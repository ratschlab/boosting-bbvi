#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import pickle
import edward as ed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.special import expit as sigmoid
from scipy.misc import logsumexp
from sklearn.metrics import roc_auc_score

from edward.models import Bernoulli, Normal, Empirical, MultivariateNormalDiag
from edward.models import Mixture, Categorical
from edward.models import RandomVariable
import edward.util

import boosting_bbvi.core.relbo as relbo
from boosting_bbvi.core.mvn import mvn
from boosting_bbvi.core.lpl import lpl
import boosting_bbvi.core.utils as utils
import blr_utils
import six

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("LMO_iter", 600, '')
tf.flags.DEFINE_integer('n_fw_iter', 100, '')
tf.flags.DEFINE_string("outdir", '/tmp', '')
tf.flags.DEFINE_string("fw_variant", 'fixed', '')
tf.flags.DEFINE_string("base_dist", 'mvn', '')

tf.flags.DEFINE_integer('seed', 0, 'The random seed to use for everything.')
ed.set_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)

bases = {'mvn': mvn, 'normal': mvn, 'lpl': lpl, 'laplace': lpl}
base = bases[FLAGS.base_dist]

def construct_base_dist(dims, iter, name=''):
  loc = tf.get_variable(name + "_loc%d" % iter, initializer=tf.random_normal(dims) + np.random.normal())
  scale = tf.nn.softplus(tf.get_variable(name + "_scale%d" % iter, initializer=tf.random_normal(dims) + np.random.normal()))
  rez = base(loc=loc, scale=scale)
  return rez

def setup_outdir():
  outdir = FLAGS.outdir
  outdir = os.path.expanduser(outdir)
  os.makedirs(outdir, exist_ok=True)
  return outdir

def get_fw_iterates(weights, target, components):
  if len(weights) == 0:
    return {}
  else:
    return {target: build_mixture(weights, components)}

def build_mixture(weights,components):
  cat = Categorical(probs=tf.convert_to_tensor(weights))
  comps = [base(loc=tf.convert_to_tensor(c['loc']),
                 scale=tf.convert_to_tensor(c['scale'])) for c in components]
  mix = Mixture(cat=cat, components=comps)
  return mix

def update_iterate(components, s):
  loc = s.loc.eval()
  scale = s.scale.eval()
  components.append({'loc': loc, 'scale': scale})
  return components

def add_bias_column(X):
  N,D = X.shape
  ret = np.c_[X, np.ones(N)]
  return ret

def euclidean_proj_simplex(v, s=1):
  """ Compute the Euclidean projection on a positive simplex
  Solves the optimisation problem (using the algorithm from [1]):
      min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
  Parameters
  ----------
  v: (n,) numpy array,
     n-dimensional vector to project
  s: int, optional, default: 1,
     radius of the simplex
  Returns
  -------
  w: (n,) numpy array,
     Euclidean projection of v on the simplex
  Notes
  -----
  The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
  Better alternatives exist for high-dimensional sparse vectors (cf. [1])
  However, this implementation still easily scales to millions of dimensions.
  References
  ----------
  [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
      John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
      International Conference on Machine Learning (ICML 2008)
      http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf

  source: https://gist.github.com/daien/1272551
  """
  assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
  n, = v.shape  # will raise ValueError if v is not 1-D
  # check if we are already on the simplex
  if v.sum() == s and np.alltrue(v >= 0):
    # best projection: itself!
    return v
  # get the array of cumulative sums of a sorted (decreasing) copy of v
  u = np.sort(v)[::-1]
  cssv = np.cumsum(u)
  # get the number of > 0 components of the optimal solution
  rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
  # compute the Lagrange multiplier associated to the simplex constraint
  theta = (cssv[rho] - s) / (rho + 1.0)
  # compute the projection by thresholding v using theta
  w = (v - theta).clip(min=0)
  return w

def fully_corrective(q, p):
  comps = q.components

  if False:
    for i,comp in enumerate(comps):
      print("component", i, "\tmean", comp.mean().eval(), "\tstddev", comp.stddev().eval())

  n_comps = len(comps)

  # randomly initialize, rather than take q.cat as the initialization
  weights = np.random.random(n_comps).astype(np.float32)
  weights /= np.sum(weights)

  n_samples = 1000
  samples = [comp.sample(n_samples).eval() for comp in comps] # comp.sample resamples each time, so eval once to keep samples fixed.

  p_log_probs = [tf.squeeze(p.log_prob(i)).eval() for i in samples]

  do_frank_wolfe = False
  if do_frank_wolfe:
    T = 1000000
  else:
    T = 100

  for t in range(T):
    # compute the gradient
    grad = np.zeros(n_comps)
    for i in range(n_comps):
      q_log_prob = Mixture(cat=Categorical(weights), components=comps).log_prob(samples[i]).eval()
      q_log_prob = np.sum(q_log_prob, axis=1)

      diff = q_log_prob - p_log_probs[i]
      grad[i] = np.mean(diff, axis=0) # take the expectation

    if do_frank_wolfe:
      min_i = np.argmin(grad)
      corner = np.zeros(weights.shape)
      corner[min_i] = 1.

      if t % 1000 == 0:
        print("grad", grad)

      duality_gap = - np.dot(grad, (corner - weights))
      print("duality gap", duality_gap)
      #assert False

      if t % 1000 == 0:
        print("duality_gap", duality_gap)

      if duality_gap < 1e-10:
        print("weights", weights, "duality gap", duality_gap, "iteration", t)
        return weights

      weights += 2. / (t + 2.) * (corner - weights)
      if t % 1000 == 0:
        print("weights", weights, t)
    else:
      # gradient step
      step_size = 0.001
      weights_prime = weights - step_size * grad
      print("before", weights_prime)
      weights_prime = euclidean_proj_simplex(weights_prime)
      print("after", weights_prime)
      if np.max(np.abs(weights - weights_prime)) < 1e-6:
        weights = weights_prime.astype(np.float32)
        break
      weights = weights_prime.astype(np.float32)

  #print("weights", weights, "duality gap", duality_gap, "iteration", t)
  return weights

def line_search(q, s, p):
  s_samples = s.sample(1000).eval()
  q_samples = q.sample(1000).eval()

  gamma = 0.5

  def get_new_weights(gamma):
    weights = q.cat.probs.eval()
    weights *= (1 - gamma)
    weights = np.append(weights, gamma).astype(np.float32)
    return weights

  comps = q.components
  comps = list(comps)
  comps.append(s)

  T = 50
  for t in range(T):
    weights = get_new_weights(gamma)

    mix = Mixture(cat=Categorical(weights), components=comps)
    s_expectation = tf.reduce_sum(mix.log_prob(s_samples), axis=1) - p.log_prob(s_samples)
    q_expectation = tf.reduce_sum(mix.log_prob(q_samples), axis=1) - p.log_prob(q_samples)
    grad = s_expectation - q_expectation
    grad = tf.reduce_mean(grad).eval()

    print("t", t, "gamma", gamma, "grad", grad)

    step_size = 0.01 / (t+1)
    gamma_prime = gamma - grad * step_size
    if gamma_prime>= 1 or gamma_prime<=0:
        gamma_prime = max(min(gamma_prime,1.),0.)

    if np.abs(gamma - gamma_prime) < 1e-6:
      gamma = gamma_prime
      break
    gamma = gamma_prime


  if gamma < 1e-5:
    gamma = 1e-5

  print("final t", t, "gamma", gamma, "grad", grad)
  return get_new_weights(gamma)

class Joint_slow:
  '''Wrapper to handle calculating the log p(y, w | X) = log [ p(y | X, w) * p(w) ] for a given sample of w.'''
  def __init__(self, X, Xtrain, w, loglik_op, sess):
    self.X = X
    self.Xtrain = Xtrain
    self.w = w
    self.loglik_op = loglik_op
    self.sess = sess

  def log_prob(self, samples):
    n_samples, n_features = samples.shape
    ret = np.zeros(n_samples)
    for i in range(n_samples):
      lik = self.sess.run(self.loglik_op, feed_dict={self.X: self.Xtrain, self.w: samples[i]})
      prior = np.sum(self.w.log_prob(samples[i]).eval())
      ret[i] = lik + prior
    return ret

class Joint:
  '''
  Wrapper to handle calculating the log p(y, w | X) = log [ p(y | X, w) *
  p(w) ] for a given sample of w.
  Should be the same as the slow version but vectorized and therefore faster.
  '''
  def __init__(self, Xtrain, ytrain, sess):
    self.Xtrain = Xtrain
    self.ytrain = ytrain
    self.sess = sess

    self.n_samples = 1000 # TODO this is hard coded and must be matched in elbo and fc.
    N, D = Xtrain.shape
    self.w = tf.placeholder(tf.float32, [D, self.n_samples])
    self.X = tf.placeholder(tf.float32, [N, D])
    #self.y = Bernoulli(logits=ed.dot(self.X, self.w))
    self.y = Bernoulli(logits=tf.matmul(self.X, self.w))
    self.prior = Normal(loc=tf.zeros([self.n_samples, D]), scale=1.0 * tf.ones([self.n_samples, D])) # TODO hard coded

  def log_prob(self, samples):
    copied_ytrain = np.repeat(self.ytrain[:, np.newaxis], self.n_samples, axis=1)
    per_sample = self.sess.run(self.y.log_prob(copied_ytrain),
        feed_dict={self.X: self.Xtrain, self.w: samples.T}).astype(np.float32)
    lik = np.sum(per_sample, axis=0)
    prior = np.sum(self.prior.log_prob(samples).eval(), axis=1)
    return lik + prior
    #return lik

def elbo(q, joint, prior, n_samples=1000):
  samples = q.sample(n_samples)
  samples = samples.eval()
  # TODO sum across elements of prior and q since features are assumed to be indepenent.
  p_log_prob = tf.reduce_mean(joint.log_prob(samples))
  q_log_prob = tf.reduce_mean(tf.reduce_sum(q.log_prob(samples), axis=1))
  elbo_samples = p_log_prob - q_log_prob
  print("elbo", "p log prob", p_log_prob.eval(), "q log prob", q_log_prob.eval())
  return tf.reduce_mean(elbo_samples).eval()

def append_and_save(alist, datum, outfile, save_func):
  alist.append(datum)
  outdir = setup_outdir()
  save_func(os.path.join(outdir, outfile), alist)
  return alist

def myloss(inference):
  n_samples = 100
  p_log_prob = [0.0] * n_samples
  q_log_prob = [0.0] * n_samples
  base_scope = tf.get_default_graph().unique_name("inference") + '/'
  for s in range(n_samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
    scope = base_scope + tf.get_default_graph().unique_name("sample")
    dict_swap = {}
    for x, qx in six.iteritems(inference.data):
      if isinstance(x, RandomVariable):
        if isinstance(qx, RandomVariable):
          qx_copy = edward.util.copy(qx, scope=scope)
          dict_swap[x] = qx_copy.value()
        else:
          dict_swap[x] = qx

    for z, qz in six.iteritems(inference.latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = edward.util.copy(qz, scope=scope)
      dict_swap[z] = qz_copy.value()
      q_log_prob[s] += tf.reduce_sum(
          #inference.scale.get(z, 1.0) *
          qz_copy.log_prob(dict_swap[z]))

    # prior
    for z in six.iterkeys(inference.latent_vars):
      z_copy = edward.util.copy(z, dict_swap, scope=scope)
      p_log_prob[s] += tf.reduce_sum(
          #inference.scale.get(z, 1.0) *
          z_copy.log_prob(dict_swap[z]))

    # likelihood
    for x in six.iterkeys(inference.data):
      if isinstance(x, RandomVariable):
        x_copy = edward.util.copy(x, dict_swap, scope=scope)
        p_log_prob[s] += tf.reduce_sum(
            #inference.scale.get(x, 1.0) *
            x_copy.log_prob(dict_swap[x]))

  p_log_prob = tf.reduce_mean(p_log_prob)
  q_log_prob = tf.reduce_mean(q_log_prob)

  loss = -(p_log_prob - q_log_prob)
  return p_log_prob, q_log_prob
  #return loss

def decompose_relbo(inference):
  """
  Copied from relbo in an effort to analyze the three parts of the RELBO.

  Build loss function. Its automatic differentiation
  is a stochastic gradient of

  $-\\text{ELBO} =
      -\mathbb{E}_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]$

  based on the reparameterization trick [@kingma2014auto].

  Computed by sampling from $q(z;\lambda)$ and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_prob = [0.0] * inference.n_samples
  q_log_prob = [0.0] * inference.n_samples
  relbo_reg_log_prob = [0.0] * inference.n_samples
  base_scope = tf.get_default_graph().unique_name("inference") + '/'
  for s in range(inference.n_samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
    scope = base_scope + tf.get_default_graph().unique_name("sample")
    dict_swap = {}
    for x, qx in six.iteritems(inference.data):
      if isinstance(x, RandomVariable):
        if isinstance(qx, RandomVariable):
          qx_copy = edward.util.copy(qx, scope=scope)
          dict_swap[x] = qx_copy.value()
        else:
          dict_swap[x] = qx

    for z, qz in six.iteritems(inference.latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = edward.util.copy(qz, scope=scope)
      dict_swap[z] = qz_copy.value()
      q_log_prob[s] += tf.reduce_sum(
          inference.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))

    # prior
    for z in six.iterkeys(inference.latent_vars):
      z_copy = edward.util.copy(z, dict_swap, scope=scope)
      p_log_prob[s] += tf.reduce_sum(
          inference.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))

    # likelihood
    for x in six.iterkeys(inference.data):
      if isinstance(x, RandomVariable):
        x_copy = edward.util.copy(x, dict_swap, scope=scope)
        p_log_prob[s] += tf.reduce_sum(
            inference.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))

    # RELBO
    for fz, qt in six.iteritems(inference.fw_iterates):
      qt_copy = edward.util.copy(qt, dict_swap, scope=scope)
      relbo_reg_log_prob[s] += tf.reduce_sum(
          #inference.scale.get(z, 1.0) * qt_copy.log_prob(fz(dict_swap)))
          inference.scale.get(z, 1.0) * qt_copy.log_prob(dict_swap[z]))
          #1.0 * qt_copy.log_prob(fz(dict_swap)))

  p_log_prob = tf.reduce_mean(p_log_prob)
  q_log_prob = tf.reduce_mean(q_log_prob)
  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())
  relbo_reg_log_prob = tf.reduce_mean(relbo_reg_log_prob)

  if inference.logging:
    tf.summary.scalar("loss/p_log_prob", p_log_prob,
                      collections=[inference._summary_key])
    tf.summary.scalar("loss/q_log_prob", q_log_prob,
                      collections=[inference._summary_key])
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=[inference._summary_key])

  annealers = {'linear': lambda t: t,
               'constant': lambda t: 1.,
               'log': lambda t: 1. + np.log(t),
               '1oversqrt': lambda t: 1. / np.sqrt(t + 1),
               '1overt': lambda t: 1. / (t + 1),
               'sqrtinc': lambda t: np.sqrt(t + 1)
               }

  relbo_reg = FLAGS.relbo_reg * annealers[FLAGS.relbo_anneal](inference.fw_iter)

  #loss = -(p_log_prob - relbo_reg * q_log_prob - reg_penalty - 0.001 * relbo_reg_log_prob)
  return p_log_prob, q_log_prob, relbo_reg_log_prob

def compute_duality_gap(joint, q, s):
  def expected_gradient(wrt):
    samples = wrt.sample(1000).eval()
    q_log_prob = np.sum(np.mean(q.log_prob(samples).eval(), axis=0))
    p_log_prob = np.mean(joint.log_prob(samples))
    return q_log_prob - p_log_prob

  egq = expected_gradient(q)
  egs = expected_gradient(s)
  print("q", egq, "s", egs)
  return egq - egs

def main(_):
  outdir = setup_outdir()
  ed.set_seed(FLAGS.seed)

  ((Xtrain, ytrain), (Xtest, ytest)) = blr_utils.get_data()
  N,D = Xtrain.shape
  N_test,D_test = Xtest.shape

  print("Xtrain")
  print(Xtrain)
  print(Xtrain.shape)

  if 'synthetic' in FLAGS.exp:
    w = Normal(loc=tf.zeros(D), scale=1.0 * tf.ones(D))
    X = tf.placeholder(tf.float32, [N, D])
    y = Bernoulli(logits=ed.dot(X, w))

    #n_posterior_samples = 100000
    n_posterior_samples = 10
    qw_empirical = Empirical(params=tf.get_variable("qw/params", [n_posterior_samples, D]))
    inference = ed.HMC({w: qw_empirical}, data={X: Xtrain, y: ytrain})
    inference.initialize(n_print=10, step_size=0.6)

    tf.global_variables_initializer().run()
    inference.run()

    empirical_samples = qw_empirical.sample(50).eval()
    #fig, ax = plt.subplots()
    #ax.scatter(posterior_samples[:,0], posterior_samples[:,1])
    #plt.show()

  weights, q_components = [], []
  ll_trains, ll_tests, bin_ac_trains, bin_ac_tests, elbos, rocs, gaps = [], [], [], [], [], [], []
  total_time, times = 0., []
  for iter in range(0, FLAGS.n_fw_iter):
    print("iter %d" % iter)
    g = tf.Graph()
    with g.as_default():
      sess = tf.InteractiveSession()
      with sess.as_default():
        tf.set_random_seed(FLAGS.seed)
        # MODEL
        w = Normal(loc=tf.zeros(D), scale=1.0 * tf.ones(D))

        X = tf.placeholder(tf.float32, [N, D])
        y = Bernoulli(logits=ed.dot(X, w))

        X_test = tf.placeholder(tf.float32, [N_test, D_test])
        y_test = Bernoulli(logits=ed.dot(X_test, w))

        qw = construct_base_dist([D], iter, 'qw')
        inference_time_start = time.time()
        inference = relbo.KLqp({w: qw}, fw_iterates=get_fw_iterates(weights, w, q_components),
            data={X: Xtrain, y: ytrain}, fw_iter=iter)
        tf.global_variables_initializer().run()
        inference.run(n_iter=FLAGS.LMO_iter)
        inference_time_end = time.time()
        total_time += float(inference_time_end - inference_time_start)

        joint = Joint(Xtrain, ytrain, sess)
        if iter > 0:
          qtw_prev = build_mixture(weights, q_components)
          gap = compute_duality_gap(joint, qtw_prev, qw)
          gaps.append(gap)
          np.savetxt(os.path.join(outdir, "gaps.csv"), gaps, delimiter=',')
          print("duality gap", gap)

        # update weights
        gamma = 2. / (iter + 2.)
        weights = [(1. - gamma) * w for w in weights]
        weights.append(gamma)

        # update components
        q_components = update_iterate(q_components, qw)

        if len(q_components) > 1 and FLAGS.fw_variant == 'fc':
          print("running fully corrective")
          # overwrite the weights
          weights = fully_corrective(build_mixture(weights, q_components), joint)

          if True:
            # remove inactivate iterates
            weights = list(weights)
            for i in reversed(range(len(weights))):
              if weights[i] == 0:
                del weights[i]
                del q_components[i]
            weights = np.array(weights) # TODO type acrobatics to make elements deletable
        elif len(q_components) > 1 and FLAGS.fw_variant == 'line_search':
          print("running line search")
          weights = line_search(build_mixture(weights[:-1], q_components[:-1]), qw, joint)

        qtw_new = build_mixture(weights, q_components)

        if False:
          for i,comp in enumerate(qtw_new.components):
            print("component", i, "\tmean", comp.mean().eval(), "\tstddev", comp.stddev().eval())

        train_lls = [sess.run(y.log_prob(ytrain), feed_dict={X: Xtrain, w: qtw_new.sample().eval()}) for _ in range(50)]
        train_lls = np.mean(train_lls, axis=0)
        ll_trains.append((np.mean(train_lls), np.std(train_lls)))

        test_lls = [sess.run(y_test.log_prob(ytest), feed_dict={X_test: Xtest, w: qtw_new.sample().eval()}) for _ in range(50)]
        test_lls = np.mean(test_lls, axis=0)
        ll_tests.append((np.mean(test_lls), np.std(test_lls)))

        logits = np.mean([np.dot(Xtest, qtw_new.sample().eval()) for _ in range(50)], axis=0)
        ypred = tf.sigmoid(logits).eval()
        roc_score = roc_auc_score(ytest, ypred)
        rocs.append(roc_score)

        print('roc_score', roc_score)
        print('ytrain', np.mean(train_lls), np.std(train_lls))
        print('ytest', np.mean(test_lls), np.std(test_lls))

        order = np.argsort(ytest)
        plt.scatter(range(len(ypred)), ypred[order], c=ytest[order])
        plt.savefig(os.path.join(outdir, 'ypred%d.pdf' % iter))
        plt.close()

        np.savetxt(os.path.join(outdir, "train_lls.csv"), ll_trains, delimiter=',')
        np.savetxt(os.path.join(outdir, "test_lls.csv"), ll_tests, delimiter=',')
        np.savetxt(os.path.join(outdir, "rocs.csv"), rocs, delimiter=',')

        x_post = ed.copy(y, {w: qtw_new})
        x_post_t = ed.copy(y_test, {w: qtw_new})

        print('log lik train', ed.evaluate('log_likelihood', data={x_post: ytrain, X:Xtrain}))
        print('log lik test',  ed.evaluate('log_likelihood', data={x_post_t: ytest, X_test:Xtest}))

        #ll_train = ed.evaluate('log_likelihood', data={x_post: ytrain, X:Xtrain})
        #ll_test = ed.evaluate('log_likelihood', data={x_post_t: ytest, X_test:Xtest})
        bin_ac_train = ed.evaluate('binary_accuracy', data={x_post: ytrain, X: Xtrain})
        bin_ac_test = ed.evaluate('binary_accuracy', data={x_post_t: ytest, X_test: Xtest})
        print('binary accuracy train', bin_ac_train)
        print('binary accuracy test', bin_ac_test)
        #latest_elbo = elbo(qtw_new, joint, w)

        #foo = ed.KLqp({w: qtw_new}, data={X: Xtrain, y: ytrain})
        #op = myloss(foo)
        #print("myloss", sess.run(op[0], feed_dict={X: Xtrain, y: ytrain}), sess.run(op[1], feed_dict={X: Xtrain, y: ytrain}))

        #append_and_save(ll_trains, ll_train, "loglik_train.csv", np.savetxt)
        #append_and_save(ll_tests, ll_train, "loglik_test.csv", np.savetxt) #append_and_save(bin_ac_trains, bin_ac_train, "bin_acc_train.csv", np.savetxt) #append_and_save(bin_ac_tests, bin_ac_test, "bin_acc_test.csv", np.savetxt)
        ##append_and_save(elbos, latest_elbo, "elbo.csv", np.savetxt)

        #print('log-likelihood train ', ll_train)
        #print('log-likelihood test ', ll_test)
        #print('binary_accuracy train ', bin_ac_train)
        #print('binary_accuracy test ', bin_ac_test)
        #print('elbo', latest_elbo)
        times.append( total_time )
        np.savetxt(os.path.join(setup_outdir() , 'times.csv'), times)

    tf.reset_default_graph()

if __name__ == "__main__":
  tf.app.run()
