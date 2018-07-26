#!/usr/bin/python 

import utils
logger = utils.get_logger()

import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import tensorflow as tf

from edward.models import (
    Categorical, Dirichlet, Empirical, InverseGamma,
    MultivariateNormalDiag, Normal, ParamMixture, Mixture)

import edward as ed
import copy
import scipy.stats as stats
from scipy.misc import logsumexp as logsumexp

import relbo
from infinite_mixture import InfiniteMixtureScipy

import time

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('outdir', '/tmp', 'directory to store all the results, models, plots, etc.')
flags.DEFINE_integer('seed', 0, 'The random seed to use for everything.')
flags.DEFINE_integer('n_fw_iter', 100, '')
flags.DEFINE_integer('LMO_iter', 1000, '')
flags.DEFINE_string('exp', 'mixture', 'select from [mixture, s_and_s (aka spike and slab), many]')
flags.DEFINE_string('fw_variant', 'fixed', '[fixed (default), line_search, fc] The Frank-Wolfe variant to use.')
#flags.DEFINE_string('decay', 'log', '[linear, log (default), squared] The decay rate to use for Lambda.')

ed.set_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)

# what we are trying to fit
if FLAGS.exp == 'mixture':
    pi = np.array([[0.4, 0.6]]).astype(np.float32)
    mus = [[1.], [-1.]]
    stds = [[.6], [.6]]
elif FLAGS.exp == 'balanced':
    pi = np.array([[0.5, 0.5]]).astype(np.float32)
    mus = [[1.], [-1.]]
    stds = [[.6], [.6]]
elif FLAGS.exp == 's_and_s':
    pi = np.array([[0.4, 0.6]]).astype(np.float32)
    mus = [[1.], [-1.]]
    stds = [[.1], [10.]]
elif FLAGS.exp == 'many':
    mus = np.array([[5.0], [10.0], [20.0], [-2]]).astype(np.float32)
    stds = np.array([[2], [2], [1], [1]]).astype(np.float32)
    pi = np.array([[1.0/3, 1.0/4, 1.0/4, 1.0/6]]).astype(np.float32)
else:
    raise KeyError("undefined experiment")

# global settings
N = 500

def build_toy_dataset(N, D=1):
    x = np.zeros((N, D), dtype=np.float32)
    ks = np.zeros(N, dtype=np.int)
    for n in range(N):
        k = np.argmax(np.random.multinomial(1, pi[0]))
        x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))
        ks[n] = k
    return x,ks

def construct_multivariatenormaldiag(dims, iter, name='', sample_shape=N):
    #loc = tf.get_variable(name + "_loc%d" % iter, dims)
    loc = tf.get_variable(name + "_loc%d" % iter, initializer=tf.random_normal(dims))
    #scale = tf.nn.softplus(tf.get_variable(name + "_scale%d" % iter, dims))
    scale = tf.nn.softplus(tf.get_variable(name + "_scale%d" % iter, initializer=tf.random_normal(dims)))
    mvn = MultivariateNormalDiag(loc=loc, scale_diag=scale, sample_shape=sample_shape)
    return mvn

def construct_normal(dims, iter, name='', sample_shape=N):
    loc = tf.get_variable(name + "_loc%d" % iter, initializer=tf.random_normal(dims) + \
            np.random.normal())
    scale = tf.get_variable(name + "_scale%d" % iter, initializer=tf.random_normal(dims))
    return Normal(loc=loc, scale=tf.nn.softplus(scale), sample_shape=N)

def elbo(q, p, n_samples=1000):
    samples = q.sample(n_samples)
    elbo_samples =  p.log_prob(samples) - q.log_prob(samples)
    elbo_samples = elbo_samples.eval()

    avg = np.mean(elbo_samples)
    std = np.std(elbo_samples)
    return avg, std

def setup_outdir():
    outdir = FLAGS.outdir
    outdir = os.path.expanduser(outdir)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def line_search_dkl(weights,locs,diags, mu_s, cov_s, x, k):
    def softmax(v):
        return np.log(1 + np.exp(v))

    N_samples  = 10

    weights = [weights]

    qt_comps = [Normal(loc=tf.convert_to_tensor(locs[i]),
                       scale=tf.convert_to_tensor(diags[i])) for i in range(len(locs))]

    qt = Mixture(cat=Categorical(probs=tf.convert_to_tensor(weights)),
            components=qt_comps, sample_shape=N)

    qt = InfiniteMixtureScipy(stats.multivariate_normal)
    qt.weights = weights[0]
    qt.params = list(zip([[l] for l in locs], [[softmax(np.dot(d,d))] for d in diags]))

    sample_q = qt.sample_n(N_samples)

    s = stats.multivariate_normal([mu_s], np.dot(np.array([cov_s]), np.array([cov_s])))
    sample_s = s.rvs(N_samples)

    new_locs = copy.copy(locs)
    new_diags = copy.copy(diags)
    new_locs.append([mu_s])
    new_diags.append([cov_s])

    gamma = 2./(k+2.)
    n_steps = 10
    prog_bar = ed.util.Progbar(n_steps)
    for it in range(n_steps):
        print("line_search iter %d, %.5f" % (it, gamma))
        new_weights = copy.copy(weights)
        new_weights[0] = [(1. - gamma) * w for w in new_weights[0]]
        new_weights[0].append(gamma)

        q_next = InfiniteMixtureScipy(stats.multivariate_normal)
        q_next.weights = new_weights[0]
        q_next.params = list(zip([[l] for l in new_locs], [[np.dot(d,d)] for d in new_diags]))

        def px_qx_ratio_log_prob(v):
            Lambda = 1.
            ret = x.log_prob([v]).eval()[0] - q_next.log_prob(v)
            ret /= Lambda
            return ret

        rez_s = [px_qx_ratio_log_prob(sample_s[ss]) for ss in range(len(sample_s))]

        rez_q = [px_qx_ratio_log_prob(sample_q[ss]) for ss in range(len(sample_q))]

        gamma = gamma + 0.1*(sum(rez_s)-sum(rez_q))/(N_samples*(it+1.))

        if gamma>= 1 or gamma<=0:
            gamma = max(min(gamma,1.),0.)
            break
    return gamma

def fully_corrective(q, p):
    comps = q.components

    n_comps = len(comps)

    # randomly initialize, rather than take q.cat as the initialization
    weights = np.random.random(n_comps).astype(np.float32)
    weights /= np.sum(weights)

    n_samples = 1000
    samples = [comp.sample(n_samples) for comp in comps]

    #S = tf.zeros([n_samples, n_comps, n_comps])
    S = []
    for j in range(n_comps):
        for i in range(n_comps):
            comp_log_prob = tf.squeeze(comps[i].log_prob(samples[j]))
            #S[:,i,j] = comp_log_prob
            S.append(comp_log_prob)
    S = tf.transpose(tf.reshape(tf.stack(S), [n_comps, n_comps, n_samples]))
    S = S.eval()

    p_log_probs = [tf.squeeze(p.log_prob(i)).eval() for i in samples]

    T = 1000000
    for t in range(T):
        grad = np.zeros(n_comps)
        for i in range(n_comps):
            part = np.zeros([n_samples, n_comps])
            #part = []
            for j in range(n_comps):
                probs = S[:,j,i]
                part[:,j] =  np.log(weights[j] + 1e-10) + probs
                #part.append(np.log(weights[j] + 1e-10).astype(np.float32) + probs)

            #part = tf.stack(part)

            #part = tf.convert_to_tensor(part.astype(np.float32))
            part = logsumexp(part, axis=1)
            #part = tf.reduce_logsumexp(part, axis=0)
            diff = part - p_log_probs[i]
            #grad[i] = tf.reduce_mean(diff, axis=0).eval()
            grad[i] = np.mean(diff, axis=0)

        min_i = np.argmin(grad)
        corner = np.zeros(weights.shape)
        corner[min_i] = 1

        if t % 1000 == 0:
            print("grad", grad)

        duality_gap = - np.dot(grad, (corner - weights))

        if t % 1000 == 0:
            print("duality_gap", duality_gap)

        if duality_gap < 1e-6:
            return weights

        weights += 2. / (t + 2.) * (corner - weights)
        if t % 1000 == 0:
            print("weights", weights, t)

    print("weights", weights, t)
    return weights

def main(argv):
    del argv

    x_train, components = build_toy_dataset(N)
    n_examples, n_features = x_train.shape

    # save the target
    outdir = setup_outdir()
    np.savez(os.path.join(outdir, 'target_dist.npz'),
            pi=pi, mus=mus, stds=stds)

    weights, comps = [], []
    elbos = []
    relbo_vals = []
    times = []
    for iter in range(FLAGS.n_fw_iter):
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(FLAGS.seed)
            sess = tf.InteractiveSession()
            with sess.as_default():
                # build model
                xcomps = [Normal(loc=tf.convert_to_tensor(mus[i]),
                                scale=tf.convert_to_tensor(stds[i])) for i in range(len(mus))]
                x = Mixture(cat=Categorical(probs=tf.convert_to_tensor(pi)), components=xcomps, sample_shape=N)

                qx = construct_normal([n_features], iter, 'qx')
                if iter > 0:
                    qtx = Mixture(cat=Categorical(probs=tf.convert_to_tensor(weights)),
                            components=[Normal(loc=c['loc'][0],
                                #scale_diag=tf.nn.softplus(c['scale_diag'])) for c in comps], sample_shape=N)
                                scale=c['scale_diag'][0]) for c in comps], sample_shape=N)
                    fw_iterates = {x: qtx}
                else:
                    fw_iterates = {}

                sess.run(tf.global_variables_initializer())

                total_time = 0
                start_inference_time = time.time()
                inference = relbo.KLqp({x: qx}, fw_iterates=fw_iterates, fw_iter=iter)
                inference.run(n_iter=FLAGS.LMO_iter)
                end_inference_time = time.time()

                total_time += end_inference_time - start_inference_time

                if iter > 0:
                    relbo_vals.append( -utils.compute_relbo(qx, fw_iterates[x], x, np.log(iter + 1)) )

                if iter == 0:
                    gamma = 1.
                elif iter > 0 and FLAGS.fw_variant == 'fixed':
                    gamma = 2. / (iter + 2.)
                elif iter > 0  and FLAGS.fw_variant == 'line_search':
                    start_line_search_time = time.time()
                    gamma = line_search_dkl(weights,
                            [c['loc'] for c in comps],
                            [c['scale_diag'] for c in comps],
                            qx.loc.eval(), qx.stddev().eval(), x, iter)
                    end_line_search_time = time.time()
                    total_time += end_line_search_time - start_line_search_time
                elif iter > 0 and FLAGS.fw_variant == 'fc':
                    gamma = 2. / (iter + 2.)

                comps.append( {'loc': qx.mean().eval(), 'scale_diag': qx.stddev().eval()} )
                weights = utils.update_weights(weights, gamma, iter)

                print("weights",     weights)
                print("comps",       [c['loc'] for c in comps])
                print("scale_diags", [c['scale_diag'] for c in comps])

                q_latest = Mixture(cat=Categorical(probs=tf.convert_to_tensor(weights)),
                        components=[MultivariateNormalDiag(**c) for c in comps], sample_shape=N)

                if FLAGS.fw_variant == "fc":
                    start_fc_time = time.time()
                    weights = fully_corrective(q_latest, x)
                    weights = list(weights)
                    for i in reversed(range(len(weights))):
                        w = weights[i]
                        if w == 0:
                            del weights[i]
                            del comps[i]
                    weights = np.array(weights)
                    end_fc_time = time.time()
                    total_time += end_fc_time - start_fc_time

                q_latest = Mixture(cat=Categorical(probs=tf.convert_to_tensor(weights)),
                        components=[MultivariateNormalDiag(**c) for c in comps], sample_shape=N)

                elbos.append(elbo(q_latest, x))

                outdir = setup_outdir()

                print("total time", total_time)
                times.append( float(total_time) )
                utils.save_times(os.path.join(outdir, 'times.csv'), times)

                elbos_filename = os.path.join(outdir, 'elbos.csv')
                logger.info("iter, %d, elbo, %.2f +/- %.2f" % (iter, *elbos[-1]) )
                np.savetxt(elbos_filename, elbos, delimiter=',')
                logger.info("saving elbos to, %s" % elbos_filename)

                relbos_filename = os.path.join(outdir, 'relbos.csv')
                np.savetxt(relbos_filename, relbo_vals, delimiter=',')
                logger.info("saving relbo values to, %s" % relbos_filename)

                for_serialization = {'locs': np.array([c['loc'] for c in comps]),
                                     'scale_diags': np.array([c['scale_diag'] for c in comps]) }
                qt_outfile = os.path.join(outdir, 'qt_iter%d.npz' % iter)
                np.savez(qt_outfile, weights=weights, **for_serialization)
                np.savez(os.path.join(outdir, 'qt_latest.npz'), weights=weights, **for_serialization)
                logger.info("saving qt to, %s" % qt_outfile)
        tf.reset_default_graph()

if __name__ == "__main__":
  tf.app.run()
