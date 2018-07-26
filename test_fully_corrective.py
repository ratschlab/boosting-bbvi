#!/usr/bin/python
# TODO set seed

from edward.models import Categorical, Normal, Mixture, MultivariateNormalDiag
import tensorflow as tf
import scipy.stats
import random

import mixture_model_relbo

def main():
    # build model
    xcomps = [Normal(loc=tf.convert_to_tensor(mixture_model_relbo.mus[i]),
                    scale=tf.convert_to_tensor(mixture_model_relbo.stds[i])) for i in range(len(mixture_model_relbo.mus))]
    x = Mixture(cat=Categorical(probs=tf.convert_to_tensor(mixture_model_relbo.pi)), components=xcomps, sample_shape=mixture_model_relbo.N)

    x_mvns = [MultivariateNormalDiag(loc=tf.convert_to_tensor(mixture_model_relbo.mus[i]),
                    scale_diag=tf.convert_to_tensor(mixture_model_relbo.stds[i])) for i in range(len(mixture_model_relbo.mus))]

    x_train, components = mixture_model_relbo.build_toy_dataset(mixture_model_relbo.N)
    n_examples, n_features = x_train.shape
    qxs = [MultivariateNormalDiag(loc=[scipy.stats.norm.rvs(1)],
        scale_diag=[scipy.stats.norm.rvs(1)]) for i in range(10)]

    truth = [MultivariateNormalDiag(loc=mixture_model_relbo.mus[i],
                scale_diag=mixture_model_relbo.stds[i]) for i in range(len(mixture_model_relbo.mus))]
    qxs.extend(truth)

    mix = Mixture(cat=Categorical(probs=[1./len(qxs)] * len(qxs)), components=qxs)

    sess = tf.InteractiveSession()
    with sess.as_default():
        mixture_model_relbo.fully_corrective(mix, x)

if __name__ == "__main__":
    main()
