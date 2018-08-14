#!/usr/bin/python

import tensorflow as tf
from edward.models import Normal

class mvn(Normal):
  def _batch_shape_tensor(self):
    return tf.constant([], dtype=dtypes.int32)

  def _batch_shape(self):
    return tf.TensorShape([])

  def _event_shape_tensor(self):
    return tf.shape(self.loc)

  def _event_shape(self):
    return self._loc.get_shape()

  def _sample_n(self, n, seed=None):
    new_shape = tf.concat([[n], self.event_shape_tensor()], 0)
    sample = tf.random_normal(new_shape, seed=seed, dtype=self.loc.dtype,
        mean=self.loc, stddev=self.scale)
    return tf.cast(sample, self.dtype)
