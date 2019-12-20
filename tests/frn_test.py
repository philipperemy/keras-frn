import unittest

import keras.backend as K
import numpy as np
import tensorflow as tf

from frn.frn import frn_layer_paper, frn_layer_keras


class FRNTest(unittest.TestCase):

    def test_1(self):
        np.random.seed(123)
        x = np.random.uniform(size=(10, 32, 32, 64))
        tau = tf.constant(np.ones(shape=(64,)), dtype=tf.double)
        beta = tf.constant(np.ones(shape=(64,)), dtype=tf.double)
        gamma = tf.constant(np.ones(shape=(64,)), dtype=tf.double)
        epsilon = np.float64(1e-6)
        a1 = frn_layer_paper(x, tau, beta, gamma, epsilon=epsilon)
        a2 = K.eval(frn_layer_keras(x, tau, beta, gamma, epsilon=epsilon))
        np.testing.assert_array_almost_equal(a1, a2)
