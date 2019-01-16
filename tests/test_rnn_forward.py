import unittest
import numpy as np
from rnn.rnn_forward import rnn_forward

class RnnForwardTest(unittest.TestCase):

    def test_rnn_forward(self):
        '''
        Tests activation and prediction produced by an RNN
        '''
        np.random.seed(1)
        x = np.random.randn(3, 10, 4)
        a0 = np.random.randn(5, 10)
        Waa = np.random.randn(5, 5)
        Wax = np.random.randn(5, 3)
        Wya = np.random.randn(2, 5)
        ba = np.random.randn(5, 1)
        by = np.random.randn(2, 1)
        parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

        a, y_pred, caches = rnn_forward(x, a0, parameters)
        print("a[4][1] = ", a[4][1])
        print("a.shape = ", a.shape)
        print("y_pred[1][3] =", y_pred[1][3])
        print("y_pred.shape = ", y_pred.shape)
        print("caches[1][1][3] =", caches[1][1][3])
        print("len(caches) = ", len(caches))

        activation_expected = [-0.99999375, 0.77911235, -0.99861469, -0.99833267]
        np.testing.assert_allclose(a[4][1], activation_expected)
        assert(a.shape == (5,10,4))

        prediction_expected = [0.79560373, 0.86224861, 0.11118257, 0.81515947]
        np.testing.assert_allclose(y_pred[1][3], prediction_expected)
        assert(y_pred.shape == (2,10,4))

        caches_expected = [-1.1425182, -0.34934272, -0.20889423, 0.58662319]
        np.testing.assert_allclose(caches[1][1][3], caches_expected)
        assert(len(caches) == 2)
