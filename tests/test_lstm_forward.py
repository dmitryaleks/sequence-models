import unittest
import numpy as np
import math
from rnn.lstm_forward import lstm_forward

class LstmForwardTest(unittest.TestCase):

    def test_rnn_forward(self):
        '''
        Tests activation and prediction produced by an LSTM
        '''
        np.random.seed(1)
        x = np.random.randn(3, 10, 7)
        a0 = np.random.randn(5, 10)
        Wf = np.random.randn(5, 5 + 3)
        bf = np.random.randn(5, 1)
        Wi = np.random.randn(5, 5 + 3)
        bi = np.random.randn(5, 1)
        Wo = np.random.randn(5, 5 + 3)
        bo = np.random.randn(5, 1)
        Wc = np.random.randn(5, 5 + 3)
        bc = np.random.randn(5, 1)
        Wy = np.random.randn(2, 5)
        by = np.random.randn(2, 1)

        parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

        a, y, c, caches = lstm_forward(x, a0, parameters)

        assert(math.isclose(a[4][3][6], 0.172117767533))
        assert(a.shape == (5,10,7))

        assert(math.isclose(y[1][4][3], 0.95087346185))
        assert(y.shape == (2,10,7))

        assert(math.isclose(c[1][2][1], -0.855544916718))

        caches_expected = [0.82797464, 0.23009474, 0.76201118, -0.22232814, -0.20075807, 0.18656139, 0.41005165]
        np.testing.assert_allclose(caches[1][1][1], caches_expected)
        assert(len(caches) == 2)
