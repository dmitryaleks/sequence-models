import unittest
import numpy as np
from rnn.lstm_cell_forward import lstm_cell_forward

class LstmCellForwardTest(unittest.TestCase):

    def test_rnn_cell_forward(self):
        '''
        Tests activation and prediction produced by a self-built basic LSTM cell
        '''
        np.random.seed(1)
        xt = np.random.randn(3, 10)
        a_prev = np.random.randn(5, 10)
        c_prev = np.random.randn(5, 10)
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

        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)

        matching_tolerance = 0.001

        a_next_four_expected = [-0.66408471, 0.0036921, 0.02088357, 0.22834167, -0.85575339, 0.00138482, 0.76566531, 0.34631421, -0.00215674, 0.43827275]
        np.testing.assert_allclose(a_next[4], a_next_four_expected, matching_tolerance)
        assert(a_next.shape == (5,10))

        c_next_two = [0.63267805, 1.00570849, 0.35504474, 0.20690913, -1.64566718, 0.11832942, 0.76449811, -0.0981561, -0.74348425, -0.26810932]
        np.testing.assert_allclose(c_next[2], c_next_two, matching_tolerance)
        assert(c_next.shape == (5,10))

        yt_pred_one_expected = [0.79913913, 0.15986619, 0.22412122, 0.15606108, 0.97057211, 0.31146381, 0.00943007, 0.12666353, 0.39380172, 0.07828381]
        np.testing.assert_allclose(yt[1], yt_pred_one_expected, matching_tolerance)
        assert(yt.shape == (2,10))

        cache_expected = [-0.16263996, 1.03729328, 0.72938082, -0.54101719, 0.02752074, -0.30821874, 0.07651101, -1.03752894, 1.41219977, -0.37647422]
        np.testing.assert_allclose(cache[1][3], cache_expected, matching_tolerance)
        assert(len(cache) == 10)

