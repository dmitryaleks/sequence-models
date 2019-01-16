import unittest
import numpy as np
from rnn.rnn_cell_forward import rnn_cell_forward

class RnnCellForwardTest(unittest.TestCase):

    def test_rnn_cell_forward(self):
        '''
        Tests activation and prediction produced by a self-built basic RNN cell
        '''
        np.random.seed(1)
        xt = np.random.randn(3, 10)
        a_prev = np.random.randn(5, 10)
        Waa = np.random.randn(5, 5)
        Wax = np.random.randn(5, 3)
        Wya = np.random.randn(2, 5)
        ba = np.random.randn(5, 1)
        by = np.random.randn(2, 1)
        parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

        a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)

        a_next_four_expected = [0.59584544, 0.18141802, 0.61311866, 0.99808218, 0.85016201, 0.99980978, -0.18887155, 0.99815551, 0.6531151, 0.82872037]
        np.testing.assert_allclose(a_next[4], a_next_four_expected)
        assert(a_next.shape == (5,10))

        yt_pred_one_expected = [0.9888161, 0.01682021, 0.21140899, 0.36817467, 0.98988387, 0.88945212, 0.36920224, 0.9966312, 0.9982559, 0.17746526]
        np.testing.assert_allclose(yt_pred[1], yt_pred_one_expected)
        assert(yt_pred.shape == (2,10))
