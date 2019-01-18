import unittest
import numpy as np
from languagemodel.model import clip

class LanguageModelText(unittest.TestCase):

    def is_close_enough(self, a, b, epsilon):
        return abs(a - b) <= epsilon

    def test_gradient_clipping(self):
        '''
        Tests gradient clipping
        '''

        np.random.seed(3)
        dWax = np.random.randn(5, 3) * 10
        dWaa = np.random.randn(5, 5) * 10
        dWya = np.random.randn(2, 5) * 10
        db = np.random.randn(5, 1) * 10
        dby = np.random.randn(2, 1) * 10
        gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
        gradients = clip(gradients, 10)

        matching_tolerance = 0.00001
        assert(gradients["dWaa"][1][2] == 10)
        assert(gradients["dWax"][3][1] == -10)
        assert(self.is_close_enough(gradients["dWya"][1][2], 0.29713815361, matching_tolerance))
        assert(gradients["db"][4] == 10)
        assert(self.is_close_enough(gradients["dby"][1], 8.45833407, matching_tolerance))
