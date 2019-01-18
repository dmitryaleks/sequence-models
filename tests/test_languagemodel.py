import unittest
import numpy as np
import os
from languagemodel.model import clip
from languagemodel.model import sample

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

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

    def test_sequence_sample(self):
        np.random.seed(2)
        _, n_a = 20, 100

        data = open(os.path.join(THIS_DIR, 'data/names.data'), 'r').read()
        data = data.lower()
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

        char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
        ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
        print(ix_to_char)

        Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
        b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
        parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}

        indices = sample(parameters, char_to_ix, 0)
        print("Sampling:")
        print("list of sampled indices:", indices)
        sampled_chars = [ix_to_char[i] for i in indices]
        print("list of sampled characters:", sampled_chars)

        sampled_chars_expected = ['l', 'q', 'x', 'n', 'm', 'i', 'j', 'v', 'x', 'f', 'm', 'k', 'l', 'f', 'u', 'o', 'u', 'n', 'c', 'b', 'a', 'u',
         'r', 'x', 'g', 'y', 'f', 'y', 'r', 'j', 'p', 'b', 'c', 'h', 'o', 'l', 'k', 'g', 'a', 'l', 'j', 'b', 'g', 'g',
         'k', 'q', 'x', 'l', 'c', '\n', '\n']

        assert(sampled_chars == sampled_chars_expected)
