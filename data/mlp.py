"""Multilayer perceptron that should be equivalent to our MLP in Rust"""

import numpy as np
import pickle

in_val = np.load('in.npy')
w1_val = np.load('w1.npy')
w2_val = np.load('w2.npy')
w3_val = np.load('w3.npy')

out_val = in_val
out_val = np.matmul(out_val, w1_val)
out_val = np.matmul(out_val, w2_val)
out_val = np.matmul(out_val, w3_val)
out_val = np.squeeze(out_val)
print(out_val)

out_true = np.load('out.npy')
np.testing.assert_equal(out_true, out_val)
