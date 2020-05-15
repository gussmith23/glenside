import numpy as np
import pickle

a_val = np.load('single_matrix_multiply_input_a.npy')
b_val = np.load('single_matrix_multiply_input_b.npy')
#a_val = np.random.rand(64, 64).astype('float64')
#np.save('single_matrix_multiply_input_a.npy', a_val)
#b_val = np.random.rand(64, 64).astype('float64')
#np.save('single_matrix_multiply_input_b.npy', b_val)

out_val = np.matmul(a_val, b_val)
#np.save('single_matrix_multiply_output.npy', out_val)
print(out_val)

out_true = np.load('single_matrix_multiply_output.npy')
np.testing.assert_equal(out_true, out_val)
