import numpy as np

in_len = 2
layer_0_len = 4
layer_1_len = 6
out_len = 2

input = np.random.rand(in_len)
weights_0 = np.random.rand(in_len, layer_0_len)
weights_1 = np.random.rand(layer_0_len, layer_1_len)
weights_2 = np.random.rand(layer_1_len, out_len)
out = np.zeros(out_len)
expected_out = np.matmul(np.matmul(np.matmul(input, weights_0), weights_1), weights_2)

for array, name in [(input, "input"), (weights_0, 'weights_0'),
                    (weights_1, 'weights_1'), (weights_2, 'weights_2'),
                    (out, 'out'), (expected_out, 'expected_out')]:
    print("float {}{} = {};".format(
        name, "".join(("[{}]".format(shape) for shape in array.shape)),
        np.array2string(array, separator=', ').replace('[', '{{ ').replace(']', ' }}')))
