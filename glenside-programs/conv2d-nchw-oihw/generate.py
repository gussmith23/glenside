import tvm
from tvm import relay
import numpy as np

activations_shape = (1, 3,  32, 32, )
weights_shape = (8, 3, 3, 3)
padding = (1, 1)
strides = (1, 1)

activations_var = relay.var(
    'activations', shape=activations_shape, dtype='float64')
weights_var = relay.var('weights', shape=weights_shape, dtype='float64')
program = relay.nn.conv2d(activations_var,
                          weights_var,
                          data_layout="NCHW",
                          kernel_layout="OIHW",
                          padding=padding)
program = relay.Function([activations_var, weights_var], program)
mod = tvm.IRModule.from_expr(program)

activations = np.random.rand(*activations_shape).astype('float64')
weights = np.random.rand(*weights_shape).astype('float64')
output = relay.create_executor(mod=mod).evaluate()(
    activations, weights).asnumpy()

with open('activations.npy', 'wb') as f:
    np.save(f, activations)
with open('weights.npy', 'wb') as f:
    np.save(f, weights)
with open('output.npy', 'wb') as f:
    np.save(f, output)
