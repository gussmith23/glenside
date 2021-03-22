import tvm
from tvm import relay
import numpy as np

a_shape = (32, 64)
b_shape = (64, 16)

a_var = relay.var(
    'activations', shape=a_shape, dtype='float64')
b_var = relay.var('weights', shape=b_shape, dtype='float64')
program = relay.nn.dense(a_var,
                         relay.transpose(b_var))

program = relay.Function([a_var, b_var], program)
mod = tvm.IRModule.from_expr(program)

a = np.random.rand(*a_shape).astype('float64')
b = np.random.rand(*b_shape).astype('float64')
output = relay.create_executor(mod=mod).evaluate()(a, b).asnumpy()

with open('a.npy', 'wb') as f:
    np.save(f, a)
with open('b.npy', 'wb') as f:
    np.save(f, b)
with open('output.npy', 'wb') as f:
    np.save(f, output)
