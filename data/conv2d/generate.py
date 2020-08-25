import tvm
from tvm import relay
import numpy as np

image_shape =  [1, 32, 32, 3]
filter_shape = [3, 3, 3, 8]

image_var = relay.var('image', shape=image_shape, dtype='float32')
weights_var = relay.var('weights', shape=filter_shape, dtype='float32')
program = relay.nn.conv2d(image_var, weights_var, data_layout="NHWC", kernel_layout="HWIO")
program = relay.Function([image_var, weights_var], program)
mod = relay.module.Module.from_expr(program)

image = np.random.rand(*image_shape).astype('float32')
weights = np.random.rand(*filter_shape).astype('float32')

output = relay.create_executor(mod=mod).evaluate()(image,weights)
