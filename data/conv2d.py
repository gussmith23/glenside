import numpy as np
import tvm
from tvm import relay

filter_x, filter_y = 3, 3
i, o = 32, 64
image_x, image_y = 512, 512
batch = 1

filters = np.random.rand(o, i, filter_x, filter_y).astype('float32')
activations = np.random.rand(batch, i, image_x, image_y).astype('float32')

filters_var = relay.var('filters',
                        shape=(o, i, filter_x, filter_y),
                        dtype='float32')
activations_var = relay.var('activations',
                            shape=(batch, i, image_x, image_y),
                            dtype='float32')

module = relay.module.Module.from_expr(
    relay.Function([activations_var, filters_var],
                   relay.nn.conv2d(activations_var, filters_var)))

ex = relay.create_executor(mod=module)

result = ex.evaluate()(activations, filters).asnumpy()

# with open('conv2d_filters.npy', 'wb') as file:
#     np.save(file, filters)
# # TODO(@gussmith23) Support batch dimension
# with open('conv2d_activations.npy', 'wb') as file:
#     np.save(file, activations[0,:,:,:])
# with open('conv2d_result.npy', 'wb') as file:
#     np.save(file, result)


