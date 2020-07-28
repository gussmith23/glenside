import tvm
from tvm import relay
from tvm.relay.testing import resnet
import numpy as np

image_shape = (3, 32, 32)

epsilon = 2e-5

model, params = resnet.get_workload(batch_size=1,
                                    num_classes=1000,
                                    num_layers=50,
                                    image_shape=image_shape,
                                    dtype='float32')

image = np.random.rand(1, *image_shape).astype('float32')
with open('image.npy', 'wb') as file:
    np.save(file, image)

# Preprocess mean
for mean_var in [
        'bn_data_moving_mean',
]:
    val = params[mean_var].asnumpy()
    val = -val
    with open(mean_var + '_negated' + '.npy', 'wb') as file:
        np.save(file, val)

# Preprocess variance
for variance_var in [
        'bn_data_moving_var',
]:
    val = params[variance_var].asnumpy()
    val = val + epsilon
    val = np.sqrt(val)
    val = 1 / val
    with open(variance_var + '_reciprocal_sqrt_plus_epsilon' + '.npy',
              'wb') as file:
        np.save(file, val)
        np.save(file, val)

for var in [
        'conv0_weight',
        'bn_data_gamma',
        'bn_data_beta',
]:
    with open(var + '.npy', 'wb') as file:
        np.save(file, params[var].asnumpy())


ex = relay.create_executor(mod=model)
result = ex.evaluate()(image, **params)

with open('result.npy', 'wb') as file:
    np.save(file, result.asnumpy())
