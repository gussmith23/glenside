import tvm
from tvm import relay
import numpy as np

image_shape = [1, 32, 32, 3]
filter_shape = [3, 3, 3, 8]
padding = (1, 1)
strides = (1, 1)

image_var = relay.var('image', shape=image_shape, dtype='float32')
weights_var = relay.var('weights', shape=filter_shape, dtype='float32')
program = relay.nn.conv2d(image_var,
                          weights_var,
                          data_layout="NHWC",
                          kernel_layout="HWIO",
                          padding=padding)
program = relay.Function([image_var, weights_var], program)
mod = relay.module.Module.from_expr(program)

image = np.random.rand(*image_shape).astype('float32')
weights = np.random.rand(*filter_shape).astype('float32')

output = relay.create_executor(mod=mod).evaluate()(image, weights).asnumpy()

# Write shapes file
with open('conv2d-shapes.json', 'w') as f:
    f.write("""
{{
    "image" : {},
    "weights" : {}
}}
    """.strip().format(image_shape, filter_shape))

# Write glenside file
with open('conv2d.glenside', 'w') as f:
    f.write("""
(access-transpose
  (compute
    dot-product
    (access-cartesian-product
      (access (access-transpose (access (access-tensor weights) 0) (list 3 0 1 2)) 1)
      (access
        (access-squeeze
          (access-squeeze
            (access-windows
              (access-pad
                (access-pad (access (access-tensor image) 4) zero-padding 1 {h_pad} {h_pad})
                zero-padding
                2
                {w_pad}
                {w_pad})
              (shape-insert-axis (shape-remove-axis (shape-of weights) 3) 0)
              (shape 1 {h_stride} {w_stride} 1))
            3)
          3)
        3)))
  (list 1 2 3 0)
  )
    """.strip().format(h_pad=padding[0],
                       w_pad=padding[1],
                       h_stride=strides[0],
                       w_stride=strides[1]))

with open('conv2d-test-harness.c', 'w') as f:
    f.write("""
#include "conv2d.c"
#include <assert.h>
#include <math.h>
#include <stdio.h>

float image{input_size} = {input};
float weights{weights_size} = {weights};
float out{out_size} = {out_zeros};
float expected_out{out_size} = {output};

int main() {{
  conv2d(out, image, weights);

  // Ensure result is what we expect.
  int i;
  for (i = 0; i < {output_num_els}; ++i) {{
    fprintf(stderr, "%f ?= %f\\n", out[i], expected_out[i]);
    assert(fabs(out[i] - expected_out[i]) < 0.00001);
  }}

  return 0;
}}

    """.strip().format(
        input_size="".join(["[{}]".format(dim) for dim in image.shape]),
        input=np.array2string(image,
                              separator=', ', edgeitems=float("inf")).replace(
                                  '[', '{ ').replace(']', ' }'),
        weights_size="".join(["[{}]".format(dim) for dim in weights.shape]),
        weights=np.array2string(weights,
                                separator=', ',
                                edgeitems=float("inf")).replace('[',
                                                                '{ ').replace(
                                                                    ']', ' }'),
        out_size="".join(["[{}]".format(dim) for dim in output.shape]),
        out_zeros=np.array2string(np.zeros(output.shape),
                                  separator=', ',
                                  edgeitems=float("inf")).replace(
                                      '[', '{ ').replace(']', ' }'),
        output=np.array2string(output,
                               separator=', ', edgeitems=float("inf")).replace(
                                   '[', '{ ').replace(']', ' }'),
        output_num_els=output.size))
