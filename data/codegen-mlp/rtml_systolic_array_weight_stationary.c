#include <assert.h>
#include <stdio.h>

void rtml_systolic_array_weight_stationary(int hardware_id, float *out,
                                           float *activations,
                                           int activations_dim_0,
                                           float *weights, int weights_dim_0,
                                           int weights_dim_1) {
  assert(activations_dim_0 == weights_dim_0);
  fprintf(stderr, "Running systolic array, hardware id %d\n", hardware_id);

  for (int col = 0; col < weights_dim_1; ++col) {
    out[col] = 0.0;
    for (int row = 0; row < activations_dim_0; ++row) {
      out[col] += activations[row] * weights[row * weights_dim_1 + col];
    }
  }
}
