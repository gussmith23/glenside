#include <assert.h>
#include <stdio.h>

void rtml_systolic_array_weight_stationary(int hardware_id, float *out,
                                           float *activations, float *weights,
                                           int input_vector_size,
                                           int output_vector_size, int batch) {
  // This is just true for now.
  assert(batch == 1);

  fprintf(stderr, "Running systolic array, hardware id %d\n", hardware_id);

  int col;
  for (col = 0; col < output_vector_size; ++col) {
    out[col] = 0.0;
    int row;
    for (row = 0; row < input_vector_size; ++row) {
      out[col] += activations[row] * weights[row * output_vector_size + col];
    }
  }
}
