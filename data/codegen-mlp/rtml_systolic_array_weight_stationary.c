#include <assert.h>
#include <stdio.h>

void rtml_systolic_array_weight_stationary(int hardware_id, float *out,
                                           float *activations, float *weights,
                                           int input_vector_size,
                                           int output_vector_size, int batch) {
  fprintf(stderr, "Running systolic array, hardware id %d\n", hardware_id);

  int batch_i;
  for (batch_i = 0; batch_i < batch; ++batch_i) {
    int col;
    for (col = 0; col < output_vector_size; ++col) {
      out[batch_i*output_vector_size + col] = 0.0;
      int row;
      for (row = 0; row < input_vector_size; ++row) {
        out[batch_i*output_vector_size + col] += activations[batch_i*input_vector_size + row] * weights[row * output_vector_size + col];
      }
    }
  }
}
