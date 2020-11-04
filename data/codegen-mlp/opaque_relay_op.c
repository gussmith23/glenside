#include <assert.h>
#include <float.h> // FLT_MIN
#include <math.h>  // needed for expf (for softmax)

/*
   Each batchnorm has 4 parameters decided during training, for each minibatch:
   var, gamma, mu, beta
   mu and var are the mean and variance of the minibatch (calculated on the
   feature dimension for FC layers, and on the channel dimension for conv
   layers) gamma and beta are hyperparameters decided during traning (same shape
   as X)

   The full batchnorm equation is
   y = (x - mu) * 1/sqrt(var+epsilon) * gamma + beta

   (epsilon is a small constant to avoid dividing by zero.)
   The coefficient tensor (1/sqrt(var+epsilon) * gamma) can be pre-computed,
   here called "coeff".

   input tensors X is assumed to be NHWC
   output tensor Y should be the same shape, also NHWC
   coeff is same shape as X
   mu is length C
   beta is same shape as X
*/
void batchNormInference(float *X, float *Y, int N, int H, int W, int C,
                        float *gamma, float *beta, float *mu, float *var,
                        float epsilon) {
  // mu is calculated on the channel dimension
  // coeff and beta have the same shape as X

  // Prepare dimensional constants
  int dim2 = C;
  int dim1 = W * dim2;
  int dim0 = H * dim1;

  for (int n = 0; n < N; n++) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        for (int c = 0; c < C; c++) {
          Y[n * dim0 + h * dim1 + w * dim2 + c] =
              (X[n * dim0 + h * dim1 + w * dim2 + c] - mu[c]) *
                  (1 / sqrt(var[n * dim0 + h * dim1 + w * dim2 + c] + epsilon) *
                   gamma[n * dim0 + h * dim1 + w * dim2 + c]) +
              beta[n * dim0 + h * dim1 + w * dim2 + c];
        }
      }
    }
  }

  return;
}

/*
Compute the 1D softmax of input tensor X and store it in output tensor Y.
Both tensors are length N.
Y = exp(X) / sum(exp(X))
 */
void softmax1D(float *X, float *Y, int N) {
  float sum = 0;
  float max = FLT_MIN;
  for (int i = 0; i < N; i++) {
    max = fmax(max, X[i]);
  }
  for (int i = 0; i < N; i++) {
    float val = expf(X[i] - max);
    Y[i] = val;
    sum += val;
  }

  for (int i = 0; i < N; i++) {
    Y[i] = Y[i] / sum;
  }

  return;
}

void softmax(float *X, float *Y, int N, int H, int W, int C) {
  return softmax1D(X, Y, N * H * W * C);
}

/*
Compute a 1-dimensional ReLU on input tensor X, storing in output tensor Y.
Both are length N.
 */
void relu1D(float *X, float *Y, int N) {
  for (int i = 0; i < N; i++)
    Y[i] = (X[i] > 0) ? X[i] : 0;
}

void relu(float *X, float *Y, int N, int H, int W, int C) {
  return relu1D(X, Y, N * H * W * C);
}

/*
Compute the average of each spatial filter (HW), collapsing it to a single
value.

Input tensor X is in format NHWC, while output tensor Y is NC.
 */
void globalAvgPool(float *X, float *Y, int N, int H, int W, int C) {

  int total = H * W;
  int Ylen = N * C;

  // Prepare dimensional constants
  int dim2 = C;
  int dim1 = W * dim2;
  int dim0 = H * dim1;

  // zero out Y
  for (int i = 0; i < Ylen; i++)
    Y[i] = 0;

  // collapse HW
  for (int n = 0; n < N; n++)
    for (int h = 0; h < H; h++)
      for (int w = 0; w < W; w++)
        for (int c = 0; c < C; c++) {
          Y[n * C + c] += X[n * dim0 + h * dim1 + w * dim2 + c];
        }

  // average
  for (int i = 0; i < Ylen; i++)
    Y[i] = Y[i] / total;
}

// element-wise add, currently does not support broadcasting
void add1D(float *X, float *Y, float *out, int N) {
  for (int i = 0; i < N; i++)
    out[i] = X[i] + Y[i];
}
void add(float *X, float *Y, float *out, int N, int H, int W, int C) {
  add1D(X, Y, out, N * H * W * C);
}

// Need a function to get max val of window size (3x3)
#define max2(x, y) ((x > y) ? x : y)
#define max4(x1, x2, x3, x4) max2(x1, max2(x2, max2(x3, x4)))

#define max6(x1, x2, x3, x4, x5, x6) max2(max4(x1, x2, x3, x4), max2(x5, x6))
#define max9(x1, x2, x3, x4, x5, x6, x7, x8, x9)                               \
  max2(x1,                                                                     \
       max2(x2,                                                                \
            max2(x3, max2(x4, max2(x5, max2(x6, max2(x7, max2(x8, x9))))))))
/*
   2d maxpool with a 3x3 filter, specialized for Resnet18
   input tensor X NHWC: (1, 112, 112, 64)
   output tensor Y NHWC: (1, 56, 56, 64)
   stride of 2, padding (1, 1)
*/
void maxpool2D3x3_resnet18_op6(float *X, float *Y) {
  // compute dimensional constants
  int Xdim1 = 64;          // C
  int Xdim2 = 112 * Xdim1; // W*C
  int Ydim1 = 64;          // C
  int Ydim2 = 56 * Ydim1;  // W*C
  for (int c = 0; c < 64; c++) {
    int inw = 0;
    int inh = 0;
    // handle first entry
    // top row and left column of input are both zero b/c of padding
    Y[c] = max4(X[(inh)*Xdim2 + (inw)*Xdim1 + c],
                X[(inh)*Xdim2 + (inw + 1) * Xdim1 + c],
                X[(inh + 1) * Xdim2 + (inw)*Xdim1 + c],
                X[(inh + 1) * Xdim2 + (inw + 1) * Xdim1 + c]);

    inw = 1;
    // handle first row after first entry
    // first row of input is zero
    for (int outw = 1; outw < 56; outw++) {
      Y[outw * Ydim1 + c] = max6(X[(inh)*Xdim2 + inw * Xdim1 + c],
                                 X[(inh)*Xdim2 + (inw + 1) * Xdim1 + c],
                                 X[(inh)*Xdim2 + (inw + 2) * Xdim1 + c],
                                 X[(inh + 1) * Xdim2 + inw * Xdim1 + c],
                                 X[(inh + 1) * Xdim2 + (inw + 1) * Xdim1 + c],
                                 X[(inh + 1) * Xdim2 + (inw + 2) * Xdim1 + c]);
      inw = inw + 2;
    }
    // populate Y except for first row
    inh = 1;
    for (int outh = 1; outh < 56; outh++) {
      inw = 0;
      // handle first column: all zeros
      Y[outh * Ydim2 + c] = max6(X[inh * Xdim2 + (inw)*Xdim1 + c],
                                 X[inh * Xdim2 + (inw + 1) * Xdim1 + c],
                                 X[(inh + 1) * Xdim2 + (inw)*Xdim1 + c],
                                 X[(inh + 1) * Xdim2 + (inw + 1) * Xdim1 + c],
                                 X[(inh + 2) * Xdim2 + (inw)*Xdim1 + c],
                                 X[(inh + 2) * Xdim2 + (inw + 1) * Xdim1 + c]);
      inw = 1;
      // move through remaining columns for normal computation
      for (int outw = 1; outw < 56; outw++) {
        Y[outh * Ydim2 + outw * Ydim1 + c] =
            max9(X[inh * Xdim2 + inw * Xdim1 + c],
                 X[inh * Xdim2 + (inw + 1) * Xdim1 + c],
                 X[inh * Xdim2 + (inw + 2) * Xdim1 + c],
                 X[(inh + 1) * Xdim2 + inw * Xdim1 + c],
                 X[(inh + 1) * Xdim2 + (inw + 1) * Xdim1 + c],
                 X[(inh + 1) * Xdim2 + (inw + 2) * Xdim1 + c],
                 X[(inh + 2) * Xdim2 + inw * Xdim1 + c],
                 X[(inh + 2) * Xdim2 + (inw + 1) * Xdim1 + c],
                 X[(inh + 2) * Xdim2 + (inw + 2) * Xdim1 + c]);
        inw = inw + 2;
      }
      inh = inh + 2;
    }
  }
}

void add_with_broadcasting(float *out, float *a, float *b, int *out_shape,
                           int out_ndims, int *a_shape, int a_ndims,
                           int *b_shape, int b_ndims) {
  // For every location in the output, calculate the indices into a and b that
  // are being added.

  assert(out_ndims > 0);
  assert(a_ndims > 0);
  assert(b_ndims > 0);
  assert(a_ndims <= out_ndims);
  assert(b_ndims <= out_ndims);

  // Calculate the number of elements in the output
  int out_len = 1;
  for (int dim_i = 0; dim_i < out_ndims; ++dim_i) {
    assert(out_shape[dim_i] > 0);
    out_len *= out_shape[dim_i];
  }

  int a_len = 1;
  for (int dim_i = 0; dim_i < a_ndims; ++dim_i) {
    assert(a_shape[dim_i] > 0);
    a_len *= a_shape[dim_i];
  }

  int b_len = 1;
  for (int dim_i = 0; dim_i < b_ndims; ++dim_i) {
    assert(b_shape[dim_i] > 0);
    b_len *= b_shape[dim_i];
  }

  // Check that broadcasting is possible
  for (int dim_i = 0; dim_i < out_ndims; ++dim_i) {
    // We align the RIGHTMOST dimensions in the tensor shapes with each other.
    // So if the output shape has 6 dimensions and a has 4, then dim_i=4 indexes
    // into dimension 4 (the 5th dimension) of the out tensor, but dimension 2
    // (The 3rd dimension) of tensor a. If the index is out of range, then the
    // value is 1.
    int dim_val_a = 0;
    assert(dim_i - (out_ndims - a_ndims) < a_ndims);
    if (dim_i - (out_ndims - a_ndims) >= 0)
      dim_val_a = a_shape[dim_i - (out_ndims - a_ndims)];
    else
      dim_val_a = 1;
    assert(dim_val_a > 0);

    int dim_val_b = 0;
    assert(dim_i - (out_ndims - b_ndims) < b_ndims);
    if (dim_i - (out_ndims - b_ndims) >= 0)
      dim_val_b = b_shape[dim_i - (out_ndims - b_ndims)];
    else
      dim_val_b = 1;
    assert(dim_val_b > 0);

    // Either a == b, or one of them is 1.
    assert(dim_val_a == dim_val_b || ((dim_val_a == 1) || (dim_val_b == 1)));
    // One of them should be equal to the output dimension.
    assert(dim_val_a == out_shape[dim_i] || dim_val_b == out_shape[dim_i]);
  }

  for (int i = 0; i < out_len; ++i) {
    int out_remaining = i;
    int out_step = out_len;
    int a_index = 0;
    int b_index = 0;
    int a_step = a_len;
    int b_step = b_len;
    for (int dim_i = 0; dim_i < out_ndims; ++dim_i) {
      int dim_val_out = out_shape[dim_i];
      assert(dim_val_out > 0);

      // Calculate the value that we're indexing this dimension at, in the out
      // tensor. out_step is the stride for this dimension. We calculate it by
      // always first dividing by the current dimension value.
      assert(out_step % dim_val_out == 0);
      out_step /= dim_val_out;
      int this_dimension_index_val = out_remaining / out_step;
      out_remaining %= out_step;

      // We align the RIGHTMOST dimensions in the tensor shapes with each other.
      // So if the output shape has 6 dimensions and a has 4, then dim_i=4
      // indexes into dimension 4 (the 5th dimension) of the out tensor, but
      // dimension 2 (The 3rd dimension) of tensor a. If the index is out of
      // range, then the value is 1.
      int dim_val_a = 0;
      assert(dim_i - (out_ndims - a_ndims) < a_ndims);
      if (dim_i - (out_ndims - a_ndims) >= 0)
        dim_val_a = a_shape[dim_i - (out_ndims - a_ndims)];
      else
        dim_val_a = 1;
      assert(dim_val_a > 0);
      assert(dim_val_a == 1 || this_dimension_index_val < dim_val_a);

      int dim_val_b = 0;
      assert(dim_i - (out_ndims - b_ndims) < b_ndims);
      if (dim_i - (out_ndims - b_ndims) >= 0)
        dim_val_b = b_shape[dim_i - (out_ndims - b_ndims)];
      else
        dim_val_b = 1;
      assert(dim_val_b > 0);
      assert(dim_val_b == 1 || this_dimension_index_val < dim_val_b);

      int this_dimension_index_val_a = 0;
      if (dim_val_a > 1)
        this_dimension_index_val_a = this_dimension_index_val;

      int this_dimension_index_val_b = 0;
      if (dim_val_b > 1)
        this_dimension_index_val_b = this_dimension_index_val;

      assert(a_step % dim_val_a == 0);
      a_step /= dim_val_a;
      a_index += a_step * this_dimension_index_val_a;

      assert(b_step % dim_val_b == 0);
      b_step /= dim_val_b;
      b_index += b_step * this_dimension_index_val_b;
    }

    // At this point, a_index and b_index should be correctly calculated, so we
    // can finally do the addition.
    assert(i < out_len);
    assert(a_index < a_len);
    assert(b_index < b_len);
    out[i] = a[a_index] + b[b_index];
  }
}
