# Glenside Program Examples

This folder contains
  a collection of example Glenside programs
  with their inputs and expected outputs.
These serve two purposes:
  first, as user-facing examples,
  and second, as test cases.
Each of these are automatically run
  by the `run-glenside-examples.rs`
  integration test.

## Adding a New Example

Inputs and outputs should be `.npy`-encoded tensors of datatype `f64`.
Currently, `run-glenside-examples.rs`
  expects

1. A single file with a `.glenside` extension, which is the source of the Glenside program,
2. A single file named `output.npy`, which is a NumPy array with `dtype=float64` of the expected result values, and finally
3. One file named `<symbol>.npy` for each symbol appearing in the program. These should also be NumPy arrays with `dtype=float64`.
