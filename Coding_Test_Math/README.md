# Coding_Test Math References

This directory is an added, no-PyTorch companion to `Coding_Test`.

The original files under `Coding_Test` are left untouched.  The files here
mirror the reference-answer interfaces with plain Python lists used as tensors:

- vector: `[d]`
- sequence/batch tensor: nested lists such as `[batch][seq][dim]`
- linear weight: `[out_features][in_features]`, matching the mathematical
  form `y = W x + b`

Dropout is treated as the identity map, which is the deterministic evaluation
case used by the original tests.

Main locations:

- `Examples/`: pure Python mathematical reference implementations.
- Problem folders such as `Layer_Norm/` and `GRPO_Loss/`: thin entry files that
  re-export the matching implementation from `Examples/`.

These files are intended for reading, interview explanation, and lightweight
formula checks without importing PyTorch.
