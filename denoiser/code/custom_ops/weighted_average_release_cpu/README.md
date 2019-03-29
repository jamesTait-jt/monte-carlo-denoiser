# Local weighted average op for Tensorflow

1. Run `make` to compile the C++ code.
2. Import the op with `from weighted_average import weighted_average`

You can use it like

```python
denoised = combine_neighborhoods(noisy_color, weights)
```

where noisy_color and weights are 4-dimensional tensors,

```
shape(noisy_color) = [b, h, w, 3]
shape(weights)     = [b, h - 2*r, w - 2*r, (2*r+1)^2]
```

where the kernel size is `(2r+1)^2`
