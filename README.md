# mentat

A machine learning library written in OCaml for OCaml. (idk what to add rn)

## Dependencies

- OCaml >= 4.14
- OpenBLAS
- dune >= 3.0

```bash
# macOS
brew install openblas

# Ubuntu/Debian
sudo apt install libopenblas-dev

# Fedora
sudo dnf install openblas-devel
```

## Building

```bash
dune build
```

## What's in it

- N-dimensional tensors
- Broadcasting for elementwise ops (`+`, `-`, `*`)
- Slicing and strided views without copying
- Matrix multiplication via SGEMM (OpenBLAS)
- Random initialization with optional seed

## Planned
- Autograd engine — reverse-mode automatic differentiation
- More BLAS bindings (`sdot`, `snrm2`, `saxpy` for gradient updates)
- Batched matmul
- Softmax, relu, and other activations
- A basic MLP to test everything end to end
