# PROTOQUANT - Dynamic Quantization with Tensor Subclassing

The protoquant package provides dynamic vector-wise
quantization and quantized arithmetic using torch.tensor subclassing.

This dynamnic quantization support is directed at a broad range of
applications, and currently tested with the PyTorch Transformner API
and Better Transformers implementation with a focus on GPU inference.

The focus on testing for Transformer Inference is non-limiting and
protoquant is broadly applicable to support broad uses for using
dynamic inference with PyTorch.


## Installation

You need to clone the repo with recursive submodules.

`git clone --recurse-submodules https://github.com/facebookexperimental/protoquant.git`

If you forget to, you can always fix this using [this
trick](https://gist.github.com/cnlohr/04de6edd3e2a75face0a68c53be2017e)

Once the repository is cloned, you will NEED to be on a GPU machine, and
then `pip install -e .` works.

If you really want to compile on a CPU machine,
[see here](https://github.com/pytorch/extension-cpp/issues/71#issuecomment-1183674660)

## License

MIT
