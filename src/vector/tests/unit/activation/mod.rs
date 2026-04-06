//! Activation function tests - split into submodules for maintainability
//!
//! Submodules:
//! - `core_activations` - clip, softmax, log_softmax, relu, sigmoid, leaky_relu, elu
//! - `advanced_activations` - gelu, swish, hardswish, mish, selu
//! - `parallel_and_simd` - aligned vectors, parallel execution, AVX-512 path tests

mod advanced_activations;
mod core_activations;
mod parallel_and_simd;
