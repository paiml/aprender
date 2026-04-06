//! Advanced activation functions: leaky_relu, elu, gelu, swish, hardswish, mish, selu
//!
//! These are separated from the basic activations (softmax, log_softmax, relu, sigmoid)
//! for file size management while remaining part of the same `activations` module.
//!
//! ## Sub-modules
//!
//! - [`parametric`]: Parametric activations (leaky_relu, elu, selu)
//! - [`smooth`]: Smooth self-gated activations (gelu, swish, hardswish, mish)

mod parametric;
mod smooth;
