//! Extracted layer operations for CudaExecutor
//!
//! Split from layer.rs (PMAT-802) to reduce module size while maintaining
//! performance through #[inline(always)] on critical paths.

mod batched;
mod cublas_prefill;
mod ffn;
/// #4971: the prefill's first token with the logits its argmax read.
mod first_token_logits;
mod forward;
mod graph_decode;
mod graphed;
mod indexed;
mod manual_graph;
mod prefill;

pub use ffn::{fused_ffn_swiglu_gpu, fused_ffn_swiglu_gpu_true_dp4a};
