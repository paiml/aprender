//! Auto-generated contract traits for compiler-enforced binding verification.
//!
//! Each trait corresponds to a YAML contract. Consumer crates `impl` the
//! trait to prove they have the required functions with correct signatures.
//!
//! Missing method = compile error. Wrong signature = compile error.
//!
//! Regenerate all: `for f in contracts/*.yaml; do pv scaffold --trait "$f" -o ...; done`
//!
//! See: docs/specifications/sub/contract-trait-enforcement.md (Section 23)

pub mod activation_kernel_v1;
pub mod adamw_kernel_v1;
pub mod attention_kernel_v1;
pub mod cross_entropy_kernel_v1;
pub mod flash_attention_v1;
pub mod gqa_kernel_v1;
pub mod layernorm_kernel_v1;
pub mod matmul_kernel_v1;
pub mod rmsnorm_kernel_v1;
pub mod rope_kernel_v1;
pub mod silu_kernel_v1;
pub mod softmax_kernel_v1;
pub mod swiglu_kernel_v1;

// Re-export all traits for convenience
pub use activation_kernel_v1::ActivationKernelV1;
pub use adamw_kernel_v1::AdamwKernelV1;
pub use attention_kernel_v1::AttentionKernelV1;
pub use cross_entropy_kernel_v1::CrossEntropyKernelV1;
pub use flash_attention_v1::FlashAttentionV1;
pub use gqa_kernel_v1::GqaKernelV1;
pub use layernorm_kernel_v1::LayernormKernelV1;
pub use matmul_kernel_v1::MatmulKernelV1;
pub use rmsnorm_kernel_v1::RmsnormKernelV1;
pub use rope_kernel_v1::RopeKernelV1;
pub use silu_kernel_v1::SiluKernelV1;
pub use softmax_kernel_v1::SoftmaxKernelV1;
pub use swiglu_kernel_v1::SwigluKernelV1;
