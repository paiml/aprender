//! GPU config, weights structures, encoder block, KV cache, GpuResidentTensor methods,
//! TransferStats, and kernel cache stats tests
//!
//! Split into submodules for maintainability:
//! - `weights`: Config creation, block weights structures, forward encoder block, KV cache
//! - `tensor`: GpuResidentTensor method coverage (transfer aliases, kernel launches, etc.)
//! - `weights_coverage`: Additional weights coverage tests (field sizes, Copy/Clone traits, etc.)

mod tensor;
mod weights;
mod weights_coverage;
