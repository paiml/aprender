// TOML-Based Syscall Clustering (Section 6.1 of single-shot-compile-tooling-spec.md)
//
// This module implements configurable, user-defined syscall clustering to replace
// hardcoded pattern matching. Addresses Open-Closed Principle violation identified
// in Toyota Way review.
//
// Scientific Foundation:
// [3] Kuhn, A., Ducasse, S., & Gîrba, T. (2007). Semantic clustering: Identifying
//     topics in source code. Information and Software Technology, 49(3).
//
// Key Innovation: Configuration-driven clustering allows future-proofing (mmap3,
// clone3) and domain-specific patterns (GPU, ML) without recompilation.

mod definition;
mod filter;
mod registry;

pub use definition::{ArgsFilter, ClusterDefinition, Severity};
pub use registry::{ClusterRegistry, FdTable};

#[cfg(test)]
mod tests;
