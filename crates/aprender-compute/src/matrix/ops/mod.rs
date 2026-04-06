//! Matrix operation modules
//!
//! This module organizes Matrix operations into focused submodules (PMAT-018):
//!
//! ## Domain Separation
//!
//! - **storage**: Memory allocation and element access (constructors, accessors)
//! - **arithmetic**: Matrix multiplication and batched operations
//! - **linear**: Linear algebra (transpose, matvec, vecmat)
//! - **ml_ops**: Machine learning operations (convolution, embedding, pooling)
//!
//! This separation allows each domain to be optimized and tested independently.
//! A matrix's memory layout is independent of its mathematical operations.

pub(super) mod arithmetic;
pub(super) mod linear;
pub(super) mod ml_ops;
pub(super) mod storage;
