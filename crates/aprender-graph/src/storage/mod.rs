//! Graph storage layer
//!
//! Provides CSR (Compressed Sparse Row) graph representation and Parquet persistence.

pub mod csr;

#[cfg(feature = "storage")]
pub mod parquet;

pub use csr::{CsrGraph, NodeId};
