//! Indexing data structures for efficient nearest neighbor search.
//!
//! This module provides approximate nearest neighbor search algorithms
//! optimized for production ML workloads.
//!
//! # Algorithms
//!
//! - **HNSW** (Hierarchical Navigable Small World): O(log n) approximate search
//!
//! # Quick Start
//!
//! ```
//! use aprender::index::hnsw::HNSWIndex;
//! use aprender::primitives::Vector;
//!
//! // Create index with M=16 connections per node
//! let mut index = HNSWIndex::new(16, 200, 0.0);
//!
//! // Add vectors at different angles (cosine distance measures angle)
//! index.add("horizontal", Vector::from_slice(&[1.0, 0.0, 0.0]));
//! index.add("diagonal", Vector::from_slice(&[1.0, 1.0, 0.0]));
//! index.add("vertical", Vector::from_slice(&[0.0, 1.0, 0.0]));
//!
//! // Search for 2 nearest neighbors to nearly horizontal vector
//! let query = Vector::from_slice(&[0.9, 0.1, 0.0]);
//! let results = index.search(&query, 2);
//!
//! assert_eq!(results.len(), 2);
//! // Results are sorted by cosine distance (closest first)
//! assert!(results[0].1 <= results[1].1);
//! ```

pub mod hnsw;

pub use hnsw::HNSWIndex;
