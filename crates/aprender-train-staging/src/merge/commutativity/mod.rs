//! ENT-031: Merge commutativity property tests
//!
//! Tests mathematical properties of merge algorithms:
//! - Commutativity: order-independence
//! - Permutation invariance: reordering models
//! - Identity: merging with self
//! - Boundary conditions: endpoint behavior

mod helpers;

mod dare_tests;
mod edge_cases;
mod property_tests;
mod slerp_tests;
mod ties_tests;
