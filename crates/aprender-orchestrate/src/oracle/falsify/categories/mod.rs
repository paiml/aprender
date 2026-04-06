//! Falsification Test Categories
//!
//! Each category provides specific types of falsification tests.

mod boundary;
mod concurrency;
mod invariant;
mod numerical;
mod parity;
mod resource;

// Public API for category types - used by external consumers
#[allow(unused_imports)]
pub use boundary::BoundaryCategory;
#[allow(unused_imports)]
pub use concurrency::ConcurrencyCategory;
#[allow(unused_imports)]
pub use invariant::InvariantCategory;
#[allow(unused_imports)]
pub use numerical::NumericalCategory;
#[allow(unused_imports)]
pub use parity::ParityCategory;
#[allow(unused_imports)]
pub use resource::ResourceCategory;
