//! Content-addressed storage for Pacha artifacts.
//!
//! This module provides BLAKE3-based content addressing for deduplication
//! and tamper detection.

mod content_address;
mod object_store;

pub use content_address::{Compression, ContentAddress};
pub use object_store::ObjectStore;
