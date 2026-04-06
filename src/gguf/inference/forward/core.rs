//! Forward pass implementations for OwnedQuantizedModel
//!
//! Contains forward, forward_cached methods.
//! These are the core inference entry points.

use crate::brick::BrickProfiler;
use crate::error::Result;
use crate::gguf::ops;
use crate::gguf::OwnedQuantizedModel;

include!("forward_cached.rs");
