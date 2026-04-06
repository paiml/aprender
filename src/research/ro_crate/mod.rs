//! RO-Crate bundling (ENT-026)
//!
//! Provides Research Object Crate (RO-Crate) packaging for
//! FAIR-compliant research data packaging.

mod descriptor;
mod entity;
mod package;

pub use descriptor::{RoCrateDescriptor, RO_CRATE_CONTEXT};
pub use entity::{EntityType, RoCrateEntity};
pub use package::{guess_mime_type, RoCrate};

#[cfg(test)]
mod tests;
