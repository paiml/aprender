//! A fixture crate: two modules, one inline, one re-export.
pub mod nn;
mod hidden {
    pub fn private_helper() {}
}
pub use hidden::private_helper as helper;
