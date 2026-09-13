//! The unresolvable case: this crate splices a GENERATED file from OUT_DIR, so
//! nothing in the tree names src/mystery.rs. Not a reader itself.
include!(concat!(env!("OUT_DIR"), "/generated_mystery.rs"));

pub fn nothing() {}
