//! A fixture crate split with `include!`, the way the tree splits large modules.
include!("split_a.rs");
include!("missing.rs");
pub mod inner {
    include!("inner_body.rs");
}
