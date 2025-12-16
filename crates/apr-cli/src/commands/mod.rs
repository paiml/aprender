//! Command implementations
//!
//! Each command follows Toyota Way principles:
//! - Genchi Genbutsu: Go and see the actual data
//! - Jidoka: Stop on quality issues
//! - Visualization: Make problems visible

pub(crate) mod debug;
pub(crate) mod diff;
pub(crate) mod inspect;
pub(crate) mod probar;
pub(crate) mod tensors;
pub(crate) mod trace;
pub(crate) mod validate;
