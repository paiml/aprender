//! Command implementations
//!
//! Each command follows Toyota Way principles:
//! - Genchi Genbutsu: Go and see the actual data
//! - Jidoka: Stop on quality issues
//! - Visualization: Make problems visible

pub(crate) mod canary;
pub(crate) mod chat;
pub(crate) mod compare_hf;
pub(crate) mod convert;
pub(crate) mod debug;
pub(crate) mod diff;
pub(crate) mod explain;
pub(crate) mod export;
pub(crate) mod flow;
pub(crate) mod hex;
pub(crate) mod import;
pub(crate) mod inspect;
pub(crate) mod lint;
pub(crate) mod merge;
pub(crate) mod probar;
pub(crate) mod profile;
pub(crate) mod run;
pub(crate) mod serve;
pub(crate) mod tensors;
pub(crate) mod trace;
pub(crate) mod tree;
pub(crate) mod tui;
pub(crate) mod validate;
