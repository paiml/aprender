//! Command implementations
//!
//! Each command follows Toyota Way principles:
//! - Genchi Genbutsu: Go and see the actual data
//! - Jidoka: Stop on quality issues
//! - Visualization: Make problems visible

pub mod bench;
pub mod canary;
pub mod cbtop;
pub mod chat;
pub mod check;
pub mod compare_hf;
pub(crate) mod convert;
pub(crate) mod debug;
pub(crate) mod diff;
pub(crate) mod distill;

pub(crate) mod eval;
pub(crate) mod explain;
pub(crate) mod export;
pub(crate) mod finetune;
pub(crate) mod flow;
pub(crate) mod hex;
pub(crate) mod import;
pub(crate) mod inspect;
pub(crate) mod lint;
pub(crate) mod merge;
pub(crate) mod oracle;
pub(crate) mod parity;
pub(crate) mod probar;
pub(crate) mod profile;
pub(crate) mod prune;
pub(crate) mod ptx_explain;
pub(crate) mod ptx_map;
pub(crate) mod publish;
pub(crate) mod pull;
pub(crate) mod qa;
pub(crate) mod qa_capability;
pub(crate) mod quantize;
pub(crate) mod rosetta;
pub(crate) mod run;
pub(crate) mod serve;
pub(crate) mod showcase;
pub(crate) mod tensors;
pub(crate) mod trace;
pub(crate) mod tree;
pub(crate) mod tui;
pub(crate) mod tune;
pub(crate) mod validate;
