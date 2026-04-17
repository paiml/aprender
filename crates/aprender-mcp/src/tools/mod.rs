//! MCP tool implementations for aprender.
//!
//! M1 shipped `apr.version`. M2 adds 8 Phase-1 tools as subprocess wrappers
//! around `apr <cmd> --json`. This module is the first M2 slice: `apr.validate`.
//! Follow-ups add run, serve, qa, trace, tensors, bench, finetune.

pub mod validate;
pub mod version;

pub use validate::validate_tool_definition;
pub use version::version_tool_definition;
