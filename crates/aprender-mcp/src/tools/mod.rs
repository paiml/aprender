//! MCP tool implementations for aprender.
//!
//! M1 ships only `apr.version`. M2 will add 8 Phase-1 tools (apr.run, apr.qa,
//! apr.serve, apr.trace, apr.tensors, apr.validate, apr.bench, apr.finetune).

pub mod version;

pub use version::version_tool_definition;
