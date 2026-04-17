//! MCP tool implementations for aprender.
//!
//! M1 shipped `apr.version`. M2 adds 8 Phase-1 tools as subprocess wrappers
//! around `apr <cmd> --json`. Shipped: `apr.validate`, `apr.tensors`, `apr.bench`.
//! Follow-ups: `apr.run`, `apr.serve`, `apr.qa`, `apr.trace`, `apr.finetune`.

pub mod bench;
pub mod subprocess;
pub mod tensors;
pub mod validate;
pub mod version;

pub use bench::bench_tool_definition;
pub use tensors::tensors_tool_definition;
pub use validate::validate_tool_definition;
pub use version::version_tool_definition;
