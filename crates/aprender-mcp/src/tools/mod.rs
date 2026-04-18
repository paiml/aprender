//! MCP tool implementations for aprender.
//!
//! M1 shipped `apr.version`. M2 adds 8 Phase-1 tools as subprocess wrappers
//! around `apr <cmd> --json`. Shipped: `apr.validate`, `apr.tensors`,
//! `apr.bench`, `apr.qa`, `apr.trace`, `apr.run`, `apr.serve`. Follow-up:
//! `apr.finetune` (streaming, likely lands in M3).

pub mod bench;
pub mod qa;
pub mod run;
pub mod serve;
pub mod subprocess;
pub mod tensors;
pub mod trace;
pub mod validate;
pub mod version;

pub use bench::bench_tool_definition;
pub use qa::qa_tool_definition;
pub use run::run_tool_definition;
pub use serve::serve_tool_definition;
pub use tensors::tensors_tool_definition;
pub use trace::trace_tool_definition;
pub use validate::validate_tool_definition;
pub use version::version_tool_definition;
