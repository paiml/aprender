//! MCP tool implementations for aprender.
//!
//! Phase-1 surface (shipped M1–M3):
//! - M1 scaffold: `apr.version`.
//! - M2 subprocess wrappers around `apr <cmd> --json`: `apr.validate`,
//!   `apr.tensors`, `apr.bench`, `apr.qa`, `apr.trace`, `apr.run`, `apr.serve`.
//! - M3 streaming slice: `apr.finetune` (opt-in `notifications/progress`
//!   per non-empty stdout line when `params._meta.progressToken` is set —
//!   see FALSIFY-MCP-PROGRESS-001) + `notifications/cancelled` →
//!   SIGTERM→SIGKILL for `apr.run` (FALSIFY-MCP-006).

pub mod args;
pub mod bench;
pub mod finetune;
pub mod port_owner;
pub mod qa;
pub mod registry;
pub mod run;
pub mod serve;
pub mod subprocess;
pub mod tensors;
pub mod trace;
pub mod validate;
pub mod version;

pub use registry::{DispatchFn, McpToolEntry, ToolIndex};

pub use bench::bench_tool_definition;
pub use finetune::finetune_tool_definition;
pub use qa::qa_tool_definition;
pub use run::run_tool_definition;
pub use serve::serve_tool_definition;
pub use tensors::tensors_tool_definition;
pub use trace::trace_tool_definition;
pub use validate::validate_tool_definition;
pub use version::version_tool_definition;
