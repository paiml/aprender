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
#[cfg(feature = "apr-tools")]
pub mod bench;
#[cfg(feature = "apr-tools")]
pub mod capability;
#[cfg(feature = "apr-tools")]
pub mod finetune;
#[cfg(feature = "apr-tools")]
pub mod port_owner;
#[cfg(feature = "apr-tools")]
pub mod qa;
pub mod registry;
#[cfg(feature = "apr-tools")]
pub mod run;
#[cfg(feature = "apr-tools")]
pub mod serve;
#[cfg(feature = "apr-tools")]
pub mod subprocess;
#[cfg(feature = "apr-tools")]
pub mod tensors;
#[cfg(feature = "apr-tools")]
pub mod trace;
#[cfg(feature = "apr-tools")]
pub mod validate;
#[cfg(feature = "apr-tools")]
pub mod version;

#[cfg(feature = "apr-tools")]
pub use registry::McpToolEntry;
pub use registry::{DispatchFn, ToolIndex};

#[cfg(feature = "apr-tools")]
pub use bench::bench_tool_definition;
#[cfg(feature = "apr-tools")]
pub use capability::capability_tool_definition;
#[cfg(feature = "apr-tools")]
pub use finetune::finetune_tool_definition;
#[cfg(feature = "apr-tools")]
pub use qa::qa_tool_definition;
#[cfg(feature = "apr-tools")]
pub use run::run_tool_definition;
#[cfg(feature = "apr-tools")]
pub use serve::serve_tool_definition;
#[cfg(feature = "apr-tools")]
pub use tensors::tensors_tool_definition;
#[cfg(feature = "apr-tools")]
pub use trace::trace_tool_definition;
#[cfg(feature = "apr-tools")]
pub use validate::validate_tool_definition;
#[cfg(feature = "apr-tools")]
pub use version::version_tool_definition;
