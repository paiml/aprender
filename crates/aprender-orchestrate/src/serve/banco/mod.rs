//! Banco: Local-first AI Workbench HTTP API
//!
//! Phase 1 delivers the API skeleton: health, model listing,
//! chat completions (with SSE streaming), and system info.
//!
//! All endpoints reuse the existing serve module orchestration:
//! `BackendSelector`, `SpilloverRouter`, `CostCircuitBreaker`,
//! `ContextManager`, and `ChatTemplateEngine`.

pub mod audit;
pub mod auth;
pub mod batch;
pub mod compat_ollama;
pub mod config;
pub mod conversations;
pub mod eval;
pub mod events;
pub mod experiment;
mod handlers;
mod handlers_audio;
mod handlers_audit;
mod handlers_batch;
mod handlers_completions;
mod handlers_config;
mod handlers_conversations;
mod handlers_data;
mod handlers_eval;
mod handlers_experiment;
#[cfg(feature = "realizar")]
mod handlers_inference;
mod handlers_mcp;
mod handlers_merge;
mod handlers_metrics;
mod handlers_models;
mod handlers_prompts;
mod handlers_rag;
mod handlers_recipes;
mod handlers_registry;
mod handlers_tokens;
mod handlers_tools;
mod handlers_train;
mod handlers_ui;
mod handlers_ws;
#[cfg(feature = "realizar")]
pub mod inference;
pub mod mcp;
pub mod metrics;
mod middleware;
pub mod model_slot;
pub mod prompts;
pub mod rag;
pub mod recipes;
pub mod router;
mod server;
pub mod state;
pub mod storage;
pub mod tools;
pub mod training;
pub mod training_engine;
pub mod types;
pub mod ui;

pub use server::start_server;
pub use state::BancoState;

#[cfg(test)]
#[path = "types_tests.rs"]
mod types_tests;

#[cfg(test)]
#[path = "state_tests.rs"]
mod state_tests;

#[cfg(test)]
#[path = "middleware_tests.rs"]
mod middleware_tests;

#[cfg(test)]
#[path = "handlers_tests.rs"]
mod handlers_tests;

#[cfg(test)]
#[path = "contract_tests.rs"]
mod contract_tests;

#[cfg(test)]
#[path = "p0_tests.rs"]
mod p0_tests;

#[cfg(test)]
#[path = "p1_tests.rs"]
mod p1_tests;

#[cfg(test)]
#[path = "conversations_tests.rs"]
mod conversations_tests;

#[cfg(test)]
#[path = "p2_tests.rs"]
mod p2_tests;

#[cfg(test)]
#[path = "model_slot_tests.rs"]
mod model_slot_tests;

#[cfg(test)]
#[path = "inference_tests.rs"]
mod inference_tests;

#[cfg(test)]
#[path = "metrics_tests.rs"]
mod metrics_tests;

#[cfg(test)]
#[path = "completions_tests.rs"]
mod completions_tests;

#[cfg(test)]
#[path = "mcp_tests.rs"]
mod mcp_tests;

#[cfg(test)]
#[path = "audio_tests.rs"]
mod audio_tests;

#[cfg(test)]
#[path = "ui_tests.rs"]
mod ui_tests;

#[cfg(test)]
#[path = "tools_tests.rs"]
mod tools_tests;

#[cfg(test)]
#[path = "events_tests.rs"]
mod events_tests;

#[cfg(test)]
#[path = "storage_tests.rs"]
mod storage_tests;

#[cfg(test)]
#[path = "recipes_tests.rs"]
mod recipes_tests;

#[cfg(test)]
#[path = "rag_tests.rs"]
mod rag_tests;

#[cfg(test)]
#[path = "eval_train_tests.rs"]
mod eval_train_tests;

#[cfg(test)]
#[path = "experiment_tests.rs"]
mod experiment_tests;

#[cfg(test)]
#[path = "training_engine_tests.rs"]
mod training_engine_tests;

#[cfg(test)]
#[path = "merge_tests.rs"]
mod merge_tests;

#[cfg(test)]
#[path = "registry_tests.rs"]
mod registry_tests;

#[cfg(test)]
#[path = "batch_tests.rs"]
mod batch_tests;
