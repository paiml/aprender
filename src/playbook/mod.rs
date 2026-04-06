//! Batuta Playbook — deterministic pipeline orchestration
//!
//! YAML-based pipelines with BLAKE3 content-addressable caching,
//! DAG-based execution ordering, and Jidoka failure policy.
//!
//! Phase 1: Local sequential execution with cache miss explanations.

#![allow(unused_imports)]

pub mod cache;
pub mod dag;
pub mod eventlog;
pub mod executor;
pub mod hasher;
pub mod parser;
pub mod template;
pub mod types;

pub use executor::{run_playbook, show_status, validate_only, RunConfig, RunResult};
pub use types::{
    InvalidationReason, LockFile, PipelineEvent, Playbook, Stage, StageStatus, TimestampedEvent,
    ValidationWarning,
};
