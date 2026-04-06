//! Local sequential pipeline executor (PB-005)
//!
//! Orchestrates: parse → build DAG → load lock → for each stage in topo order:
//! resolve template → hash → check cache → execute → hash outputs → update lock → log event.
//!
//! Implements Jidoka: stop on first failure (only policy in Phase 1).
//! Remote targets return error: "Remote execution requires Phase 2 (PB-006)".

#[path = "executor_stages.rs"]
mod stages;

#[path = "executor_command.rs"]
mod command;

use super::cache;
use super::dag;
use super::eventlog;
use super::parser;
use super::types::*;
use anyhow::Result;
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

// Re-export public items
use command::execute_command;
use command::CommandError;
pub use command::{show_status, validate_only};
use stages::{
    check_remote_target, compute_stage_hashes, evaluate_cache, finalize_run, handle_frozen,
    handle_stage_failure, handle_stage_success,
};

/// Configuration for a playbook run
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// Path to the playbook YAML
    pub playbook_path: PathBuf,

    /// Only run these stages (None = all)
    pub stage_filter: Option<Vec<String>>,

    /// Force re-run (ignore cache)
    pub force: bool,

    /// Dry-run mode (Phase 6, no-op in Phase 1)
    pub dry_run: bool,

    /// Parameter overrides from CLI
    pub param_overrides: HashMap<String, serde_yaml_ng::Value>,
}

/// Result of a playbook run
#[derive(Debug, Clone)]
pub struct RunResult {
    pub stages_run: u32,
    pub stages_cached: u32,
    pub stages_failed: u32,
    pub total_duration: std::time::Duration,
    pub lock_file: Option<LockFile>,
}

/// Internal context shared across the execution pipeline
struct ExecutionContext {
    run_id: String,
    playbook: Playbook,
    dag_result: dag::PlaybookDag,
    existing_lock: Option<LockFile>,
    lock: LockFile,
    stages_run: u32,
    stages_cached: u32,
    stages_failed: u32,
    rerun_stages: HashSet<String>,
}

/// Hashed inputs for a single stage, computed before cache check
pub(crate) struct StageHashes {
    pub(crate) resolved_cmd: String,
    pub(crate) cmd_hash: String,
    pub(crate) dep_hashes: Vec<(String, String)>,
    pub(crate) dep_locks: Vec<DepLock>,
    pub(crate) params_hash: String,
    pub(crate) cache_key: String,
}

/// Whether to skip (cached) or execute the stage.
pub(crate) enum CacheAction {
    Cached,
    Execute,
}

/// Execute a playbook
pub async fn run_playbook(config: &RunConfig) -> Result<RunResult> {
    let total_start = Instant::now();

    let mut ctx = prepare_execution(config)?;
    let stages_to_run = select_stages(&ctx.dag_result, &config.stage_filter);

    for stage_name in &stages_to_run {
        execute_single_stage(&mut ctx, stage_name, config).await?;
    }

    finalize_run(
        &ctx.playbook,
        &ctx.run_id,
        ctx.stages_run,
        ctx.stages_cached,
        ctx.stages_failed,
        total_start.elapsed(),
        ctx.lock,
        config,
    )
}

/// Execute one stage: check cache, run command, update lock.
async fn execute_single_stage(
    ctx: &mut ExecutionContext,
    stage_name: &str,
    config: &RunConfig,
) -> Result<()> {
    let stage = ctx
        .playbook
        .stages
        .get(stage_name)
        .ok_or_else(|| anyhow::anyhow!("stage '{}' not found in playbook", stage_name))?;

    check_remote_target(stage_name, stage, &ctx.playbook)?;

    if handle_frozen(stage, config.force, stage_name, &config.playbook_path) {
        ctx.stages_cached += 1;
        return Ok(());
    }

    tracing::debug!(
        "stage '{}': executing via raw sh -c (bashrs purification deferred to Phase 2)",
        stage_name
    );

    let hashes = compute_stage_hashes(stage_name, stage, &ctx.playbook)?;
    let cache_action = evaluate_cache(
        stage_name,
        stage,
        &hashes,
        &ctx.existing_lock,
        &ctx.dag_result,
        &ctx.rerun_stages,
        config,
    );

    if matches!(cache_action, CacheAction::Cached) {
        ctx.stages_cached += 1;
        return Ok(());
    }

    let started_at = eventlog::now_iso8601();
    let stage_start = Instant::now();
    let exec_result = execute_command(&hashes.resolved_cmd).await;
    let duration = stage_start.elapsed();
    let completed_at = eventlog::now_iso8601();

    match exec_result {
        Ok(()) => {
            ctx.stages_run += 1;
            ctx.rerun_stages.insert(stage_name.to_string());
            handle_stage_success(
                stage_name,
                stage,
                &hashes,
                &started_at,
                &completed_at,
                duration,
                &mut ctx.lock,
                &config.playbook_path,
            )?;
        }
        Err(e) => {
            ctx.stages_failed += 1;
            handle_stage_failure(
                stage_name,
                stage,
                &hashes,
                &started_at,
                &completed_at,
                duration,
                e,
                &mut ctx.lock,
                &ctx.playbook,
                config,
                &ctx.run_id,
            )?;
        }
    }
    Ok(())
}

/// Parse playbook, build DAG, initialize lock file and run context.
fn prepare_execution(config: &RunConfig) -> Result<ExecutionContext> {
    let mut playbook = parser::parse_playbook_file(&config.playbook_path)?;
    let warnings = parser::validate_playbook(&playbook)?;
    for w in &warnings {
        tracing::warn!("playbook validation: {}", w);
    }

    for (k, v) in &config.param_overrides {
        playbook.params.insert(k.clone(), v.clone());
    }

    let dag_result = dag::build_dag(&playbook)?;
    let run_id = eventlog::generate_run_id();

    let _ = eventlog::append_event(
        &config.playbook_path,
        PipelineEvent::RunStarted {
            playbook: playbook.name.clone(),
            run_id: run_id.clone(),
            batuta_version: env!("CARGO_PKG_VERSION").to_string(),
        },
    );

    let existing_lock = cache::load_lock_file(&config.playbook_path)?;

    let mut lock = LockFile {
        schema: "1.0".to_string(),
        playbook: playbook.name.clone(),
        generated_at: eventlog::now_iso8601(),
        generator: format!("batuta {}", env!("CARGO_PKG_VERSION")),
        blake3_version: "1.8".to_string(),
        params_hash: None,
        stages: IndexMap::new(),
    };

    if let Some(ref el) = existing_lock {
        for (name, stage_lock) in &el.stages {
            lock.stages.insert(name.clone(), stage_lock.clone());
        }
    }

    Ok(ExecutionContext {
        run_id,
        playbook,
        dag_result,
        existing_lock,
        lock,
        stages_run: 0,
        stages_cached: 0,
        stages_failed: 0,
        rerun_stages: HashSet::new(),
    })
}

/// Filter and order stages according to DAG topology and optional stage filter.
fn select_stages(dag_result: &dag::PlaybookDag, stage_filter: &Option<Vec<String>>) -> Vec<String> {
    if let Some(ref filter) = stage_filter {
        dag_result.topo_order.iter().filter(|s| filter.contains(s)).cloned().collect()
    } else {
        dag_result.topo_order.clone()
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
#[path = "executor_tests.rs"]
mod tests;

#[cfg(test)]
#[allow(non_snake_case)]
#[path = "executor_tests_integration.rs"]
mod tests_integration;
