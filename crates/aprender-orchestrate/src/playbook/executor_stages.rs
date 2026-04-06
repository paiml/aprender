//! Stage-level execution helpers for the playbook executor.
//!
//! Handles cache evaluation, hashing, success/failure processing,
//! and run finalization.

use super::{CacheAction, RunConfig, StageHashes};
use crate::playbook::cache::{self, CacheDecision};
use crate::playbook::dag;
use crate::playbook::eventlog;
use crate::playbook::hasher;
use crate::playbook::template;
use crate::playbook::types::*;
use anyhow::{bail, Context, Result};
use std::collections::HashSet;
use std::path::Path;

/// Reject stages targeting remote hosts (Phase 2 not yet available).
/// Allow localhost and 127.0.0.1 (M8 fix).
pub(super) fn check_remote_target(
    stage_name: &str,
    stage: &Stage,
    playbook: &Playbook,
) -> Result<()> {
    if let Some(ref target_name) = stage.target {
        if let Some(target) = playbook.targets.get(target_name) {
            if let Some(ref host) = target.host {
                let is_local = host == "localhost" || host == "127.0.0.1";
                if !is_local {
                    bail!(
                        "stage '{}' targets remote host '{}': Remote execution requires Phase 2 (PB-006)",
                        stage_name,
                        target_name
                    );
                }
            }
        }
    }
    Ok(())
}

/// Handle frozen stages: return true if stage should be skipped (cached).
pub(super) fn handle_frozen(
    stage: &Stage,
    force: bool,
    stage_name: &str,
    playbook_path: &Path,
) -> bool {
    if stage.frozen && !force {
        let _ = eventlog::append_event(
            playbook_path,
            PipelineEvent::StageCached {
                stage: stage_name.to_string(),
                cache_key: "frozen".to_string(),
                reason: "stage is frozen".to_string(),
            },
        );
        println!("  {} CACHED (frozen)", stage_name);
        true
    } else {
        false
    }
}

/// Resolve templates and compute all hashes for a stage.
pub(super) fn compute_stage_hashes(
    stage_name: &str,
    stage: &Stage,
    playbook: &Playbook,
) -> Result<StageHashes> {
    let resolved_cmd = template::resolve_template(
        &stage.cmd,
        &playbook.params,
        &stage.params,
        &stage.deps,
        &stage.outs,
    )
    .with_context(|| format!("stage '{}' template resolution failed", stage_name))?;

    let cmd_hash = hasher::hash_cmd(&resolved_cmd);

    let (dep_hashes, dep_locks) = hash_dependencies(&stage.deps)?;

    let deps_combined =
        hasher::combine_deps_hashes(&dep_hashes.iter().map(|(_, h)| h.clone()).collect::<Vec<_>>());

    let param_refs = hasher::effective_param_keys(&stage.params, &stage.cmd);
    let params_hash = hasher::hash_params(&playbook.params, &param_refs)?;
    let cache_key = hasher::compute_cache_key(&cmd_hash, &deps_combined, &params_hash);

    Ok(StageHashes { resolved_cmd, cmd_hash, dep_hashes, dep_locks, params_hash, cache_key })
}

/// Hash all dependencies for a stage, returning parallel vecs of hashes and locks.
fn hash_dependencies(deps: &[Dependency]) -> Result<(Vec<(String, String)>, Vec<DepLock>)> {
    let mut dep_hashes: Vec<(String, String)> = Vec::new();
    let mut dep_locks: Vec<DepLock> = Vec::new();

    for dep in deps {
        let dep_path = Path::new(&dep.path);
        if dep_path.exists() {
            let result = hasher::hash_dep(dep_path)?;
            dep_hashes.push((dep.path.clone(), result.hash.clone()));
            dep_locks.push(DepLock {
                path: dep.path.clone(),
                hash: result.hash,
                file_count: Some(result.file_count),
                total_bytes: Some(result.total_bytes),
            });
        } else {
            dep_hashes.push((dep.path.clone(), String::new()));
            dep_locks.push(DepLock {
                path: dep.path.clone(),
                hash: String::new(),
                file_count: None,
                total_bytes: None,
            });
        }
    }

    Ok((dep_hashes, dep_locks))
}

/// Check cache and log the appropriate event, returning the action to take.
pub(super) fn evaluate_cache(
    stage_name: &str,
    stage: &Stage,
    hashes: &StageHashes,
    existing_lock: &Option<LockFile>,
    dag_result: &dag::PlaybookDag,
    rerun_stages: &HashSet<String>,
    config: &RunConfig,
) -> CacheAction {
    let upstream_reruns: Vec<String> = dag_result
        .predecessors
        .get(stage_name)
        .map(|preds: &Vec<String>| {
            preds.iter().filter(|p| rerun_stages.contains(p.as_str())).cloned().collect()
        })
        .unwrap_or_default();

    let decision = cache::check_cache(
        stage_name,
        &hashes.cache_key,
        &hashes.cmd_hash,
        &hashes.dep_hashes,
        &hashes.params_hash,
        existing_lock,
        config.force,
        &upstream_reruns,
    );

    match decision {
        CacheDecision::Hit => {
            let _ = eventlog::append_event(
                &config.playbook_path,
                PipelineEvent::StageCached {
                    stage: stage_name.to_string(),
                    cache_key: hashes.cache_key.clone(),
                    reason: "cache_key matches lock".to_string(),
                },
            );
            println!("  {} CACHED", stage_name);
            CacheAction::Cached
        }
        CacheDecision::Miss { ref reasons } => {
            let reason_str: Vec<String> = reasons.iter().map(|r| r.to_string()).collect();
            let _ = eventlog::append_event(
                &config.playbook_path,
                PipelineEvent::StageStarted {
                    stage: stage_name.to_string(),
                    target: stage.target.clone().unwrap_or_else(|| "local".to_string()),
                    cache_miss_reason: reason_str.join("; "),
                },
            );
            println!("  {} RUNNING ({})", stage_name, reason_str.join("; "));
            CacheAction::Execute
        }
    }
}

/// Process successful stage execution: hash outputs, update lock, log event.
#[allow(clippy::too_many_arguments)]
pub(super) fn handle_stage_success(
    stage_name: &str,
    stage: &Stage,
    hashes: &StageHashes,
    started_at: &str,
    completed_at: &str,
    duration: std::time::Duration,
    lock: &mut LockFile,
    playbook_path: &Path,
) -> Result<()> {
    let out_locks = hash_outputs(stage_name, &stage.outs)?;

    let outs_hash = if out_locks.is_empty() {
        None
    } else {
        Some(hasher::combine_deps_hashes(
            &out_locks.iter().map(|o| o.hash.clone()).collect::<Vec<_>>(),
        ))
    };

    lock.stages.insert(
        stage_name.to_string(),
        StageLock {
            status: StageStatus::Completed,
            started_at: Some(started_at.to_string()),
            completed_at: Some(completed_at.to_string()),
            duration_seconds: Some(duration.as_secs_f64()),
            target: stage.target.clone(),
            deps: hashes.dep_locks.clone(),
            params_hash: Some(hashes.params_hash.clone()),
            outs: out_locks,
            cmd_hash: Some(hashes.cmd_hash.clone()),
            cache_key: Some(hashes.cache_key.clone()),
        },
    );

    let _ = eventlog::append_event(
        playbook_path,
        PipelineEvent::StageCompleted {
            stage: stage_name.to_string(),
            duration_seconds: duration.as_secs_f64(),
            outs_hash,
        },
    );

    println!("  {} COMPLETED ({:.1}s)", stage_name, duration.as_secs_f64());

    Ok(())
}

/// Hash all output artifacts for a completed stage.
fn hash_outputs(stage_name: &str, outs: &[Output]) -> Result<Vec<OutLock>> {
    let mut out_locks: Vec<OutLock> = Vec::new();
    for out in outs {
        let out_path = Path::new(&out.path);
        if out_path.exists() {
            let result = hasher::hash_dep(out_path)?;
            out_locks.push(OutLock {
                path: out.path.clone(),
                hash: result.hash,
                file_count: Some(result.file_count),
                total_bytes: Some(result.total_bytes),
                remote: out.remote.clone(),
            });
        } else {
            tracing::warn!(
                "stage '{}' completed but output '{}' does not exist",
                stage_name,
                out.path
            );
            out_locks.push(OutLock {
                path: out.path.clone(),
                hash: String::new(),
                file_count: None,
                total_bytes: None,
                remote: out.remote.clone(),
            });
        }
    }
    Ok(out_locks)
}

/// Process failed stage execution: update lock, log event, enforce Jidoka policy.
#[allow(clippy::too_many_arguments)]
pub(super) fn handle_stage_failure(
    stage_name: &str,
    stage: &Stage,
    hashes: &StageHashes,
    started_at: &str,
    completed_at: &str,
    duration: std::time::Duration,
    error: anyhow::Error,
    lock: &mut LockFile,
    playbook: &Playbook,
    config: &RunConfig,
    run_id: &str,
) -> Result<()> {
    let (exit_code, error_msg) = match error.downcast_ref::<super::CommandError>() {
        Some(ce) => (ce.exit_code, ce.stderr.clone()),
        None => (None, error.to_string()),
    };

    lock.stages.insert(
        stage_name.to_string(),
        StageLock {
            status: StageStatus::Failed,
            started_at: Some(started_at.to_string()),
            completed_at: Some(completed_at.to_string()),
            duration_seconds: Some(duration.as_secs_f64()),
            target: stage.target.clone(),
            deps: hashes.dep_locks.clone(),
            params_hash: Some(hashes.params_hash.clone()),
            outs: vec![],
            cmd_hash: Some(hashes.cmd_hash.clone()),
            cache_key: None,
        },
    );

    let _ = eventlog::append_event(
        &config.playbook_path,
        PipelineEvent::StageFailed {
            stage: stage_name.to_string(),
            exit_code,
            error: error_msg.clone(),
            retry_attempt: None,
        },
    );

    eprintln!("  {} FAILED: {}", stage_name, error_msg);

    if playbook.policy.failure == FailurePolicy::StopOnFirst {
        let _ = eventlog::append_event(
            &config.playbook_path,
            PipelineEvent::RunFailed {
                playbook: playbook.name.clone(),
                run_id: run_id.to_string(),
                error: format!("stage '{}' failed (Jidoka: stop_on_first)", stage_name),
            },
        );

        if playbook.policy.lock_file {
            lock.generated_at = eventlog::now_iso8601();
            cache::save_lock_file(lock, &config.playbook_path)?;
        }

        bail!("stage '{}' failed: {} (Jidoka: stop_on_first policy)", stage_name, error_msg);
    }

    Ok(())
}

/// Save final lock file and return the run result.
pub(super) fn finalize_run(
    playbook: &Playbook,
    run_id: &str,
    stages_run: u32,
    stages_cached: u32,
    stages_failed: u32,
    total_duration: std::time::Duration,
    mut lock: LockFile,
    config: &RunConfig,
) -> Result<super::RunResult> {
    let _ = eventlog::append_event(
        &config.playbook_path,
        PipelineEvent::RunCompleted {
            playbook: playbook.name.clone(),
            run_id: run_id.to_string(),
            stages_run,
            stages_cached,
            stages_failed,
            total_seconds: total_duration.as_secs_f64(),
        },
    );

    if playbook.policy.lock_file {
        lock.generated_at = eventlog::now_iso8601();
        let all_param_keys: Vec<String> = playbook.params.keys().cloned().collect();
        lock.params_hash = Some(hasher::hash_params(&playbook.params, &all_param_keys)?);
        cache::save_lock_file(&lock, &config.playbook_path)?;
    }

    Ok(super::RunResult {
        stages_run,
        stages_cached,
        stages_failed,
        total_duration,
        lock_file: Some(lock),
    })
}
