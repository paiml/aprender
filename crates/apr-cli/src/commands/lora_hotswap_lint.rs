//! `apr lora-hotswap-lint` — CRUX-C-16 LoRA hotswap observation linter.
//!
//! Reads a JSON observation file that captures a single LoRA hotswap run
//! and dispatches four classifiers (hotswap_parity, load_latency,
//! adapter_compat, unload_restore). Emits a text or `--json` report.
//!
//! Spec: `contracts/crux-C-16-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.
//!
//! Observation schema (top-level keys; all optional — missing sections
//! skip the corresponding classifier):
//!
//!   {
//!     "hotswap_parity": {
//!       "merged_tokens":  [1, 2, 3],
//!       "hotswap_tokens": [1, 2, 3]
//!     },
//!     "load_latency": {
//!       "samples_seconds": [0.1, 0.2, 0.5],
//!       "budget_seconds":  2.0
//!     },
//!     "adapter_compat": {
//!       "base_sha256":              "abc123",
//!       "adapter_base_sha256":      "abc123",
//!       "base_module_names":        ["q_proj", "k_proj", "v_proj"],
//!       "adapter_target_modules":   ["q_proj", "v_proj"],
//!       "adapter_rank":             64
//!     },
//!     "unload_restore": {
//!       "fresh_tokens":        [1, 2, 3],
//!       "after_unload_tokens": [1, 2, 3]
//!     }
//!   }

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::commands::lora_hotswap_classifier as clf;
use crate::error::{CliError, Result};

pub(crate) fn run(observation_file: &Path, json: bool) -> Result<()> {
    if !observation_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(observation_file)));
    }

    let body = std::fs::read_to_string(observation_file)?;
    let obs: Value = serde_json::from_str(&body).map_err(|e| {
        CliError::InvalidFormat(format!(
            "apr lora-hotswap-lint: failed to parse JSON from {}: {e}",
            observation_file.display()
        ))
    })?;

    let hotswap_parity = classify_hotswap_parity(&obs);
    let load_latency = classify_load_latency(&obs);
    let adapter_compat = classify_adapter_compat(&obs);
    let unload_restore = classify_unload_restore(&obs);

    let fail_reasons: Vec<String> = [
        hotswap_parity.as_ref().and_then(hotswap_parity_fail_reason),
        load_latency.as_ref().and_then(load_latency_fail_reason),
        adapter_compat.as_ref().and_then(adapter_compat_fail_reason),
        unload_restore.as_ref().and_then(unload_restore_fail_reason),
    ]
    .into_iter()
    .flatten()
    .collect();

    print_report(
        observation_file,
        hotswap_parity.as_ref(),
        load_latency.as_ref(),
        adapter_compat.as_ref(),
        unload_restore.as_ref(),
        json,
    );

    if fail_reasons.is_empty() {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(fail_reasons.join("; ")))
    }
}

fn classify_hotswap_parity(obs: &Value) -> Option<clf::HotswapParityOutcome> {
    let sec = obs.get("hotswap_parity")?.as_object()?;
    let merged = tokens_array(sec.get("merged_tokens")?)?;
    let hotswap = tokens_array(sec.get("hotswap_tokens")?)?;
    Some(clf::classify_hotswap_parity(&merged, &hotswap))
}

fn classify_load_latency(obs: &Value) -> Option<clf::LoadLatencyOutcome> {
    let sec = obs.get("load_latency")?.as_object()?;
    let samples: Vec<f64> = sec
        .get("samples_seconds")?
        .as_array()?
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect();
    let budget = sec.get("budget_seconds")?.as_f64()?;
    Some(clf::classify_load_latency(&samples, budget))
}

fn classify_adapter_compat(obs: &Value) -> Option<clf::AdapterCompatOutcome> {
    let sec = obs.get("adapter_compat")?.as_object()?;
    let base_sha = sec.get("base_sha256")?.as_str()?;
    let adapter_sha = sec.get("adapter_base_sha256")?.as_str()?;
    let base_modules: Vec<String> = sec
        .get("base_module_names")?
        .as_array()?
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    let adapter_modules: Vec<String> = sec
        .get("adapter_target_modules")?
        .as_array()?
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    let rank = sec.get("adapter_rank")?.as_u64()? as u32;

    let base_refs: Vec<&str> = base_modules.iter().map(String::as_str).collect();
    let adapter_refs: Vec<&str> = adapter_modules.iter().map(String::as_str).collect();

    Some(clf::classify_adapter_compat(
        base_sha,
        adapter_sha,
        &base_refs,
        &adapter_refs,
        rank,
    ))
}

fn classify_unload_restore(obs: &Value) -> Option<clf::UnloadRestoreOutcome> {
    let sec = obs.get("unload_restore")?.as_object()?;
    let fresh = tokens_array(sec.get("fresh_tokens")?)?;
    let after = tokens_array(sec.get("after_unload_tokens")?)?;
    Some(clf::classify_unload_restore(&fresh, &after))
}

fn tokens_array(v: &Value) -> Option<Vec<u32>> {
    Some(
        v.as_array()?
            .iter()
            .filter_map(|t| t.as_u64().map(|n| n as u32))
            .collect(),
    )
}

fn hotswap_parity_fail_reason(o: &clf::HotswapParityOutcome) -> Option<String> {
    match o {
        clf::HotswapParityOutcome::Ok => None,
        clf::HotswapParityOutcome::EmptinessMismatch {
            merged_empty,
            hotswap_empty,
        } => Some(format!(
            "FALSIFY-CRUX-C-16-001 hotswap_parity: emptiness mismatch merged_empty={merged_empty} hotswap_empty={hotswap_empty}"
        )),
        clf::HotswapParityOutcome::LengthMismatch {
            merged_len,
            hotswap_len,
        } => Some(format!(
            "FALSIFY-CRUX-C-16-001 hotswap_parity: length mismatch merged={merged_len} hotswap={hotswap_len}"
        )),
        clf::HotswapParityOutcome::TokenDivergence {
            at_index,
            merged_token,
            hotswap_token,
        } => Some(format!(
            "FALSIFY-CRUX-C-16-001 hotswap_parity: token divergence at idx {at_index}: merged={merged_token} hotswap={hotswap_token}"
        )),
    }
}

fn load_latency_fail_reason(o: &clf::LoadLatencyOutcome) -> Option<String> {
    match o {
        clf::LoadLatencyOutcome::Ok { .. } => None,
        clf::LoadLatencyOutcome::InvalidInput { reason } => Some(format!(
            "FALSIFY-CRUX-C-16-002 load_latency: invalid input: {reason}"
        )),
        clf::LoadLatencyOutcome::Exceeded {
            p99_seconds,
            budget_seconds,
        } => Some(format!(
            "FALSIFY-CRUX-C-16-002 load_latency: P99 {p99_seconds:.3}s exceeds budget {budget_seconds:.3}s"
        )),
    }
}

fn adapter_compat_fail_reason(o: &clf::AdapterCompatOutcome) -> Option<String> {
    match o {
        clf::AdapterCompatOutcome::Ok => None,
        clf::AdapterCompatOutcome::EmptyBaseSha256 => Some(
            "FALSIFY-CRUX-C-16-003 adapter_compat: base_sha256 is empty".to_string(),
        ),
        clf::AdapterCompatOutcome::EmptyAdapterBaseSha256 => Some(
            "FALSIFY-CRUX-C-16-003 adapter_compat: adapter_base_sha256 is empty".to_string(),
        ),
        clf::AdapterCompatOutcome::BaseSha256Mismatch => Some(
            "FALSIFY-CRUX-C-16-003 adapter_compat: base sha256 mismatch".to_string(),
        ),
        clf::AdapterCompatOutcome::EmptyTargetModules => Some(
            "FALSIFY-CRUX-C-16-003 adapter_compat: adapter target modules list is empty"
                .to_string(),
        ),
        clf::AdapterCompatOutcome::UnknownTargetModules { unknown } => Some(format!(
            "FALSIFY-CRUX-C-16-003 adapter_compat: unknown target modules: {unknown:?}"
        )),
        clf::AdapterCompatOutcome::RankTooSmall { rank } => Some(format!(
            "FALSIFY-CRUX-C-16-003 adapter_compat: rank {rank} below LORA_RANK_MIN={}",
            clf::LORA_RANK_MIN
        )),
        clf::AdapterCompatOutcome::RankTooLarge { rank } => Some(format!(
            "FALSIFY-CRUX-C-16-003 adapter_compat: rank {rank} above LORA_RANK_MAX={}",
            clf::LORA_RANK_MAX
        )),
    }
}

fn unload_restore_fail_reason(o: &clf::UnloadRestoreOutcome) -> Option<String> {
    match o {
        clf::UnloadRestoreOutcome::Ok => None,
        clf::UnloadRestoreOutcome::EmptinessMismatch {
            fresh_empty,
            after_unload_empty,
        } => Some(format!(
            "FALSIFY-CRUX-C-16-004 unload_restore: emptiness mismatch fresh_empty={fresh_empty} after_unload_empty={after_unload_empty}"
        )),
        clf::UnloadRestoreOutcome::LengthMismatch {
            fresh_len,
            after_unload_len,
        } => Some(format!(
            "FALSIFY-CRUX-C-16-004 unload_restore: length mismatch fresh={fresh_len} after_unload={after_unload_len}"
        )),
        clf::UnloadRestoreOutcome::TokenDivergence {
            at_index,
            fresh_token,
            after_unload_token,
        } => Some(format!(
            "FALSIFY-CRUX-C-16-004 unload_restore: token divergence at idx {at_index}: fresh={fresh_token} after_unload={after_unload_token}"
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn print_report(
    path: &Path,
    hotswap_parity: Option<&clf::HotswapParityOutcome>,
    load_latency: Option<&clf::LoadLatencyOutcome>,
    adapter_compat: Option<&clf::AdapterCompatOutcome>,
    unload_restore: Option<&clf::UnloadRestoreOutcome>,
    json: bool,
) {
    if json {
        let v = serde_json::json!({
            "observation_path": path.display().to_string(),
            "hotswap_parity":   hotswap_parity.map(|o| format!("{o:?}")),
            "load_latency":     load_latency.map(|o| format!("{o:?}")),
            "adapter_compat":   adapter_compat.map(|o| format!("{o:?}")),
            "unload_restore":   unload_restore.map(|o| format!("{o:?}")),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string())
        );
    } else {
        println!("lora-hotswap-lint report for {}", path.display());
        print_line(
            "  hotswap_parity: ",
            hotswap_parity.map(|o| format!("{o:?}")),
        );
        print_line("  load_latency:   ", load_latency.map(|o| format!("{o:?}")));
        print_line(
            "  adapter_compat: ",
            adapter_compat.map(|o| format!("{o:?}")),
        );
        print_line(
            "  unload_restore: ",
            unload_restore.map(|o| format!("{o:?}")),
        );
    }
}

fn print_line(prefix: &str, v: Option<String>) {
    match v {
        Some(s) => println!("{prefix}{s}"),
        None => println!("{prefix}(missing fields — classifier skipped)"),
    }
}
