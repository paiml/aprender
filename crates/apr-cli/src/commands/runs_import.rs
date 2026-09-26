//! `apr runs import <dir>`: backfill loose run directories into pacha
//! (EXT-001 EXT-10, I-3, FALSIFY-EXT-008).
//!
//! Each child of `<dir>` is one legacy run. Import reads only what is on disk:
//! - checkpoint sidecars `ckpt/epoch-NNN.metadata.json` become `epoch.<field>`
//!   metric series at step = epoch;
//! - a result file `r.json` gives its `status` and `per_step_metrics`;
//! - model files (`.apr`, `.safetensors`, `.gguf`) become artifact paths.
//!
//! Nothing is invented. Every run is tagged `provenance=backfilled`. A field
//! the source does not hold stays NULL: no start or end time, no hash (a
//! backfill does not re-read hundreds of GB of checkpoints), and status
//! `unknown` unless `r.json` records one. A directory with nothing
//! structured is skipped with a reason, and one imported before (same
//! `source_path`) is skipped too, so a re-run adds nothing.
//!
//! Like `apr runs gc`, the default is the plan; `--yes` writes it, after
//! checking free space on the pacha home (R-6).

use crate::error::{CliError, Result};
use entrenar::tracking::pacha::PachaBackend;
use entrenar::tracking::storage::TrackingBackend;
use entrenar::tracking::{ulid::new_ulid, Run, RunStatus};
use pacha::RegistryConfig;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Refuse to write with less than this free on the pacha home's filesystem.
pub(crate) const MIN_FREE_BYTES: u64 = 5 << 30;

/// Model files recorded as artifacts.
const MODEL_EXTS: [&str; 3] = ["apr", "safetensors", "gguf"];

/// What import does with one directory.
#[derive(Debug, Clone)]
pub(crate) enum Outcome {
    /// Import this run (its id is minted at write time).
    Import(Box<Run>),
    /// Leave the directory out, for this reason.
    Skip(String),
}

/// One child of the import root and its outcome.
#[derive(Debug, Clone)]
pub(crate) struct Planned {
    pub source: PathBuf,
    pub outcome: Outcome,
}

/// The backfilled run for `dir`, or why there is none. Pure: reads `dir`
/// and nothing else.
pub(crate) fn plan_dir(dir: &Path) -> Outcome {
    let source = std::path::absolute(dir).unwrap_or_else(|_| dir.to_path_buf());
    let mut run = Run {
        run_id: String::new(),
        run_name: dir.file_name().map(|n| n.to_string_lossy().into_owned()),
        experiment_name: "backfill".into(),
        status: RunStatus::Unknown,
        params: HashMap::new(),
        metrics: HashMap::new(),
        artifacts: Vec::new(),
        tags: HashMap::from([
            ("provenance".into(), "backfilled".into()),
            ("source_path".into(), source.display().to_string()),
        ]),
        start_time_ms: None,
        end_time_ms: None,
    };
    let mut files = Vec::new();
    collect_files(dir, 2, &mut files);
    files.sort();
    if files.is_empty() {
        return Outcome::Skip("empty directory".into());
    }
    let mut unreadable = Vec::new();
    for f in &files {
        let name = f
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let ext = f.extension().and_then(|e| e.to_str()).unwrap_or_default();
        if MODEL_EXTS.contains(&ext) {
            run.artifacts.push(f.display().to_string());
        } else if name.ends_with(".metadata.json") {
            if read_epoch_metadata(f, &mut run).is_none() {
                unreadable.push(name);
            }
        } else if name == "r.json" && read_result(f, &mut run).is_none() {
            unreadable.push(name);
        }
    }
    if !unreadable.is_empty() {
        run.params.insert("unreadable".into(), unreadable.join(","));
    }
    if run.artifacts.is_empty() && run.metrics.is_empty() {
        return Outcome::Skip(format!(
            "nothing structured: no model file, checkpoint metadata or r.json ({} file(s))",
            files.len()
        ));
    }
    Outcome::Import(Box::new(run))
}

/// Regular files under `dir`, `depth` levels deep.
fn collect_files(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for path in entries
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
    {
        if path.is_file() {
            out.push(path);
        } else if path.is_dir() && depth > 0 {
            collect_files(&path, depth - 1, out);
        }
    }
}

fn read_json(path: &Path) -> Option<serde_json::Map<String, serde_json::Value>> {
    match serde_json::from_slice(&std::fs::read(path).ok()?).ok()? {
        serde_json::Value::Object(map) => Some(map),
        _ => None,
    }
}

fn push_metric(run: &mut Run, key: String, value: &serde_json::Value, step: u64) {
    if let Some(v) = value.as_f64().filter(|v| v.is_finite()) {
        run.metrics.entry(key).or_default().push((v, step));
    }
}

/// `epoch-NNN.metadata.json`: every finite numeric field but `epoch` becomes
/// `epoch.<field>` at step = the file's `epoch`. No `epoch` field, no step:
/// the file is unreadable rather than guessed from its name.
fn read_epoch_metadata(path: &Path, run: &mut Run) -> Option<()> {
    let map = read_json(path)?;
    let epoch = map.get("epoch")?.as_u64()?;
    for (k, v) in &map {
        if k != "epoch" {
            push_metric(run, format!("epoch.{k}"), v, epoch);
        }
    }
    Some(())
}

/// `r.json`: `status` ("OK" is completed; any other value is kept verbatim
/// as `recorded_status` and the status stays unknown), `final_val_loss`, and
/// `per_step_metrics` rows keyed by their own `step`.
fn read_result(path: &Path, run: &mut Run) -> Option<()> {
    let map = read_json(path)?;
    if let Some(status) = map.get("status").and_then(|s| s.as_str()) {
        if status == "OK" {
            run.status = RunStatus::Completed;
        } else {
            run.params
                .insert("recorded_status".into(), status.to_string());
        }
    }
    if let Some(v) = map.get("final_val_loss") {
        push_metric(run, "final_val_loss".into(), v, 0);
    }
    for row in map
        .get("per_step_metrics")
        .and_then(|r| r.as_array())
        .into_iter()
        .flatten()
    {
        let Some(step) = row.get("step").and_then(serde_json::Value::as_u64) else {
            continue;
        };
        for (k, v) in row.as_object().into_iter().flatten() {
            if k != "step" {
                push_metric(run, k.clone(), v, step);
            }
        }
    }
    Some(())
}

/// The plan for every child directory of `root`, skipping those already
/// imported into `backend` (matched by `source_path`).
pub(crate) fn plan(root: &Path, backend: Option<&PachaBackend>) -> Result<Vec<Planned>> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root)?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    let mut imported = HashMap::new();
    if let Some(b) = backend {
        for run in b.list_runs().map_err(pacha_err)? {
            if run.tags.get("provenance").map(String::as_str) == Some("backfilled") {
                if let Some(src) = run.tags.get("source_path") {
                    imported.insert(src.clone(), run.run_id.clone());
                }
            }
        }
    }
    Ok(dirs
        .into_iter()
        .map(|dir| {
            let outcome = match plan_dir(&dir) {
                Outcome::Import(run) => match imported.get(&run.tags["source_path"]) {
                    Some(id) => Outcome::Skip(format!("already imported as {id}")),
                    None => Outcome::Import(run),
                },
                skip => skip,
            };
            Planned {
                source: dir,
                outcome,
            }
        })
        .collect())
}

fn pacha_err(e: impl std::fmt::Display) -> CliError {
    CliError::ValidationFailed(format!("pacha: {e}"))
}

/// Free bytes on the filesystem holding `path` (its nearest existing
/// ancestor), from `df -Pk`.
fn free_bytes(path: &Path) -> Result<u64> {
    let mut at = path;
    while !at.exists() {
        at = at.parent().unwrap_or(Path::new("/"));
    }
    let out = std::process::Command::new("df")
        .arg("-Pk")
        .arg(at)
        .output()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .nth(1)
        .and_then(|l| l.split_whitespace().nth(3))
        .and_then(|kb| kb.parse::<u64>().ok())
        .map(|kb| kb * 1024)
        .ok_or_else(|| CliError::ValidationFailed(format!("df failed for {}", at.display())))
}

/// R-6: refuse a backfill write with under [`MIN_FREE_BYTES`] free.
fn refuse_low_space(free: u64, home: &Path) -> Result<()> {
    if free < MIN_FREE_BYTES {
        return Err(CliError::ValidationFailed(format!(
            "R-6: {} GiB free under {}, need {} GiB before backfill",
            free >> 30,
            home.display(),
            MIN_FREE_BYTES >> 30
        )));
    }
    Ok(())
}

/// Write every `Import` in `planned` to the pacha home `home`; returns the
/// minted run ids.
pub(crate) fn apply(home: &Path, planned: &[Planned]) -> Result<Vec<String>> {
    refuse_low_space(free_bytes(home)?, home)?;
    pacha::Registry::open(RegistryConfig::new(home)).map_err(pacha_err)?;
    let mut backend = open_backend(home)?;
    let mut ids = Vec::new();
    for p in planned {
        if let Outcome::Import(run) = &p.outcome {
            let mut run = (**run).clone();
            run.run_id = new_ulid();
            backend.save_run(&run).map_err(pacha_err)?;
            ids.push(run.run_id);
        }
    }
    Ok(ids)
}

fn open_backend(home: &Path) -> Result<PachaBackend> {
    PachaBackend::open(
        &RegistryConfig::new(home).db_path(),
        &home.join("tracking-metrics.db"),
    )
    .map_err(pacha_err)
}

/// `apr runs import <dir> [--yes] [--json] [--pacha-home DIR]`.
pub(crate) fn run_import(
    dir: &Path,
    yes: bool,
    json: bool,
    pacha_home: Option<&Path>,
) -> Result<()> {
    let home = match pacha_home {
        Some(h) => h.to_path_buf(),
        None => dirs::home_dir()
            .map(|h| h.join(".pacha"))
            .ok_or_else(|| CliError::ValidationFailed("no home directory for ~/.pacha".into()))?,
    };
    let backend = if RegistryConfig::new(&home).db_path().exists() {
        Some(open_backend(&home)?)
    } else {
        None
    };
    let planned = plan(dir, backend.as_ref())?;
    drop(backend);
    let ids = if yes {
        apply(&home, &planned)?
    } else {
        Vec::new()
    };
    println!("{}", render_plan(&planned, &ids, yes, json));
    Ok(())
}

/// The plan (or what was written) as `apr runs import` prints it.
fn render_plan(planned: &[Planned], ids: &[String], applied: bool, json: bool) -> String {
    use std::fmt::Write as _;
    let imports = planned
        .iter()
        .filter(|p| matches!(p.outcome, Outcome::Import(_)))
        .count();
    if json {
        let rows: Vec<serde_json::Value> = planned
            .iter()
            .map(|p| match &p.outcome {
                Outcome::Import(run) => serde_json::json!({
                    "source": p.source,
                    "action": "import",
                    "status": format!("{:?}", run.status).to_lowercase(),
                    "metrics": run.metrics.len(),
                    "artifacts": run.artifacts.len(),
                }),
                Outcome::Skip(reason) => serde_json::json!({
                    "source": p.source,
                    "action": "skip",
                    "reason": reason,
                }),
            })
            .collect();
        return serde_json::json!({
                "applied": applied,
                "dirs": planned.len(),
                "import": imports,
                "skip": planned.len() - imports,
                "run_ids": ids,
            "rows": rows,
        })
        .to_string();
    }
    let mut out = String::new();
    for p in planned {
        match &p.outcome {
            Outcome::Import(run) => writeln!(
                out,
                "IMPORT {} status={:?} metrics={} artifacts={}",
                p.source.display(),
                run.status,
                run.metrics.len(),
                run.artifacts.len()
            ),
            Outcome::Skip(reason) => writeln!(out, "SKIP   {} ({reason})", p.source.display()),
        }
        .expect("write to String");
    }
    write!(
        out,
        "{} dir(s): {imports} import, {} skip{}",
        planned.len(),
        planned.len() - imports,
        if applied {
            format!("; wrote {} run(s)", ids.len())
        } else {
            "; plan only, pass --yes to write".into()
        }
    )
    .expect("write to String");
    out
}

#[cfg(test)]
#[path = "runs_import_tests.rs"]
mod tests;
