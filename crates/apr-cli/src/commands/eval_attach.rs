//! EXT-09 (aprender#4391): `apr eval` and `apr qa --json` attach their result to
//! the evaluated file in pacha's `evals` table, keyed on the file's sha256.
//!
//! Recording never changes the command's outcome. A file pacha has not
//! registered warns and is still evaluated and still recorded under its sha256,
//! so registering it later (EXT-05 writes `ModelCard.extra["sha256"]`) links the
//! rows. A registry that cannot be opened or written warns and the eval stands.

use pacha::registry::{EvalRecord, Registry, RegistryConfig};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

/// pacha's home, the same one `apr runs fsck` reads.
pub(crate) fn default_pacha_home() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".pacha"))
}

/// sha256 hex of a file, streamed.
pub(crate) fn file_sha256(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex_lower(&h.finalize()))
}

/// sha256 hex of bytes (a suite's items or its gate list).
pub(crate) fn bytes_sha256(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

fn hex_lower(b: &[u8]) -> String {
    use std::fmt::Write;
    b.iter()
        .fold(String::with_capacity(b.len() * 2), |mut s, x| {
            let _ = write!(s, "{x:02x}");
            s
        })
}

/// What one eval produced, before it is keyed to a file and an engine.
pub(crate) struct EvalOutcome<'a> {
    pub suite: &'a str,
    pub suite_manifest_sha: String,
    pub score: f64,
    pub n: u64,
}

/// What recording did, for the caller's warning and for tests.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Attached {
    /// Recorded against a registered model.
    Registered,
    /// Recorded, but no registered model card carries this sha256.
    Unregistered,
}

/// Record `outcome` for `model` in the registry at `pacha_home`.
pub(crate) fn attach_to(
    pacha_home: &Path,
    model: &Path,
    outcome: &EvalOutcome<'_>,
) -> Result<Attached, String> {
    if !model.is_file() {
        return Err(format!(
            "{} is not a file, so it has no sha256",
            model.display()
        ));
    }
    let model_sha =
        file_sha256(model).map_err(|e| format!("cannot hash {}: {e}", model.display()))?;
    let reg = Registry::open(RegistryConfig::new(pacha_home))
        .map_err(|e| format!("cannot open pacha at {}: {e}", pacha_home.display()))?;
    let row = EvalRecord {
        model_sha: model_sha.clone(),
        suite: outcome.suite.to_string(),
        suite_manifest_sha: outcome.suite_manifest_sha.clone(),
        score: outcome.score,
        n: outcome.n,
        engine_version: env!("CARGO_PKG_VERSION").to_string(),
        engine_sha: env!("APR_GIT_SHA").to_string(),
        host: hostname::get()
            .ok()
            .and_then(|h| h.into_string().ok())
            .unwrap_or_default(),
        ts: chrono::Utc::now(),
    };
    let registered = reg
        .is_model_sha_registered(&model_sha)
        .map_err(|e| format!("pacha lookup failed: {e}"))?;
    reg.record_eval(&row)
        .map_err(|e| format!("pacha refused the eval row: {e}"))?;
    Ok(if registered {
        Attached::Registered
    } else {
        Attached::Unregistered
    })
}

/// Record into the default pacha home and turn every problem into a warning on stderr.
pub(crate) fn attach(model: &Path, outcome: &EvalOutcome<'_>) {
    let Some(home) = default_pacha_home() else {
        eprintln!("⚠ eval not recorded: no home directory for pacha");
        return;
    };
    match attach_to(&home, model, outcome) {
        Ok(Attached::Registered) => {}
        Ok(Attached::Unregistered) => eprintln!(
            "⚠ {} is not a registered model; its {} eval is recorded under its sha256 only",
            model.display(),
            outcome.suite
        ),
        Err(e) => eprintln!("⚠ eval not recorded: {e}"),
    }
}

/// The `apr qa` row: the share of executed gates that passed, over the gates executed.
/// The manifest is the gate registry this binary runs, so a binary with a different gate
/// set never shares a suite manifest. A report that executed no gate has no score.
pub(crate) fn qa_outcome(report: &super::qa::QaReport) -> Option<EvalOutcome<'static>> {
    let executed: Vec<_> = report.gates.iter().filter(|g| !g.skipped).collect();
    if executed.is_empty() {
        return None;
    }
    let passed = executed.iter().filter(|g| g.passed).count();
    Some(EvalOutcome {
        suite: "apr-qa",
        suite_manifest_sha: bytes_sha256(report.gates_registered.join("\n").as_bytes()),
        score: passed as f64 / executed.len() as f64,
        n: executed.len() as u64,
    })
}

/// The evals recorded for `model_sha` in the registry at `pacha_home`. A home with no
/// registry yet has none, and reading never creates one.
pub(crate) fn evals_for(pacha_home: &Path, model_sha: &str) -> Result<Vec<EvalRecord>, String> {
    let config = RegistryConfig::new(pacha_home);
    if !config.db_path().is_file() {
        return Ok(Vec::new());
    }
    Registry::open(config)
        .and_then(|r| r.list_evals(model_sha))
        .map_err(|e| format!("cannot read evals from {}: {e}", pacha_home.display()))
}

/// The `apr runs show` block for the run's output model.
pub(crate) fn evals_text(model_sha: &str, evals: &[EvalRecord]) -> String {
    let mut s = format!("  Output model: sha256 {model_sha}\n");
    if evals.is_empty() {
        s.push_str("    evals: none recorded\n");
    }
    for e in evals {
        s.push_str(&format!(
            "    {:<24} score {:<12} n {:<8} apr {} ({}) on {} at {}\n",
            e.suite,
            e.score,
            e.n,
            e.engine_version,
            e.engine_sha,
            e.host,
            e.ts.to_rfc3339()
        ));
    }
    s
}

#[cfg(test)]
#[path = "eval_attach_tests.rs"]
mod tests;
