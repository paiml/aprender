//! `apr model gate`: the M-CR and M0..M6 gates against pacha (EXT-001 §3.6, rows EXT-12
//! aprender#4394 and EXT-13 aprender#4393).
//!
//! The gates live in `model_gate`; this file is only the pacha lookups and the CLI surface.

use super::model_gate::{run, GateEnv, GateEvidence, GateInputs, GateReceipt, RECEIPT};
use super::model_gate_m2::M2Prereg;
use crate::error::{CliError, Result};
use entrenar::tracking::pacha::PachaBackend;
use entrenar::tracking::storage::{TrackingBackend, TrackingStorageError};
use pacha::data::{AdmittedManifest, SealedItems};
use pacha::{Registry, RegistryConfig};
use std::path::{Path, PathBuf};

/// The real lookups: lineage runs in the tracking backend, manifests in the registry.
struct PachaEnv {
    registry: Registry,
    runs: PachaBackend,
}

impl PachaEnv {
    fn open(home: &Path) -> Result<Self> {
        let err = |e: &dyn std::fmt::Display| CliError::ValidationFailed(format!("pacha: {e}"));
        let registry = Registry::open(RegistryConfig::new(home)).map_err(|e| err(&e))?;
        let runs = PachaBackend::open(
            &RegistryConfig::new(home).db_path(),
            &home.join("tracking-metrics.db"),
        )
        .map_err(|e| err(&e))?;
        Ok(Self { registry, runs })
    }
}

impl GateEnv for PachaEnv {
    fn run_resolves(&self, run_id: &str) -> std::result::Result<bool, String> {
        match self.runs.load_run(run_id) {
            Ok(_) => Ok(true),
            Err(TrackingStorageError::RunNotFound(_)) => Ok(false),
            Err(e) => Err(e.to_string()),
        }
    }

    fn dataset_manifest(
        &self,
        canonical_sha256: &str,
    ) -> std::result::Result<Option<AdmittedManifest>, String> {
        self.registry
            .get_dataset_manifest(canonical_sha256)
            .map_err(|e| e.to_string())
    }
}

/// What `apr model gate` was given.
pub(crate) struct GateArgs<'a> {
    pub dir: &'a Path,
    pub evidence: &'a Path,
    pub sealed: &'a Path,
    pub engine_tarball: Option<&'a Path>,
    pub fetched: Option<&'a Path>,
    pub pacha_home: Option<&'a Path>,
    pub out: Option<&'a Path>,
    pub json: bool,
}

fn print_table(r: &GateReceipt) {
    println!("{} {}  manifest {}", r.line, r.version, r.manifest_sha256);
    for g in &r.gates {
        println!("  {} {}", g.gate, if g.green { "GREEN" } else { "RED" });
        for f in &g.findings {
            println!("      {f}");
        }
    }
    println!("all_green: {}", r.all_green);
}

/// Run every gate, write the receipt, and fail unless all eight are green.
///
/// # Errors
///
/// `InvalidInput` when the evidence or sealed set cannot be read; `ValidationFailed` when
/// pacha cannot be opened, the manifest is unreadable, or any gate is RED (the receipt is
/// still written, so a RED run leaves its findings on disk).
pub(crate) fn run_gate(a: &GateArgs<'_>) -> Result<()> {
    let text = std::fs::read_to_string(a.evidence)
        .map_err(|e| CliError::InvalidInput(format!("{}: {e}", a.evidence.display())))?;
    let evidence: GateEvidence = serde_json::from_str(&text)
        .map_err(|e| CliError::InvalidInput(format!("{}: {e}", a.evidence.display())))?;
    let sealed = SealedItems::load_dir(a.sealed)
        .map_err(|e| CliError::InvalidInput(format!("{}: {e}", a.sealed.display())))?;
    let home = match a.pacha_home {
        Some(h) => h.to_path_buf(),
        None => dirs::home_dir()
            .map(|h| h.join(".pacha"))
            .ok_or_else(|| CliError::ValidationFailed("no home directory for ~/.pacha".into()))?,
    };
    let env = PachaEnv::open(&home)?;
    let receipt = run(
        &M2Prereg::REX_001,
        &GateInputs {
            dir: a.dir,
            evidence: &evidence,
            sealed: &sealed,
            engine_tarball: a.engine_tarball,
            fetched: a.fetched,
            env: &env,
        },
    )
    .map_err(CliError::ValidationFailed)?;

    let out: PathBuf = a.out.map_or_else(|| a.dir.join(RECEIPT), Path::to_path_buf);
    let body = serde_json::to_string_pretty(&receipt)
        .map_err(|e| CliError::ValidationFailed(format!("receipt: {e}")))?;
    std::fs::write(&out, format!("{body}\n"))
        .map_err(|e| CliError::ValidationFailed(format!("{}: {e}", out.display())))?;
    if a.json {
        println!("{body}");
    } else {
        print_table(&receipt);
        println!("receipt: {}", out.display());
    }
    if receipt.all_green {
        Ok(())
    } else {
        let red: Vec<&str> = receipt
            .gates
            .iter()
            .filter(|g| !g.green)
            .map(|g| g.gate)
            .collect();
        Err(CliError::ValidationFailed(format!(
            "model gate RED: {}",
            red.join(", ")
        )))
    }
}

#[cfg(test)]
#[path = "model_gate_cli_tests.rs"]
mod tests;
