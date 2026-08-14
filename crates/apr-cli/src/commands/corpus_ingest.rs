//! `apr corpus-ingest` — dry-run scaffolding for the SHIP-TWO-001 MODEL-2
//! Python-code pretraining corpus ingest pipeline.
//!
//! This module used to be a second `[[bin]]` in this crate,
//! `apr-corpus-ingest`. It is now reached as `apr corpus-ingest <COMMAND>`,
//! so `cargo install aprender` places exactly one binary in `~/.cargo/bin`.
//!
//! Implements task #91: minimal, NETWORK-FREE scaffolding that reads and
//! validates `contracts/dataset-thestack-python-v1.yaml` (C-DATA-THESTACK-PYTHON,
//! v1.0.0 PROPOSED) and emits a planned `dry-run-manifest.yaml` with the
//! contract fields filled in with placeholder values plus a timestamp.
//!
//! The full download / license-filter / PII-scrub / MinHash-LSH dedup /
//! hash-by-file-sha256 split pipeline is out of scope for this task. Later
//! work slots into the `run` subcommand (currently absent by design — a
//! missing subcommand is a louder signal than a stubbed one that lies).
//!
//! Hard constraints honored:
//!   * NO network calls.
//!   * NO writes outside the directory named by `--output-dir` (default
//!     `./output/`).
//!   * Only `serde`, `serde_yaml` and `anyhow` here — the clap definitions
//!     moved to `CorpusIngestCommands` in `extended_commands.rs` when the
//!     standalone binary was folded into `apr`.
//!   * Does NOT touch `crates/aprender-train/` (task #89) or
//!     `crates/apr-cli/src/commands/tokenize.rs` (task #90).

use crate::error::CliError;
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_yaml::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Default contract path, relative to workspace root.
pub const DEFAULT_CONTRACT_PATH: &str = "contracts/dataset-thestack-python-v1.yaml";

/// Default output directory for the dry-run manifest.
pub const DEFAULT_OUTPUT_DIR: &str = "output";

/// Minimum counts required by the contract spec.
/// The contract declares 7 invariants (INV-DATA-001..007), 5 falsification
/// tests (FALSIFY-DATA-001..005), and 5 gates (GATE-DATA-001..005).
const MIN_INVARIANTS: usize = 7;
const MIN_FALSIFICATIONS: usize = 5;
const MIN_GATES: usize = 5;

/// Required top-level keys on the contract YAML.
const REQUIRED_TOP_KEYS: &[&str] = &[
    "source",
    "license_whitelist",
    "pii_scrub",
    "deduplication",
    "split",
    "budget",
];

/// Structural view of the corpus contract used by both subcommands.
/// We do NOT model every field — only the load-bearing structure that the
/// scaffold must type-check. Everything else is carried through as
/// `serde_yaml::Value` so we can round-trip unknown fields untouched.
///
/// The `source`, `license_whitelist`, `pii_scrub`, `deduplication`, `split`,
/// and `budget` fields MUST be present on the YAML (a missing field aborts
/// deserialization — that is their job here), but they are not read as Rust
/// values in the dry-run. The real ingest (future task) destructures them.
#[derive(Debug, Deserialize)]
struct CorpusContract {
    contract_id: String,
    version: String,
    #[serde(default)]
    status: Option<String>,

    #[allow(dead_code)]
    source: Value,
    #[allow(dead_code)]
    license_whitelist: Value,
    #[allow(dead_code)]
    pii_scrub: Value,
    #[allow(dead_code)]
    deduplication: Value,
    #[allow(dead_code)]
    split: Value,
    #[allow(dead_code)]
    budget: Value,

    #[serde(default)]
    invariants: Vec<ContractItem>,
    #[serde(default)]
    falsification: Vec<ContractItem>,
    #[serde(default)]
    gates: Vec<ContractItem>,
}

#[derive(Debug, Deserialize)]
struct ContractItem {
    id: String,
    #[serde(default)]
    name: Option<String>,
    // Present so the contract validates; not surfaced by the dry-run yet.
    #[allow(dead_code)]
    #[serde(default)]
    description: Option<String>,
}

/// Report produced by `validate_contract`. Mirrors the invariants the caller
/// cares about so tests can assert them without re-parsing the YAML.
#[derive(Debug)]
struct ValidationReport {
    contract_id: String,
    version: String,
    status: Option<String>,
    invariants: usize,
    falsification: usize,
    gates: usize,
    present_top_keys: Vec<String>,
}

/// Dispatch one `apr corpus-ingest` command.
///
/// # Errors
///
/// Returns [`CliError::InvalidInput`] when the contract does not parse or does
/// not meet its declared structural minimums, and [`CliError::Io`] when the
/// manifest cannot be written.
pub fn run(command: &crate::CorpusIngestCommands) -> std::result::Result<(), CliError> {
    let outcome = match command {
        crate::CorpusIngestCommands::Plan {
            contract,
            output_dir,
        } => run_plan(contract, output_dir),
        crate::CorpusIngestCommands::ValidateContract { path } => {
            validate_contract(path).map(|report| print_validation_report(&report))
        }
    };
    // `anyhow` carries the full context chain; flatten it into apr's error
    // taxonomy so the exit code is the one apr documents (4 = bad input).
    outcome.map_err(|e| CliError::InvalidInput(format!("{e:#}")))
}

/// `plan` — loads the contract, validates it, prints a summary, writes a
/// `dry-run-manifest.yaml` skeleton populated with TODO placeholders for every
/// field the real ingest must fill in.
fn run_plan(contract_path: &Path, output_dir: &Path) -> Result<()> {
    let yaml = fs::read_to_string(contract_path).with_context(|| {
        format!(
            "failed to read corpus contract at {}",
            contract_path.display()
        )
    })?;
    let contract: CorpusContract =
        serde_yaml::from_str(&yaml).with_context(|| "contract YAML failed to deserialize")?;
    assert_structural_minimums(&contract)?;

    print_plan_summary(&contract, contract_path);

    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create output directory {}", output_dir.display()))?;
    let manifest_path = output_dir.join("dry-run-manifest.yaml");
    let manifest = build_dry_run_manifest(&contract, contract_path);
    let rendered =
        serde_yaml::to_string(&manifest).with_context(|| "failed to serialize dry-run manifest")?;
    fs::write(&manifest_path, rendered)
        .with_context(|| format!("failed to write {}", manifest_path.display()))?;

    println!();
    println!("wrote dry-run manifest: {}", manifest_path.display());
    println!("(no network calls, no downloads — this is scaffolding only)");
    Ok(())
}

/// `validate-contract` implementation. Returns a report; caller decides how
/// to present it.
fn validate_contract(path: &Path) -> Result<ValidationReport> {
    let yaml = fs::read_to_string(path)
        .with_context(|| format!("failed to read contract at {}", path.display()))?;

    // First pass: assert the raw mapping has the required top-level keys.
    // Using a raw Value here means a missing REQUIRED_TOP_KEYS entry fails
    // with a more pointed error than a serde derive missing-field message.
    let raw: Value =
        serde_yaml::from_str(&yaml).with_context(|| "contract YAML is not valid YAML")?;
    let mapping = raw
        .as_mapping()
        .ok_or_else(|| anyhow!("contract root must be a YAML mapping"))?;
    let mut present_top_keys = Vec::new();
    for key in REQUIRED_TOP_KEYS {
        if mapping.contains_key(Value::String((*key).to_string())) {
            present_top_keys.push((*key).to_string());
        } else {
            bail!("contract missing required top-level key: {key}");
        }
    }

    // Second pass: strong-typed deserialize so invariants[]/falsification[]/
    // gates[] have real `id` fields.
    let contract: CorpusContract = serde_yaml::from_str(&yaml)
        .with_context(|| "contract YAML failed to deserialize into typed schema")?;
    assert_structural_minimums(&contract)?;

    Ok(ValidationReport {
        contract_id: contract.contract_id,
        version: contract.version,
        status: contract.status,
        invariants: contract.invariants.len(),
        falsification: contract.falsification.len(),
        gates: contract.gates.len(),
        present_top_keys,
    })
}

/// Enforce the minimum counts declared in the contract header comment:
/// 7 invariants, 5 falsification tests, 5 gates. Uses `bail!` so the
/// resulting `anyhow::Error` carries a human-readable message.
fn assert_structural_minimums(contract: &CorpusContract) -> Result<()> {
    if contract.invariants.len() < MIN_INVARIANTS {
        bail!(
            "contract declares {} invariants; spec requires at least {}",
            contract.invariants.len(),
            MIN_INVARIANTS
        );
    }
    if contract.falsification.len() < MIN_FALSIFICATIONS {
        bail!(
            "contract declares {} falsification tests; spec requires at least {}",
            contract.falsification.len(),
            MIN_FALSIFICATIONS
        );
    }
    if contract.gates.len() < MIN_GATES {
        bail!(
            "contract declares {} gates; spec requires at least {}",
            contract.gates.len(),
            MIN_GATES
        );
    }
    for item in contract
        .invariants
        .iter()
        .chain(&contract.falsification)
        .chain(&contract.gates)
    {
        if item.id.trim().is_empty() {
            bail!("contract has an item with empty id");
        }
    }
    Ok(())
}

fn print_plan_summary(contract: &CorpusContract, contract_path: &Path) {
    println!("== C-DATA-THESTACK-PYTHON ingest plan (dry-run) ==");
    println!("contract file : {}", contract_path.display());
    println!("contract_id   : {}", contract.contract_id);
    println!("version       : {}", contract.version);
    if let Some(status) = &contract.status {
        println!("status        : {status}");
    }
    println!("invariants    : {}", contract.invariants.len());
    for inv in &contract.invariants {
        println!("  - {}", inv.id);
    }
    println!("falsification : {}", contract.falsification.len());
    for f in &contract.falsification {
        let name = f.name.as_deref().unwrap_or("");
        println!("  - {} {}", f.id, name);
    }
    println!("gates         : {}", contract.gates.len());
    for g in &contract.gates {
        let name = g.name.as_deref().unwrap_or("");
        println!("  - {} {}", g.id, name);
    }
    println!();
    println!("planned steps (none executed in dry-run):");
    println!("  1. pin HF dataset revision_sha + record raw_tar_sha256");
    println!("  2. license filter via go-license-detector (SPDX whitelist)");
    println!("  3. PII scrub (file-level rejection on pattern match)");
    println!("  4. MinHash-LSH dedup, shingle=5, perms=128, seed=42");
    println!("  5. hash-by-file-sha256 deterministic train/val split");
    println!("  6. emit shards + manifest + provenance + corpus_sha256");
}

fn print_validation_report(report: &ValidationReport) {
    println!("contract_id   : {}", report.contract_id);
    println!("version       : {}", report.version);
    if let Some(status) = &report.status {
        println!("status        : {status}");
    }
    println!(
        "top-level     : {} keys present",
        report.present_top_keys.len()
    );
    for key in &report.present_top_keys {
        println!("  - {key}");
    }
    println!("invariants    : {}", report.invariants);
    println!("falsification : {}", report.falsification);
    println!("gates         : {}", report.gates);
    println!("VALIDATION: PASS");
}

/// Dry-run manifest emitted by `plan`. Field names mirror
/// `manifest_required_fields` in the contract. Every value is a TODO
/// placeholder because this is scaffolding — the real ingest fills them in.
#[derive(Serialize)]
struct DryRunManifest {
    // Provenance of the dry-run itself.
    dry_run: bool,
    generated_at_utc: String,
    generator: &'static str,
    contract_path: String,
    contract_id: String,
    contract_version: String,

    // Counts echoed from the contract so downstream readers can sanity-check
    // that the manifest was produced against the expected contract shape.
    invariant_count: usize,
    falsification_count: usize,
    gate_count: usize,

    // Required output fields per `manifest_required_fields` in the contract.
    // All TODO until the real ingest lands (task #91 follow-up).
    source: BTreeMap<&'static str, String>,
    counts: BTreeMap<&'static str, String>,
    corpus_sha256: String,

    // Listing of declared gates so `apr qa`-style tools can scaffold against
    // a known set without re-parsing the contract.
    gates: Vec<String>,
}

fn build_dry_run_manifest(contract: &CorpusContract, contract_path: &Path) -> DryRunManifest {
    let mut source = BTreeMap::new();
    source.insert("revision_sha", placeholder("source.revision_sha"));
    source.insert("raw_tar_sha256", placeholder("source.raw_tar_sha256"));
    source.insert("ingest_date_utc", placeholder("source.ingest_date_utc"));

    let mut counts = BTreeMap::new();
    counts.insert("train_token_count", placeholder("counts.train_token_count"));
    counts.insert("val_token_count", placeholder("counts.val_token_count"));
    counts.insert("train_file_count", placeholder("counts.train_file_count"));
    counts.insert("val_file_count", placeholder("counts.val_file_count"));
    counts.insert(
        "pii_rejected_count",
        placeholder("counts.pii_rejected_count"),
    );
    counts.insert(
        "license_rejected_count",
        placeholder("counts.license_rejected_count"),
    );
    counts.insert(
        "dedup_rejected_count",
        placeholder("counts.dedup_rejected_count"),
    );

    DryRunManifest {
        dry_run: true,
        generated_at_utc: now_utc_iso8601(),
        generator: concat!("apr-corpus-ingest ", env!("CARGO_PKG_VERSION")),
        contract_path: contract_path.display().to_string(),
        contract_id: contract.contract_id.clone(),
        contract_version: contract.version.clone(),
        invariant_count: contract.invariants.len(),
        falsification_count: contract.falsification.len(),
        gate_count: contract.gates.len(),
        source,
        counts,
        corpus_sha256: placeholder("corpus_sha256"),
        gates: contract.gates.iter().map(|g| g.id.clone()).collect(),
    }
}

fn placeholder(field: &str) -> String {
    format!("TODO: {field} — fill in on real ingest")
}

/// Minimal, dependency-free ISO-8601 UTC timestamp. Avoids pulling `chrono`
/// into this binary: the constraints already pre-declare the dep set
/// (serde, serde_yaml, anyhow, clap) and chrono isn't on the list.
fn now_utc_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Civil-date conversion (Howard Hinnant, "date" algorithms).
    let days = (secs / 86_400) as i64;
    let sod = secs % 86_400;
    let (y, m, d) = civil_from_days(days);
    let hour = sod / 3600;
    let minute = (sod % 3600) / 60;
    let second = sod % 60;
    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Convert days-since-1970-01-01 to (year, month, day). Based on
/// H. Hinnant's `civil_from_days` — no external deps.
fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m, d)
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Resolve the real corpus contract path relative to the workspace root.
    /// Tests run with `CARGO_MANIFEST_DIR = .../crates/apr-cli`, so we walk
    /// up two levels to reach the contracts directory.
    fn real_contract_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root resolvable from CARGO_MANIFEST_DIR")
            .join("contracts/dataset-thestack-python-v1.yaml")
    }

    #[test]
    fn parses_real_contract_with_structural_invariants() {
        let path = real_contract_path();
        assert!(path.exists(), "real contract missing at {}", path.display());
        let yaml = fs::read_to_string(&path).expect("read real contract");
        let contract: CorpusContract = serde_yaml::from_str(&yaml).expect("contract deserializes");
        assert_structural_minimums(&contract).expect("structural minimums hold");

        assert!(
            contract.invariants.len() >= MIN_INVARIANTS,
            "expected >= {} invariants, got {}",
            MIN_INVARIANTS,
            contract.invariants.len()
        );
        assert!(
            contract.falsification.len() >= MIN_FALSIFICATIONS,
            "expected >= {} falsification tests, got {}",
            MIN_FALSIFICATIONS,
            contract.falsification.len()
        );
        assert!(
            contract.gates.len() >= MIN_GATES,
            "expected >= {} gates, got {}",
            MIN_GATES,
            contract.gates.len()
        );

        // IDs are non-empty and follow the documented prefixes.
        for inv in &contract.invariants {
            assert!(
                inv.id.starts_with("INV-DATA-"),
                "bad invariant id: {}",
                inv.id
            );
        }
        for f in &contract.falsification {
            assert!(
                f.id.starts_with("FALSIFY-DATA-"),
                "bad falsification id: {}",
                f.id
            );
        }
        for g in &contract.gates {
            assert!(g.id.starts_with("GATE-DATA-"), "bad gate id: {}", g.id);
        }
    }

    #[test]
    fn validate_contract_passes_on_real_contract_and_reports_top_keys() {
        let path = real_contract_path();
        let report = validate_contract(&path).expect("validation passes");
        assert_eq!(report.contract_id, "C-DATA-THESTACK-PYTHON");
        assert_eq!(report.invariants, 7);
        assert_eq!(report.falsification, 5);
        assert_eq!(report.gates, 5);

        // Every REQUIRED_TOP_KEYS entry must be present.
        for key in REQUIRED_TOP_KEYS {
            assert!(
                report.present_top_keys.iter().any(|k| k == *key),
                "missing required top-level key: {key}"
            );
        }
        assert_eq!(report.present_top_keys.len(), REQUIRED_TOP_KEYS.len());
    }
}
