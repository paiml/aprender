//! OBS-05 nightly perf-ledger checks (APR-OBS-001 v1.3 §2.1–§2.8, §3, §10 E1/E2/E8/E12;
//! contract `apr-perf-ledger-v1`).
//!
//! The reader side of the ledger the recorder writes. Every §3 falsifier for
//! `apr-perf-ledger-v1` is a refusal here:
//!
//! - a row is admitted only with the full §2.1 identity block (version and sha included),
//!   a proven GPU claim (§2.5), an ABBA/BAAB order of exactly `B = 3` blocks (§2.4) and a
//!   passing quiescence proof (§2.6); anything else is counted as absent, never as partial;
//! - a night is RED when the ledger is empty or a DECLARED series has no admissible row;
//! - two rows are divided only within identity (host, backend, tier, band, model): a
//!   mismatch is fatal, a different apr build is annotated (it is the signal);
//! - the chain (E8) catches a mid-file edit, and only the anchor catches tail truncation;
//! - the ledger file must not be writable by the subject's uid (E12);
//! - E1/E2 are computed from blocks, and a block whose quiescence failed at start or end
//!   is voided and never reaches the median.

use crate::obs_backfill::{known, MAX_LINE_BYTES};
use crate::obs_epoch::{canonical, median, PERF_SCHEMA};
use chrono::NaiveDate;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// §2.1 fields that must be present, non-null, non-empty and not `"unknown"`.
pub const IDENTITY: [&str; 20] = [
    "schema",
    "ts",
    "host",
    "apr_version",
    "apr_tag",
    "crate_tarball_sha256",
    "binary_sha256",
    "build_identity",
    "comparator",
    "model_id",
    "model_sha256",
    "backend",
    "quiescence_proof",
    "tier",
    "band",
    "order",
    "epoch_id",
    "lease_id",
    "prev_row_sha256",
    "request_id",
];

/// Backends whose claim needs a `gpu_proof` (§2.5).
pub const GPU_BACKENDS: [&str; 3] = ["cuda", "wgpu", "metal"];
/// Blocks per band (§2.4).
pub const BLOCKS: usize = 3;

/// Why a row is not admitted. Every variant counts the row as absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// Not one JSON object on one line, or over [`MAX_LINE_BYTES`].
    Malformed(String),
    /// `schema` is not `apr-perf-ledger-v1`.
    WrongSchema(String),
    /// Identity fields absent, null, `""` or `"unknown"` (named).
    Inadmissible(Vec<String>),
    /// A GPU backend with a null `gpu_proof`: excluded from GPU series (§2.5).
    BackendUnproven,
    /// `backend = cpu` with a non-null `gpu_proof`: the row contradicts itself.
    CpuWithGpuProof,
    /// `kernel_path` null without a `kernel_path_reason`.
    KernelPathUnexplained,
    /// Not `design = "ABBA"` with a seed and exactly [`BLOCKS`] `ABBA`/`BAAB` blocks.
    NotAbba(String),
    /// Quiescence failed (§2.6): the reasons, e.g. `no_lease`, `foreign_gpu_procs`.
    HostUnproven(Vec<&'static str>),
}

/// Why a quiescence proof fails (§2.6). Empty ⇒ quiescent.
#[must_use]
pub fn quiescence_failures(q: &Value) -> Vec<&'static str> {
    let mut why = Vec::new();
    if !q.is_object() {
        why.push("no_proof");
        return why;
    }
    if !known(q.get("lease_id")) {
        why.push("no_lease");
    }
    match q.get("foreign_gpu_procs") {
        Some(Value::Array(a)) if a.is_empty() => {}
        Some(Value::Array(_)) => why.push("foreign_gpu_procs"),
        _ => why.push("foreign_gpu_procs_unread"),
    }
    match q.get("train_active") {
        Some(Value::Bool(false)) => {}
        Some(Value::Bool(true)) => why.push("train_active"),
        _ => why.push("train_active_unread"),
    }
    why
}

fn order_error(order: &Value) -> Option<String> {
    if order.get("design").and_then(Value::as_str) != Some("ABBA") {
        return Some(format!("design {:?}", order.get("design")));
    }
    if !order.get("seed").is_some_and(Value::is_u64) {
        return Some("seed not recorded".into());
    }
    let blocks = order.get("blocks").and_then(Value::as_array);
    match blocks {
        Some(b) if b.len() == BLOCKS => b
            .iter()
            .find(|x| !matches!(x.as_str(), Some("ABBA" | "BAAB")))
            .map(|x| format!("block {x}")),
        Some(b) => Some(format!("{} blocks, need {BLOCKS}", b.len())),
        None => Some("blocks absent".into()),
    }
}

/// Admit one row, or say why it counts as absent.
pub fn admit_row(row: &Value) -> Result<(), Refusal> {
    if !row.is_object() {
        return Err(Refusal::Malformed("not a JSON object".into()));
    }
    let missing: Vec<String> = IDENTITY
        .iter()
        .filter(|f| !known(row.get(**f)))
        .map(|f| (*f).to_string())
        .collect();
    let mut missing = missing;
    for key in ["gpu_proof", "kernel_path"] {
        if row.get(key).is_none() {
            missing.push(key.to_string());
        }
    }
    if !missing.is_empty() {
        return Err(Refusal::Inadmissible(missing));
    }
    let schema = row["schema"].as_str().unwrap_or_default();
    if schema != PERF_SCHEMA {
        return Err(Refusal::WrongSchema(schema.to_string()));
    }
    let gpu = GPU_BACKENDS.contains(&row["backend"].as_str().unwrap_or_default());
    match (gpu, row["gpu_proof"].is_null()) {
        (true, true) => return Err(Refusal::BackendUnproven),
        (false, false) => return Err(Refusal::CpuWithGpuProof),
        _ => {}
    }
    if row["kernel_path"].is_null() && !known(row.get("kernel_path_reason")) {
        return Err(Refusal::KernelPathUnexplained);
    }
    if let Some(e) = order_error(&row["order"]) {
        return Err(Refusal::NotAbba(e));
    }
    let q = quiescence_failures(&row["quiescence_proof"]);
    if !q.is_empty() {
        return Err(Refusal::HostUnproven(q));
    }
    Ok(())
}

/// Parse and admit one ledger line.
pub fn admit_line(line: &str) -> Result<Value, Refusal> {
    if line.len() > MAX_LINE_BYTES {
        return Err(Refusal::Malformed(format!(
            "{} bytes > {MAX_LINE_BYTES}",
            line.len()
        )));
    }
    let row: Value = serde_json::from_str(line).map_err(|e| Refusal::Malformed(e.to_string()))?;
    admit_row(&row)?;
    Ok(row)
}

/// A forjar-declared series: never derived from the rows present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Series {
    /// forjar machine name.
    pub host: String,
    /// `cpu` / `cuda` / `wgpu` / `metal`.
    pub backend: String,
    /// `engine` / `serve`.
    pub tier: String,
}

/// One night's liveness verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct NightVerdict {
    /// The ledger had no line at all.
    pub empty: bool,
    /// Declared series with no admissible row and no skip row tonight.
    pub missing: Vec<Series>,
    /// Declared series that wrote `{state: "skipped"}` (§2.8): not admissible, not a gap.
    pub skipped: Vec<Series>,
    /// Lines of tonight refused, by line index.
    pub refused: Vec<(usize, Refusal)>,
}

impl NightVerdict {
    /// RED: an empty ledger or a declared series missing (R-2). A single skip is judged
    /// by the 14-night liveness rule (E7), not here.
    #[must_use]
    pub fn red(&self) -> bool {
        self.empty || !self.missing.is_empty()
    }
}

fn night_of(row: &Value) -> Option<NaiveDate> {
    let ts = row.get("ts")?.as_str()?;
    NaiveDate::parse_from_str(ts.get(..10)?, "%Y-%m-%d").ok()
}

fn is_series(row: &Value, s: &Series) -> bool {
    row.get("host").and_then(Value::as_str) == Some(s.host.as_str())
        && row.get("backend").and_then(Value::as_str) == Some(s.backend.as_str())
        && row.get("tier").and_then(Value::as_str) == Some(s.tier.as_str())
}

/// Judge one night of a ledger against the declared series.
#[must_use]
pub fn judge_night(lines: &[&str], declared: &[Series], night: NaiveDate) -> NightVerdict {
    let mut admitted = Vec::new();
    let mut skips = Vec::new();
    let mut refused = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let parsed: Option<Value> = serde_json::from_str(line).ok();
        if let Some(v) = &parsed {
            if night_of(v) != Some(night) {
                continue;
            }
            if v.get("state").and_then(Value::as_str) == Some("skipped") {
                skips.push(v.clone());
                continue;
            }
        }
        match admit_line(line) {
            Ok(v) => admitted.push(v),
            Err(r) => refused.push((i, r)),
        }
    }
    let mut missing = Vec::new();
    let mut skipped = Vec::new();
    for s in declared {
        if admitted.iter().any(|r| is_series(r, s)) {
            continue;
        }
        if skips.iter().any(|r| is_series(r, s)) {
            skipped.push(s.clone());
        } else {
            missing.push(s.clone());
        }
    }
    NightVerdict {
        empty: lines.iter().all(|l| l.trim().is_empty()),
        missing,
        skipped,
        refused,
    }
}

/// Fields two rows must share before any ratio between them (§2.1, S-4).
pub const RATIO_IDENTITY: [&str; 5] = ["host", "backend", "tier", "band", "model_sha256"];
/// Fields that may differ across nights of one host, annotated (the apr build is the signal).
pub const ANNOTATED: [&str; 2] = ["crate_tarball_sha256", "binary_sha256"];

/// `Ok(annotations)` if `a` and `b` may be divided; `Err(fields)` is FATAL.
pub fn comparable(a: &Value, b: &Value) -> Result<Vec<&'static str>, Vec<&'static str>> {
    let fatal: Vec<&'static str> = RATIO_IDENTITY
        .iter()
        .copied()
        .filter(|f| a.get(*f).is_none() || a.get(*f) != b.get(*f))
        .collect();
    if !fatal.is_empty() {
        return Err(fatal);
    }
    Ok(ANNOTATED
        .iter()
        .copied()
        .filter(|f| a.get(*f) != b.get(*f))
        .collect())
}

/// `h_k = sha256(canon(row_k))`, lowercase hex.
#[must_use]
pub fn row_hash(row: &Value) -> String {
    format!("{:x}", Sha256::digest(canonical(row).as_bytes()))
}

/// E8 genesis `h_0 = sha256(schema ‖ host ‖ backend ‖ epoch_start)`, byte concatenation.
#[must_use]
pub fn genesis(schema: &str, host: &str, backend: &str, epoch_start: &str) -> String {
    let mut h = Sha256::new();
    for part in [schema, host, backend, epoch_start] {
        h.update(part.as_bytes());
    }
    format!("{:x}", h.finalize())
}

/// Why a chain does not verify (§2.2, §10 T5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainError {
    /// Line `i` is not JSON.
    Unparsable(usize),
    /// Row `i`'s `prev_row_sha256` is not `h_{i−1}`: an edit, deletion or reorder at or before `i`.
    Break(usize),
    /// Fewer rows than the anchor recorded: tail truncation.
    BelowAnchor {
        /// Rows present now.
        rows: usize,
        /// Rows the anchor recorded.
        anchored: usize,
    },
    /// Row `n_rows − 1` does not hash to the anchor's `h_n`: the anchored prefix changed.
    AnchorMismatch,
}

/// A RED-job anchor `(n_rows, h_n)` (E8).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Anchor {
    /// Rows the ledger had when anchored.
    pub n_rows: usize,
    /// `h_{n_rows}` — the hash of the last anchored row.
    pub h_n: String,
}

/// Verify the chain from `h0` and, if given, the anchor. Returns the head `(n_rows, h_n)`.
pub fn verify_chain(
    lines: &[&str],
    h0: &str,
    anchor: Option<&Anchor>,
) -> Result<Anchor, ChainError> {
    let mut hashes = Vec::with_capacity(lines.len());
    let mut prev = h0.to_string();
    for (i, line) in lines.iter().enumerate() {
        let row: Value = serde_json::from_str(line).map_err(|_| ChainError::Unparsable(i))?;
        if row.get("prev_row_sha256").and_then(Value::as_str) != Some(prev.as_str()) {
            return Err(ChainError::Break(i));
        }
        prev = row_hash(&row);
        hashes.push(prev.clone());
    }
    if let Some(a) = anchor {
        if lines.len() < a.n_rows {
            return Err(ChainError::BelowAnchor {
                rows: lines.len(),
                anchored: a.n_rows,
            });
        }
        let at = if a.n_rows == 0 {
            h0.to_string()
        } else {
            hashes[a.n_rows - 1].clone()
        };
        if at != a.h_n {
            return Err(ChainError::AnchorMismatch);
        }
    }
    Ok(Anchor {
        n_rows: lines.len(),
        h_n: prev,
    })
}

/// E12: the ledger's owner is not a subject uid, and neither group nor other may write it.
pub fn writer_check(owner_uid: u32, mode: u32, subject_uids: &[u32]) -> Result<(), String> {
    if subject_uids.contains(&owner_uid) {
        return Err(format!("ledger owned by subject uid {owner_uid}"));
    }
    if mode & 0o022 != 0 {
        return Err(format!(
            "ledger mode {:o} is group/other writable",
            mode & 0o777
        ));
    }
    Ok(())
}

/// [`writer_check`] on a file's metadata.
pub fn ledger_writer_check(path: &std::path::Path, subject_uids: &[u32]) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;
    let m = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    writer_check(m.uid(), m.mode(), subject_uids)
}

/// Block order (§2.4, E1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// Slots A, B, B, A.
    Abba,
    /// Slots B, A, A, B.
    Baab,
}

/// One ABBA block as the recorder measured it.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    /// Start of the block's first slot, seconds (blocks need not be contiguous, §2.8).
    pub start_s: u64,
    /// ABBA or BAAB.
    pub order: Order,
    /// x (> 0, higher is better) in slot order t, t+w, t+2w, t+3w.
    pub x: [f64; 4],
    /// Quiescence held at block start.
    pub quiet_start: bool,
    /// Quiescence held at block end.
    pub quiet_end: bool,
}

/// E1 `ℓ_b`, pairing by adjacency.
#[must_use]
pub fn block_stat(b: &Block) -> f64 {
    let [s0, s1, s2, s3] = b.x;
    let (a1, b1, a2, b2) = match b.order {
        Order::Abba => (s0, s1, s3, s2),
        Order::Baab => (s1, s0, s2, s3),
    };
    0.5 * ((a1 / b1).ln() + (a2 / b2).ln())
}

/// Why a band has no `L_n` tonight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NightError {
    /// Fewer than [`BLOCKS`] valid blocks by window end: written as a skip (§2.8).
    Skipped {
        /// Valid blocks measured.
        blocks_valid: usize,
    },
    /// A valid block holds a non-positive or non-finite x.
    BadSample(usize),
}

/// E2 `L_n = med ℓ_b` over the first [`BLOCKS`] valid blocks in time order.
/// A voided block (quiescence failed at start or end) never enters the median.
pub fn nightly_stat(blocks: &[Block]) -> Result<f64, NightError> {
    let mut valid: Vec<&Block> = blocks
        .iter()
        .filter(|b| b.quiet_start && b.quiet_end)
        .collect();
    valid.sort_by_key(|b| b.start_s);
    if valid.len() < BLOCKS {
        return Err(NightError::Skipped {
            blocks_valid: valid.len(),
        });
    }
    let mut ls = Vec::with_capacity(BLOCKS);
    for (i, b) in valid.iter().take(BLOCKS).enumerate() {
        if !b.x.iter().all(|x| x.is_finite() && *x > 0.0) {
            return Err(NightError::BadSample(i));
        }
        ls.push(block_stat(b));
    }
    median(&ls).ok_or(NightError::Skipped { blocks_valid: 0 })
}

#[cfg(test)]
#[path = "obs_ledger_tests.rs"]
mod tests;
