//! `apr parity-oracle` — per-position logit parity against llama.cpp@d1d3c3396 (aprender#4444).
//!
//! `apr parity` compares apr's GPU against apr's CPU: one implementation, two
//! backends. The REX-04 receipt (`docs/audits/rex-001/rex-04-receipt.md`) found
//! that no apr verb compares apr's logits against an INDEPENDENT implementation,
//! so a `rex-cell-admission-v1` cell had nothing to declare as `parity.oracle`.
//! This verb fills that slot (PRM-001 v2 item 3a).
//!
//! Inputs are two APRRAWLG files over the SAME token ids:
//! * `--reference`: written by the vendored llama.cpp producer
//!   (`evidence/parity/l0-1/intel/qwen35-cpu-reference/producer/apr_raw_logits.cpp`,
//!   built against the pin `scripts/llama_pin.toml` `build_commit`);
//! * `--subject`: apr's logits for every position, in the same layout. OR
//! * `--model`: apr produces the subject in-process, running realizar's Qwen3.5 CPU
//!   forward over the REFERENCE's own token ids, so the ids agree by construction.
//!   The receipt records the forward and the model's sha256; `--subject-out` keeps
//!   the produced bytes, whose sha256 is the one in the receipt.
//!
//! The ids are compared first. An oracle fed different input measures nothing,
//! so an id mismatch is a refusal and never a RED (see the memory lesson "an oracle
//! fed the SUT's own input inherits its defect").
//!
//! Layout (little-endian): `char[8] "APRRAWLG" | u32 version=1 | i32 n_pos |
//! i32 n_vocab | i32 token_ids[n_pos] | f32 logits[n_pos*n_vocab]`.
//!
//! Verdict: GREEN iff min over positions of cosine(ref_i, sub_i) >= `--threshold`.
//! The threshold has NO default. It is required together with a non-empty
//! `--threshold-basis`, because a threshold without a basis is not a threshold
//! (`evidence/parity/thresholds.yaml`). The written receipt's sha256 and the
//! fields `{oracle, cosine, threshold, threshold_basis, receipt_sha256}` are
//! exactly `rex-cell-admission-v1`'s `Parity` block.
//!
//! Exit: 0 GREEN; 13 RED (`ParityFailed`); 4 unreadable/incomparable input;
//! 5 too few positions or a missing basis.

use std::path::Path;

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::threshold_arg;
use crate::error::{CliError, Result};

/// The oracle string a cell declares, byte for byte (`admission.rs` `LLAMA_CPP`).
/// Its commit must equal `scripts/llama_pin.toml` `build_commit`, and a test pins that.
pub(crate) const ORACLE: &str = "llama.cpp@d1d3c3396";

/// Receipt schema id.
pub(crate) const SCHEMA: &str = "apr-parity-oracle/v1";

/// `evidence/parity/thresholds.yaml` `min_positions` (basis: I8 / CF-4, #1864).
pub(crate) const DEFAULT_MIN_POSITIONS: usize = 64;

const MAGIC: &[u8; 8] = b"APRRAWLG";
const HEADER_LEN: usize = 20;

/// One parsed APRRAWLG file.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RawLogits {
    pub n_vocab: usize,
    pub token_ids: Vec<i32>,
    /// Row-major `[n_pos][n_vocab]`.
    pub logits: Vec<f32>,
}

impl RawLogits {
    pub(crate) fn n_pos(&self) -> usize {
        self.token_ids.len()
    }

    fn row(&self, pos: usize) -> &[f32] {
        &self.logits[pos * self.n_vocab..(pos + 1) * self.n_vocab]
    }
}

fn read_i32(buf: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// Parse an APRRAWLG buffer. `what` names the file in the error.
pub(crate) fn parse_raw_logits(buf: &[u8], what: &str) -> Result<RawLogits> {
    let bad = |m: String| CliError::InvalidFormat(format!("{what}: {m}"));
    if buf.len() < HEADER_LEN || &buf[..8] != MAGIC {
        return Err(bad(
            "not an APRRAWLG file (bad magic or short header)".into()
        ));
    }
    let version = read_i32(buf, 8);
    if version != 1 {
        return Err(bad(format!("APRRAWLG version {version}, expected 1")));
    }
    let (n_pos, n_vocab) = (read_i32(buf, 12), read_i32(buf, 16));
    if n_pos <= 0 || n_vocab <= 0 {
        return Err(bad(format!(
            "n_pos={n_pos} n_vocab={n_vocab}; both must be > 0"
        )));
    }
    let (n_pos, n_vocab) = (n_pos as usize, n_vocab as usize);
    let off = HEADER_LEN + 4 * n_pos;
    let want = n_pos
        .checked_mul(n_vocab)
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| n.checked_add(off));
    if want != Some(buf.len()) {
        return Err(bad(format!(
            "size {} does not match header (n_pos={n_pos}, n_vocab={n_vocab})",
            buf.len()
        )));
    }
    let token_ids = (0..n_pos)
        .map(|i| read_i32(buf, HEADER_LEN + 4 * i))
        .collect();
    let logits = buf[off..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(RawLogits {
        n_vocab,
        token_ids,
        logits,
    })
}

/// Serialize to APRRAWLG (the inverse of [`parse_raw_logits`]; used by fixtures).
pub(crate) fn encode_raw_logits(raw: &RawLogits) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + 4 * (raw.n_pos() + raw.logits.len()));
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(raw.n_pos() as i32).to_le_bytes());
    out.extend_from_slice(&(raw.n_vocab as i32).to_le_bytes());
    for id in &raw.token_ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    for v in &raw.logits {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// One position's comparison.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct PositionRow {
    pub pos: usize,
    pub token_id: i32,
    pub cosine: f64,
    pub argmax_reference: usize,
    pub argmax_subject: usize,
    pub max_abs_diff: f64,
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    best
}

/// Cosine in f64. A zero-norm row yields NaN, which the verdict treats as RED.
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn compare_position(reference: &RawLogits, subject: &RawLogits, pos: usize) -> PositionRow {
    let (r, s) = (reference.row(pos), subject.row(pos));
    let max_abs_diff = r
        .iter()
        .zip(s)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
        .fold(0.0, f64::max);
    PositionRow {
        pos,
        token_id: reference.token_ids[pos],
        cosine: cosine(r, s),
        argmax_reference: argmax(r),
        argmax_subject: argmax(s),
        max_abs_diff,
    }
}

/// Refuse a pair that cannot be compared: shape, ids, or non-finite logits.
pub(crate) fn check_comparable(reference: &RawLogits, subject: &RawLogits) -> Result<()> {
    if reference.n_vocab != subject.n_vocab || reference.n_pos() != subject.n_pos() {
        return Err(CliError::InvalidFormat(format!(
            "shape mismatch: reference [{}x{}] vs subject [{}x{}]",
            reference.n_pos(),
            reference.n_vocab,
            subject.n_pos(),
            subject.n_vocab
        )));
    }
    if let Some(i) =
        (0..reference.n_pos()).find(|&i| reference.token_ids[i] != subject.token_ids[i])
    {
        return Err(CliError::InvalidFormat(format!(
            "token ids differ at position {i} (reference {} vs subject {}): the oracle was not fed the subject's input, so nothing is measured",
            reference.token_ids[i], subject.token_ids[i]
        )));
    }
    for (what, raw) in [("reference", reference), ("subject", subject)] {
        let bad = raw.logits.iter().filter(|v| !v.is_finite()).count();
        if bad > 0 {
            return Err(CliError::InvalidFormat(format!(
                "{what}: {bad} non-finite logits"
            )));
        }
    }
    Ok(())
}

/// Where a file came from, as the receipt records it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct InputRecord {
    pub path: String,
    pub sha256: String,
    /// Set when apr produced this input in-process (`--model`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub producer: Option<Producer>,
}

/// What produced an in-process subject: the forward, and the exact model file.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct Producer {
    pub forward: &'static str,
    pub model_path: String,
    pub model_sha256: String,
}

/// The forward `--model` runs. Per-token on purpose: it is the path whose every
/// position has a logits row, and the same loop the #3091 evidence harness used.
pub(crate) const QWEN35_CPU_FORWARD: &str =
    "realizar Qwen35Model::forward_single_qwen35 (CPU, one token per call)";

/// Where the subject logits come from.
pub(crate) enum Subject<'a> {
    /// An APRRAWLG file written elsewhere.
    File(&'a Path),
    /// Produced here from a Qwen3.5 GGUF; optionally saved to `save`.
    Model {
        model: &'a Path,
        save: Option<&'a Path>,
    },
}

/// The receipt. `receipt_sha256` is the sha256 of these serialized bytes.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct Receipt {
    pub schema: &'static str,
    pub oracle: &'static str,
    pub verdict: &'static str,
    /// Min cosine over positions (admission `Parity.cosine`).
    pub cosine: f64,
    pub min_cosine_pos: usize,
    pub threshold: f64,
    pub threshold_basis: String,
    pub n_positions: usize,
    pub n_vocab: usize,
    pub positions_below_threshold: Vec<usize>,
    pub argmax_mismatches: Vec<usize>,
    pub max_abs_diff: f64,
    pub reference: InputRecord,
    pub subject: InputRecord,
    pub per_position: Vec<PositionRow>,
}

/// Compare two comparable files and build the receipt. Pure; no I/O.
pub(crate) fn judge(
    reference: &RawLogits,
    subject: &RawLogits,
    threshold: f64,
    threshold_basis: &str,
    inputs: (InputRecord, InputRecord),
) -> Receipt {
    let rows: Vec<PositionRow> = (0..reference.n_pos())
        .map(|p| compare_position(reference, subject, p))
        .collect();
    // A NaN cosine (zero-norm row) ranks as -inf: it is the reported min and it is below.
    let rank = |r: &PositionRow| {
        if r.cosine.is_nan() {
            f64::NEG_INFINITY
        } else {
            r.cosine
        }
    };
    let min_row = rows
        .iter()
        .min_by(|a, b| rank(a).total_cmp(&rank(b)))
        .expect("n_pos > 0 is checked at parse");
    let below: Vec<usize> = rows
        .iter()
        .filter(|r| r.cosine.is_nan() || r.cosine < threshold)
        .map(|r| r.pos)
        .collect();
    Receipt {
        schema: SCHEMA,
        oracle: ORACLE,
        verdict: if below.is_empty() { "GREEN" } else { "RED" },
        cosine: min_row.cosine,
        min_cosine_pos: min_row.pos,
        threshold,
        threshold_basis: threshold_basis.to_string(),
        n_positions: rows.len(),
        n_vocab: reference.n_vocab,
        positions_below_threshold: below,
        argmax_mismatches: rows
            .iter()
            .filter(|r| r.argmax_reference != r.argmax_subject)
            .map(|r| r.pos)
            .collect(),
        max_abs_diff: rows.iter().map(|r| r.max_abs_diff).fold(0.0, f64::max),
        reference: inputs.0,
        subject: inputs.1,
        per_position: rows,
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn load(path: &Path, what: &str) -> Result<(RawLogits, InputRecord)> {
    let buf = std::fs::read(path)
        .map_err(|e| CliError::InvalidFormat(format!("{what} {}: {e}", path.display())))?;
    let raw = parse_raw_logits(&buf, what)?;
    let rec = InputRecord {
        path: path.display().to_string(),
        sha256: sha256_hex(&buf),
        producer: None,
    };
    Ok((raw, rec))
}

fn check_args(threshold: f64, threshold_basis: &str) -> Result<()> {
    threshold_arg::guard("--threshold", threshold, threshold_arg::COSINE)?;
    if threshold_basis.trim().is_empty() {
        return Err(CliError::ValidationFailed(
            "apr parity-oracle: --threshold-basis is empty; a threshold without a basis is not a threshold".into(),
        ));
    }
    Ok(())
}

/// Run `forward(token, pos)` over `token_ids` and keep EVERY row. Pure: the forward
/// is injected, so the refusals are testable without a model. An id outside
/// `0..vocab` is refused before the forward sees it (the embedding lookup is an
/// unchecked slice), and rows must all have one length.
pub(crate) fn produce_subject(
    token_ids: &[i32],
    vocab: usize,
    mut forward: impl FnMut(u32, usize) -> Result<Vec<f32>>,
) -> Result<RawLogits> {
    let mut n_vocab = None;
    let mut logits = Vec::new();
    for (pos, &id) in token_ids.iter().enumerate() {
        let token = u32::try_from(id)
            .ok()
            .filter(|&t| usize::try_from(t).is_ok_and(|t| t < vocab))
            .ok_or_else(|| {
                CliError::InvalidFormat(format!(
                    "apr parity-oracle: reference token id {id} at position {pos} is outside the model's vocab of {vocab}"
                ))
            })?;
        let row = forward(token, pos)?;
        let width = *n_vocab.get_or_insert(row.len());
        if row.len() != width || width == 0 {
            return Err(CliError::InvalidFormat(format!(
                "apr parity-oracle: the forward returned {} logits at position {pos}, {width} before",
                row.len()
            )));
        }
        logits.extend_from_slice(&row);
    }
    Ok(RawLogits {
        n_vocab: n_vocab.unwrap_or(0),
        token_ids: token_ids.to_vec(),
        logits,
    })
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| CliError::InvalidFormat(format!("--model {}: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// `--model`: produce the subject, optionally save it, and record what produced it.
/// The recorded sha256 is of the encoded bytes, i.e. of the `--subject-out` file.
fn subject_from_model(
    model: &Path,
    save: Option<&Path>,
    token_ids: &[i32],
) -> Result<(RawLogits, InputRecord)> {
    let model_sha256 = sha256_file(model)?;
    let raw = qwen35_cpu_subject(model, token_ids)?;
    let bytes = encode_raw_logits(&raw);
    if let Some(out) = save {
        std::fs::write(out, &bytes)?;
    }
    let rec = InputRecord {
        path: save.map_or_else(|| "<in-process>".to_string(), |p| p.display().to_string()),
        sha256: sha256_hex(&bytes),
        producer: Some(Producer {
            forward: QWEN35_CPU_FORWARD,
            model_path: model.display().to_string(),
            model_sha256,
        }),
    };
    Ok((raw, rec))
}

#[cfg(feature = "inference")]
fn qwen35_cpu_subject(model: &Path, token_ids: &[i32]) -> Result<RawLogits> {
    use realizar::gguf::forward_qwen35::Qwen35Model;
    use realizar::gguf::MappedGGUFModel;

    let mapped = MappedGGUFModel::from_path(model)
        .map_err(|e| CliError::InvalidFormat(format!("--model {}: {e}", model.display())))?;
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).map_err(|e| {
        CliError::ValidationFailed(format!("--model: not a loadable Qwen3.5 base model: {e}"))
    })?;
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).map_err(|e| {
            CliError::ValidationFailed(format!(
                "--model: the Qwen3.5 hybrid layers will not load: {e}"
            ))
        })?;
    let mut state = qwen.new_state(token_ids.len() + 1);
    produce_subject(token_ids, base.config().vocab_size, |token, pos| {
        qwen.forward_single_qwen35(token, &mut state, pos)
            .map_err(|e| CliError::ValidationFailed(format!("apr forward at position {pos}: {e}")))
    })
}

#[cfg(not(feature = "inference"))]
fn qwen35_cpu_subject(_model: &Path, _token_ids: &[i32]) -> Result<RawLogits> {
    Err(CliError::ValidationFailed(
        "apr parity-oracle --model needs apr built with the `inference` feature".into(),
    ))
}

/// Run the verb: load, refuse the incomparable, judge, write the receipt, report.
pub(crate) fn run(
    reference: &Path,
    subject: Subject<'_>,
    threshold: f64,
    threshold_basis: &str,
    min_positions: usize,
    receipt_out: &Path,
    json: bool,
) -> Result<()> {
    check_args(threshold, threshold_basis)?;
    let (ref_raw, ref_rec) = load(reference, "reference")?;
    // Before the subject: with `--model` the subject is a full forward pass.
    if ref_raw.n_pos() < min_positions {
        return Err(CliError::ValidationFailed(format!(
            "apr parity-oracle: {} positions < --min-positions {min_positions}; too few to judge",
            ref_raw.n_pos()
        )));
    }
    let (sub_raw, sub_rec) = match subject {
        Subject::File(path) => load(path, "subject")?,
        Subject::Model { model, save } => subject_from_model(model, save, &ref_raw.token_ids)?,
    };
    check_comparable(&ref_raw, &sub_raw)?;
    let receipt = judge(
        &ref_raw,
        &sub_raw,
        threshold,
        threshold_basis,
        (ref_rec, sub_rec),
    );
    let bytes = serde_json::to_vec_pretty(&receipt)
        .map_err(|e| CliError::InvalidFormat(format!("serialize receipt: {e}")))?;
    std::fs::write(receipt_out, &bytes)?;
    let receipt_sha256 = sha256_hex(&bytes);
    report(&receipt, receipt_out, &receipt_sha256, json);
    if receipt.verdict == "GREEN" {
        Ok(())
    } else {
        Err(CliError::ParityFailed(format!(
            "{ORACLE}: min cosine {:.6} at position {} < threshold {} ({} of {} positions below)",
            receipt.cosine,
            receipt.min_cosine_pos,
            threshold,
            receipt.positions_below_threshold.len(),
            receipt.n_positions
        )))
    }
}

fn report(receipt: &Receipt, receipt_out: &Path, receipt_sha256: &str, json: bool) {
    if json {
        // `parity` is the rex-cell-admission-v1 `Parity` block, ready to paste.
        let body = serde_json::json!({
            "verdict": receipt.verdict,
            "receipt": receipt_out.display().to_string(),
            "parity": {
                "oracle": receipt.oracle,
                "cosine": receipt.cosine,
                "threshold": receipt.threshold,
                "threshold_basis": receipt.threshold_basis,
                "receipt_sha256": receipt_sha256,
            },
        });
        println!("{body}");
        return;
    }
    println!(
        "{}: {} — min cosine {:.6} at pos {} (threshold {}), {} positions, {} argmax mismatches, max |diff| {:.6}",
        receipt.oracle,
        receipt.verdict,
        receipt.cosine,
        receipt.min_cosine_pos,
        receipt.threshold,
        receipt.n_positions,
        receipt.argmax_mismatches.len(),
        receipt.max_abs_diff
    );
    println!("receipt: {} sha256 {receipt_sha256}", receipt_out.display());
}

#[cfg(test)]
#[path = "parity_oracle_tests.rs"]
mod tests;
