//! M1b artifact quality (EXT-001 §10 C1/C4, row EXT-27, aprender#4409).
//!
//! KL divergence against the BF16 reference and top-1 agreement, for our quant and for
//! each competitor arm (D-9), measured by the pinned llama.cpp on a committed corpus.
//! The gate is on the delta of the gap, never the gap (R-15): per arm,
//! `kl_gap = ours.kl_mean − arm.kl_mean` and `top1_gap = arm.top1 − ours.top1`, and M1b
//! is RED when either widens versus the previous release's record on the same corpus,
//! reference and pins. The first record only sets the baseline.

use super::model_gate::{row, GateRow, ReleaseManifest, M1_LLAMA_CPP_PIN};
use pacha::registry::is_sha256_hex;
use serde::{Deserialize, Serialize};

/// How far a KL gap may move before it has widened. `[A]`: llama.cpp's KL is a sum
/// over threads, so a re-run is not bit-identical; the first repeat records the noise.
pub(crate) const M1B_KL_EPS: f64 = 1e-4;
/// How far a top-1 gap may move before it has widened (0.1 pp). `[A]`, as above.
pub(crate) const M1B_TOP1_EPS: f64 = 1e-3;

/// One measured artifact: its bytes, and its quality against the reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct QualityArm {
    pub arm: String,
    pub file_sha256: String,
    /// Mean KL(reference ‖ arm) over the corpus positions, in nats.
    pub kl_mean: f64,
    /// Fraction of positions whose argmax equals the reference's.
    pub top1: f64,
}

/// One release's measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct QualityRecord {
    pub version: String,
    pub llama_cpp_commit: String,
    pub corpus_sha256: String,
    pub reference_sha256: String,
    pub ours: QualityArm,
    pub arms: Vec<QualityArm>,
}

/// M1b evidence: this release's record, the previous release's, and the receipt the
/// card cites.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct M1bEvidence {
    pub current: QualityRecord,
    /// `None` only for the first release of a line, which sets the baseline.
    #[serde(default)]
    pub baseline: Option<QualityRecord>,
    pub receipt_id: String,
}

/// One arm's gap now and at the baseline.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ArmGap {
    pub arm: String,
    pub kl_gap: f64,
    pub top1_gap: f64,
    /// `None`: a new or re-pinned arm, not compared.
    pub baseline_kl_gap: Option<f64>,
    pub baseline_top1_gap: Option<f64>,
    pub widened: bool,
}

/// What M1b adds to the gate receipt.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct M1bReport {
    pub first_record: bool,
    pub gaps: Vec<ArmGap>,
    /// The card block: absolute values per arm, each line citing the receipt, no ratio.
    pub card_markdown: String,
}

/// Run M1b against the release manifest `m`.
pub(crate) fn gate(m: &ReleaseManifest, ev: Option<&M1bEvidence>) -> (GateRow, Option<M1bReport>) {
    let _ = (m, ev);
    (row("M1b", Vec::new(), String::new()), None)
}

/// The card block for `rec`: one table row per arm, absolute values only.
pub(crate) fn card_markdown(rec: &QualityRecord, receipt_id: &str) -> String {
    let _ = (rec, receipt_id);
    String::new()
}

#[cfg(test)]
#[path = "model_gate_m1b_tests.rs"]
mod tests;
