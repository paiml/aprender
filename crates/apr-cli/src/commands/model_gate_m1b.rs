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
    let Some(ev) = ev else {
        return (
            row("M1b", vec!["no M1b quality evidence".into()], String::new()),
            None,
        );
    };
    let mut findings = Vec::new();
    if ev.receipt_id.trim().is_empty() {
        findings.push("no receipt id for the card's quality figures".into());
    }
    validate(&ev.current, "current", &mut findings);
    if !m
        .files
        .iter()
        .any(|f| f.sha256 == ev.current.ours.file_sha256)
    {
        findings.push(format!(
            "our measured file {} is not a file of this release",
            ev.current.ours.file_sha256
        ));
    }
    if let Some(b) = &ev.baseline {
        validate(b, "baseline", &mut findings);
        comparable(&ev.current, b, &mut findings);
    }
    let gaps = gaps(&ev.current, ev.baseline.as_ref());
    if ev.baseline.is_some() && gaps.iter().all(|g| g.baseline_kl_gap.is_none()) {
        findings.push("no arm is comparable with the baseline: nothing was gated".into());
    }
    for g in gaps.iter().filter(|g| g.widened) {
        if let Some(b) = g.baseline_kl_gap.filter(|b| g.kl_gap > b + M1B_KL_EPS) {
            findings.push(format!(
                "KL gap to {} widened: {:.6} → {:.6} nats",
                g.arm, b, g.kl_gap
            ));
        }
        if let Some(b) = g
            .baseline_top1_gap
            .filter(|b| g.top1_gap > b + M1B_TOP1_EPS)
        {
            findings.push(format!(
                "top-1 gap to {} widened: {:.6} → {:.6}",
                g.arm, b, g.top1_gap
            ));
        }
    }
    let first_record = ev.baseline.is_none();
    let checked = if first_record {
        format!(
            "first record: baseline set for {} arms",
            ev.current.arms.len()
        )
    } else {
        format!(
            "{} arms compared with {}, none widened",
            gaps.iter().filter(|g| g.baseline_kl_gap.is_some()).count(),
            ev.baseline.as_ref().map_or("", |b| b.version.as_str())
        )
    };
    let report = M1bReport {
        first_record,
        gaps,
        card_markdown: card_markdown(&ev.current, &ev.receipt_id),
    };
    (row("M1b", findings, checked), Some(report))
}

/// Every field a gap is computed from is a measurement, not a placeholder.
fn validate(rec: &QualityRecord, which: &str, findings: &mut Vec<String>) {
    if !rec.llama_cpp_commit.starts_with(M1_LLAMA_CPP_PIN) {
        findings.push(format!(
            "{which} record measured against llama.cpp {}, the pin is {M1_LLAMA_CPP_PIN}",
            rec.llama_cpp_commit
        ));
    }
    for (name, sha) in [
        ("corpus", &rec.corpus_sha256),
        ("reference", &rec.reference_sha256),
    ] {
        if !is_sha256_hex(sha) {
            findings.push(format!("{which} {name} sha256 is not a sha256: {sha:?}"));
        }
    }
    if rec.arms.is_empty() {
        findings.push(format!("{which} record has no competitor arm"));
    }
    let mut seen = std::collections::BTreeSet::new();
    for a in std::iter::once(&rec.ours).chain(&rec.arms) {
        if !is_sha256_hex(&a.file_sha256) {
            findings.push(format!("{which} arm {}: file is not a sha256", a.arm));
        }
        if !(a.kl_mean.is_finite() && a.kl_mean >= 0.0) {
            findings.push(format!(
                "{which} arm {}: KL {} is not a KL",
                a.arm, a.kl_mean
            ));
        }
        if !(0.0..=1.0).contains(&a.top1) {
            findings.push(format!(
                "{which} arm {}: top-1 {} is not a fraction",
                a.arm, a.top1
            ));
        }
        if !seen.insert(a.arm.as_str()) {
            findings.push(format!("{which} arm {} appears twice", a.arm));
        }
    }
}

/// R-15 compares like with like: the same corpus, reference and comparator.
fn comparable(cur: &QualityRecord, base: &QualityRecord, findings: &mut Vec<String>) {
    for (what, c, b) in [
        ("corpus", &cur.corpus_sha256, &base.corpus_sha256),
        ("reference", &cur.reference_sha256, &base.reference_sha256),
        ("llama.cpp", &cur.llama_cpp_commit, &base.llama_cpp_commit),
    ] {
        if c != b {
            findings.push(format!(
                "the {what} changed since the baseline ({b} → {c}): not comparable"
            ));
        }
    }
}

/// Per arm: `kl_gap = ours − arm`, `top1_gap = arm − ours` (positive = we trail). An
/// arm is compared only when the baseline measured the same bytes under the same id.
fn gaps(cur: &QualityRecord, base: Option<&QualityRecord>) -> Vec<ArmGap> {
    let gap = |ours: &QualityArm, a: &QualityArm| (ours.kl_mean - a.kl_mean, a.top1 - ours.top1);
    cur.arms
        .iter()
        .map(|a| {
            let (kl_gap, top1_gap) = gap(&cur.ours, a);
            let prev = base.and_then(|b| {
                b.arms
                    .iter()
                    .find(|p| p.arm == a.arm && p.file_sha256 == a.file_sha256)
                    .map(|p| gap(&b.ours, p))
            });
            let widened = prev
                .is_some_and(|(bk, bt)| kl_gap > bk + M1B_KL_EPS || top1_gap > bt + M1B_TOP1_EPS);
            ArmGap {
                arm: a.arm.clone(),
                kl_gap,
                top1_gap,
                baseline_kl_gap: prev.map(|p| p.0),
                baseline_top1_gap: prev.map(|p| p.1),
                widened,
            }
        })
        .collect()
}

/// The card block for `rec`: one table row per arm, absolute values only.
pub(crate) fn card_markdown(rec: &QualityRecord, receipt_id: &str) -> String {
    let mut md = String::from(
        "Artifact quality against the BF16 reference, measured per arm \
         (lower KL and higher top-1 agreement are closer to the reference).\n\n\
         | arm | file sha256 | mean KL (nats) | top-1 agreement (%) |\n\
         |---|---|---|---|\n",
    );
    for a in std::iter::once(&rec.ours).chain(&rec.arms) {
        let short = a.file_sha256.get(..12).unwrap_or(&a.file_sha256);
        md.push_str(&format!(
            "| {} | `{short}` | {:.4} | {:.2} [receipt:{receipt_id}] |\n",
            a.arm,
            a.kl_mean,
            a.top1 * 100.0
        ));
    }
    md
}

#[cfg(test)]
#[path = "model_gate_m1b_tests.rs"]
mod tests;
