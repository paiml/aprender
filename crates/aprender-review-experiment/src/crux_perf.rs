//! `crux-perf-receipt-v1`: one comparable speed receipt per rc per cell (CRUX perf history, G2).
//!
//! Contract: `contracts/crux-perf-receipt-v1.yaml`. RED stub: the F1–F5 tests land first.

use serde::{Deserialize, Serialize};

/// Schema tag every receipt carries.
pub const SCHEMA_VERSION: &str = "crux-perf-receipt-v1";

/// The measurement cell. Any field change starts a new series.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cell {
    pub host: String,
    pub gpu: String,
    pub driver_cuda: String,
    pub model: String,
    pub gguf_sha256: String,
    pub quant: String,
    pub ctx: u32,
    pub batch: u32,
    pub prompt_set_sha: String,
}

impl Cell {
    /// Content id of the cell.
    #[must_use]
    pub fn id(&self) -> String {
        String::new()
    }
}

/// The same-session competitor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Competitor {
    pub name: String,
    pub version: Option<String>,
    pub binary_sha256: String,
}

/// One phase's summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseStats {
    pub median: f64,
    pub p95: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
}

impl PhaseStats {
    #[must_use]
    pub fn from_samples(_xs: &[f64], _seed: u64) -> Option<Self> {
        None
    }
}

/// A timed phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    Ttft,
    Prefill,
    Decode,
}

/// The three phases.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phases {
    pub ttft: PhaseStats,
    pub prefill: PhaseStats,
    pub decode: PhaseStats,
}

impl Phases {
    #[must_use]
    pub fn from_samples(_t: &[f64], _p: &[f64], _d: &[f64], _seed: u64) -> Option<Self> {
        None
    }
}

/// apr / competitor medians.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ratios {
    pub ttft: f64,
    pub prefill: f64,
    pub decode: f64,
}

/// The receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CruxPerfReceipt {
    pub schema_version: String,
    pub tag: String,
    pub rc_binary_sha256: String,
    pub cell_id: String,
    pub cell: Cell,
    pub competitor: Competitor,
    pub prompt_set_sha: String,
    pub n_runs: usize,
    pub apr: Phases,
    pub competitor_phases: Phases,
    pub ratio_vs_competitor: Ratios,
    pub t1_topk_sha: Option<String>,
    pub t2_blob_sha: Option<String>,
    pub trace_overhead_pct: Option<f64>,
}

/// Why a receipt or a comparison is refused.
#[derive(Debug, Clone, PartialEq)]
pub enum Refusal {
    CompetitorVersionMissing,
    CellIdMismatch { claimed: String, derived: String },
    NewSeries { current: String, other: String },
    BinaryShaMismatch { receipt: String, tag_asset: String },
    RatioInconsistent(Phase),
}

/// What a T2 blob may be used for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum T2Status {
    Absent,
    Refused,
    AttributionOnly,
    Diffable,
}

/// A gate rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    Prev,
    Released,
    Rolling3,
}

/// One RED reason.
#[derive(Debug, Clone, PartialEq)]
pub struct Reason {
    pub phase: Phase,
    pub rule: Rule,
    pub delta: f64,
}

/// The gate outcome.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Gate {
    pub reasons: Vec<Reason>,
}

impl Gate {
    #[must_use]
    pub fn red(&self) -> bool {
        false
    }
}

impl CruxPerfReceipt {
    #[must_use]
    pub fn new(
        tag: String,
        rc_binary_sha256: String,
        cell: Cell,
        competitor: Competitor,
        apr: Phases,
        competitor_phases: Phases,
    ) -> Self {
        let prompt_set_sha = cell.prompt_set_sha.clone();
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            tag,
            rc_binary_sha256,
            cell_id: cell.id(),
            cell,
            competitor,
            prompt_set_sha,
            n_runs: 0,
            apr,
            competitor_phases,
            ratio_vs_competitor: Ratios { ttft: 0.0, prefill: 0.0, decode: 0.0 },
            t1_topk_sha: None,
            t2_blob_sha: None,
            trace_overhead_pct: None,
        }
    }

    /// # Errors
    /// The first refusal.
    pub fn check(&self, _tag_asset_sha256: &str) -> Result<(), Refusal> {
        Ok(())
    }

    #[must_use]
    pub fn holds(&self, tag_asset_sha256: &str) -> bool {
        self.check(tag_asset_sha256).is_ok()
    }

    #[must_use]
    pub fn t2_status(&self) -> T2Status {
        T2Status::Absent
    }
}

/// # Errors
/// A comparison across cells.
pub fn gate(
    _cur: &CruxPerfReceipt,
    _prev: Option<&CruxPerfReceipt>,
    _released: Option<&CruxPerfReceipt>,
    _last3: &[&CruxPerfReceipt],
) -> Result<Gate, Refusal> {
    Ok(Gate::default())
}

#[cfg(test)]
#[path = "crux_perf_tests.rs"]
mod tests;
