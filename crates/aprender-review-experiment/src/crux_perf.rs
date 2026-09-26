//! `crux-perf-receipt-v1`: one comparable speed receipt per rc per cell (CRUX perf history, G2).
//!
//! Contract: `contracts/crux-perf-receipt-v1.yaml` (Refs infra#1057). A receipt measures the
//! sha-verified rc binary against a competitor in the same session, on the same cell and GGUF,
//! over the PRM-S1 replay set. Every phase is a TIME (ttft ms, prefill and decode ms/token), so
//! lower is better and a positive delta is always a regression.
//!
//! T0 (this receipt's phases) is the gate. T1 and T2 are recorded by their content sha only: the
//! T2 layer-trace blob is private and never enters the repo, and it is usable only when the
//! tracer's overhead on the same rc was measured.

use crate::corpus::sha256_hex;
use crate::replay::{Engine, Row};
use crate::stats::{percentile, percentile_sorted, SplitMix64};
use serde::{Deserialize, Serialize};

/// Schema tag every receipt carries.
pub const SCHEMA_VERSION: &str = "crux-perf-receipt-v1";
/// Fewest samples a phase statistic is computed from.
pub const MIN_RUNS: usize = 5;
/// Bootstrap resamples for the median CI.
pub const BOOTSTRAP: usize = 2000;
/// Tracer overhead above which T2 is attribution-only.
pub const TRACE_OVERHEAD_MAX_PCT: f64 = 5.0;
/// D_prev: the CI lower bound past this is RED.
pub const D_PREV: f64 = 0.10;
/// D_released: the median past this vs the last released tag is RED.
pub const D_RELEASED: f64 = 0.05;
/// D_rolling3: the median past this vs the mean of the last three is RED.
pub const D_ROLLING3: f64 = 0.03;

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
    /// Content id of the cell: the first 16 hex of the sha256 of all nine fields, NUL-separated.
    #[must_use]
    pub fn id(&self) -> String {
        let ctx = self.ctx.to_string();
        let batch = self.batch.to_string();
        let fields = [
            self.host.as_str(),
            &self.gpu,
            &self.driver_cuda,
            &self.model,
            &self.gguf_sha256,
            &self.quant,
            &ctx,
            &batch,
            &self.prompt_set_sha,
        ];
        let mut h = sha256_hex(fields.join("\0").as_bytes());
        h.truncate(16);
        h
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
    /// Bootstrap 95% CI of the median.
    pub ci_lo: f64,
    pub ci_hi: f64,
    /// Samples the statistics came from.
    pub n: usize,
}

impl PhaseStats {
    /// `None` for fewer than [`MIN_RUNS`] samples or any non-finite or non-positive one: a
    /// dropped sample is never replaced by a median of the rest.
    #[must_use]
    pub fn from_samples(xs: &[f64], seed: u64) -> Option<Self> {
        if xs.len() < MIN_RUNS || xs.iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return None;
        }
        let mut rng = SplitMix64::new(seed);
        let mut medians: Vec<f64> = (0..BOOTSTRAP)
            .map(|_| {
                let s: Vec<f64> = (0..xs.len()).map(|_| xs[rng.below(xs.len())]).collect();
                percentile(&s, 0.5)
            })
            .collect();
        medians.sort_by(f64::total_cmp);
        Some(Self {
            median: percentile(xs, 0.5),
            p95: percentile(xs, 0.95),
            ci_lo: percentile_sorted(&medians, 0.025),
            ci_hi: percentile_sorted(&medians, 0.975),
            n: xs.len(),
        })
    }
}

/// Why replay rows give no phases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RowsError {
    NoRows,
    MixedSets,
    /// This item's server reported no timing (apr before SRV-TIM-001).
    UntimedRow(String),
    TooFewSamples(usize),
}

/// A timed phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    Ttft,
    Prefill,
    Decode,
}

const PHASES: [Phase; 3] = [Phase::Ttft, Phase::Prefill, Phase::Decode];

/// The fewest samples behind any phase of either engine: the receipt's `n_runs`.
fn min_runs(apr: &Phases, competitor: &Phases) -> usize {
    PHASES
        .iter()
        .flat_map(|p| [apr.get(*p).n, competitor.get(*p).n])
        .min()
        .unwrap_or(0)
}

/// The three phases: ttft ms, prefill ms/token, decode ms/token.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phases {
    pub ttft: PhaseStats,
    pub prefill: PhaseStats,
    pub decode: PhaseStats,
}

impl Phases {
    /// One engine's phases from a PRM-S1 replay run: one sample per item, as
    /// ttft = `prompt_ms`, prefill = 1000 / `prompt_tps`, decode = 1000 / `decode_tps`.
    ///
    /// # Errors
    /// No rows for `engine`, rows from more than one set, or any row with a null or
    /// non-positive server timing: a figure is never the median of the rows that were timed.
    pub fn from_replay_rows(rows: &[Row], engine: Engine, seed: u64) -> Result<Self, RowsError> {
        let mine: Vec<&Row> = rows.iter().filter(|r| r.engine == engine).collect();
        let Some(first) = mine.first() else {
            return Err(RowsError::NoRows);
        };
        if mine.iter().any(|r| r.set_sha != first.set_sha) {
            return Err(RowsError::MixedSets);
        }
        let per_tok = |tps: Option<f64>| tps.filter(|t| *t > 0.0).map(|t| 1000.0 / t);
        let mut t = Vec::with_capacity(mine.len());
        let mut p = Vec::with_capacity(mine.len());
        let mut d = Vec::with_capacity(mine.len());
        for r in &mine {
            match (r.prompt_ms, per_tok(r.prompt_tps), per_tok(r.decode_tps)) {
                (Some(a), Some(b), Some(c)) => {
                    t.push(a);
                    p.push(b);
                    d.push(c);
                }
                _ => return Err(RowsError::UntimedRow(r.diff_sha256.clone())),
            }
        }
        Self::from_samples(&t, &p, &d, seed).ok_or(RowsError::TooFewSamples(mine.len()))
    }

    /// All three from raw samples; `None` if any phase is.
    #[must_use]
    pub fn from_samples(ttft: &[f64], prefill: &[f64], decode: &[f64], seed: u64) -> Option<Self> {
        Some(Self {
            ttft: PhaseStats::from_samples(ttft, seed)?,
            prefill: PhaseStats::from_samples(prefill, seed ^ 1)?,
            decode: PhaseStats::from_samples(decode, seed ^ 2)?,
        })
    }

    #[must_use]
    pub fn get(&self, p: Phase) -> &PhaseStats {
        match p {
            Phase::Ttft => &self.ttft,
            Phase::Prefill => &self.prefill,
            Phase::Decode => &self.decode,
        }
    }
}

/// apr / competitor medians, per phase. Above 1 is apr slower.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ratios {
    pub ttft: f64,
    pub prefill: f64,
    pub decode: f64,
}

impl Ratios {
    fn of(apr: &Phases, competitor: &Phases) -> Self {
        let r = |p| apr.get(p).median / competitor.get(p).median;
        Self {
            ttft: r(Phase::Ttft),
            prefill: r(Phase::Prefill),
            decode: r(Phase::Decode),
        }
    }

    #[must_use]
    pub fn get(&self, p: Phase) -> f64 {
        match p {
            Phase::Ttft => self.ttft,
            Phase::Prefill => self.prefill,
            Phase::Decode => self.decode,
        }
    }
}

/// One rc's T0 receipt for one cell.
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
    SchemaVersion(String),
    CompetitorVersionMissing,
    CellIdMismatch { claimed: String, derived: String },
    PromptSetMismatch,
    TooFewRuns(usize),
    NewSeries { current: String, other: String },
    BinaryShaMismatch { receipt: String, tag_asset: String },
    RatioInconsistent(Phase),
}

/// What a T2 blob may be used for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum T2Status {
    Absent,
    /// No measured tracer overhead: the blob cannot be told from the tracer's own cost.
    Refused,
    /// Overhead above [`TRACE_OVERHEAD_MAX_PCT`]: explains a RED, never diffed as speed.
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

/// One RED reason: `delta` is the relative slowdown that crossed the rule's bound.
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
        !self.reasons.is_empty()
    }
}

impl CruxPerfReceipt {
    /// A receipt with its derived fields — cell id, prompt set, run count, ratios — computed,
    /// never supplied.
    #[must_use]
    pub fn new(
        tag: String,
        rc_binary_sha256: String,
        cell: Cell,
        competitor: Competitor,
        apr: Phases,
        competitor_phases: Phases,
    ) -> Self {
        let ratio_vs_competitor = Ratios::of(&apr, &competitor_phases);
        let n_runs = min_runs(&apr, &competitor_phases);
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            tag,
            rc_binary_sha256,
            cell_id: cell.id(),
            prompt_set_sha: cell.prompt_set_sha.clone(),
            cell,
            competitor,
            n_runs,
            apr,
            competitor_phases,
            ratio_vs_competitor,
            t1_topk_sha: None,
            t2_blob_sha: None,
            trace_overhead_pct: None,
        }
    }

    /// Admissibility against the rc tag's published asset sha.
    ///
    /// # Errors
    /// The first refusal, in schema → provenance → consistency order.
    pub fn check(&self, tag_asset_sha256: &str) -> Result<(), Refusal> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(Refusal::SchemaVersion(self.schema_version.clone()));
        }
        if tag_asset_sha256.is_empty()
            || !tag_asset_sha256.eq_ignore_ascii_case(&self.rc_binary_sha256)
        {
            return Err(Refusal::BinaryShaMismatch {
                receipt: self.rc_binary_sha256.clone(),
                tag_asset: tag_asset_sha256.to_string(),
            });
        }
        if self
            .competitor
            .version
            .as_deref()
            .is_none_or(|v| v.trim().is_empty())
        {
            return Err(Refusal::CompetitorVersionMissing);
        }
        let derived = self.cell.id();
        if derived != self.cell_id {
            return Err(Refusal::CellIdMismatch {
                claimed: self.cell_id.clone(),
                derived,
            });
        }
        if self.prompt_set_sha != self.cell.prompt_set_sha {
            return Err(Refusal::PromptSetMismatch);
        }
        let n = min_runs(&self.apr, &self.competitor_phases);
        if self.n_runs != n || n < MIN_RUNS {
            return Err(Refusal::TooFewRuns(n));
        }
        let want = Ratios::of(&self.apr, &self.competitor_phases);
        for p in PHASES {
            let (got, want) = (self.ratio_vs_competitor.get(p), want.get(p));
            if (got - want).abs() > 1e-9 * want.abs().max(1.0) {
                return Err(Refusal::RatioInconsistent(p));
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn holds(&self, tag_asset_sha256: &str) -> bool {
        self.check(tag_asset_sha256).is_ok()
    }

    /// What the T2 blob may be used for. A refused T2 never voids T0.
    #[must_use]
    pub fn t2_status(&self) -> T2Status {
        if self.t2_blob_sha.is_none() {
            return T2Status::Absent;
        }
        match self.trace_overhead_pct {
            Some(o) if o.is_finite() && o <= TRACE_OVERHEAD_MAX_PCT => T2Status::Diffable,
            Some(o) if o.is_finite() => T2Status::AttributionOnly,
            _ => T2Status::Refused,
        }
    }
}

/// The history gate for `cur` against the previous rc, the last released tag and the last
/// three receipts (all of the same cell). RED if, for any phase, the CI lower bound is more
/// than [`D_PREV`] past the previous median, the median more than [`D_RELEASED`] past the
/// released median, or more than [`D_ROLLING3`] past the mean of the last three medians.
/// `last3` of any other length is not evaluated.
///
/// # Errors
/// `cur`'s own cell id does not match its fields, or any comparand is another cell.
pub fn gate(
    cur: &CruxPerfReceipt,
    prev: Option<&CruxPerfReceipt>,
    released: Option<&CruxPerfReceipt>,
    last3: &[&CruxPerfReceipt],
) -> Result<Gate, Refusal> {
    let derived = cur.cell.id();
    if derived != cur.cell_id {
        return Err(Refusal::CellIdMismatch {
            claimed: cur.cell_id.clone(),
            derived,
        });
    }
    for o in prev.iter().chain(released.iter()).chain(last3.iter()) {
        if o.cell != cur.cell || o.cell_id != cur.cell_id {
            return Err(Refusal::NewSeries {
                current: cur.cell_id.clone(),
                other: o.cell_id.clone(),
            });
        }
    }
    let mut g = Gate::default();
    for phase in PHASES {
        let c = cur.apr.get(phase);
        let mut push = |rule, delta: f64, bound| {
            if delta > bound {
                g.reasons.push(Reason { phase, rule, delta });
            }
        };
        if let Some(p) = prev {
            push(Rule::Prev, c.ci_lo / p.apr.get(phase).median - 1.0, D_PREV);
        }
        if let Some(r) = released {
            push(
                Rule::Released,
                c.median / r.apr.get(phase).median - 1.0,
                D_RELEASED,
            );
        }
        if last3.len() == 3 {
            let mean = last3.iter().map(|r| r.apr.get(phase).median).sum::<f64>() / 3.0;
            push(Rule::Rolling3, c.median / mean - 1.0, D_ROLLING3);
        }
    }
    Ok(g)
}

/// What an admitted cell lacks for one rc (G3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gap {
    /// No admissible T0 receipt for the cell at this tag.
    T0 { cell_id: String },
    /// A T0 receipt with no T1 profile sha.
    T1 { cell_id: String },
}

/// G3: the gaps in `tag`'s perf history over the admitted cells. Empty is GREEN.
///
/// A receipt counts toward `tag` only when it is for `tag` and holds against the tag's asset
/// sha: a receipt of another binary or another tag never fills a cell.
/// An empty admitted-cell list returns no gaps; the caller must refuse it rather than print
/// GREEN.
#[must_use]
pub fn coverage_gaps(
    tag: &str,
    tag_asset_sha256: &str,
    admitted_cells: &[String],
    receipts: &[CruxPerfReceipt],
) -> Vec<Gap> {
    let mut gaps = Vec::new();
    for cell_id in admitted_cells {
        let here: Vec<&CruxPerfReceipt> = receipts
            .iter()
            .filter(|r| r.tag == tag && &r.cell_id == cell_id && r.holds(tag_asset_sha256))
            .collect();
        if here.is_empty() {
            gaps.push(Gap::T0 {
                cell_id: cell_id.clone(),
            });
        } else if !here.iter().any(|r| {
            r.t1_topk_sha
                .as_deref()
                .is_some_and(|s| !s.trim().is_empty())
        }) {
            gaps.push(Gap::T1 {
                cell_id: cell_id.clone(),
            });
        }
    }
    gaps
}

#[cfg(test)]
#[path = "crux_perf_tests.rs"]
mod tests;
