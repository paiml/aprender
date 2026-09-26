//! EXT-28 (aprender#4410): the C2 speed arms (EXT-001 §10 C2).
//!
//! On one REX cell, `apr serve` and each competitor arm (llama.cpp — the existing
//! oracle —, Ollama, mistral.rs) serve the blessed model and are measured by ONE
//! client (`evidence/ext-001/EXT-28/c2_cell.sh`): TTFT, ITL, e2e, load time and peak
//! RSS, at m=1, resident, text-only. One [`CellReceipt`] per (tag, cell) records
//! every arm as `measured`, `not_run` or `refused`, each measured arm carrying its
//! EXT-26 comparator block.
//!
//! S-14: an arm that is not like-for-like with apr — another quant type, a vision
//! path in the served file, m>1, other CPUs/threads/prompt/length — is refused, never
//! normalised. [`classify`] refuses it; [`check_cell`] turns a receipt RED that
//! records such an arm as measured.
//!
//! The cell becomes one EXT-19 ledger row ([`ledger_row`]); the apr/llama.cpp ratio
//! is computed only inside that ledger's ratchet (T28), which is FALSIFY-EXT-022.

use super::comparator::{check_block, is_sha256, ComparatorBlock};
use super::speed_ledger::{ArmSpeed, Measured, NotRun, Outcome, Row, REFERENCE_ARM};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The engine under test.
pub(crate) const APR_ARM: &str = "apr";
/// Every competitor arm a measured cell must account for (EXT-001 §10 C2).
pub(crate) const COMPETITOR_ARMS: [&str; 3] = [REFERENCE_ARM, "ollama", "mistral.rs"];
/// REX §2.1's device cells. Provisional until REX-08 rules; coverage is over these.
pub(crate) const PROVISIONAL_CELLS: [&str; 6] = [
    "intel-wgpu",
    "intel-cpu",
    "lambda-cpu",
    "gx10-cuda",
    "mini-cpu",
    "mini-metal",
];

/// Measured iterations per arm, pre-registered before the records it judges were
/// taken (cop ruling on FALSIFY-EXT-022, 2026-09-26).
pub(crate) const PREREGISTERED_ITERATIONS: u32 = 5;
/// The pre-registered ledger statistic: the best (highest) decode rate of the
/// interleaved iterations. Host contention only ever slows a run, so the best
/// iteration is the one least disturbed by it.
pub(crate) const PREREGISTERED_STATISTIC: &str = "best_of_n_decode";
/// Quiet-host amendment (cop ruling, 2026-09-26): the reserved cpuset's busy
/// share, sampled over 1 s with every arm idle just before each measured
/// iteration, may not exceed this. Host-wide load1 cannot gate a reserved
/// cpuset (the rest of the host stays busy), so it is recorded, not gated.
pub(crate) const PREREGISTERED_MAX_CPUSET_BUSY_PCT: f64 = 10.0;

/// What every arm on a cell must share with apr's run (S-14).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Conditions {
    pub cpus: String,
    pub threads: u32,
    /// Concurrent requests: C2 is m=1.
    pub concurrency: u32,
    pub iterations: u32,
    /// How `decode_tok_s` summarises the iterations ([`PREREGISTERED_STATISTIC`]).
    pub statistic: String,
    pub max_tokens: u32,
    pub prompt_sha256: String,
    /// The systemd unit (in the reserved slice) the whole record ran under.
    pub isolation_unit: String,
    /// `Cpus_allowed_list` read from the harness's own /proc status: proof the
    /// cpuset engaged, not a restatement of `cpus`.
    pub cpus_allowed_list: String,
}

/// TTFT, ITL and e2e are medians over the measured iterations; `decode_tok_s` is
/// the pre-registered statistic over `decode_tok_s_iters`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Timing {
    pub load_ms: f64,
    pub ttft_ms: f64,
    pub itl_ms: f64,
    pub e2e_ms: f64,
    pub decode_tok_s: f64,
    pub peak_rss_kb: u64,
    /// Every measured iteration's decode rate, in run order.
    pub decode_tok_s_iters: Vec<f64>,
    /// The host's 1-minute load average when each iteration started.
    pub loadavg_1m_iters: Vec<f64>,
    /// Host 1-minute load average when the record started (recorded precondition).
    pub load1_at_start: f64,
    /// Busy % of the reserved cpuset over 1 s, arms idle, before each iteration.
    pub cpuset_busy_pct_iters: Vec<f64>,
}

/// The best of the iterations (NaN when there are none).
pub(crate) fn best_of(iters: &[f64]) -> f64 {
    iters.iter().copied().fold(f64::NAN, f64::max)
}

/// One arm's run on one cell, as `c2_cell.sh` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ArmRecord {
    pub arm: String,
    pub version: String,
    /// sha256 of the server executable that ran.
    pub engine_sha256: String,
    /// sha256 of the model file the server loaded.
    pub model_sha256: String,
    /// GGUF `general.file_type` of that file (15 = Q4_K_M).
    pub file_type: u32,
    /// Vision-tower tensors (`v.*`, `mm.*`) in that file.
    pub vision_tensors: u32,
    pub served_model: String,
    #[serde(default)]
    pub ollama_manifest_digest: Option<String>,
    pub conditions: Conditions,
    pub timing: Timing,
    pub comparator: ComparatorBlock,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ArmOutcome {
    Measured(ArmRecord),
    NotRun {
        reason: String,
    },
    /// Not comparable (S-14) or not loadable. A run that happened is kept as a
    /// report-only record and never enters the ledger.
    Refused {
        reason: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        record: Option<ArmRecord>,
    },
}

/// One (tag, cell): every arm's outcome, or why the cell was not run at all.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CellReceipt {
    pub tag: String,
    pub cell: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub not_run: Option<String>,
    #[serde(default)]
    pub arms: BTreeMap<String, ArmOutcome>,
}

/// S-14: why `other` is not like-for-like with apr's run `apr` (empty: it is).
pub(crate) fn unlike(apr: &ArmRecord, other: &ArmRecord) -> Vec<String> {
    let mut why = Vec::new();
    if other.conditions.concurrency != 1 {
        why.push(format!("m={} (C2 is m=1)", other.conditions.concurrency));
    }
    if other.conditions != apr.conditions {
        why.push(format!(
            "conditions {:?} differ from apr's {:?}",
            other.conditions, apr.conditions
        ));
    }
    if other.file_type != apr.file_type {
        why.push(format!(
            "quant type {} differs from apr's {}",
            other.file_type, apr.file_type
        ));
    }
    if other.vision_tensors > 0 {
        why.push(format!(
            "the served file carries a vision path ({} vision tensors)",
            other.vision_tensors
        ));
    }
    if other.arm == REFERENCE_ARM && other.model_sha256 != apr.model_sha256 {
        why.push("the reference arm must serve apr's exact file".into());
    }
    why
}

/// Measured when like-for-like with apr, else refused with the run kept.
pub(crate) fn classify(apr: &ArmRecord, other: ArmRecord) -> ArmOutcome {
    let why = unlike(apr, &other);
    if why.is_empty() {
        ArmOutcome::Measured(other)
    } else {
        ArmOutcome::Refused {
            reason: format!("S-14 not like-for-like: {}", why.join("; ")),
            record: Some(other),
        }
    }
}

fn check_record(key: &str, r: &ArmRecord, findings: &mut Vec<String>) {
    if r.arm != key {
        findings.push(format!("arm `{key}` holds a record for `{}`", r.arm));
    }
    for (what, sha) in [
        ("engine_sha256", &r.engine_sha256),
        ("model_sha256", &r.model_sha256),
        ("prompt_sha256", &r.conditions.prompt_sha256),
    ] {
        if !is_sha256(sha) {
            findings.push(format!("arm `{key}`: `{what}` is not a sha256"));
        }
    }
    let t = &r.timing;
    for (what, x) in [
        ("load_ms", t.load_ms),
        ("ttft_ms", t.ttft_ms),
        ("itl_ms", t.itl_ms),
        ("e2e_ms", t.e2e_ms),
        ("decode_tok_s", t.decode_tok_s),
    ] {
        if !(x.is_finite() && x > 0.0) {
            findings.push(format!("arm `{key}`: {what} {x} is not a measurement"));
        }
    }
    if t.peak_rss_kb == 0 {
        findings.push(format!("arm `{key}`: no peak RSS"));
    }
    let c = &r.conditions;
    if c.iterations != PREREGISTERED_ITERATIONS || c.statistic != PREREGISTERED_STATISTIC {
        findings.push(format!(
            "arm `{key}`: ran {} iterations under `{}`; pre-registered is {PREREGISTERED_ITERATIONS} under `{PREREGISTERED_STATISTIC}`",
            c.iterations, c.statistic
        ));
    }
    let n = c.iterations as usize;
    if t.decode_tok_s_iters.len() != n || t.loadavg_1m_iters.len() != n {
        findings.push(format!(
            "arm `{key}`: {} decode samples and {} loadavg samples for {n} iterations",
            t.decode_tok_s_iters.len(),
            t.loadavg_1m_iters.len()
        ));
    }
    if t.loadavg_1m_iters
        .iter()
        .any(|x| !(x.is_finite() && *x >= 0.0))
    {
        findings.push(format!(
            "arm `{key}`: a loadavg sample is not a measurement"
        ));
    }
    // Quiet-host amendment: the record ran in a reserved cpuset, proven from the
    // harness's own affinity, and that cpuset was idle before every iteration.
    if c.isolation_unit.trim().is_empty() {
        findings.push(format!("arm `{key}`: no isolation unit (quiet-host rule)"));
    }
    if c.cpus_allowed_list != c.cpus {
        findings.push(format!(
            "arm `{key}`: ran with Cpus_allowed_list `{}`, not the reserved cpuset `{}`",
            c.cpus_allowed_list, c.cpus
        ));
    }
    if !(t.load1_at_start.is_finite() && t.load1_at_start >= 0.0) {
        findings.push(format!("arm `{key}`: load1 at start is not a measurement"));
    }
    if t.cpuset_busy_pct_iters.len() != n {
        findings.push(format!(
            "arm `{key}`: {} cpuset busy samples for {n} iterations",
            t.cpuset_busy_pct_iters.len()
        ));
    }
    if let Some(b) = t
        .cpuset_busy_pct_iters
        .iter()
        .find(|b| !(b.is_finite() && (0.0..=PREREGISTERED_MAX_CPUSET_BUSY_PCT).contains(*b)))
    {
        findings.push(format!(
            "arm `{key}`: the reserved cpuset was {b}% busy before an iteration (max {PREREGISTERED_MAX_CPUSET_BUSY_PCT}%)"
        ));
    }
    // The statistic is recomputed from the samples, never taken on the record's word.
    if best_of(&t.decode_tok_s_iters) != t.decode_tok_s {
        findings.push(format!(
            "arm `{key}`: decode_tok_s {} is not the best of its iterations {:?}",
            t.decode_tok_s, t.decode_tok_s_iters
        ));
    }
    match serde_json::to_value(&r.comparator) {
        Ok(v) => {
            if let Err(e) = check_block(&v) {
                findings.push(format!("arm `{key}`: {e}"));
            }
        }
        Err(e) => findings.push(format!("arm `{key}`: {e}")),
    }
}

/// Every finding that makes `c` unusable as C2 evidence (empty: GREEN).
pub(crate) fn check_cell(c: &CellReceipt) -> Vec<String> {
    let mut findings = Vec::new();
    if c.tag.trim().is_empty() || c.cell.trim().is_empty() {
        findings.push("empty `tag` or `cell`".into());
    }
    if let Some(reason) = &c.not_run {
        if reason.trim().is_empty() {
            findings.push("`not_run` needs a reason".into());
        }
        if !c.arms.is_empty() {
            findings.push("a `not_run` cell records no arms".into());
        }
        return findings;
    }
    let apr = match c.arms.get(APR_ARM) {
        Some(ArmOutcome::Measured(r)) => r,
        _ => {
            findings.push("a measured cell needs a measured `apr` arm".into());
            return findings;
        }
    };
    for name in c.arms.keys() {
        if name != APR_ARM && !COMPETITOR_ARMS.contains(&name.as_str()) {
            findings.push(format!("unknown arm `{name}`"));
        }
    }
    for name in COMPETITOR_ARMS {
        if !c.arms.contains_key(name) {
            findings.push(format!(
                "arm `{name}` has no outcome (measured, not_run or refused)"
            ));
        }
    }
    for (name, outcome) in &c.arms {
        match outcome {
            ArmOutcome::Measured(r) => {
                check_record(name, r, &mut findings);
                let why = unlike(apr, r);
                if name != APR_ARM && !why.is_empty() {
                    findings.push(format!(
                        "arm `{name}` recorded as measured but {} (S-14: refuse it)",
                        why.join("; ")
                    ));
                }
            }
            ArmOutcome::NotRun { reason } | ArmOutcome::Refused { reason, .. }
                if reason.trim().is_empty() =>
            {
                findings.push(format!("arm `{name}` needs a reason"));
            }
            ArmOutcome::NotRun { .. } | ArmOutcome::Refused { .. } => {}
        }
    }
    if apr.conditions.concurrency != 1 {
        findings.push(format!("apr ran at m={}", apr.conditions.concurrency));
    }
    if apr.vision_tensors > 0 {
        findings.push("apr's served file carries a vision path".into());
    }
    findings
}

/// The EXT-19 ledger row for a GREEN cell receipt whose bytes hash to
/// `receipt_sha256`. A cell without a measured reference arm has no ratchet input
/// and becomes `not_run`, naming why.
pub(crate) fn ledger_row(c: &CellReceipt, receipt_sha256: &str) -> Result<Row, Vec<String>> {
    let findings = check_cell(c);
    if !findings.is_empty() {
        return Err(findings);
    }
    let not_run = |reason: String| Row {
        tag: c.tag.clone(),
        cell: c.cell.clone(),
        outcome: Outcome::NotRun(NotRun { reason }),
    };
    if let Some(reason) = &c.not_run {
        return Ok(not_run(reason.clone()));
    }
    let apr = match c.arms.get(APR_ARM) {
        Some(ArmOutcome::Measured(r)) => r,
        _ => unreachable!("check_cell requires a measured apr arm"),
    };
    match c.arms.get(REFERENCE_ARM) {
        Some(ArmOutcome::Measured(_)) => {}
        Some(ArmOutcome::NotRun { reason } | ArmOutcome::Refused { reason, .. }) => {
            return Ok(not_run(format!("{REFERENCE_ARM} not measured: {reason}")));
        }
        None => unreachable!("check_cell requires every competitor arm"),
    }
    let arms = COMPETITOR_ARMS
        .iter()
        .filter_map(|name| match c.arms.get(*name) {
            Some(ArmOutcome::Measured(r)) => Some(ArmSpeed {
                arm: r.arm.clone(),
                decode_tok_s: r.timing.decode_tok_s,
                comparator: r.comparator.clone(),
            }),
            _ => None,
        })
        .collect();
    Ok(Row {
        tag: c.tag.clone(),
        cell: c.cell.clone(),
        outcome: Outcome::Measured(Measured {
            apr_decode_tok_s: apr.timing.decode_tok_s,
            receipt_sha256: receipt_sha256.to_string(),
            arms,
        }),
    })
}

#[cfg(test)]
#[path = "speed_arms_tests.rs"]
mod tests;
