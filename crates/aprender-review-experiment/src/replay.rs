//! PRM-S1: the `review-replay-v1` speed benchmark (PRM-001 v3 §2.4 and §9;
//! contract `review-replay-v1`).
//!
//! A frozen set of real quorum diffs (inputs only, never a lane's output) is
//! replayed through `apr serve` and through llama.cpp `d1d3c3396` on the same
//! GGUF and the same cell. This module builds the set, defines the receipt row
//! and folds rows into the §9 speed block. It does no I/O.
//!
//! - **Set.** Four strata by composed input tokens: `2k` (0, 2048], `8k`
//!   (2048, 8192], `16k` (8192, 16384], `32k` (16384, 32768]. [`PER_STRATUM`]
//!   items each. One item per `repo#PR`, one per diff sha. Every candidate that
//!   hits the sealed test set (diff sha, hunk fingerprint or near-dup cluster)
//!   is dropped. Selection inside a stratum is `sha256(seed ‖ diff_sha)` order,
//!   so it is frozen by `(version, seed, candidates)`. A short stratum is an
//!   error: the set is never padded.
//! - **Rows** (`review-replay-receipt-v1`), one per `(engine, item)`.
//! - **Summary.** Refuses a mixed set, a GGUF or cell mismatch, unequal item
//!   coverage between engines, duplicate rows, any not-run row, or a missing
//!   voter latency sample. Every ratio is `apr / llama_cpp`: TTFT wants ≤ 2.0,
//!   prefill and decode want ≥ 1.0. A server that reports no timing yields
//!   `null`, never a number.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::contamination::Index;
use crate::corpus::sha256_hex;
use crate::receipt::Verdict;
use crate::stats::{percentile, percentile_sorted, SplitMix64};

pub const SCHEMA: &str = "review-replay-v1";
pub const ROW_SCHEMA: &str = "review-replay-receipt-v1";
pub const PER_STRATUM: usize = 50;
/// Bootstrap resamples for the p95 interval.
pub const BOOTSTRAP: usize = 2000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Stratum {
    #[serde(rename = "2k")]
    K2,
    #[serde(rename = "8k")]
    K8,
    #[serde(rename = "16k")]
    K16,
    #[serde(rename = "32k")]
    K32,
}

impl Stratum {
    pub const ALL: [Self; 4] = [Self::K2, Self::K8, Self::K16, Self::K32];

    /// Inclusive upper bound in input tokens.
    #[must_use]
    pub fn upper(self) -> u64 {
        match self {
            Self::K2 => 2048,
            Self::K8 => 8192,
            Self::K16 => 16384,
            Self::K32 => 32768,
        }
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::K2 => "2k",
            Self::K8 => "8k",
            Self::K16 => "16k",
            Self::K32 => "32k",
        }
    }

    /// The stratum of an input of `tokens` tokens; `None` for 0 or > 32768.
    #[must_use]
    pub fn of(tokens: u64) -> Option<Self> {
        if tokens == 0 {
            return None;
        }
        Self::ALL.into_iter().find(|s| tokens <= s.upper())
    }
}

/// A candidate diff, before sealing and stratification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// `owner/repo#N`.
    pub group: String,
    pub diff_sha256: String,
    /// Composed prompt tokens, counted by the replay GGUF's tokenizer.
    pub input_tokens: u64,
    pub diff: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Item {
    pub group: String,
    pub diff_sha256: String,
    pub input_tokens: u64,
    pub stratum: Stratum,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Set {
    pub version: String,
    pub seed: u64,
    pub per_stratum: usize,
    pub items: Vec<Item>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// Nothing is sealed: the exclusion would prove nothing.
    EmptySealedIndex,
    /// Strata that could not be filled.
    Short(Vec<Short>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Short {
    pub stratum: Stratum,
    pub have: usize,
    pub need: usize,
}

fn rank_key(seed: u64, diff_sha: &str) -> String {
    sha256_hex(format!("{seed}\u{0}{diff_sha}").as_bytes())
}

/// Build the frozen set (see module docs).
///
/// # Errors
/// [`BuildError::EmptySealedIndex`] when `sealed` is empty, and
/// [`BuildError::Short`] listing every stratum with fewer than `per_stratum`
/// eligible items.
pub fn build(
    version: &str,
    seed: u64,
    per_stratum: usize,
    cands: &[Candidate],
    sealed: &Index,
) -> Result<Set, BuildError> {
    if sealed.is_empty() {
        return Err(BuildError::EmptySealedIndex);
    }
    // Deterministic order before the one-per-group / one-per-sha dedup, so
    // input order never decides which duplicate survives.
    let mut eligible: Vec<(String, Item)> = cands
        .iter()
        .filter(|c| {
            sealed
                .scan(&format!("{}\n{}", c.diff_sha256, c.diff), &c.group)
                .is_empty()
        })
        .filter_map(|c| {
            Stratum::of(c.input_tokens).map(|stratum| {
                (
                    rank_key(seed, &c.diff_sha256),
                    Item {
                        group: c.group.clone(),
                        diff_sha256: c.diff_sha256.clone(),
                        input_tokens: c.input_tokens,
                        stratum,
                    },
                )
            })
        })
        .collect();
    eligible.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.group.cmp(&b.1.group)));
    let mut groups = BTreeSet::new();
    let mut shas = BTreeSet::new();
    let mut by: BTreeMap<Stratum, Vec<Item>> = BTreeMap::new();
    for (_, it) in eligible {
        if groups.contains(&it.group) || shas.contains(&it.diff_sha256) {
            continue;
        }
        let bucket = by.entry(it.stratum).or_default();
        if bucket.len() < per_stratum {
            groups.insert(it.group.clone());
            shas.insert(it.diff_sha256.clone());
            bucket.push(it);
        }
    }
    let short: Vec<Short> = Stratum::ALL
        .into_iter()
        .filter_map(|s| {
            let have = by.get(&s).map_or(0, Vec::len);
            (have < per_stratum).then_some(Short {
                stratum: s,
                have,
                need: per_stratum,
            })
        })
        .collect();
    if !short.is_empty() {
        return Err(BuildError::Short(short));
    }
    Ok(Set {
        version: version.to_string(),
        seed,
        per_stratum,
        items: Stratum::ALL
            .into_iter()
            .flat_map(|s| by.remove(&s).unwrap_or_default())
            .collect(),
    })
}

impl Set {
    /// JSONL: a header line, then one item per line.
    #[must_use]
    pub fn render(&self) -> String {
        let head: serde_json::Map<String, serde_json::Value> = [
            ("schema", serde_json::Value::from(SCHEMA)),
            ("version", self.version.as_str().into()),
            ("seed", self.seed.into()),
            ("per_stratum", self.per_stratum.into()),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let mut out = serde_json::Value::Object(head).to_string();
        out.push('\n');
        for it in &self.items {
            out.push_str(&serde_json::to_string(it).unwrap_or_default());
            out.push('\n');
        }
        out
    }

    /// sha256 of [`Set::render`]; every row carries it.
    #[must_use]
    pub fn sha(&self) -> String {
        sha256_hex(self.render().as_bytes())
    }

    /// Parse a rendered set.
    ///
    /// # Errors
    /// A missing or foreign header, or a malformed item line.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut lines = text.lines();
        let head: serde_json::Value = lines
            .next()
            .and_then(|l| serde_json::from_str(l).ok())
            .ok_or("no header")?;
        if head["schema"] != SCHEMA {
            return Err(format!("schema is {}, want {SCHEMA}", head["schema"]));
        }
        let version = head["version"].as_str().ok_or("no version")?.to_string();
        let seed = head["seed"].as_u64().ok_or("no seed")?;
        let per_stratum = head["per_stratum"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("no per_stratum")?;
        let items = lines
            .enumerate()
            .map(|(i, l)| serde_json::from_str(l).map_err(|e| format!("item {}: {e}", i + 1)))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            version,
            seed,
            per_stratum,
            items,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Engine {
    Apr,
    LlamaCpp,
}

/// One `review-replay-receipt-v1` row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Row {
    pub schema: String,
    pub replay_version: String,
    pub set_sha: String,
    pub diff_sha256: String,
    pub stratum: Stratum,
    pub engine: Engine,
    pub engine_version: String,
    /// The apr tag under test (the same on the llama.cpp rows of that run).
    pub apr_tag: String,
    pub cell: String,
    pub gguf_sha256: String,
    pub wall_ms: f64,
    /// Server-reported prompt (prefill) time: the TTFT of a non-streamed call.
    pub prompt_ms: Option<f64>,
    pub prompt_tps: Option<f64>,
    pub decode_tps: Option<f64>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub peak_rss_mb: Option<f64>,
    pub verdict: Verdict,
}

/// Engine A over engine B (`apr / llama_cpp`). `ratio` is null when either
/// side is null or B is zero.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ratio {
    pub apr: Option<f64>,
    pub llama_cpp: Option<f64>,
    pub ratio: Option<f64>,
}

impl Ratio {
    fn of(apr: Option<f64>, llama_cpp: Option<f64>) -> Self {
        let ratio = match (apr, llama_cpp) {
            (Some(a), Some(b)) if b > 0.0 => Some(a / b),
            _ => None,
        };
        Self {
            apr,
            llama_cpp,
            ratio,
        }
    }
}

/// The §9 speed block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Speed {
    pub replay_version: String,
    pub set_sha: String,
    pub apr_tag: String,
    pub cell: String,
    pub gguf_sha256: String,
    pub n_items: usize,
    pub p50_s: f64,
    pub p95_s: f64,
    pub p95_ci: [f64; 2],
    pub queue_budget_p95_s: f64,
    pub queue_budget_lane: String,
    pub within_budget: bool,
    pub ttft_ms: Ratio,
    pub prefill_tps: BTreeMap<Stratum, Ratio>,
    pub decode_tps: Ratio,
    pub peak_rss_mb: Ratio,
    pub parse_rate: f64,
    pub verdict_identity_vs_prev_tag: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SummaryError {
    NoRows,
    MixedSet,
    Mismatch(&'static str),
    DuplicateRow { engine: Engine, diff_sha256: String },
    CoverageDiffers,
    NotRun { engine: Engine, diff_sha256: String },
    NoVoterLatency,
    PrevSetDiffers,
    PrevCoverageDiffers,
}

/// Per-engine rows keyed by diff sha.
type ByItem<'a> = BTreeMap<&'a str, &'a Row>;

fn index(rows: &[Row]) -> Result<[ByItem<'_>; 2], SummaryError> {
    let mut out = [BTreeMap::new(), BTreeMap::new()];
    for r in rows {
        if let Verdict::NotRun(_) = r.verdict {
            return Err(SummaryError::NotRun {
                engine: r.engine,
                diff_sha256: r.diff_sha256.clone(),
            });
        }
        let slot = usize::from(r.engine == Engine::LlamaCpp);
        if out[slot].insert(r.diff_sha256.as_str(), r).is_some() {
            return Err(SummaryError::DuplicateRow {
                engine: r.engine,
                diff_sha256: r.diff_sha256.clone(),
            });
        }
    }
    Ok(out)
}

fn median(xs: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    // A single null makes the engine's figure null: never a median of a subset.
    let v: Option<Vec<f64>> = xs.collect();
    v.filter(|v| !v.is_empty()).map(|v| percentile(&v, 0.5))
}

fn p95_ci(wall_s: &[f64], seed: u64) -> [f64; 2] {
    let mut rng = SplitMix64::new(seed);
    let mut stats: Vec<f64> = (0..BOOTSTRAP)
        .map(|_| {
            let s: Vec<f64> = (0..wall_s.len())
                .map(|_| wall_s[rng.below(wall_s.len())])
                .collect();
            percentile(&s, 0.95)
        })
        .collect();
    stats.sort_by(f64::total_cmp);
    [
        percentile_sorted(&stats, 0.025),
        percentile_sorted(&stats, 0.975),
    ]
}

/// Fold one run's rows into the §9 block.
///
/// `voters` is `(lane, total seconds per review)` for every counted voter;
/// the queue budget is the largest lane p95. `prev` is the previous tag's
/// apr rows on the same set, for verdict identity.
///
/// # Errors
/// See [`SummaryError`]; every refusal names what was inconsistent.
pub fn summarize(
    rows: &[Row],
    voters: &[(String, Vec<f64>)],
    prev: Option<&[Row]>,
    seed: u64,
) -> Result<Speed, SummaryError> {
    let first = rows.first().ok_or(SummaryError::NoRows)?;
    for r in rows {
        if r.set_sha != first.set_sha || r.replay_version != first.replay_version {
            return Err(SummaryError::MixedSet);
        }
        if r.gguf_sha256 != first.gguf_sha256 {
            return Err(SummaryError::Mismatch("gguf_sha256"));
        }
        if r.cell != first.cell {
            return Err(SummaryError::Mismatch("cell"));
        }
        if r.apr_tag != first.apr_tag {
            return Err(SummaryError::Mismatch("apr_tag"));
        }
    }
    let [apr, llama] = index(rows)?;
    if apr.is_empty() || !apr.keys().eq(llama.keys()) {
        return Err(SummaryError::CoverageDiffers);
    }
    if voters.is_empty() || voters.iter().any(|(_, v)| v.is_empty()) {
        return Err(SummaryError::NoVoterLatency);
    }
    let (queue_budget_lane, queue_budget_p95_s) = voters
        .iter()
        .map(|(lane, v)| (lane.clone(), percentile(v, 0.95)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or(SummaryError::NoVoterLatency)?;

    let wall_s: Vec<f64> = apr.values().map(|r| r.wall_ms / 1000.0).collect();
    let p95_s = percentile(&wall_s, 0.95);
    let pair = |f: fn(&Row) -> Option<f64>, keep: &dyn Fn(&Row) -> bool| {
        Ratio::of(
            median(apr.values().filter(|r| keep(r)).map(|r| f(r))),
            median(llama.values().filter(|r| keep(r)).map(|r| f(r))),
        )
    };
    let all = |_: &Row| true;
    let prefill_tps = Stratum::ALL
        .into_iter()
        .filter(|s| apr.values().any(|r| r.stratum == *s))
        .map(|s| (s, pair(|r| r.prompt_tps, &|r: &Row| r.stratum == s)))
        .collect();
    let peak = |m: &ByItem<'_>| -> Option<f64> {
        let v: Option<Vec<f64>> = m.values().map(|r| r.peak_rss_mb).collect();
        v.and_then(|v| v.into_iter().reduce(f64::max))
    };
    let parsed = apr
        .values()
        .filter(|r| matches!(r.verdict, Verdict::Pass | Verdict::Fail))
        .count();

    let verdict_identity_vs_prev_tag = match prev {
        None => None,
        Some(prev) => {
            let mut p = BTreeMap::new();
            for r in prev.iter().filter(|r| r.engine == Engine::Apr) {
                if r.set_sha != first.set_sha {
                    return Err(SummaryError::PrevSetDiffers);
                }
                p.insert(r.diff_sha256.as_str(), r.verdict);
            }
            if !p.keys().eq(apr.keys()) {
                return Err(SummaryError::PrevCoverageDiffers);
            }
            let same = apr
                .values()
                .filter(|r| p[r.diff_sha256.as_str()] == r.verdict)
                .count();
            Some(same as f64 / apr.len() as f64)
        }
    };

    Ok(Speed {
        replay_version: first.replay_version.clone(),
        set_sha: first.set_sha.clone(),
        apr_tag: first.apr_tag.clone(),
        cell: first.cell.clone(),
        gguf_sha256: first.gguf_sha256.clone(),
        n_items: apr.len(),
        p50_s: percentile(&wall_s, 0.5),
        p95_s,
        p95_ci: p95_ci(&wall_s, seed),
        queue_budget_p95_s,
        queue_budget_lane,
        within_budget: p95_s <= queue_budget_p95_s,
        ttft_ms: pair(|r| r.prompt_ms, &all),
        prefill_tps,
        decode_tps: pair(|r| r.decode_tps, &all),
        peak_rss_mb: Ratio::of(peak(&apr), peak(&llama)),
        parse_rate: parsed as f64 / apr.len() as f64,
        verdict_identity_vs_prev_tag,
    })
}

#[cfg(test)]
#[path = "replay_tests.rs"]
mod tests;
