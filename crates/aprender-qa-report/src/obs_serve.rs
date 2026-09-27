//! OBS-13 serve tier (APR-OBS-001 v1.3 §2.4, §2.8, §10 E5/E6; contract `apr-perf-ledger-v1`).
//!
//! The deterministic half of the serve tier, for the recorder (infra-8d, §2.7) to call:
//!
//! - which bands run tonight: three gated bands every night, then one rotating band
//!   (`c = 4` on even nights, `c = 8` on odd nights), in the §2.8 priority order;
//! - the ladder (PP-24): a band above either server's admitted slots is `NA`, with the
//!   budget recorded, never run at a lower `c`;
//! - the `llama-server` launch for a band (`-np c`, `-c c·n_ctx_slot`, `n_ctx_slot ≥ 640`)
//!   and the W5 long-context prompt derived from W1;
//! - PP-28: a band with any retained sample whose `completion_tokens != 128` is fatal;
//! - E6 `η(c)` / `ς(c)`, E5 `ln ω` over two matched rows, and the §2.8 lease length;
//! - the skip row a band writes when it has fewer than three valid blocks.
//!
//! Nothing here runs a server or opens a ledger (R-10): the recorder measures, this computes.

use crate::obs_ledger::BLOCKS;
use chrono::{Datelike, NaiveDate};
use serde_json::{Map, Value};

/// Serve-run window, seconds (§2.4, PP-LLAMA §5.1).
pub const WINDOW_S: u64 = 60;
/// Warmup before the window, seconds.
pub const WARMUP_S: u64 = 15;
/// Cooldown after the window, seconds.
pub const COOLDOWN_S: u64 = 10;
/// One serve run: warmup + window + cooldown = 85 s.
pub const RUN_S: u64 = WARMUP_S + WINDOW_S + COOLDOWN_S;
/// Minimum per-slot context for `llama-server` (§2.4).
pub const MIN_CTX_SLOT: u32 = 640;
/// Greedy generation length every retained sample must reach (§2.4, PP-28).
pub const MAX_TOKENS: u32 = 128;
/// W1 prompt length, nominal and tolerance (§2.4).
pub const W1_PROMPT_TOKENS: u32 = 512;
/// W1 prompt tolerance, ± tokens.
pub const W1_PROMPT_TOL: u32 = 8;
/// W5 long-context prompt length (§2.4).
pub const W5_PROMPT_TOKENS: usize = 30_720;
/// Lease length over the measured p95 block wall (§2.8).
pub const LEASE_FACTOR: f64 = 1.25;

/// Prompt workload (§2.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workload {
    /// `prompts-w1.jsonl`, 512 ± 8 tokens per prompt.
    W1,
    /// W1 concatenated in file order, cycled, truncated to [`W5_PROMPT_TOKENS`].
    W5,
}

impl Workload {
    /// The `workload_id` a ledger row carries.
    #[must_use]
    pub fn id(self) -> &'static str {
        match self {
            Self::W1 => "W1",
            Self::W5 => "W5",
        }
    }

    /// Longest prompt this workload may send, in tokens.
    #[must_use]
    pub fn max_prompt_tokens(self) -> u32 {
        match self {
            Self::W1 => W1_PROMPT_TOKENS + W1_PROMPT_TOL,
            Self::W5 => W5_PROMPT_TOKENS as u32,
        }
    }
}

/// One serve band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Band {
    /// Concurrent clients (`llama-server -np c`).
    pub c: u32,
    /// Context the band is named by.
    pub ctx: u32,
    /// Prompt workload.
    pub workload: Workload,
    /// Gated every night (true) or rotating, report-only (false).
    pub gated: bool,
}

impl Band {
    /// The `band` object a ledger row carries (same shape as the engine tier's).
    #[must_use]
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("c".into(), Value::from(self.c));
        m.insert("ctx".into(), Value::from(self.ctx));
        m.insert("workload_id".into(), Value::from(self.workload.id()));
        Value::Object(m)
    }
}

/// The three gated bands, in priority order (§2.4).
pub const GATED: [Band; 3] = [
    Band {
        c: 1,
        ctx: 4096,
        workload: Workload::W1,
        gated: true,
    },
    Band {
        c: 16,
        ctx: 4096,
        workload: Workload::W1,
        gated: true,
    },
    Band {
        c: 1,
        ctx: 32_768,
        workload: Workload::W5,
        gated: true,
    },
];

/// The rotating band for `night`: `c = 4` on even nights, `c = 8` on odd nights.
///
/// Night parity is `night.num_days_from_ce() % 2`, so it alternates across month and year
/// ends (day-of-month parity would repeat `c = 8` on the 31st and the 1st).
#[must_use]
pub fn rotating(night: NaiveDate) -> Band {
    let c = if night.num_days_from_ce() % 2 == 0 {
        4
    } else {
        8
    };
    Band {
        c,
        ctx: 4096,
        workload: Workload::W1,
        gated: false,
    }
}

/// Tonight's serve bands in §2.8 priority order: gated first, rotating last.
/// (The engine tier runs between them; it is not a serve band.)
#[must_use]
pub fn bands_for_night(night: NaiveDate) -> Vec<Band> {
    let mut v = GATED.to_vec();
    v.push(rotating(night));
    v
}

/// Why a band does not run tonight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BandRefusal {
    /// PP-24: `c` exceeds a server's admitted slots. Recorded as `NA` with the budget.
    AboveCeiling {
        /// Band concurrency.
        c: u32,
        /// Slots apr serve reported.
        slots_apr: u32,
        /// Slots llama-server reported.
        slots_llama: u32,
    },
    /// `n_ctx_slot` below [`MIN_CTX_SLOT`] or below the workload's prompt + generation.
    SlotTooSmall {
        /// Per-slot context offered.
        n_ctx_slot: u32,
        /// Per-slot context needed.
        needed: u32,
    },
}

/// PP-24 ladder: a band runs only when `c ≤ min(slots_apr, slots_llama)`.
pub fn ladder(band: &Band, slots_apr: u32, slots_llama: u32) -> Result<(), BandRefusal> {
    if band.c > slots_apr.min(slots_llama) {
        return Err(BandRefusal::AboveCeiling {
            c: band.c,
            slots_apr,
            slots_llama,
        });
    }
    Ok(())
}

/// Per-slot context a band needs: its longest prompt plus [`MAX_TOKENS`], and at least
/// [`MIN_CTX_SLOT`].
///
/// For W1 this is 648, not 640: the §2.4 floor covers the nominal 512 + 128 and not the
/// +8 tolerance, so a 520-token prompt at `n_ctx_slot = 640` would be truncated.
#[must_use]
pub fn needed_ctx_slot(band: &Band) -> u32 {
    (band.workload.max_prompt_tokens() + MAX_TOKENS).max(MIN_CTX_SLOT)
}

/// `llama-server` arguments for `band` (§2.4): `-np c -c (c · n_ctx_slot)`, with the
/// pinned defaults (PP-LLAMA §5.3) left to the caller.
pub fn llama_server_args(band: &Band, n_ctx_slot: u32) -> Result<Vec<String>, BandRefusal> {
    let needed = needed_ctx_slot(band);
    if n_ctx_slot < needed {
        return Err(BandRefusal::SlotTooSmall { n_ctx_slot, needed });
    }
    Ok(vec![
        "-np".into(),
        band.c.to_string(),
        "-c".into(),
        (u64::from(band.c) * u64::from(n_ctx_slot)).to_string(),
    ])
}

/// W5: the W1 prompts' tokens concatenated in file order, cycled, truncated to
/// [`W5_PROMPT_TOKENS`]. `None` if W1 has no tokens (cycling an empty set never ends).
#[must_use]
pub fn w5_prompt(w1: &[Vec<u32>]) -> Option<Vec<u32>> {
    let total: usize = w1.iter().map(Vec::len).sum();
    if total == 0 {
        return None;
    }
    Some(
        w1.iter()
            .flatten()
            .copied()
            .cycle()
            .take(W5_PROMPT_TOKENS)
            .collect(),
    )
}

/// PP-28: indices of retained samples whose `completion_tokens` is not [`MAX_TOKENS`].
/// Any index makes the band fatal for the night.
#[must_use]
pub fn short_completions(completion_tokens: &[u32]) -> Vec<usize> {
    completion_tokens
        .iter()
        .enumerate()
        .filter(|(_, n)| **n != MAX_TOKENS)
        .map(|(i, _)| i)
        .collect()
}

/// E6 `η(c) = agg(c) / (c · agg(1))`. `None` on a non-positive or non-finite input.
#[must_use]
pub fn eta(agg_c: f64, agg_1: f64, c: u32) -> Option<f64> {
    if c == 0 || !(agg_c.is_finite() && agg_1.is_finite()) || agg_c <= 0.0 || agg_1 <= 0.0 {
        return None;
    }
    Some(agg_c / (f64::from(c) * agg_1))
}

/// E6 `ς(c) = dec_ratio(c) / agg_ratio(c)`. `None` on a non-positive or non-finite input.
#[must_use]
pub fn varsigma(dec_ratio: f64, agg_ratio: f64) -> Option<f64> {
    if !(dec_ratio.is_finite() && agg_ratio.is_finite()) || dec_ratio <= 0.0 || agg_ratio <= 0.0 {
        return None;
    }
    Some(dec_ratio / agg_ratio)
}

/// Fields the serve and engine rows must share before ω (§10 E5).
pub const OMEGA_IDENTITY: [&str; 5] = [
    "host",
    "backend",
    "binary_sha256",
    "model_sha256",
    "epoch_id",
];

/// Why ω is not defined for a pair of rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OmegaRefusal {
    /// A row is not the tier/band E5 needs (serve decode at c = 1, engine tg128).
    WrongRow(&'static str),
    /// [`OMEGA_IDENTITY`] fields absent or different (named).
    Identity(Vec<&'static str>),
    /// Rows from different nights (`ts` dates differ or are unparsable).
    DifferentNight,
    /// `L_n` absent or non-finite on a row.
    NoStatistic(&'static str),
}

fn night_of(row: &Value) -> Option<&str> {
    row.get("ts")
        .and_then(Value::as_str)
        .and_then(|t| t.get(..10))
}

/// E5 `ln ω_n = L^{serve,dec,c=1}_n − L^{engine,tg128}_n`, over two ledger rows.
///
/// `serve` must be tier `serve` at band `c = 1`, `engine` tier `engine`; both must share
/// [`OMEGA_IDENTITY`] and the same night. The caller picks the decode / tg128 rows; this
/// refuses any pair E5 does not define instead of computing a quotient across identity
/// (the §1 join prohibition).
pub fn ln_omega(serve: &Value, engine: &Value) -> Result<f64, OmegaRefusal> {
    if serve.get("tier").and_then(Value::as_str) != Some("serve") {
        return Err(OmegaRefusal::WrongRow("serve row is not tier serve"));
    }
    if serve.pointer("/band/c").and_then(Value::as_u64) != Some(1) {
        return Err(OmegaRefusal::WrongRow("serve row is not band c = 1"));
    }
    if engine.get("tier").and_then(Value::as_str) != Some("engine") {
        return Err(OmegaRefusal::WrongRow("engine row is not tier engine"));
    }
    let bad: Vec<&'static str> = OMEGA_IDENTITY
        .iter()
        .copied()
        .filter(|f| serve.get(*f).map_or(true, Value::is_null) || serve.get(*f) != engine.get(*f))
        .collect();
    if !bad.is_empty() {
        return Err(OmegaRefusal::Identity(bad));
    }
    match (night_of(serve), night_of(engine)) {
        (Some(a), Some(b)) if a == b && NaiveDate::parse_from_str(a, "%Y-%m-%d").is_ok() => {}
        _ => return Err(OmegaRefusal::DifferentNight),
    }
    let l = |row: &Value, which| {
        row.get("L_n")
            .and_then(Value::as_f64)
            .filter(|x| x.is_finite())
            .ok_or(OmegaRefusal::NoStatistic(which))
    };
    Ok(l(serve, "serve")? - l(engine, "engine")?)
}

/// §2.8 lease length for one block: `1.25 × p95 block wall`, whole seconds, rounded up.
/// `None` until a p95 has been measured (the `[A]` factor never runs on an assumed wall).
#[must_use]
pub fn lease_len_s(p95_block_wall_s: Option<f64>) -> Option<u64> {
    let p = p95_block_wall_s.filter(|p| p.is_finite() && *p > 0.0)?;
    Some((LEASE_FACTOR * p).ceil() as u64)
}

/// Planned wall of one serve ABBA block before any p95 exists: 4 runs × [`RUN_S`].
pub const PLANNED_BLOCK_S: u64 = 4 * RUN_S;

/// The §2.8 skip row for a band with fewer than [`BLOCKS`] valid blocks by window end:
/// reported, never a gap. `None` if the band in fact has enough blocks.
#[must_use]
pub fn skip_row(
    band: &Band,
    reason: &str,
    lease_holder: &str,
    blocks_valid: usize,
) -> Option<Value> {
    if blocks_valid >= BLOCKS {
        return None;
    }
    let mut m = Map::new();
    m.insert("tier".into(), Value::from("serve"));
    m.insert("band".into(), band.to_json());
    m.insert("state".into(), Value::from("skipped"));
    m.insert("reason".into(), Value::from(reason));
    m.insert("lease_holder".into(), Value::from(lease_holder));
    m.insert("blocks_valid".into(), Value::from(blocks_valid));
    Some(Value::Object(m))
}

#[cfg(test)]
#[path = "obs_serve_tests.rs"]
mod tests;
