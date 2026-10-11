//! The V2 serve receipt (APR-071 Train B, #4953).
//!
//! > `apr serve` keeps the model loaded across requests, and its receipt
//! > reports `load_ms` and TTFT beside pp512 and tg128.
//!
//! Built from ONE [`BenchmarkReport`] whose server this producer started
//! itself, so `load_ms` (spawn → first healthy answer) and every request
//! timing come from the same process. The previous producer was a Python
//! OpenAI client that recorded none of the four numbers.
//!
//! The builder REFUSES rather than reports when the report cannot carry the
//! claim. Every refusal names the missing measurement:
//!
//! - no `load_ms` — the server was not started by this producer, so its load
//!   was never observed;
//! - the server process was not alive at the end of the last run, or its own
//!   `/health` `uptime_sec` (counted from the moment it finished loading) does
//!   not cover the span since ready — a restart, by a wrapper or a supervisor,
//!   would hide a per-request load;
//! - a load, uptime, span, TTFT or latency that is not a finite number;
//! - a failed request, zero requests, or no streamed TTFT;
//! - a request whose TTFT is not below `load_ms` — a request that waited as
//!   long as a whole load cannot show that the model stayed loaded. This is a
//!   BOUND, not a proof: a warm in-process reload can be faster than the cold
//!   load. The process-continuity check above is the proof against restarts;
//!   V1 compares this TTFT with llama.cpp's on the same host;
//! - a workload that is not pp512/tg128: the median prompt is outside
//!   512 ± [`PROMPT_TOLERANCE`], or a completion is not exactly 128 tokens;
//! - no `compute_mode` — `/health` did not report a device after the runs.
//!   The Qwen3.5 hybrid falls back to CPU when it refuses its GPU path at the
//!   first request, so neither the build nor the host label says where the
//!   numbers came from; its `/health` reports the live backend. Other model
//!   paths report the device of the resident model. A `cpu` answer is
//!   recorded as it is, never refused and never relabelled.

use super::benchmark::BenchmarkReport;
use super::loadtest::RequestDetail;
use serde::{Deserialize, Serialize};

/// Receipt schema id. Bump it on any field change.
pub const SCHEMA: &str = "apr-serve-receipt-v1";
/// pp512: the prompt length the receipt is named for.
pub const PROMPT_TOKENS: u32 = 512;
/// tg128: the completion length the receipt is named for.
pub const COMPLETION_TOKENS: u32 = 128;
/// Allowed relative distance of the MEDIAN prompt from 512. A corpus is cut
/// for one tokenizer and served under another, so exact equality would refuse
/// every honest run. Every completion must still be exactly 128 tokens.
pub const PROMPT_TOLERANCE: f64 = 0.05;
/// Clock slack between the server's uptime and this producer's span, seconds.
pub const UPTIME_SLACK_S: f64 = 1.0;

/// Identity of what was measured. The caller supplies it; nothing here guesses.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReceiptMeta {
    /// `apr` version (`CARGO_PKG_VERSION`).
    pub apr_version: String,
    /// Commit the `apr` binary was built from.
    pub apr_commit: String,
    /// sha256 of the model file the server loaded.
    pub model_sha256: String,
    /// Host label (a declared label, not a measured hostname).
    pub host: String,
    /// The workload source, e.g. the corpus path.
    pub workload: String,
}

/// p50 / p95 / max of one per-request quantity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Spread {
    /// Median.
    pub p50: f64,
    /// 95th percentile (nearest rank).
    pub p95: f64,
    /// Maximum.
    pub max: f64,
}

/// The receipt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServeReceipt {
    /// [`SCHEMA`].
    pub schema: String,
    /// What was measured.
    #[serde(flatten)]
    pub meta: ReceiptMeta,
    /// Server spawn → first healthy answer, in ms.
    pub load_ms: f64,
    /// The server's own uptime after the last run, s; covers the measured span.
    pub server_uptime_s: f64,
    /// The device the server's `/health` reported after the last run (`gpu`
    /// or `cpu`); live for the Qwen3.5 hybrid, the resident model otherwise.
    pub compute_mode: String,
    /// Measured requests, all served by the one process that loaded once.
    pub n: usize,
    /// How every summary below is formed.
    pub method: String,
    /// Time to first token per request, ms.
    pub ttft_ms: Spread,
    /// Prefill rate: prompt tokens ÷ TTFT, tok/s, median over requests.
    pub pp512_tok_s: f64,
    /// Decode rate: (completion tokens − 1) ÷ (latency − TTFT), tok/s, median.
    pub tg128_tok_s: f64,
    /// Median prompt tokens as the server counted them.
    pub prompt_tokens_p50: f64,
    /// TTFT max ÷ load_ms. Below 1 on every receipt; the one-load evidence.
    pub max_ttft_over_load: f64,
}

/// Build the receipt, or name the measurement the report lacks.
///
/// # Errors
///
/// A human-readable refusal; see the module docs for the list.
pub fn build(report: &BenchmarkReport, meta: ReceiptMeta) -> Result<ServeReceipt, String> {
    let (load_ms, uptime) = one_process(report)?;
    let compute_mode = report.server_compute_mode_after.clone().ok_or(
        "no compute_mode: /health did not report a device after the runs, so a GPU path \
         that fell back to CPU would read as GPU",
    )?;
    let reqs = measured_requests(report)?;
    let prompt_p50 = workload_prompt_p50(&reqs)?;
    let ttft = spread(reqs.iter().map(|r| r.ttft_ms).collect());
    if ttft.max >= load_ms {
        return Err(format!(
            "a request waited {} ms for its first token, not below load_ms {load_ms}: this run \
             cannot show that the model stayed loaded",
            ttft.max
        ));
    }
    let pp = median(
        reqs.iter()
            .map(|r| f64::from(r.prompt_tokens) * 1000.0 / r.ttft_ms)
            .collect(),
    );
    let tg = median(
        reqs.iter()
            .map(|r| f64::from(r.completion_tokens - 1) * 1000.0 / (r.latency_ms - r.ttft_ms))
            .collect(),
    );
    Ok(ServeReceipt {
        schema: SCHEMA.to_string(),
        meta,
        load_ms,
        server_uptime_s: uptime,
        compute_mode,
        n: reqs.len(),
        method: "per-request values; p50 = median, p95 = nearest rank, over all measured runs; \
                 warmup excluded"
            .to_string(),
        max_ttft_over_load: ttft.max / load_ms,
        ttft_ms: ttft,
        pp512_tok_s: pp,
        tg128_tok_s: tg,
        prompt_tokens_p50: prompt_p50,
    })
}

/// The load was observed, and ONE server process lived through every run:
/// returns `(load_ms, server uptime s)`.
fn one_process(report: &BenchmarkReport) -> Result<(f64, f64), String> {
    let load_ms = report.load_ms.ok_or(
        "no load_ms: the server was not started by this producer (--start), so its load was \
         never observed",
    )?;
    if !load_ms.is_finite() || load_ms <= 0.0 {
        return Err(format!("load_ms is {load_ms}, not a measured load"));
    }
    if report.server_alive_after != Some(true) {
        return Err(
            "the server process was not alive after the last run; a restart would hide \
                    a per-request load"
                .to_string(),
        );
    }
    let (uptime, span) = match (report.server_uptime_after_s, report.ready_to_end_s) {
        (Some(u), Some(s)) => (u, s),
        _ => {
            return Err(
                "no server uptime: /health did not report uptime_sec after the last run, so \
                 a restart between requests cannot be ruled out"
                    .to_string(),
            )
        }
    };
    // A NaN fails every comparison and an infinite uptime covers any span, so
    // both would pass the restart check below without measuring anything.
    if !uptime.is_finite() || !span.is_finite() {
        return Err(format!(
            "server uptime {uptime} s or run span {span} s is not a measured time"
        ));
    }
    if uptime + UPTIME_SLACK_S < span {
        return Err(format!(
            "the server reports {uptime:.1} s of uptime after a {span:.1} s run: it restarted \
             after it was ready, so it did not stay loaded"
        ));
    }
    Ok((load_ms, uptime))
}

/// Every measured request, provided all succeeded, streamed a first token, and
/// completed exactly [`COMPLETION_TOKENS`].
fn measured_requests(report: &BenchmarkReport) -> Result<Vec<&RequestDetail>, String> {
    let failed: u64 = report.runs.iter().map(|r| r.failed).sum();
    if failed > 0 {
        return Err(format!(
            "{failed} failed request(s); a receipt over the survivors is not one"
        ));
    }
    let reqs: Vec<&RequestDetail> = report
        .runs
        .iter()
        .flat_map(|r| &r.request_details)
        .collect();
    if reqs.is_empty() {
        return Err("zero measured requests".to_string());
    }
    if let Some(r) = reqs.iter().find(|r| {
        !r.ttft_ms.is_finite()
            || r.ttft_ms <= 0.0
            || !r.latency_ms.is_finite()
            || r.latency_ms <= r.ttft_ms
    }) {
        return Err(format!(
            "a request has no streamed TTFT (ttft {} ms, latency {} ms); run with --stream",
            r.ttft_ms, r.latency_ms
        ));
    }
    if let Some(r) = reqs
        .iter()
        .find(|r| r.completion_tokens != COMPLETION_TOKENS)
    {
        return Err(format!(
            "a completion has {} tokens, not {COMPLETION_TOKENS}: not tg{COMPLETION_TOKENS} \
             (the corpus needs max_tokens {COMPLETION_TOKENS} and ignore_eos)",
            r.completion_tokens
        ));
    }
    Ok(reqs)
}

/// The median prompt, provided it is pp512 within [`PROMPT_TOLERANCE`].
fn workload_prompt_p50(reqs: &[&RequestDetail]) -> Result<f64, String> {
    let prompt_p50 = median(reqs.iter().map(|r| f64::from(r.prompt_tokens)).collect());
    let want = f64::from(PROMPT_TOKENS);
    if ((prompt_p50 - want) / want).abs() > PROMPT_TOLERANCE {
        return Err(format!(
            "median prompt is {prompt_p50} tokens, outside {PROMPT_TOKENS} ± {:.0}%: not \
             pp{PROMPT_TOKENS}",
            PROMPT_TOLERANCE * 100.0
        ));
    }
    Ok(prompt_p50)
}

fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(f64::total_cmp);
    v
}

fn median(v: Vec<f64>) -> f64 {
    let v = sorted(v);
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn spread(v: Vec<f64>) -> Spread {
    let p50 = median(v.clone());
    let v = sorted(v);
    let n = v.len();
    // Nearest rank: ceil(0.95 n), 1-based.
    let rank = ((0.95 * n as f64).ceil() as usize).clamp(1, n);
    Spread {
        p50,
        p95: v[rank - 1],
        max: v[n - 1],
    }
}

#[cfg(test)]
#[path = "serve_receipt_tests.rs"]
mod tests;
