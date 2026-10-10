//! #4971 V3-d: the perf041 probe's decisions, in Rust.
//!
//! `scripts/perf041_batched_parity_probe.py` decodes one prompt twice alone
//! (`m = 1`, the reference), then as `c` identical concurrent requests for each
//! `c` of the ladder, and writes a witness whose bands the V3 shape check
//! ([`super::v3_shape`]) reads. This module is that probe's judgement, ported
//! (C301): what makes a sample comparable, when the reference is stable, how a
//! band is scored, and the run's verdict. Every function here is pure in what
//! came back over the wire, so it compiles and is tested under default
//! features. The requests themselves are fired behind the `llm` feature.
//!
//! The port adds the margin. Every request asks for `logprobs` with the two
//! best tokens per step, so a band records the reference's top-2 margin at the
//! token where the batch parted from it (`top2_margin_at_divergence`), and the
//! V3 shape check can tell a near-tie flip from a defect.
//!
//! Three verdicts, as in the script: PASS, FAIL (a slot parted from its own
//! batch before `declared_min`, or froze), UNMEASURABLE (the run could not
//! decide). A run in which no band at `c > 1` was measured is UNMEASURABLE,
//! never PASS.
//!
//! This is not a gate. No merge, queue or release job runs it.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::margin::{token_ids, TokenLogprob};
use super::metrics::percentile;
use super::protocol::{matrix_block, ProtocolParams, PERF_MATRIX_SOURCE};
use super::witness::{BatchInvariance, BatchInvarianceWitness};

/// The witness layout this module writes. The script's is 2; 3 adds the margin.
pub const WITNESS_VERSION: u32 = 3;

/// The `probe` field: the script's name, with the port marked.
pub const PROBE: &str = "perf041-rs";

/// What the scheduler logs after each decode step,
/// `[PMAT-044] Batch m=N done` (`crates/aprender-serve/src/api/cuda_batch_scheduler.rs`).
const BATCH_MARK: &str = "Batch m=";

/// Knobs that change which kernels decode, recorded rather than inherited.
/// A key that is unset is recorded as `null`, so the witness says so.
pub const ENV_KEYS: [&str; 9] = [
    "CUDA_BATCH_WINDOW_MS",
    "CUBLAS_GEMM_THRESHOLD",
    "FP8_DECODE",
    "FP8_PREFILL",
    "BATCHED_PREFILL",
    "MULTI_PROMPT_PREFILL",
    "APR_DECODE_GEMM",
    "ITERATION_SCHEDULER",
    "FUSED_GATE_UP",
];

/// The numbers a probe run compares against, from `scripts/perf-matrix.yaml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbePolicy {
    /// `witness.min_agree_tokens`: tokens a batch's slots must agree for.
    pub declared_min: u32,
    /// `protocol.n_predict`: tokens every request generates (`ignore_eos`).
    pub n_predict: u32,
    /// `witness.max_constant_run`: a run of one token this long is a frozen slot.
    pub max_constant_run: u32,
}

/// One request, as the probe saw it come back.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeSample {
    /// One logprob entry per generated token, in order.
    pub logprobs: Vec<TokenLogprob>,
    /// `usage.completion_tokens`, as the server reported it.
    pub completion_tokens: Option<u32>,
    /// The terminal chunk's `finish_reason`.
    pub finish_reason: Option<String>,
    /// S7: this request's decode rate, from [`live_decode_tok_s`].
    pub decode_tok_s: Option<f64>,
}

/// §4.4.3 `decode_tok_s` for one streamed request, as
/// [`RequestSample::decode_tok_s`](super::metrics::RequestSample::decode_tok_s)
/// defines it: `(completion tokens − 1) / (last token arrival − first)`, from
/// the client's arrival times.
///
/// `None` unless the server declared the stream `live` (PP-27): a replayed
/// stream's arrival times time the replay, not the decode, and an undeclared
/// stream is not assumed live. `None` too with fewer than two tokens or
/// arrivals, or no time between them, where the rate is undefined, not zero.
/// The server's own `timings` cannot stand in: the batched path drops them
/// (`cuda_batch_scheduler.rs`), and S7 is a claim about that path.
#[must_use]
pub fn live_decode_tok_s(live: bool, arrivals: &[Duration], completion_tokens: u32) -> Option<f64> {
    if !live || arrivals.len() < 2 || completion_tokens < 2 {
        return None;
    }
    let span = arrivals
        .last()?
        .saturating_sub(*arrivals.first()?)
        .as_secs_f64();
    (span > 0.0).then(|| f64::from(completion_tokens - 1) / span)
}

/// A request that came back, or why it did not.
pub type SampleResult = Result<ProbeSample, String>;

/// The sample when it can be compared, else why not (PP-28): the server must
/// have generated exactly `n_predict` tokens and sent one logprob entry each.
fn accepted(sample: &SampleResult, n_predict: u32) -> Result<&ProbeSample, String> {
    let sample = sample.as_ref().map_err(Clone::clone)?;
    if sample.completion_tokens != Some(n_predict) {
        return Err(format!(
            "completion_tokens={} != n_predict={n_predict} (PP-28)",
            sample
                .completion_tokens
                .map_or_else(|| "none".to_string(), |n| n.to_string())
        ));
    }
    let entries = sample.logprobs.len();
    if u32::try_from(entries).ok() != Some(n_predict) {
        return Err(format!(
            "{entries} logprob entries != n_predict={n_predict}; the request must ask for logprobs"
        ));
    }
    Ok(sample)
}

/// Tokens on which two streams agree from the start.
fn agreement(a: &[TokenLogprob], b: &[TokenLogprob]) -> u32 {
    let ids = token_ids(&[a, b]);
    let agree = ids[0]
        .iter()
        .zip(&ids[1])
        .take_while(|(x, y)| x == y)
        .count();
    u32::try_from(agree).unwrap_or(u32::MAX)
}

/// The `m = 1` reference: two solo decodes of the prompt, which must agree to
/// `declared_min` before a batched difference can be laid at batching's door.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProbeReference {
    /// Both decodes were comparable and agreed to `declared_min`.
    pub stable: bool,
    /// The first decode's `usage.completion_tokens`.
    pub tokens: Option<u32>,
    /// Tokens on which the two decodes agree from the start.
    pub self_divergence_at: Option<u32>,
    /// Why the reference is not stable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Judge the two `m = 1` decodes.
#[must_use]
pub fn check_reference(
    first: &SampleResult,
    second: &SampleResult,
    policy: &ProbePolicy,
) -> ProbeReference {
    let mut reference = ProbeReference {
        tokens: first.as_ref().ok().and_then(|s| s.completion_tokens),
        ..ProbeReference::default()
    };
    let pair = accepted(first, policy.n_predict)
        .map_err(|why| format!("ref#1: {why}"))
        .and_then(|a| {
            accepted(second, policy.n_predict)
                .map(|b| (a, b))
                .map_err(|why| format!("ref#2: {why}"))
        });
    let (a, b) = match pair {
        Ok(pair) => pair,
        Err(reason) => {
            reference.reason = Some(reason);
            return reference;
        }
    };
    let agree = agreement(&a.logprobs, &b.logprobs);
    reference.self_divergence_at = Some(agree);
    if agree < policy.declared_min {
        reference.reason = Some(format!(
            "the two m=1 references diverge at token {agree} < declared_min {}; a batched \
             difference could not be laid at batching's door",
            policy.declared_min
        ));
    } else {
        reference.stable = true;
    }
    reference
}

/// One slot of a band, as recorded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeSlot {
    /// The slot's index in the band.
    pub i: u32,
    /// `usage.completion_tokens`.
    pub completion_tokens: Option<u32>,
    /// The terminal chunk's `finish_reason`.
    pub finish_reason: Option<String>,
    /// Why the slot could not be compared.
    pub refused: Option<String>,
    /// S7: the request's decode rate, `null` when it was not timed.
    #[serde(default)]
    pub decode_tok_s: Option<f64>,
}

impl ProbeSlot {
    fn of(i: usize, sample: &SampleResult, n_predict: u32) -> Self {
        let ok = sample.as_ref().ok();
        Self {
            i: u32::try_from(i).unwrap_or(u32::MAX),
            completion_tokens: ok.and_then(|s| s.completion_tokens),
            finish_reason: ok.and_then(|s| s.finish_reason.clone()),
            refused: accepted(sample, n_predict).err(),
            decode_tok_s: ok.and_then(|s| s.decode_tok_s),
        }
    }
}

/// One band of the witness, in the shape the V3 shape check reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeBand {
    /// Concurrent identical requests fired.
    pub c: u32,
    /// The largest batch the scheduler logged while this band ran.
    pub m_formed: u32,
    /// PP-26 v3.1's verdict on the band.
    pub result: BatchInvariance,
    /// The script's `divergence_at`: the first token at which the `m = 1`
    /// reference and the batch part, or the shorter stream's length when they
    /// never do. Recorded, not gated. `None` when the band was not measured.
    pub divergence_at: Option<u32>,
    /// The reference's top-2 margin at `divergence_at`, when they part.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top2_margin_at_divergence: Option<f64>,
    /// PP-26 (a): the shortest agreement of any slot with slot 0.
    pub intra_agree_to: Option<u32>,
    /// PP-26 (b): the longest run of one token in any slot.
    pub max_constant_run: Option<u32>,
    /// Tokens the slots had to agree for.
    pub declared_min: u32,
    /// The run of one token that marks a frozen slot.
    pub max_constant_run_declared: u32,
    /// Why the band is not PASS.
    pub reason: Option<String>,
    /// S7: the median of the slots' decode rates, from [`band_decode_tok_s`].
    #[serde(default)]
    pub decode_tok_s: Option<f64>,
    /// Every slot fired.
    pub slots: Vec<ProbeSlot>,
}

/// §4.4.3 — a band's per-request decode rate: the median of its slots'.
/// `None` when the band has no slot or any slot was not timed: the median of
/// the slots that happened to be timed is not the band's, and V3 reads `None`
/// as S7 unmeasured, never assumed.
#[must_use]
pub fn band_decode_tok_s(slots: &[ProbeSlot]) -> Option<f64> {
    let mut rates = slots
        .iter()
        .map(|s| s.decode_tok_s)
        .collect::<Option<Vec<f64>>>()?;
    rates.sort_by(f64::total_cmp);
    percentile(&rates, 0.50)
}

impl ProbeBand {
    fn unmeasurable(c: u32, m_formed: u32, policy: &ProbePolicy, slots: Vec<ProbeSlot>) -> Self {
        Self {
            c,
            m_formed,
            result: BatchInvariance::Unmeasurable,
            divergence_at: None,
            top2_margin_at_divergence: None,
            intra_agree_to: None,
            max_constant_run: None,
            declared_min: policy.declared_min,
            max_constant_run_declared: policy.max_constant_run,
            reason: None,
            decode_tok_s: band_decode_tok_s(&slots),
            slots,
        }
    }
}

/// Why the band cannot be decided, before any token is compared.
fn band_refusal(c: u32, m_formed: u32, slots: &[ProbeSlot]) -> Option<String> {
    if u32::try_from(slots.len()).ok() != Some(c) {
        return Some(format!("fired c={c} requests, {} came back", slots.len()));
    }
    let refusals: Vec<String> = slots
        .iter()
        .filter_map(|s| s.refused.as_ref().map(|why| format!("slot {}: {why}", s.i)))
        .collect();
    if !refusals.is_empty() {
        return Some(refusals.join("; "));
    }
    // A batch is two or more sequences in one step, so `m_formed < 2` is the
    // word "batch", not a threshold. c = 1 needs none: its witness is the
    // reference's agreement with itself.
    (c > 1 && m_formed < 2).then(|| {
        format!(
            "no batch with m>1 formed in this band's window (max m={m_formed}); the batched \
             path was never exercised, so nothing about it was witnessed"
        )
    })
}

/// Why a measured band FAILed, in PP-26 v3.1's two rules.
fn fail_reason(witness: &BatchInvarianceWitness, policy: &ProbePolicy) -> Option<String> {
    let mut reasons = Vec::new();
    let run = witness.max_constant_run.unwrap_or(0);
    if policy.max_constant_run > 0 && run >= policy.max_constant_run {
        reasons.push(format!(
            "a slot repeated one token {run} times, at or above max_constant_run={} (#2753 signature)",
            policy.max_constant_run
        ));
    }
    let intra = witness.intra_agree_to.unwrap_or(0);
    if intra < policy.declared_min {
        reasons.push(format!(
            "slots of one batch disagree at token {intra} < declared_min={}",
            policy.declared_min
        ));
    }
    (!reasons.is_empty()).then(|| reasons.join("; "))
}

/// Score one band: `samples` are the `c` slots, `m1` the reference stream,
/// `m_formed` the largest batch the scheduler logged while the band ran.
#[must_use]
pub fn evaluate_band(
    c: u32,
    m_formed: u32,
    m1: &[TokenLogprob],
    samples: &[SampleResult],
    policy: &ProbePolicy,
) -> ProbeBand {
    let slots = samples
        .iter()
        .enumerate()
        .map(|(i, s)| ProbeSlot::of(i, s, policy.n_predict))
        .collect::<Vec<_>>();
    let refusal = band_refusal(c, m_formed, &slots);
    let mut band = ProbeBand::unmeasurable(c, m_formed, policy, slots);
    if refusal.is_some() {
        band.reason = refusal;
        return band;
    }
    let streams: Vec<&[TokenLogprob]> = samples
        .iter()
        .filter_map(|s| s.as_ref().ok())
        .map(|s| s.logprobs.as_slice())
        .collect();
    let witness = BatchInvarianceWitness::compare_logprobs(
        m1,
        &streams,
        policy.declared_min,
        policy.max_constant_run,
    );
    let shorter = m1.len().min(streams.first().map_or(0, |s| s.len()));
    band.divergence_at = witness
        .divergence_at
        .or_else(|| u32::try_from(shorter).ok());
    band.top2_margin_at_divergence = witness.top2_margin_at_divergence;
    band.intra_agree_to = witness.intra_agree_to;
    band.max_constant_run = witness.max_constant_run;
    band.result = witness.batch_invariance;
    band.reason = match band.result {
        BatchInvariance::Pass => None,
        BatchInvariance::Fail => fail_reason(&witness, policy),
        BatchInvariance::Unmeasurable => Some(format!(
            "no slot reached declared_min={}",
            policy.declared_min
        )),
    };
    band
}

/// The run's verdict: FAIL on any failed band; UNMEASURABLE when the reference
/// was not stable, a band could not be decided, or no band at `c > 1` ran.
#[must_use]
pub fn run_verdict(reference: &ProbeReference, bands: &[ProbeBand]) -> BatchInvariance {
    if bands.iter().any(|b| b.result == BatchInvariance::Fail) {
        return BatchInvariance::Fail;
    }
    let undecided = bands
        .iter()
        .any(|b| b.result == BatchInvariance::Unmeasurable);
    if !reference.stable || undecided || !bands.iter().any(|b| b.c > 1) {
        return BatchInvariance::Unmeasurable;
    }
    BatchInvariance::Pass
}

/// The largest `m` in the scheduler's `Batch m=N done` lines of `log`, or 0.
#[must_use]
pub fn max_batch_formed(log: &str) -> u32 {
    log.match_indices(BATCH_MARK)
        .filter_map(|(at, _)| {
            let rest = &log[at + BATCH_MARK.len()..];
            let end = rest
                .find(|ch: char| !ch.is_ascii_digit())
                .unwrap_or(rest.len());
            let (digits, tail) = rest.split_at(end);
            tail.starts_with(" done")
                .then_some(digits)
                .and_then(|d| d.parse::<u32>().ok())
        })
        .max()
        .unwrap_or(0)
}

/// The server log's length now, so a band reads only what it caused. A log
/// that cannot be read has length 0.
#[must_use]
pub fn log_offset(server_log: &Path) -> u64 {
    std::fs::metadata(server_log).map_or(0, |m| m.len())
}

/// The largest batch the scheduler logged after `offset` bytes, 0 when the log
/// cannot be read. Reading from the band's own offset is what keeps a batch
/// formed during c=4 from being credited to c=8.
#[must_use]
pub fn batch_formed_since(server_log: &Path, offset: u64) -> u32 {
    let mut bytes = Vec::new();
    let read = std::fs::File::open(server_log).and_then(|mut file| {
        file.seek(SeekFrom::Start(offset))?;
        file.read_to_end(&mut bytes)
    });
    read.map_or(0, |_| max_batch_formed(&String::from_utf8_lossy(&bytes)))
}

/// The sampler every request carries.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProbeSampler {
    /// 0 for the witness: greedy.
    pub temperature: f64,
    /// Pinned so two runs draw alike.
    pub seed: u64,
    /// So every request generates exactly `max_tokens`.
    pub ignore_eos: bool,
    /// `protocol.n_predict`.
    pub max_tokens: u32,
    /// The best tokens asked for per step; 2 gives the top-2 margin.
    pub top_logprobs: u8,
}

/// The best tokens asked for per step: two, so a band can record the top-2 margin.
pub const TOP_LOGPROBS: u8 = 2;

/// What configures a probe run, all of it from `scripts/perf-matrix.yaml`
/// (PP-33): the policy a band is judged by, the sampler every request carries,
/// and the declared ladder.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeMatrix {
    /// `witness.min_agree_tokens`, `protocol.n_predict`, `witness.max_constant_run`.
    pub policy: ProbePolicy,
    /// `protocol.sampler`, with `max_tokens = protocol.n_predict` and [`TOP_LOGPROBS`].
    pub sampler: ProbeSampler,
    /// `ladder.declared`: the concurrencies fired, in order.
    pub ladder: Vec<u32>,
}

impl ProbeMatrix {
    /// Read from the compiled-in matrix.
    ///
    /// # Errors
    /// As [`Self::from_matrix_source`].
    pub fn from_matrix() -> Result<Self, String> {
        Self::from_matrix_source(PERF_MATRIX_SOURCE)
    }

    /// Read from an explicit document. The script falls back to its own
    /// literals when a key is absent and notes it; this refuses instead and
    /// names the key, because a probe judged by numbers the matrix does not
    /// hold is the drift PP-33 exists to prevent.
    ///
    /// # Errors
    /// When a block or a key is missing; when the ladder is empty or holds a
    /// 0; when `max_constant_run` is 0; or when `min_agree_tokens` exceeds
    /// `n_predict`, so no run could pass.
    pub fn from_matrix_source(source: &str) -> Result<Self, String> {
        let protocol = ProtocolParams::from_matrix_source(source)?;
        let witness: MatrixProbeWitness =
            serde_yaml_ng::from_value(matrix_block(source, "witness")?)
                .map_err(|e| format!("perf-matrix.yaml `witness:` block: {e}"))?;
        let ladder: MatrixLadder = serde_yaml_ng::from_value(matrix_block(source, "ladder")?)
            .map_err(|e| format!("perf-matrix.yaml `ladder:` block: {e}"))?;
        let policy = ProbePolicy {
            declared_min: witness.min_agree_tokens,
            n_predict: protocol.n_predict,
            max_constant_run: witness.max_constant_run,
        };
        check_matrix(&policy, &ladder.declared)?;
        Ok(Self {
            policy,
            sampler: ProbeSampler {
                temperature: protocol.sampler.temperature,
                seed: protocol.sampler.seed,
                ignore_eos: protocol.sampler.ignore_eos,
                max_tokens: protocol.n_predict,
                top_logprobs: TOP_LOGPROBS,
            },
            ladder: ladder.declared,
        })
    }
}

/// Refuse a matrix under which no run could be judged.
fn check_matrix(policy: &ProbePolicy, ladder: &[u32]) -> Result<(), String> {
    if ladder.is_empty() || ladder.contains(&0) {
        return Err(format!(
            "perf-matrix.yaml `ladder.declared` must list concurrencies >= 1, got {ladder:?}"
        ));
    }
    if policy.max_constant_run == 0 {
        return Err(
            "perf-matrix.yaml `witness.max_constant_run` is 0: every slot would read as frozen"
                .to_string(),
        );
    }
    if policy.declared_min > policy.n_predict {
        return Err(format!(
            "perf-matrix.yaml `witness.min_agree_tokens` ({}) exceeds `protocol.n_predict` ({}): no \
             slot could agree for that long",
            policy.declared_min, policy.n_predict
        ));
    }
    Ok(())
}

/// The two `witness:` keys the probe reads. The rest of the block is
/// governance and is ignored, as in every `Matrix*` reader in
/// [`super::protocol`].
#[derive(Debug, Deserialize)]
struct MatrixProbeWitness {
    min_agree_tokens: u32,
    max_constant_run: u32,
}

/// The `ladder:` key the probe reads.
#[derive(Debug, Deserialize)]
struct MatrixLadder {
    declared: Vec<u32>,
}

/// The served model: the file name, never a host path, and its bytes' sha256.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeModel {
    /// The file name only.
    pub path: Option<String>,
    /// The file's sha256.
    pub sha256: Option<String>,
}

/// The witness a probe run writes. Identity fields the caller did not give
/// stay `null`; the V3 shape check names each one missing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeWitness {
    /// [`WITNESS_VERSION`].
    pub witness_version: u32,
    /// [`PROBE`].
    pub probe: String,
    /// The host the server ran on.
    pub host: Option<String>,
    /// The commit the server was built from.
    pub commit: Option<String>,
    /// The serving binary's sha256.
    pub binary_sha256: Option<String>,
    /// The served model.
    pub model: ProbeModel,
    /// The prompt's sha256.
    pub prompt_sha256: String,
    /// The sampler on every request.
    pub sampler: ProbeSampler,
    /// Tokens a batch's slots had to agree for.
    pub declared_min: u32,
    /// [`ENV_KEYS`] as this process saw them.
    pub env: BTreeMap<String, Option<String>>,
    /// The `m = 1` reference.
    pub reference: ProbeReference,
    /// One per `c` fired, in ladder order.
    pub bands: Vec<ProbeBand>,
    /// [`run_verdict`].
    pub result: BatchInvariance,
}

/// The prompt's sha256, as 64 lowercase hex characters.
#[must_use]
pub fn prompt_sha256(prompt: &str) -> String {
    format!("{:x}", Sha256::digest(prompt.as_bytes()))
}

/// [`ENV_KEYS`] as this process sees them; an unset key is `None`.
#[must_use]
pub fn env_block() -> BTreeMap<String, Option<String>> {
    ENV_KEYS
        .iter()
        .map(|key| ((*key).to_string(), std::env::var(key).ok()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::margin::TopLogprob;
    use super::super::v3_shape::{check_v3_shape, ShapeVerdict};
    use super::*;

    const POLICY: ProbePolicy = ProbePolicy {
        declared_min: 4,
        n_predict: 8,
        max_constant_run: 5,
    };

    /// A token whose step's runner-up trails it by `gap`.
    fn tok(text: &str, gap: f64) -> TokenLogprob {
        let top = |token: &str, logprob: f64| TopLogprob {
            token: token.to_string(),
            logprob: Some(logprob),
            bytes: Some(token.as_bytes().to_vec()),
        };
        TokenLogprob {
            token: text.to_string(),
            logprob: Some(-0.1),
            bytes: Some(text.as_bytes().to_vec()),
            top_logprobs: vec![top(text, -0.1), top("~", -0.1 - gap)],
        }
    }

    /// `n_predict` tokens named by `words`, every step decided by a margin of 1.
    fn stream(words: &[&str]) -> Vec<TokenLogprob> {
        words.iter().map(|w| tok(w, 1.0)).collect()
    }

    /// The decode rate every [`sample`] was timed at.
    const RATE: f64 = 40.0;

    fn sample(tokens: Vec<TokenLogprob>) -> SampleResult {
        timed(tokens, Some(RATE))
    }

    fn timed(tokens: Vec<TokenLogprob>, decode_tok_s: Option<f64>) -> SampleResult {
        Ok(ProbeSample {
            completion_tokens: u32::try_from(tokens.len()).ok(),
            finish_reason: Some("length".to_string()),
            logprobs: tokens,
            decode_tok_s,
        })
    }

    const TEXT: [&str; 8] = ["a", "b", "c", "d", "e", "f", "g", "h"];

    fn reference() -> Vec<TokenLogprob> {
        stream(&TEXT)
    }

    fn band_of(c: u32, m_formed: u32, slots: &[SampleResult]) -> ProbeBand {
        evaluate_band(c, m_formed, &reference(), slots, &POLICY)
    }

    fn copies(c: u32, tokens: &[TokenLogprob]) -> Vec<SampleResult> {
        (0..c).map(|_| sample(tokens.to_vec())).collect()
    }

    #[test]
    fn identical_slots_pass_and_record_the_full_length_as_divergence_at() {
        let band = band_of(4, 4, &copies(4, &reference()));
        assert_eq!(band.result, BatchInvariance::Pass, "{band:?}");
        assert_eq!(band.divergence_at, Some(8));
        assert_eq!(band.top2_margin_at_divergence, None);
        assert_eq!(band.reason, None);
        assert_eq!(band.decode_tok_s, Some(RATE));
    }

    /// S7's rate, as `RequestSample::decode_tok_s` defines it, and every case
    /// where it is undefined rather than zero.
    #[test]
    fn live_decode_tok_s_times_only_a_live_stream_with_two_arrivals() {
        let ms = Duration::from_millis;
        let arrivals = [ms(100), ms(150), ms(300)];
        let near = |got: Option<f64>, want: f64| got.is_some_and(|r| (r - want).abs() < 1e-9);
        assert!(
            near(live_decode_tok_s(true, &arrivals, 9), 40.0),
            "8 tokens over 0.2 s"
        );
        assert!(
            near(live_decode_tok_s(true, &arrivals, 2), 5.0),
            "1 token over 0.2 s"
        );
        assert_eq!(
            live_decode_tok_s(false, &arrivals, 9),
            None,
            "a replay is not timed"
        );
        assert_eq!(live_decode_tok_s(true, &arrivals[..1], 9), None);
        assert_eq!(live_decode_tok_s(true, &[], 9), None);
        assert_eq!(live_decode_tok_s(true, &arrivals, 1), None);
        assert_eq!(
            live_decode_tok_s(true, &[ms(5), ms(5)], 9),
            None,
            "no time between"
        );
    }

    #[test]
    fn a_band_rate_is_the_slots_median_and_unmeasured_if_any_slot_is() {
        let slot = |decode_tok_s| ProbeSlot {
            i: 0,
            completion_tokens: Some(8),
            finish_reason: None,
            refused: None,
            decode_tok_s,
        };
        let odd = [slot(Some(10.0)), slot(Some(30.0)), slot(Some(20.0))];
        assert_eq!(band_decode_tok_s(&odd), Some(20.0));
        assert_eq!(band_decode_tok_s(&odd[..2]), Some(20.0));
        assert_eq!(band_decode_tok_s(&[slot(Some(10.0)), slot(None)]), None);
        assert_eq!(band_decode_tok_s(&[]), None);
    }

    /// The script's `witness_constant_token_m3`: #2753, one id forever.
    #[test]
    fn a_frozen_batch_fails_and_names_the_signature() {
        let frozen = stream(&["z"; 8]);
        let band = band_of(3, 3, &copies(3, &frozen));
        assert_eq!(band.result, BatchInvariance::Fail, "{band:?}");
        assert!(
            band.reason.as_deref().is_some_and(|r| r.contains("#2753")),
            "{band:?}"
        );
    }

    /// `witness_no_batch_formed_is_unmeasurable`: c > 1 with no batch logged.
    #[test]
    fn a_band_where_no_batch_formed_is_unmeasurable_never_a_pass() {
        let band = band_of(4, 1, &copies(4, &reference()));
        assert_eq!(band.result, BatchInvariance::Unmeasurable);
        assert!(band
            .reason
            .as_deref()
            .is_some_and(|r| r.contains("no batch with m>1")));
        assert_eq!(band.divergence_at, None);
    }

    /// c = 1 needs no batch.
    #[test]
    fn the_c1_band_passes_with_m_formed_1() {
        let band = band_of(1, 1, &copies(1, &reference()));
        assert_eq!(band.result, BatchInvariance::Pass, "{band:?}");
    }

    /// `witness_short_slot_is_unmeasurable`: PP-28 per slot.
    #[test]
    fn a_short_slot_makes_the_band_unmeasurable_and_names_the_slot() {
        let mut slots = copies(4, &reference());
        slots[2] = sample(reference()[..6].to_vec());
        let band = band_of(4, 4, &slots);
        assert_eq!(band.result, BatchInvariance::Unmeasurable);
        let reason = band.reason.unwrap_or_default();
        assert!(
            reason.contains("slot 2: completion_tokens=6 != n_predict=8"),
            "{reason}"
        );
    }

    #[test]
    fn a_slot_without_logprob_entries_is_refused_by_name() {
        let mut slots = copies(2, &reference());
        slots[1] = Ok(ProbeSample {
            logprobs: Vec::new(),
            completion_tokens: Some(8),
            finish_reason: None,
            decode_tok_s: None,
        });
        let band = band_of(2, 2, &slots);
        assert_eq!(band.result, BatchInvariance::Unmeasurable);
        assert!(band
            .reason
            .unwrap_or_default()
            .contains("must ask for logprobs"));
    }

    #[test]
    fn a_failed_request_or_a_missing_slot_is_unmeasurable() {
        let mut slots = copies(2, &reference());
        slots[0] = Err("connection refused".to_string());
        let band = band_of(2, 2, &slots);
        assert_eq!(band.result, BatchInvariance::Unmeasurable);
        assert_eq!(
            band.decode_tok_s, None,
            "a slot that never came back was not timed"
        );
        assert!(band
            .reason
            .unwrap_or_default()
            .contains("slot 0: connection refused"));
        let band = band_of(4, 4, &copies(3, &reference()));
        assert_eq!(band.result, BatchInvariance::Unmeasurable);
        assert!(band
            .reason
            .unwrap_or_default()
            .contains("fired c=4 requests, 3 came back"));
    }

    /// `witness_intra_batch_disagree_m4`: slot 3 parts from slot 0 at 2 < 4.
    #[test]
    fn slots_that_disagree_before_declared_min_fail() {
        let mut slots = copies(4, &reference());
        let mut odd = reference();
        odd[2] = tok("X", 1.0);
        slots[3] = sample(odd);
        let band = band_of(4, 4, &slots);
        assert_eq!(band.result, BatchInvariance::Fail, "{band:?}");
        assert_eq!(band.intra_agree_to, Some(2));
        assert!(band
            .reason
            .unwrap_or_default()
            .contains("disagree at token 2"));
    }

    /// `witness_kernel_family_flip_recorded_ok`, now with the margin: the whole
    /// batch parts from m=1 at a near-tie, which is recorded and still PASS.
    #[test]
    fn a_batch_that_parts_from_m1_at_a_near_tie_passes_and_records_the_margin() {
        let mut m1 = reference();
        m1[1] = tok("b", 0.01);
        let mut flipped = reference();
        flipped[1] = tok("~", 0.01);
        let slots = copies(4, &flipped);
        let band = evaluate_band(4, 4, &m1, &slots, &POLICY);
        assert_eq!(band.result, BatchInvariance::Pass, "{band:?}");
        assert_eq!(band.divergence_at, Some(1));
        let margin = band.top2_margin_at_divergence.expect("margin recorded");
        assert!((margin - 0.01).abs() < 1e-9, "{margin}");
    }

    #[test]
    fn the_reference_is_stable_only_when_both_decodes_agree_to_declared_min() {
        let ok = check_reference(&sample(reference()), &sample(reference()), &POLICY);
        assert!(ok.stable, "{ok:?}");
        assert_eq!(ok.self_divergence_at, Some(8));
        let mut early = reference();
        early[2] = tok("X", 1.0);
        let unstable = check_reference(&sample(reference()), &sample(early), &POLICY);
        assert!(!unstable.stable);
        assert_eq!(unstable.self_divergence_at, Some(2));
        let short = check_reference(
            &sample(reference()[..3].to_vec()),
            &sample(reference()),
            &POLICY,
        );
        assert!(!short.stable);
        assert!(short
            .reason
            .unwrap_or_default()
            .starts_with("ref#1: completion_tokens=3"));
        let gone = check_reference(&sample(reference()), &Err("timeout".into()), &POLICY);
        assert_eq!(gone.reason.as_deref(), Some("ref#2: timeout"));
    }

    fn passed(c: u32) -> ProbeBand {
        band_of(c, c, &copies(c, &reference()))
    }

    #[test]
    fn the_run_passes_only_with_a_measured_batched_band_and_a_stable_reference() {
        let stable = ProbeReference {
            stable: true,
            ..ProbeReference::default()
        };
        let full = vec![passed(1), passed(4)];
        assert_eq!(run_verdict(&stable, &full), BatchInvariance::Pass);
        // `witness_c1_only_is_not_a_pass`.
        assert_eq!(
            run_verdict(&stable, &[passed(1)]),
            BatchInvariance::Unmeasurable
        );
        assert_eq!(run_verdict(&stable, &[]), BatchInvariance::Unmeasurable);
        assert_eq!(
            run_verdict(&ProbeReference::default(), &full),
            BatchInvariance::Unmeasurable
        );
        let undecided = vec![passed(4), band_of(8, 1, &copies(8, &reference()))];
        assert_eq!(
            run_verdict(&stable, &undecided),
            BatchInvariance::Unmeasurable
        );
        let frozen = band_of(3, 3, &copies(3, &stream(&["z"; 8])));
        assert_eq!(
            run_verdict(&stable, &[passed(4), frozen]),
            BatchInvariance::Fail
        );
    }

    /// `witness_cross_band_log_not_credited`: the scheduler's lines are read
    /// from the band's own offset.
    #[test]
    fn batch_formed_since_reads_only_past_the_offset() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = dir.path().join("server.log");
        std::fs::write(
            &log,
            "[PMAT-044] Batch m=4 done in 3.0ms (9.0 tok/s/slot)\n",
        )
        .expect("write");
        let offset = log_offset(&log);
        assert_eq!(batch_formed_since(&log, 0), 4);
        assert_eq!(batch_formed_since(&log, offset), 0);
        let mut text = std::fs::read_to_string(&log).expect("read");
        text.push_str("[PMAT-044] Batch m=2 done in 1.0ms (9.0 tok/s/slot)\n");
        std::fs::write(&log, text).expect("append");
        assert_eq!(batch_formed_since(&log, offset), 2);
        assert_eq!(batch_formed_since(&dir.path().join("absent.log"), 0), 0);
        assert_eq!(log_offset(&dir.path().join("absent.log")), 0);
    }

    #[test]
    fn max_batch_formed_takes_the_largest_well_formed_line() {
        let log = "Batch m=3 done\nBatch m=16 done in 2ms\nBatch m=99 started\nBatch m= done\nBatch m=8 done";
        assert_eq!(max_batch_formed(log), 16);
        assert_eq!(max_batch_formed(""), 0);
        assert_eq!(max_batch_formed("Batch m=12"), 0);
    }

    /// What V3's floor admits: `declared_min` at the perf matrix's 64.
    const V3_POLICY: ProbePolicy = ProbePolicy {
        declared_min: 64,
        n_predict: 96,
        max_constant_run: 5,
    };

    /// `n_predict` distinct tokens, every step decided by a margin of 1.
    fn v3_reference() -> Vec<TokenLogprob> {
        (0..V3_POLICY.n_predict)
            .map(|i| tok(&format!("t{i} "), 1.0))
            .collect()
    }

    /// A band of `c` slots streaming `slot` against `m1`, each timed at `rate`.
    fn v3_band(c: u32, m1: &[TokenLogprob], slot: &[TokenLogprob], rate: Option<f64>) -> ProbeBand {
        let slots: Vec<SampleResult> = (0..c).map(|_| timed(slot.to_vec(), rate)).collect();
        evaluate_band(c, c, m1, &slots, &V3_POLICY)
    }

    fn v3_passed(c: u32, rate: f64) -> ProbeBand {
        v3_band(c, &v3_reference(), &v3_reference(), Some(rate))
    }

    /// A clean run whose c=4 band decodes at `c4_rate` and the rest at 40.
    fn v3_bands(c4_rate: Option<f64>) -> Vec<ProbeBand> {
        let m1 = v3_reference();
        vec![
            v3_passed(1, RATE),
            v3_band(4, &m1, &m1, c4_rate),
            v3_passed(8, RATE),
            v3_passed(16, RATE),
        ]
    }

    fn witness(bands: Vec<ProbeBand>) -> ProbeWitness {
        let reference = ProbeReference {
            stable: true,
            ..ProbeReference::default()
        };
        ProbeWitness {
            witness_version: WITNESS_VERSION,
            probe: PROBE.to_string(),
            host: Some("lambda".to_string()),
            commit: Some("f".repeat(40)),
            binary_sha256: Some("a".repeat(64)),
            model: ProbeModel {
                path: Some("Qwen3.5-4B-Q4_K_M.gguf".to_string()),
                sha256: Some("c".repeat(64)),
            },
            prompt_sha256: prompt_sha256("Write an essay on compilers."),
            sampler: ProbeSampler {
                temperature: 0.0,
                seed: 0,
                ignore_eos: true,
                max_tokens: V3_POLICY.n_predict,
                top_logprobs: 2,
            },
            declared_min: V3_POLICY.declared_min,
            env: env_block(),
            result: run_verdict(&reference, &bands),
            reference,
            bands,
        }
    }

    /// Producer against consumer: what this module writes for a clean run is
    /// what the V3 shape check admits. A band that never parts from m=1 must
    /// still record `divergence_at`, or V3 reads it as unmeasured.
    #[test]
    fn a_clean_run_writes_a_witness_the_v3_shape_check_admits() {
        let w = witness(v3_bands(Some(RATE)));
        assert_eq!(w.result, BatchInvariance::Pass);
        let text = serde_json::to_string_pretty(&w).expect("serialize");
        assert_eq!(check_v3_shape(&text), ShapeVerdict::Admissible, "{text}");
        let back: ProbeWitness = serde_json::from_str(&text).expect("round trip");
        assert_eq!(back, w);
    }

    /// The same, with the c=8 band parting from m=1 at a near-tie: admitted on
    /// the margin. Without the margin it would not be.
    #[test]
    fn a_near_tie_divergence_is_admitted_by_v3_only_with_its_margin() {
        let mut m1 = v3_reference();
        m1[1] = tok("t1 ", 0.01);
        let mut flipped = v3_reference();
        flipped[1] = tok("~", 0.01);
        let flip = v3_band(8, &m1, &flipped, Some(RATE));
        assert_eq!(flip.divergence_at, Some(1), "{flip:?}");
        let mut w = witness(v3_bands(Some(RATE)));
        w.bands[2] = flip;
        let text = serde_json::to_string(&w).expect("serialize");
        assert_eq!(check_v3_shape(&text), ShapeVerdict::Admissible, "{text}");
        w.bands[2].top2_margin_at_divergence = None;
        let text = serde_json::to_string(&w).expect("serialize");
        assert!(
            matches!(check_v3_shape(&text), ShapeVerdict::NotAdmissible(_)),
            "{text}"
        );
    }

    #[test]
    fn missing_identity_stays_null_and_v3_names_it() {
        let mut w = witness(v3_bands(Some(RATE)));
        w.commit = None;
        let text = serde_json::to_string(&w).expect("serialize");
        assert!(text.contains("\"commit\":null"), "{text}");
        assert_eq!(
            check_v3_shape(&text),
            ShapeVerdict::NotAdmissible(vec!["identity field commit missing".to_string()])
        );
    }

    /// FALSIFY-SSP-017 through the producer: the rates the probe records are
    /// the rates V3 reads. c=4 at 49% of c=1 is refused by name; at 50% it is
    /// admitted; one untimed c=4 slot leaves S7 unmeasured.
    #[test]
    fn the_band_rates_the_probe_writes_are_what_v3_s7_reads() {
        let check = |c4_rate| {
            check_v3_shape(&serde_json::to_string(&witness(v3_bands(c4_rate))).expect("serialize"))
        };
        assert_eq!(
            check(Some(19.6)),
            ShapeVerdict::NotAdmissible(vec![
                "S7 (FALSIFY-CB-004): per-request decode at c=4 is 19.6 tok/s, 49.0% of c=1's 40 tok/s; needs at least 50%".to_string()
            ])
        );
        assert_eq!(check(Some(20.0)), ShapeVerdict::Admissible);
        let unmeasured = ShapeVerdict::NotAdmissible(vec![
            "S7 (FALSIFY-CB-004) is unmeasured: no positive per-request decode_tok_s at c=4"
                .to_string(),
        ]);
        assert_eq!(check(None), unmeasured);
        let mut w = witness(v3_bands(Some(RATE)));
        w.bands[1].slots[2].decode_tok_s = None;
        w.bands[1].decode_tok_s = band_decode_tok_s(&w.bands[1].slots);
        let text = serde_json::to_string(&w).expect("serialize");
        assert!(text.contains("\"decode_tok_s\":null"), "{text}");
        assert_eq!(check_v3_shape(&text), unmeasured);
    }

    #[test]
    fn env_block_records_every_key_and_prompt_sha256_is_64_hex() {
        let env = env_block();
        assert_eq!(env.len(), ENV_KEYS.len());
        let sha = prompt_sha256("");
        assert_eq!(
            sha,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    /// A matrix with every key the probe reads, plus a governance key it ignores.
    fn matrix(witness: &str, ladder: &str) -> String {
        format!(
            "protocol:\n  window_ms: 1000\n  warmup_requests_per_worker: 1\n  quiesce_ms: 0\n  \
             cooldown_ms: 0\n  n_predict: 32\n  replicates_min: 1\n  interleaved: true\n  \
             sampler: {{temperature: 0.0, seed: 9, ignore_eos: true}}\nwitness:\n{witness}  \
             author: spec-owner\nladder:\n{ladder}"
        )
    }

    const WITNESS_OK: &str = "  min_agree_tokens: 24\n  max_constant_run: 6\n";

    #[test]
    fn the_matrix_configures_the_policy_the_sampler_and_the_ladder() {
        let m = ProbeMatrix::from_matrix_source(&matrix(WITNESS_OK, "  declared: [1, 2, 3]\n"))
            .expect("a complete matrix reads");
        assert_eq!(
            m.policy,
            ProbePolicy {
                declared_min: 24,
                n_predict: 32,
                max_constant_run: 6
            }
        );
        assert_eq!(
            m.sampler,
            ProbeSampler {
                temperature: 0.0,
                seed: 9,
                ignore_eos: true,
                max_tokens: 32,
                top_logprobs: 2
            }
        );
        assert_eq!(m.ladder, vec![1, 2, 3]);
    }

    #[test]
    fn the_shipped_matrix_configures_the_probe_as_the_other_readers_read_it() {
        let m = ProbeMatrix::from_matrix().expect("the shipped matrix configures the probe");
        let protocol = ProtocolParams::from_matrix().expect("the shipped protocol block");
        let declared_min =
            super::super::protocol::witness_min_agree_tokens_from(PERF_MATRIX_SOURCE)
                .expect("the shipped witness block");
        assert_eq!(m.policy.declared_min, declared_min);
        assert_eq!(m.policy.n_predict, protocol.n_predict);
        assert_eq!(m.sampler.max_tokens, protocol.n_predict);
        assert_eq!(m.sampler.seed, protocol.sampler.seed);
        assert!(
            m.ladder.iter().any(|&c| c > 1),
            "a ladder with no c > 1 could never PASS: {:?}",
            m.ladder
        );
    }

    #[test]
    fn a_matrix_that_cannot_judge_a_run_is_refused_by_name() {
        for (witness, ladder, named) in [
            (
                "  min_agree_tokens: 24\n",
                "  declared: [1, 2]\n",
                "max_constant_run",
            ),
            (WITNESS_OK, "  derive_from: []\n", "declared"),
            (WITNESS_OK, "  declared: []\n", "ladder.declared"),
            (WITNESS_OK, "  declared: [1, 0]\n", "ladder.declared"),
            (
                "  min_agree_tokens: 24\n  max_constant_run: 0\n",
                "  declared: [1, 2]\n",
                "max_constant_run",
            ),
            (
                "  min_agree_tokens: 33\n  max_constant_run: 6\n",
                "  declared: [1, 2]\n",
                "min_agree_tokens",
            ),
        ] {
            let err = ProbeMatrix::from_matrix_source(&matrix(witness, ladder)).expect_err(named);
            assert!(err.contains(named), "{named}: {err}");
        }
        let no_ladder =
            matrix(WITNESS_OK, "  declared: [1]\n").replace("ladder:\n  declared: [1]\n", "");
        let err = ProbeMatrix::from_matrix_source(&no_ladder).expect_err("no ladder block");
        assert!(err.contains("`ladder:`"), "{err}");
    }
}
