//! PP-LLAMA-001 v3.0 §5.1 / §7.4 — band derivation: boundary effects, drain,
//! the four request counters that must never be conflated, the correctness and
//! streaming witnesses, and the status the band ends up with.
//!
//! # Why this exists
//!
//! `scripts/perf_gate.sh:42` fails any receipt whose `drain_ms` is absent. On
//! `62d23d8d1`, `grep -rn "drain_ms" --include="*.rs" crates` returned **zero
//! lines**: nothing in the workspace could produce the field, so Arm C rejected
//! every receipt that could ever be measured. A gate that can only fail is the
//! mirror of one that can only pass.
//!
//! # What `drain_ms` MEANS (PP-10), stated before it is computed
//!
//! The measurement window opens at offset `0` and **closes at `T`**. No new
//! request is issued at or after `T` (PP-10). Every request issued before `T` is
//! then *drained* — allowed to run on past `T` to completion or timeout.
//!
//! > `drain_ms` = (last settlement of any pre-`T` request) − `T`, clamped at 0.
//!
//! It is the length of the **drain phase**, not a property of any one request,
//! and it is `0` when nothing was still in flight at `T`. The `SUSPECT` rule
//! reads it exactly that way: `drain_ms > 0.5 × window` means one request
//! dominated the window and the band must be re-run longer.
//!
//! # The conflation this module refuses to make
//!
//! "A request that timed out during drain increments `timeouts`; one
//! **abandoned at drain deadline** increments `truncated`."
//!
//! `truncated` therefore means *the drain deadline arrived while this request
//! was still running*. It does **not** mean `finish_reason == "length"`.
//! W1 (§5.1) generates with `n_predict = 128` and **EOS ignored**, so every
//! single healthy W1 request ends with `finish_reason == "length"`. `agg`'s
//! numerator is over "completed, **non-truncated**" requests — so reading
//! `truncated` in the finish-reason sense empties the numerator and reports
//! `0 tok/s` for a perfectly healthy server. The two senses are named apart
//! here on purpose: [`Outcome::AbandonedAtDrain`] is the drain sense and is the
//! only thing that increments `truncated`.
//!
//! # Timeouts are their own counter, and it is checked, not just named
//!
//! §3 fixes a hard **120 s per request**. [`Outcome::Timeout`] and
//! [`Outcome::Failed`] are distinct counters (PP-5 makes `timeouts > 0` fatal to
//! a band's ratio, while a transport error is a different fault), and
//! [`BandInput::derive`] *verifies* the label against the request's own
//! duration: a `Timeout` that did not reach the timeout, or a `Failed` that
//! exceeded it, is refused rather than counted.
//!
//! # v3: three witnesses a band must carry, and what happens without them
//!
//! | witness | rule | absent or failing |
//! |---|---|---|
//! | [`BatchInvarianceWitness`] (PP-26) | the tokens were right | `INVALID-CORRECTNESS` at `c > 1`; **no** `agg`/`dec`/`prefill` written |
//! | [`StreamMode`] + [`StreamWitness`] (PP-27) | the stream was live | `dec`/`ttft`/`itl` move to `unproduced`; `NONCONFORMANT-VALID` |
//! | `n_predict` (PP-28) | every retained sample ran to length | `short_of_n_predict > 0`; `NONCONFORMANT-VALID` |
//!
//! A failing witness does **not** make [`BandInput::derive`] return `Err`. The
//! band still renders, with the numbers it is entitled to and a status that
//! says what it lacks — because the evidence of a bad run is the point of
//! keeping it (Appendix C's `validity_by_band`). Only a band that contradicts
//! its own clock (a request issued after `T`, a mislabelled timeout) is refused
//! outright: that is not a bad measurement, it is not a measurement.
//!
//! # Nothing here is defaulted
//!
//! Every number below is derived from per-request timestamps supplied by the
//! caller. There is no constructor that accepts a `drain_ms` scalar, because a
//! caller-supplied `drain_ms` is indistinguishable from a fabricated one — the
//! same rule `scripts/lib/bench_receipt.py` already applies to ratios ("a stated
//! ratio that its own samples do not produce is a fabricated measurement").

use serde::{Deserialize, Serialize};

use super::bootstrap::{median_decode_tok_s, paired_ratio_lcb};
use super::join::{BandRatios, JoinKey, Ratio, RatioMethod};
use super::metrics::RequestSample;
use super::protocol::{stream_live_ttft_over_e2e_max, INTERLEAVED, REPLICATES};
use super::receipt::RunId;
use super::replicate::MIN_REPLICATES;
use super::samples::SamplesFile;
use super::witness::BatchInvarianceWitness;

/// §3 — the hard per-request timeout, in milliseconds.
pub const REQUEST_TIMEOUT_MS: f64 = 120_000.0;

/// PP-10 — `drain_ms > DRAIN_SUSPECT_FRACTION × window_ms` is annotated `SUSPECT`.
pub const DRAIN_SUSPECT_FRACTION: f64 = 0.5;

/// The receipt schema version this producer writes. `3` is PP-LLAMA-001 v3.0;
/// a receipt without the key is version 2 and is historical (PP-4).
pub const SCHEMA_VERSION: u32 = 3;

/// P-5 — the one-sided confidence the verdict is taken at.
pub const VERDICT_CONFIDENCE: f64 = 0.95;

/// How one sampled request ended. The four variants are mutually exclusive, so
/// `requested == completed + timeouts + truncated + errors` holds by
/// construction rather than by convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// Returned a usable response, inside the window or during the drain.
    Completed,
    /// Reached the §3 hard 120 s timeout.
    Timeout,
    /// Still running when the drain deadline arrived (PP-10). Increments
    /// `truncated`. **Not** `finish_reason == "length"` — see the module docs.
    AbandonedAtDrain,
    /// Any other fault: transport, non-2xx, unparseable body. Counted apart
    /// from [`Outcome::Timeout`] because they are different defects.
    Failed,
}

/// PP-27 — what the **server** declared on the first SSE chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamMode {
    /// Tokens were emitted as they were produced.
    Live,
    /// The answer was produced first and replayed as a stream. Every
    /// client-side latency metric is then a property of the replay, not of the
    /// server.
    Replayed,
}

/// PP-27 — the client's own verdict, independent of what the server declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamVerdict {
    /// The stream is live: either the server declared `live` and
    /// `median(ttft / e2e)` agrees, or the server declared nothing and the
    /// client's own ratio establishes it.
    Live,
    /// Server said `replayed`, or said `live` and the ratio contradicts it.
    Replayed,
    /// The server declared nothing **and** the client's ratio does not
    /// establish liveness either. Not the same as `replayed` — nothing said
    /// the answer was pre-computed — and not a pass: no half of the dual
    /// witness supports a latency metric.
    Undeclared,
}

/// PP-27 — which half of the dual witness the verdict rests on.
///
/// Upstream `llama-server` declares no `stream_mode` on its SSE chunks and is
/// not going to start. Reading "undeclared" as "not live" made **every**
/// comparator band `NONCONFORMANT-VALID`, so no baseline could ever be
/// conformant and the parity arm could never reach a verdict — a rule about a
/// field the oracle does not emit, dressed as a finding about the oracle.
/// The client's `median(ttft / e2e)` is a measurement of the same fact, so it
/// stands on its own; this field records that it had to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamWitnessSource {
    /// The server declared a mode and the client did not contradict it.
    Server,
    /// The server declared nothing, or declared `live` and the client's ratio
    /// overruled it. Either way the verdict is the client's measurement.
    Client,
}

/// PP-27 — the client-side half of the dual witness.
///
/// On a live stream the first token arrives long before the last, so
/// `ttft / e2e` is small. On a replayed one the whole answer lands at once and
/// the ratio approaches 1. The threshold is
/// `stream.live_ttft_over_e2e_max` in `perf-matrix.yaml` (PP-33).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamWitness {
    /// Median over completed requests of `ttft_ms / (settled_ms − issued_ms)`.
    pub client_ttft_over_e2e_median: f64,
    /// The verdict the two halves reach together.
    pub verdict: StreamVerdict,
    /// Which half the verdict rests on.
    pub source: StreamWitnessSource,
}

/// §7.4 — the band status vocabulary. Six tokens, no more.
///
/// `Skip` is not a status. `SUSPECT_DISPATCH` is not a status; a dispatch
/// anomaly is a finding with a mechanism, and this band carries `suspect[]`
/// annotations for it instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BandStatus {
    /// Conformant receipt, fresh pin, PP-26 passed, a baseline present.
    Measured,
    /// Temporary; in the denominator. Needs an owner.
    Unmeasured,
    /// Permanent; out of the denominator. Needs `decided_by`.
    Na,
    /// PP-26 absent or failed on this band. Can never be a baseline.
    InvalidCorrectness,
    /// Historical record; cited, never a baseline.
    NonconformantValid,
    /// PP-20 tripped: the comparator pin expired before the run started.
    ComparatorStale,
}

impl BandStatus {
    /// The §7.4 wire token. Spelled out rather than derived from the variant
    /// name because two of them are not the variant name
    /// (`NA`, `INVALID-CORRECTNESS`), and a reader of the receipt matches these
    /// strings exactly.
    #[must_use]
    pub fn wire_token(self) -> &'static str {
        match self {
            Self::Measured => "MEASURED",
            Self::Unmeasured => "UNMEASURED",
            Self::Na => "NA",
            Self::InvalidCorrectness => "INVALID-CORRECTNESS",
            Self::NonconformantValid => "NONCONFORMANT-VALID",
            Self::ComparatorStale => "COMPARATOR_STALE",
        }
    }

    /// The §7.4 vocabulary, in table order. The single source for the
    /// vocabulary test and for any validator that needs the closed set.
    #[must_use]
    pub fn vocabulary() -> [Self; 6] {
        [
            Self::Measured,
            Self::Unmeasured,
            Self::Na,
            Self::InvalidCorrectness,
            Self::NonconformantValid,
            Self::ComparatorStale,
        ]
    }

    /// P-4 / §7.4 — may a band with this status be a comparator baseline?
    #[must_use]
    pub fn baseline_eligible(self) -> bool {
        self == Self::Measured
    }

    /// §7.4 — where this status sits in the precedence order. **Lower wins.**
    ///
    /// `INVALID-CORRECTNESS > COMPARATOR_STALE > NA > NONCONFORMANT-VALID >
    /// UNMEASURED > MEASURED`, defined **once** here.
    ///
    /// The order is not arbitrary and each step is a different question:
    ///
    /// - `INVALID-CORRECTNESS` first, because §7.0 asks "were the tokens
    ///   right?" before "how fast?". A `c > 1` band with no passing witness has
    ///   no throughput at all, and a fresher comparator pin does not give it
    ///   one — which is exactly the inversion this rank fixes: the render pass
    ///   used to stamp `COMPARATOR_STALE` over it, and a band that reported no
    ///   numbers came out labelled as if its only problem were an expired pin.
    /// - `COMPARATOR_STALE` next: PP-20 blocks a band that would OTHERWISE have
    ///   been `MEASURED`/`UNMEASURED`, and says so rather than hiding it under
    ///   the weaker tokens below.
    /// - `NA` above `NONCONFORMANT-VALID`: an `NA` band is permanently out of
    ///   the denominator — usually because it never ran — and "the band that
    ///   did not run did not interleave" is not a finding about the run.
    #[must_use]
    pub fn rank(self) -> u8 {
        match self {
            Self::InvalidCorrectness => 0,
            Self::ComparatorStale => 1,
            Self::Na => 2,
            Self::NonconformantValid => 3,
            Self::Unmeasured => 4,
            Self::Measured => 5,
        }
    }

    /// The stronger of two statuses under [`Self::rank`].
    ///
    /// Every place that has to combine two verdicts goes through this, so the
    /// precedence cannot be spelled twice and drift.
    #[must_use]
    pub fn stronger_of(self, other: Self) -> Self {
        if other.rank() < self.rank() {
            other
        } else {
            self
        }
    }
}

/// PP-24 — which lane capped admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Lane {
    /// The subject, `apr serve`.
    Apr,
    /// The comparator, `llama-server`.
    Llama,
}

impl Lane {
    /// The wire token.
    #[must_use]
    pub fn wire_token(self) -> &'static str {
        match self {
            Self::Apr => "apr",
            Self::Llama => "llama",
        }
    }
}

/// PP-24 — a band that could not run because a lane admitted fewer slots than
/// the band's `c`. Server-reported; a harness-computed cap is schema-fatal
/// (PP-13).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionCap {
    /// Which lane capped.
    pub lane: Lane,
    /// The slot count that lane reported.
    pub cap: u32,
}

/// §4.7.1 / §7.4 — a band's comparator posture.
///
/// [`Self::Measured`] is the only variant carrying numbers, and it has no
/// public constructor: [`BandInput::join_status`] is the only way to make one,
/// and it refuses a cross-run baseline (PP-3), a join-key mismatch (PP-22) and
/// a timed-out lane (PP-5). A bare `agg_ratio` scalar is therefore
/// unrepresentable rather than merely discouraged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum ComparatorStatus {
    /// Permanent exclusion. Needs the decision recorded.
    NotApplicable {
        /// Who decided, e.g. `perf-matrix.yaml`.
        decided_by: String,
        /// Why the comparator cannot exist for this cell.
        reason: String,
        /// PP-24 — the server-reported ceiling that put this band out of the
        /// ladder, when that is the reason.
        budget: Option<String>,
    },
    /// Temporary. Counted against the denominator; needs an owner.
    Unmeasured {
        /// Who owes the measurement.
        owner: String,
        /// Why it has not been measured yet.
        reason: String,
        /// PP-24 — set when the band was not run because a lane admitted fewer
        /// slots than `c`.
        admission_capped: Option<AdmissionCap>,
    },
    /// PP-3 — a same-run comparator lane and the ratios formed from it.
    ///
    /// The payload is a [`MeasuredJoin`], whose fields are **private** and
    /// whose constructor is crate-private: outside this crate the variant can
    /// be matched and read but not built, so PP-3, PP-22 and PP-5 cannot be
    /// stepped around with a struct literal.
    Measured(MeasuredJoin),
}

/// PP-3 / PP-22 / P-5 — a comparator lane that passed the join, and the ratios
/// it produced.
///
/// # Why the fields are private
///
/// While `ComparatorStatus::Measured` was a struct variant with public fields,
/// **any** caller could write
///
/// ```text
/// ComparatorStatus::Measured { baseline: Box::new(band), ratios }
/// ```
///
/// and attach a baseline from another run, another band, or a lane that timed
/// out — the three things [`BandInput::join_status_in`] exists to refuse. The
/// refusals lived in a function nothing forced anyone to call. Wrapping the
/// payload in a type whose only constructor is `pub(crate)` and which is
/// reachable only from that function makes the refusals structural: outside
/// `aprender-test-lib` there is no expression that produces one.
///
/// Reading is unrestricted — [`Self::baseline`] and [`Self::ratios`] — because
/// a receipt renderer must be able to see what it is rendering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredJoin {
    /// The comparator lane's band, derived under the same rules.
    baseline: Box<DerivedBand>,
    /// P-5 — the ratios, each with the bound its verdict is taken on.
    ratios: BandRatios,
}

impl MeasuredJoin {
    /// The only constructor, crate-private and called from exactly one place:
    /// [`BandInput::join_status_in`], after PP-3, PP-22 and PP-5 have all been
    /// checked.
    pub(crate) fn sealed(baseline: DerivedBand, ratios: BandRatios) -> Self {
        Self {
            baseline: Box::new(baseline),
            ratios,
        }
    }

    /// The comparator lane's band.
    #[must_use]
    pub fn baseline(&self) -> &DerivedBand {
        &self.baseline
    }

    /// P-5 — the ratios formed against that band.
    #[must_use]
    pub fn ratios(&self) -> &BandRatios {
        &self.ratios
    }
}

impl ComparatorStatus {
    /// A temporary posture with no admission cap — the common case.
    #[must_use]
    pub fn unmeasured(owner: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Unmeasured {
            owner: owner.into(),
            reason: reason.into(),
            admission_capped: None,
        }
    }

    /// A permanent exclusion with no reported budget.
    #[must_use]
    pub fn not_applicable(decided_by: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::NotApplicable {
            decided_by: decided_by.into(),
            reason: reason.into(),
            budget: None,
        }
    }

    /// The wire token `perf_gate.sh` reads from `band.comparator_status`.
    ///
    /// The legacy spelling `NOT_APPLICABLE` is kept here deliberately: §7.4's
    /// `NA` lives in the new per-band `status` field, and changing this token
    /// would break every existing reader for no gain.
    #[must_use]
    pub fn wire_token(&self) -> &'static str {
        match self {
            Self::NotApplicable { .. } => "NOT_APPLICABLE",
            Self::Unmeasured { .. } => "UNMEASURED",
            Self::Measured(_) => "MEASURED",
        }
    }
}

/// The comparator lane's configuration, as it enters the PP-22 join key.
///
/// Every field is `Option` because a lane that did not report one has not
/// reported one — and `None` does not match `Some(x)` in
/// [`JoinKey::refuse_mismatch`], so an unreported field refuses the join
/// rather than acting as a wildcard.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaneConfig {
    /// `-c` per slot on the comparator; `props.n_ctx / props.total_slots`.
    pub n_ctx_slot: Option<u32>,
    /// KV cache type, e.g. `f16`.
    pub kv_type: Option<String>,
    /// Flash attention.
    pub fa: Option<bool>,
    /// `-b`. `Some(1)` is refused by the join key (§5.3).
    pub n_batch: Option<u32>,
}

/// One sampled request's terminal record. All offsets are milliseconds from the
/// window opening at `0`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestOutcome {
    /// When the client issued the request.
    pub issued_ms: f64,
    /// When the request reached its terminal state.
    pub settled_ms: f64,
    /// How it ended.
    pub outcome: Outcome,
    /// Generated (completion) tokens. Must be non-zero for a completion:
    /// a zero-token response is a failure, not a fast request.
    pub generated_tokens: u32,
    /// Prompt tokens the server reported (§3: all token counts are the
    /// server's `usage`).
    pub prompt_tokens: u32,
    /// PP-28 — the `n_predict` this request was issued with, when it differs
    /// from the band's (W2's ragged mixture). `None` means "the band's".
    pub expected_tokens: Option<u32>,
    /// Time to first token, when the transport streamed. `None` for a
    /// non-streaming client, which genuinely cannot observe it.
    pub ttft_ms: Option<f64>,
    /// Server-reported prefill duration for this request
    /// (`timings.prompt_ms` on llama.cpp, the `apr` equivalent per PP-2).
    /// `None` when the server reported none — never a client-side estimate
    /// (PP-13).
    pub prefill_ms: Option<f64>,
    /// Concurrent requests in flight at the instant this one was issued. The
    /// direct per-request evidence that the client was actually concurrent
    /// (PP-8).
    pub in_flight_at_start: u32,
    /// Absolute arrival offsets of each generated token, when the transport
    /// streamed. Empty for a non-streaming client.
    pub token_times_ms: Vec<f64>,
}

impl RequestOutcome {
    /// A terminal record with nothing observed beyond the clock and the count.
    ///
    /// The streaming, server-timing and concurrency facts are added with the
    /// builders below, so a caller that never observed one cannot accidentally
    /// supply a plausible value for it.
    #[must_use]
    pub fn new(issued_ms: f64, settled_ms: f64, outcome: Outcome, generated_tokens: u32) -> Self {
        Self {
            issued_ms,
            settled_ms,
            outcome,
            generated_tokens,
            prompt_tokens: 0,
            expected_tokens: None,
            ttft_ms: None,
            prefill_ms: None,
            in_flight_at_start: 0,
            token_times_ms: Vec::new(),
        }
    }

    /// A completed request. The common case.
    #[must_use]
    pub fn completed(issued_ms: f64, settled_ms: f64, generated_tokens: u32) -> Self {
        Self::new(issued_ms, settled_ms, Outcome::Completed, generated_tokens)
    }

    /// Record what the transport streamed (PP-27).
    #[must_use]
    pub fn streamed(mut self, ttft_ms: f64, token_times_ms: Vec<f64>) -> Self {
        self.ttft_ms = Some(ttft_ms);
        self.token_times_ms = token_times_ms;
        self
    }

    /// Record the server's prompt-token count and prefill duration (PP-2).
    #[must_use]
    pub fn server_prefill(mut self, prompt_tokens: u32, prefill_ms: f64) -> Self {
        self.prompt_tokens = prompt_tokens;
        self.prefill_ms = Some(prefill_ms);
        self
    }

    /// Record the server's prompt-token count without a prefill duration.
    #[must_use]
    pub fn with_prompt_tokens(mut self, prompt_tokens: u32) -> Self {
        self.prompt_tokens = prompt_tokens;
        self
    }

    /// Record the `n_predict` this request was issued with (PP-28).
    #[must_use]
    pub fn expecting(mut self, expected_tokens: u32) -> Self {
        self.expected_tokens = Some(expected_tokens);
        self
    }

    /// Record the client's in-flight count at issue (PP-8).
    #[must_use]
    pub fn in_flight(mut self, in_flight_at_start: u32) -> Self {
        self.in_flight_at_start = in_flight_at_start;
        self
    }

    /// Wall-clock duration of the request.
    #[must_use]
    pub fn duration_ms(&self) -> f64 {
        self.settled_ms - self.issued_ms
    }

    /// §3 — per-request `dec = (tokens − 1) / (last − first)`.
    /// `None` unless the transport streamed at least two tokens.
    #[must_use]
    pub fn decode_tok_per_sec(&self) -> Option<f64> {
        let (first, last) = (self.token_times_ms.first()?, self.token_times_ms.last()?);
        let span_s = (last - first) / 1000.0;
        let n = self.token_times_ms.len();
        if n < 2 || span_s <= 0.0 {
            return None;
        }
        Some((n as f64 - 1.0) / span_s)
    }

    /// §3 — this request's inter-token gaps, in milliseconds.
    #[must_use]
    pub fn itl_gaps_ms(&self) -> Vec<f64> {
        self.token_times_ms
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect()
    }

    /// PP-27 — this request's `ttft / e2e`. `None` without a first-token
    /// instant or a positive duration.
    #[must_use]
    pub fn ttft_over_e2e(&self) -> Option<f64> {
        let ttft = self.ttft_ms?;
        let e2e = self.duration_ms();
        if e2e <= 0.0 {
            return None;
        }
        Some(ttft / e2e)
    }

    /// The same record in the seconds-based shape the §4.3 estimators resample.
    #[must_use]
    pub fn to_sample(&self, index: usize, in_flight_fallback: u32) -> RequestSample {
        RequestSample {
            index,
            worker: 0,
            start_s: self.issued_ms / 1000.0,
            end_s: self.settled_ms / 1000.0,
            token_times_s: self.token_times_ms.iter().map(|t| t / 1000.0).collect(),
            generated_tokens: self.generated_tokens,
            prompt_tokens: self.prompt_tokens,
            outcome: self.outcome,
            in_flight_at_start: if self.in_flight_at_start == 0 {
                in_flight_fallback as usize
            } else {
                self.in_flight_at_start as usize
            },
            drained: false,
        }
    }

    /// PP-7 — the row this request contributes to the receipt's per-band
    /// `samples[]`. `token_times_ms` stays in the gzipped side file: it is the
    /// bulk of the payload and the receipt links it by digest instead.
    #[must_use]
    pub fn to_row(&self, index: usize) -> SampleRow {
        SampleRow {
            index,
            issued_ms: self.issued_ms,
            settled_ms: self.settled_ms,
            outcome: self.outcome,
            generated_tokens: self.generated_tokens,
            prompt_tokens: self.prompt_tokens,
            ttft_ms: self.ttft_ms,
            in_flight_at_start: self.in_flight_at_start,
        }
    }
}

/// PP-7 — one row of a band's `samples[]`, as it appears in the receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampleRow {
    /// Position in issue order.
    pub index: usize,
    /// Offset of the request start from the window opening.
    pub issued_ms: f64,
    /// Offset of the terminal state.
    pub settled_ms: f64,
    /// How it ended.
    pub outcome: Outcome,
    /// Server-reported completion tokens.
    pub generated_tokens: u32,
    /// Server-reported prompt tokens.
    pub prompt_tokens: u32,
    /// Time to first token, when the transport streamed.
    pub ttft_ms: Option<f64>,
    /// In-flight requests when this one was issued (PP-8).
    pub in_flight_at_start: u32,
}

/// Receipt-level facts a band needs in order to know its own status.
///
/// Passed in rather than read from a global, so `derive_at(2)` can render a v2
/// band under v2 rules and `derive` can render a v3 one under v3 rules in the
/// same test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandContext {
    /// The receipt's `schema_version`. PP-26's `INVALID-CORRECTNESS` rule and
    /// PP-4's metric-presence rule apply from 3 on; a version-2 receipt is
    /// historical and neither is applied to it.
    pub schema_version: u32,
    /// Replicates the cell ran (§4.3: fewer than five is `NONCONFORMANT`).
    pub replicates: u32,
    /// Whether those replicates alternated.
    pub interleaved: bool,
    /// PP-20 — the comparator pin expired before the run started.
    pub comparator_stale: bool,
    /// PP-27 threshold, from `perf-matrix.yaml`.
    pub stream_live_ttft_over_e2e_max: f64,
}

impl Default for BandContext {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            replicates: REPLICATES as u32,
            interleaved: INTERLEAVED,
            comparator_stale: false,
            stream_live_ttft_over_e2e_max: stream_live_ttft_over_e2e_max(),
        }
    }
}

impl BandContext {
    /// The context for a receipt at `schema_version`, everything else at the
    /// conformant value.
    #[must_use]
    pub fn at_schema_version(schema_version: u32) -> Self {
        Self {
            schema_version,
            ..Self::default()
        }
    }
}

/// One band's raw input: the window it ran, and every request it sampled.
#[derive(Debug, Clone, PartialEq)]
pub struct BandInput {
    /// Fixed concurrency `c`.
    pub concurrency: u32,
    /// `T` — the window close, in milliseconds from window open.
    pub window_ms: f64,
    /// Which replicate this band is, 1-based.
    pub replicate: u32,
    /// Every sampled request's terminal record.
    pub requests: Vec<RequestOutcome>,
    /// This cell's comparator posture.
    pub comparator: ComparatorStatus,
    /// PP-28 — the `n_predict` every retained sample was issued with. `None`
    /// only for a workload that does not pin one.
    pub n_predict: Option<u32>,
    /// PP-27 — what the server declared on the first chunk.
    pub stream_mode: Option<StreamMode>,
    /// PP-26 — the band's correctness witness.
    pub witness: Option<BatchInvarianceWitness>,
    /// PP-7 — the retained gzipped samples file, when one was written.
    pub samples_file: Option<SamplesFile>,
    /// PP-22 — the lane configuration that enters the join key.
    pub lane: LaneConfig,
    /// PP-26 — which lane measured this band.
    ///
    /// The batch-invariance witness is a claim about the **subject**: "did
    /// `apr serve` return the same tokens under batching as it did alone?" The
    /// comparator is the oracle the subject is measured against, and demanding
    /// a witness of it would make `llama-server` `INVALID-CORRECTNESS` for
    /// failing to be witnessed against itself. So a `Llama` band is exempt
    /// (see [`Self::invalid_correctness`]) and carries no witness at all.
    pub role: Lane,
    /// §4.4.2 — protocol departures the DRIVER observed while running this
    /// band (a shrunken window, a warmup that did not complete, a sample floor
    /// that was not met).
    ///
    /// Printed-only is not recorded: a violation the operator saw scroll past
    /// and the receipt did not carry is a receipt that reads conformant. Each
    /// entry sends the band to `NONCONFORMANT-VALID` and is named in
    /// `unproduced_fields`.
    pub conformance_violations: Vec<String>,
}

impl BandInput {
    /// A band with the minimum a measurement needs: its window, its requests
    /// and its comparator posture.
    ///
    /// Everything else — the correctness witness, the stream declaration, the
    /// `n_predict` pin, the lane configuration — is recorded with the builders
    /// below. A band that records none of them renders as
    /// `NONCONFORMANT-VALID` and says why, which is the honest posture for a
    /// run that observed none of them.
    #[must_use]
    pub fn new(
        concurrency: u32,
        window_ms: f64,
        requests: Vec<RequestOutcome>,
        comparator: ComparatorStatus,
    ) -> Self {
        Self {
            concurrency,
            window_ms,
            replicate: 1,
            requests,
            comparator,
            n_predict: None,
            stream_mode: None,
            witness: None,
            samples_file: None,
            lane: LaneConfig::default(),
            role: Lane::Apr,
            conformance_violations: Vec::new(),
        }
    }

    /// PP-26 — which lane measured this band. Defaults to [`Lane::Apr`], the
    /// subject; a comparator lane must say so, because saying nothing is what
    /// copied the subject's witness onto the oracle.
    #[must_use]
    pub fn role(mut self, role: Lane) -> Self {
        self.role = role;
        self
    }

    /// §4.4.2 — the driver's protocol departures for this band.
    #[must_use]
    pub fn conformance_violations(mut self, violations: Vec<String>) -> Self {
        self.conformance_violations = violations;
        self
    }

    /// Which replicate this band is, 1-based (§4.3).
    #[must_use]
    pub fn replicate(mut self, replicate: u32) -> Self {
        self.replicate = replicate;
        self
    }

    /// PP-28 — the `n_predict` every retained sample was issued with.
    #[must_use]
    pub fn n_predict(mut self, n_predict: u32) -> Self {
        self.n_predict = Some(n_predict);
        self
    }

    /// PP-27 — what the server declared on the first chunk.
    #[must_use]
    pub fn stream_mode(mut self, stream_mode: StreamMode) -> Self {
        self.stream_mode = Some(stream_mode);
        self
    }

    /// PP-26 — the band's correctness witness.
    #[must_use]
    pub fn witness(mut self, witness: BatchInvarianceWitness) -> Self {
        self.witness = Some(witness);
        self
    }

    /// PP-7 — the retained gzipped samples file.
    #[must_use]
    pub fn samples_file(mut self, samples_file: SamplesFile) -> Self {
        self.samples_file = Some(samples_file);
        self
    }

    /// PP-22 — the lane configuration that enters the join key.
    #[must_use]
    pub fn lane(mut self, lane: LaneConfig) -> Self {
        self.lane = lane;
        self
    }

    /// Derive every quantity from the sampled requests, under v3 rules.
    ///
    /// # Errors
    /// When the band contradicts its own clock: an empty band, a non-positive
    /// window, a request issued at or after `T` (PP-10), a settlement before
    /// its issue, a zero-token completion, an abandonment that did not happen
    /// during the drain, or a `Timeout`/`Failed` label that the request's own
    /// duration contradicts.
    pub fn derive(&self) -> Result<DerivedBand, String> {
        self.derive_in(&BandContext::default())
    }

    /// [`Self::derive`] under the rules of a given schema version.
    ///
    /// # Errors
    /// As [`Self::derive`].
    pub fn derive_at(&self, schema_version: u32) -> Result<DerivedBand, String> {
        self.derive_in(&BandContext::at_schema_version(schema_version))
    }

    /// [`Self::derive`] with the receipt-level facts spelled out.
    ///
    /// # Errors
    /// As [`Self::derive`].
    pub fn derive_in(&self, ctx: &BandContext) -> Result<DerivedBand, String> {
        self.validate()?;
        let drain_ms = self.drain_ms();
        let span_ms = self.span_ms();
        let tokens_total = self.tokens_total();
        let short_of_n_predict = self.short_of_n_predict();
        let stream_witness = self.stream_witness(ctx.stream_live_ttft_over_e2e_max);
        // PP-27: the verdict, not the declaration. An undeclared stream the
        // client measured as live IS live; a declared-live stream the client
        // measured as a replay is not.
        let stream_live = stream_witness.is_some_and(|w| w.verdict == StreamVerdict::Live);
        let invalid_correctness = self.invalid_correctness(ctx);

        let mut unproduced = Vec::new();
        let latency = if stream_live {
            Latency::from(self)
        } else {
            unproduced.push(self.stream_reason(stream_witness.as_ref()));
            Latency::none()
        };
        let mut prefill = self.prefill_tok_per_sec();
        if prefill.is_none() {
            unproduced.push(format!(
                "PP-4 c={}: prefill_tok_per_sec — no request carried a server-reported \
                 `timings.prompt_ms`, and a client-side prefill estimate is exactly the \
                 harness-inferred field PP-13 refuses",
                self.concurrency
            ));
        }
        let mut aggregate = Some(rate_per_sec(tokens_total as f64, span_ms));
        let mut latency = latency;
        if invalid_correctness {
            aggregate = None;
            prefill = None;
            latency.decode_tok_per_sec = None;
            unproduced.push(format!(
                "P-4 c={}: aggregate_tok_per_sec, decode_tok_per_sec and prefill_tok_per_sec — \
                 the band's batch-invariance witness (PP-26) is {} , so its throughput is not \
                 reported, not gated and never a baseline",
                self.concurrency,
                self.witness.as_ref().map_or_else(
                    || "absent".to_string(),
                    |w| format!("{:?}", w.batch_invariance)
                )
            ));
        }
        if short_of_n_predict > 0 {
            unproduced.push(format!(
                "PP-28 c={}: {short_of_n_predict} of {} completed requests did not reach \
                 n_predict — the sampler pin was not honoured, so this band is a record and not \
                 a baseline",
                self.concurrency,
                self.count(Outcome::Completed)
            ));
        }
        for violation in &self.conformance_violations {
            unproduced.push(format!(
                "§4.4.2 c={}: protocol violation observed by the driver — {violation}. The band \
                 is NONCONFORMANT-VALID: a record, cited, never a baseline.",
                self.concurrency
            ));
        }
        // PP-4: a band that reports numbers reports all three of them.
        let metrics_complete =
            aggregate.is_some() && latency.decode_tok_per_sec.is_some() && prefill.is_some();
        let status = self.status(
            ctx,
            invalid_correctness,
            stream_live,
            short_of_n_predict,
            metrics_complete,
        );
        if status == BandStatus::NonconformantValid {
            unproduced.push(format!(
                "§7.4 c={}: this band is NONCONFORMANT-VALID — a historical record, cited, never \
                 a baseline",
                self.concurrency
            ));
        }

        Ok(DerivedBand {
            concurrency: self.concurrency,
            replicate: self.replicate,
            window_ms: self.window_ms,
            drain_ms,
            suspect: self.suspect(drain_ms),
            requested: self.requests.len(),
            completed: self.count(Outcome::Completed),
            timeouts: self.count(Outcome::Timeout),
            truncated: self.count(Outcome::AbandonedAtDrain),
            errors: self.count(Outcome::Failed),
            short_of_n_predict,
            tokens_total,
            span_ms,
            aggregate_tok_per_sec: aggregate,
            decode_tok_per_sec: latency.decode_tok_per_sec,
            prefill_tok_per_sec: prefill,
            ttft_p50_ms: latency.ttft_p50_ms,
            ttft_p95_ms: latency.ttft_p95_ms,
            itl_p50_ms: latency.itl_p50_ms,
            itl_p95_ms: latency.itl_p95_ms,
            latencies_ms: self.latencies_ms(),
            samples: self.sample_rows(),
            samples_file: self.samples_file.clone(),
            stream_mode: self.stream_mode,
            stream_witness,
            witness: self.witness.clone(),
            status,
            join_key: None,
            run_id: None,
            unproduced,
            comparator: self.comparator.clone(),
        })
    }

    /// PP-3 / PP-22 / PP-5 — form the comparator posture for a subject band
    /// against a same-run comparator lane.
    ///
    /// This is the **only** constructor of [`ComparatorStatus::Measured`], and
    /// therefore the only way a [`Ratio`] enters a receipt.
    ///
    /// # Errors
    /// When the two lanes come from different runs (PP-3), when the join keys
    /// differ or either is a `-b 1` cripple (PP-22), when either lane timed out
    /// (PP-5), or when either lane fails to derive.
    pub fn join_status(
        subject: &Self,
        comparator: &Self,
        subject_key: &JoinKey,
        comparator_key: &JoinKey,
        run_ids: (&RunId, &RunId),
    ) -> Result<ComparatorStatus, String> {
        Self::join_status_in(
            subject,
            comparator,
            subject_key,
            comparator_key,
            run_ids,
            &BandContext::default(),
        )
    }

    /// [`Self::join_status`] with the receipt-level facts spelled out.
    ///
    /// # Errors
    /// As [`Self::join_status`].
    pub fn join_status_in(
        subject: &Self,
        comparator: &Self,
        subject_key: &JoinKey,
        comparator_key: &JoinKey,
        run_ids: (&RunId, &RunId),
        ctx: &BandContext,
    ) -> Result<ComparatorStatus, String> {
        let (subject_run, comparator_run) = run_ids;
        if subject_run != comparator_run {
            return Err(format!(
                "PP-3: the comparator lane is run_id {} and the subject is run_id {} — a ratio is \
                 representable only against a baseline from the SAME run; two runs saw two \
                 thermal states, two free-VRAM figures and two schedulers",
                comparator_run.as_str(),
                subject_run.as_str()
            ));
        }
        subject_key.refuse_mismatch(comparator_key)?;
        let subject_band = subject.derive_in(ctx)?;
        let comparator_band = comparator.derive_in(ctx)?;
        for (lane, band) in [("subject", &subject_band), ("comparator", &comparator_band)] {
            if band.timeouts > 0 {
                return Err(format!(
                    "PP-5: the {lane} lane at c={} recorded {} timeouts — a timed-out band cannot \
                     carry a ratio, because the requests that did not return are exactly the ones \
                     the ratio would have to account for",
                    band.concurrency, band.timeouts
                ));
            }
        }
        let ratios = ratios_of(subject, comparator, &subject_band, &comparator_band)?;
        Ok(ComparatorStatus::Measured(MeasuredJoin::sealed(
            comparator_band
                .with_run_id(comparator_run.clone())
                .with_join_key(comparator_key.clone()),
            ratios,
        )))
    }

    /// [`Self::join_status`], applied: the subject's derived band carrying the
    /// joined comparator.
    ///
    /// # Errors
    /// As [`Self::join_status`].
    pub fn join(
        subject: &Self,
        comparator: &Self,
        subject_key: &JoinKey,
        comparator_key: &JoinKey,
        run_ids: (&RunId, &RunId),
    ) -> Result<DerivedBand, String> {
        let status = Self::join_status(subject, comparator, subject_key, comparator_key, run_ids)?;
        let joined = Self {
            comparator: status,
            ..subject.clone()
        };
        Ok(joined
            .derive()?
            .with_run_id(run_ids.0.clone())
            .with_join_key(subject_key.clone()))
    }

    fn completed_iter(&self) -> impl Iterator<Item = &RequestOutcome> {
        self.requests
            .iter()
            .filter(|r| r.outcome == Outcome::Completed)
    }

    fn count(&self, outcome: Outcome) -> usize {
        self.requests
            .iter()
            .filter(|r| r.outcome == outcome)
            .count()
    }

    fn tokens_total(&self) -> u64 {
        self.completed_iter()
            .map(|r| u64::from(r.generated_tokens))
            .sum()
    }

    /// PP-28 — completed requests whose token count missed the pin.
    ///
    /// The per-request `expected_tokens` wins over the band's `n_predict` so a
    /// ragged workload (W2) is not counted short for being ragged. When neither
    /// is declared nothing is counted: a band that never pinned `n_predict`
    /// cannot be short of it, and it says so through its status instead.
    fn short_of_n_predict(&self) -> usize {
        self.completed_iter()
            .filter(|r| {
                r.expected_tokens
                    .or(self.n_predict)
                    .is_some_and(|want| r.generated_tokens != want)
            })
            .count()
    }

    /// PP-27 — the two halves of the streaming witness, resolved.
    ///
    /// | server declared | client ratio | verdict | source | latency metrics |
    /// |---|---|---|---|---|
    /// | `replayed` | anything | `Replayed` | server | withheld |
    /// | `live` | `<= live_max` | `Live` | server | produced |
    /// | `live` | `> live_max` | `Replayed` | client | withheld (disagreement) |
    /// | nothing | `<= live_max` | `Live` | client | produced |
    /// | nothing | `> live_max` | `Undeclared` | client | withheld |
    ///
    /// The fourth row is the one that changed: upstream `llama-server` never
    /// declares a mode, and treating that as "not live" made every comparator
    /// band `NONCONFORMANT-VALID` — no baseline could ever be conformant. The
    /// client's ratio measures the same fact and carries the verdict alone.
    ///
    /// The fifth row is deliberately NOT `Replayed`: nothing said the answer
    /// was pre-computed. It is "no evidence of a live stream", and it is not a
    /// pass either.
    fn stream_witness(&self, live_max: f64) -> Option<StreamWitness> {
        let ratios: Vec<f64> = self
            .completed_iter()
            .filter_map(RequestOutcome::ttft_over_e2e)
            .collect();
        let median = percentile(&sorted(ratios), 0.50)?;
        let client_live = median <= live_max;
        let (verdict, source) = match (self.stream_mode, client_live) {
            (Some(StreamMode::Replayed), _) => {
                (StreamVerdict::Replayed, StreamWitnessSource::Server)
            }
            (Some(StreamMode::Live), true) => (StreamVerdict::Live, StreamWitnessSource::Server),
            (Some(StreamMode::Live), false) => {
                (StreamVerdict::Replayed, StreamWitnessSource::Client)
            }
            (None, true) => (StreamVerdict::Live, StreamWitnessSource::Client),
            (None, false) => (StreamVerdict::Undeclared, StreamWitnessSource::Client),
        };
        Some(StreamWitness {
            client_ttft_over_e2e_median: median,
            verdict,
            source,
        })
    }

    fn stream_reason(&self, witness: Option<&StreamWitness>) -> String {
        let verdict = witness.map_or(StreamVerdict::Undeclared, |w| w.verdict);
        let observed = witness.map_or_else(
            || "no completed request reported a first-token instant".to_string(),
            |w| format!("median(ttft/e2e)={:.3}", w.client_ttft_over_e2e_median),
        );
        format!(
            "PP-27 c={}: decode_tok_per_sec, ttft_ms p50/p95 and itl_ms p50/p95 — stream verdict \
             {verdict:?} ({observed}); a latency computed off a replayed or undeclared stream is a \
             property of the replay, not of the server",
            self.concurrency
        )
    }

    /// P-4 / PP-26 — `c > 1` on the **subject** lane needs a passing
    /// batch-invariance witness. `c = 1` forms no batch and needs none, and the
    /// comparator lane is not the thing being witnessed.
    ///
    /// PP-26 asks whether `apr serve` returns the same tokens under batching as
    /// it does alone. The comparator is the ORACLE that question is asked
    /// against; requiring a witness of it would mark `llama-server`
    /// `INVALID-CORRECTNESS` for not being witnessed against itself, and — as
    /// the producer copied the subject's witness onto the comparator band — a
    /// subject-side PASS would have silently vouched for the oracle too.
    fn invalid_correctness(&self, ctx: &BandContext) -> bool {
        self.role == Lane::Apr
            && ctx.schema_version >= SCHEMA_VERSION
            && self.concurrency > 1
            && !self
                .witness
                .as_ref()
                .is_some_and(BatchInvarianceWitness::passed)
    }

    /// §7.4 — every applicable verdict, folded through
    /// [`BandStatus::stronger_of`].
    ///
    /// Each rule contributes a candidate and the precedence lives in one place
    /// ([`BandStatus::rank`]) rather than in the order of early returns, which
    /// is how `COMPARATOR_STALE` came to be stamped over `INVALID-CORRECTNESS`
    /// at render time.
    fn status(
        &self,
        ctx: &BandContext,
        invalid_correctness: bool,
        stream_live: bool,
        short_of_n_predict: usize,
        metrics_complete: bool,
    ) -> BandStatus {
        // PP-5 predates v3 and applies to every receipt; PP-27, PP-28, PP-4
        // and §4.3's replicate floor are v3 rules and are NOT applied
        // retroactively to a v2-dated receipt, which is historical either way
        // (`baseline_eligible` is false for anything but MEASURED).
        let v3 = ctx.schema_version >= SCHEMA_VERSION;
        let nonconformant = self.count(Outcome::Timeout) > 0
            || !self.conformance_violations.is_empty()
            || (v3
                && (!ctx.interleaved
                    || (ctx.replicates as usize) < MIN_REPLICATES
                    || !stream_live
                    || short_of_n_predict > 0
                    || !metrics_complete));
        let mut status = match self.comparator {
            ComparatorStatus::Measured(_) => BandStatus::Measured,
            ComparatorStatus::NotApplicable { .. } => BandStatus::Na,
            ComparatorStatus::Unmeasured { .. } => BandStatus::Unmeasured,
        };
        if nonconformant {
            status = status.stronger_of(BandStatus::NonconformantValid);
        }
        if ctx.comparator_stale {
            status = status.stronger_of(BandStatus::ComparatorStale);
        }
        if invalid_correctness {
            status = status.stronger_of(BandStatus::InvalidCorrectness);
        }
        status
    }

    /// PP-10 — last settlement of any request, minus `T`, clamped at 0.
    fn drain_ms(&self) -> f64 {
        let last = self
            .requests
            .iter()
            .map(|r| r.settled_ms)
            .fold(f64::NEG_INFINITY, f64::max);
        (last - self.window_ms).max(0.0)
    }

    /// §3 — last completion minus first request start.
    fn span_ms(&self) -> f64 {
        let first = self
            .requests
            .iter()
            .map(|r| r.issued_ms)
            .fold(f64::INFINITY, f64::min);
        let last = self
            .completed_iter()
            .map(|r| r.settled_ms)
            .fold(f64::NEG_INFINITY, f64::max);
        (last - first).max(0.0)
    }

    fn suspect(&self, drain_ms: f64) -> Vec<String> {
        if self.window_ms > 0.0 && drain_ms > DRAIN_SUSPECT_FRACTION * self.window_ms {
            return vec![format!(
                "SUSPECT PP-10 c={}: drain_ms={drain_ms:.1} > 0.5 x window_ms={:.1} — one \
                 request dominated the window; re-run this band with a longer window",
                self.concurrency, self.window_ms
            )];
        }
        Vec::new()
    }

    fn latencies_ms(&self) -> Vec<f64> {
        self.completed_iter()
            .map(RequestOutcome::duration_ms)
            .collect()
    }

    fn sample_rows(&self) -> Vec<SampleRow> {
        self.requests
            .iter()
            .enumerate()
            .map(|(i, r)| r.to_row(i))
            .collect()
    }

    /// The §4.3 request-unit shape of this band's completed requests.
    fn request_samples(&self) -> Vec<RequestSample> {
        self.requests
            .iter()
            .enumerate()
            .map(|(i, r)| r.to_sample(i, self.concurrency))
            .collect()
    }

    fn decode_median(&self) -> Option<f64> {
        let rates: Vec<f64> = self
            .completed_iter()
            .filter_map(RequestOutcome::decode_tok_per_sec)
            .collect();
        percentile(&sorted(rates), 0.50)
    }

    fn ttft_percentile(&self, p: f64) -> Option<f64> {
        let v: Vec<f64> = self.completed_iter().filter_map(|r| r.ttft_ms).collect();
        percentile(&sorted(v), p)
    }

    fn itl_percentile(&self, p: f64) -> Option<f64> {
        let v: Vec<f64> = self
            .completed_iter()
            .flat_map(RequestOutcome::itl_gaps_ms)
            .collect();
        percentile(&sorted(v), p)
    }

    /// §3 `prefill` — `Σ prompt_tokens / Σ prefill_ms` over the completed
    /// requests that carry a **server-reported** prefill duration.
    ///
    /// `None` when none of them do. There is no client-side fallback: PP-13
    /// makes a harness-computed value schema-fatal, and the whole point of
    /// naming `prefill_source: "server"` beside the number is that the reader
    /// can tell the difference.
    fn prefill_tok_per_sec(&self) -> Option<f64> {
        let mut tokens = 0_u64;
        let mut ms = 0.0_f64;
        for r in self.completed_iter() {
            if let Some(p) = r.prefill_ms {
                if p > 0.0 {
                    tokens += u64::from(r.prompt_tokens);
                    ms += p;
                }
            }
        }
        if ms <= 0.0 || tokens == 0 {
            return None;
        }
        Some(tokens as f64 / (ms / 1000.0))
    }

    fn validate(&self) -> Result<(), String> {
        if self.requests.is_empty() {
            return Err(format!(
                "band c={}: no sampled requests — a band over zero requests is a vacuous pass, \
                 not a measurement",
                self.concurrency
            ));
        }
        // NaN is caught explicitly: a negated `>` would let it through.
        if self.window_ms.is_nan() || self.window_ms <= 0.0 {
            return Err(format!(
                "band c={}: window_ms={} — the window must have positive length or `drain_ms` \
                 and the SUSPECT fraction are both undefined",
                self.concurrency, self.window_ms
            ));
        }
        for (i, r) in self.requests.iter().enumerate() {
            validate_request(self.concurrency, i, r, self.window_ms)?;
        }
        Ok(())
    }
}

/// The five streaming-only figures, produced or withheld together.
struct Latency {
    decode_tok_per_sec: Option<f64>,
    ttft_p50_ms: Option<f64>,
    ttft_p95_ms: Option<f64>,
    itl_p50_ms: Option<f64>,
    itl_p95_ms: Option<f64>,
}

impl Latency {
    fn from(band: &BandInput) -> Self {
        Self {
            decode_tok_per_sec: band.decode_median(),
            ttft_p50_ms: band.ttft_percentile(0.50),
            ttft_p95_ms: band.ttft_percentile(0.95),
            itl_p50_ms: band.itl_percentile(0.50),
            itl_p95_ms: band.itl_percentile(0.95),
        }
    }

    fn none() -> Self {
        Self {
            decode_tok_per_sec: None,
            ttft_p50_ms: None,
            ttft_p95_ms: None,
            itl_p50_ms: None,
            itl_p95_ms: None,
        }
    }
}

/// §4.3 — the three ratios, each by its own estimator.
fn ratios_of(
    subject: &BandInput,
    comparator: &BandInput,
    subject_band: &DerivedBand,
    comparator_band: &DerivedBand,
) -> Result<BandRatios, String> {
    let agg = window_ratio(
        subject_band.aggregate_tok_per_sec,
        comparator_band.aggregate_tok_per_sec,
    )
    .ok_or_else(|| {
        format!(
            "P-5: neither lane at c={} produced an aggregate throughput, so there is no agg ratio \
             to form",
            subject_band.concurrency
        )
    })?;
    // A ratio of two suppressed numbers is not a ratio. `dec` is a paired
    // bootstrap over the raw request samples, which exist whatever the band
    // decided about them — so without this guard a lane whose decode was
    // withheld as unreliable (a replayed stream, an unwitnessed batch) still
    // contributed a `dec` ratio computed from exactly the samples the band
    // refused to report. `prefill` and `agg` divide the DERIVED figures and are
    // already `None` when either lane withheld one; `dec` has to be told.
    let dec = if subject_band.decode_tok_per_sec.is_some()
        && comparator_band.decode_tok_per_sec.is_some()
    {
        paired_ratio_lcb(
            &subject.request_samples(),
            &comparator.request_samples(),
            median_decode_tok_s,
            VERDICT_CONFIDENCE,
        )
    } else {
        None
    };
    let prefill = window_ratio(
        subject_band.prefill_tok_per_sec,
        comparator_band.prefill_tok_per_sec,
    );
    Ok(BandRatios { agg, dec, prefill })
}

/// §4.3 replicate unit, from a single replicate: the point estimate with an
/// explicitly absent bound. One replicate bounds no variance.
fn window_ratio(subject: Option<f64>, comparator: Option<f64>) -> Option<Ratio> {
    let (s, c) = (subject?, comparator?);
    if c <= 0.0 {
        return None;
    }
    Some(Ratio::reporting_only(
        s / c,
        RatioMethod::ReplicateTLower,
        1,
    ))
}

/// Everything §5.1, §3 and §7.4 derive from a [`BandInput`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivedBand {
    /// Fixed concurrency `c`.
    pub concurrency: u32,
    /// Which replicate this band is, 1-based.
    pub replicate: u32,
    /// `T`, in milliseconds.
    pub window_ms: f64,
    /// PP-10 — length of the drain phase, in milliseconds.
    pub drain_ms: f64,
    /// PP-10 `SUSPECT` annotations; empty when the band is clean.
    pub suspect: Vec<String>,
    /// Requests issued before `T`.
    pub requested: usize,
    /// Requests that returned a usable response.
    pub completed: usize,
    /// Requests that reached the 120 s hard timeout.
    pub timeouts: usize,
    /// Requests abandoned at the drain deadline (PP-10 sense).
    pub truncated: usize,
    /// Requests that failed for any other reason.
    pub errors: usize,
    /// PP-28 — completed requests that did not reach `n_predict`.
    pub short_of_n_predict: usize,
    /// Σ generated tokens over completed requests — `agg`'s numerator.
    pub tokens_total: u64,
    /// `agg`'s denominator: last completion − first request start, in ms.
    pub span_ms: f64,
    /// §3 `agg` — wall-clock aggregate throughput. `None` on an
    /// `INVALID-CORRECTNESS` band, which reports no throughput at all.
    pub aggregate_tok_per_sec: Option<f64>,
    /// §3 `dec` — median per-request decode rate. `None` without a live stream.
    pub decode_tok_per_sec: Option<f64>,
    /// §3 `prefill` — server-reported. `None` without server timings.
    pub prefill_tok_per_sec: Option<f64>,
    /// p50 time-to-first-token. `None` without a live stream.
    pub ttft_p50_ms: Option<f64>,
    /// p95 time-to-first-token. `None` without a live stream.
    pub ttft_p95_ms: Option<f64>,
    /// p50 of the pooled inter-token gaps. `None` without a live stream.
    pub itl_p50_ms: Option<f64>,
    /// p95 of the pooled inter-token gaps. `None` without a live stream.
    pub itl_p95_ms: Option<f64>,
    /// Per-request end-to-end latencies of completed requests (PP-7).
    pub latencies_ms: Vec<f64>,
    /// PP-7 — the per-request rows this band carries in the receipt.
    pub samples: Vec<SampleRow>,
    /// PP-7 — the gzipped side file the rows' token times went to.
    pub samples_file: Option<SamplesFile>,
    /// PP-27 — what the server declared.
    pub stream_mode: Option<StreamMode>,
    /// PP-27 — what the client independently observed.
    pub stream_witness: Option<StreamWitness>,
    /// PP-26 — the correctness witness.
    pub witness: Option<BatchInvarianceWitness>,
    /// §7.4 — the status this band ended up with.
    pub status: BandStatus,
    /// PP-22 — the key this band joins on. Set at render time.
    pub join_key: Option<JoinKey>,
    /// PP-3 — the run this band belongs to. Set at render time.
    pub run_id: Option<RunId>,
    /// Fields this client could not produce, each with its reason. Never
    /// silently omitted and never filled with a plausible number.
    pub unproduced: Vec<String>,
    /// This cell's comparator posture.
    pub comparator: ComparatorStatus,
}

impl DerivedBand {
    /// Attach the PP-22 join key. Done at render time, where the receipt-level
    /// half of the key (host, model, protocol) is in scope.
    #[must_use]
    pub fn with_join_key(mut self, key: JoinKey) -> Self {
        self.join_key = Some(key);
        self
    }

    /// Attach the PP-3 run id.
    #[must_use]
    pub fn with_run_id(mut self, run_id: RunId) -> Self {
        self.run_id = Some(run_id);
        self
    }

    /// PP-20 — mark the band `COMPARATOR_STALE`. Applied at render time,
    /// because the pin expiry lives in the receipt's provenance and not in any
    /// band.
    ///
    /// Folded through [`BandStatus::stronger_of`], never assigned. An
    /// assignment here **overwrote** `INVALID-CORRECTNESS`: a `c > 1` band with
    /// no witness reported no throughput at all, and came out of the render
    /// labelled as though its only defect were an expired comparator pin. §7.4
    /// puts correctness first, and there is now one definition of that order.
    #[must_use]
    pub fn marked_comparator_stale(mut self, pin_expiry: &str, started_utc: &str) -> Self {
        self.status = self.status.stronger_of(BandStatus::ComparatorStale);
        self.unproduced.push(format!(
            "PP-20 c={}: the comparator pin expired {pin_expiry}, before this run started \
             {started_utc} — every ratio on this band is COMPARATOR_STALE and blocks MEASURED \
             until the pin is refreshed",
            self.concurrency
        ));
        self
    }

    /// §7.4 — may this band be a comparator baseline?
    #[must_use]
    pub fn baseline_eligible(&self) -> bool {
        self.status.baseline_eligible()
    }
}

/// Every per-request rule, applied to one record.
fn validate_request(c: u32, i: usize, r: &RequestOutcome, window_ms: f64) -> Result<(), String> {
    let at = format!("band c={c} request[{i}]");
    if r.issued_ms >= window_ms {
        return Err(format!(
            "{at}: issued_ms={} >= T={window_ms} — PP-10: no request is issued at or after the \
             window close, and its tokens are never counted",
            r.issued_ms
        ));
    }
    if r.settled_ms < r.issued_ms {
        return Err(format!(
            "{at}: settled_ms={} precedes issued_ms={}",
            r.settled_ms, r.issued_ms
        ));
    }
    validate_outcome(&at, r, window_ms)
}

/// The label a request carries must agree with the clock.
fn validate_outcome(at: &str, r: &RequestOutcome, window_ms: f64) -> Result<(), String> {
    let d = r.duration_ms();
    match r.outcome {
        Outcome::Completed if r.generated_tokens == 0 => Err(format!(
            "{at}: completed with zero generated tokens — a zero-token response is a failure, not \
             a fast request"
        )),
        Outcome::Timeout if d < REQUEST_TIMEOUT_MS => Err(format!(
            "{at}: labelled Timeout but ran {d:.1} ms < the {REQUEST_TIMEOUT_MS} ms hard timeout \
             (§3) — that is a Failed, and the two are separate counters"
        )),
        Outcome::Failed if d >= REQUEST_TIMEOUT_MS => Err(format!(
            "{at}: labelled Failed but ran {d:.1} ms >= the {REQUEST_TIMEOUT_MS} ms hard timeout \
             — that is a Timeout, which PP-5 makes fatal to this band's ratio"
        )),
        Outcome::AbandonedAtDrain if r.settled_ms < window_ms => Err(format!(
            "{at}: labelled AbandonedAtDrain but settled at {}, before T={window_ms} — a request \
             can only be abandoned during the drain",
            r.settled_ms
        )),
        _ => Ok(()),
    }
}

fn rate_per_sec(count: f64, span_ms: f64) -> f64 {
    if span_ms <= 0.0 {
        return 0.0;
    }
    count / (span_ms / 1000.0)
}

fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// Percentile by linear interpolation between order statistics.
///
/// Returns `None` for an empty slice: a percentile of nothing is undefined, and
/// returning `0.0` there is how an empty band reads as an instant one.
#[must_use]
pub fn percentile(sorted_ascending: &[f64], p: f64) -> Option<f64> {
    match sorted_ascending.len() {
        0 => None,
        1 => Some(sorted_ascending[0]),
        n => {
            let idx = (n as f64 - 1.0) * p;
            let lo = idx.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            let frac = idx - lo as f64;
            Some(sorted_ascending[lo].mul_add(1.0 - frac, sorted_ascending[hi] * frac))
        }
    }
}

#[cfg(test)]
mod tests {
    // The `<selftest-name>__<sentence>` spelling is load-bearing: PP-29's
    // `scripts/spec_conformance.sh` joins the §6 invariant table to the test
    // list on the prefix before the double underscore.
    #![allow(non_snake_case)]
    use super::*;
    use crate::perf_gate::receipt::{TokenCountingMethod, Workload};
    use crate::perf_gate::witness::BatchInvariance;

    /// A completed request that generated `tokens`, issued at `issued_ms` and
    /// settling `dur_ms` later. Non-streaming: no ttft, no token times.
    fn done(issued_ms: f64, dur_ms: f64, tokens: u32) -> RequestOutcome {
        RequestOutcome::completed(issued_ms, issued_ms + dur_ms, tokens)
    }

    /// The same, streamed live: the first token lands early and the rest are
    /// spread over the request, so `ttft/e2e` is far below the live threshold.
    fn streamed(issued_ms: f64, dur_ms: f64, tokens: u32) -> RequestOutcome {
        let ttft = dur_ms * 0.08;
        let times: Vec<f64> = (0..tokens)
            .map(|k| issued_ms + ttft + f64::from(k) * (dur_ms - ttft) / f64::from(tokens))
            .collect();
        done(issued_ms, dur_ms, tokens)
            .streamed(ttft, times)
            .server_prefill(512, dur_ms * 0.05)
    }

    fn unmeasured() -> ComparatorStatus {
        ComparatorStatus::unmeasured("perf-gate", "no comparator lane on this cell yet (PP-25)")
    }

    /// A band at `c = 1`. A single stream forms no batch, so PP-26's witness is
    /// out of scope here and these tests exercise the drain and counter rules
    /// on their own; the correctness rules have `conformant_band(c > 1)`.
    fn band(window_ms: f64, requests: Vec<RequestOutcome>) -> BandInput {
        BandInput::new(1, window_ms, requests, unmeasured())
    }

    fn passing_witness() -> BatchInvarianceWitness {
        let tokens: Vec<u32> = (0..128).collect();
        BatchInvarianceWitness::compare(&tokens, &tokens, 64).formed_at(4, "perf041")
    }

    /// A band that satisfies every v3 rule, so status derivation can be tested
    /// one departure at a time.
    fn conformant_band(concurrency: u32) -> BandInput {
        let requests: Vec<RequestOutcome> = (0..8)
            .map(|i| streamed(f64::from(i) * 100.0, 90.0 + f64::from(i), 128))
            .collect();
        BandInput::new(concurrency, 1000.0, requests, unmeasured())
            .n_predict(128)
            .stream_mode(StreamMode::Live)
            .witness(passing_witness())
    }

    /// THE POINT OF THE TICKET, half one: nothing in flight at `T` means the
    /// drain phase had zero length. `0.0` here is a measurement, not a default.
    #[test]
    fn drain_ms_is_zero_when_nothing_straddles_the_window_close() {
        let d = band(1000.0, vec![done(0.0, 100.0, 128), done(200.0, 100.0, 128)])
            .derive()
            .expect("valid band");
        assert_eq!(d.drain_ms, 0.0);
        assert!(d.suspect.is_empty(), "{:?}", d.suspect);
    }

    /// THE POINT OF THE TICKET, half two: the SAME code path over a band whose
    /// last request ran past `T` yields a DIFFERENT, non-zero `drain_ms`. A
    /// defaulted field cannot do this.
    #[test]
    fn drain_ms_varies_with_actual_drain_behaviour() {
        let quiet = band(1000.0, vec![done(0.0, 100.0, 128), done(900.0, 50.0, 128)])
            .derive()
            .expect("valid band");
        let straggler = band(1000.0, vec![done(0.0, 100.0, 128), done(900.0, 350.0, 128)])
            .derive()
            .expect("valid band");
        assert_eq!(quiet.drain_ms, 0.0);
        assert!((straggler.drain_ms - 250.0).abs() < 1e-9, "{straggler:?}");
        assert_ne!(quiet.drain_ms, straggler.drain_ms);
    }

    /// PP-10 — `drain_ms > 0.5 x window` is annotated SUSPECT.
    #[test]
    fn a_dominating_request_is_annotated_suspect() {
        let d = band(1000.0, vec![done(0.0, 50.0, 128), done(900.0, 700.0, 128)])
            .derive()
            .expect("valid band");
        assert!((d.drain_ms - 600.0).abs() < 1e-9, "{d:?}");
        assert_eq!(d.suspect.len(), 1, "{:?}", d.suspect);
        assert!(d.suspect[0].contains("drain_ms"), "{:?}", d.suspect);
    }

    /// And the annotation discriminates: just under the fraction stays clean.
    #[test]
    fn a_drain_just_under_half_the_window_is_not_suspect() {
        let d = band(1000.0, vec![done(0.0, 50.0, 128), done(900.0, 599.0, 128)])
            .derive()
            .expect("valid band");
        assert!((d.drain_ms - 499.0).abs() < 1e-9, "{d:?}");
        assert!(d.suspect.is_empty(), "{:?}", d.suspect);
    }

    /// PP-10, the registered mutation: "issue one request after `T` and count
    /// its tokens". The band is refused, so the tokens can never be counted.
    #[test]
    fn a_request_issued_at_or_after_t_is_refused() {
        let at_t = band(1000.0, vec![done(0.0, 10.0, 8), done(1000.0, 10.0, 8)]).derive();
        let after_t = band(1000.0, vec![done(0.0, 10.0, 8), done(1500.0, 10.0, 8)]).derive();
        for (label, got) in [("at T", at_t), ("after T", after_t)] {
            let err = got.expect_err(label);
            assert!(err.contains("PP-10"), "{label}: {err}");
        }
    }

    /// THE CONFLATION THIS TICKET EXISTS TO PREVENT. Every W1 request stops at
    /// `n_predict = 128` with EOS ignored, i.e. every one has
    /// `finish_reason == "length"`. Read in the finish-reason sense, `truncated`
    /// would be 8 and `agg`'s "completed, non-truncated" numerator would be
    /// EMPTY. In the drain sense it is 0 and the numerator is 1024 tokens.
    #[test]
    fn max_tokens_truncation_is_not_drain_truncation() {
        let reqs: Vec<RequestOutcome> = (0..8)
            .map(|i| done(f64::from(i) * 100.0, 90.0, 128))
            .collect();
        let d = band(1000.0, reqs).derive().expect("valid band");
        assert_eq!(
            d.truncated, 0,
            "no request was abandoned at the drain deadline"
        );
        assert_eq!(d.completed, 8);
        assert_eq!(d.tokens_total, 1024, "the numerator must not be emptied");
        assert!(
            d.aggregate_tok_per_sec.expect("agg") > 0.0,
            "{:?}",
            d.aggregate_tok_per_sec
        );
    }

    /// The drain sense: a request still running at the drain deadline.
    #[test]
    fn an_abandoned_request_increments_truncated_not_completed() {
        let abandoned = RequestOutcome::new(900.0, 1400.0, Outcome::AbandonedAtDrain, 12);
        let d = band(1000.0, vec![done(0.0, 100.0, 128), abandoned])
            .derive()
            .expect("valid band");
        assert_eq!(d.truncated, 1);
        assert_eq!(d.completed, 1);
        assert_eq!(
            d.tokens_total, 128,
            "an abandoned request contributes no tokens"
        );
        assert!((d.drain_ms - 400.0).abs() < 1e-9, "{d:?}");
    }

    /// An abandonment that happened before `T` is a contradiction, not a count.
    #[test]
    fn an_abandonment_before_the_window_close_is_refused() {
        let bogus = RequestOutcome::new(100.0, 200.0, Outcome::AbandonedAtDrain, 1);
        let err = band(1000.0, vec![done(0.0, 10.0, 8), bogus])
            .derive()
            .expect_err("settled before T");
        assert!(err.contains("only be abandoned during the drain"), "{err}");
    }

    /// §3 — `timeouts` is its own counter, and the label is checked against
    /// the clock rather than trusted.
    #[test]
    fn timeouts_and_failures_are_separate_and_both_are_verified() {
        let timeout = RequestOutcome::new(10.0, 10.0 + REQUEST_TIMEOUT_MS, Outcome::Timeout, 0);
        let failure = RequestOutcome::new(20.0, 45.0, Outcome::Failed, 0);
        let with_both = band(
            1000.0,
            vec![done(0.0, 10.0, 8), timeout.clone(), failure.clone()],
        );
        let d = with_both.derive().expect("valid band");
        assert_eq!(d.timeouts, 1);
        assert_eq!(d.errors, 1);
        assert_eq!(
            d.requested,
            d.completed + d.timeouts + d.truncated + d.errors,
            "the four counters must partition the requests"
        );
        assert_eq!(
            d.status,
            BandStatus::NonconformantValid,
            "PP-5: a band that timed out is a record, at every schema version"
        );
        assert_eq!(
            with_both.derive_at(2).expect("renders").status,
            BandStatus::NonconformantValid,
            "…including a v2-dated one"
        );
    }

    /// A `Timeout` that did not reach 120 s is a mislabelled failure. PP-5 makes
    /// `timeouts > 0` fatal to a band's ratio, so the label must be earned.
    #[test]
    fn a_short_request_cannot_be_labelled_a_timeout() {
        let liar = RequestOutcome::new(0.0, 50.0, Outcome::Timeout, 0);
        let err = band(1000.0, vec![done(500.0, 10.0, 8), liar])
            .derive()
            .expect_err("50 ms is not a timeout");
        assert!(err.contains("hard timeout"), "{err}");
    }

    /// And the converse: a request that ran past the hard timeout is a timeout,
    /// not a generic error that would leave `timeouts == 0`.
    #[test]
    fn an_over_long_request_cannot_be_labelled_a_plain_failure() {
        let liar = RequestOutcome::new(0.0, REQUEST_TIMEOUT_MS + 1.0, Outcome::Failed, 0);
        let err = band(1000.0, vec![done(500.0, 10.0, 8), liar])
            .derive()
            .expect_err("past the hard timeout");
        assert!(err.contains("PP-5"), "{err}");
    }

    /// A zero-token response is a failure, not a fast request.
    #[test]
    fn a_zero_token_completion_is_refused() {
        let err = band(1000.0, vec![done(0.0, 10.0, 0)])
            .derive()
            .expect_err("zero tokens");
        assert!(err.contains("zero-token"), "{err}");
    }

    #[test]
    fn an_empty_band_is_refused() {
        let err = band(1000.0, Vec::new()).derive().expect_err("no requests");
        assert!(err.contains("vacuous"), "{err}");
    }

    #[test]
    fn a_non_positive_window_is_refused() {
        let err = band(0.0, vec![done(-10.0, 5.0, 8)])
            .derive()
            .expect_err("zero window");
        assert!(err.contains("window_ms"), "{err}");
    }

    /// §3 — the denominator is wall-clock (last completion − first start),
    /// never the mean of per-request rates.
    #[test]
    fn aggregate_is_wall_clock_over_the_whole_span() {
        // Two requests, 100 tokens each, spanning 0 -> 2000 ms.
        let d = band(
            2500.0,
            vec![done(0.0, 500.0, 100), done(1000.0, 1000.0, 100)],
        )
        .derive()
        .expect("valid band");
        assert!((d.span_ms - 2000.0).abs() < 1e-9, "{d:?}");
        assert!(
            (d.aggregate_tok_per_sec.expect("agg") - 100.0).abs() < 1e-9,
            "200 tokens over 2 s = 100 tok/s, got {:?}",
            d.aggregate_tok_per_sec
        );
    }

    /// A non-streaming client cannot observe TTFT or ITL. It says so instead of
    /// emitting a plausible number — and at v3 the band is NONCONFORMANT-VALID,
    /// because PP-27 requires streaming and PP-4 requires `dec`.
    #[test]
    fn a_non_streaming_band_names_what_it_could_not_produce() {
        let d = band(1000.0, vec![done(0.0, 100.0, 128)])
            .derive()
            .expect("valid band");
        assert_eq!(d.ttft_p50_ms, None);
        assert_eq!(d.itl_p95_ms, None);
        assert_eq!(d.decode_tok_per_sec, None);
        assert_eq!(d.prefill_tok_per_sec, None);
        assert_eq!(d.status, BandStatus::NonconformantValid);
        let notes = d.unproduced.join("\n");
        assert!(notes.contains("PP-27"), "{notes}");
        assert!(notes.contains("PP-4"), "{notes}");
    }

    /// A live-streaming client produces them, and then only the receipt-level
    /// notes remain.
    #[test]
    fn a_streaming_band_produces_ttft_itl_and_decode() {
        let one = RequestOutcome::completed(0.0, 500.0, 5)
            .streamed(100.0, vec![100.0, 200.0, 300.0, 400.0, 500.0])
            .server_prefill(512, 90.0);
        let d = BandInput::new(1, 1000.0, vec![one], unmeasured())
            .stream_mode(StreamMode::Live)
            .n_predict(5)
            .derive()
            .expect("valid band");
        assert_eq!(d.ttft_p50_ms, Some(100.0));
        assert_eq!(d.itl_p50_ms, Some(100.0));
        // (5 - 1) tokens over (500 - 100) ms = 10 tok/s.
        assert_eq!(d.decode_tok_per_sec, Some(10.0));
        // 512 prompt tokens over 90 ms.
        assert!(
            (d.prefill_tok_per_sec.expect("prefill") - 512.0 / 0.09).abs() < 1e-6,
            "{:?}",
            d.prefill_tok_per_sec
        );
        assert!(d.unproduced.is_empty(), "{:?}", d.unproduced);
        assert_eq!(d.status, BandStatus::Unmeasured, "no comparator lane");
    }

    #[test]
    fn percentile_of_nothing_is_undefined_not_zero() {
        assert_eq!(percentile(&[], 0.5), None);
        assert_eq!(percentile(&[7.0], 0.95), Some(7.0));
        assert_eq!(percentile(&[0.0, 10.0], 0.5), Some(5.0));
    }

    #[test]
    fn comparator_status_renders_the_token_the_gate_reads() {
        assert_eq!(unmeasured().wire_token(), "UNMEASURED");
        let na = ComparatorStatus::not_applicable("perf-matrix.yaml", "vLLM has no aarch64 build");
        assert_eq!(na.wire_token(), "NOT_APPLICABLE");
    }

    /// §7.4 — the closed vocabulary, spelled exactly. Two of the six are not
    /// their variant name, which is why the table is written out.
    #[test]
    fn status_tokens_are_exactly_the_section_7_4_vocabulary() {
        let table = [
            (BandStatus::Measured, "MEASURED"),
            (BandStatus::Unmeasured, "UNMEASURED"),
            (BandStatus::Na, "NA"),
            (BandStatus::InvalidCorrectness, "INVALID-CORRECTNESS"),
            (BandStatus::NonconformantValid, "NONCONFORMANT-VALID"),
            (BandStatus::ComparatorStale, "COMPARATOR_STALE"),
        ];
        assert_eq!(table.len(), BandStatus::vocabulary().len());
        for (status, token) in table {
            assert_eq!(status.wire_token(), token);
            assert!(
                BandStatus::vocabulary().contains(&status),
                "{token} missing from the vocabulary"
            );
        }
        assert_ne!(
            BandStatus::Na.wire_token(),
            "NOT_APPLICABLE",
            "§7.4 spells it NA; NOT_APPLICABLE is the legacy comparator_status token"
        );
        assert!(BandStatus::Measured.baseline_eligible());
        for s in BandStatus::vocabulary() {
            if s != BandStatus::Measured {
                assert!(!s.baseline_eligible(), "{s:?} may not be a baseline");
            }
        }
    }

    /// PP-28 must-fire: a completed sample that stopped short of `n_predict`.
    #[test]
    fn a_completed_sample_short_of_n_predict_is_counted() {
        let mut b = conformant_band(1);
        b.requests[3].generated_tokens = 67;
        let d = b.derive().expect("the band still renders");
        assert_eq!(d.short_of_n_predict, 1);
        assert_eq!(d.status, BandStatus::NonconformantValid);
        assert!(
            d.aggregate_tok_per_sec.is_some(),
            "the evidence still renders; PP-28 is not fatal to the receipt"
        );
        let notes = d.unproduced.join("\n");
        assert!(notes.contains("PP-28"), "{notes}");
    }

    /// PP-28 must-not-fire: every retained sample at `n_predict`.
    #[test]
    fn thirty_of_thirty_at_n_predict_pass() {
        let requests: Vec<RequestOutcome> = (0..30)
            .map(|i| streamed(f64::from(i) * 30.0, 90.0 + f64::from(i), 128))
            .collect();
        let d = BandInput::new(1, 1000.0, requests, unmeasured())
            .n_predict(128)
            .stream_mode(StreamMode::Live)
            .derive()
            .expect("valid band");
        assert_eq!(d.short_of_n_predict, 0);
        assert_eq!(d.completed, 30);
        assert_eq!(d.status, BandStatus::Unmeasured);
    }

    /// A band with short samples is a record, never a baseline.
    #[test]
    fn a_band_with_short_samples_is_nonconformant() {
        let mut b = conformant_band(4);
        for r in &mut b.requests {
            r.generated_tokens = 112;
        }
        let d = b.derive().expect("renders");
        assert_eq!(d.short_of_n_predict, 8);
        assert_eq!(d.status, BandStatus::NonconformantValid);
        assert!(!d.baseline_eligible());
    }

    /// A per-request `expected_tokens` overrides the band's pin, so a ragged
    /// workload is not counted short for being ragged.
    #[test]
    fn a_per_request_expectation_overrides_the_band_pin() {
        let mut b = conformant_band(1);
        b.requests[0].generated_tokens = 64;
        assert_eq!(b.derive().expect("renders").short_of_n_predict, 1);
        b.requests[0] = b.requests[0].clone().expecting(64);
        assert_eq!(b.derive().expect("renders").short_of_n_predict, 0);
    }

    /// PP-27 must-fire: a replayed stream withholds every latency metric.
    #[test]
    fn a_replayed_stream_sends_latency_to_unproduced() {
        let d = conformant_band(1)
            .stream_mode(StreamMode::Replayed)
            .derive()
            .expect("renders");
        assert_eq!(d.decode_tok_per_sec, None);
        assert_eq!(d.ttft_p95_ms, None);
        assert_eq!(d.itl_p95_ms, None);
        assert_eq!(
            d.stream_witness.expect("witness").verdict,
            StreamVerdict::Replayed
        );
        assert_eq!(d.status, BandStatus::NonconformantValid);
    }

    /// And the client's half can overrule a server that SAYS live: a stream
    /// whose first token arrives with the last one is a replay however it is
    /// labelled.
    #[test]
    fn a_server_claiming_live_is_overruled_by_the_client_witness() {
        let late = RequestOutcome::completed(0.0, 500.0, 4)
            .streamed(499.0, vec![499.0, 499.5, 499.8, 500.0])
            .server_prefill(512, 40.0);
        let d = BandInput::new(1, 1000.0, vec![late], unmeasured())
            .stream_mode(StreamMode::Live)
            .n_predict(4)
            .derive()
            .expect("renders");
        let w = d.stream_witness.expect("witness");
        assert!(w.client_ttft_over_e2e_median > 0.95, "{w:?}");
        assert_eq!(w.verdict, StreamVerdict::Replayed);
        assert_eq!(d.decode_tok_per_sec, None);
    }

    /// PP-27's threshold is an exclusive one: a ratio exactly at
    /// `stream.live_ttft_over_e2e_max` is still live, one hair above is a
    /// replay. Without this the `>` could be a `>=` and nothing would notice.
    #[test]
    fn the_stream_threshold_is_exclusive_at_the_declared_maximum() {
        let ctx = BandContext {
            stream_live_ttft_over_e2e_max: 0.95,
            ..BandContext::default()
        };
        let at_threshold = |ratio: f64| {
            let e2e = 1000.0;
            let ttft = ratio * e2e;
            let one = RequestOutcome::completed(0.0, e2e, 4)
                .streamed(ttft, vec![ttft, ttft + 10.0, ttft + 20.0, ttft + 30.0])
                .server_prefill(512, 40.0);
            BandInput::new(1, 2_000.0, vec![one], unmeasured())
                .stream_mode(StreamMode::Live)
                .n_predict(4)
                .derive_in(&ctx)
                .expect("renders")
        };
        assert_eq!(
            at_threshold(0.95).stream_witness.expect("witness").verdict,
            StreamVerdict::Live,
            "exactly at the maximum is still live"
        );
        assert_eq!(
            at_threshold(0.951).stream_witness.expect("witness").verdict,
            StreamVerdict::Replayed
        );
    }

    /// PP-27, the rule as it now stands: a server that declares nothing does
    /// **not** thereby make its band nonconformant. Upstream `llama-server`
    /// declares no `stream_mode` and is not going to; reading its silence as
    /// "not live" made every comparator band `NONCONFORMANT-VALID`, so no
    /// baseline could ever be conformant and the parity arm could not reach a
    /// verdict — a rule about a field the oracle does not emit.
    ///
    /// The client's `median(ttft / e2e)` measures the same fact and carries the
    /// verdict alone: `Live`, sourced `Client`, with `stream_mode` still null on
    /// the wire because the server really did declare nothing.
    #[test]
    fn an_undeclared_stream_the_client_measured_as_live_is_live() {
        let d = BandInput::new(1, 1000.0, conformant_band(1).requests, unmeasured())
            .n_predict(128)
            .derive()
            .expect("renders");
        let w = d.stream_witness.expect("witness");
        assert_eq!(w.verdict, StreamVerdict::Live);
        assert_eq!(
            w.source,
            StreamWitnessSource::Client,
            "the server said nothing"
        );
        assert_eq!(d.stream_mode, None, "and the receipt still says so");
        assert!(d.decode_tok_per_sec.is_some(), "a live stream has a dec");
        assert_eq!(
            d.status,
            BandStatus::Unmeasured,
            "no comparator lane, but conformant"
        );
    }

    /// The other polarity: silence plus a client ratio that does NOT establish
    /// liveness is `Undeclared` — not `Replayed`, because nothing said the
    /// answer was pre-computed, and not a pass either. Every latency metric is
    /// withheld.
    #[test]
    fn an_undeclared_stream_the_client_cannot_call_live_is_undeclared() {
        // Every token arrives with the last one: ttft/e2e ≈ 1.
        let requests: Vec<RequestOutcome> = (0..6)
            .map(|i| {
                let issued = f64::from(i) * 10.0;
                RequestOutcome::completed(issued, issued + 100.0 + f64::from(i), 128)
                    .streamed(99.0, vec![issued + 99.0, issued + 99.5, issued + 100.0])
            })
            .collect();
        let d = BandInput::new(1, 1000.0, requests, unmeasured())
            .n_predict(128)
            .derive()
            .expect("renders");
        let w = d.stream_witness.expect("witness");
        assert_eq!(w.verdict, StreamVerdict::Undeclared);
        assert_eq!(w.source, StreamWitnessSource::Client);
        assert_eq!(d.decode_tok_per_sec, None);
        assert_eq!(d.ttft_p50_ms, None);
        assert_eq!(d.itl_p95_ms, None);
        assert_eq!(d.status, BandStatus::NonconformantValid);
    }

    /// PP-26 must-fire: #2753's constant-token batch. The band renders and
    /// reports NO throughput at all.
    #[test]
    fn a_constant_token_batch_is_invalid_correctness() {
        let m1: Vec<u32> = (0..128).map(|i| 1000 + i).collect();
        let failing = BatchInvarianceWitness::compare(&m1, &vec![474_u32; 128], 64)
            .formed_at(3, "scripts/perf041_batched_parity_probe.py");
        let d = conformant_band(4)
            .witness(failing)
            .derive()
            .expect("the band still renders");
        assert_eq!(d.status, BandStatus::InvalidCorrectness);
        assert_eq!(d.aggregate_tok_per_sec, None);
        assert_eq!(d.decode_tok_per_sec, None);
        assert_eq!(d.prefill_tok_per_sec, None);
    }

    /// PP-26 must-not-fire.
    #[test]
    fn identical_128_token_prefixes_pass() {
        let d = conformant_band(4).derive().expect("renders");
        assert_eq!(
            d.witness.expect("witness").batch_invariance,
            BatchInvariance::Pass
        );
        assert_eq!(d.status, BandStatus::Unmeasured, "no comparator lane");
        assert!(d.aggregate_tok_per_sec.is_some());
    }

    /// An `INVALID-CORRECTNESS` band names the three metrics it withheld, so a
    /// reader can tell "not measured" from "measured and wrong".
    #[test]
    fn an_invalid_correctness_band_reports_no_throughput() {
        let d = conformant_band(8)
            .witness(BatchInvarianceWitness::compare(&[1, 2, 3], &[9, 9, 9], 64))
            .derive()
            .expect("renders");
        assert_eq!(d.status, BandStatus::InvalidCorrectness);
        assert!(!d.baseline_eligible());
        let notes = d.unproduced.join("\n");
        assert!(notes.contains("aggregate_tok_per_sec"), "{notes}");
        assert!(notes.contains("decode_tok_per_sec"), "{notes}");
        assert!(notes.contains("prefill_tok_per_sec"), "{notes}");
    }

    /// `c = 1` forms no batch, so it needs no witness and stays valid without
    /// one. Applying the rule there would make every single-stream band invalid.
    #[test]
    fn c1_needs_no_witness() {
        let mut b = conformant_band(1);
        b.witness = None;
        let d = b.derive().expect("renders");
        assert_ne!(d.status, BandStatus::InvalidCorrectness);
        assert!(d.aggregate_tok_per_sec.is_some());

        // …and the same band at c=4 without one is INVALID-CORRECTNESS.
        let mut wider = conformant_band(4);
        wider.witness = None;
        assert_eq!(
            wider.derive().expect("renders").status,
            BandStatus::InvalidCorrectness
        );
    }

    /// PP-4 — a v2-dated receipt is historical: the v3 rules are not applied to
    /// it retroactively, and it is not a baseline either.
    #[test]
    fn a_v2_receipt_is_historical_not_a_baseline() {
        let mut b = conformant_band(4);
        b.witness = None;
        b.stream_mode = None;
        let v2 = b.derive_at(2).expect("renders");
        assert_ne!(v2.status, BandStatus::InvalidCorrectness);
        assert!(
            v2.aggregate_tok_per_sec.is_some(),
            "a v2 band keeps its throughput"
        );
        assert!(!v2.baseline_eligible(), "but is never a baseline");
        assert_eq!(
            b.derive_at(3).expect("renders").status,
            BandStatus::InvalidCorrectness,
            "the same band at v3"
        );
    }

    /// PP-4 — a band reporting numbers must report all three. `prefill` absent
    /// is a departure, not a silent omission.
    #[test]
    fn a_measured_band_without_prefill_is_nonconformant() {
        let mut b = conformant_band(1);
        for r in &mut b.requests {
            r.prefill_ms = None;
        }
        let d = b.derive().expect("renders");
        assert_eq!(d.prefill_tok_per_sec, None);
        assert_eq!(d.status, BandStatus::NonconformantValid);
        assert!(d.unproduced.join("\n").contains("PP-13"));
    }

    /// §3 — `prefill` is `Σ prompt_tokens / Σ prefill_ms`, over the requests
    /// that carry a server-reported duration and no others.
    #[test]
    fn prefill_is_prompt_tokens_over_server_prefill_ms() {
        let a = RequestOutcome::completed(0.0, 500.0, 8)
            .streamed(
                40.0,
                vec![40.0, 100.0, 200.0, 300.0, 350.0, 400.0, 450.0, 500.0],
            )
            .server_prefill(500, 100.0);
        let b = RequestOutcome::completed(10.0, 520.0, 8)
            .streamed(
                40.0,
                vec![50.0, 110.0, 210.0, 310.0, 360.0, 410.0, 460.0, 520.0],
            )
            .server_prefill(300, 100.0);
        // A third request the server gave no timing for contributes nothing.
        let c = RequestOutcome::completed(20.0, 530.0, 8)
            .streamed(
                40.0,
                vec![60.0, 120.0, 220.0, 320.0, 370.0, 420.0, 470.0, 530.0],
            )
            .with_prompt_tokens(9_999);
        // And a fourth whose server reported a ZERO prefill duration: a
        // zero-length prefill is not a measurement, and admitting it would make
        // the sum's denominator smaller and the rate larger.
        let zero = RequestOutcome::completed(30.0, 540.0, 8)
            .streamed(
                40.0,
                vec![70.0, 130.0, 230.0, 330.0, 380.0, 430.0, 480.0, 540.0],
            )
            .server_prefill(7_777, 0.0);
        let d = BandInput::new(1, 1000.0, vec![a, b, c, zero], unmeasured())
            .stream_mode(StreamMode::Live)
            .n_predict(8)
            .derive()
            .expect("renders");
        // 800 prompt tokens over 200 ms = 4000 tok/s.
        assert!(
            (d.prefill_tok_per_sec.expect("prefill") - 4_000.0).abs() < 1e-9,
            "{:?}",
            d.prefill_tok_per_sec
        );
    }

    /// §4.3 — five interleaved replicates is the floor; a receipt that ran
    /// three is a record.
    #[test]
    fn fewer_than_five_replicates_makes_the_band_nonconformant() {
        let b = conformant_band(1);
        let five = BandContext {
            replicates: 5,
            ..BandContext::default()
        };
        let three = BandContext {
            replicates: 3,
            ..BandContext::default()
        };
        assert_eq!(
            b.derive_in(&five).expect("renders").status,
            BandStatus::Unmeasured
        );
        assert_eq!(
            b.derive_in(&three).expect("renders").status,
            BandStatus::NonconformantValid
        );
    }

    /// §4.3 — and replicates that did not alternate are a record too.
    #[test]
    fn a_non_interleaved_receipt_makes_the_band_nonconformant() {
        let ctx = BandContext {
            interleaved: false,
            ..BandContext::default()
        };
        assert_eq!(
            conformant_band(1).derive_in(&ctx).expect("renders").status,
            BandStatus::NonconformantValid
        );
    }

    /// PP-20 — a stale pin is its own status, ahead of NONCONFORMANT.
    #[test]
    fn a_stale_pin_renders_comparator_stale() {
        let ctx = BandContext {
            comparator_stale: true,
            ..BandContext::default()
        };
        let d = conformant_band(1).derive_in(&ctx).expect("renders");
        assert_eq!(d.status, BandStatus::ComparatorStale);
        assert!(!d.baseline_eligible());
    }

    // -- PP-3 / PP-22 / PP-5: the join -------------------------------------

    fn jkey(c: u32) -> JoinKey {
        JoinKey {
            host: "lambda".to_string(),
            workload: Workload::W1,
            band: c,
            model: "qwen2.5-coder-7b-apache-q4k-v1".to_string(),
            quant: "Q4_K_M".to_string(),
            tokenization: TokenCountingMethod::ClientTokenizer,
            window_ms: 1_000,
            replicates: 5,
            interleaved: true,
            n_ctx_slot: Some(1024),
            kv_type: Some("f16".to_string()),
            fa: Some(true),
            n_batch: Some(2048),
            n_predict: 128,
        }
    }

    fn same_run() -> RunId {
        RunId::derive("2026-09-02T10:11:12.345Z", "lambda", &"a".repeat(64), 4242)
    }

    fn another_run() -> RunId {
        RunId::derive("2026-09-02T11:00:00.000Z", "lambda", &"a".repeat(64), 4243)
    }

    /// PP-3's must-not-fire: a same-run comparator lane joins, and the ratios
    /// that come out carry the estimator each metric's unit demands.
    #[test]
    fn ratio_paired__a_same_run_baseline_joins() {
        let subject = conformant_band(1);
        let comparator = conformant_band(1);
        let id = same_run();
        let status = BandInput::join_status(&subject, &comparator, &jkey(1), &jkey(1), (&id, &id))
            .expect("a same-run, same-key, timeout-free join");

        let ComparatorStatus::Measured(join) = &status else {
            panic!("expected Measured, got {status:?}");
        };
        let (baseline, ratios) = (join.baseline(), join.ratios());
        assert_eq!(
            baseline.run_id.as_ref(),
            Some(&id),
            "PP-3: the baseline says which run it came from"
        );
        assert_eq!(baseline.join_key.as_ref(), Some(&jkey(1)));
        assert_eq!(status.wire_token(), "MEASURED");

        // Identical lanes are parity, by both estimators.
        assert!((ratios.agg.point - 1.0).abs() < 1e-9, "{:?}", ratios.agg);
        assert_eq!(ratios.agg.method, RatioMethod::ReplicateTLower);
        assert!(
            ratios.agg.lcb95.is_none(),
            "one replicate bounds no variance (§4.3)"
        );
        let dec = ratios.dec.as_ref().expect("a live stream has a dec ratio");
        assert_eq!(dec.method, RatioMethod::PairedPercentileBootstrap);
        assert!((dec.point - 1.0).abs() < 1e-9, "{dec:?}");
        assert!(dec.lcb95.is_some(), "the request unit does bound");
        assert!(ratios.prefill.is_some(), "both lanes reported prefill");

        // …and the joined band is MEASURED end to end.
        let joined =
            BandInput::join(&subject, &comparator, &jkey(1), &jkey(1), (&id, &id)).expect("joins");
        assert_eq!(joined.status, BandStatus::Measured);
        assert!(joined.baseline_eligible());
    }

    /// PP-3's must-fire: a baseline from another invocation saw another thermal
    /// state, another free-VRAM figure and another scheduler.
    #[test]
    fn a_baseline_from_another_run_is_refused() {
        let subject = conformant_band(1);
        let comparator = conformant_band(1);
        let (mine, theirs) = (same_run(), another_run());
        assert_ne!(mine, theirs);
        let err =
            BandInput::join_status(&subject, &comparator, &jkey(1), &jkey(1), (&mine, &theirs))
                .expect_err("cross-run baseline");
        assert!(err.contains("PP-3"), "{err}");
        assert!(err.contains("SAME run"), "{err}");
    }

    /// PP-22 at the join, not merely at the key: a c=4 subject against a c=16
    /// comparator never reaches the estimator.
    #[test]
    fn a_key_mismatch_stops_the_join_before_any_ratio_is_computed() {
        let id = same_run();
        let err = BandInput::join_status(
            &conformant_band(4),
            &conformant_band(16),
            &jkey(4),
            &jkey(16),
            (&id, &id),
        )
        .expect_err("c=4 against c=16");
        assert!(err.contains("band: 4 != 16"), "{err}");
    }

    /// PP-5's must-fire: the requests that did not return are exactly the ones
    /// a ratio would have to account for.
    #[test]
    fn a_timed_out_band_cannot_carry_a_ratio() {
        let id = same_run();
        let mut timed_out = conformant_band(1);
        timed_out.requests.push(RequestOutcome::new(
            10.0,
            10.0 + REQUEST_TIMEOUT_MS,
            Outcome::Timeout,
            0,
        ));
        assert_eq!(
            timed_out.derive().expect("renders").timeouts,
            1,
            "control: the band itself still renders its evidence"
        );

        let subject_side = BandInput::join_status(
            &timed_out,
            &conformant_band(1),
            &jkey(1),
            &jkey(1),
            (&id, &id),
        )
        .expect_err("the subject timed out");
        assert!(subject_side.contains("PP-5"), "{subject_side}");
        assert!(subject_side.contains("subject"), "{subject_side}");

        let comparator_side = BandInput::join_status(
            &conformant_band(1),
            &timed_out,
            &jkey(1),
            &jkey(1),
            (&id, &id),
        )
        .expect_err("the comparator timed out");
        assert!(comparator_side.contains("comparator"), "{comparator_side}");

        // …and the clean pair still joins, so the refusal is about the timeout.
        BandInput::join_status(
            &conformant_band(1),
            &conformant_band(1),
            &jkey(1),
            &jkey(1),
            (&id, &id),
        )
        .expect("a clean pair joins");
    }

    /// The ratio direction is subject over comparator, and it MOVES: a faster
    /// subject gives a ratio above 1.
    #[test]
    fn the_joined_ratio_is_subject_over_comparator() {
        let id = same_run();
        let subject = conformant_band(1);
        // Halve every comparator request's duration: twice the throughput.
        let mut fast_comparator = conformant_band(1);
        for r in &mut fast_comparator.requests {
            let dur = r.settled_ms - r.issued_ms;
            r.settled_ms = r.issued_ms + dur / 2.0;
            let first = r.token_times_ms[0];
            for t in &mut r.token_times_ms {
                *t = first + (*t - first) / 2.0;
            }
        }
        let status =
            BandInput::join_status(&subject, &fast_comparator, &jkey(1), &jkey(1), (&id, &id))
                .expect("joins");
        let ComparatorStatus::Measured(join) = &status else {
            panic!("expected Measured");
        };
        let ratios = join.ratios();
        assert!(
            ratios.agg.point < 1.0,
            "a slower subject is below parity: {:?}",
            ratios.agg
        );
        let dec = ratios.dec.as_ref().expect("dec ratio");
        assert!((dec.point - 0.5).abs() < 0.02, "{dec:?}");
    }

    // -- §7.4 precedence, pairwise ----------------------------------------

    /// §7.4's order, as a table. Every adjacent pair, both ways round, so a
    /// flipped comparison in `rank` or an inverted `stronger_of` is caught by
    /// name rather than by a downstream status happening to differ.
    #[test]
    fn the_status_precedence_is_a_total_order_correctness_first() {
        use BandStatus::{
            ComparatorStale, InvalidCorrectness, Measured, Na, NonconformantValid, Unmeasured,
        };
        let strongest_first = [
            InvalidCorrectness,
            ComparatorStale,
            Na,
            NonconformantValid,
            Unmeasured,
            Measured,
        ];
        for (i, strong) in strongest_first.iter().enumerate() {
            for weak in &strongest_first[i + 1..] {
                assert_eq!(
                    strong.stronger_of(*weak),
                    *strong,
                    "{strong:?} must win over {weak:?}"
                );
                assert_eq!(
                    weak.stronger_of(*strong),
                    *strong,
                    "…in either argument order"
                );
            }
            assert_eq!(strong.stronger_of(*strong), *strong, "idempotent");
        }
        // The vocabulary and the order are the same six tokens: a status added
        // to one and not the other would rank arbitrarily.
        assert_eq!(strongest_first.len(), BandStatus::vocabulary().len());
    }

    /// MUST-FIRE, the inversion itself: a `c > 1` band with no witness under an
    /// EXPIRED comparator pin stays `INVALID-CORRECTNESS`.
    ///
    /// `marked_comparator_stale` used to assign the status, so this band came
    /// out labelled `COMPARATOR_STALE` — a reader would have concluded the only
    /// thing wrong was an out-of-date pin, while the band in fact reported no
    /// throughput at all because nothing established the tokens were right.
    #[test]
    fn an_unwitnessed_batch_under_a_stale_pin_stays_invalid_correctness() {
        let ctx = BandContext {
            comparator_stale: true,
            ..BandContext::default()
        };
        let unwitnessed = BandInput::new(4, 1000.0, conformant_band(4).requests, unmeasured())
            .n_predict(128)
            .stream_mode(StreamMode::Live);
        let d = unwitnessed
            .derive_in(&ctx)
            .expect("renders")
            .marked_comparator_stale("2026-01-01T00:00:00.000Z", "2026-09-02T10:11:12.345Z");
        assert_eq!(d.status, BandStatus::InvalidCorrectness);
        assert_eq!(
            d.aggregate_tok_per_sec, None,
            "and it reports no throughput"
        );
        assert!(!d.baseline_eligible());
        // REVERT -> the same band WITH a witness is COMPARATOR_STALE, which is
        // what the stale pin alone is supposed to say.
        let witnessed = conformant_band(4)
            .derive_in(&ctx)
            .expect("renders")
            .marked_comparator_stale("2026-01-01T00:00:00.000Z", "2026-09-02T10:11:12.345Z");
        assert_eq!(witnessed.status, BandStatus::ComparatorStale);
    }

    /// `NA` outranks `NONCONFORMANT-VALID`: a band excluded permanently — one
    /// that usually never ran at all — is not first a finding about how it ran.
    #[test]
    fn a_not_applicable_band_is_na_even_when_it_is_also_nonconformant() {
        let ctx = BandContext {
            interleaved: false,
            ..BandContext::default()
        };
        let na = ComparatorStatus::not_applicable("perf-matrix.yaml", "no Metal path (#2841)");
        let d = BandInput::new(1, 1000.0, conformant_band(1).requests, na)
            .n_predict(128)
            .stream_mode(StreamMode::Live)
            .derive_in(&ctx)
            .expect("renders");
        assert_eq!(d.status, BandStatus::Na);
        // …and the same departure over an UNMEASURED comparator is the weaker
        // token, so this is a fact about NA and not about the departure.
        let d2 = conformant_band(1).derive_in(&ctx).expect("renders");
        assert_eq!(d2.status, BandStatus::NonconformantValid);
    }

    // -- PP-3 / PP-22 / PP-5: the payload has no public constructor ---------

    /// PP-3 must-not-fire, at the type level: `ComparatorStatus::Measured`'s
    /// payload is a [`MeasuredJoin`] whose fields are private and whose only
    /// constructor is `pub(crate)`. Outside this crate there is no expression
    /// that builds one, so a baseline from another run — or another band, or a
    /// lane that timed out — cannot be attached by writing a struct literal.
    ///
    /// The compile-fail half cannot be a `#[test]`; it is this, which does not
    /// compile from `apr-cli`:
    ///
    /// ```text
    /// ComparatorStatus::Measured(MeasuredJoin { baseline, ratios })  // private fields
    /// ComparatorStatus::Measured(MeasuredJoin::sealed(band, ratios)) // private fn
    /// ```
    ///
    /// What IS public is reading, which the receipt renderer needs.
    #[test]
    fn ratio_paired__the_measured_payload_is_read_only_outside_the_join() {
        let id = same_run();
        let status = BandInput::join_status(
            &conformant_band(1),
            &conformant_band(1),
            &jkey(1),
            &jkey(1),
            (&id, &id),
        )
        .expect("joins");
        let ComparatorStatus::Measured(join) = &status else {
            panic!("expected Measured");
        };
        assert_eq!(join.baseline().concurrency, 1);
        assert_eq!(join.baseline().run_id.as_ref(), Some(&id));
        assert!((join.ratios().agg.point - 1.0).abs() < 1e-9);
    }

    // -- PP-26: the witness is about the SUBJECT ---------------------------

    /// PP-26 must-not-fire on the oracle: a **comparator**-lane band at `c > 1`
    /// with no witness is NOT `INVALID-CORRECTNESS` and keeps its throughput.
    ///
    /// The witness answers "does `apr serve` return the same tokens under
    /// batching as it does alone?". `llama-server` is what that question is
    /// asked against; demanding it witness itself would red every baseline, and
    /// the producer's workaround — copying the SUBJECT's witness onto the
    /// comparator band — had a subject-side PASS vouching for the oracle.
    #[test]
    fn a_comparator_lane_band_needs_no_batch_invariance_witness() {
        let subject = BandInput::new(4, 1000.0, conformant_band(4).requests, unmeasured())
            .n_predict(128)
            .stream_mode(StreamMode::Live);
        let subject_band = subject.clone().derive().expect("renders");
        assert_eq!(
            subject_band.status,
            BandStatus::InvalidCorrectness,
            "the SUBJECT still needs one"
        );

        let comparator_band = subject.role(Lane::Llama).derive().expect("renders");
        assert_ne!(comparator_band.status, BandStatus::InvalidCorrectness);
        assert!(
            comparator_band.aggregate_tok_per_sec.is_some(),
            "the oracle's throughput is not withheld for a witness it is not the subject of"
        );
        assert!(
            comparator_band.witness.is_none(),
            "and it carries no witness of its own"
        );
    }

    /// …and `c = 1` on either lane needs none, so the exemption above is about
    /// the LANE and not about the concurrency.
    #[test]
    fn the_comparator_exemption_is_about_the_lane_not_the_band_width() {
        let one = BandInput::new(1, 1000.0, conformant_band(1).requests, unmeasured())
            .n_predict(128)
            .stream_mode(StreamMode::Live);
        assert_ne!(
            one.clone().derive().expect("renders").status,
            BandStatus::InvalidCorrectness
        );
        assert_ne!(
            one.role(Lane::Llama).derive().expect("renders").status,
            BandStatus::InvalidCorrectness
        );
    }

    // -- P-5: a ratio of two withheld numbers -------------------------------

    /// MUST-FIRE: `ratios.dec` is `None` when either lane's decode was
    /// suppressed as unreliable.
    ///
    /// `dec` is a paired bootstrap over the RAW request samples, which survive
    /// whatever the band decided about them — so a lane whose decode was
    /// withheld (here: a replayed stream) still produced a `dec` ratio computed
    /// from exactly the samples the band refused to report.
    #[test]
    fn a_lane_with_suppressed_decode_forms_no_dec_ratio() {
        let id = same_run();
        let replayed = BandInput::new(1, 1000.0, conformant_band(1).requests, unmeasured())
            .n_predict(128)
            .stream_mode(StreamMode::Replayed)
            .witness(passing_witness());
        assert_eq!(
            replayed.derive().expect("renders").decode_tok_per_sec,
            None,
            "the fixture's decode must actually be withheld"
        );

        let status = BandInput::join_status(
            &conformant_band(1),
            &replayed,
            &jkey(1),
            &jkey(1),
            (&id, &id),
        )
        .expect("joins");
        let ComparatorStatus::Measured(join) = &status else {
            panic!("expected Measured");
        };
        assert!(
            join.ratios().dec.is_none(),
            "a ratio whose denominator the band refused to report is not a ratio: {:?}",
            join.ratios().dec
        );
        // REVERT -> GREEN: two live lanes do form one.
        let live = BandInput::join_status(
            &conformant_band(1),
            &conformant_band(1),
            &jkey(1),
            &jkey(1),
            (&id, &id),
        )
        .expect("joins");
        let ComparatorStatus::Measured(join) = &live else {
            panic!("expected Measured");
        };
        assert!(join.ratios().dec.is_some());
    }

    /// The same rule for `prefill`: a lane with no server timings forms no
    /// prefill ratio, and the numerator alone is not one.
    #[test]
    fn a_lane_without_server_prefill_forms_no_prefill_ratio() {
        let id = same_run();
        let no_timings: Vec<RequestOutcome> = (0..8)
            .map(|i| {
                let (issued, dur) = (f64::from(i) * 100.0, 90.0 + f64::from(i));
                let ttft = dur * 0.08;
                let times: Vec<f64> = (0..128)
                    .map(|k| issued + ttft + f64::from(k) * (dur - ttft) / 128.0)
                    .collect();
                RequestOutcome::completed(issued, issued + dur, 128)
                    .with_prompt_tokens(512)
                    .streamed(ttft, times)
            })
            .collect();
        let bare = BandInput::new(1, 1000.0, no_timings, unmeasured())
            .n_predict(128)
            .stream_mode(StreamMode::Live)
            .witness(passing_witness());
        assert_eq!(bare.derive().expect("renders").prefill_tok_per_sec, None);

        let status =
            BandInput::join_status(&conformant_band(1), &bare, &jkey(1), &jkey(1), (&id, &id))
                .expect("joins");
        let ComparatorStatus::Measured(join) = &status else {
            panic!("expected Measured");
        };
        assert!(join.ratios().prefill.is_none());
    }

    // -- §4.4.2: the driver's protocol violations reach the receipt ---------

    /// MUST-FIRE: a protocol departure the DRIVER observed makes the band
    /// `NONCONFORMANT-VALID` and is named in `unproduced_fields`.
    ///
    /// The producer printed these to stdout and dropped them. A violation the
    /// operator watched scroll past and the receipt did not carry is a receipt
    /// that reads conformant — and the receipt is the only thing the gate sees.
    #[test]
    fn a_driver_protocol_violation_reaches_the_band_and_its_status() {
        let clean = conformant_band(1).derive().expect("renders");
        assert_eq!(clean.status, BandStatus::Unmeasured);

        let violated = conformant_band(1)
            .conformance_violations(vec![
                "window closed after 30 samples, below the max(30, 8c) floor".to_string(),
            ])
            .derive()
            .expect("renders");
        assert_eq!(violated.status, BandStatus::NonconformantValid);
        assert!(
            violated
                .unproduced
                .iter()
                .any(|u| u.contains("below the max(30, 8c) floor") && u.contains("§4.4.2")),
            "the violation text itself must be on the receipt: {:?}",
            violated.unproduced
        );
    }

    /// PP-7 — every band carries its own rows, and the token times are NOT in
    /// them (they live in the gz side file the receipt links by digest).
    #[test]
    fn a_band_carries_one_sample_row_per_request() {
        let d = conformant_band(1).derive().expect("renders");
        assert_eq!(d.samples.len(), d.requested);
        assert_eq!(d.samples[0].index, 0);
        assert_eq!(d.samples[0].generated_tokens, 128);
        assert_eq!(d.samples[0].prompt_tokens, 512);
        assert!(d.samples[0].ttft_ms.is_some());
        let json = serde_json::to_string(&d.samples[0]).expect("serialises");
        assert!(
            !json.contains("token_times"),
            "token times stay in the side file: {json}"
        );
    }
}
