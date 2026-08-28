//! §4.4.7 boundary effects and drain — the producer for `drain_ms`, and for the
//! four request counters that must never be conflated.
//!
//! # Why this exists
//!
//! `scripts/perf_gate.sh:42` fails any receipt whose `drain_ms` is absent. On
//! `62d23d8d1`, `grep -rn "drain_ms" --include="*.rs" crates` returned **zero
//! lines**: nothing in the workspace could produce the field, so Arm C rejected
//! every receipt that could ever be measured. A gate that can only fail is the
//! mirror of one that can only pass.
//!
//! # What `drain_ms` MEANS (§4.4.7), stated before it is computed
//!
//! The measurement window opens at offset `0` and **closes at `T`**. No new
//! request is issued at or after `T` (I-14). Every request issued before `T` is
//! then *drained* — allowed to run on past `T` to completion or timeout.
//!
//! > `drain_ms` = (last settlement of any pre-`T` request) − `T`, clamped at 0.
//!
//! It is the length of the **drain phase**, not a property of any one request,
//! and it is `0` when nothing was still in flight at `T`. §4.4.7's `SUSPECT`
//! rule reads it exactly that way: `drain_ms > 0.5 × window` means one request
//! dominated the window and the band must be re-run longer.
//!
//! # The conflation this module refuses to make
//!
//! §4.4.7: "A request that timed out during drain increments `timeouts`; one
//! **abandoned at drain deadline** increments `truncated`."
//!
//! `truncated` therefore means *the drain deadline arrived while this request
//! was still running*. It does **not** mean `finish_reason == "length"`.
//! W1 (§4.3.1) generates with `max_tokens = 128` and **EOS ignored**, so every
//! single healthy W1 request ends with `finish_reason == "length"`. §4.4.3 puts
//! `agg_tok_s`'s numerator over "completed, **non-truncated**" requests — so
//! reading `truncated` in the finish-reason sense empties the numerator and
//! reports `0 tok/s` for a perfectly healthy server. The two senses are named
//! apart here on purpose: [`Outcome::AbandonedAtDrain`] is the §4.4.7 sense and
//! is the only thing that increments `truncated`.
//!
//! # Timeouts are their own counter, and it is checked, not just named
//!
//! §4.4.3 fixes a hard **120 s per request**. [`Outcome::Timeout`] and
//! [`Outcome::Failed`] are distinct counters (I-5 makes `timeouts > 0` fatal to
//! a host's ratio, while a transport error is a different fault), and
//! [`BandInput::derive`] *verifies* the label against the request's own
//! duration: a `Timeout` that did not reach the timeout, or a `Failed` that
//! exceeded it, is refused rather than counted.
//!
//! # Nothing here is defaulted
//!
//! Every number below is derived from per-request timestamps supplied by the
//! caller. There is no constructor that accepts a `drain_ms` scalar, because a
//! caller-supplied `drain_ms` is indistinguishable from a fabricated one — the
//! same rule `scripts/lib/bench_receipt.py` already applies to ratios ("a stated
//! ratio that its own samples do not produce is a fabricated measurement").

use serde::{Deserialize, Serialize};

/// §4.4.3 — the hard per-request timeout, in milliseconds.
pub const REQUEST_TIMEOUT_MS: f64 = 120_000.0;

/// §4.4.7 — `drain_ms > DRAIN_SUSPECT_FRACTION × window_ms` is annotated `SUSPECT`.
pub const DRAIN_SUSPECT_FRACTION: f64 = 0.5;

/// How one sampled request ended. The four variants are mutually exclusive, so
/// `requested == completed + timeouts + truncated + errors` holds by
/// construction rather than by convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// Returned a usable response, inside the window or during the drain.
    Completed,
    /// Reached the §4.4.3 hard 120 s timeout.
    Timeout,
    /// Still running when the drain deadline arrived (§4.4.7). Increments
    /// `truncated`. **Not** `finish_reason == "length"` — see the module docs.
    AbandonedAtDrain,
    /// Any other fault: transport, non-2xx, unparseable body. Counted apart
    /// from [`Outcome::Timeout`] because they are different defects.
    Failed,
}

/// One sampled request's terminal record. All offsets are milliseconds from the
/// window opening at `0`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestOutcome {
    /// When the client issued the request.
    pub issued_ms: f64,
    /// When the request reached its terminal state.
    pub settled_ms: f64,
    /// How it ended.
    pub outcome: Outcome,
    /// Generated (completion) tokens. Must be non-zero for a completion:
    /// Arm C treats a zero-token response as a failure, not a fast request.
    pub generated_tokens: u32,
    /// Time to first token, when the transport streamed. `None` for a
    /// non-streaming client, which genuinely cannot observe it.
    pub ttft_ms: Option<f64>,
    /// Absolute arrival offsets of each generated token, when the transport
    /// streamed. Empty for a non-streaming client.
    pub token_times_ms: Vec<f64>,
}

impl RequestOutcome {
    /// Wall-clock duration of the request.
    #[must_use]
    pub fn duration_ms(&self) -> f64 {
        self.settled_ms - self.issued_ms
    }

    /// §4.4.3 — per-request `decode_tok_s = (tokens − 1) / (last − first)`.
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

    /// §4.4.3 — this request's inter-token gaps, in milliseconds.
    #[must_use]
    pub fn itl_gaps_ms(&self) -> Vec<f64> {
        self.token_times_ms
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect()
    }
}

/// §4.7.1 — a band's comparator posture.
///
/// There is deliberately **no `Measured` variant**. A comparator ratio needs a
/// baseline object that itself passes every receipt rule (I-3) and a comparator
/// lane driven by the same client binary (I-15, PERF-019). This producer cannot
/// derive one, so it cannot emit one: an unbacked `agg_ratio` is the exact
/// fabrication this epic exists to remove.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparatorStatus {
    /// Permanent exclusion. Needs the decision recorded, per §4.7.1.
    NotApplicable {
        /// Who decided, e.g. `perf-matrix.yaml`.
        decided_by: String,
        /// Why the comparator cannot exist for this cell.
        reason: String,
    },
    /// Temporary. Counted against the denominator; needs an owner.
    Unmeasured {
        /// Who owes the measurement.
        owner: String,
        /// Why it has not been measured yet.
        reason: String,
    },
}

impl ComparatorStatus {
    /// The wire token `perf_gate.sh` reads from `band.comparator_status`.
    #[must_use]
    pub fn wire_token(&self) -> &'static str {
        match self {
            Self::NotApplicable { .. } => "NOT_APPLICABLE",
            Self::Unmeasured { .. } => "UNMEASURED",
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
    /// Every sampled request's terminal record.
    pub requests: Vec<RequestOutcome>,
    /// This cell's comparator posture.
    pub comparator: ComparatorStatus,
}

/// Everything §4.4.7 and §4.4.3 derive from a [`BandInput`].
#[derive(Debug, Clone, PartialEq)]
pub struct DerivedBand {
    /// Fixed concurrency `c`.
    pub concurrency: u32,
    /// `T`, in milliseconds.
    pub window_ms: f64,
    /// §4.4.7 — length of the drain phase, in milliseconds.
    pub drain_ms: f64,
    /// §4.4.7 `SUSPECT` annotations; empty when the band is clean.
    pub suspect: Vec<String>,
    /// Requests issued before `T`.
    pub requested: usize,
    /// Requests that returned a usable response.
    pub completed: usize,
    /// Requests that reached the 120 s hard timeout.
    pub timeouts: usize,
    /// Requests abandoned at the drain deadline (§4.4.7 sense).
    pub truncated: usize,
    /// Requests that failed for any other reason.
    pub errors: usize,
    /// Σ generated tokens over completed requests — `agg_tok_s`'s numerator.
    pub tokens_total: u64,
    /// `agg_tok_s`'s denominator: last completion − first request start, in ms.
    pub span_ms: f64,
    /// §4.4.3 — wall-clock aggregate throughput.
    pub aggregate_tok_per_sec: f64,
    /// §4.4.3 — median per-request decode rate. `None` without streaming.
    pub decode_tok_per_sec: Option<f64>,
    /// p50 time-to-first-token. `None` without streaming.
    pub ttft_p50_ms: Option<f64>,
    /// p95 time-to-first-token. `None` without streaming.
    pub ttft_p95_ms: Option<f64>,
    /// p50 of the pooled inter-token gaps. `None` without streaming.
    pub itl_p50_ms: Option<f64>,
    /// p95 of the pooled inter-token gaps. `None` without streaming.
    pub itl_p95_ms: Option<f64>,
    /// Per-request end-to-end latencies of completed requests (§4.4.5, I-4).
    pub latencies_ms: Vec<f64>,
    /// §4.5 fields this client could not produce, each with its reason. Never
    /// silently omitted and never filled with a plausible number.
    pub unproduced: Vec<String>,
    /// This cell's comparator posture.
    pub comparator: ComparatorStatus,
}

impl BandInput {
    /// Derive every §4.4.3/§4.4.7 quantity from the sampled requests.
    ///
    /// # Errors
    /// When the band violates the measurement protocol: an empty band, a
    /// non-positive window, a request issued at or after `T` (I-14), a
    /// settlement before its issue, a zero-token completion, an abandonment
    /// that did not happen during the drain, or a `Timeout`/`Failed` label that
    /// the request's own duration contradicts.
    pub fn derive(&self) -> Result<DerivedBand, String> {
        self.validate()?;
        let drain_ms = self.drain_ms();
        let span_ms = self.span_ms();
        let tokens_total = self.tokens_total();
        Ok(DerivedBand {
            concurrency: self.concurrency,
            window_ms: self.window_ms,
            drain_ms,
            suspect: self.suspect(drain_ms),
            requested: self.requests.len(),
            completed: self.count(Outcome::Completed),
            timeouts: self.count(Outcome::Timeout),
            truncated: self.count(Outcome::AbandonedAtDrain),
            errors: self.count(Outcome::Failed),
            tokens_total,
            span_ms,
            aggregate_tok_per_sec: rate_per_sec(tokens_total as f64, span_ms),
            decode_tok_per_sec: self.decode_median(),
            ttft_p50_ms: self.ttft_percentile(0.50),
            ttft_p95_ms: self.ttft_percentile(0.95),
            itl_p50_ms: self.itl_percentile(0.50),
            itl_p95_ms: self.itl_percentile(0.95),
            latencies_ms: self.latencies_ms(),
            unproduced: self.unproduced(),
            comparator: self.comparator.clone(),
        })
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

    /// §4.4.7 — last settlement of any request, minus `T`, clamped at 0.
    fn drain_ms(&self) -> f64 {
        let last = self
            .requests
            .iter()
            .map(|r| r.settled_ms)
            .fold(f64::NEG_INFINITY, f64::max);
        (last - self.window_ms).max(0.0)
    }

    /// §4.4.3 — last completion minus first request start.
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
                "SUSPECT §4.4.7 c={}: drain_ms={drain_ms:.1} > 0.5 x window_ms={:.1} — one \
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

    /// §4.5 fields this client could not observe, named rather than invented.
    fn unproduced(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.ttft_percentile(0.50).is_none() {
            out.push(format!(
                "§4.5 c={}: ttft_ms p50/p95 — the transport did not stream, so the client never \
                 observed a first-token instant",
                self.concurrency
            ));
        }
        if self.itl_percentile(0.50).is_none() {
            out.push(format!(
                "§4.5 c={}: itl_ms p50/p95 and decode_tok_per_sec — the transport did not stream, \
                 so there are no per-token arrival times to pool",
                self.concurrency
            ));
        }
        out
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

/// Every per-request rule of §4.4.3 and §4.4.7, applied to one record.
fn validate_request(c: u32, i: usize, r: &RequestOutcome, window_ms: f64) -> Result<(), String> {
    let at = format!("band c={c} request[{i}]");
    if r.issued_ms >= window_ms {
        return Err(format!(
            "{at}: issued_ms={} >= T={window_ms} — I-14: no request is issued at or after the \
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

/// The label a request carries must agree with the clock (§4.4.3, §4.4.7).
fn validate_outcome(at: &str, r: &RequestOutcome, window_ms: f64) -> Result<(), String> {
    let d = r.duration_ms();
    match r.outcome {
        Outcome::Completed if r.generated_tokens == 0 => Err(format!(
            "{at}: completed with zero generated tokens — Arm C: a zero-token response is a \
             failure, not a fast request"
        )),
        Outcome::Timeout if d < REQUEST_TIMEOUT_MS => Err(format!(
            "{at}: labelled Timeout but ran {d:.1} ms < the {REQUEST_TIMEOUT_MS} ms hard timeout \
             (§4.4.3) — that is a Failed, and the two are separate counters"
        )),
        Outcome::Failed if d >= REQUEST_TIMEOUT_MS => Err(format!(
            "{at}: labelled Failed but ran {d:.1} ms >= the {REQUEST_TIMEOUT_MS} ms hard timeout \
             — that is a Timeout, which I-5 makes fatal to this host's ratio"
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
    use super::*;

    /// A completed request that generated `tokens`, issued at `issued_ms` and
    /// settling `dur_ms` later. Non-streaming: no ttft, no token times.
    fn done(issued_ms: f64, dur_ms: f64, tokens: u32) -> RequestOutcome {
        RequestOutcome {
            issued_ms,
            settled_ms: issued_ms + dur_ms,
            outcome: Outcome::Completed,
            generated_tokens: tokens,
            ttft_ms: None,
            token_times_ms: Vec::new(),
        }
    }

    fn unmeasured() -> ComparatorStatus {
        ComparatorStatus::Unmeasured {
            owner: "perf-gate".to_string(),
            reason: "no comparator lane on this cell yet (PERF-019)".to_string(),
        }
    }

    fn band(window_ms: f64, requests: Vec<RequestOutcome>) -> BandInput {
        BandInput {
            concurrency: 4,
            window_ms,
            requests,
            comparator: unmeasured(),
        }
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

    /// §4.4.7 — `drain_ms > 0.5 x window` is annotated SUSPECT.
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

    /// I-14, the registered mutation: "issue one request after `T` and count its
    /// tokens". The band is refused, so the tokens can never be counted.
    #[test]
    fn a_request_issued_at_or_after_t_is_refused() {
        let at_t = band(1000.0, vec![done(0.0, 10.0, 8), done(1000.0, 10.0, 8)]).derive();
        let after_t = band(1000.0, vec![done(0.0, 10.0, 8), done(1500.0, 10.0, 8)]).derive();
        for (label, got) in [("at T", at_t), ("after T", after_t)] {
            let err = got.expect_err(label);
            assert!(err.contains("I-14"), "{label}: {err}");
        }
    }

    /// THE CONFLATION THIS TICKET EXISTS TO PREVENT. Every W1 request stops at
    /// `max_tokens = 128` with EOS ignored, i.e. every one has
    /// `finish_reason == "length"`. Read in the finish-reason sense, `truncated`
    /// would be 8 and §4.4.3's "completed, non-truncated" numerator would be
    /// EMPTY. In the §4.4.7 sense it is 0 and the numerator is 1024 tokens.
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
        assert!(d.aggregate_tok_per_sec > 0.0, "{d:?}");
    }

    /// The §4.4.7 sense: a request still running at the drain deadline.
    #[test]
    fn an_abandoned_request_increments_truncated_not_completed() {
        let abandoned = RequestOutcome {
            issued_ms: 900.0,
            settled_ms: 1400.0,
            outcome: Outcome::AbandonedAtDrain,
            generated_tokens: 12,
            ttft_ms: None,
            token_times_ms: Vec::new(),
        };
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
        let bogus = RequestOutcome {
            issued_ms: 100.0,
            settled_ms: 200.0,
            outcome: Outcome::AbandonedAtDrain,
            generated_tokens: 1,
            ttft_ms: None,
            token_times_ms: Vec::new(),
        };
        let err = band(1000.0, vec![done(0.0, 10.0, 8), bogus])
            .derive()
            .expect_err("settled before T");
        assert!(err.contains("only be abandoned during the drain"), "{err}");
    }

    /// §4.4.3 — `timeouts` is its own counter, and the label is checked against
    /// the clock rather than trusted.
    #[test]
    fn timeouts_and_failures_are_separate_and_both_are_verified() {
        let timeout = RequestOutcome {
            issued_ms: 10.0,
            settled_ms: 10.0 + REQUEST_TIMEOUT_MS,
            outcome: Outcome::Timeout,
            generated_tokens: 0,
            ttft_ms: None,
            token_times_ms: Vec::new(),
        };
        let failure = RequestOutcome {
            issued_ms: 20.0,
            settled_ms: 45.0,
            outcome: Outcome::Failed,
            generated_tokens: 0,
            ttft_ms: None,
            token_times_ms: Vec::new(),
        };
        let d = band(1000.0, vec![done(0.0, 10.0, 8), timeout, failure])
            .derive()
            .expect("valid band");
        assert_eq!(d.timeouts, 1);
        assert_eq!(d.errors, 1);
        assert_eq!(
            d.requested,
            d.completed + d.timeouts + d.truncated + d.errors,
            "the four counters must partition the requests"
        );
    }

    /// A `Timeout` that did not reach 120 s is a mislabelled failure. I-5 makes
    /// `timeouts > 0` fatal to a host's ratio, so the label must be earned.
    #[test]
    fn a_short_request_cannot_be_labelled_a_timeout() {
        let liar = RequestOutcome {
            issued_ms: 0.0,
            settled_ms: 50.0,
            outcome: Outcome::Timeout,
            generated_tokens: 0,
            ttft_ms: None,
            token_times_ms: Vec::new(),
        };
        let err = band(1000.0, vec![done(500.0, 10.0, 8), liar])
            .derive()
            .expect_err("50 ms is not a timeout");
        assert!(err.contains("hard timeout"), "{err}");
    }

    /// And the converse: a request that ran past the hard timeout is a timeout,
    /// not a generic error that would leave `timeouts == 0`.
    #[test]
    fn an_over_long_request_cannot_be_labelled_a_plain_failure() {
        let liar = RequestOutcome {
            issued_ms: 0.0,
            settled_ms: REQUEST_TIMEOUT_MS + 1.0,
            outcome: Outcome::Failed,
            generated_tokens: 0,
            ttft_ms: None,
            token_times_ms: Vec::new(),
        };
        let err = band(1000.0, vec![done(500.0, 10.0, 8), liar])
            .derive()
            .expect_err("past the hard timeout");
        assert!(err.contains("I-5"), "{err}");
    }

    /// Arm C: "a zero-token response is a failure, not a fast request".
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

    /// §4.4.3 — the denominator is wall-clock (last completion − first start),
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
            (d.aggregate_tok_per_sec - 100.0).abs() < 1e-9,
            "200 tokens over 2 s = 100 tok/s, got {}",
            d.aggregate_tok_per_sec
        );
    }

    /// A non-streaming client cannot observe TTFT or ITL. It says so instead of
    /// emitting a plausible number.
    #[test]
    fn a_non_streaming_band_names_what_it_could_not_produce() {
        let d = band(1000.0, vec![done(0.0, 100.0, 128)])
            .derive()
            .expect("valid band");
        assert_eq!(d.ttft_p50_ms, None);
        assert_eq!(d.itl_p95_ms, None);
        assert_eq!(d.decode_tok_per_sec, None);
        assert_eq!(d.unproduced.len(), 2, "{:?}", d.unproduced);
        assert!(d.unproduced.iter().all(|s| s.contains("did not stream")));
    }

    /// A streaming client produces them, and then `unproduced` is empty.
    #[test]
    fn a_streaming_band_produces_ttft_itl_and_decode() {
        let streamed = RequestOutcome {
            issued_ms: 0.0,
            settled_ms: 500.0,
            outcome: Outcome::Completed,
            generated_tokens: 5,
            ttft_ms: Some(100.0),
            token_times_ms: vec![100.0, 200.0, 300.0, 400.0, 500.0],
        };
        let d = band(1000.0, vec![streamed]).derive().expect("valid band");
        assert_eq!(d.ttft_p50_ms, Some(100.0));
        assert_eq!(d.itl_p50_ms, Some(100.0));
        // (5 - 1) tokens over (500 - 100) ms = 10 tok/s.
        assert_eq!(d.decode_tok_per_sec, Some(10.0));
        assert!(d.unproduced.is_empty(), "{:?}", d.unproduced);
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
        let na = ComparatorStatus::NotApplicable {
            decided_by: "perf-matrix.yaml".to_string(),
            reason: "vLLM has no aarch64 build".to_string(),
        };
        assert_eq!(na.wire_token(), "NOT_APPLICABLE");
    }
}
