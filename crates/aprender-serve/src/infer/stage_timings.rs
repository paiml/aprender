//! PMAT-3598 row 1 (#3542) — where the time went, as a thing a consumer can read.
//!
//! **Why this exists.** `apr run` reported one number: wall clock. On a 4B model that number was 14 s
//! against llama.cpp's 1.8 s, and from it the mechanism was named twice and was wrong twice inside
//! fifteen minutes — *"it is load"* died to a 2-character prompt costing 6.8 s, *"prefill at 4.7 tok/s"*
//! died to 144→288 words costing only ~2 s. Three candidate mechanisms and one number that admits all
//! three. **Slow is allowed; invisible is not.**
//!
//! **Every field is `Option`, and absent means NOT MEASURED — never zero.** A stage that reports `0.0`
//! because nobody timed it is the defect this row exists to end: it reads as "free" and it is the
//! cheapest possible lie. The CPU and wgpu paths do not split prefill from decode today, so on those
//! backends those two fields are `None` and say so.
//!
//! **The residual is a field, not a rounding error.** [`StageTimings::unattributed_ms`] is
//! `wall − Σ(measured stages)` and is always present. It is what makes "the fields sum to the wall
//! clock" an assertion a test can make rather than a claim in a comment: they sum exactly, by
//! construction, and the part nobody attributed is visible instead of smeared across the parts that
//! were. A large `unattributed_ms` is a finding — it says the instrument is missing a stage.

use std::time::Instant;

/// One `apr run` broken into the stages that can each be attributed to a cause.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StageTimings {
    /// Reading and building the model on the HOST: mmap, prefault, tensor wiring.
    pub load_ms: Option<f64>,
    /// Moving the weights HOST → DEVICE. `None` on any backend that never transfers (CPU), which is
    /// a different statement from `Some(0.0)`.
    pub h2d_ms: Option<f64>,
    /// The pre-generation first-token validation the CUDA path runs before it will use the GPU.
    ///
    /// It is a whole forward pass, it happens before the user's first token, and until this row it
    /// was invisible — folded into a wall clock that nobody could decompose. It is its own field
    /// because attributing it to `prefill_ms` would report the model as twice as slow to prefill as
    /// it is, and attributing it to `load_ms` would hide that it scales with the prompt.
    pub validate_ms: Option<f64>,
    /// The CPU reference forward inside [`Self::validate_ms`], when the guard reports its halves.
    ///
    /// Split out because #3604 caches the whole validation, and its before/after receipt has to
    /// record WHICH mechanism disappeared, not merely that something got faster. "Not blocking the
    /// fix" and "not needed in the receipt" are different claims.
    pub validate_ref_ms: Option<f64>,
    /// The GPU probe forward inside [`Self::validate_ms`]. Absent on any guard that times itself
    /// as one unit — the dense path does, and says so by absence rather than by halving the total.
    pub validate_probe_ms: Option<f64>,
    /// Processing the prompt.
    pub prefill_ms: Option<f64>,
    /// Generating the output tokens.
    pub decode_ms: Option<f64>,
    /// Tokens actually generated — the denominator any rate here is computed against.
    pub tokens_out: usize,
    /// `wall − Σ(measured stages)`. Always present; see the module note.
    pub unattributed_ms: f64,
    /// Which generate path produced these numbers, so a reader knows which fields could be measured.
    pub backend: String,
    /// Total wall clock for the run, measured once at the outermost boundary.
    pub wall_ms: f64,
}

impl StageTimings {
    /// Sum of the stages that were actually measured. `None`s contribute nothing — they are not zero.
    #[must_use]
    pub fn measured_sum_ms(&self) -> f64 {
        // `validate_ref_ms` and `validate_probe_ms` are INSIDE `validate_ms` and are deliberately
        // absent here: adding them would double-count the guard and the books would not close.
        [
            self.load_ms,
            self.h2d_ms,
            self.validate_ms,
            self.prefill_ms,
            self.decode_ms,
        ]
        .iter()
        .flatten()
        .sum()
    }

    /// Close the books: record the wall clock and make the residual explicit.
    ///
    /// Called once, at the outermost boundary. After this the invariant
    /// `measured_sum_ms() + unattributed_ms == wall_ms` holds exactly, which is what the falsifier
    /// asserts — a tolerance is then a statement about how much is UNATTRIBUTED, not about whether
    /// the arithmetic works.
    pub fn close(&mut self, wall_ms: f64) {
        self.wall_ms = wall_ms;
        self.unattributed_ms = wall_ms - self.measured_sum_ms();
    }

    /// The names of the stages this run could measure, in order — for the report.
    #[must_use]
    pub fn measured(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        for (name, val) in [
            ("load_ms", self.load_ms),
            ("h2d_ms", self.h2d_ms),
            ("validate_ms", self.validate_ms),
            ("validate_ref_ms", self.validate_ref_ms),
            ("validate_probe_ms", self.validate_probe_ms),
            ("prefill_ms", self.prefill_ms),
            ("decode_ms", self.decode_ms),
        ] {
            if val.is_some() {
                v.push(name);
            }
        }
        v
    }
}

/// A deliberate delay planted in one stage, for the falsifier.
///
/// **A timing row that cannot localise a planted delay is decoration.** `APR_STAGE_DELAY_MS` takes
/// `<stage>:<ms>` (e.g. `h2d:500`) and sleeps inside exactly that stage, so a test can assert the
/// delay lands in that field AND in no other. Reading an env var in the measured path is deliberate:
/// the alternative is a test double that measures a different code path from the one that ships.
///
/// Unset, malformed, or naming an unknown stage → no delay, silently. This never fails a run.
#[must_use]
pub fn planted_delay(stage: &str) -> Option<std::time::Duration> {
    let spec = std::env::var("APR_STAGE_DELAY_MS").ok()?;
    let (want, ms) = spec.split_once(':')?;
    if want != stage {
        return None;
    }
    Some(std::time::Duration::from_millis(ms.trim().parse().ok()?))
}

/// Time `f`, adding any delay planted for `stage`, and return `(value, elapsed_ms)`.
pub fn timed<T>(stage: &str, f: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    if let Some(d) = planted_delay(stage) {
        std::thread::sleep(d);
    }
    let out = f();
    (out, start.elapsed().as_secs_f64() * 1000.0)
}

#[cfg(test)]
#[path = "stage_timings_tests.rs"]
mod tests;
