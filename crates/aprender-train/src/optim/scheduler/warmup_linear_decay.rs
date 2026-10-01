//! Warmup + LINEAR decay learning rate scheduler — SetFit's reference schedule.
//!
//! # Why this exists when the repo already has three schedulers
//!
//! `LinearWarmupLR` warms up and then holds the peak rate CONSTANT, and
//! `WarmupCosineDecayLR` decays on a cosine. SetFit's reference recipe uses
//! `sentence-transformers`' default, which is HuggingFace's
//! `get_linear_schedule_with_warmup`: linear warmup, then LINEAR decay to zero.
//! Substituting either existing scheduler would be an undeclared deviation from the
//! reference recipe that no gate in this milestone would catch — the loss curve would look
//! perfectly plausible either way.

use super::LRScheduler;
use crate::optim::Optimizer;

/// Warmup + linear decay learning rate scheduler.
///
/// - Phase 1 (warmup): linear increase from 0 to `lr_max` over `warmup_steps`.
/// - Phase 2 (decay): linear decrease from `lr_max` to `lr_min` over the remaining steps.
///
/// # Two reference facts are pinned here, and neither is an off-by-one
///
/// **1. `get_lr()` at step 0 with `warmup_steps > 0` is EXACTLY 0.0.** HuggingFace
/// implements the schedule as a `LambdaLR` whose lambda is
/// `current_step / max(1, num_warmup_steps)`, so lambda(0) = 0 and the first optimizer step
/// genuinely runs at a learning rate of zero. This is deliberate reference fidelity. It is
/// recorded on the type so the 03-05 loop ordering is not "fixed" later by a reader who
/// believes a first step at lr 0 must be a bug.
///
/// **2. The warmup step COUNT is a ceiling, not a rounding.** See
/// [`warmup_steps_from_ratio`].
///
/// # Zero-division guards
///
/// `warmup_steps == 0` (no warmup at all) and `decay_steps == 0` (`total_steps` at or below
/// `warmup_steps`) each have an explicit branch, mirroring `WarmupCosineDecayLR`.
pub struct WarmupLinearDecayLR {
    lr_max: f32,
    lr_min: f32,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl WarmupLinearDecayLR {
    /// Create a new warmup + linear decay scheduler.
    ///
    /// # Arguments
    /// * `lr_max` - Peak learning rate, reached at the end of warmup
    /// * `lr_min` - Floor learning rate, reached at `total_steps` (0.0 for the reference)
    /// * `warmup_steps` - Number of warmup steps, normally [`warmup_steps_from_ratio`]
    /// * `total_steps` - Total training steps, warmup included
    #[must_use]
    pub fn new(lr_max: f32, lr_min: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self { lr_max, lr_min, warmup_steps, total_steps, current_step: 0 }
    }

    /// Apply the current learning rate to an optimizer.
    pub fn apply<O: Optimizer>(&self, optimizer: &mut O) {
        optimizer.set_lr(self.get_lr());
    }

    /// The step counter, exposed so a training loop can record it in provenance.
    #[must_use]
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// The resolved warmup step count.
    #[must_use]
    pub fn warmup_steps(&self) -> usize {
        self.warmup_steps
    }

    /// The resolved total step count.
    #[must_use]
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }
}

/// `warmup_steps = ceil(warmup_ratio * total_steps)` — the reference formula.
///
/// # This is `ceil`, and that is not a stylistic choice
///
/// HuggingFace's `TrainingArguments.get_warmup_steps` computes
/// `math.ceil(num_training_steps * warmup_ratio)`. An earlier draft of this phase's plan
/// specified `round`, which was a guess: the two disagree for any product with a fractional
/// part below 0.5 — `total_steps = 11, ratio = 0.1` gives `ceil = 2` and `round = 1` — and
/// on a one-epoch few-shot run that is a materially different schedule for the first steps.
///
/// This is deliberately the SINGLE implementation. Plan 03-05 must call it rather than
/// recompute the product, so there is no second place for the rounding rule to drift.
///
/// A `ratio` outside `[0.0, 1.0]` is unreachable through `SetFitTrainConfig`, whose
/// `warmup_ratio` knob is validated to that closed range. The result is nonetheless clamped
/// to `total_steps`, which makes `warmup_steps <= total_steps` a fact about this function
/// rather than an obligation on its callers — and that is what discharges the cross-field
/// "warmup exceeds total" rule structurally instead of by a runtime comparison somebody
/// could forget.
#[must_use]
pub fn warmup_steps_from_ratio(total_steps: u64, ratio: f64) -> u64 {
    if !ratio.is_finite() || ratio <= 0.0 || total_steps == 0 {
        return 0;
    }
    #[allow(clippy::cast_precision_loss)]
    let product = (total_steps as f64) * ratio;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let steps = product.ceil() as u64;
    steps.min(total_steps)
}

impl LRScheduler for WarmupLinearDecayLR {
    fn get_lr(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            // `current_step < warmup_steps` already implies `warmup_steps > 0`, so the
            // division below is total. The zero-warmup case falls through to the decay arm.
            // Reference fidelity: at step 0 this is exactly 0.0.
            #[allow(clippy::cast_precision_loss)]
            let progress = self.current_step as f32 / self.warmup_steps as f32;
            return self.lr_max * progress;
        }

        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps);
        if decay_steps == 0 {
            return self.lr_min;
        }

        let decay_step = self.current_step - self.warmup_steps;
        if decay_step >= decay_steps {
            return self.lr_min;
        }

        #[allow(clippy::cast_precision_loss)]
        let progress = decay_step as f32 / decay_steps as f32;
        self.lr_min + (self.lr_max - self.lr_min) * (1.0 - progress)
    }

    fn step(&mut self) {
        self.current_step += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    #[test]
    fn warmup_linear_lr_at_step_zero_is_exactly_zero() {
        let scheduler = WarmupLinearDecayLR::new(1.0, 0.0, 10, 100);
        // EXACTLY zero, asserted as an equality rather than a tolerance: the reference's
        // lambda(0) is 0, and a near-zero value would mean a different first step.
        assert_eq!(scheduler.get_lr(), 0.0);
    }

    #[test]
    fn warmup_linear_lr_ramps_linearly_through_warmup() {
        let mut scheduler = WarmupLinearDecayLR::new(1.0, 0.0, 10, 100);
        for expected_step in 0..10_usize {
            #[allow(clippy::cast_precision_loss)]
            let expected = expected_step as f32 / 10.0;
            assert!(
                (scheduler.get_lr() - expected).abs() < EPS,
                "step {expected_step}: expected {expected}, got {}",
                scheduler.get_lr(),
            );
            scheduler.step();
        }
    }

    #[test]
    fn warmup_linear_lr_peaks_at_the_end_of_warmup() {
        let mut scheduler = WarmupLinearDecayLR::new(2e-5, 0.0, 10, 100);
        for _ in 0..10 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 2e-5).abs() < EPS);
    }

    #[test]
    fn warmup_linear_lr_descends_linearly_after_warmup() {
        // warmup 10, total 110 -> 100 decay steps. Halfway through the decay the rate is
        // half the peak.
        let mut scheduler = WarmupLinearDecayLR::new(1.0, 0.0, 10, 110);
        for _ in 0..60 {
            scheduler.step();
        }
        assert!(
            (scheduler.get_lr() - 0.5).abs() < EPS,
            "mid-decay must be half the peak, got {}",
            scheduler.get_lr(),
        );
    }

    #[test]
    fn warmup_linear_lr_is_exactly_lr_min_at_total_steps() {
        let mut scheduler = WarmupLinearDecayLR::new(1.0, 0.0, 10, 100);
        for _ in 0..100 {
            scheduler.step();
        }
        assert_eq!(scheduler.get_lr(), 0.0);
        // And it stays there rather than going negative.
        scheduler.step();
        assert_eq!(scheduler.get_lr(), 0.0);
    }

    #[test]
    fn warmup_linear_lr_honours_a_non_zero_floor() {
        let mut scheduler = WarmupLinearDecayLR::new(1.0, 0.25, 0, 10);
        for _ in 0..10 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.25).abs() < EPS);
    }

    #[test]
    fn warmup_linear_lr_zero_warmup_guard_starts_at_the_peak() {
        let scheduler = WarmupLinearDecayLR::new(1.0, 0.0, 0, 100);
        assert!((scheduler.get_lr() - 1.0).abs() < EPS);
    }

    #[test]
    fn warmup_linear_lr_zero_decay_guard_returns_the_floor() {
        // total_steps <= warmup_steps leaves no decay window at all.
        let mut scheduler = WarmupLinearDecayLR::new(1.0, 0.1, 10, 10);
        for _ in 0..10 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.1).abs() < EPS);
    }

    #[test]
    fn warmup_linear_accessors_report_the_resolved_schedule() {
        let mut scheduler = WarmupLinearDecayLR::new(1.0, 0.0, 7, 70);
        assert_eq!(scheduler.warmup_steps(), 7);
        assert_eq!(scheduler.total_steps(), 70);
        assert_eq!(scheduler.current_step(), 0);
        scheduler.step();
        assert_eq!(scheduler.current_step(), 1);
    }

    // ─── warmup_steps_from_ratio: the reference rounding rule ───────────────

    #[test]
    fn warmup_linear_steps_from_ratio_uses_ceil_not_round() {
        // THE divergence case. 11 * 0.1 = 1.1000000000000001; ceil is 2, round is 1.
        assert_eq!(warmup_steps_from_ratio(11, 0.1), 2);
        // A second, larger divergence so the first is not an artifact of one input.
        // 101 * 0.001 = 0.101; ceil is 1, round is 0.
        assert_eq!(warmup_steps_from_ratio(101, 0.001), 1);
        // And a case where they agree, so the test is not merely detecting "always ceil".
        assert_eq!(warmup_steps_from_ratio(100, 0.1), 10);
    }

    #[test]
    fn warmup_linear_steps_from_ratio_table() {
        let table = [
            (0_u64, 0.1_f64, 0_u64),
            (10, 0.0, 0),
            (10, 0.1, 1),
            (10, 0.15, 2),
            (10, 1.0, 10),
            (587, 0.1, 59),
            (1, 0.1, 1),
        ];
        for (total, ratio, expected) in table {
            assert_eq!(
                warmup_steps_from_ratio(total, ratio),
                expected,
                "total={total} ratio={ratio}",
            );
        }
    }

    #[test]
    fn warmup_linear_steps_from_ratio_never_exceeds_total_steps() {
        // The structural discharge of the cross-field rule: there is no (total, ratio) in
        // the knob's validated range for which warmup exceeds total, and the clamp makes
        // that true even outside it.
        for total in [0_u64, 1, 2, 7, 16, 100, 587, 10_000] {
            for ratio in [0.0_f64, 0.001, 0.1, 0.5, 0.999, 1.0, 1.5, f64::INFINITY] {
                let steps = warmup_steps_from_ratio(total, ratio);
                assert!(
                    steps <= total,
                    "total={total} ratio={ratio} produced warmup {steps} > total",
                );
            }
        }
    }

    #[test]
    fn warmup_linear_steps_from_ratio_rejects_nonsense_ratios_by_returning_zero() {
        assert_eq!(warmup_steps_from_ratio(100, f64::NAN), 0);
        assert_eq!(warmup_steps_from_ratio(100, -0.5), 0);
        assert_eq!(warmup_steps_from_ratio(100, 0.0), 0);
    }
}
