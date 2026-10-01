//! NeuralProphet events: one additive `Linear(E, 1)` block over indicator columns,
//! composed BESIDE [`crate::np::NpModel`] rather than forked into it (D-30, SC3).
//!
//! # Why a block beside the model and not a field inside it
//!
//! [`crate::np::NpModel::forward`] returns a `Tensor`, so the event term is **one `.add()`
//! away**. Composing rather than forking is what keeps the whole `np::parity` ladder — the
//! SC3 correctness bar and every D-10 training invariant — measuring the SAME model it
//! measured before events existed. The block carries its own parameter set, and BOTH sets
//! are handed to ONE `AdamW`: a second optimiser would give the event weights their own
//! moment estimates and their own schedule, which is a different model, not a wiring detail.
//!
//! # Why the columns are the Prophet expansion
//!
//! One column per `(event, window offset)`, events in INSERTION order and offsets ascending
//! within an event — the same expansion the Prophet arm already applies to
//! [`crate::types::HolidayArg`] (`prophet::columns`). That is what makes D-29's
//! one-argument-two-models claim literal rather than approximate: the two arms expand the
//! same argument into the same columns in the same order.
//!
//! # The measured evidence behind this module
//!
//! Spike 013 (`.claude/skills/spike-findings-aprender/sources/013-np-events-autograd`):
//! on a 1200-point synthetic daily series of scale 30.43 with a planted `+8.0` per active
//! indicator, the six learned weights came back `+7.769, +8.293, +7.990, +8.043, +8.249,
//! +8.462` — worst error 0.462, 5.8 % of the planted effect. The tape measured 25 entries
//! per step with events against 20 without: a fixed `+5` that does NOT grow with the column
//! count, so `clear_graph()` still empties it. Seed 42 twice was bit-identical and seed 43
//! differed, so the determinism check can fail.
//!
//! Spike 013 measured the LAG-FREE path only. Plan 06.1-05 measured the AR paths, which D-30
//! put in scope, and the finding is recorded on
//! `events::tests::a_planted_effect_is_recovered_with_lags` and in the
//! `preconditions:` of `equations.event_effect_recovery_rel`: at `n_lags > 0` the event
//! block and the AR term are only jointly identified once the fit CONVERGES, and at the
//! epoch budget `np::door_epochs` would configure the per-column recovery bar does not hold.
//!
//! The bar those numbers are asserted against is `equations.event_effect_recovery_rel` in
//! `contracts/neuralprophet-parity-v1.yaml`, read at test time through
//! [`crate::test_support::equation_tolerance`]. It is never written as a literal here: a bar
//! that lives in a test can be loosened without the contract noticing (D-15).
//!
//! # How this module is reached
//!
//! From the door, since plan 06.1-06 (D-29). `forecast::forecast` accepts `holidays` on the
//! `"neuralprophet"` arm at `n_lags = 0`, builds an [`EventDesign`] from them and passes it
//! into every `np::TrainConfig`; the per-event contributions are published as response
//! components (D-31). Everything here is therefore on the request path, and the caller-facing
//! bounds that hold it are the four hoisted holiday bounds in `forecast::forecast`
//! (name length, window range, date counts, column count) plus the C-08 event-column price
//! against `MAX_NP_TRAIN_COST`.
//!
//! # What this module does NOT do
//!
//! It is not reachable at `n_lags > 0`: the door refuses that combination, because the event
//! block and the autoregressive term are jointly identified only once the fit converges, and
//! the per-column effect recovery that requires is not established at the door's epoch budget.

use crate::np::Rng;
use aprender::autograd::Tensor;
use aprender::nn::{Linear, Module};
use std::collections::HashSet;

/// One caller-supplied event: a name, the days it occurs on, and a window around each day.
///
/// Mirrors [`crate::types::HolidayArg`] field for field (with `days` already parsed to civil
/// day numbers), because the door hands ONE argument to both model arms.
#[derive(Clone, Debug)]
pub struct EventSpec {
    /// The event name. Becomes a per-event component key on the response (D-31, plan 06.1-06).
    pub name: String,
    /// The days the event occurs on, as day numbers from [`crate::dates::days_from_civil`].
    pub days: Vec<i64>,
    /// Days BEFORE each date to include (<= 0).
    pub lower_window: i64,
    /// Days AFTER each date to include (>= 0).
    pub upper_window: i64,
}

/// `(event index, offset)` per column, in the Prophet column order.
///
/// Events in INSERTION order; offsets ascending within an event. The order is the contract
/// between this expansion and the learned weight vector: `weights()[j]` is the effect of
/// `cols[j]`, and a reordering here silently relabels every per-event component.
#[must_use]
pub fn event_columns(events: &[EventSpec]) -> Vec<(usize, i64)> {
    let mut cols = Vec::new();
    for (ei, e) in events.iter().enumerate() {
        for off in e.lower_window..=e.upper_window {
            cols.push((ei, off));
        }
    }
    cols
}

/// One membership set per event, hoisted out of the row loop.
///
/// The `prophet::holiday_day_sets` pattern: building the set inside the row loop turns an
/// `O(rows * columns)` design build into `O(rows * columns * dates_per_event)`.
#[must_use]
pub fn event_day_sets(events: &[EventSpec]) -> Vec<HashSet<i64>> {
    events
        .iter()
        .map(|e| e.days.iter().copied().collect())
        .collect()
}

/// Push one indicator per column for `day` onto `out`.
///
/// `sets` must be the [`event_day_sets`] slice for the SAME event list `cols` was built
/// from — one set per event, in order.
///
/// # The lookup is TOTAL (IN-02), and the prototype's direct indexing is not
///
/// `ei` comes from `cols` and subscripts `sets`, a DIFFERENT argument that no type ties to
/// it. Spike 013's prototype wrote `sets[ei]`, which makes a mismatched pair an
/// out-of-bounds panic raised inside a library — the same defect `prophet::feature_row` was
/// converted away from. A mismatch yields a zero column here. Neither a panic nor a zero
/// column is CORRECT output for a caller who built the slices wrong; what the total form
/// buys is that a library does not abort the caller's process over it. Both in-crate callers
/// build both slices from the same [`EventDesign`], so this cannot change any shipped result.
pub fn event_row(day: i64, cols: &[(usize, i64)], sets: &[HashSet<i64>], out: &mut Vec<f32>) {
    for &(ei, off) in cols {
        // `d + off == day` iff `d == day - off`, the rearrangement `feature_row` uses.
        let hit = sets.get(ei).is_some_and(|s| s.contains(&(day - off)));
        out.push(f32::from(u8::from(hit)));
    }
}

/// The expanded event design: the specs, their columns and their hoisted membership sets.
///
/// Carrying the three together is what lets the PREDICT paths build an indicator row for a
/// future day. A block alone cannot: the weights say what an active column is worth, not
/// which days activate it.
#[derive(Clone, Debug)]
pub struct EventDesign {
    /// The caller's events, in insertion order.
    pub events: Vec<EventSpec>,
    /// `(event index, offset)` per column — [`event_columns`] of `events`.
    pub cols: Vec<(usize, i64)>,
    /// One membership set per event — [`event_day_sets`] of `events`.
    pub sets: Vec<HashSet<i64>>,
}

impl EventDesign {
    /// Expand a list of events into columns and hoisted membership sets, ONCE.
    #[must_use]
    pub fn new(events: Vec<EventSpec>) -> Self {
        let cols = event_columns(&events);
        let sets = event_day_sets(&events);
        EventDesign { events, cols, sets }
    }

    /// The number of indicator columns, i.e. the width of the event block.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.cols.len()
    }

    /// Push this design's indicator row for `day` onto `out`.
    pub fn row(&self, day: i64, out: &mut Vec<f32>) {
        event_row(day, &self.cols, &self.sets, out);
    }

    /// The row-major indicator matrix for `days`, `days.len() * dim()` entries.
    #[must_use]
    pub fn rows(&self, days: &[i64]) -> Vec<f32> {
        let mut out = Vec::with_capacity(days.len() * self.dim());
        for &day in days {
            self.row(day, &mut out);
        }
        out
    }

    /// The event NAME behind column `ci`, or `None` if `ci` is out of range.
    ///
    /// Total for the same reason [`event_row`] is: `cols` and `events` are separate fields on
    /// a struct whose fields are all `pub`.
    #[must_use]
    pub fn column_event_name(&self, ci: usize) -> Option<&str> {
        let (ei, _) = *self.cols.get(ci)?;
        self.events.get(ei).map(|e| e.name.as_str())
    }
}

/// The additive event term: `Linear(E, 1)` with no bias, trained beside `NpModel`.
///
/// No bias on purpose — `NpModel` already carries one, and a second additive constant is
/// unidentifiable against it.
pub struct EventBlock {
    lin: Linear,
    dim: usize,
}

impl EventBlock {
    /// Initialise a block of `dim` columns from the crate's `Rng`.
    ///
    /// `std = sqrt(1 / dim)` mirrors the seasonality block's `sqrt(1 / (2 * order))` scaling
    /// for a block of this width (`np::NpModel::new`), so the event term enters training at
    /// the same scale as every other linear term rather than dominating or vanishing.
    ///
    /// # Panics
    ///
    /// Panics if `dim` is zero: `Linear::without_bias(0, 1)` builds an empty weight whose
    /// transpose trips trueno's `Contract transpose: input is empty` deep inside the forward
    /// pass, which is a confusing place to learn that a caller passed no events. Callers hold
    /// `Option<EventBlock>` and construct `None` for an empty design.
    #[must_use]
    pub fn new(dim: usize, rng: &mut Rng) -> Self {
        assert!(
            dim > 0,
            "an EventBlock of width 0 is not a block: Linear::without_bias(0, 1) fails \
             inside the forward pass. Hold None for an empty design instead."
        );
        let mut lin = Linear::without_bias(dim, 1);
        let std = (1.0 / dim as f64).sqrt();
        lin.set_weight(
            Tensor::from_vec(
                (0..dim).map(|_| (rng.normal() * std) as f32).collect(),
                &[1, dim],
            )
            .requires_grad(),
        );
        EventBlock { lin, dim }
    }

    /// `[b, dim]` indicators to a `[b, 1]` additive contribution.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.lin.forward(x)
    }

    /// The block's parameter tensors, for extending into the ONE `AdamW` parameter vector.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.lin.parameters_mut()
    }

    /// The number of scalar parameters — `dim`, since there is no bias.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.lin.num_parameters()
    }

    /// The learned per-column weights, in [`event_columns`] order.
    #[must_use]
    pub fn weights(&self) -> Vec<f32> {
        self.lin.weight().data().to_vec()
    }

    /// The block's column count.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::{event_columns, event_day_sets, event_row, EventBlock, EventDesign, EventSpec};
    use crate::dates::days_from_civil;
    use crate::np::{
        door_epochs, door_lr_sweep, n_training_samples, train, NpData, NpModel, Rng, TrainConfig,
        TrainLog,
    };
    use crate::test_support::{constant_u64, equation_tolerance};

    /// The planted per-indicator magnitude. The recovery BAR is contract-read; this is the
    /// harness's own INPUT, which is not a bar and belongs here.
    const EFFECT: f64 = 8.0;
    /// Spike 013's series length, which this module keeps so its numbers are comparable.
    const POINTS: usize = 1_200;
    /// The epoch budget for the tests that are about WIRING (determinism, the tape, the
    /// optimiser vector) rather than about accuracy. Deliberately small: those tests do not
    /// read a weight's value, so paying for convergence in them buys nothing.
    const WIRING_EPOCHS: usize = 60;

    /// The two events spike 013 planted: a three-day promo window and a three-day
    /// Black-Friday window — six columns in all.
    ///
    /// Occurrences cover the whole 1200-day span. That is load-bearing rather than tidy: a
    /// fixed occurrence list on a LONGER series leaves its second half event-free, and
    /// `np::sample_weight` weights the second half MORE (`newer_w` = 2), so the probe would
    /// measure its own generator. Re-measured with and without the fix: at 2400 points
    /// lag-free, the worst recovery error moved from 0.034-0.038 to 0.017-0.025.
    fn planted_events() -> Vec<EventSpec> {
        events_spanning(POINTS)
    }

    /// [`planted_events`] with its occurrence list sized to cover `points` days from
    /// 2018-01-01.
    fn events_spanning(points: usize) -> Vec<EventSpec> {
        let last_year: i64 = 2018 + (points / 365) as i64 + 1;
        vec![
            EventSpec {
                name: "promo".into(),
                days: (2018..last_year)
                    .flat_map(|yr| [days_from_civil(yr, 3, 15), days_from_civil(yr, 9, 15)])
                    .collect(),
                lower_window: -1,
                upper_window: 1,
            },
            EventSpec {
                name: "blackfriday".into(),
                days: (2018..last_year)
                    .map(|yr| days_from_civil(yr, 11, 25))
                    .collect(),
                lower_window: 0,
                upper_window: 2,
            },
        ]
    }

    /// A synthetic daily series carrying a KNOWN additive effect on every active indicator,
    /// so "was it recovered?" has an answer rather than an impression.
    ///
    /// The effect is planted THROUGH the same [`EventDesign::row`] expansion the block reads,
    /// so a recovery result is a statement about the block and not about the harness agreeing
    /// with itself about which days are events.
    fn synthetic(events: &[EventSpec]) -> (Vec<i64>, Vec<f64>) {
        let start = days_from_civil(2018, 1, 1);
        let design = EventDesign::new(events.to_vec());
        let mut ds = Vec::with_capacity(POINTS);
        let mut y = Vec::with_capacity(POINTS);
        let mut rng = Rng::new(7);
        let mut row: Vec<f32> = Vec::new();
        for i in 0..POINTS {
            let day = start + i as i64;
            let t = i as f64;
            let mut v = 100.0
                + 0.02 * t
                + 6.0 * (2.0 * std::f64::consts::PI * t / 365.25).sin()
                + 3.0 * (2.0 * std::f64::consts::PI * t / 7.0).sin();
            row.clear();
            design.row(day, &mut row);
            v += EFFECT * f64::from(row.iter().sum::<f32>());
            v += 0.4 * rng.normal();
            ds.push(day);
            y.push(v);
        }
        (ds, y)
    }

    /// The series every test in this module trains on — ALWAYS the planted-six series, so an
    /// events-OFF control sees exactly the same data as its events-ON counterpart.
    fn planted_series() -> (NpData, Vec<EventSpec>) {
        let evs = planted_events();
        let (ds, y) = synthetic(&evs);
        (NpData::new(&ds, &y, ds.len(), 10, 0.9), evs)
    }

    fn cfg_for(
        events: &[EventSpec],
        n_lags: usize,
        seed: u64,
        max_lr: f64,
        epochs: usize,
    ) -> TrainConfig {
        TrainConfig {
            n_lags,
            ar_layers: if n_lags > 0 { vec![32] } else { vec![] },
            max_lr,
            epochs: Some(epochs),
            batch: None,
            weight_decay: 1e-3,
            huber_beta: 0.3,
            newer_w: 2.0,
            seed,
            event_design: if events.is_empty() {
                None
            } else {
                Some(EventDesign::new(events.to_vec()))
            },
            regressors: None,
        }
    }

    /// One fit at a pinned learning rate, for the tests about wiring rather than accuracy.
    fn fit(events: &[EventSpec], n_lags: usize, seed: u64) -> (NpData, TrainLog) {
        let (d, _) = planted_series();
        let cfg = cfg_for(events, n_lags, seed, 0.03, WIRING_EPOCHS);
        let (_m, log) = train(&d, &cfg, false);
        (d, log)
    }

    /// Fit the way the shipped recipe selects: sweep [`door_lr_sweep`] and keep the rate with
    /// the lowest FINAL TRAIN LOSS (D-10 — never the test error).
    ///
    /// The two recovery bars use this rather than a pinned rate, so what they certify is the
    /// shipped SELECTION RULE and not a hand-picked learning rate. It matters here: at
    /// `n_lags = 7`, a pinned 0.03 at the converged budget measured a worst relative error of
    /// 0.2743 on seed 7 while train-loss selection picked 0.1 and measured 0.0668.
    fn fit_lr_selected_by_train_loss(
        d: &NpData,
        events: &[EventSpec],
        n_lags: usize,
        seed: u64,
        epochs: usize,
    ) -> (f64, TrainLog) {
        let mut best: Option<(f64, f64, TrainLog)> = None;
        for &lr in door_lr_sweep(n_lags) {
            let cfg = cfg_for(events, n_lags, seed, lr, epochs);
            let (_m, log) = train(d, &cfg, false);
            let fl = *log.epoch_loss.last().unwrap_or(&f64::INFINITY);
            if fl.is_finite()
                && best
                    .as_ref()
                    .is_none_or(|b: &(f64, f64, TrainLog)| fl < b.0)
            {
                best = Some((fl, lr, log));
            }
        }
        let (_fl, lr, log) =
            best.expect("at least one rate in the shipped sweep must produce a finite loss");
        (lr, log)
    }

    /// Assert EVERY learned weight recovers the planted magnitude to the CONTRACT bar.
    ///
    /// Per column, never on an average: an average hides one column that learned nothing.
    fn assert_every_column_recovers(d: &NpData, log: &TrainLog, what: &str) {
        // The bar is READ, never written. A literal here could be loosened without
        // `contracts/neuralprophet-parity-v1.yaml` ever noticing (D-15).
        let bar = equation_tolerance("neuralprophet-parity-v1", "event_effect_recovery_rel");
        let weights = log.event_weights();
        assert_eq!(
            weights.len(),
            log.n_event_cols,
            "{what}: the trainer must record one weight per event column"
        );
        assert!(
            !weights.is_empty(),
            "{what}: zero event weights would make every assertion below vacuous"
        );
        let denorm: Vec<f64> = weights.iter().map(|w| f64::from(*w) * d.scale).collect();
        let mut worst = 0.0f64;
        let mut worst_col = 0usize;
        for (j, v) in denorm.iter().enumerate() {
            let rel = (v - EFFECT).abs() / EFFECT.abs();
            if rel > worst {
                worst = rel;
                worst_col = j;
            }
        }
        println!(
            "{what}: {} columns, worst relative recovery error {worst:.4} on column \
             {worst_col} against the contract bar {bar}; denormalised weights {denorm:?}",
            denorm.len()
        );
        for (j, v) in denorm.iter().enumerate() {
            let rel = (v - EFFECT).abs() / EFFECT.abs();
            assert!(
                rel <= bar,
                "{what}: event column {j} recovered {v:+.3} against a planted {EFFECT:+.1} — \
                 a relative error of {rel:.4}, over the contract bar {bar}. All denormalised \
                 weights: {denorm:?}"
            );
        }
    }

    /// SC3, lag-free, at the budget the DOOR itself would configure.
    ///
    /// `np::door_epochs` for this geometry is 110, and the bar holds there with room to
    /// spare: measured worst relative error 0.0597-0.0628 across seeds 42, 43, 7 and 11.
    #[test]
    fn a_planted_effect_is_recovered_lag_free() {
        let (d, evs) = planted_series();
        let epochs = door_epochs(POINTS, n_training_samples(&d, 0), 0);
        let (lr, log) = fit_lr_selected_by_train_loss(&d, &evs, 0, 42, epochs);
        println!("lag-free: {epochs} epochs (np::door_epochs), train-loss-selected lr {lr}");
        assert_every_column_recovers(&d, &log, "lag-free");
    }

    /// SC3 at `n_lags = 7` — D-30's substance, and the half spike 013 never measured.
    ///
    /// # The budget is the CONVERGED one, and that is a finding rather than a convenience
    ///
    /// This test trains for `constants.auto_epochs_cap_with_lags` epochs, NOT for the
    /// `np::door_epochs` budget its lag-free twin uses. MEASURED 2026-09-21 on this series,
    /// with the learning rate selected by train loss in every cell (worst relative error per
    /// column, four seeds):
    ///
    /// | points | epochs | seed 42 | seed 43 | seed 7 | seed 11 |
    /// |---|---|---|---|---|---|
    /// | 1200 | 110 (`door_epochs`) | 0.0915 | 0.2293 | 0.4660 | 0.3326 |
    /// | 1200 | 320 (the AR cap) | **0.0528** | **0.0659** | **0.0668** | **0.0667** |
    /// | 2400 |  90 (`door_epochs`) | 0.2191 | 0.3597 | 0.3897 | 0.2954 |
    /// | 2400 | 320 (the AR cap) | **0.0677** | **0.0628** | 0.1663 | **0.0901** |
    ///
    /// So the bar holds at the AR cap on this geometry and does NOT hold at the budget the
    /// door would configure — and at 2400 points one seed is still over even at the cap. The
    /// mechanism is visible in every failing row and is exactly the one D-30 predicted: the
    /// FIRST column of each event, whose seven-day lag window contains no earlier day of the
    /// same window, recovers correctly at every budget (7.7-8.7 against a planted 8.0), while
    /// the TRAILING columns, whose lag windows overlap the event, drift anywhere from 4.3 to
    /// 10.7. The AR term can read the effect out of the lags, so the split between the two
    /// terms is not identified until the fit converges.
    ///
    /// This is recorded as an OPEN ITEM for plan 06.1-06 (`preconditions` on
    /// `equations.event_effect_recovery_rel`), which is the plan that opens the
    /// `neuralprophet` arm to `holidays` and therefore chooses the epoch budget the DOOR
    /// configures. Nothing here reaches a caller: the arm is still refused.
    #[test]
    fn a_planted_effect_is_recovered_with_lags() {
        let (d, evs) = planted_series();
        let epochs = constant_u64("neuralprophet-parity-v1", "auto_epochs_cap_with_lags") as usize;
        let door = door_epochs(POINTS, n_training_samples(&d, 7), 7);
        assert!(
            epochs > door,
            "this test's premise is that the AR arm needs MORE than the door's budget \
             ({door}); if the contract's cap ({epochs}) no longer exceeds it, the finding \
             recorded on this test has changed and must be re-measured"
        );
        let (lr, log) = fit_lr_selected_by_train_loss(&d, &evs, 7, 42, epochs);
        println!(
            "n_lags=7: {epochs} epochs (constants.auto_epochs_cap_with_lags; door_epochs \
             would be {door}), train-loss-selected lr {lr}"
        );
        assert_every_column_recovers(&d, &log, "n_lags=7");
    }

    /// The CONTROL that stops a recovery bar passing on a block that learned nothing useful.
    ///
    /// A recovery bar only inspects the weights. This inspects the FIT: events ON must reach
    /// a lower final train loss than events OFF on the same series and the same seed.
    #[test]
    fn events_on_beats_events_off_on_the_same_seed() {
        let evs = planted_events();
        let (_d, on) = fit(&evs, 0, 42);
        let (_d2, off) = fit(&[], 0, 42);
        let on_loss = *on.epoch_loss.last().expect("events ON trained some epochs");
        let off_loss = *off
            .epoch_loss
            .last()
            .expect("events OFF trained some epochs");
        println!(
            "final train loss: events ON {on_loss:.6} ({} cols) vs events OFF {off_loss:.6}",
            on.n_event_cols
        );
        assert_eq!(
            off.n_event_cols, 0,
            "the OFF control must carry no event columns"
        );
        assert!(
            on_loss < off_loss,
            "events ON must fit the planted effect BETTER than events OFF on the same seed: \
             ON {on_loss:.6} vs OFF {off_loss:.6}. A recovery bar that passes while this \
             fails means the weights look right and the model does not"
        );
    }

    /// Determinism WITH a falsification control: seed 43 must differ, or the check is vacuous.
    ///
    /// The bar is `equations.event_training_determinism_bitwise`, read from the contract
    /// rather than written here. It is EXACTLY `0.0` and the comparison is on BIT PATTERNS:
    /// `f64` equality would call `-0.0` and `+0.0` equal and two NaNs unequal, and neither
    /// is the statement this bar makes. Reading it keeps the number in one place even
    /// though a zero tolerance looks like it could not drift — a later loosening to `1e-12`
    /// would then have to happen in the contract, where it is visible.
    #[test]
    fn fixed_seed_event_training_is_bit_identical_and_a_different_seed_differs() {
        let tol = equation_tolerance(
            "neuralprophet-parity-v1",
            "event_training_determinism_bitwise",
        );
        assert_eq!(
            tol, 0.0,
            "this check compares BIT PATTERNS, which is only the right instrument at a \
             tolerance of exactly 0.0; the contract says {tol}"
        );
        let evs = planted_events();
        let (_d, a) = fit(&evs, 0, 42);
        let (_d, b) = fit(&evs, 0, 42);
        let (_d, c) = fit(&evs, 0, 43);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&a.event_weights()),
            bits(&b.event_weights()),
            "seed 42 twice must produce BIT-IDENTICAL event weights"
        );
        assert_eq!(
            a.epoch_loss.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.epoch_loss.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "seed 42 twice must produce a BIT-IDENTICAL loss trace"
        );
        assert_ne!(
            bits(&a.event_weights()),
            bits(&c.event_weights()),
            "seed 43 must DIFFER from seed 42, or this determinism check cannot fail and \
             proves nothing"
        );
    }

    /// The block's parameters are actually IN the vector handed to `AdamW`.
    ///
    /// Without this, a block that is built and forwarded but never extended into the
    /// optimiser's parameter vector still contributes to the forward pass — its weights
    /// simply never move — and the recovery bar would then fail with a confusing "effect not
    /// recovered" rather than the true "the block was never optimised". This test names the
    /// real cause.
    #[test]
    fn the_event_block_weights_are_actually_optimised() {
        let (d, evs) = planted_series();
        let dim = EventDesign::new(evs.clone()).dim();

        // ONE optimiser step is enough: the question is whether the weights MOVE.
        let cfg = TrainConfig {
            batch: Some(64),
            ..cfg_for(&evs, 0, 42, 0.03, 1)
        };
        let (_m, log) = train(&d, &cfg, false);

        // The initial weights, reproduced from the SAME rng sequence the trainer uses: a
        // fresh Rng(seed), an NpModel to consume the model's draws, then the block.
        let mut rng = Rng::new(42);
        let _model = NpModel::new(&d, 0, &[], &mut rng);
        let before = EventBlock::new(dim, &mut rng).weights();
        let after = log.event_weights();
        assert_eq!(before.len(), after.len(), "the block width must not change");
        assert!(
            before
                .iter()
                .zip(&after)
                .any(|(x, y)| x.to_bits() != y.to_bits()),
            "not one event weight moved across {} optimiser step(s): the block is built and \
             forwarded but its parameters were never extended into the AdamW vector. \
             before {before:?} after {after:?}",
            log.steps
        );

        // ...and the vector handed to AdamW carries the block. `n_opt_tensors` is recorded by
        // the trainer at the moment the optimiser is constructed, so this is the vector that
        // was actually handed over and not a reconstruction of it.
        let (_d2, off) = fit(&[], 0, 42);
        assert_eq!(
            log.n_opt_tensors,
            off.n_opt_tensors + 1,
            "the AdamW parameter vector must carry the block's ONE weight tensor beside the \
             model's: events ON handed over {} tensors, events OFF {}",
            log.n_opt_tensors,
            off.n_opt_tensors
        );
        assert_eq!(
            log.n_params,
            off.n_params + dim,
            "the recorded parameter COUNT must grow by exactly the block's width"
        );
    }

    /// The tape does not grow with the event-column count, so `clear_graph()` still empties it.
    ///
    /// Two VERY different column counts against the events-OFF baseline, asserting the same
    /// fixed increment at both. A single column count could not tell a fixed increment from a
    /// linear one.
    ///
    /// The bar is `equations.event_tape_length_flat_in_columns`, read from the contract. It
    /// is `0.0` because this is a COUNT: the increment is equal or it is not, and an
    /// approximate tape length would not mean anything.
    #[test]
    fn the_tape_does_not_grow_with_the_event_column_count() {
        let tol = equation_tolerance(
            "neuralprophet-parity-v1",
            "event_tape_length_flat_in_columns",
        );
        assert_eq!(
            tol, 0.0,
            "a tape length is a COUNT, so the only meaningful tolerance is exactly 0.0; \
             the contract says {tol}"
        );
        // Two events x 3 offsets = 6 columns, against 40 events x 7 offsets = 280 columns.
        let wide: Vec<EventSpec> = (0..40)
            .map(|k: u32| EventSpec {
                name: format!("e{k}"),
                days: (2018..2022)
                    .map(|yr| days_from_civil(yr, 1 + (k % 12), 1 + (k % 28)))
                    .collect(),
                lower_window: -3,
                upper_window: 3,
            })
            .collect();
        let narrow = planted_events();

        let (_d, off) = fit(&[], 0, 42);
        let (_d, small) = fit(&narrow, 0, 42);
        let (_d, large) = fit(&wide, 0, 42);

        assert_eq!(small.n_event_cols, 6, "the narrow design is six columns");
        assert_eq!(large.n_event_cols, 280, "the wide design is 280 columns");
        let d_small = small.tape_len_per_step as i64 - off.tape_len_per_step as i64;
        let d_large = large.tape_len_per_step as i64 - off.tape_len_per_step as i64;
        println!(
            "tape per step: OFF {} | E=6 {} (+{d_small}) | E=280 {} (+{d_large})",
            off.tape_len_per_step, small.tape_len_per_step, large.tape_len_per_step
        );
        assert!(
            d_small > 0,
            "the event block must add SOMETHING to the tape, or this test is measuring a \
             block that never ran"
        );
        assert_eq!(
            d_small, d_large,
            "the per-step tape increment must be the SAME at 6 and at 280 event columns. It \
             is {d_small} and {d_large}, which means the tape grows with the column count \
             and `clear_graph()` is clearing an ever-larger graph"
        );
    }

    /// The expansion order: events in insertion order, offsets ascending within an event.
    #[test]
    fn event_columns_are_in_insertion_order_with_ascending_offsets() {
        let evs = vec![
            EventSpec {
                name: "b_second_by_name".into(),
                days: vec![10],
                lower_window: -2,
                upper_window: 1,
            },
            EventSpec {
                name: "a_first_by_name".into(),
                days: vec![20],
                lower_window: 0,
                upper_window: 2,
            },
        ];
        let cols = event_columns(&evs);
        assert_eq!(
            cols,
            vec![(0, -2), (0, -1), (0, 0), (0, 1), (1, 0), (1, 1), (1, 2)],
            "INSERTION order, not name order: the first-listed event owns the first columns, \
             and its offsets ascend from lower_window to upper_window"
        );

        // ...and the indicators land on the columns that order names. Day 10 is
        // `b_second_by_name` at offset 0, which is column index 2.
        let sets = event_day_sets(&evs);
        let mut row = Vec::new();
        event_row(10, &cols, &sets, &mut row);
        assert_eq!(row, vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        row.clear();
        // Day 21 is `a_first_by_name` at offset +1: column 5.
        event_row(21, &cols, &sets, &mut row);
        assert_eq!(row, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

        // The design agrees with the free functions, and names the column's owning event.
        let design = EventDesign::new(evs);
        assert_eq!(design.dim(), 7);
        assert_eq!(design.column_event_name(0), Some("b_second_by_name"));
        assert_eq!(design.column_event_name(6), Some("a_first_by_name"));
        assert_eq!(
            design.column_event_name(7),
            None,
            "total lookup past the end"
        );
    }

    /// IN-02: a mismatched `sets` slice yields a zero column, never a panic in a library.
    #[test]
    fn event_row_is_a_total_lookup_on_a_mismatched_slice() {
        let evs = planted_events();
        let cols = event_columns(&evs);
        let mut row = Vec::new();
        // The caller cleared the events but kept the columns — the shape `feature_row`'s
        // IN-02 comment describes, reachable here because every EventDesign field is `pub`.
        event_row(days_from_civil(2018, 3, 15), &cols, &[], &mut row);
        assert_eq!(row.len(), cols.len(), "one entry per column, still");
        assert!(
            row.iter().all(|v| *v == 0.0),
            "a mismatched membership slice must produce zero columns, not an out-of-bounds \
             panic raised inside a library"
        );
    }

    /// An event on a FUTURE day moves the forecast on all three predict paths.
    ///
    /// D-30's other half: spike 013 measured only training. A block that trains but is not
    /// added on the predict path learns the effect and then never emits it, which no
    /// recovery bar can see.
    #[test]
    fn a_future_event_moves_every_predict_path() {
        let (d, evs) = planted_series();
        let design = EventDesign::new(evs.clone());

        for n_lags in [0usize, 7] {
            let cfg = cfg_for(&evs, n_lags, 42, 0.03, WIRING_EPOCHS);
            let (m, log) = train(&d, &cfg, false);
            let block = log.events.as_ref().expect("a block was trained");

            // A future window that IS an event (the promo days the generator plants) against
            // one that is not, on the same horizon.
            let last = d.grid_days[d.grid_days.len() - 1];
            // The horizon is DERIVED so it reaches the next event day rather than being a
            // round number that might miss one. A fixed 40-day horizon did exactly that:
            // the series ends 2021-04-14 and the next planted occurrence is 2021-09-15.
            let is_event = |day: i64| {
                let mut r = Vec::new();
                design.row(day, &mut r);
                r.iter().sum::<f32>() > 0.0
            };
            let reach = (1..=400i64)
                .find(|k| is_event(last + k))
                .expect("a planted occurrence must fall within 400 days of the series end");
            let future: Vec<i64> = (1..=reach + 3).map(|k| last + k).collect();
            let hit = future.iter().filter(|day| is_event(**day)).count();
            assert!(
                hit > 0,
                "the probe horizon must contain at least one event day, or it cannot \
                 distinguish a wired predict path from an unwired one"
            );

            let (with, without) = if n_lags == 0 {
                (
                    crate::np::predict_ts(&d, &m, &future, Some((&design, block)), None),
                    crate::np::predict_ts(&d, &m, &future, None, None),
                )
            } else {
                (
                    crate::np::predict_ar_recursive(&d, &m, &future, Some((&design, block)), None),
                    crate::np::predict_ar_recursive(&d, &m, &future, None, None),
                )
            };
            let moved = with
                .iter()
                .zip(&without)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            println!(
                "n_lags={n_lags}: {hit} event day(s) in the horizon, {moved} of {} predicted \
                 rows moved when the event channel is supplied",
                future.len()
            );
            assert!(
                moved >= hit,
                "supplying the event channel must move at least the {hit} event day(s) it \
                 covers; only {moved} row(s) changed, so the predict path is not adding the \
                 block's contribution"
            );

            // The one-step path, on HISTORY rows, for the lagged arm.
            if n_lags > 0 {
                let idx: Vec<usize> = (n_lags..d.grid_days.len().min(400)).collect();
                let a = crate::np::predict_ar_1step(&d, &m, &idx, Some((&design, block)), None);
                let b = crate::np::predict_ar_1step(&d, &m, &idx, None, None);
                let moved_1step = a
                    .iter()
                    .zip(&b)
                    .filter(|(x, y)| x.to_bits() != y.to_bits())
                    .count();
                assert!(
                    moved_1step > 0,
                    "predict_ar_1step must add the event contribution too, or the fitted \
                     values the residual band is built from ignore every event"
                );
            }
        }
    }
}
