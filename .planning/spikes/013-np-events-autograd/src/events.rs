//! The prototype: NeuralProphet `add_events` as one additive `Linear(E, 1)` block over
//! indicator columns, composed BESIDE the shipped `NpModel` rather than forked into it.
//!
//! One indicator per (event, window offset) on the daily grid, exactly as the Prophet arm
//! already expands `HolidayArg` — so the same argument serves both models, which is the
//! change request's stated requirement.

use aprender::autograd::Tensor;
use aprender::nn::{Linear, Module};
use aprender_forecast::np::Rng;

#[derive(Clone, Debug)]
pub struct EventSpec {
    pub name: String,
    pub days: Vec<i64>,
    pub lower_window: i64,
    pub upper_window: i64,
}

/// `(event index, offset)` per column, in the Prophet column order: events in insertion
/// order, offsets ascending within an event.
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

/// One membership set per event, hoisted out of the row loop — the `holiday_day_sets`
/// pattern the Prophet arm already uses.
#[must_use]
pub fn event_day_sets(events: &[EventSpec]) -> Vec<std::collections::HashSet<i64>> {
    events.iter().map(|e| e.days.iter().copied().collect()).collect()
}

pub fn event_row(
    day: i64,
    cols: &[(usize, i64)],
    sets: &[std::collections::HashSet<i64>],
    out: &mut Vec<f32>,
) {
    for &(ei, off) in cols {
        // `d + off == day` iff `d == day - off`, the same rearrangement `feature_row` uses.
        out.push(f32::from(u8::from(sets[ei].contains(&(day - off)))));
    }
}

pub struct EventBlock {
    pub lin: Linear,
    pub dim: usize,
}

impl EventBlock {
    /// NP initialises an additive event block like any other linear term. `std = sqrt(1/dim)`
    /// mirrors the seasonality block's `sqrt(1/(2*order))` scaling for a block of this width.
    pub fn new(dim: usize, rng: &mut Rng) -> Self {
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
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.lin.forward(x)
    }
    pub fn n_params(&self) -> usize {
        self.lin.num_parameters()
    }
    pub fn weights(&self) -> Vec<f32> {
        self.lin.weight().data().to_vec()
    }
}
