//! Histogram legend and histogram brush (APR-MAP-001 MAP-7 slice 1).
//!
//! The histogram bins a lens over its domain: equal-width bins on a linear scale, equal-width
//! bins in `log10` space on a log scale. A cell with data is one whose value has a `color_t`.
//! The brush selects cells by raw value; both bounds are inclusive.

use std::collections::BTreeSet;

use super::lens::{Lens, Scale};
use super::{check_finite, EntityId, MapError};

/// Bin edges over `(lo, hi)`: equal steps in value space on a linear scale, equal steps in
/// `log10` space on a log scale. `bins + 1` values.
fn edges_for(scale: Scale, (lo, hi): (f64, f64), bins: usize) -> Vec<f64> {
    let steps = (0..=bins).map(|i| i as f64 / bins as f64);
    match scale {
        Scale::Linear => steps.map(|s| lo + s * (hi - lo)).collect(),
        Scale::Log => {
            let (llo, lhi) = (lo.log10(), hi.log10());
            steps.map(|s| 10f64.powf(llo + s * (lhi - llo))).collect()
        }
    }
}

/// Bin index of a position `t` in `[0, 1]`, with `t == 1` in the last bin.
fn bin_of(t: f64, bins: usize) -> usize {
    let scaled = (t * bins as f64).floor() as usize;
    scaled.min(bins - 1)
}

/// Bin count of the legend histogram.
pub const HISTOGRAM_BINS: usize = 16;

/// Equal-width bin counts over a lens domain.
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    edges: Vec<f64>,
    counts: Vec<u32>,
}

impl Histogram {
    /// Bins `lens` into `bins` equal-width bins over its domain. A cell counts when its value has
    /// a position on the lens scale, so the counts sum to the cells with data.
    pub fn of(lens: &Lens, bins: usize) -> Result<Self, MapError> {
        if bins == 0 {
            return Err(MapError::NonPositive("bins"));
        }
        let mut counts = vec![0u32; bins];
        for t in lens
            .values()
            .iter()
            .flatten()
            .filter_map(|&v| lens.color_t(v))
        {
            let bin = bin_of(t, bins);
            counts[bin] = counts[bin].saturating_add(1);
        }
        Ok(Self {
            edges: edges_for(lens.scale(), lens.domain(), bins),
            counts,
        })
    }

    /// Bin edges, `bins + 1` values in ascending order.
    pub fn edges(&self) -> &[f64] {
        &self.edges
    }

    /// Count of cells with data per bin.
    pub fn counts(&self) -> &[u32] {
        &self.counts
    }

    /// Sum of all bin counts.
    pub fn total(&self) -> u64 {
        self.counts.iter().map(|&c| u64::from(c)).sum()
    }

    /// Moves one cell from the bin of its old value to the bin of its new one, as `of` would bin
    /// them. A value with no position on the lens scale holds no bin, on either side.
    pub(super) fn move_cell(&mut self, lens: &Lens, old: Option<f64>, new: Option<f64>) {
        let bins = self.counts.len();
        if let Some(bin) = old.and_then(|v| lens.color_t(v)).map(|t| bin_of(t, bins)) {
            self.counts[bin] = self.counts[bin].saturating_sub(1);
        }
        if let Some(bin) = new.and_then(|v| lens.color_t(v)).map(|t| bin_of(t, bins)) {
            self.counts[bin] = self.counts[bin].saturating_add(1);
        }
    }
}

/// A brush over value space: the closed interval `[lo, hi]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Brush {
    lo: f64,
    hi: f64,
}

impl Brush {
    /// A brush over `[lo, hi]`. Refuses a non-finite bound, and `lo > hi`; `lo == hi` is a
    /// single value.
    pub fn new(lo: f64, hi: f64) -> Result<Self, MapError> {
        check_finite(lo, "brush")?;
        check_finite(hi, "brush")?;
        if lo > hi {
            return Err(MapError::InvertedBounds);
        }
        Ok(Self { lo, hi })
    }

    /// Inclusive lower bound.
    pub fn lo(&self) -> f64 {
        self.lo
    }

    /// Inclusive upper bound.
    pub fn hi(&self) -> f64 {
        self.hi
    }

    /// True when `v` lies in `[lo, hi]`, both bounds included.
    pub fn contains(&self, v: f64) -> bool {
        (self.lo..=self.hi).contains(&v)
    }

    /// Ids of the cells with data whose value the brush contains. Cells without data are never
    /// selected, whatever the bounds.
    pub fn select(&self, lens: &Lens) -> BTreeSet<EntityId> {
        lens.values()
            .iter()
            .enumerate()
            .filter_map(|(cell, v)| {
                let v = (*v)?;
                let id = u64::try_from(cell).ok()?;
                (lens.color_t(v).is_some() && self.contains(v)).then_some(id)
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "histogram_tests.rs"]
mod tests;
