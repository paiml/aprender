//! Histogram legend and histogram brush (APR-MAP-001 MAP-7 slice 1).
//!
//! The histogram bins a lens over its domain: equal-width bins on a linear scale, equal-width
//! bins in `log10` space on a log scale. A cell with data is one whose value has a `color_t`.
//! The brush selects cells by raw value; both bounds are inclusive.

use std::collections::BTreeSet;

use super::lens::Lens;
use super::{EntityId, MapError};

/// Bin count of the legend histogram.
pub const HISTOGRAM_BINS: usize = 16;

/// Equal-width bin counts over a lens domain.
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    edges: Vec<f64>,
    counts: Vec<u32>,
}

impl Histogram {
    /// Bins `lens` into `bins` bins. RED stub: returns an empty histogram of the right shape.
    pub fn of(lens: &Lens, bins: usize) -> Result<Self, MapError> {
        let _ = lens;
        Ok(Self {
            edges: vec![0.0; bins + 1],
            counts: vec![0; bins],
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
}

/// A brush over value space: the closed interval `[lo, hi]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Brush {
    lo: f64,
    hi: f64,
}

impl Brush {
    /// A brush over `[lo, hi]`. RED stub: no validation yet.
    pub fn new(lo: f64, hi: f64) -> Result<Self, MapError> {
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

    /// True when `v` lies in `[lo, hi]`. RED stub: always false.
    pub fn contains(&self, v: f64) -> bool {
        let _ = v;
        false
    }

    /// Ids of the cells with data whose value the brush contains. RED stub: empty.
    pub fn select(&self, lens: &Lens) -> BTreeSet<EntityId> {
        let _ = lens;
        BTreeSet::new()
    }
}

#[cfg(test)]
#[path = "histogram_tests.rs"]
mod tests;
