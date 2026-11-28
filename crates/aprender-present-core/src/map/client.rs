//! Headless lens client (APR-MAP-001 MAP-8 slice 1): applies lens deltas to a registry and keeps
//! each lens histogram current. No socket, no clock, no drawing of its own.
//!
//! A delta is applied all or nothing. Its sequence number must be exactly one after the last one
//! applied to its lens; a gap is refused, not skipped. Every cell is checked before any value is
//! written.

use std::collections::BTreeMap;

use super::{
    camera::Camera2D,
    delta::Delta,
    hex::HexGrid,
    histogram::{Histogram, HISTOGRAM_BINS},
    interact::{Interaction, Report},
    layers::LensId,
    lens::LensRegistry,
    MapError,
};

/// Headless client: a grid, its lenses and their histograms, and the last sequence per lens.
pub struct Client {
    map: Interaction,
    lenses: LensRegistry,
    histograms: BTreeMap<LensId, Histogram>,
    applied: BTreeMap<LensId, u64>,
}

impl Client {
    /// Starts a client over `grid`. Refuses a registry that does not cover the grid's cells, and
    /// an active lens the registry does not hold.
    pub fn new(
        grid: HexGrid,
        camera: Camera2D,
        lenses: LensRegistry,
        active: Option<LensId>,
    ) -> Result<(Self, Report), MapError> {
        if lenses.cells() != grid.cell_count() {
            return Err(MapError::Mismatch("registry"));
        }
        if let Some(id) = active.filter(|&id| lenses.get(id).is_none()) {
            return Err(MapError::UnknownLens(id));
        }
        let histograms = lenses
            .iter()
            .map(|lens| Histogram::of(lens, HISTOGRAM_BINS).map(|h| (lens.id(), h)))
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let (map, report) = Interaction::new(grid, camera, active);
        Ok((
            Self {
                map,
                lenses,
                histograms,
                applied: BTreeMap::new(),
            },
            report,
        ))
    }

    /// Switches the active lens; refuses an id the registry does not hold.
    pub fn switch_lens(&mut self, lens: LensId, now_ms: u64) -> Result<Report, MapError> {
        self.map.switch_lens(&self.lenses, lens, now_ms)
    }

    /// Applies one delta: writes its cells, keeps the histogram equal to a rebuild, records the
    /// sequence number and tells the scheduler. Refuses, with no change at all, an unregistered
    /// lens, a sequence number that is not the next one, and a cell outside the grid.
    pub fn apply(&mut self, delta: &Delta) -> Result<Report, MapError> {
        let id = delta.lens();
        if self.lenses.get(id).is_none() {
            return Err(MapError::UnknownLens(id));
        }
        check_sequence(delta.seq(), self.last_seq(id))?;
        let cells = self.lenses.cells();
        let writes = delta
            .cells()
            .iter()
            .map(|&(cell, value)| -> Result<(usize, Option<f64>), MapError> {
                let index = usize::try_from(cell)
                    .ok()
                    .filter(|_| cell < cells)
                    .ok_or(MapError::CellOutOfRange(cell))?;
                Ok((index, value))
            })
            .collect::<Result<Vec<_>, MapError>>()?;

        let lens = self.lenses.get_mut(id).ok_or(MapError::UnknownLens(id))?;
        let hist = self
            .histograms
            .get_mut(&id)
            .ok_or(MapError::UnknownLens(id))?;
        for (index, value) in writes {
            let old = lens.values().get(index).copied().flatten();
            hist.move_cell(lens, old, value);
            lens.set_value(index, value);
        }
        self.applied.insert(id, delta.seq());
        self.map.lens_delta(id)
    }

    /// The interaction core.
    pub fn map(&self) -> &Interaction {
        &self.map
    }

    /// The lens registry, with every applied delta written in.
    pub fn lenses(&self) -> &LensRegistry {
        &self.lenses
    }

    /// Histogram of `lens`, kept equal to a rebuild from scratch.
    pub fn histogram(&self, lens: LensId) -> Option<&Histogram> {
        self.histograms.get(&lens)
    }

    /// Sequence number of the last delta applied to `lens`; 0 before any.
    pub fn last_seq(&self, lens: LensId) -> u64 {
        self.applied.get(&lens).copied().unwrap_or(0)
    }
}

/// Checks a delta's sequence number against the last one applied to its lens. It must be exactly
/// one more: an equal or earlier number is stale, a later one is a gap.
fn check_sequence(seq: u64, last: u64) -> Result<(), MapError> {
    if seq <= last {
        return Err(MapError::Stale { seq, last });
    }
    // `last < seq <= u64::MAX`, so `last + 1` cannot overflow.
    let expected = last + 1;
    if seq == expected {
        Ok(())
    } else {
        Err(MapError::Gap { expected, seq })
    }
}

#[cfg(test)]
#[path = "client_tests.rs"]
mod tests;
