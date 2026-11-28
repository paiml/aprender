//! Headless lens client (APR-MAP-001 MAP-8 slice 1): applies lens deltas to a registry and keeps
//! each lens histogram current. No socket, no clock, no drawing of its own.
//!
//! RED stub: `apply` changes nothing yet. The sequence rules and the registry checks are in the
//! tests.

use std::collections::BTreeMap;

use super::{
    camera::Camera2D,
    delta::Delta,
    hex::HexGrid,
    histogram::{Histogram, HISTOGRAM_BINS},
    interact::{Interaction, Report},
    layers::{LensId, Renders},
    lens::LensRegistry,
    urlstate::Nav,
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
    /// RED stub: builds without the registry and active-lens checks.
    pub fn new(
        grid: HexGrid,
        camera: Camera2D,
        lenses: LensRegistry,
        active: Option<LensId>,
    ) -> Result<(Self, Report), MapError> {
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

    /// RED stub: applies nothing.
    pub fn apply(&mut self, delta: &Delta) -> Result<Report, MapError> {
        let _ = delta;
        Ok(Report {
            renders: Renders::default(),
            frame_requested: false,
            nav: Nav::Nothing,
        })
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

#[cfg(test)]
#[path = "client_tests.rs"]
mod tests;
