//! Lens registry (APR-MAP-001 MAP-7 slice 1): one value per hex cell, a scale and a stated domain.
//!
//! Headless and target-neutral. `color_t` maps a value to a position in `[0, 1]` on the lens
//! scale; a value that has no position (non-finite, or non-positive on a log scale) yields `None`.

use std::collections::BTreeMap;

use super::{LensId, MapError};

/// How a lens maps values to a position in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scale {
    Linear,
    Log,
}

/// A lens: one optional value per cell, indexed by cell id.
#[derive(Debug, Clone, PartialEq)]
pub struct Lens {
    id: LensId,
    name: String,
    scale: Scale,
    domain: (f64, f64),
    values: Vec<Option<f64>>,
}

impl Lens {
    /// Builds a lens. RED stub: performs no validation yet.
    pub fn new(
        id: LensId,
        name: &str,
        scale: Scale,
        domain: (f64, f64),
        values: Vec<Option<f64>>,
    ) -> Result<Self, MapError> {
        Ok(Self {
            id,
            name: name.to_owned(),
            scale,
            domain,
            values,
        })
    }

    /// Lens id.
    pub fn id(&self) -> LensId {
        self.id
    }

    /// Display name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Scale.
    pub fn scale(&self) -> Scale {
        self.scale
    }

    /// The stated `(lo, hi)` domain.
    pub fn domain(&self) -> (f64, f64) {
        self.domain
    }

    /// One value per cell; `None` is no data.
    pub fn values(&self) -> &[Option<f64>] {
        &self.values
    }

    /// Position of `v` on the lens scale in `[0, 1]`. RED stub: always `None`.
    pub fn color_t(&self, _v: f64) -> Option<f64> {
        None
    }
}

/// Registry of lenses for one grid. Every lens covers exactly `cells` cells.
#[derive(Debug, Clone, PartialEq)]
pub struct LensRegistry {
    cells: u32,
    lenses: BTreeMap<LensId, Lens>,
}

impl LensRegistry {
    /// An empty registry for a grid of `cells` cells.
    pub fn new(cells: u32) -> Self {
        Self {
            cells,
            lenses: BTreeMap::new(),
        }
    }

    /// Cell count every registered lens must cover.
    pub fn cells(&self) -> u32 {
        self.cells
    }

    /// Registers a lens. RED stub: stores without checks.
    pub fn add(&mut self, lens: Lens) -> Result<(), MapError> {
        self.lenses.insert(lens.id, lens);
        Ok(())
    }

    /// The lens with this id.
    pub fn get(&self, id: LensId) -> Option<&Lens> {
        self.lenses.get(&id)
    }

    /// Number of registered lenses.
    pub fn len(&self) -> usize {
        self.lenses.len()
    }

    /// True when no lens is registered.
    pub fn is_empty(&self) -> bool {
        self.lenses.is_empty()
    }
}

#[cfg(test)]
#[path = "lens_tests.rs"]
mod tests;
