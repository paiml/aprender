//! Lens registry (APR-MAP-001 MAP-7 slice 1): one value per hex cell, a scale and a stated domain.
//!
//! Headless and target-neutral. `color_t` maps a value to a position in `[0, 1]` on the lens
//! scale; a value that has no position (non-finite, or non-positive on a log scale) yields `None`.

use std::collections::BTreeMap;

use super::{check_finite, LensId, MapError};

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
    /// Builds a lens. Refuses a domain that is not finite or not strictly ordered, and a log
    /// domain whose lower bound is not positive.
    pub fn new(
        id: LensId,
        name: &str,
        scale: Scale,
        domain: (f64, f64),
        values: Vec<Option<f64>>,
    ) -> Result<Self, MapError> {
        check_finite(domain.0, "domain")?;
        check_finite(domain.1, "domain")?;
        if domain.0 >= domain.1 {
            return Err(MapError::EmptyDomain);
        }
        if scale == Scale::Log && domain.0 <= 0.0 {
            return Err(MapError::NonPositive("domain"));
        }
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

    /// Replaces the value of the cell at `index`. The caller has checked the index against the grid,
    /// and every registered lens covers the same cells.
    pub(super) fn set_value(&mut self, index: usize, value: Option<f64>) {
        if let Some(slot) = self.values.get_mut(index) {
            *slot = value;
        }
    }

    /// Position of `v` on the lens scale, clamped to `[0, 1]`. `None` for a non-finite value, and
    /// for a non-positive value on a log scale. Never NaN.
    pub fn color_t(&self, v: f64) -> Option<f64> {
        let (lo, hi) = self.domain;
        let t = match self.scale {
            Scale::Linear if v.is_finite() => (v - lo) / (hi - lo),
            Scale::Log if v.is_finite() && v > 0.0 => {
                (v.log10() - lo.log10()) / (hi.log10() - lo.log10())
            }
            _ => return None,
        };
        if t.is_nan() {
            None
        } else {
            Some(t.clamp(0.0, 1.0))
        }
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

    /// Registers a lens. Refuses a lens whose values do not cover the grid, and an id that is
    /// already registered.
    pub fn add(&mut self, lens: Lens) -> Result<(), MapError> {
        if lens.values.len() != self.cells as usize {
            return Err(MapError::Mismatch("lens"));
        }
        if self.lenses.contains_key(&lens.id) {
            return Err(MapError::Duplicate("lens"));
        }
        self.lenses.insert(lens.id, lens);
        Ok(())
    }

    /// The lens with this id.
    pub fn get(&self, id: LensId) -> Option<&Lens> {
        self.lenses.get(&id)
    }

    /// Every registered lens, in id order.
    pub fn iter(&self) -> impl Iterator<Item = &Lens> {
        self.lenses.values()
    }

    /// The lens with this id, for a delta to write into.
    pub(super) fn get_mut(&mut self, id: LensId) -> Option<&mut Lens> {
        self.lenses.get_mut(&id)
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
