//! Lens delta: the wire format for one batch of cell updates (APR-MAP-001 MAP-8 slice 1).
//!
//! RED stub: the signatures are final, the bodies are not.

use super::{LensId, MapError};

/// Header bytes: lens id (`u32`), sequence number (`u64`), entry count (`u32`).
pub const HEADER_BYTES: usize = 16;
/// Bytes per entry across the three columns: cell id (`u32`), tag (`u8`), value (`f64`).
pub const ENTRY_BYTES: usize = 13;
/// Tag of a cell with no data.
pub const TAG_NONE: u8 = 0x00;
/// Tag of a cell whose value follows.
pub const TAG_SOME: u8 = 0x01;

/// One batch of cell updates for one lens. `None` clears a cell.
#[derive(Debug, Clone, PartialEq)]
pub struct Delta {
    lens: LensId,
    seq: u64,
    cells: Vec<(u32, Option<f64>)>,
}

impl Delta {
    /// RED stub: builds without validating.
    pub fn new(lens: LensId, seq: u64, cells: Vec<(u32, Option<f64>)>) -> Result<Self, MapError> {
        Ok(Self { lens, seq, cells })
    }

    /// Lens id.
    pub fn lens(&self) -> LensId {
        self.lens
    }

    /// Sequence number.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// `(cell id, new value)` pairs in wire order.
    pub fn cells(&self) -> &[(u32, Option<f64>)] {
        &self.cells
    }

    /// RED stub: encodes nothing yet.
    pub fn encode(&self) -> Result<Vec<u8>, MapError> {
        Ok(Vec::new())
    }

    /// RED stub: refuses everything.
    pub fn decode(bytes: &[u8], cells: u32) -> Result<Self, MapError> {
        let _ = (bytes, cells);
        Err(MapError::Malformed("delta"))
    }
}

#[cfg(test)]
#[path = "delta_tests.rs"]
mod tests;
