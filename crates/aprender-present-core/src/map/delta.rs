//! Lens delta: the wire format for one batch of cell updates (APR-MAP-001 MAP-8 slice 1).
//!
//! Little-endian and columnar. A 16-byte header holds the lens id (`u32`), the sequence number
//! (`u64`) and the entry count (`u32`). Three columns of `count` entries follow: the cell ids
//! (`u32`), the tags (`u8`, [`TAG_NONE`] or [`TAG_SOME`]) and the value bits (`f64`). A no-data
//! entry holds zero bits in its value slot and decode ignores them. The total length is exactly
//! `16 + 13 * count` bytes.

use super::{check_finite, LensId, MapError};

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
    /// Builds a delta. Refuses a value that is NaN or infinite, since the wire format carries only
    /// finite values.
    pub fn new(lens: LensId, seq: u64, cells: Vec<(u32, Option<f64>)>) -> Result<Self, MapError> {
        for &(_, value) in &cells {
            if let Some(v) = value {
                check_finite(v, "delta value")?;
            }
        }
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

    /// Encodes the delta in the layout of the module header. Refuses more entries than a `u32`
    /// count can name.
    pub fn encode(&self) -> Result<Vec<u8>, MapError> {
        let count = u32::try_from(self.cells.len()).map_err(|_| MapError::OutOfRange("delta"))?;
        let mut out = Vec::with_capacity(HEADER_BYTES + ENTRY_BYTES * self.cells.len());
        out.extend_from_slice(&self.lens.to_le_bytes());
        out.extend_from_slice(&self.seq.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend(self.cells.iter().flat_map(|&(cell, _)| cell.to_le_bytes()));
        out.extend(
            self.cells
                .iter()
                .map(|&(_, value)| if value.is_some() { TAG_SOME } else { TAG_NONE }),
        );
        out.extend(
            self.cells
                .iter()
                .flat_map(|&(_, value)| value.unwrap_or(0.0).to_bits().to_le_bytes()),
        );
        Ok(out)
    }

    /// Decodes a delta for a grid of `cells` cells. Refuses a buffer that is shorter or longer
    /// than its header declares, a cell id outside the grid, an unknown tag and a non-finite
    /// value. The declared length is checked before any allocation, so no input can make decode
    /// reserve more memory than the buffer it was given.
    pub fn decode(bytes: &[u8], cells: u32) -> Result<Self, MapError> {
        let mut at = 0;
        let lens = u32::from_le_bytes(take(bytes, &mut at)?);
        let seq = u64::from_le_bytes(take(bytes, &mut at)?);
        let count = u32::from_le_bytes(take(bytes, &mut at)?);
        check_length(bytes.len(), count)?;
        let n = usize::try_from(count).map_err(|_| MapError::Truncated("delta"))?;

        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            ids.push(u32::from_le_bytes(take(bytes, &mut at)?));
        }
        let mut tags = Vec::with_capacity(n);
        for _ in 0..n {
            let [tag] = take(bytes, &mut at)?;
            tags.push(tag);
        }
        let mut values = Vec::with_capacity(n);
        for _ in 0..n {
            values.push(f64::from_bits(u64::from_le_bytes(take(bytes, &mut at)?)));
        }

        let mut entries = Vec::with_capacity(n);
        for ((cell, tag), value) in ids.into_iter().zip(tags).zip(values) {
            if cell >= cells {
                return Err(MapError::CellOutOfRange(cell));
            }
            let entry = match tag {
                TAG_NONE => None,
                TAG_SOME => {
                    check_finite(value, "delta value")?;
                    Some(value)
                }
                other => return Err(MapError::UnknownTag(other)),
            };
            entries.push((cell, entry));
        }
        Ok(Self {
            lens,
            seq,
            cells: entries,
        })
    }
}

/// Reads the next `N` bytes at `at` and advances it. A buffer that ends first is `Truncated`.
fn take<const N: usize>(bytes: &[u8], at: &mut usize) -> Result<[u8; N], MapError> {
    let truncated = MapError::Truncated("delta");
    let end = at.checked_add(N).ok_or(truncated)?;
    let chunk = bytes.get(*at..end).ok_or(truncated)?;
    *at = end;
    chunk.try_into().map_err(|_| truncated)
}

/// Checks a buffer of `len` bytes against the length its header declares for `count` entries.
fn check_length(len: usize, count: u32) -> Result<(), MapError> {
    let want = HEADER_BYTES as u64 + u64::from(count) * ENTRY_BYTES as u64;
    let have = u64::try_from(len).unwrap_or(u64::MAX);
    match have.cmp(&want) {
        std::cmp::Ordering::Less => Err(MapError::Truncated("delta")),
        std::cmp::Ordering::Equal => Ok(()),
        std::cmp::Ordering::Greater => Err(MapError::TrailingBytes("delta")),
    }
}

#[cfg(test)]
#[path = "delta_tests.rs"]
mod tests;
