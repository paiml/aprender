//! MAP-8 column feed hub (PMAT-7552, epic #7507).
//!
//! Named `f32` columns, each with a monotonic sequence number and a bounded
//! ring of recent change batches. A client that last saw `seq = n` asks
//! [`ColumnFeed::resume`]: it gets the missing batches when the ring still
//! holds them, otherwise a full snapshot. [`column_hash`] lets the client
//! check its mirror against the server after a resume.
//!
//! Headless: no network code lives here.

use std::collections::BTreeMap;

/// FNV-1a 64-bit offset basis.
pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// One applied change batch: its sequence number and the `(cell, value)` writes.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    /// Sequence number assigned by [`ColumnFeed::apply`].
    pub seq: u64,
    /// `(cell index, new value)` pairs, in apply order.
    pub changes: Vec<(u32, f32)>,
}

/// Answer to a resume request.
#[derive(Debug, Clone, PartialEq)]
pub enum Resume {
    /// Every batch after the client's seq is still in the ring, in seq order.
    Replay(Vec<Batch>),
    /// The ring no longer covers the gap: the full column at `seq`.
    Snapshot {
        /// Column sequence number the values correspond to.
        seq: u64,
        /// Full column values at `seq`.
        values: Vec<f32>,
    },
}

/// Errors from [`ColumnFeed`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeedError {
    /// No column with this name exists.
    UnknownColumn(String),
    /// A column with this name already exists.
    DuplicateColumn(String),
    /// A change addressed a cell at or beyond the column length.
    CellOutOfRange {
        /// Column the change addressed.
        column: String,
        /// Offending cell index.
        cell: u32,
        /// Column length in cells.
        len: usize,
    },
    /// The client claims a seq newer than the server has.
    FutureSeq {
        /// Column the resume addressed.
        column: String,
        /// Seq the client reported.
        last_seen: u64,
        /// Current server seq.
        current: u64,
    },
}

impl std::fmt::Display for FeedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownColumn(c) => write!(f, "unknown column `{c}`"),
            Self::DuplicateColumn(c) => write!(f, "column `{c}` already exists"),
            Self::CellOutOfRange { column, cell, len } => {
                write!(f, "cell {cell} out of range for `{column}` (len {len})")
            },
            Self::FutureSeq {
                column,
                last_seen,
                current,
            } => {
                write!(
                    f,
                    "`{column}` resume seq {last_seen} is ahead of server seq {current}"
                )
            },
        }
    }
}

impl std::error::Error for FeedError {}

/// Named `f32` columns with per-column sequence numbers and bounded batch rings.
#[derive(Debug)]
pub struct ColumnFeed {
    ring_capacity: usize,
    columns: BTreeMap<String, Column>,
}

#[derive(Debug)]
struct Column {
    values: Vec<f32>,
    seq: u64,
    ring: std::collections::VecDeque<Batch>,
}

impl ColumnFeed {
    /// Create an empty feed that keeps the last `ring_capacity` batches per column.
    #[must_use]
    pub fn new(ring_capacity: usize) -> Self {
        Self {
            ring_capacity,
            columns: BTreeMap::new(),
        }
    }

    /// Add a column of `cells` zeroed cells at seq 0.
    ///
    /// # Errors
    /// Returns [`FeedError::DuplicateColumn`] if the name is already taken.
    pub fn add_column(&mut self, name: &str, cells: usize) -> Result<(), FeedError> {
        if self.columns.contains_key(name) {
            return Err(FeedError::DuplicateColumn(name.to_owned()));
        }
        let col = Column {
            values: vec![0.0; cells],
            seq: 0,
            ring: Default::default(),
        };
        self.columns.insert(name.to_owned(), col);
        Ok(())
    }

    /// Apply a change batch atomically and return the column's new seq.
    ///
    /// # Errors
    /// Returns [`FeedError::UnknownColumn`] or [`FeedError::CellOutOfRange`]
    /// without changing any state.
    pub fn apply(&mut self, column: &str, changes: &[(u32, f32)]) -> Result<u64, FeedError> {
        let ring_capacity = self.ring_capacity;
        let col = self
            .columns
            .get_mut(column)
            .ok_or_else(|| FeedError::UnknownColumn(column.to_owned()))?;
        let resolved = resolve_cells(column, col.values.len(), changes)?;
        for (index, value) in resolved {
            col.values[index] = value;
        }
        col.seq += 1;
        col.ring.push_back(Batch {
            seq: col.seq,
            changes: changes.to_vec(),
        });
        while col.ring.len() > ring_capacity {
            col.ring.pop_front();
        }
        Ok(col.seq)
    }

    /// Resume a client that last saw `last_seen_seq` on `column`.
    ///
    /// # Errors
    /// Returns [`FeedError::UnknownColumn`] or [`FeedError::FutureSeq`].
    pub fn resume(&self, column: &str, last_seen_seq: u64) -> Result<Resume, FeedError> {
        let col = self.column(column)?;
        if last_seen_seq > col.seq {
            return Err(FeedError::FutureSeq {
                column: column.to_owned(),
                last_seen: last_seen_seq,
                current: col.seq,
            });
        }
        let missing = col.seq - last_seen_seq;
        let ring_len = col.ring.len();
        match usize::try_from(missing) {
            // The last `missing` ring entries are exactly seqs last_seen+1..=seq.
            Ok(m) if m <= ring_len => Ok(Resume::Replay(
                col.ring.iter().skip(ring_len - m).cloned().collect(),
            )),
            _ => Ok(Resume::Snapshot {
                seq: col.seq,
                values: col.values.clone(),
            }),
        }
    }

    /// Current seq of `column`.
    ///
    /// # Errors
    /// Returns [`FeedError::UnknownColumn`] if the column does not exist.
    pub fn seq(&self, column: &str) -> Result<u64, FeedError> {
        self.column(column).map(|c| c.seq)
    }

    /// Current values of `column`.
    ///
    /// # Errors
    /// Returns [`FeedError::UnknownColumn`] if the column does not exist.
    pub fn values(&self, column: &str) -> Result<&[f32], FeedError> {
        self.column(column).map(|c| c.values.as_slice())
    }

    fn column(&self, column: &str) -> Result<&Column, FeedError> {
        self.columns
            .get(column)
            .ok_or_else(|| FeedError::UnknownColumn(column.to_owned()))
    }
}

/// Validate every cell of a batch before any write, so a rejected batch changes nothing.
fn resolve_cells(
    column: &str,
    len: usize,
    changes: &[(u32, f32)],
) -> Result<Vec<(usize, f32)>, FeedError> {
    changes
        .iter()
        .map(|&(cell, value)| {
            usize::try_from(cell)
                .ok()
                .filter(|&index| index < len)
                .map(|index| (index, value))
                .ok_or_else(|| FeedError::CellOutOfRange {
                    column: column.to_owned(),
                    cell,
                    len,
                })
        })
        .collect()
}

/// Fold `bytes` into an FNV-1a 64-bit running hash.
#[must_use]
pub fn fnv1a_extend(hash: u64, bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(hash, |h, &b| (h ^ u64::from(b)).wrapping_mul(FNV_PRIME))
}

/// Deterministic hash of a column: FNV-1a over the little-endian bytes of each value.
#[must_use]
pub fn column_hash(values: &[f32]) -> u64 {
    values
        .iter()
        .fold(FNV_OFFSET, |h, v| fnv1a_extend(h, &v.to_le_bytes()))
}

#[cfg(test)]
mod feed_tests;

#[cfg(test)]
mod reconnect_tests;
