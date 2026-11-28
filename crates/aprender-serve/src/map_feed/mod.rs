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
        let _ = (column, changes);
        Ok(0)
    }

    /// Resume a client that last saw `last_seen_seq` on `column`.
    ///
    /// # Errors
    /// Returns [`FeedError::UnknownColumn`] or [`FeedError::FutureSeq`].
    pub fn resume(&self, column: &str, last_seen_seq: u64) -> Result<Resume, FeedError> {
        let _ = (column, last_seen_seq);
        Ok(Resume::Replay(Vec::new()))
    }

    /// Current seq of `column`.
    ///
    /// # Errors
    /// Returns [`FeedError::UnknownColumn`] if the column does not exist.
    pub fn seq(&self, column: &str) -> Result<u64, FeedError> {
        let _ = column;
        Ok(0)
    }

    /// Current values of `column`.
    ///
    /// # Errors
    /// Returns [`FeedError::UnknownColumn`] if the column does not exist.
    pub fn values(&self, column: &str) -> Result<&[f32], FeedError> {
        let _ = column;
        Ok(&[])
    }
}

/// Fold `bytes` into an FNV-1a 64-bit running hash.
#[must_use]
pub fn fnv1a_extend(hash: u64, bytes: &[u8]) -> u64 {
    let _ = bytes;
    hash
}

/// Deterministic hash of a column: FNV-1a over the little-endian bytes of each value.
#[must_use]
pub fn column_hash(values: &[f32]) -> u64 {
    let _ = values;
    0
}

#[cfg(test)]
mod feed_tests;

#[cfg(test)]
mod reconnect_tests;
