//! MAP-8 WebSocket route `/map/feed/:column` (PMAT-7553, epic #7507).
//!
//! A thin axum handler over [`super::session::Session`]. Server feature only; all
//! encoding and resume logic lives in the target-neutral modules.

use std::sync::{Arc, Mutex};

use tokio::sync::broadcast;

use super::{Batch, ColumnFeed, FeedError};

/// A batch applied to a named column, as broadcast to live connections.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveBatch {
    /// Column the batch was applied to.
    pub column: String,
    /// The applied batch, with its seq.
    pub batch: Batch,
}

/// Shared server state: the column feed plus the live broadcast channel.
///
/// [`FeedHub::apply`] mutates the feed and broadcasts the batch while holding the same lock,
/// so a connection that resumed under that lock never misses or reorders a live batch.
#[derive(Clone)]
pub struct FeedHub {
    feed: Arc<Mutex<ColumnFeed>>,
    live: broadcast::Sender<LiveBatch>,
}

impl FeedHub {
    /// A hub whose columns keep `ring_capacity` batches, with a live channel of `live_capacity`.
    #[must_use]
    pub fn new(ring_capacity: usize, live_capacity: usize) -> Self {
        todo!("RED: FeedHub::new({ring_capacity}, {live_capacity})")
    }

    /// Add a zeroed column of `cells` cells.
    ///
    /// # Errors
    /// Returns [`FeedError::DuplicateColumn`] if the name is taken.
    pub fn add_column(&self, name: &str, cells: usize) -> Result<(), FeedError> {
        todo!("RED: add_column({name}, {cells})")
    }

    /// Apply a batch and broadcast it. Returns the column's new seq.
    ///
    /// # Errors
    /// Returns the [`FeedError`] from [`ColumnFeed::apply`]; nothing is broadcast on error.
    pub fn apply(&self, column: &str, changes: &[(u32, f32)]) -> Result<u64, FeedError> {
        todo!("RED: apply({column}, {} changes)", changes.len())
    }

    /// Run `f` against the feed under its lock.
    pub fn with_feed<T>(&self, f: impl FnOnce(&ColumnFeed) -> T) -> T {
        todo!("RED: with_feed")
    }
}

/// The axum router serving `GET /map/feed/:column` as a WebSocket upgrade.
#[must_use]
pub fn router(hub: FeedHub) -> axum::Router {
    todo!("RED: router")
}
