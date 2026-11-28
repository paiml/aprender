//! MAP-8 WebSocket route `/map/feed/:column` (PMAT-7553, epic #7507).
//!
//! A thin axum handler over [`super::session::Session`]. Server feature only; all
//! encoding and resume logic lives in the target-neutral modules.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use tokio::sync::broadcast;

use super::codec::decode_hello;
use super::session::Session;
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
        // broadcast::channel panics on capacity 0.
        let (live, _) = broadcast::channel(live_capacity.max(1));
        Self {
            feed: Arc::new(Mutex::new(ColumnFeed::new(ring_capacity))),
            live,
        }
    }

    fn lock(&self) -> MutexGuard<'_, ColumnFeed> {
        // A panic in another holder leaves the feed consistent (apply is all-or-nothing).
        self.feed.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Add a zeroed column of `cells` cells.
    ///
    /// # Errors
    /// Returns [`FeedError::DuplicateColumn`] if the name is taken.
    pub fn add_column(&self, name: &str, cells: usize) -> Result<(), FeedError> {
        self.lock().add_column(name, cells)
    }

    /// Apply a batch and broadcast it. Returns the column's new seq.
    ///
    /// # Errors
    /// Returns the [`FeedError`] from [`ColumnFeed::apply`]; nothing is broadcast on error.
    pub fn apply(&self, column: &str, changes: &[(u32, f32)]) -> Result<u64, FeedError> {
        let mut feed = self.lock();
        let seq = feed.apply(column, changes)?;
        // Broadcast under the same lock: live receivers see batches in seq order.
        // An Err here only means no connection is listening.
        let _ = self.live.send(LiveBatch {
            column: column.to_owned(),
            batch: Batch {
                seq,
                changes: changes.to_vec(),
            },
        });
        Ok(seq)
    }

    /// Run `f` against the feed under its lock.
    pub fn with_feed<T>(&self, f: impl FnOnce(&ColumnFeed) -> T) -> T {
        f(&self.lock())
    }
}

/// The axum router serving `GET /map/feed/:column` as a WebSocket upgrade.
#[must_use]
pub fn router(hub: FeedHub) -> Router {
    Router::new()
        .route("/map/feed/:column", get(ws_feed))
        .with_state(hub)
}

async fn ws_feed(
    ws: WebSocketUpgrade,
    Path(column): Path<String>,
    State(hub): State<FeedHub>,
) -> Response {
    ws.on_upgrade(move |socket| serve_socket(socket, hub, column))
}

/// Read the client's Hello. `None` if the socket closes or the first binary frame is bad.
async fn read_hello(socket: &mut WebSocket) -> Option<u64> {
    loop {
        match socket.recv().await {
            Some(Ok(Message::Binary(bytes))) => return decode_hello(&bytes).ok(),
            Some(Ok(Message::Close(_)) | Err(_)) | None => return None,
            Some(Ok(_)) => {},
        }
    }
}

async fn serve_socket(mut socket: WebSocket, hub: FeedHub, column: String) {
    // Subscribe before resuming: a batch applied during the resume is then also in the
    // live channel, and the session drops it if the resume already covered it.
    let mut live = hub.subscribe();
    let Some(last_seen) = read_hello(&mut socket).await else {
        return;
    };
    let mut session = Session::new(&column);
    let Ok(frames) = hub.with_feed(|feed| session.hello(feed, last_seen)) else {
        return;
    };
    for frame in frames {
        if socket.send(Message::Binary(frame)).await.is_err() {
            return;
        }
    }
    loop {
        tokio::select! {
            incoming = socket.recv() => match incoming {
                Some(Ok(Message::Close(_)) | Err(_)) | None => return,
                Some(Ok(_)) => {},
            },
            received = live.recv() => match received {
                Ok(live_batch) if live_batch.column == column => {
                    match session.live(&live_batch.batch) {
                        Ok(Some(frame)) => {
                            if socket.send(Message::Binary(frame)).await.is_err() {
                                return;
                            }
                        },
                        Ok(None) => {},
                        // A gap or a lagged channel: close, the client resumes from its seq.
                        Err(_) => return,
                    }
                },
                Ok(_) => {},
                Err(broadcast::error::RecvError::Lagged(_) | broadcast::error::RecvError::Closed) => {
                    return;
                },
            },
        }
    }
}

impl FeedHub {
    fn subscribe(&self) -> broadcast::Receiver<LiveBatch> {
        self.live.subscribe()
    }
}
