//! MAP-8 per-connection session (PMAT-7553, epic #7507). Target-neutral: it turns feed
//! state into encoded frames and never touches a socket.
//!
//! Protocol: the client sends Hello with its `last_seen_seq`. [`Session::hello`] answers
//! with the resume frames (Batch frames, or one Snapshot). After that the route forwards
//! each live batch through [`Session::live`], which drops anything the resume already sent.

use super::codec::{encode_batch, encode_snapshot};
use super::{column_hash, Batch, ColumnFeed, FeedError, Resume};

/// One connection's view of one column.
#[derive(Debug)]
pub struct Session {
    column: String,
    sent_seq: u64,
}

/// A live batch arrived out of order: a batch was missed on this connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionGap {
    /// The seq the session needed next.
    pub expected: u64,
    /// The seq that arrived.
    pub got: u64,
}

impl std::fmt::Display for SessionGap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "live batch {} arrived, expected {}",
            self.got, self.expected
        )
    }
}

impl std::error::Error for SessionGap {}

impl Session {
    /// A session for `column` that has sent nothing yet.
    #[must_use]
    pub fn new(column: &str) -> Self {
        Self {
            column: column.to_owned(),
            sent_seq: 0,
        }
    }

    /// Seq of the last frame this session has sent (or resumed to).
    #[must_use]
    pub fn sent_seq(&self) -> u64 {
        self.sent_seq
    }

    /// Answer a Hello: the encoded resume frames, in send order. Advances `sent_seq`
    /// to the column's seq.
    ///
    /// # Errors
    /// Returns the [`FeedError`] from [`ColumnFeed::resume`] (unknown column, or a client
    /// seq ahead of the server).
    pub fn hello(
        &mut self,
        feed: &ColumnFeed,
        last_seen_seq: u64,
    ) -> Result<Vec<Vec<u8>>, FeedError> {
        let frames = match feed.resume(&self.column, last_seen_seq)? {
            Resume::Replay(batches) => batches.iter().map(encode_batch).collect(),
            Resume::Snapshot { seq, values } => {
                vec![encode_snapshot(seq, column_hash(&values), &values)]
            },
        };
        self.sent_seq = feed.seq(&self.column)?;
        Ok(frames)
    }

    /// Forward one live batch. `Ok(None)` if the resume already covered it, `Ok(Some(frame))`
    /// if it is the next batch, [`SessionGap`] if a batch was skipped.
    ///
    /// # Errors
    /// Returns [`SessionGap`] when `batch.seq` is beyond `sent_seq + 1`.
    pub fn live(&mut self, batch: &Batch) -> Result<Option<Vec<u8>>, SessionGap> {
        if batch.seq <= self.sent_seq {
            return Ok(None);
        }
        let expected = self.sent_seq.saturating_add(1);
        if batch.seq != expected {
            return Err(SessionGap {
                expected,
                got: batch.seq,
            });
        }
        self.sent_seq = batch.seq;
        Ok(Some(encode_batch(batch)))
    }
}
