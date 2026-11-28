//! Session tests: the Hello answer on the replay and snapshot paths, live forwarding, gaps.

use crate::map_feed::codec::{decode_server_frame, ServerFrame};
use crate::map_feed::session::{Session, SessionGap};
use crate::map_feed::{column_hash, Batch, ColumnFeed, FeedError};

fn feed_with(cap: usize) -> ColumnFeed {
    let mut feed = ColumnFeed::new(cap);
    feed.add_column("col", 4).expect("add col");
    feed
}

fn decode_all(frames: &[Vec<u8>]) -> Vec<ServerFrame> {
    frames
        .iter()
        .map(|f| decode_server_frame(f).expect("session frame decodes"))
        .collect()
}

fn batch_seq(frame: &ServerFrame) -> u64 {
    match frame {
        ServerFrame::Batch(b) => b.seq,
        other => panic!("expected Batch, got {other:?}"),
    }
}

#[test]
fn map_feed_session_hello_replay_sends_missing_batches_in_order() {
    let mut feed = feed_with(8);
    for v in 0..3 {
        feed.apply("col", &[(v, v as f32)]).expect("apply");
    }
    let mut session = Session::new("col");
    let frames = session.hello(&feed, 1).expect("hello");
    let decoded = decode_all(&frames);
    assert_eq!(decoded.len(), 2, "client at seq 1 misses batches 2 and 3");
    assert_eq!(batch_seq(&decoded[0]), 2);
    assert_eq!(batch_seq(&decoded[1]), 3);
    assert_eq!(session.sent_seq(), 3);
}

#[test]
fn map_feed_session_hello_current_seq_sends_nothing() {
    let mut feed = feed_with(8);
    feed.apply("col", &[(0, 1.0)]).expect("apply");
    let mut session = Session::new("col");
    let frames = session.hello(&feed, 1).expect("hello");
    assert!(frames.is_empty());
    assert_eq!(session.sent_seq(), 1);
}

#[test]
fn map_feed_session_hello_snapshot_when_ring_overflowed() {
    let mut feed = feed_with(2);
    for v in 0..5 {
        feed.apply("col", &[(v % 4, (v + 10) as f32)])
            .expect("apply");
    }
    let mut session = Session::new("col");
    let frames = session.hello(&feed, 0).expect("hello");
    let decoded = decode_all(&frames);
    assert_eq!(
        decoded.len(),
        1,
        "overflowed ring answers with one Snapshot"
    );
    match &decoded[0] {
        ServerFrame::Snapshot { seq, hash, values } => {
            assert_eq!(*seq, 5);
            assert_eq!(values.as_slice(), feed.values("col").expect("values"));
            assert_eq!(*hash, column_hash(values));
        },
        other => panic!("expected Snapshot, got {other:?}"),
    }
    assert_eq!(session.sent_seq(), 5);
}

#[test]
fn map_feed_session_hello_future_seq_is_feed_error() {
    let feed = feed_with(8);
    let mut session = Session::new("col");
    assert!(matches!(
        session.hello(&feed, 3),
        Err(FeedError::FutureSeq { .. })
    ));
}

#[test]
fn map_feed_session_hello_unknown_column_is_feed_error() {
    let feed = feed_with(8);
    let mut session = Session::new("nope");
    assert!(matches!(
        session.hello(&feed, 0),
        Err(FeedError::UnknownColumn(_))
    ));
}

#[test]
fn map_feed_session_live_drops_batches_the_resume_already_sent() {
    let mut feed = feed_with(8);
    feed.apply("col", &[(0, 1.0)]).expect("apply");
    feed.apply("col", &[(1, 2.0)]).expect("apply");
    let mut session = Session::new("col");
    session.hello(&feed, 0).expect("hello");
    let covered = Batch {
        seq: 2,
        changes: vec![(1, 2.0)],
    };
    assert_eq!(session.live(&covered), Ok(None));
    assert_eq!(session.sent_seq(), 2);
}

#[test]
fn map_feed_session_live_forwards_the_next_batch_as_a_frame() {
    let mut feed = feed_with(8);
    feed.apply("col", &[(0, 1.0)]).expect("apply");
    let mut session = Session::new("col");
    session.hello(&feed, 1).expect("hello");
    let next = Batch {
        seq: 2,
        changes: vec![(3, -4.5)],
    };
    let frame = session
        .live(&next)
        .expect("no gap")
        .expect("next batch is forwarded");
    assert_eq!(
        decode_server_frame(&frame).expect("decodes"),
        ServerFrame::Batch(next)
    );
    assert_eq!(session.sent_seq(), 2);
}

#[test]
fn map_feed_session_live_gap_is_an_error_naming_both_seqs() {
    let mut feed = feed_with(8);
    feed.apply("col", &[(0, 1.0)]).expect("apply");
    let mut session = Session::new("col");
    session.hello(&feed, 1).expect("hello");
    let skipped = Batch {
        seq: 3,
        changes: vec![(0, 9.0)],
    };
    assert_eq!(
        session.live(&skipped),
        Err(SessionGap {
            expected: 2,
            got: 3
        })
    );
    assert_eq!(session.sent_seq(), 1, "a gap must not advance the session");
}
