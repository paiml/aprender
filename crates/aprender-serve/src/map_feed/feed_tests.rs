//! Unit tests for `ColumnFeed` semantics, error paths and `column_hash`.

use super::*;

fn feed_with(cap: usize) -> ColumnFeed {
    let mut f = ColumnFeed::new(cap);
    f.add_column("lat", 4).expect("add lat");
    f.add_column("lon", 4).expect("add lon");
    f
}

#[test]
fn map_feed_seq_is_monotonic_per_column() {
    let mut f = feed_with(8);
    assert_eq!(f.apply("lat", &[(0, 1.0)]).expect("a1"), 1);
    assert_eq!(f.apply("lat", &[(1, 2.0)]).expect("a2"), 2);
    assert_eq!(f.apply("lon", &[(0, 3.0)]).expect("b1"), 1);
    assert_eq!(f.seq("lat").expect("seq lat"), 2);
    assert_eq!(f.seq("lon").expect("seq lon"), 1);
}

#[test]
fn map_feed_seq_does_not_advance_on_error() {
    let mut f = feed_with(8);
    f.apply("lat", &[(9, 1.0)]).expect_err("oob");
    f.apply("nope", &[(0, 1.0)]).expect_err("unknown");
    assert_eq!(f.seq("lat").expect("seq"), 0);
    assert_eq!(f.apply("lat", &[(0, 1.0)]).expect("first good"), 1);
}

#[test]
fn map_feed_unknown_column_errors_and_leaves_state_unchanged() {
    let mut f = feed_with(8);
    f.apply("lat", &[(0, 5.0)]).expect("seed");
    let before = f.values("lat").expect("values").to_vec();
    let err = f.apply("nope", &[(0, 9.0)]).expect_err("unknown column");
    assert_eq!(err, FeedError::UnknownColumn("nope".to_owned()));
    assert_eq!(f.values("lat").expect("values"), before.as_slice());
    assert_eq!(f.seq("lat").expect("seq"), 1);
}

#[test]
fn map_feed_out_of_range_cell_errors_and_leaves_state_unchanged() {
    let mut f = feed_with(8);
    f.apply("lat", &[(0, 5.0)]).expect("seed");
    let before = f.values("lat").expect("values").to_vec();
    // First change is valid, second is at len (4): the whole batch is rejected.
    let err = f.apply("lat", &[(1, 7.0), (4, 8.0)]).expect_err("oob");
    assert_eq!(
        err,
        FeedError::CellOutOfRange {
            column: "lat".to_owned(),
            cell: 4,
            len: 4
        }
    );
    assert_eq!(f.values("lat").expect("values"), before.as_slice());
    assert_eq!(f.seq("lat").expect("seq"), 1);
}

#[test]
fn map_feed_last_write_in_a_batch_wins() {
    let mut f = feed_with(8);
    f.apply("lat", &[(2, 1.0), (2, 3.0)]).expect("apply");
    assert_eq!(f.values("lat").expect("values")[2], 3.0);
}

#[test]
fn map_feed_add_column_rejects_duplicate_name() {
    let mut f = feed_with(8);
    let err = f.add_column("lat", 9).expect_err("duplicate");
    assert_eq!(err, FeedError::DuplicateColumn("lat".to_owned()));
}

#[test]
fn map_feed_resume_same_seq_is_empty_replay() {
    let mut f = feed_with(8);
    f.apply("lat", &[(0, 1.0)]).expect("a1");
    f.apply("lat", &[(1, 2.0)]).expect("a2");
    assert_eq!(
        f.resume("lat", 2).expect("resume"),
        Resume::Replay(Vec::new())
    );
}

#[test]
fn map_feed_resume_future_seq_is_error() {
    let mut f = feed_with(8);
    f.apply("lat", &[(0, 1.0)]).expect("a1");
    let err = f.resume("lat", 2).expect_err("future");
    assert_eq!(
        err,
        FeedError::FutureSeq {
            column: "lat".to_owned(),
            last_seen: 2,
            current: 1
        }
    );
}

#[test]
fn map_feed_resume_unknown_column_is_error() {
    let f = feed_with(8);
    let err = f.resume("nope", 0).expect_err("unknown");
    assert_eq!(err, FeedError::UnknownColumn("nope".to_owned()));
}

#[test]
fn map_feed_resume_replays_exactly_the_missing_batches_in_order() {
    let mut f = feed_with(8);
    for i in 0..6_u32 {
        f.apply("lat", &[(i % 4, i as f32)]).expect("apply");
    }
    match f.resume("lat", 2).expect("resume") {
        Resume::Replay(batches) => {
            let seqs: Vec<u64> = batches.iter().map(|b| b.seq).collect();
            assert_eq!(seqs, vec![3, 4, 5, 6]);
            assert_eq!(batches[0].changes, vec![(2, 2.0)]);
            assert_eq!(batches[3].changes, vec![(1, 5.0)]);
        },
        other => panic!("expected replay, got {other:?}"),
    }
}

#[test]
fn map_feed_resume_at_exact_ring_boundary_still_replays() {
    // cap 4, 6 applied: missing from seq 2 is 4 batches == cap, so replay.
    let mut f = feed_with(4);
    for i in 0..6_u32 {
        f.apply("lat", &[(0, i as f32)]).expect("apply");
    }
    match f.resume("lat", 2).expect("resume") {
        Resume::Replay(batches) => {
            let seqs: Vec<u64> = batches.iter().map(|b| b.seq).collect();
            assert_eq!(seqs, vec![3, 4, 5, 6]);
        },
        other => panic!("expected replay at boundary, got {other:?}"),
    }
}

#[test]
fn map_feed_resume_one_past_ring_boundary_is_snapshot() {
    // cap 4, 6 applied: from seq 1 needs 5 batches > cap, so snapshot.
    let mut f = feed_with(4);
    for i in 0..6_u32 {
        f.apply("lat", &[(0, i as f32)]).expect("apply");
    }
    match f.resume("lat", 1).expect("resume") {
        Resume::Snapshot { seq, values } => {
            assert_eq!(seq, 6);
            assert_eq!(values, f.values("lat").expect("values").to_vec());
        },
        other => panic!("expected snapshot past boundary, got {other:?}"),
    }
}

#[test]
fn map_feed_ring_evicts_oldest_batches() {
    let mut f = feed_with(3);
    for i in 0..10_u32 {
        f.apply("lat", &[(0, i as f32)]).expect("apply");
    }
    match f.resume("lat", 7).expect("resume") {
        Resume::Replay(batches) => {
            let seqs: Vec<u64> = batches.iter().map(|b| b.seq).collect();
            assert_eq!(seqs, vec![8, 9, 10]);
        },
        other => panic!("expected replay of the 3 newest, got {other:?}"),
    }
    assert!(matches!(
        f.resume("lat", 6).expect("resume"),
        Resume::Snapshot { .. }
    ));
}

#[test]
fn map_feed_snapshot_from_zero_carries_current_values_and_seq() {
    let mut f = feed_with(1);
    f.apply("lat", &[(0, 1.5)]).expect("a1");
    f.apply("lat", &[(3, -2.0)]).expect("a2");
    match f.resume("lat", 0).expect("resume") {
        Resume::Snapshot { seq, values } => {
            assert_eq!(seq, 2);
            assert_eq!(values, vec![1.5, 0.0, 0.0, -2.0]);
        },
        other => panic!("expected snapshot, got {other:?}"),
    }
}

#[test]
fn map_feed_fnv1a_matches_published_vector() {
    // FNV-1a 64 of the single byte "a".
    assert_eq!(fnv1a_extend(FNV_OFFSET, b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a_extend(FNV_OFFSET, b""), FNV_OFFSET);
}

#[test]
fn map_feed_column_hash_is_deterministic_and_order_sensitive() {
    let a = [1.5_f32, -2.0];
    assert_eq!(column_hash(&a), column_hash(&a));
    assert_ne!(column_hash(&[1.0, 0.0]), column_hash(&[0.0, 1.0]));
}

#[test]
fn map_feed_hash_sees_sign_bit_in_last_byte() {
    // The sign bit lives in the 4th (last) byte of each f32.
    assert_ne!(column_hash(&[1.0]), column_hash(&[-1.0]));
    assert_ne!(column_hash(&[0.0]), column_hash(&[-0.0]));
}
