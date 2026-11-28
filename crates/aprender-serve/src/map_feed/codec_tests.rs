//! Codec tests: round-trips, exact byte layout, and error (never panic) cases.

use crate::map_feed::codec::*;
use crate::map_feed::Batch;

fn batch(seq: u64, changes: &[(u32, f32)]) -> Batch {
    Batch {
        seq,
        changes: changes.to_vec(),
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn map_feed_codec_hello_round_trips() {
    for seq in [0_u64, 1, 42, u64::MAX] {
        assert_eq!(decode_hello(&encode_hello(seq)), Ok(seq));
    }
}

#[test]
fn map_feed_codec_batch_round_trips_every_field() {
    let original = batch(9, &[(0, 1.5), (7, -2.25), (3, 1.0e-30)]);
    let frame = decode_server_frame(&encode_batch(&original)).expect("decode batch");
    assert_eq!(frame, ServerFrame::Batch(original));
}

#[test]
fn map_feed_codec_empty_batch_round_trips() {
    let original = batch(5, &[]);
    let frame = decode_server_frame(&encode_batch(&original)).expect("decode empty batch");
    assert_eq!(frame, ServerFrame::Batch(original));
}

#[test]
fn map_feed_codec_snapshot_round_trips_bit_exactly() {
    // -0.0 and a NaN payload only survive a bit-level encoding.
    let values = [0.5_f32, -0.0, 1.0e-30, f32::from_bits(0x7fc0_1234)];
    let frame = decode_server_frame(&encode_snapshot(12, 0xDEAD_BEEF_0102_0304, &values))
        .expect("decode snapshot");
    match frame {
        ServerFrame::Snapshot {
            seq,
            hash,
            values: got,
        } => {
            assert_eq!(seq, 12);
            assert_eq!(hash, 0xDEAD_BEEF_0102_0304);
            assert_eq!(bits(&got), bits(&values));
        },
        other => panic!("expected Snapshot, got {other:?}"),
    }
}

#[test]
fn map_feed_codec_batch_layout_is_little_endian() {
    // tag 0 | seq 1 (u64) | n 1 (u32) | cell 2 (u32) | 1.0f32 = 0x3F80_0000
    let expected: Vec<u8> = vec![
        0, 1, 0, 0, 0, 0, 0, 0, 0, // tag, seq
        1, 0, 0, 0, // n
        2, 0, 0, 0, // cell
        0, 0, 0x80, 0x3f, // value
    ];
    assert_eq!(encode_batch(&batch(1, &[(2, 1.0)])), expected);
}

#[test]
fn map_feed_codec_snapshot_layout_is_little_endian() {
    // tag 1 | seq 2 | hash 0x0102030405060708 | n 1 | 1.0f32
    let expected: Vec<u8> = vec![
        1, // tag
        2, 0, 0, 0, 0, 0, 0, 0, // seq
        8, 7, 6, 5, 4, 3, 2, 1, // hash
        1, 0, 0, 0, // n
        0, 0, 0x80, 0x3f, // value
    ];
    assert_eq!(encode_snapshot(2, 0x0102_0304_0506_0708, &[1.0]), expected);
}

#[test]
fn map_feed_codec_truncated_batch_is_err_at_every_prefix() {
    let full = encode_batch(&batch(3, &[(0, 1.0), (1, 2.0)]));
    for len in 0..full.len() {
        let prefix = full.get(..len).expect("prefix in range");
        assert!(
            decode_server_frame(prefix).is_err(),
            "prefix of {len} bytes decoded"
        );
    }
}

#[test]
fn map_feed_codec_truncated_snapshot_is_err_at_every_prefix() {
    let full = encode_snapshot(4, 77, &[1.0, 2.0, 3.0]);
    for len in 0..full.len() {
        let prefix = full.get(..len).expect("prefix in range");
        assert!(
            decode_server_frame(prefix).is_err(),
            "prefix of {len} bytes decoded"
        );
    }
}

#[test]
fn map_feed_codec_truncated_hello_is_err() {
    assert!(decode_hello(&[]).is_err());
    assert!(decode_hello(&[1, 2, 3]).is_err());
    assert!(decode_hello(&encode_hello(7).get(..7).expect("7 bytes")).is_err());
}

#[test]
fn map_feed_codec_unknown_tag_is_err() {
    let mut frame = encode_batch(&batch(1, &[(0, 1.0)]));
    if let Some(tag) = frame.first_mut() {
        *tag = 7;
    }
    assert_eq!(decode_server_frame(&frame), Err(CodecError::UnknownTag(7)));
    assert_eq!(decode_server_frame(&[9]), Err(CodecError::UnknownTag(9)));
}

#[test]
fn map_feed_codec_trailing_bytes_are_err() {
    let mut frame = encode_batch(&batch(1, &[(0, 1.0)]));
    frame.push(0);
    assert_eq!(
        decode_server_frame(&frame),
        Err(CodecError::TrailingBytes(1))
    );
    let mut hello = encode_hello(3);
    hello.push(0);
    assert_eq!(decode_hello(&hello), Err(CodecError::TrailingBytes(1)));
}

#[test]
fn map_feed_codec_oversized_count_is_err_not_a_panic() {
    // tag 0 | seq 0 | n = u32::MAX with no body: must fail without allocating n entries.
    let mut frame = vec![0_u8];
    frame.extend_from_slice(&0_u64.to_le_bytes());
    frame.extend_from_slice(&u32::MAX.to_le_bytes());
    assert!(decode_server_frame(&frame).is_err());
    let mut snap = vec![1_u8];
    snap.extend_from_slice(&0_u64.to_le_bytes());
    snap.extend_from_slice(&0_u64.to_le_bytes());
    snap.extend_from_slice(&u32::MAX.to_le_bytes());
    assert!(decode_server_frame(&snap).is_err());
}
