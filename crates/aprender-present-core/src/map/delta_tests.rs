//! Wire-format tests for the lens delta (MAP-8 slice 1): the stated layout byte for byte, and each
//! refusal with its exact error. Property and fuzz tests are in `props`.

use super::*;

#[path = "delta_props.rs"]
mod props;

/// Cell count of the 117 by 90 map grid.
const CELLS: u32 = 10_530;

fn sample() -> Delta {
    Delta::new(7, 3, vec![(5, Some(1.5)), (9, None)]).expect("finite values")
}

#[test]
fn layout_is_little_endian_and_columnar() {
    let bytes = sample().encode().expect("encodes");
    assert_eq!(bytes.len(), 42, "16 header bytes + 2 entries x 13");
    assert_eq!(&bytes[..4], &[7u8, 0, 0, 0], "lens id, u32 little-endian");
    assert_eq!(
        &bytes[4..12],
        &3u64.to_le_bytes(),
        "sequence, u64 little-endian"
    );
    assert_eq!(&bytes[12..16], &[2u8, 0, 0, 0], "entry count, u32");
    assert_eq!(&bytes[16..24], &[5u8, 0, 0, 0, 9, 0, 0, 0], "cell column");
    assert_eq!(&bytes[24..26], &[TAG_SOME, TAG_NONE], "tag column");
    assert_eq!(&bytes[26..34], &1.5f64.to_le_bytes(), "value column: 1.5");
    assert_eq!(&bytes[34..42], &[0u8; 8], "a no-data entry holds zero bits");
}

#[test]
fn tag_bytes_are_zero_for_no_data_and_one_for_a_value() {
    assert_eq!(TAG_NONE, 0);
    assert_eq!(TAG_SOME, 1);
}

#[test]
fn empty_delta_is_a_bare_header_and_round_trips() {
    let empty = Delta::new(1, 1, vec![]).expect("empty");
    let bytes = empty.encode().expect("encodes");
    assert_eq!(bytes.len(), HEADER_BYTES);
    assert_eq!(Delta::decode(&bytes, CELLS), Ok(empty));
}

#[test]
fn sample_round_trips_exactly() {
    let d = sample();
    let bytes = d.encode().expect("encodes");
    assert_eq!(Delta::decode(&bytes, CELLS), Ok(d));
}

#[test]
fn sequence_keeps_its_full_width() {
    let d = Delta::new(1, u64::MAX, vec![(0, Some(-0.0))]).expect("finite");
    let bytes = d.encode().expect("encodes");
    let back = Delta::decode(&bytes, CELLS).expect("decodes");
    assert_eq!(back.seq(), u64::MAX);
}

#[test]
fn every_proper_prefix_is_refused_as_truncated() {
    let bytes = sample().encode().expect("encodes");
    let refused = (0..bytes.len())
        .filter(|&n| Delta::decode(&bytes[..n], CELLS) == Err(MapError::Truncated("delta")))
        .count();
    assert_eq!(refused, 42, "every length from 0 to 41 bytes");
}

#[test]
fn bytes_after_the_declared_end_are_refused() {
    let mut bytes = sample().encode().expect("encodes");
    bytes.push(0);
    assert_eq!(
        Delta::decode(&bytes, CELLS),
        Err(MapError::TrailingBytes("delta"))
    );
    bytes.push(0);
    assert_eq!(
        Delta::decode(&bytes, CELLS),
        Err(MapError::TrailingBytes("delta"))
    );
}

#[test]
fn a_declared_count_larger_than_the_buffer_is_refused_before_any_allocation() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(
        Delta::decode(&bytes, CELLS),
        Err(MapError::Truncated("delta"))
    );
}

#[test]
fn cell_ids_must_be_inside_the_grid() {
    let edge = Delta::new(1, 1, vec![(CELLS - 1, None)]).expect("finite");
    let bytes = edge.encode().expect("encodes");
    assert_eq!(Delta::decode(&bytes, CELLS), Ok(edge));

    let outside = Delta::new(1, 1, vec![(CELLS, None)]).expect("finite");
    let bytes = outside.encode().expect("encodes");
    assert_eq!(
        Delta::decode(&bytes, CELLS),
        Err(MapError::CellOutOfRange(CELLS))
    );
    assert!(
        Delta::decode(&bytes, CELLS + 1).is_ok(),
        "the bound is the grid's cell count"
    );
}

#[test]
fn an_unknown_tag_byte_is_refused() {
    let mut bytes = sample().encode().expect("encodes");
    bytes[24] = 2;
    assert_eq!(Delta::decode(&bytes, CELLS), Err(MapError::UnknownTag(2)));
    bytes[24] = 0xFF;
    assert_eq!(
        Delta::decode(&bytes, CELLS),
        Err(MapError::UnknownTag(0xFF))
    );
}

#[test]
fn a_nan_value_is_refused() {
    let mut bytes = sample().encode().expect("encodes");
    bytes[26..34].copy_from_slice(&f64::NAN.to_le_bytes());
    assert_eq!(
        Delta::decode(&bytes, CELLS),
        Err(MapError::NonFinite("delta value"))
    );
}

#[test]
fn an_infinite_value_is_refused_too() {
    let mut bytes = sample().encode().expect("encodes");
    bytes[26..34].copy_from_slice(&f64::INFINITY.to_le_bytes());
    assert_eq!(
        Delta::decode(&bytes, CELLS),
        Err(MapError::NonFinite("delta value"))
    );
}

#[test]
fn building_a_delta_refuses_non_finite_values() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            Delta::new(1, 1, vec![(0, Some(bad))]),
            Err(MapError::NonFinite("delta value"))
        );
    }
    assert!(Delta::new(1, 1, vec![(0, None)]).is_ok());
}
