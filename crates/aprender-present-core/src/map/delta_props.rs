//! Property and fuzz tests for the lens delta wire format (MAP-8 slice 1). Included from
//! `delta_tests`, so the shared fixtures are in scope through `super`.

use super::*;
use proptest::prelude::*;

fn finite() -> impl Strategy<Value = f64> {
    any::<f64>().prop_filter("finite", |v| v.is_finite())
}

fn delta_strategy() -> impl Strategy<Value = Delta> {
    (
        any::<u32>(),
        any::<u64>(),
        proptest::collection::vec((0..CELLS, proptest::option::of(finite())), 0..64),
    )
        .prop_map(|(lens, seq, cells)| Delta::new(lens, seq, cells).expect("finite values"))
}

proptest! {
    // decode(encode(d)) == d for every lens id, sequence number and cell list inside the grid.
    #[test]
    fn round_trip_returns_the_same_delta(d in delta_strategy()) {
        let bytes = d.encode().expect("encodes");
        prop_assert_eq!(Delta::decode(&bytes, CELLS), Ok(d));
    }

    // Random bytes of any length: decode returns Ok or Err and never panics. An Ok delta names
    // only cells inside the grid and only finite values.
    #[test]
    fn decode_of_random_bytes_never_panics(bytes in proptest::collection::vec(any::<u8>(), 0..512)) {
        if let Ok(d) = Delta::decode(&bytes, CELLS) {
            prop_assert!(d.cells().iter().all(|&(c, v)| c < CELLS && v.is_none_or(f64::is_finite)));
        }
    }

    // A valid encoding with one byte overwritten and then cut short: never a panic.
    #[test]
    fn decode_of_damaged_valid_bytes_never_panics(
        d in delta_strategy(),
        at in any::<usize>(),
        byte in any::<u8>(),
        cut in any::<usize>(),
    ) {
        let mut bytes = d.encode().expect("encodes");
        if !bytes.is_empty() {
            let i = at % bytes.len();
            bytes[i] = byte;
        }
        let keep = cut % (bytes.len() + 1);
        let _ = Delta::decode(&bytes[..keep], CELLS);
    }
}
