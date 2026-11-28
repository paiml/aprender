//! Decode fuzz: `decode` must not panic on arbitrary input. Any input it accepts
//! must encode, and the encoding must decode back to the same state.

use super::{decode, encode};
use proptest::prelude::*;

/// Decodes `text`. If it is accepted, the state must encode and re-decode unchanged.
fn check_decode_total(text: &str) {
    if let Ok(state) = decode(text) {
        let encoded = encode(&state).expect("a decoded state encodes");
        let back = decode(&encoded).expect("encoded hash decodes");
        assert!(back.bitwise_eq(&state), "decode is not stable on {text:?}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 1000, failure_persistence: None, ..ProptestConfig::default() })]

    #[test]
    fn decode_never_panics_on_arbitrary_bytes(bytes in proptest::collection::vec(any::<u8>(), 0..96)) {
        let text = String::from_utf8_lossy(&bytes).into_owned();
        check_decode_total(&text);
    }

    #[test]
    fn decode_never_panics_on_arbitrary_text(text in "\\PC{0,96}") {
        check_decode_total(&text);
    }

    #[test]
    fn decode_never_panics_on_hash_shaped_text(text in "[a-z0-9&=#%,.+\\-]{0,96}") {
        check_decode_total(&text);
    }
}
