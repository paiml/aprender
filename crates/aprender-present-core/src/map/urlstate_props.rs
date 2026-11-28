//! Property tests for the URL state codec: FALSIFY-MAP-URL-001 (round trip), -002 (canonical
//! encode), and the bounded-length properties. Kept apart from `urlstate_tests.rs` to stay under
//! the 500-line file gate. Included as a child of the `tests` module.

use super::*;
use crate::map::{MAX_COORD, MAX_SCALE, MIN_SCALE};
use proptest::prelude::*;

fn camera_strategy() -> impl Strategy<Value = Option<Camera2D>> {
    proptest::option::of(
        (
            -MAX_COORD..MAX_COORD,
            -MAX_COORD..MAX_COORD,
            MIN_SCALE..MAX_SCALE,
            0.0..MAX_COORD,
            0.0..MAX_COORD,
        )
            .prop_map(|(x, y, s, w, h)| Camera2D::new((x, y), s, (w, h)).expect("in range")),
    )
}

fn key_strategy() -> impl Strategy<Value = String> {
    "[a-z&=#%+ é😀]{1,6}".prop_filter("typed keys are not extension keys", |k| {
        !matches!(k.as_str(), "cam" | "sel" | "lens")
    })
}

fn state_strategy() -> impl Strategy<Value = UrlState> {
    (
        camera_strategy(),
        proptest::option::of(any::<u64>()),
        proptest::option::of(any::<u32>()),
        proptest::collection::btree_map(key_strategy(), "[a-z&=#%+ é😀0-9]{0,12}", 0..4),
    )
        .prop_map(|(camera, sel, lens, extra)| {
            let mut state = UrlState::new(camera, sel, lens);
            for (k, v) in &extra {
                state.insert_extra(k, v).expect("extension key");
            }
            state
        })
}

proptest! {
    // FALSIFY-MAP-URL-001: decode(encode(s)) == s, including full-precision camera values.
    #[test]
    fn round_trip_returns_the_same_state(state in state_strategy()) {
        let hash = encode(&state).expect("encodes");
        prop_assert_eq!(decode(&hash), Ok(state));
    }

    // FALSIFY-MAP-URL-002: encode(decode(encode(s))) == encode(s), and encode is repeatable.
    #[test]
    fn encode_is_canonical_and_repeatable(state in state_strategy()) {
        let once = encode(&state).expect("encodes");
        let again = encode(&state).expect("encodes");
        prop_assert_eq!(&once, &again);
        let decoded = decode(&once).expect("decodes");
        prop_assert_eq!(encode(&decoded), Ok(once));
    }

    // Any input: decode returns Ok or Err, never panics; an Ok state re-encodes canonically.
    #[test]
    fn decode_of_arbitrary_text_never_panics(text in "\\PC{0,300}") {
        if let Ok(state) = decode(&text) {
            let hash = encode(&state).expect("decoded state encodes");
            prop_assert_eq!(decode(&hash), Ok(state));
        }
    }

    #[test]
    fn decode_of_arbitrary_bytes_never_panics(bytes in proptest::collection::vec(any::<u8>(), 0..200)) {
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let _ = decode(&text);
    }
}

proptest! {
    // Inputs long enough to reach the limit: a decoded state always re-encodes within it, and
    // one that is refused was refused for length or syntax, never a panic.
    #[test]
    fn long_inputs_near_the_limit_decode_or_refuse_cleanly(
        text in "[a&=%+ é#]{3900,4300}"
    ) {
        match decode(&text) {
            Ok(state) => {
                let hash = encode(&state).expect("decoded state encodes");
                prop_assert!(hash.len() <= MAX_URL_LEN);
                prop_assert_eq!(decode(&hash), Ok(state));
            }
            Err(err) => prop_assert!(matches!(
                err,
                MapError::OutOfRange("url") | MapError::Malformed(_)
            )),
        }
    }

    // Alphanumeric keys of any length: a key-only pair encodes as `key=`, so it decodes exactly
    // when that fits the limit, and the refusal is the length error, never a panic.
    #[test]
    fn long_keys_decode_exactly_when_their_encoding_fits(text in "[a-z]{4090,4100}") {
        let result = decode(&text);
        prop_assert_eq!(result.is_ok(), text.len() < MAX_URL_LEN);
        if let Err(err) = result {
            prop_assert_eq!(err, MapError::OutOfRange("url"));
        }
    }

    // Values long enough to reach the limit: insert_extra accepts exactly when the state encodes.
    // On refusal the state is unchanged and the state with that pair is unencodable.
    #[test]
    fn insert_extra_accepts_exactly_the_encodable_states(
        value in "[a%+ é&]{1300,1500}"
    ) {
        let mut state = nearly_full_state();
        let before = state.clone();
        match state.insert_extra("k", &value) {
            Ok(()) => {
                let hash = encode(&state).expect("accepted state encodes");
                prop_assert!(hash.len() <= MAX_URL_LEN);
            }
            Err(err) => {
                prop_assert_eq!(err, MapError::OutOfRange("url"));
                prop_assert_eq!(&state, &before);
                let mut trial = before.clone();
                trial.extra.insert("k".to_owned(), value.to_owned());
                prop_assert!(encode(&trial).is_err());
            }
        }
    }
}
