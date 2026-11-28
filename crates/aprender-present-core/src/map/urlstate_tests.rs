//! Tests for the URL state codec and history policy. FALSIFY-MAP-URL-001 (round trip),
//! -002 (canonical encode), -003 (unknown keys survive) and -004 (push versus replace) live here,
//! with the error paths and the hash-syntax edge cases.

use super::super::{MAX_COORD, MAX_SCALE, MIN_SCALE};
use super::*;
use proptest::prelude::*;

const VIEW: (f64, f64) = (800.0, 600.0);

fn cam(centre: (f64, f64), scale: f64) -> Camera2D {
    Camera2D::new(centre, scale, VIEW).expect("valid camera")
}

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

fn cam_hash(hash: &str) -> Result<UrlState, MapError> {
    decode(hash)
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

#[test]
fn empty_state_encodes_to_empty_hash_and_back() {
    let state = UrlState::default();
    assert_eq!(encode(&state), Ok(String::new()));
    assert_eq!(decode(""), Ok(UrlState::default()));
    assert_eq!(decode("#"), Ok(UrlState::default()));
    assert_eq!(decode("&&"), Ok(UrlState::default()));
}

#[test]
fn typed_keys_encode_in_fixed_order_with_extras_sorted() {
    let mut state = UrlState::new(Some(cam((1.5, -2.0), 2.0)), Some(7), Some(3));
    state.insert_extra("zz", "1").expect("key");
    state.insert_extra("aa", "2").expect("key");
    assert_eq!(
        encode(&state).as_deref(),
        Ok("cam=1.5,-2,2,800,600&sel=7&lens=3&aa=2&zz=1")
    );
}

#[test]
fn camera_keeps_full_precision() {
    let c = cam(
        (0.123_456_789_012_345_67, -9.876_543_210_987_654e8),
        3.141_592_653_589_793,
    );
    let state = UrlState::new(Some(c), None, None);
    let back = decode(&encode(&state).expect("encodes")).expect("decodes");
    assert_eq!(back.camera(), Some(c));
}

// FALSIFY-MAP-URL-003: a committed hash with an unregistered key keeps that key and value.
#[test]
fn unknown_key_survives_decode_then_encode() {
    let fixture = "cam=1,2,4,800,600&zz%20k=v%26w&sel=9";
    let state = cam_hash(fixture).expect("fixture decodes");
    assert_eq!(state.extra().get("zz k").map(String::as_str), Some("v&w"));
    let out = encode(&state).expect("encodes");
    assert!(out.contains("zz%20k=v%26w"), "re-encoded: {out}");
    assert_eq!(state.selection(), Some(9));
}

#[test]
fn plus_is_literal_and_round_trips() {
    let state = decode("tag=a+b").expect("decodes");
    assert_eq!(state.extra().get("tag").map(String::as_str), Some("a+b"));
    assert_eq!(encode(&state).as_deref(), Ok("tag=a%2Bb"));
}

#[test]
fn lowercase_hex_is_accepted_and_normalised() {
    let state = decode("k=%c3%a9").expect("decodes");
    assert_eq!(state.extra().get("k").map(String::as_str), Some("é"));
    assert_eq!(encode(&state).as_deref(), Ok("k=%C3%A9"));
}

#[test]
fn pair_without_equals_has_an_empty_value() {
    let state = decode("flag").expect("decodes");
    assert_eq!(state.extra().get("flag").map(String::as_str), Some(""));
    assert_eq!(encode(&state).as_deref(), Ok("flag="));
}

#[test]
fn decode_rejects_bad_percent_escapes() {
    for bad in ["k=%", "k=%2", "k=%G1", "k=%1G", "%zz=1"] {
        assert_eq!(
            decode(bad),
            Err(MapError::Malformed("percent escape")),
            "{bad}"
        );
    }
}

#[test]
fn decode_rejects_invalid_utf8() {
    assert_eq!(decode("k=%FF"), Err(MapError::Malformed("utf-8")));
}

#[test]
fn decode_rejects_duplicate_keys() {
    assert_eq!(decode("sel=1&sel=2"), Err(MapError::Malformed("sel")));
    assert_eq!(decode("lens=1&lens=2"), Err(MapError::Malformed("lens")));
    assert_eq!(
        decode("cam=0,0,1,1,1&cam=0,0,1,1,1"),
        Err(MapError::Malformed("cam"))
    );
    assert_eq!(
        decode("k=1&k=2"),
        Err(MapError::Malformed("duplicate url key"))
    );
}

#[test]
fn decode_rejects_malformed_typed_values() {
    assert_eq!(decode("sel="), Err(MapError::Malformed("sel")));
    assert_eq!(decode("sel=abc"), Err(MapError::Malformed("sel")));
    assert_eq!(decode("sel=-1"), Err(MapError::Malformed("sel")));
    assert_eq!(decode("lens=4294967296"), Err(MapError::Malformed("lens")));
    assert_eq!(decode("cam=1,2,3,4"), Err(MapError::Malformed("cam")));
    assert_eq!(decode("cam=1,2,3,4,5,6"), Err(MapError::Malformed("cam")));
    assert_eq!(decode("cam=1,x,3,4,5"), Err(MapError::Malformed("cam")));
}

#[test]
fn decode_passes_camera_validation_through() {
    assert_eq!(
        decode("cam=NaN,0,1,1,1"),
        Err(MapError::NonFinite("centre"))
    );
    assert_eq!(decode("cam=0,0,0,1,1"), Err(MapError::NonPositive("scale")));
    assert_eq!(
        decode("cam=0,0,1,-1,1"),
        Err(MapError::Negative("viewport"))
    );
}

#[test]
fn decode_rejects_oversized_input() {
    let long = "k=".to_owned() + &"a".repeat(MAX_URL_LEN);
    assert_eq!(decode(&long), Err(MapError::OutOfRange("url")));
}

#[test]
fn encode_rejects_oversized_output_and_round_trip_still_holds_below_it() {
    let mut state = UrlState::default();
    state
        .insert_extra("k", &"%".repeat(MAX_URL_LEN))
        .expect("key");
    assert_eq!(encode(&state), Err(MapError::OutOfRange("url")));

    let mut small = UrlState::default();
    small.insert_extra("k", &"é".repeat(100)).expect("key");
    let hash = encode(&small).expect("encodes");
    assert_eq!(decode(&hash), Ok(small));
}

#[test]
fn insert_extra_rejects_empty_and_typed_keys() {
    let mut state = UrlState::default();
    assert_eq!(
        state.insert_extra("", "x"),
        Err(MapError::Malformed("url key"))
    );
    for typed in ["cam", "sel", "lens"] {
        assert_eq!(
            state.insert_extra(typed, "x"),
            Err(MapError::Malformed("url key"))
        );
    }
    assert!(state.extra().is_empty());
}

#[test]
fn accessors_expose_the_typed_keys() {
    let c = cam((1.0, 2.0), 3.0);
    let state = UrlState::new(Some(c), Some(5), Some(6));
    assert_eq!(state.camera(), Some(c));
    assert_eq!(state.selection(), Some(5));
    assert_eq!(state.lens(), Some(6));
    assert!(state.extra().is_empty());
}

// FALSIFY-MAP-URL-004: a camera drag adds no entries and is replaced once still; a selection
// and a lens switch each push one entry.
#[test]
fn camera_drag_adds_no_entries_and_replaces_after_debounce() {
    let mut policy = HistoryPolicy::new(UrlState::default());
    let mut pushes = 0;
    for (i, scale) in [1.0, 1.1, 1.2, 1.3].into_iter().enumerate() {
        let event = UrlEvent::Camera(cam((i as f64, 0.0), scale));
        let nav = policy.apply(event, 100 * i as u64).expect("applies");
        if matches!(nav, Nav::Push(_)) {
            pushes += 1;
        }
        assert_eq!(nav, Nav::Nothing);
    }
    assert_eq!(pushes, 0);
    assert!(policy.camera_pending());
    // Debounce runs from the last move at t = 300.
    assert_eq!(policy.poll(300 + CAMERA_DEBOUNCE_MS - 1), Ok(Nav::Nothing));
    let Nav::Replace(hash) = policy.poll(300 + CAMERA_DEBOUNCE_MS).expect("polls") else {
        panic!("expected a replace after the debounce interval");
    };
    assert!(!policy.camera_pending());
    let decoded = decode(&hash).expect("decodes");
    assert_eq!(decoded.camera(), Some(cam((3.0, 0.0), 1.3)));
    assert_eq!(policy.poll(10_000), Ok(Nav::Nothing));
}

#[test]
fn selection_pushes_one_entry_and_decodes_to_it() {
    let mut policy = HistoryPolicy::new(UrlState::default());
    let Nav::Push(hash) = policy
        .apply(UrlEvent::Select(Some(42)), 0)
        .expect("applies")
    else {
        panic!("selection must push");
    };
    assert_eq!(decode(&hash).expect("decodes").selection(), Some(42));
    let Nav::Push(cleared) = policy.apply(UrlEvent::Select(None), 1).expect("applies") else {
        panic!("clearing the selection must push");
    };
    assert_eq!(decode(&cleared).expect("decodes").selection(), None);
}

#[test]
fn lens_switch_pushes_one_entry_and_decodes_to_it() {
    let mut policy = HistoryPolicy::new(UrlState::default());
    let Nav::Push(hash) = policy.apply(UrlEvent::Lens(Some(8)), 0).expect("applies") else {
        panic!("lens switch must push");
    };
    assert_eq!(decode(&hash).expect("decodes").lens(), Some(8));
}

#[test]
fn push_carries_pending_camera_and_clears_the_debounce() {
    let mut policy = HistoryPolicy::new(UrlState::default());
    policy
        .apply(UrlEvent::Camera(cam((5.0, 5.0), 2.0)), 0)
        .expect("applies");
    let Nav::Push(hash) = policy
        .apply(UrlEvent::Select(Some(1)), 10)
        .expect("applies")
    else {
        panic!("selection must push");
    };
    let decoded = decode(&hash).expect("decodes");
    assert_eq!(decoded.camera(), Some(cam((5.0, 5.0), 2.0)));
    assert_eq!(decoded.selection(), Some(1));
    assert!(!policy.camera_pending());
    assert_eq!(policy.poll(10_000), Ok(Nav::Nothing));
}

#[test]
fn failed_push_leaves_state_and_pending_camera_unchanged() {
    let mut state = UrlState::default();
    state
        .insert_extra("k", &"%".repeat(MAX_URL_LEN))
        .expect("key");
    let mut policy = HistoryPolicy::new(state.clone());
    assert_eq!(
        policy.apply(UrlEvent::Select(Some(3)), 0),
        Err(MapError::OutOfRange("url"))
    );
    assert_eq!(policy.state(), &state);
    assert_eq!(
        policy.apply(UrlEvent::Lens(Some(3)), 0),
        Err(MapError::OutOfRange("url"))
    );
    assert_eq!(policy.state(), &state);
}

#[test]
fn failed_debounced_replace_keeps_the_camera_pending() {
    let mut state = UrlState::default();
    state
        .insert_extra("k", &"%".repeat(MAX_URL_LEN))
        .expect("key");
    let mut policy = HistoryPolicy::new(state);
    policy
        .apply(UrlEvent::Camera(cam((0.0, 0.0), 1.0)), 0)
        .expect("camera applies");
    assert_eq!(
        policy.poll(CAMERA_DEBOUNCE_MS),
        Err(MapError::OutOfRange("url"))
    );
    assert!(policy.camera_pending());
}

#[test]
fn history_policy_exposes_the_loaded_state() {
    let loaded = decode("sel=4&lens=2&x=y").expect("decodes");
    let policy = HistoryPolicy::new(loaded.clone());
    assert_eq!(policy.state(), &loaded);
    assert!(!policy.camera_pending());
}

#[test]
fn poll_before_any_camera_move_does_nothing() {
    let mut policy = HistoryPolicy::new(UrlState::default());
    assert_eq!(policy.poll(0), Ok(Nav::Nothing));
}

#[test]
fn poll_with_time_going_backwards_does_not_replace() {
    let mut policy = HistoryPolicy::new(UrlState::default());
    policy
        .apply(UrlEvent::Camera(cam((0.0, 0.0), 1.0)), 1000)
        .expect("applies");
    assert_eq!(policy.poll(0), Ok(Nav::Nothing));
    assert!(policy.camera_pending());
}

#[test]
fn map_error_for_malformed_displays_the_name() {
    assert_eq!(MapError::Malformed("sel").to_string(), "sel is malformed");
}
