//! Tests for the URL state codec and history policy. FALSIFY-MAP-URL-001 (round trip),
//! -002 (canonical encode), -003 (unknown keys survive) and -004 (push versus replace) live here,
//! with the error paths and the hash-syntax edge cases. The property tests are in `props`.

use super::*;

#[path = "urlstate_props.rs"]
mod props;

const VIEW: (f64, f64) = (800.0, 600.0);

fn cam(centre: (f64, f64), scale: f64) -> Camera2D {
    Camera2D::new(centre, scale, VIEW).expect("valid camera")
}

fn cam_hash(hash: &str) -> Result<UrlState, MapError> {
    decode(hash)
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
        0.577_215_664_901_532_9,
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

/// A state whose encoding is `k=` plus 4089 bytes: 4091 of the 4096-byte limit.
fn nearly_full_state() -> UrlState {
    let mut state = UrlState::default();
    state.insert_extra("k", &"%".repeat(1363)).expect("fits");
    state
}

#[test]
fn encode_rejects_oversized_output_and_round_trip_still_holds_below_it() {
    let mut state = nearly_full_state();
    assert_eq!(encode(&state).map(|h| h.len()), Ok(4091));
    // Typed keys are set by the constructor or the policy, not by insert_extra, so a state can
    // reach the limit only this way; the child test module may set the field directly.
    state.selection = Some(u64::MAX);
    assert_eq!(encode(&state), Err(MapError::OutOfRange("url")));

    let mut small = UrlState::default();
    small.insert_extra("k", &"é".repeat(100)).expect("key");
    let hash = encode(&small).expect("encodes");
    assert_eq!(decode(&hash), Ok(small));
}

#[test]
fn insert_extra_refuses_a_pair_that_cannot_be_encoded_and_restores_the_state() {
    let mut state = nearly_full_state();
    let before = state.clone();
    // Replacing `k` with a longer value would overflow the limit.
    assert_eq!(
        state.insert_extra("k", &"%".repeat(1400)),
        Err(MapError::OutOfRange("url"))
    );
    assert_eq!(state, before);
    // A new key that does not fit is refused and removed again.
    assert_eq!(
        state.insert_extra("zz", "1234567890"),
        Err(MapError::OutOfRange("url"))
    );
    assert_eq!(state, before);
}

#[test]
fn insert_extra_refuses_raw_input_past_the_limit_without_allocating_it() {
    let mut state = UrlState::default();
    let huge = "%".repeat(1_000_000);
    assert_eq!(
        state.insert_extra("k", &huge),
        Err(MapError::OutOfRange("url"))
    );
    assert_eq!(
        state.insert_extra(&huge, "v"),
        Err(MapError::OutOfRange("url"))
    );
    assert!(state.extra().is_empty());
}

#[test]
fn insert_extra_replaces_an_existing_pair_within_the_limit() {
    let mut state = UrlState::default();
    state.insert_extra("k", "1").expect("key");
    state.insert_extra("k", "2").expect("replace");
    assert_eq!(state.extra().get("k").map(String::as_str), Some("2"));
}

#[test]
fn decode_refuses_an_empty_key() {
    assert_eq!(decode("=foo"), Err(MapError::Malformed("url key")));
    assert_eq!(decode("="), Err(MapError::Malformed("url key")));
    assert_eq!(decode("a=1&=foo"), Err(MapError::Malformed("url key")));
    // An escaped '=' is a non-empty key and is accepted.
    let state = decode("%3D=x").expect("key '=' is valid");
    assert_eq!(state.extra().get("=").map(String::as_str), Some("x"));
}

#[test]
fn decode_refuses_inputs_whose_canonical_encoding_is_too_long() {
    // Accepted length on input, but the canonical re-encoding exceeds the limit.
    let long_key = "a".repeat(MAX_URL_LEN);
    assert_eq!(decode(&long_key), Err(MapError::OutOfRange("url")));
    let plus_value = format!("k={}", "+".repeat(1366));
    assert_eq!(plus_value.len(), 1368);
    assert_eq!(decode(&plus_value), Err(MapError::OutOfRange("url")));
}

#[test]
fn decode_measures_length_after_the_leading_hash() {
    // A full-length encoding with a leading '#' is 4097 bytes in total but 4096 after it.
    let body = format!("k={}", "a".repeat(4094));
    assert_eq!(body.len(), MAX_URL_LEN);
    let hashed = format!("#{body}");
    assert_eq!(decode(&hashed).map(|s| encode(&s)), Ok(Ok(body.clone())));
    assert_eq!(decode(&body).map(|s| encode(&s)), Ok(Ok(body)));
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
    let state = nearly_full_state();
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

// Finding (Haiku, 01:10Z): a camera move that cannot be encoded must be refused, not stored, or
// every later poll fails and the URL stays stalled for the session.
#[test]
fn camera_move_refused_when_the_state_cannot_encode() {
    let mut policy = HistoryPolicy::new(nearly_full_state());
    let before = policy.state().clone();
    assert_eq!(
        policy.apply(UrlEvent::Camera(cam((0.0, 0.0), 1.0)), 0),
        Err(MapError::OutOfRange("url"))
    );
    assert_eq!(policy.state(), &before);
    assert!(!policy.camera_pending());
    assert_eq!(policy.poll(10_000), Ok(Nav::Nothing));
}

// Finding (agy lane 1): -0.0 compares equal to 0.0, so it must encode to the same hash.
#[test]
fn negative_zero_camera_encodes_like_positive_zero() {
    let pos = UrlState::new(Some(cam((0.0, 0.0), 1.0)), None, None);
    let neg = UrlState::new(Some(cam((-0.0, -0.0), 1.0)), None, None);
    assert_eq!(pos, neg);
    assert_eq!(encode(&neg), encode(&pos));
    assert_eq!(encode(&neg).as_deref(), Ok("cam=0,0,1,800,600"));
}

// Finding (agy lane 1): the Some(old) rollback in insert_extra must run. The value fits the
// per-value bound, so only the total encode can refuse it.
#[test]
fn failed_replace_restores_the_previous_value_of_the_same_key() {
    let mut state = UrlState::default();
    state.insert_extra("a", &"%".repeat(1000)).expect("fits");
    state.insert_extra("b", "x").expect("fits");
    let before = state.clone();
    assert_eq!(
        state.insert_extra("b", &"%".repeat(400)),
        Err(MapError::OutOfRange("url"))
    );
    assert_eq!(state.extra().get("b").map(String::as_str), Some("x"));
    assert_eq!(state, before);
}

// Finding (agy lane 1): a refused push must leave the pending camera pending. Camera 17 bytes
// plus the 4073-byte pair fit; the selection adds 25 bytes and overflows.
#[test]
fn failed_push_keeps_the_pending_camera() {
    let mut state = UrlState::default();
    state.insert_extra("k", &"%".repeat(1351)).expect("fits");
    let mut policy = HistoryPolicy::new(state);
    policy
        .apply(UrlEvent::Camera(cam((0.0, 0.0), 1.0)), 0)
        .expect("camera fits");
    assert!(policy.camera_pending());
    let before = policy.state().clone();
    assert_eq!(
        policy.apply(UrlEvent::Select(Some(u64::MAX)), 10),
        Err(MapError::OutOfRange("url"))
    );
    assert!(policy.camera_pending());
    assert_eq!(policy.state(), &before);
    assert!(matches!(
        policy.poll(CAMERA_DEBOUNCE_MS),
        Ok(Nav::Replace(_))
    ));
}

#[test]
fn map_error_for_malformed_displays_the_name() {
    assert_eq!(MapError::Malformed("sel").to_string(), "sel is malformed");
}
