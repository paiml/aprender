//! FALSIFY-MAP-URL-001..004 for contract `aprender-viz-map-url-state-v1`.
//! Each test names the planted mutation it must turn red.

use super::generators::map_url_state;
use super::{decode, encode, Camera, FakeHistory, HistoryPolicy, MapUrlError, MapUrlState};
use proptest::prelude::*;

/// Asserts that `state` survives encode then decode, bit for bit.
fn assert_round_trip(state: &MapUrlState) {
    let hash = encode(state).expect("state is finite");
    let back = decode(&hash).expect("encoded hash decodes");
    assert!(
        back.bitwise_eq(state),
        "round trip changed the state: {hash:?}"
    );
}

// ---------------------------------------------------------------------------
// FALSIFY-MAP-URL-001: decode(encode(s)) == s
// planted: camera centre and scale written with two fixed decimals
// ---------------------------------------------------------------------------

#[test]
fn falsify_map_url_001_round_trip_empty_state() {
    assert_round_trip(&MapUrlState::default());
}

#[test]
fn falsify_map_url_001_round_trip_full_precision_camera() {
    let state = MapUrlState {
        camera: Some(Camera {
            x: 0.300_000_000_000_000_04,
            y: 1.0 / 3.0,
            log2_zoom: 12.345_678_901_234_567,
        }),
        ..MapUrlState::default()
    };
    assert_round_trip(&state);
}

#[test]
fn falsify_map_url_001_round_trip_keeps_negative_zero() {
    let state = MapUrlState {
        camera: Some(Camera {
            x: -0.0,
            y: 0.0,
            log2_zoom: -0.0,
        }),
        ..MapUrlState::default()
    };
    let hash = encode(&state).expect("-0.0 is finite");
    assert!(hash.contains("at=-0,0,-0"), "sign of zero lost: {hash}");
    assert_round_trip(&state);
}

#[test]
fn falsify_map_url_001_non_finite_camera_is_rejected() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let state = MapUrlState {
            camera: Some(Camera {
                x: bad,
                y: 0.0,
                log2_zoom: 0.0,
            }),
            ..MapUrlState::default()
        };
        assert_eq!(encode(&state), Err(MapUrlError::NonFinite));
    }
    assert_eq!(decode("at=NaN,0,0"), Err(MapUrlError::NonFinite));
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 1000, failure_persistence: None, ..ProptestConfig::default() })]

    #[test]
    fn falsify_map_url_001_round_trip_property(state in map_url_state()) {
        let hash = encode(&state).expect("generated states are finite");
        let back = decode(&hash).expect("encoded hash decodes");
        prop_assert!(back.bitwise_eq(&state), "round trip changed the state: {}", hash);
    }
}

// ---------------------------------------------------------------------------
// FALSIFY-MAP-URL-002: encode(decode(encode(s))) == encode(s), and encode is pure
// planted: a per-call sequence number written into the hash
// ---------------------------------------------------------------------------

#[test]
fn falsify_map_url_002_empty_state_is_canonical() {
    let state = MapUrlState::default();
    let first = encode(&state).expect("empty state encodes");
    let second = encode(&state).expect("empty state encodes");
    assert_eq!(first, second, "two encodes of the empty state differ");
    let again = encode(&decode(&first).expect("decodes")).expect("encodes");
    assert_eq!(again, first, "re-encode of the empty state differs");
}

#[test]
fn falsify_map_url_002_key_order_is_fixed_then_extension_then_unknown() {
    let mut state = MapUrlState {
        lens: Some("gdp".to_owned()),
        city: Some("Austin".to_owned()),
        state: Some("TX".to_owned()),
        camera: Some(Camera {
            x: 1.0,
            y: 2.0,
            log2_zoom: 3.0,
        }),
        advisor: Some("ada".to_owned()),
        layers: vec!["roads".to_owned(), "parks".to_owned()],
        ..MapUrlState::default()
    };
    state
        .extensions
        .insert("x-tint".to_owned(), "warm".to_owned());
    state
        .extensions
        .insert("x-legend".to_owned(), "on".to_owned());
    state.opaque.insert("zz".to_owned(), "1".to_owned());
    state.opaque.insert("aa".to_owned(), "2".to_owned());
    assert_eq!(
        encode(&state).expect("encodes"),
        "lens=gdp&city=Austin&state=TX&at=1,2,3&advisor=ada&layers=roads,parks&x-legend=on&x-tint=warm&aa=2&zz=1"
    );
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 1000, failure_persistence: None, ..ProptestConfig::default() })]

    #[test]
    fn falsify_map_url_002_encode_is_canonical_property(state in map_url_state()) {
        let first = encode(&state).expect("generated states are finite");
        let second = encode(&state).expect("generated states are finite");
        prop_assert_eq!(&first, &second, "two encodes of one state differ");
        let decoded = decode(&first).expect("encoded hash decodes");
        let again = encode(&decoded).expect("decoded state encodes");
        prop_assert_eq!(&again, &first, "encode(decode(encode(s))) differs from encode(s)");
    }
}

// ---------------------------------------------------------------------------
// FALSIFY-MAP-URL-003: an unregistered key survives decode then encode
// planted: decoder keeps only registered keys and drops the rest
// ---------------------------------------------------------------------------

const FIXTURE_HASH: &str = "#lens=gdp&zz-future=a%2Fb%3D1&at=1,2,3&x-tint=warm";

#[test]
fn falsify_map_url_003_unregistered_key_survives_decode_then_encode() {
    let state = decode(FIXTURE_HASH).expect("fixture decodes");
    assert_eq!(
        state.opaque.get("zz-future").map(String::as_str),
        Some("a/b=1"),
        "unregistered key not kept as opaque"
    );
    let out = encode(&state).expect("fixture state encodes");
    assert!(
        out.contains("zz-future=a%2Fb%3D1"),
        "unregistered pair lost on re-save: {out}"
    );
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 1000, failure_persistence: None, ..ProptestConfig::default() })]

    #[test]
    fn falsify_map_url_003_unknown_keys_survive_property(state in map_url_state()) {
        let hash = encode(&state).expect("generated states are finite");
        let back = decode(&hash).expect("encoded hash decodes");
        for (key, value) in &state.opaque {
            prop_assert_eq!(back.opaque.get(key), Some(value), "opaque key {} lost", key);
        }
    }
}

// ---------------------------------------------------------------------------
// FALSIFY-MAP-URL-004: selection and lens switch push one entry each;
// camera drags add none and replace after the debounce.
// planted: camera moves push a history entry on every change
// ---------------------------------------------------------------------------

const DEBOUNCE: u64 = 5;

fn fresh() -> (HistoryPolicy, FakeHistory) {
    (
        HistoryPolicy::new(MapUrlState::default(), DEBOUNCE),
        FakeHistory::new(""),
    )
}

/// Asserts that `actual` is present and equal to `expected` bit for bit. The
/// expected value must come from the test's own input, never from the policy.
fn assert_camera_bits(actual: Option<Camera>, expected: Camera, what: &str) {
    let Some(actual) = actual else {
        panic!("{what}: camera is missing");
    };
    assert_eq!(
        (
            actual.x.to_bits(),
            actual.y.to_bits(),
            actual.log2_zoom.to_bits()
        ),
        (
            expected.x.to_bits(),
            expected.y.to_bits(),
            expected.log2_zoom.to_bits()
        ),
        "{what}"
    );
}

#[test]
fn falsify_map_url_004_camera_drag_adds_no_entries_and_replaces_after_debounce() {
    let (mut policy, mut history) = fresh();
    let before = history.entries();
    let mut last_input: Option<Camera> = None;
    for i in 0_u32..20 {
        let tick = u64::from(i);
        let camera = Camera {
            x: f64::from(i) * 0.25,
            y: f64::from(i) * -0.5,
            log2_zoom: 3.0 + f64::from(i) / 7.0,
        };
        policy.camera_move(tick, camera).expect("camera is finite");
        last_input = Some(camera);
        assert!(
            !policy.tick(tick, &mut history).expect("tick encodes"),
            "replace fired before the debounce"
        );
    }
    assert_eq!(
        history.entries(),
        before,
        "camera drag added history entries"
    );
    assert!(
        !policy.tick(23, &mut history).expect("tick encodes"),
        "replace fired 4 ticks after the last move, debounce is 5"
    );
    assert!(
        policy.tick(24, &mut history).expect("tick encodes"),
        "no replace 5 ticks after the last move"
    );
    assert_eq!(history.entries(), before, "replace added an entry");
    let last_input = last_input.expect("the drag made moves");
    let decoded = decode(history.current_hash()).expect("hash decodes");
    assert_camera_bits(
        decoded.camera,
        last_input,
        "replaced URL hash does not carry the last camera passed to camera_move",
    );
    assert!(
        decoded.bitwise_eq(policy.state()),
        "URL hash does not decode to the policy state: {}",
        history.current_hash()
    );
}

#[test]
fn falsify_map_url_004_selection_pushes_exactly_one_entry() {
    let (mut policy, mut history) = fresh();
    let before = history.entries();
    policy
        .select(Some("Austin"), &mut history)
        .expect("selection encodes");
    assert_eq!(
        history.entries(),
        before + 1,
        "selection did not push one entry"
    );
    let decoded = decode(history.current_hash()).expect("hash decodes");
    assert_eq!(decoded.city.as_deref(), Some("Austin"));
}

#[test]
fn falsify_map_url_004_lens_switch_pushes_exactly_one_entry() {
    let (mut policy, mut history) = fresh();
    let before = history.entries();
    policy
        .switch_lens("gdp", &mut history)
        .expect("lens switch encodes");
    assert_eq!(
        history.entries(),
        before + 1,
        "lens switch did not push one entry"
    );
    let decoded = decode(history.current_hash()).expect("hash decodes");
    assert_eq!(decoded.lens.as_deref(), Some("gdp"));
}

#[test]
fn falsify_map_url_004_selection_after_drag_carries_the_camera() {
    let (mut policy, mut history) = fresh();
    let camera = Camera {
        x: 10.5,
        y: -2.25,
        log2_zoom: 4.125,
    };
    policy.camera_move(0, camera).expect("camera is finite");
    policy
        .select(Some("Austin"), &mut history)
        .expect("selection encodes");
    assert_eq!(
        history.entries(),
        2,
        "drag then selection should add one entry"
    );
    assert!(
        !policy.camera_pending(),
        "selection push should absorb the pending camera replace"
    );
    let decoded = decode(history.current_hash()).expect("hash decodes");
    assert_camera_bits(
        decoded.camera,
        camera,
        "pushed URL hash does not carry the camera passed to camera_move",
    );
    assert!(
        decoded.bitwise_eq(policy.state()),
        "pushed hash does not decode to the policy state"
    );
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 1000, failure_persistence: None, ..ProptestConfig::default() })]

    #[test]
    fn falsify_map_url_004_selection_decodes_to_selection_property(city in "[a-zA-Z&=#%\u{e9}]{1,8}") {
        let (mut policy, mut history) = fresh();
        policy.select(Some(city.as_str()), &mut history).expect("selection encodes");
        prop_assert_eq!(history.entries(), 2);
        let decoded = decode(history.current_hash()).expect("hash decodes");
        prop_assert_eq!(decoded.city.as_deref(), Some(city.as_str()));
    }
}
