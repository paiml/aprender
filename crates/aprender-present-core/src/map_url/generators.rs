//! proptest strategies for map URL states (MAP-URL-INV-001..003).

use super::state::{Camera, MapUrlState, EXTENSION_KEYS, FIXED_KEYS};
use proptest::collection::{btree_map, vec};
use proptest::option;
use proptest::prelude::*;

/// Finite f64 values: the edges, signed zeros, and full-precision draws.
pub fn finite_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.0_f64),
        Just(-0.0_f64),
        Just(f64::MIN_POSITIVE),
        Just(f64::MAX),
        Just(f64::MIN),
        -1.0e6_f64..1.0e6_f64,
        any::<f64>().prop_filter("finite", |v| v.is_finite()),
    ]
}

/// A finite camera.
pub fn camera() -> impl Strategy<Value = Camera> {
    (finite_f64(), finite_f64(), finite_f64()).prop_map(|(x, y, log2_zoom)| Camera {
        x,
        y,
        log2_zoom,
    })
}

/// Free text with the characters the codec must escape, plus non-ASCII.
fn text() -> impl Strategy<Value = String> {
    prop_oneof!["[a-zA-Z0-9&=#%,+ \u{e9}\u{4e2d}-]{0,10}", "\\PC{0,8}",]
}

fn layer() -> impl Strategy<Value = String> {
    "[a-z&=#%,\u{e9}]{1,6}"
}

/// A key that no fixed or extension name claims, so it must come back opaque.
fn unknown_key() -> impl Strategy<Value = String> {
    "[a-z&=#%\u{e9}-]{1,6}".prop_filter("not a known key", |k| {
        !FIXED_KEYS.contains(&k.as_str()) && !EXTENSION_KEYS.contains(&k.as_str())
    })
}

/// Any map URL state, including the empty state.
pub fn map_url_state() -> impl Strategy<Value = MapUrlState> {
    (
        option::of(text()),
        option::of(text()),
        option::of(text()),
        option::of(camera()),
        option::of(text()),
        vec(layer(), 0..4),
        btree_map(prop::sample::select(EXTENSION_KEYS.to_vec()), text(), 0..3),
        btree_map(unknown_key(), text(), 0..3),
    )
        .prop_map(
            |(lens, city, state, camera, advisor, layers, extensions, opaque)| MapUrlState {
                lens,
                city,
                state,
                camera,
                advisor,
                layers,
                extensions: extensions
                    .into_iter()
                    .map(|(k, v)| (k.to_owned(), v))
                    .collect(),
                opaque,
            },
        )
}
