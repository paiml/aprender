//! Tests for the zoom-bucketed sprite atlas (MAP-5). The golden is a committed literal: the FNV-1a
//! hash of the full RGBA bytes of one sprite, plus its dimensions and a few exact pixels.

use super::*;

const GOLDEN_KEY: &str = "hex-forest";
const GOLDEN_SEED: u64 = 7;
/// Bucket for radius 8.0 px: floor(4 * log2(8)) = 12, side 16.
const GOLDEN_BUCKET: u32 = 12;
const GOLDEN_SIDE: u32 = 16;
const GOLDEN_FNV: u64 = 0x58c3_3b46_100a_0ec3;
/// Exact pixels of the golden sprite (row-major, x then y).
const GOLDEN_PIXEL_CENTRE: [u8; 4] = [234, 84, 102, 255];
const GOLDEN_PIXEL_3_9: [u8; 4] = [221, 81, 87, 255];

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn pixel(s: &Sprite, x: usize, y: usize) -> [u8; 4] {
    let n = s.side() as usize;
    let at = (y * n + x) * 4;
    [
        s.rgba()[at],
        s.rgba()[at + 1],
        s.rgba()[at + 2],
        s.rgba()[at + 3],
    ]
}

#[test]
fn size_bucket_is_floor_of_four_log2_radius() {
    assert_eq!(size_bucket(1.0), Ok(0));
    assert_eq!(size_bucket(2.0), Ok(4));
    assert_eq!(size_bucket(8.0), Ok(12));
    assert_eq!(size_bucket(16.0), Ok(16));
    assert_eq!(size_bucket(256.0), Ok(32));
    assert_eq!(size_bucket(255.99), Ok(31));
}

#[test]
fn size_bucket_switches_exactly_at_a_quarter_octave_boundary() {
    // Both sides of the boundary at radius 2.0 (bucket 4): just below falls to 3, at and above is 4.
    let below = 2.0 * (1.0 - 1e-9);
    let above = 2.0 * (1.0 + 1e-9);
    assert_eq!(size_bucket(below), Ok(3), "just below 2.0");
    assert_eq!(size_bucket(2.0), Ok(4), "at 2.0");
    assert_eq!(size_bucket(above), Ok(4), "just above 2.0");
}

#[test]
fn size_bucket_rejects_non_finite_non_positive_and_out_of_range_radius() {
    assert_eq!(size_bucket(f64::NAN), Err(MapError::NonFinite("radius")));
    assert_eq!(
        size_bucket(f64::INFINITY),
        Err(MapError::NonFinite("radius"))
    );
    assert_eq!(size_bucket(0.0), Err(MapError::NonPositive("radius")));
    assert_eq!(size_bucket(-2.0), Err(MapError::NonPositive("radius")));
    assert_eq!(size_bucket(0.5), Err(MapError::OutOfRange("radius")));
    assert_eq!(size_bucket(256.5), Err(MapError::OutOfRange("radius")));
    assert_eq!(size_bucket(1e300), Err(MapError::OutOfRange("radius")));
}

#[test]
fn side_for_bucket_is_rounded_quarter_octave_of_two_radius() {
    assert_eq!(side_for_bucket(0), Ok(2));
    assert_eq!(side_for_bucket(1), Ok(2));
    assert_eq!(side_for_bucket(12), Ok(16));
    assert_eq!(side_for_bucket(13), Ok(19));
    assert_eq!(side_for_bucket(21), Ok(76));
    assert_eq!(side_for_bucket(MAX_SPRITE_BUCKET), Ok(MAX_SPRITE_SIDE));
}

#[test]
fn side_for_bucket_never_shrinks_and_stays_bounded() {
    let mut prev = 0;
    for b in 0..=MAX_SPRITE_BUCKET {
        let side = side_for_bucket(b).expect("bucket in range");
        assert!(side >= prev, "side fell at bucket {b}");
        assert!(side <= MAX_SPRITE_SIDE, "side {side} over the cap at {b}");
        prev = side;
    }
}

#[test]
fn side_for_bucket_rejects_a_bucket_above_the_cap() {
    assert_eq!(
        side_for_bucket(MAX_SPRITE_BUCKET + 1),
        Err(MapError::OutOfRange("bucket"))
    );
}

#[test]
fn golden_sprite_matches_committed_hash_dimensions_and_pixels() {
    let s = generate_sprite(GOLDEN_KEY, GOLDEN_SEED, GOLDEN_BUCKET).expect("golden bucket");
    assert_eq!(s.side(), GOLDEN_SIDE);
    assert_eq!(s.rgba().len(), (GOLDEN_SIDE * GOLDEN_SIDE * 4) as usize);
    assert_eq!(fnv1a(s.rgba()), GOLDEN_FNV, "golden hash");
    // Corner is outside the disc: transparent black.
    assert_eq!(pixel(&s, 0, 0), [0, 0, 0, 0]);
    // Centre pixel is inside the disc: opaque.
    assert_eq!(pixel(&s, 8, 8), GOLDEN_PIXEL_CENTRE);
    assert_eq!(pixel(&s, 3, 9), GOLDEN_PIXEL_3_9);
}

#[test]
fn same_key_in_two_buckets_gives_two_different_sizes() {
    let small = generate_sprite(GOLDEN_KEY, GOLDEN_SEED, 13).expect("bucket 13");
    let large = generate_sprite(GOLDEN_KEY, GOLDEN_SEED, 21).expect("bucket 21");
    assert_eq!(small.side(), 19);
    assert_eq!(large.side(), 76);
    assert_ne!(small.side(), large.side());
    assert_ne!(small.rgba().len(), large.rgba().len());
}

#[test]
fn different_seed_gives_different_bytes_and_same_seed_is_repeatable() {
    let a = generate_sprite(GOLDEN_KEY, 7, GOLDEN_BUCKET).expect("seed 7");
    let b = generate_sprite(GOLDEN_KEY, 8, GOLDEN_BUCKET).expect("seed 8");
    let again = generate_sprite(GOLDEN_KEY, 7, GOLDEN_BUCKET).expect("seed 7 again");
    assert_ne!(a.rgba(), b.rgba(), "different seed must change the bytes");
    assert_eq!(a, again, "same inputs must give identical bytes");
}

#[test]
fn different_key_gives_different_bytes() {
    let a = generate_sprite("hex-forest", GOLDEN_SEED, GOLDEN_BUCKET).expect("forest");
    let b = generate_sprite("hex-river", GOLDEN_SEED, GOLDEN_BUCKET).expect("river");
    assert_ne!(a.rgba(), b.rgba(), "different key must change the bytes");
}

#[test]
fn disc_is_opaque_inside_and_transparent_black_outside() {
    let s = generate_sprite(GOLDEN_KEY, GOLDEN_SEED, GOLDEN_BUCKET).expect("golden bucket");
    let n = s.side() as usize;
    let mut opaque = 0usize;
    for y in 0..n {
        for x in 0..n {
            let [r, g, b, a] = pixel(&s, x, y);
            match a {
                255 => opaque += 1,
                0 => assert_eq!(
                    [r, g, b],
                    [0, 0, 0],
                    "transparent pixel ({x},{y}) has colour"
                ),
                other => panic!("alpha {other} at ({x},{y}) is neither 0 nor 255"),
            }
        }
    }
    assert!(opaque > 0 && opaque < n * n, "disc must be a proper subset");
}

#[test]
fn generate_sprite_propagates_a_bucket_error() {
    assert_eq!(
        generate_sprite(GOLDEN_KEY, GOLDEN_SEED, MAX_SPRITE_BUCKET + 1),
        Err(MapError::OutOfRange("bucket"))
    );
}

#[test]
fn atlas_starts_empty_and_counts_each_generation_once() {
    let mut atlas = SpriteAtlas::new();
    assert!(atlas.is_empty());
    assert_eq!(atlas.generations(), 0);

    let first = atlas
        .sprite(GOLDEN_KEY, GOLDEN_SEED, GOLDEN_BUCKET)
        .expect("miss")
        .clone();
    assert_eq!(atlas.generations(), 1);
    assert_eq!(atlas.len(), 1);
    assert!(!atlas.is_empty());

    // Same key, seed and bucket: a cache hit must not paint again.
    let hit = atlas
        .sprite(GOLDEN_KEY, GOLDEN_SEED, GOLDEN_BUCKET)
        .expect("hit")
        .clone();
    assert_eq!(atlas.generations(), 1, "cache hit regenerated");
    assert_eq!(atlas.len(), 1);
    assert_eq!(first, hit);
    assert_eq!(
        first,
        generate_sprite(GOLDEN_KEY, GOLDEN_SEED, GOLDEN_BUCKET).expect("direct")
    );
}

#[test]
fn atlas_keeps_one_entry_per_key_seed_and_bucket() {
    let mut atlas = SpriteAtlas::default();
    let side_small = atlas
        .sprite(GOLDEN_KEY, GOLDEN_SEED, 13)
        .expect("13")
        .side();
    let side_large = atlas
        .sprite(GOLDEN_KEY, GOLDEN_SEED, 21)
        .expect("21")
        .side();
    assert_eq!((side_small, side_large), (19, 76));
    assert_eq!(atlas.generations(), 2);
    atlas.sprite(GOLDEN_KEY, 9, 13).expect("new seed");
    assert_eq!(atlas.generations(), 3, "new seed is a miss");
    assert_eq!(atlas.len(), 3);
}

#[test]
fn atlas_error_leaves_the_cache_and_counter_untouched() {
    let mut atlas = SpriteAtlas::new();
    let err = atlas.sprite(GOLDEN_KEY, GOLDEN_SEED, MAX_SPRITE_BUCKET + 1);
    assert_eq!(err.err(), Some(MapError::OutOfRange("bucket")));
    assert_eq!(atlas.generations(), 0);
    assert!(atlas.is_empty());
}

#[test]
fn sprite_accessors_report_side_and_byte_length() {
    let s = generate_sprite(GOLDEN_KEY, GOLDEN_SEED, 0).expect("bucket 0");
    assert_eq!(s.side(), 2);
    assert_eq!(s.rgba().len(), 16);
}

#[test]
fn every_bucket_side_equals_the_documented_formula() {
    for bucket in 0..=MAX_SPRITE_BUCKET {
        let want = (2.0 * (f64::from(bucket) / 4.0).exp2()).round();
        let got = side_for_bucket(bucket).expect("bucket in range");
        assert_eq!(f64::from(got), want, "bucket {bucket}");
    }
    assert_eq!(side_for_bucket(22), Ok(91));
}
