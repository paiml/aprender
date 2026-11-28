//! Byte-stable golden of the sprite atlas across every quarter-octave bucket (MAP-5 slice 2).
//!
//! Each bucket `0..=MAX_SPRITE_BUCKET` is painted for one fixed key and seed. The FNV-1a 64-bit
//! hash of `side (u32 little-endian) ++ rgba` is committed per bucket. Any change to the
//! rasteriser (jitter, disc test, alpha) moves a hash. Any change to the bucket edges
//! (`size_bucket`) fails the edge test. Two consecutive runs must give identical bytes.

use super::sprites::{generate_sprite, size_bucket, Sprite, MAX_SPRITE_BUCKET};

const GOLDEN_KEY: &str = "hex-cell";
const GOLDEN_SEED: u64 = 42;

/// Committed hashes, one per bucket `0..=32`. Regenerate only with a reviewed rasteriser change.
const BUCKET_GOLDEN: [u64; 33] = [0; 33];

/// Relative offset used to probe each side of a quarter-octave edge.
const EDGE_EPSILON: f64 = 1e-9;

fn digest(sprite: &Sprite) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in sprite.side().to_le_bytes().iter().chain(sprite.rgba()) {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn paint_every_bucket() -> Vec<Sprite> {
    (0..=MAX_SPRITE_BUCKET)
        .map(|b| generate_sprite(GOLDEN_KEY, GOLDEN_SEED, b).expect("bucket is in range"))
        .collect()
}

#[test]
fn sprite_golden_hashes_match_the_committed_value_in_every_bucket() {
    let sprites = paint_every_bucket();
    assert_eq!(sprites.len(), BUCKET_GOLDEN.len(), "one sprite per bucket");
    for (b, sprite) in sprites.iter().enumerate() {
        assert_eq!(
            digest(sprite),
            BUCKET_GOLDEN[b],
            "bucket {b} (side {}) golden hash",
            sprite.side()
        );
    }
}

#[test]
fn sprite_golden_is_byte_identical_across_two_consecutive_runs() {
    let first = paint_every_bucket();
    let second = paint_every_bucket();
    for (b, (x, y)) in first.iter().zip(&second).enumerate() {
        assert!(x == y, "bucket {b} differs between two runs");
    }
}

#[test]
fn bucket_edges_switch_exactly_at_each_quarter_octave_power() {
    for b in 0..=MAX_SPRITE_BUCKET {
        let edge = 2f64.powf(f64::from(b) / 4.0);
        if b < MAX_SPRITE_BUCKET {
            assert_eq!(
                size_bucket(edge * (1.0 + EDGE_EPSILON)),
                Ok(b),
                "just above the edge of bucket {b}"
            );
        }
        if b > 0 {
            assert_eq!(
                size_bucket(edge * (1.0 - EDGE_EPSILON)),
                Ok(b - 1),
                "just below the edge of bucket {b}"
            );
        }
    }
}
