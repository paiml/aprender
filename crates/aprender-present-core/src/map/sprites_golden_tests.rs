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
const BUCKET_GOLDEN: [u64; 33] = [
    0x7a1bac4f24f9e885,
    0x65396d674b024abe,
    0x5e7af89e39708817,
    0xaa72ac56c51a864b,
    0xbfc3e025cfef15df,
    0x2d6f4935ced35f84,
    0x63f9d46bed6c5305,
    0xfebba9b54f3691e0,
    0x3bd43420032fe54b,
    0x8aab6178d242e851,
    0x1bba2e4cd01f53bf,
    0x3a3d5a376ce92e52,
    0xb512316297181deb,
    0xee0f9bf4fe7d6f8c,
    0x2e8e47fdbb5fbff5,
    0x9cb97ef5961f3be4,
    0x467ba8b143e19dff,
    0x026acd9be5ae12f1,
    0x5897b67310d9acf5,
    0x961cce19b6d85c4e,
    0x004250561346e736,
    0x48571065c59d4635,
    0x342387f38b111848,
    0x11112a4a9583a94a,
    0xdd0dad96891e22a1,
    0x66012925a90d5305,
    0x008a0bbdf91dafc3,
    0x1f2edd27efef08f0,
    0x3025e613f94cb7f4,
    0x95718e2e3a639e58,
    0x226b692128022514,
    0xad2997af55d88beb,
    0xf0fe6ee964044f2c,
];

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
