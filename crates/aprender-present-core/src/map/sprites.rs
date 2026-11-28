//! Zoom-bucketed sprite atlas from a seeded generator (APR-MAP-001 MAP-5).
//!
//! A sprite is a square RGBA8 buffer, painted once per `(key, seed, size bucket)` and then
//! served from a cache. The size bucket is quarter-octave: `bucket = floor(4 * log2(r))` for a
//! radius `r` in screen pixels. Pixels are integer-only, drawn from a splitmix64 stream seeded
//! by the key, the seed and the bucket, so the bytes are identical on every platform.

use std::collections::btree_map::Entry;
use std::collections::BTreeMap;

use super::{check_scale, MapError};

/// Smallest radius, in screen pixels, that has a sprite. Below it the bucket would be negative.
pub const MIN_SPRITE_RADIUS_PX: f64 = 1.0;
/// Largest radius, in screen pixels, that has a sprite. Bounds every allocation.
pub const MAX_SPRITE_RADIUS_PX: f64 = 256.0;
/// Largest size bucket: `floor(4 * log2(MAX_SPRITE_RADIUS_PX))`.
pub const MAX_SPRITE_BUCKET: u32 = 32;
/// Largest sprite side in pixels, reached at `MAX_SPRITE_BUCKET`. Bounds each sprite to 1 MiB.
pub const MAX_SPRITE_SIDE: u32 = 512;

/// `2^(q/4)` for `q` in `0..4`, in thousandths, for the integer side computation.
const QUARTER_OCTAVE_MILLI: [u64; 4] = [1000, 1189, 1414, 1682];

/// Quarter-octave size bucket for a radius in screen pixels: `floor(4 * log2(r))`.
///
/// Rejects non-finite and non-positive radii with `NonFinite` / `NonPositive`, and radii outside
/// `[MIN_SPRITE_RADIUS_PX, MAX_SPRITE_RADIUS_PX]` with `OutOfRange`.
pub fn size_bucket(radius_px: f64) -> Result<u32, MapError> {
    let r = check_scale(radius_px, "radius")?;
    if !(MIN_SPRITE_RADIUS_PX..=MAX_SPRITE_RADIUS_PX).contains(&r) {
        return Err(MapError::OutOfRange("radius"));
    }
    // r is in [1, 256], so 4 * log2(r) is in [0, 32] and the cast is exact for the floor.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let bucket = (4.0 * r.log2()).floor() as u32;
    Ok(bucket)
}

/// Side length in pixels of the sprite for a size bucket: `round(2 * 2^(bucket/4))`, in integers.
///
/// Rejects a bucket above `MAX_SPRITE_BUCKET` with `OutOfRange`.
pub fn side_for_bucket(bucket: u32) -> Result<u32, MapError> {
    if bucket > MAX_SPRITE_BUCKET {
        return Err(MapError::OutOfRange("bucket"));
    }
    let k = bucket / 4;
    let q = (bucket % 4) as usize;
    // 2 * 2^(q/4) * 2^k, in thousandths, then rounded half-up to whole pixels. At the largest
    // bucket this is 862, so the u64 product and the u32 result cannot overflow.
    let milli = 2 * QUARTER_OCTAVE_MILLI[q] * (1u64 << k);
    Ok(((milli + 500) / 1000) as u32)
}

/// FNV-1a 64-bit hash. Used to turn a key string into a stable integer.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// splitmix64 pseudo-random stream. Integer-only, so the output is the same on every target.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// A square RGBA8 sprite, `side * side * 4` bytes, row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sprite {
    side: u32,
    rgba: Vec<u8>,
}

impl Sprite {
    /// Side length in pixels.
    pub fn side(&self) -> u32 {
        self.side
    }

    /// Row-major RGBA8 bytes, `side * side * 4` long.
    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }
}

/// Paints one sprite for a key, seed and size bucket. Pure: the same inputs give the same bytes.
///
/// Pixels inside the disc of diameter `side` carry a key-coloured base with seeded jitter and
/// full alpha; pixels outside are transparent black.
pub fn generate_sprite(key: &str, seed: u64, bucket: u32) -> Result<Sprite, MapError> {
    let side = side_for_bucket(bucket)?;
    let key_hash = fnv1a(key.as_bytes());
    let base = [
        (key_hash & 0xff) as i32,
        ((key_hash >> 8) & 0xff) as i32,
        ((key_hash >> 16) & 0xff) as i32,
    ];
    let mut rng = SplitMix64 {
        state: key_hash
            ^ seed.rotate_left(17)
            ^ u64::from(bucket).wrapping_mul(0x9E37_79B9_7F4A_7C15),
    };
    let n = side as usize;
    let mut rgba = vec![0u8; n * n * 4];
    let s = i64::from(side);
    for y in 0..n {
        for x in 0..n {
            // Doubled offsets from the centre, so the disc test stays in integers.
            let dx = 2 * x as i64 + 1 - s;
            let dy = 2 * y as i64 + 1 - s;
            if dx * dx + dy * dy > s * s {
                continue;
            }
            let at = (y * n + x) * 4;
            for c in 0..3 {
                let jitter = (rng.next_u64() % 31) as i32 - 15;
                // Clamped to 0..=255 first, so the cast is exact.
                rgba[at + c] = (base[c] + jitter).clamp(0, 255) as u8;
            }
            rgba[at + 3] = 255;
        }
    }
    Ok(Sprite { side, rgba })
}

/// Cache of sprites keyed by `(key, seed, bucket)`. A sprite is generated only on a miss.
#[derive(Debug, Default)]
pub struct SpriteAtlas {
    cache: BTreeMap<(String, u64, u32), Sprite>,
    generations: u64,
}

impl SpriteAtlas {
    /// Empty atlas with no generations counted.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sprite for `(key, seed, bucket)`. Generates and stores it on a miss, otherwise returns the
    /// cached sprite without regenerating it.
    pub fn sprite(&mut self, key: &str, seed: u64, bucket: u32) -> Result<&Sprite, MapError> {
        match self.cache.entry((key.to_owned(), seed, bucket)) {
            Entry::Occupied(hit) => Ok(hit.into_mut()),
            Entry::Vacant(slot) => {
                let sprite = generate_sprite(key, seed, bucket)?;
                self.generations += 1;
                Ok(slot.insert(sprite))
            }
        }
    }

    /// Number of sprites painted since the atlas was made (cache misses).
    pub fn generations(&self) -> u64 {
        self.generations
    }

    /// Number of sprites currently cached.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether no sprite is cached.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

#[cfg(test)]
#[path = "sprites_tests.rs"]
mod tests;
