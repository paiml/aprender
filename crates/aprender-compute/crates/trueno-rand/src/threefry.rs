//! Threefry 4x64-20 counter-based RNG.
//!
//! # Contract: rand-threefry-v1.yaml
//!
//! Threefry is a counter-based PRNG from Salmon et al. (SC 2011).
//! It uses 20 rounds of rotation and XOR to produce 4 u64s per counter value.
//! Designed for hardware without fast 64-bit multiply (backup to Philox).
//!
//! ## Properties
//! - Deterministic: same (key, counter) → same output
//! - Parallel-safe: each counter value is independent
//! - Period: 2^256 per key
//! - No multiplication — uses only add, rotate, XOR

/// Threefry 4x64-20 counter-based RNG.
///
/// Counter-based PRNG using 20 rounds of mix operations (rotation + XOR).
/// Each call to `generate` produces 4 u64 values from a 256-bit counter
/// and 256-bit key.
#[derive(Debug, Clone)]
pub struct Threefry4x64 {
    key: [u64; 4],
    counter: [u64; 4],
}

// Rotation constants for Threefry 4x64 (from the original paper, Table 4)
const R_4X64: [[u32; 2]; 8] =
    [[14, 16], [52, 57], [23, 40], [5, 37], [25, 33], [46, 12], [58, 22], [32, 32]];

// Skein key schedule constant (2^64 / golden ratio)
const KS_PARITY: u64 = 0x1BD1_1BDA_A9FC_1A22;

impl Threefry4x64 {
    /// Create a new Threefry RNG with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { key: [seed, seed.wrapping_mul(0x9E37_79B9_7F4A_7C15), 0, 0], counter: [0, 0, 0, 0] }
    }

    /// Create with explicit key and counter (for reproducible parallel generation).
    pub fn with_key_counter(key: [u64; 4], counter: [u64; 4]) -> Self {
        Self { key, counter }
    }

    /// Advance the counter by 1 and generate 4 u64 values.
    pub fn next_4u64(&mut self) -> [u64; 4] {
        let result = threefry4x64_20(self.counter, self.key);
        self.increment_counter();
        result
    }

    /// Generate a single u64.
    pub fn next_u64(&mut self) -> u64 {
        self.next_4u64()[0]
    }

    /// Generate a f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        // Use top 23 bits of first u64 output
        let v = self.next_u64();
        ((v >> 41) as u32) as f32 * (1.0 / (1u32 << 23) as f32)
    }

    /// Fill a buffer with uniform f32 values in [0, 1).
    pub fn fill_uniform(&mut self, output: &mut [f32]) {
        let mut i = 0;
        while i < output.len() {
            let vals = self.next_4u64();
            for v in &vals {
                if i >= output.len() {
                    break;
                }
                output[i] = ((*v >> 41) as u32) as f32 * (1.0 / (1u32 << 23) as f32);
                i += 1;
            }
        }
    }

    /// Fill a buffer with normal(0,1) values using Box-Muller transform.
    pub fn fill_normal(&mut self, output: &mut [f32]) {
        let mut i = 0;
        while i < output.len() {
            let u1 = self.next_f32().max(f32::MIN_POSITIVE);
            let u2 = self.next_f32();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;

            output[i] = r * theta.cos();
            i += 1;
            if i < output.len() {
                output[i] = r * theta.sin();
                i += 1;
            }
        }
    }

    /// Generate values for a specific counter (stateless, for GPU-style parallel generation).
    pub fn generate_at(key: [u64; 4], counter: [u64; 4]) -> [u64; 4] {
        threefry4x64_20(counter, key)
    }

    fn increment_counter(&mut self) {
        self.counter[0] = self.counter[0].wrapping_add(1);
        if self.counter[0] == 0 {
            self.counter[1] = self.counter[1].wrapping_add(1);
            if self.counter[1] == 0 {
                self.counter[2] = self.counter[2].wrapping_add(1);
                if self.counter[2] == 0 {
                    self.counter[3] = self.counter[3].wrapping_add(1);
                }
            }
        }
    }
}

/// Apply a single Threefry mix: v0 += v1; v1 = rotl(v1, r) ^ v0.
#[inline]
fn mix(v0: u64, v1: u64, r: u32) -> (u64, u64) {
    let v0 = v0.wrapping_add(v1);
    let v1 = v1.rotate_left(r) ^ v0;
    (v0, v1)
}

/// Full Threefry 4x64-20: 20 rounds with key injection every 4 rounds.
fn threefry4x64_20(counter: [u64; 4], key: [u64; 4]) -> [u64; 4] {
    // Expanded key schedule: ks[4] = KS_PARITY ^ ks[0] ^ ks[1] ^ ks[2] ^ ks[3]
    let ks4 = KS_PARITY ^ key[0] ^ key[1] ^ key[2] ^ key[3];

    let mut x = [
        counter[0].wrapping_add(key[0]),
        counter[1].wrapping_add(key[1]),
        counter[2].wrapping_add(key[2]),
        counter[3].wrapping_add(key[3]),
    ];

    // 20 rounds, key injection after rounds 4, 8, 12, 16, 20
    let ks = [key[0], key[1], key[2], key[3], ks4];
    for round in 0..20 {
        let rc = R_4X64[round % 8];
        if round % 2 == 0 {
            let (a, b) = mix(x[0], x[1], rc[0]);
            let (c, d) = mix(x[2], x[3], rc[1]);
            x = [a, b, c, d];
        } else {
            let (a, b) = mix(x[0], x[3], rc[0]);
            let (c, d) = mix(x[2], x[1], rc[1]);
            x = [a, d, c, b];
        }

        // Key injection every 4 rounds
        if (round + 1) % 4 == 0 {
            let s = (round + 1) / 4;
            x[0] = x[0].wrapping_add(ks[s % 5]);
            x[1] = x[1].wrapping_add(ks[(s + 1) % 5]);
            x[2] = x[2].wrapping_add(ks[(s + 2) % 5].wrapping_add(s as u64));
            x[3] = x[3].wrapping_add(ks[(s + 3) % 5].wrapping_add(s as u64));
        }
    }

    x
}
