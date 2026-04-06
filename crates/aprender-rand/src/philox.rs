//! Philox 4x32-10 counter-based RNG.
//!
//! # Contract: rand-philox-v1.yaml
//!
//! Philox is a counter-based PRNG from Salmon et al. (SC 2011).
//! It uses 10 rounds of multiplication and XOR to produce 4 u32s
//! per counter value.
//!
//! ## Properties
//! - Deterministic: same (key, counter) → same output
//! - Parallel-safe: each counter value is independent
//! - Statistical quality: passes BigCrush test suite
//! - Period: 2^128 per key

/// Philox 4x32-10 counter-based RNG.
///
/// Each call to `generate` produces 4 u32 values from a 128-bit counter
/// and 64-bit key. Counter-based design means any output can be computed
/// independently — ideal for GPU parallelism.
#[derive(Debug, Clone)]
pub struct Philox4x32 {
    key: [u32; 2],
    counter: [u32; 4],
}

// Philox round constants (from the original paper)
const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9; // golden ratio
const PHILOX_W1: u32 = 0xBB67_AE85; // sqrt(3)-1

impl Philox4x32 {
    /// Create a new Philox RNG with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { key: [seed as u32, (seed >> 32) as u32], counter: [0, 0, 0, 0] }
    }

    /// Create with explicit key and counter (for reproducible parallel generation).
    pub fn with_key_counter(key: [u32; 2], counter: [u32; 4]) -> Self {
        Self { key, counter }
    }

    /// Advance the counter by 1 and generate 4 u32 values.
    pub fn next_4u32(&mut self) -> [u32; 4] {
        let result = philox4x32_10(self.counter, self.key);
        self.increment_counter();
        result
    }

    /// Generate a single u32.
    pub fn next_u32(&mut self) -> u32 {
        self.next_4u32()[0]
    }

    /// Generate a f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        // Use 23 bits of mantissa for uniform [0, 1)
        (self.next_u32() >> 9) as f32 * (1.0 / (1u32 << 23) as f32)
    }

    /// Fill a buffer with uniform f32 values in [0, 1).
    pub fn fill_uniform(&mut self, output: &mut [f32]) {
        let mut i = 0;
        while i < output.len() {
            let vals = self.next_4u32();
            for v in &vals {
                if i >= output.len() {
                    break;
                }
                output[i] = (*v >> 9) as f32 * (1.0 / (1u32 << 23) as f32);
                i += 1;
            }
        }
    }

    /// Fill a buffer with normal(0,1) values using Box-Muller transform.
    pub fn fill_normal(&mut self, output: &mut [f32]) {
        let mut i = 0;
        while i < output.len() {
            let u1 = self.next_f32().max(f32::MIN_POSITIVE); // Avoid log(0)
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
    pub fn generate_at(key: [u32; 2], counter: [u32; 4]) -> [u32; 4] {
        philox4x32_10(counter, key)
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

/// One round of Philox: multiply, XOR, and key-add.
#[inline]
fn philox4x32_round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let lo0 = (u64::from(PHILOX_M0) * u64::from(ctr[0])) as u32;
    let hi0 = ((u64::from(PHILOX_M0) * u64::from(ctr[0])) >> 32) as u32;
    let lo1 = (u64::from(PHILOX_M1) * u64::from(ctr[2])) as u32;
    let hi1 = ((u64::from(PHILOX_M1) * u64::from(ctr[2])) >> 32) as u32;

    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

/// Full Philox 4x32-10: 10 rounds with key schedule.
fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut ctr = counter;
    let mut k = key;

    ctr = philox4x32_round(ctr, k);
    k[0] = k[0].wrapping_add(PHILOX_W0);
    k[1] = k[1].wrapping_add(PHILOX_W1);

    ctr = philox4x32_round(ctr, k);
    k[0] = k[0].wrapping_add(PHILOX_W0);
    k[1] = k[1].wrapping_add(PHILOX_W1);

    ctr = philox4x32_round(ctr, k);
    k[0] = k[0].wrapping_add(PHILOX_W0);
    k[1] = k[1].wrapping_add(PHILOX_W1);

    ctr = philox4x32_round(ctr, k);
    k[0] = k[0].wrapping_add(PHILOX_W0);
    k[1] = k[1].wrapping_add(PHILOX_W1);

    ctr = philox4x32_round(ctr, k);
    k[0] = k[0].wrapping_add(PHILOX_W0);
    k[1] = k[1].wrapping_add(PHILOX_W1);

    ctr = philox4x32_round(ctr, k);
    k[0] = k[0].wrapping_add(PHILOX_W0);
    k[1] = k[1].wrapping_add(PHILOX_W1);

    ctr = philox4x32_round(ctr, k);
    k[0] = k[0].wrapping_add(PHILOX_W0);
    k[1] = k[1].wrapping_add(PHILOX_W1);

    ctr = philox4x32_round(ctr, k);
    k[0] = k[0].wrapping_add(PHILOX_W0);
    k[1] = k[1].wrapping_add(PHILOX_W1);

    ctr = philox4x32_round(ctr, k);
    k[0] = k[0].wrapping_add(PHILOX_W0);
    k[1] = k[1].wrapping_add(PHILOX_W1);

    philox4x32_round(ctr, k)
}
