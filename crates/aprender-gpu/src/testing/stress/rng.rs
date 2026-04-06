//! Simple PCG32 RNG for stress testing.
//!
//! Used when the `simular` feature is not enabled.

/// Simple PCG32 RNG for stress testing (no external deps in core)
/// Used when simular feature is not enabled
#[derive(Debug, Clone)]
pub struct StressRng {
    state: u64,
    inc: u64,
}

impl StressRng {
    /// Create new RNG with seed
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (seed << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// Generate next u32
    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.inc);
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        (xorshifted >> rot) | (xorshifted << ((!rot).wrapping_add(1) & 31))
    }

    /// Generate next u64
    pub fn next_u64(&mut self) -> u64 {
        let high = self.next_u32() as u64;
        let low = self.next_u32() as u64;
        (high << 32) | low
    }

    /// Generate f32 in [0, 1)
    pub fn gen_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / u32::MAX as f64) as f32
    }

    /// Generate u32 in range [min, max)
    pub fn gen_range_u32(&mut self, min: u32, max: u32) -> u32 {
        if max <= min {
            return min;
        }
        let range = max - min;
        min + (self.next_u32() % range)
    }
}
