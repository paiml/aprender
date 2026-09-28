//! A seeded, dependency-free generator (splitmix64), so every failure reproduces from the seed its
//! receipt records (KTEST-001 §3.3).

/// splitmix64 (Steele, Lea, Flood 2014). Not cryptographic; chosen because it is tiny, has no
/// dependencies, and gives the same stream on every host.
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1), with 53 random bits.
    #[allow(clippy::cast_precision_loss)] // 53-bit integers convert exactly
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in [lo, hi).
    #[allow(clippy::cast_possible_truncation)] // the point: an f32 input
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        (f64::from(lo) + self.unit() * (f64::from(hi) - f64::from(lo))) as f32
    }

    /// Standard normal (Box–Muller).
    #[allow(clippy::cast_possible_truncation)] // the point: an f32 input
    pub fn normal(&mut self) -> f32 {
        let u1 = 1.0 - self.unit(); // (0, 1]: ln is finite
        let u2 = self.unit();
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }

    /// Uniform in 0..n.
    ///
    /// # Panics
    /// If `n == 0`.
    #[allow(clippy::cast_possible_truncation)] // the remainder is below n, a usize
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "SplitMix64::below(0)");
        (self.next_u64() % n as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let (mut a, mut b) = (SplitMix64::new(7), SplitMix64::new(7));
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        assert_ne!(SplitMix64::new(7).next_u64(), SplitMix64::new(8).next_u64());
    }

    #[test]
    fn ranges_hold() {
        let mut r = SplitMix64::new(1);
        for _ in 0..10_000 {
            let u = r.unit();
            assert!((0.0..1.0).contains(&u));
            let x = r.uniform(-2.0, 3.0);
            assert!((-2.0..3.0).contains(&x));
            assert!(r.below(5) < 5);
            assert!(r.normal().is_finite());
        }
    }
}
