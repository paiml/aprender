//! Aggregation oracle (MAP-8 slice 1): after a seeded sequence of deltas the client's incremental
//! histogram equals a brute-force rebuild over all 10,530 cells. The oracle is written here, not
//! taken from `Histogram::of`, so the two are independent.

use super::*;
use crate::map::histogram::{Histogram, HISTOGRAM_BINS};

/// xorshift64*: a fixed seed gives the same delta sequence on every run.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Bin counts of the cells with data over `[lo, hi]`, rebuilt from scratch over every cell.
fn oracle(values: &[Option<f64>], lo: f64, hi: f64, bins: usize) -> Vec<u32> {
    let mut counts = vec![0u32; bins];
    for &value in values {
        let Some(v) = value else { continue };
        let t = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
        let bin = ((t * bins as f64).floor() as usize).min(bins - 1);
        counts[bin] += 1;
    }
    counts
}

#[test]
fn incremental_histogram_equals_a_from_scratch_rebuild_after_sixty_seeded_deltas() {
    let mut c = client(Some(1));
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for seq in 1..=60u64 {
        let entries = 1 + rng.below(50);
        let cells: Vec<(u32, Option<f64>)> = (0..entries)
            .map(|_| {
                let cell = u32::try_from(rng.below(u64::from(CELLS))).expect("fits a cell id");
                let value = (rng.below(4) != 0).then(|| rng.unit() * 120.0 - 10.0);
                (cell, value)
            })
            .collect();
        let rep = c.apply(&delta(1, seq, &cells)).expect("applies");
        assert_eq!(rep.renders, r(0, 1, 0), "delta {seq}");
        let want = oracle(&values(&c, 1), 0.0, 100.0, HISTOGRAM_BINS);
        assert_eq!(
            c.histogram(1).expect("histogram").counts(),
            want.as_slice(),
            "after delta {seq}"
        );
    }
    let scratch =
        Histogram::of(c.lenses().get(1).expect("lens 1"), HISTOGRAM_BINS).expect("histogram");
    assert_eq!(c.histogram(1), Some(&scratch));
}
