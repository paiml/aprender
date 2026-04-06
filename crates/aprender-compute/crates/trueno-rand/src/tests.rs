//! RNG tests — statistical properties and determinism.

use crate::philox::Philox4x32;
use crate::threefry::Threefry4x64;

// ============================================================================
// Determinism tests
// ============================================================================

#[test]
fn test_deterministic_same_seed() {
    let mut rng1 = Philox4x32::new(42);
    let mut rng2 = Philox4x32::new(42);

    for _ in 0..100 {
        assert_eq!(rng1.next_4u32(), rng2.next_4u32());
    }
}

#[test]
fn test_different_seeds_differ() {
    let mut rng1 = Philox4x32::new(1);
    let mut rng2 = Philox4x32::new(2);
    assert_ne!(rng1.next_4u32(), rng2.next_4u32());
}

#[test]
fn test_stateless_generation() {
    let key = [42u32, 0];
    let counter = [7, 0, 0, 0];
    let result1 = Philox4x32::generate_at(key, counter);
    let result2 = Philox4x32::generate_at(key, counter);
    assert_eq!(result1, result2);
}

#[test]
fn test_different_counters_differ() {
    let key = [42u32, 0];
    let c1 = [0, 0, 0, 0];
    let c2 = [1, 0, 0, 0];
    assert_ne!(Philox4x32::generate_at(key, c1), Philox4x32::generate_at(key, c2));
}

// ============================================================================
// Uniform distribution tests
// ============================================================================

#[test]
fn test_uniform_range() {
    let mut rng = Philox4x32::new(123);
    let mut buf = vec![0.0f32; 10000];
    rng.fill_uniform(&mut buf);

    for &v in &buf {
        assert!((0.0..1.0).contains(&v), "Uniform value out of range: {v}");
    }
}

#[test]
fn test_uniform_mean() {
    let mut rng = Philox4x32::new(456);
    let mut buf = vec![0.0f32; 100_000];
    rng.fill_uniform(&mut buf);

    let mean: f32 = buf.iter().sum::<f32>() / buf.len() as f32;
    assert!((mean - 0.5).abs() < 0.01, "Uniform mean {mean} too far from 0.5");
}

#[test]
fn test_uniform_variance() {
    let mut rng = Philox4x32::new(789);
    let mut buf = vec![0.0f32; 100_000];
    rng.fill_uniform(&mut buf);

    let mean: f64 = buf.iter().map(|&v| f64::from(v)).sum::<f64>() / buf.len() as f64;
    let var: f64 = buf
        .iter()
        .map(|&v| {
            let d = f64::from(v) - mean;
            d * d
        })
        .sum::<f64>()
        / buf.len() as f64;

    // Uniform[0,1) variance = 1/12 ≈ 0.0833
    assert!((var - 1.0 / 12.0).abs() < 0.005, "Uniform variance {var} too far from 1/12");
}

// ============================================================================
// Normal distribution tests
// ============================================================================

#[test]
fn test_normal_mean_and_variance() {
    let mut rng = Philox4x32::new(101);
    let mut buf = vec![0.0f32; 100_000];
    rng.fill_normal(&mut buf);

    let mean: f64 = buf.iter().map(|&v| f64::from(v)).sum::<f64>() / buf.len() as f64;
    let var: f64 = buf
        .iter()
        .map(|&v| {
            let d = f64::from(v) - mean;
            d * d
        })
        .sum::<f64>()
        / buf.len() as f64;

    assert!(mean.abs() < 0.02, "Normal mean {mean} too far from 0");
    assert!((var - 1.0).abs() < 0.05, "Normal variance {var} too far from 1.0");
}

#[test]
fn test_normal_no_nan_or_inf() {
    let mut rng = Philox4x32::new(202);
    let mut buf = vec![0.0f32; 10000];
    rng.fill_normal(&mut buf);

    for &v in &buf {
        assert!(v.is_finite(), "Normal produced non-finite: {v}");
    }
}

// ============================================================================
// Counter overflow tests
// ============================================================================

#[test]
fn test_counter_wraparound() {
    let mut rng = Philox4x32::with_key_counter([0, 0], [u32::MAX, 0, 0, 0]);
    let v1 = rng.next_4u32();
    let v2 = rng.next_4u32(); // Should wrap counter[0] and increment counter[1]
    assert_ne!(v1, v2);
}

// ============================================================================
// Chi-squared uniformity test (bucket test)
// ============================================================================

#[test]
fn test_chi_squared_uniformity() {
    let mut rng = Philox4x32::new(303);
    let n = 100_000;
    let buckets = 100;
    let mut counts = vec![0u32; buckets];

    for _ in 0..n {
        let v = rng.next_f32();
        let idx = (v * buckets as f32) as usize;
        let idx = idx.min(buckets - 1);
        counts[idx] += 1;
    }

    let expected = n as f64 / buckets as f64;
    let chi_sq: f64 = counts
        .iter()
        .map(|&c| {
            let diff = f64::from(c) - expected;
            diff * diff / expected
        })
        .sum();

    // Chi-squared critical value for 99 df, p=0.001 ≈ 148.2
    assert!(chi_sq < 150.0, "Chi-squared {chi_sq} exceeds threshold (poor uniformity)");
}

// ============================================================================
// Property-based tests
// ============================================================================

// ============================================================================
// Threefry 4x64-20 tests
// ============================================================================

#[test]
fn test_threefry_deterministic_same_seed() {
    let mut rng1 = Threefry4x64::new(42);
    let mut rng2 = Threefry4x64::new(42);

    for _ in 0..100 {
        assert_eq!(rng1.next_4u64(), rng2.next_4u64());
    }
}

#[test]
fn test_threefry_different_seeds_differ() {
    let mut rng1 = Threefry4x64::new(1);
    let mut rng2 = Threefry4x64::new(2);
    assert_ne!(rng1.next_4u64(), rng2.next_4u64());
}

#[test]
fn test_threefry_stateless_generation() {
    let key = [42u64, 0, 0, 0];
    let counter = [7, 0, 0, 0];
    let result1 = Threefry4x64::generate_at(key, counter);
    let result2 = Threefry4x64::generate_at(key, counter);
    assert_eq!(result1, result2);
}

#[test]
fn test_threefry_different_counters_differ() {
    let key = [42u64, 0, 0, 0];
    let c1 = [0, 0, 0, 0];
    let c2 = [1, 0, 0, 0];
    assert_ne!(Threefry4x64::generate_at(key, c1), Threefry4x64::generate_at(key, c2));
}

#[test]
fn test_threefry_uniform_range() {
    let mut rng = Threefry4x64::new(123);
    let mut buf = vec![0.0f32; 10000];
    rng.fill_uniform(&mut buf);

    for &v in &buf {
        assert!((0.0..1.0).contains(&v), "Threefry uniform out of range: {v}");
    }
}

#[test]
fn test_threefry_uniform_mean() {
    let mut rng = Threefry4x64::new(456);
    let mut buf = vec![0.0f32; 100_000];
    rng.fill_uniform(&mut buf);

    let mean: f32 = buf.iter().sum::<f32>() / buf.len() as f32;
    assert!((mean - 0.5).abs() < 0.01, "Threefry uniform mean {mean} too far from 0.5");
}

#[test]
fn test_threefry_normal_mean_and_variance() {
    let mut rng = Threefry4x64::new(101);
    let mut buf = vec![0.0f32; 100_000];
    rng.fill_normal(&mut buf);

    let mean: f64 = buf.iter().map(|&v| f64::from(v)).sum::<f64>() / buf.len() as f64;
    let var: f64 = buf
        .iter()
        .map(|&v| {
            let d = f64::from(v) - mean;
            d * d
        })
        .sum::<f64>()
        / buf.len() as f64;

    assert!(mean.abs() < 0.02, "Threefry normal mean {mean} too far from 0");
    assert!((var - 1.0).abs() < 0.05, "Threefry normal variance {var} too far from 1.0");
}

#[test]
fn test_threefry_counter_wraparound() {
    let mut rng = Threefry4x64::with_key_counter([0, 0, 0, 0], [u64::MAX, 0, 0, 0]);
    let v1 = rng.next_4u64();
    let v2 = rng.next_4u64();
    assert_ne!(v1, v2);
}

#[test]
fn test_threefry_normal_no_nan_or_inf() {
    let mut rng = Threefry4x64::new(202);
    let mut buf = vec![0.0f32; 10000];
    rng.fill_normal(&mut buf);

    for &v in &buf {
        assert!(v.is_finite(), "Threefry normal produced non-finite: {v}");
    }
}

// ============================================================================
// Rng trait tests (Contract: rand-philox-v1.yaml §6B-C)
// ============================================================================

use crate::Rng;

#[test]
fn test_rng_trait_philox_uniform() {
    let mut rng = Philox4x32::new(999);
    let rng_trait: &mut dyn Rng = &mut rng;
    let mut buf = vec![0.0f32; 1000];
    rng_trait.fill_uniform(&mut buf);
    for &v in &buf {
        assert!((0.0..1.0).contains(&v), "Rng trait uniform out of range: {v}");
    }
}

#[test]
fn test_rng_trait_philox_normal() {
    let mut rng = Philox4x32::new(888);
    let rng_trait: &mut dyn Rng = &mut rng;
    let mut buf = vec![0.0f32; 1000];
    rng_trait.fill_normal(&mut buf);
    for &v in &buf {
        assert!(v.is_finite(), "Rng trait normal non-finite: {v}");
    }
}

#[test]
fn test_rng_trait_threefry_uniform() {
    let mut rng = Threefry4x64::new(777);
    let rng_trait: &mut dyn Rng = &mut rng;
    let mut buf = vec![0.0f32; 1000];
    rng_trait.fill_uniform(&mut buf);
    for &v in &buf {
        assert!((0.0..1.0).contains(&v), "Rng trait threefry uniform out of range: {v}");
    }
}

#[test]
fn test_rng_trait_threefry_normal() {
    let mut rng = Threefry4x64::new(666);
    let rng_trait: &mut dyn Rng = &mut rng;
    let mut buf = vec![0.0f32; 1000];
    rng_trait.fill_normal(&mut buf);
    for &v in &buf {
        assert!(v.is_finite(), "Rng trait threefry normal non-finite: {v}");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_deterministic(seed: u64) {
            let mut rng1 = Philox4x32::new(seed);
            let mut rng2 = Philox4x32::new(seed);
            prop_assert_eq!(rng1.next_4u32(), rng2.next_4u32());
        }

        #[test]
        fn prop_uniform_in_range(seed: u64) {
            let mut rng = Philox4x32::new(seed);
            for _ in 0..100 {
                let v = rng.next_f32();
                prop_assert!((0.0..1.0).contains(&v), "out of range: {v}");
            }
        }

        #[test]
        fn prop_threefry_deterministic(seed: u64) {
            let mut rng1 = Threefry4x64::new(seed);
            let mut rng2 = Threefry4x64::new(seed);
            prop_assert_eq!(rng1.next_4u64(), rng2.next_4u64());
        }

        #[test]
        fn prop_threefry_uniform_in_range(seed: u64) {
            let mut rng = Threefry4x64::new(seed);
            for _ in 0..100 {
                let v = rng.next_f32();
                prop_assert!((0.0..1.0).contains(&v), "out of range: {v}");
            }
        }
    }
}

// ============================================================================
// POPPERIAN FALSIFICATION: Adversarial edge cases (GH #152)
// ============================================================================

#[test]
fn test_falsify_fill_uniform_empty_buffer() {
    let mut rng = Philox4x32::new(0);
    let mut buf: Vec<f32> = vec![];
    rng.fill_uniform(&mut buf); // should not panic
    assert!(buf.is_empty());
}

#[test]
fn test_falsify_fill_normal_empty_buffer() {
    let mut rng = Philox4x32::new(0);
    let mut buf: Vec<f32> = vec![];
    rng.fill_normal(&mut buf); // should not panic
    assert!(buf.is_empty());
}

#[test]
fn test_falsify_fill_uniform_single_element() {
    let mut rng = Philox4x32::new(42);
    let mut buf = vec![0.0_f32; 1];
    rng.fill_uniform(&mut buf);
    assert!((0.0..1.0).contains(&buf[0]), "Single element out of range: {}", buf[0]);
}

#[test]
fn test_falsify_fill_normal_single_element() {
    let mut rng = Philox4x32::new(42);
    let mut buf = vec![0.0_f32; 1];
    rng.fill_normal(&mut buf);
    assert!(buf[0].is_finite(), "Single element non-finite: {}", buf[0]);
}

#[test]
fn test_falsify_sequential_calls_independent() {
    // Two sequential fill_uniform calls should produce different values
    let mut rng = Philox4x32::new(99);
    let mut buf1 = vec![0.0_f32; 100];
    let mut buf2 = vec![0.0_f32; 100];
    rng.fill_uniform(&mut buf1);
    rng.fill_uniform(&mut buf2);
    // Very unlikely to be identical
    let same = buf1.iter().zip(buf2.iter()).filter(|(&a, &b)| (a - b).abs() < 1e-10).count();
    assert!(same < 5, "Sequential fills too similar: {same}/100 identical");
}

#[test]
fn test_falsify_seed_zero() {
    // Seed 0 should still produce valid output (not degenerate)
    let mut rng = Philox4x32::new(0);
    let mut buf = vec![0.0_f32; 1000];
    rng.fill_uniform(&mut buf);
    let mean: f32 = buf.iter().sum::<f32>() / buf.len() as f32;
    assert!((mean - 0.5).abs() < 0.1, "Seed 0 mean: {mean}");
    // Not all same value
    let first = buf[0];
    let all_same = buf.iter().all(|&v| (v - first).abs() < 1e-10);
    assert!(!all_same, "Seed 0 produced degenerate constant output");
}

#[test]
fn test_falsify_threefry_empty_buffers() {
    let mut rng = Threefry4x64::new(0);
    let mut buf: Vec<f32> = vec![];
    rng.fill_uniform(&mut buf);
    assert!(buf.is_empty());
    rng.fill_normal(&mut buf);
    assert!(buf.is_empty());
}

#[test]
fn test_falsify_kolmogorov_smirnov_uniform() {
    // KS test: max deviation of empirical CDF from theoretical U(0,1)
    let mut rng = Philox4x32::new(404);
    let n = 10_000;
    let mut buf = vec![0.0_f32; n];
    rng.fill_uniform(&mut buf);
    buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut max_d = 0.0_f64;
    for (i, &v) in buf.iter().enumerate() {
        let empirical = (i + 1) as f64 / n as f64;
        let theoretical = f64::from(v);
        let d = (empirical - theoretical).abs();
        if d > max_d {
            max_d = d;
        }
    }
    // KS critical value at α=0.01 for n=10000 ≈ 1.63/sqrt(n) ≈ 0.0163
    assert!(max_d < 0.02, "KS statistic {max_d} exceeds threshold (distribution not uniform)");
}

#[test]
fn test_falsify_normal_no_extreme_outliers() {
    // For N(0,1), values beyond 6σ should be extremely rare (< 1 in 500M)
    let mut rng = Philox4x32::new(505);
    let mut buf = vec![0.0_f32; 100_000];
    rng.fill_normal(&mut buf);
    let outliers = buf.iter().filter(|&&v| v.abs() > 6.0).count();
    assert!(outliers == 0, "Got {outliers} values beyond 6σ in 100K samples");
}
