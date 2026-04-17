//! Weight initialization utilities (C-INIT-001).
//!
//! Provides `rand_normal_seeded` for proper random normal initialization,
//! matching HuggingFace LLaMA's `normal(0, initializer_range)`.
//!
//! Replaces the legacy sinusoidal `sin(i * const) * scale` placeholder
//! that caused a 16x convergence gap vs PyTorch (entrenar#309).
//!
//! References:
//! - Touvron et al. (2023) LLaMA: arxiv 2302.13971
//! - He et al. (2015) Kaiming init: arxiv 1502.01852
//! - HuggingFace LlamaPreTrainedModel._init_weights

use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Default initializer range matching HuggingFace LLaMA config.
pub const INITIALIZER_RANGE: f32 = 0.02;

/// Generate `n` random normal values with mean=0 and std=`INITIALIZER_RANGE`.
///
/// Uses a deterministic seed derived from `base_seed` and `name` for
/// reproducibility (C-INIT-001, FALSIFY-INIT-003).
///
/// # Contract (C-INIT-001)
///
/// - `E[result[i]] ≈ 0`
/// - `Var[result[i]] ≈ INITIALIZER_RANGE^2`
/// - Same `(base_seed, name)` → identical output
/// - Different `(base_seed, name)` → different output
pub fn rand_normal_seeded(n: usize, base_seed: u64, name: &str) -> Vec<f32> {
    // Derive per-parameter seed from base_seed + name hash
    let name_hash = hash_name(name);
    let seed = base_seed.wrapping_add(name_hash);
    let mut rng = SmallRng::seed_from_u64(seed);

    let std_dev = INITIALIZER_RANGE;
    (0..n)
        .map(|_| {
            // Box-Muller transform: two uniform → one normal
            let u1: f32 = rand::Rng::random::<f32>(&mut rng).max(1e-7);
            let u2: f32 = rand::Rng::random::<f32>(&mut rng);
            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()) * std_dev
        })
        .collect()
}

/// Simple string hash for name-based seed derivation.
fn hash_name(name: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
    for byte in name.bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(0x0100_0000_01b3); // FNV-1a prime
    }
    h
}

/// Global seed for weight initialization (set from training config).
///
/// Defaults to 42. Set via `set_init_seed()` before `Transformer::new()`.
static INIT_SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(42);

/// Set the global initialization seed (called from training config).
pub fn set_init_seed(seed: u64) {
    INIT_SEED.store(seed, std::sync::atomic::Ordering::SeqCst);
}

/// Get the current initialization seed.
pub fn get_init_seed() -> u64 {
    INIT_SEED.load(std::sync::atomic::Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rand_normal_seeded_deterministic() {
        let a = rand_normal_seeded(100, 42, "test");
        let b = rand_normal_seeded(100, 42, "test");
        assert_eq!(a, b, "Same seed+name must produce identical output");
    }

    #[test]
    fn test_rand_normal_seeded_different_seeds() {
        let a = rand_normal_seeded(100, 42, "test");
        let b = rand_normal_seeded(100, 123, "test");
        assert_ne!(a, b, "Different seeds must produce different output");
    }

    #[test]
    fn test_rand_normal_seeded_different_names() {
        let a = rand_normal_seeded(100, 42, "w_q");
        let b = rand_normal_seeded(100, 42, "w_k");
        assert_ne!(a, b, "Different names must produce different output");
    }

    #[test]
    fn test_rand_normal_seeded_statistics() {
        let data = rand_normal_seeded(10000, 42, "stats_test");
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let variance: f32 =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std = variance.sqrt();

        assert!(mean.abs() < 0.005, "Mean should be near 0, got {mean}");
        assert!(
            (std - INITIALIZER_RANGE).abs() < 0.005,
            "Std should be near {INITIALIZER_RANGE}, got {std}"
        );
    }

    #[test]
    fn test_rand_normal_seeded_no_sinusoidal_pattern() {
        // FALSIFY-INIT-001: autocorrelation at lag 1 should be < 0.1
        let data = rand_normal_seeded(1000, 42, "autocorr_test");
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let var: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let autocorr: f32 = data.windows(2).map(|w| (w[0] - mean) * (w[1] - mean)).sum::<f32>()
            / (data.len() as f32 * var);
        assert!(
            autocorr.abs() < 0.1,
            "Autocorrelation should be < 0.1 (no sinusoidal pattern), got {autocorr}"
        );
    }
}
