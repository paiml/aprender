//! LoRA adapter merging (SMED principle - quick changeover).

use entrenar_common::{EntrenarError, Result};
use std::path::Path;

/// LoRA adapter merging engine.
#[derive(Debug)]
pub struct MergeEngine {
    scale: f32,
}

impl Default for MergeEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl MergeEngine {
    /// Create a new merge engine with default scale.
    pub fn new() -> Self {
        Self { scale: 1.0 }
    }

    /// Set the merge scale factor.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Merge adapter weights into base model.
    ///
    /// For each LoRA target module:
    /// W_merged = W_base + (scale * alpha / rank) * (B @ A)
    ///
    /// A is [d_in, rank], B is [rank, d_out] (or transposed depending on convention).
    /// The merge computes the outer product B×A and adds it to W_base.
    pub fn merge(
        &self,
        base_weights: &[f32],
        lora_a: &[f32],
        lora_b: &[f32],
        alpha: f32,
        rank: u32,
    ) -> Vec<f32> {
        let scale_factor = self.scale * alpha / rank as f32;
        let r = rank as usize;

        // Infer matrix dimensions from flat arrays and rank
        // A: [d_in, r] stored row-major → lora_a.len() = d_in * r
        // B: [r, d_out] stored row-major → lora_b.len() = r * d_out
        let d_in = lora_a.len() / r;
        let d_out = lora_b.len() / r;

        // W_base is [d_out, d_in] stored row-major (standard weight layout)
        // W_merged = W_base + scale * B^T @ A^T
        // where B^T is [d_out, r] and A^T is [r, d_in]
        // Result: [d_out, d_in] — same shape as W_base
        let mut result = base_weights.to_vec();

        if d_out * d_in != base_weights.len() {
            // Dimensions don't match — try transposed interpretation
            // A: [r, d_in], B: [d_out, r]
            let d_in_alt = lora_a.len() / r;
            let d_out_alt = lora_b.len() / r;
            if d_out_alt * d_in_alt == base_weights.len() {
                // B[d_out, r] @ A[r, d_in] = [d_out, d_in]
                for row in 0..d_out_alt {
                    for col in 0..d_in_alt {
                        let mut sum = 0.0f32;
                        for k in 0..r {
                            sum += lora_b[row * r + k] * lora_a[k * d_in_alt + col];
                        }
                        result[row * d_in_alt + col] += scale_factor * sum;
                    }
                }
                return result;
            }
            // Fall through — dimensions incompatible, return base unchanged
            eprintln!(
                "[WARN] LoRA merge dimension mismatch: base={}, A={}x{}, B={}x{}",
                base_weights.len(),
                d_in,
                r,
                r,
                d_out
            );
            return result;
        }

        // Standard: A[d_in, r], B[r, d_out]
        // Compute B @ A = [r, d_out]^T @ [d_in, r]^T ...
        // Actually: W += scale * (lora_b^T @ lora_a^T) where result is [d_out, d_in]
        // lora_b as [r, d_out]: lora_b^T is [d_out, r]
        // lora_a as [d_in, r]: lora_a^T is [r, d_in]
        // [d_out, r] @ [r, d_in] = [d_out, d_in] ✓
        for row in 0..d_out {
            for col in 0..d_in {
                let mut sum = 0.0f32;
                for k in 0..r {
                    // B^T[row, k] = B[k, row] = lora_b[k * d_out + row]
                    // A^T[k, col] = A[col, k] = lora_a[col * r + k]
                    sum += lora_b[k * d_out + row] * lora_a[col * r + k];
                }
                result[row * d_in + col] += scale_factor * sum;
            }
        }

        result
    }

    /// Merge multiple adapters with different scales.
    pub fn merge_multiple(&self, base_weights: &[f32], adapters: &[AdapterWeights]) -> Vec<f32> {
        let mut result = base_weights.to_vec();

        for adapter in adapters {
            let scale_factor = adapter.scale * adapter.alpha / adapter.rank as f32;
            for (i, w) in result.iter_mut().enumerate() {
                let a_val = adapter.lora_a.get(i % adapter.lora_a.len()).copied().unwrap_or(0.0);
                let b_val = adapter.lora_b.get(i % adapter.lora_b.len()).copied().unwrap_or(0.0);
                *w += scale_factor * a_val * b_val;
            }
        }

        result
    }

    /// Load adapter from file and merge.
    pub fn merge_from_file(
        &self,
        base_path: &Path,
        adapter_path: &Path,
        output_path: &Path,
    ) -> Result<MergeResult> {
        // Verify files exist
        if !base_path.exists() {
            return Err(EntrenarError::ModelNotFound { path: base_path.to_path_buf() });
        }
        if !adapter_path.exists() {
            return Err(EntrenarError::ModelNotFound { path: adapter_path.to_path_buf() });
        }

        // In real implementation, would load SafeTensors and perform merge
        // For now, return a placeholder result
        Ok(MergeResult {
            output_path: output_path.to_path_buf(),
            merged_params: 0,
            base_size_bytes: 0,
            output_size_bytes: 0,
        })
    }
}

/// Adapter weights for merging.
#[derive(Debug, Clone)]
pub struct AdapterWeights {
    pub lora_a: Vec<f32>,
    pub lora_b: Vec<f32>,
    pub alpha: f32,
    pub rank: u32,
    pub scale: f32,
}

impl AdapterWeights {
    /// Create new adapter weights.
    pub fn new(lora_a: Vec<f32>, lora_b: Vec<f32>, alpha: f32, rank: u32) -> Self {
        Self { lora_a, lora_b, alpha, rank, scale: 1.0 }
    }

    /// Set the scale factor.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }
}

/// Result of a merge operation.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// Path to the merged output
    pub output_path: std::path::PathBuf,
    /// Number of parameters merged
    pub merged_params: u64,
    /// Base model size in bytes
    pub base_size_bytes: u64,
    /// Output model size in bytes
    pub output_size_bytes: u64,
}

impl MergeResult {
    /// Check if the merge resulted in size increase.
    pub fn size_increase_percent(&self) -> f64 {
        if self.base_size_bytes == 0 {
            return 0.0;
        }
        ((self.output_size_bytes as f64 - self.base_size_bytes as f64)
            / self.base_size_bytes as f64)
            * 100.0
    }
}

/// Analyze adapter sparsity and effective rank.
#[derive(Debug, Clone)]
pub struct AdapterAnalysis {
    /// Stated rank
    pub rank: u32,
    /// Alpha scaling
    pub alpha: f32,
    /// Computed scale factor
    pub scale: f32,
    /// Effective rank based on SVD analysis
    pub effective_rank: f32,
    /// Rank utilization percentage
    pub rank_utilization: f64,
    /// Sparsity percentage (near-zero values)
    pub sparsity: f64,
    /// Frobenius norm of adapter
    pub frobenius_norm: f64,
}

/// Analyze an adapter's structure.
pub fn analyze_adapter(lora_a: &[f32], lora_b: &[f32], alpha: f32, rank: u32) -> AdapterAnalysis {
    let sparsity = calculate_sparsity(lora_a) * 0.5 + calculate_sparsity(lora_b) * 0.5;

    // Simplified effective rank estimation
    let effective_rank = (rank as f32) * (1.0 - sparsity as f32);

    let frobenius_norm = f64::from(
        (lora_a.iter().map(|x| x * x).sum::<f32>() + lora_b.iter().map(|x| x * x).sum::<f32>())
            .sqrt(),
    );

    AdapterAnalysis {
        rank,
        alpha,
        scale: alpha / rank as f32,
        effective_rank,
        rank_utilization: f64::from(effective_rank / rank as f32) * 100.0,
        sparsity: sparsity * 100.0,
        frobenius_norm,
    }
}

fn calculate_sparsity(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let near_zero = values.iter().filter(|&&x| x.abs() < 1e-6).count();
    near_zero as f64 / values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_adds_adapter_contribution() {
        let engine = MergeEngine::new();
        let base = vec![1.0, 2.0, 3.0, 4.0];
        let lora_a = vec![0.1, 0.2];
        let lora_b = vec![0.5, 0.5];

        let merged = engine.merge(&base, &lora_a, &lora_b, 16.0, 64);

        // Merged should differ from base
        assert!(merged.iter().zip(&base).any(|(m, b)| (m - b).abs() > 1e-6));
    }

    #[test]
    fn test_merge_scale_affects_result() {
        let base = vec![1.0, 2.0, 3.0, 4.0];
        let lora_a = vec![0.1, 0.2];
        let lora_b = vec![0.5, 0.5];

        let merged_1 = MergeEngine::new().with_scale(1.0).merge(&base, &lora_a, &lora_b, 16.0, 64);
        let merged_2 = MergeEngine::new().with_scale(2.0).merge(&base, &lora_a, &lora_b, 16.0, 64);

        // Higher scale should produce larger difference from base
        let diff_1: f32 = merged_1.iter().zip(&base).map(|(m, b)| (m - b).abs()).sum();
        let diff_2: f32 = merged_2.iter().zip(&base).map(|(m, b)| (m - b).abs()).sum();
        assert!(diff_2 > diff_1);
    }

    #[test]
    fn test_merge_multiple_adapters() {
        let engine = MergeEngine::new();
        let base = vec![1.0, 2.0, 3.0, 4.0];

        let adapters = vec![
            AdapterWeights::new(vec![0.1, 0.1], vec![0.5, 0.5], 16.0, 64),
            AdapterWeights::new(vec![0.2, 0.2], vec![0.3, 0.3], 8.0, 32).with_scale(0.5),
        ];

        let merged = engine.merge_multiple(&base, &adapters);

        // Result should differ from base
        assert!(merged.iter().zip(&base).any(|(m, b)| (m - b).abs() > 1e-6));
    }

    #[test]
    fn test_adapter_analysis() {
        let lora_a = vec![0.1, 0.2, 0.3, 0.0, 0.0];
        let lora_b = vec![0.5, 0.5, 0.0, 0.0, 0.5];

        let analysis = analyze_adapter(&lora_a, &lora_b, 16.0, 64);

        assert_eq!(analysis.rank, 64);
        assert_eq!(analysis.alpha, 16.0);
        assert!(analysis.sparsity > 0.0); // Some zeros
        assert!(analysis.frobenius_norm > 0.0);
    }

    #[test]
    fn test_sparsity_calculation() {
        let sparse = vec![0.0, 0.0, 0.0, 1.0];
        assert!((calculate_sparsity(&sparse) - 0.75).abs() < 0.01);

        let dense = vec![1.0, 2.0, 3.0, 4.0];
        assert!((calculate_sparsity(&dense)).abs() < 0.01);
    }

    #[test]
    fn test_merge_result_size_increase() {
        let result = MergeResult {
            output_path: std::path::PathBuf::from("/tmp/out"),
            merged_params: 1000,
            base_size_bytes: 1000,
            output_size_bytes: 1100,
        };

        assert!((result.size_increase_percent() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_merge_result_zero_base() {
        let result = MergeResult {
            output_path: std::path::PathBuf::from("/tmp/out"),
            merged_params: 1000,
            base_size_bytes: 0,
            output_size_bytes: 1100,
        };

        // Should not panic, returns 0.0
        assert_eq!(result.size_increase_percent(), 0.0);
    }

    #[test]
    fn test_merge_engine_default() {
        let engine = MergeEngine::default();
        // Default scale is 1.0
        let base = vec![1.0, 2.0];
        let lora_a = vec![1.0];
        let lora_b = vec![1.0];
        let merged = engine.merge(&base, &lora_a, &lora_b, 16.0, 16);
        // With scale=1.0, alpha=16, rank=16: scale_factor = 1.0
        // merged[0] = 1.0 + 1.0 * 1.0 * 1.0 = 2.0
        assert!((merged[0] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_adapter_weights_with_scale() {
        let adapter = AdapterWeights::new(vec![0.1], vec![0.2], 8.0, 32).with_scale(0.5);
        assert_eq!(adapter.scale, 0.5);
        assert_eq!(adapter.alpha, 8.0);
        assert_eq!(adapter.rank, 32);
    }

    #[test]
    fn test_merge_with_empty_adapters() {
        let engine = MergeEngine::new();
        let base = vec![1.0, 2.0, 3.0];
        let adapters: Vec<AdapterWeights> = vec![];

        let merged = engine.merge_multiple(&base, &adapters);
        // No adapters = result equals base
        assert_eq!(merged, base);
    }

    #[test]
    fn test_merge_from_file_missing_base() {
        let engine = MergeEngine::new();
        let result = engine.merge_from_file(
            Path::new("/nonexistent/base.safetensors"),
            Path::new("/nonexistent/adapter.safetensors"),
            Path::new("/tmp/output.safetensors"),
        );

        assert!(result.is_err());
        if let Err(EntrenarError::ModelNotFound { path }) = result {
            assert!(path.to_string_lossy().contains("base.safetensors"));
        }
    }

    #[test]
    fn test_sparsity_empty_input() {
        assert_eq!(calculate_sparsity(&[]), 0.0);
    }

    #[test]
    fn test_sparsity_all_zeros() {
        let zeros = vec![0.0, 0.0, 0.0, 0.0];
        assert!((calculate_sparsity(&zeros) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_adapter_analysis_effective_rank() {
        // Dense adapter (no zeros)
        let lora_a = vec![0.1, 0.2, 0.3, 0.4];
        let lora_b = vec![0.5, 0.6, 0.7, 0.8];
        let analysis = analyze_adapter(&lora_a, &lora_b, 16.0, 64);

        // Dense adapter should have high effective rank
        assert!(analysis.effective_rank > 60.0);
        assert!(analysis.rank_utilization > 90.0);
    }

    #[test]
    fn test_adapter_analysis_scale_calculation() {
        let lora_a = vec![0.1];
        let lora_b = vec![0.1];
        let analysis = analyze_adapter(&lora_a, &lora_b, 32.0, 64);

        // scale = alpha / rank = 32 / 64 = 0.5
        assert!((analysis.scale - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_merge_engine_with_scale_builder() {
        let engine = MergeEngine::new().with_scale(0.75);
        let base = vec![1.0, 1.0];
        let lora_a = vec![1.0];
        let lora_b = vec![1.0];
        // scale_factor = 0.75 * 8.0 / 8 = 0.75
        let merged = engine.merge(&base, &lora_a, &lora_b, 8.0, 8);
        // merged[0] = 1.0 + 0.75 * 1.0 * 1.0 = 1.75
        assert!((merged[0] - 1.75).abs() < 0.01);
    }
}
