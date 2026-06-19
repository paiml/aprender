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
    /// Uses the STANDARD PEFT adapter layout produced by `apr finetune`
    /// (`crates/apr-cli/src/commands/finetune.rs:842,845`):
    /// `lora_a` is [rank, d_in] and `lora_b` is [d_out, rank], both row-major.
    /// The merge computes ΔW = B @ A (shape [d_out, d_in]) and adds scale·ΔW to W_base.
    /// This matches the in-repo correct twin `QLoRALayer::merge_to_f32`.
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

        // Infer matrix dimensions from flat arrays and rank (PEFT layout).
        // A: [r, d_in] stored row-major → lora_a.len() = r * d_in
        // B: [d_out, r] stored row-major → lora_b.len() = d_out * r
        let d_in = lora_a.len() / r;
        let d_out = lora_b.len() / r;

        // W_base is [d_out, d_in] stored row-major (standard weight layout).
        // W_merged = W_base + scale * (B @ A)
        // where B is [d_out, r] and A is [r, d_in]
        // Result: [d_out, d_in] — same shape as W_base
        let mut result = base_weights.to_vec();

        if d_out * d_in != base_weights.len() {
            // Defensive re-derivation of dims (same PEFT layout: A:[r,d_in], B:[d_out,r]).
            let d_in_alt = lora_a.len() / r;
            let d_out_alt = lora_b.len() / r;
            if d_out_alt * d_in_alt == base_weights.len() {
                // ΔW = B[d_out, r] @ A[r, d_in] = [d_out, d_in]
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

        // PEFT layout: A is [r, d_in], B is [d_out, r].
        // ΔW = B @ A = [d_out, r] @ [r, d_in] = [d_out, d_in] ✓
        // W_merged = W_base + scale · ΔW (same indexing as QLoRALayer::merge_to_f32).
        for row in 0..d_out {
            for col in 0..d_in {
                let mut sum = 0.0f32;
                for k in 0..r {
                    // B[row, k] = lora_b[row * r + k]
                    // A[k, col] = lora_a[k * d_in + col]
                    sum += lora_b[row * r + k] * lora_a[k * d_in + col];
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
                let a_val = adapter
                    .lora_a
                    .get(i % adapter.lora_a.len())
                    .copied()
                    .unwrap_or(0.0);
                let b_val = adapter
                    .lora_b
                    .get(i % adapter.lora_b.len())
                    .copied()
                    .unwrap_or(0.0);
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
            return Err(EntrenarError::ModelNotFound {
                path: base_path.to_path_buf(),
            });
        }
        if !adapter_path.exists() {
            return Err(EntrenarError::ModelNotFound {
                path: adapter_path.to_path_buf(),
            });
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
        Self {
            lora_a,
            lora_b,
            alpha,
            rank,
            scale: 1.0,
        }
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
        let base = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2] weight matrix
                                             // rank=1: A=[2,1], B=[1,2]
        let lora_a = vec![0.1, 0.2];
        let lora_b = vec![0.5, 0.5];

        let merged = engine.merge(&base, &lora_a, &lora_b, 16.0, 1);

        // Merged should differ from base
        assert!(merged.iter().zip(&base).any(|(m, b)| (m - b).abs() > 1e-6));
    }

    #[test]
    fn test_merge_scale_affects_result() {
        let base = vec![1.0, 2.0, 3.0, 4.0];
        // rank=1: A=[2,1], B=[1,2]
        let lora_a = vec![0.1, 0.2];
        let lora_b = vec![0.5, 0.5];

        let merged_1 = MergeEngine::new()
            .with_scale(1.0)
            .merge(&base, &lora_a, &lora_b, 16.0, 1);
        let merged_2 = MergeEngine::new()
            .with_scale(2.0)
            .merge(&base, &lora_a, &lora_b, 16.0, 1);

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
        // Default scale is 1.0; rank=1: A=[1,1], B=[1,1], base=[1,1]
        let base = vec![1.0];
        let lora_a = vec![1.0];
        let lora_b = vec![1.0];
        let merged = engine.merge(&base, &lora_a, &lora_b, 16.0, 1);
        // scale_factor = 1.0 * 16.0 / 1 = 16.0
        // merged[0] = 1.0 + 16.0 * 1.0 * 1.0 = 17.0
        assert!((merged[0] - 17.0).abs() < 0.01);
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
        // rank=1: A=[1,1], B=[1,1], base=[1,1]
        let base = vec![1.0];
        let lora_a = vec![1.0];
        let lora_b = vec![1.0];
        // scale_factor = 0.75 * 8.0 / 1 = 6.0
        let merged = engine.merge(&base, &lora_a, &lora_b, 8.0, 1);
        // merged[0] = 1.0 + 6.0 * 1.0 * 1.0 = 7.0
        assert!((merged[0] - 7.0).abs() < 0.01);
    }

    /// PMAT-854 falsifier: `MergeEngine::merge` must honor the STANDARD PEFT
    /// adapter layout produced by `apr finetune` — A:[rank,d_in], B:[d_out,rank]
    /// (see `crates/apr-cli/src/commands/finetune.rs:842,845`). The merge must fold
    /// `W += scale·(B@A)` using THAT indexing, matching the in-repo correct twin
    /// `QLoRALayer::merge_to_f32` (`crates/aprender-train/src/lora/qlora.rs:240-245`).
    ///
    /// Repro (d_in=3, d_out=2, rank=2, alpha=2 → scale=alpha/rank=1, W=zeros):
    ///   A = [[1,0,0],[0,2,0]]  ([rank,d_in])
    ///   B = identity           ([d_out,rank])
    ///   B@A = [[1,0,0],[0,2,0]]
    /// Correct merge → [1,0,0, 0,2,0]. The pre-fix (transposed) code yielded
    /// [1,0,2, 0,0,0] (max abs error 2.0) because it read BOTH factors transposed.
    #[test]
    fn merge_uses_peft_layout() {
        // PEFT layout: A is [rank, d_in], B is [d_out, rank].
        let a: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0]; // [2,3] = [rank, d_in]
        let b: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0]; // [2,2] = [d_out, rank] identity
        let w_base: Vec<f32> = vec![0.0; 6]; // [d_out=2, d_in=3]

        // alpha=2, rank=2 → scale_factor = 1.0 * 2.0 / 2 = 1.0
        let merged = MergeEngine::new().merge(&w_base, &a, &b, 2.0, 2);

        // B@A = [[1,0,0],[0,2,0]] → W + 1.0*(B@A).
        let expected = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        let max_abs_diff = merged
            .iter()
            .zip(expected.iter())
            .map(|(m, e)| (m - e).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs_diff < 1e-6,
            "PMAT-854: MergeEngine::merge does NOT honor PEFT layout (A:[rank,d_in], \
             B:[d_out,rank]). Got {merged:?}, expected {expected:?} (= W + scale·(B@A)). \
             The pre-fix transposed code yields [1,0,2, 0,0,0]."
        );
    }

    /// BEAT-LORA-MERGE — Pillar-3 (Unsloth) correctness beat (PMAT-747).
    ///
    /// The other half of "replace Unsloth's QLoRA pipeline" (NF4 quant ≡ bitsandbytes
    /// is PMAT-745; this is fine-tune→merge). apr's `MergeEngine::merge` folds the
    /// LoRA delta `scale·(B@A)` into the base weight; this gate proves the MERGED
    /// weights produce a forward pass NUMERICALLY EQUIVALENT to applying the LoRA
    /// factors unmerged — i.e. the merge is mathematically faithful, contract-gated,
    /// where PEFT/Unsloth ship merge_and_unload with no such equivalence guarantee.
    ///
    /// The reference is computed INDEPENDENTLY from the A,B factors via a different
    /// path (x @ A @ B), so this is not tautological: a transpose/indexing bug in
    /// `merge` would diverge. Self-contained (CPU, deterministic).
    #[test]
    fn beat_lora_merge_forward_equivalence() {
        // dims chosen so d_out != d_in (unambiguous merge path).
        let (d_in, d_out, r) = (4usize, 3usize, 2usize);
        let n = 2usize; // batch

        // PEFT layout (matches `apr finetune`): A:[rank,d_in] row-major (2x4),
        // B:[d_out,rank] row-major (3x2); W_base:[d_out,d_in] (3x4).
        let a: Vec<f32> = vec![0.10, -0.20, 0.05, 0.30, -0.10, 0.25, 0.40, -0.15]; // 2x4 [rank,d_in]
        let b: Vec<f32> = vec![0.20, -0.30, 0.10, 0.15, 0.05, -0.25]; // 3x2 [d_out,rank]
        let w_base: Vec<f32> = (0..d_out * d_in).map(|i| (i as f32 - 6.0) * 0.07).collect(); // 3x4
        let x: Vec<f32> = vec![0.5, -0.2, 0.3, 0.1, -0.4, 0.6, 0.2, -0.1]; // 2x4
        let (alpha, rank) = (4.0_f32, r as u32);
        let scale_factor = 1.0 * alpha / rank as f32; // default scale 1.0 → 2.0

        // apr merge: fold delta into base.
        let merged = MergeEngine::new().merge(&w_base, &a, &b, alpha, rank);

        // Forward with MERGED weights: y_m[i,row] = sum_col x[i,col]*merged[row,col].
        let mut y_merged = vec![0.0f32; n * d_out];
        for i in 0..n {
            for row in 0..d_out {
                let mut s = 0.0;
                for col in 0..d_in {
                    s += x[i * d_in + col] * merged[row * d_in + col];
                }
                y_merged[i * d_out + row] = s;
            }
        }

        // INDEPENDENT reference: base forward + scale·(x @ A^T @ B^T) where ΔW = B@A.
        // PEFT layout: A[k,col]=a[k*d_in+col]; B[row,k]=b[row*r+k].
        // (x@A^T)[i,k] = sum_col x[i,col]*A[k,col]; then (..@B^T)[i,row] = sum_k xat[i,k]*B[row,k].
        let mut y_ref = vec![0.0f32; n * d_out];
        for i in 0..n {
            // base contribution
            for row in 0..d_out {
                let mut base = 0.0;
                for col in 0..d_in {
                    base += x[i * d_in + col] * w_base[row * d_in + col];
                }
                y_ref[i * d_out + row] = base;
            }
            // LoRA contribution via the factors (different computation path)
            let mut xat = vec![0.0f32; r];
            for (k, xat_k) in xat.iter_mut().enumerate() {
                let mut s = 0.0;
                for col in 0..d_in {
                    s += x[i * d_in + col] * a[k * d_in + col]; // A[k,col]
                }
                *xat_k = s;
            }
            for row in 0..d_out {
                let mut s = 0.0;
                for (k, &xat_k) in xat.iter().enumerate() {
                    s += xat_k * b[row * r + k]; // B[row,k]
                }
                y_ref[i * d_out + row] += scale_factor * s;
            }
        }

        let max_abs_diff = y_merged
            .iter()
            .zip(y_ref.iter())
            .map(|(m, r)| (m - r).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs_diff < 1e-4,
            "LoRA merge NOT forward-equivalent: max|y_merged - y_factored|={max_abs_diff:.6} \
             (a transpose/indexing bug in MergeEngine::merge). y_merged={y_merged:?} y_ref={y_ref:?}"
        );
        println!(
            "BEAT-LORA-MERGE: merged-forward ≡ factored-LoRA-forward — max|Δ|={max_abs_diff:.2e}"
        );
    }
}
