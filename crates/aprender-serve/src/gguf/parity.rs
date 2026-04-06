//! PMAT-232: GPU/CPU Parity Check
//!
//! Runs the same input through both GPU and CPU forward passes
//! and reports exactly where they diverge.
//!
//! Toyota Way principle: "Go and see" (genchi genbutsu) — don't hypothesize
//! about where the bug is. Run both backends and let the data tell you.
//!
//! See contracts/layer-parity-v1.yaml for the full specification.

use crate::error::{RealizarError, Result};
use crate::gguf::{OwnedQuantizedKVCache, OwnedQuantizedModelCuda};

/// Result of a single-token parity check between GPU and CPU
#[derive(Debug)]
pub struct ParityResult {
    /// Token ID that was processed
    pub token_id: u32,
    /// Position in sequence
    pub position: usize,
    /// CPU logits (full vocab)
    pub cpu_logits: Vec<f32>,
    /// GPU logits (full vocab)
    pub gpu_logits: Vec<f32>,
    /// CPU argmax token ID
    pub cpu_argmax: u32,
    /// GPU argmax token ID
    pub gpu_argmax: u32,
    /// Maximum absolute difference between CPU and GPU logits
    pub max_abs_diff: f32,
    /// Index of maximum difference
    pub max_diff_idx: usize,
    /// Number of NaN in CPU logits
    pub cpu_nan_count: usize,
    /// Number of NaN in GPU logits
    pub gpu_nan_count: usize,
}

impl ParityResult {
    /// Whether the GPU and CPU agree on the argmax token
    pub fn argmax_matches(&self) -> bool {
        self.cpu_argmax == self.gpu_argmax
    }

    /// Whether both outputs are numerically clean (no NaN)
    pub fn is_clean(&self) -> bool {
        self.cpu_nan_count == 0 && self.gpu_nan_count == 0
    }
}

impl std::fmt::Display for ParityResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.argmax_matches() && self.is_clean() {
            "OK"
        } else if !self.is_clean() {
            "FAIL (NaN)"
        } else {
            "FAIL (mismatch)"
        };

        writeln!(
            f,
            "Parity check for token {} at position {}:",
            self.token_id, self.position
        )?;
        writeln!(f, "  Status: {}", status)?;
        writeln!(
            f,
            "  CPU argmax: {} (logit={:.4})",
            self.cpu_argmax,
            self.cpu_logits
                .get(self.cpu_argmax as usize)
                .unwrap_or(&f32::NAN)
        )?;
        writeln!(
            f,
            "  GPU argmax: {} (logit={:.4})",
            self.gpu_argmax,
            self.gpu_logits
                .get(self.gpu_argmax as usize)
                .unwrap_or(&f32::NAN)
        )?;
        writeln!(
            f,
            "  Max diff: {:.6} at index {}",
            self.max_abs_diff, self.max_diff_idx
        )?;
        writeln!(
            f,
            "  CPU NaN: {}, GPU NaN: {}",
            self.cpu_nan_count, self.gpu_nan_count
        )?;
        writeln!(
            f,
            "  CPU logits[0..10]: {:?}",
            &self.cpu_logits[..10.min(self.cpu_logits.len())]
        )?;
        writeln!(
            f,
            "  GPU logits[0..10]: {:?}",
            &self.gpu_logits[..10.min(self.gpu_logits.len())]
        )?;
        Ok(())
    }
}

/// Run parity check: process the same tokens through CPU and GPU, compare logits.
///
/// This is the core diagnostic tool for PMAT-232. Instead of hypothesizing about
/// where the GPU diverges from CPU, this function tells you EXACTLY:
/// - Whether the logits match
/// - The maximum difference
/// - Whether either path produces NaN
///
/// # Arguments
///
/// * `cuda_model` - The CUDA-accelerated model
/// * `tokens` - Token IDs to process (e.g., chat-templated prompt)
///
/// # Returns
///
/// Parity results for each token position. The LAST result is the most important
/// as it produces the logits used for the first generated token.
pub fn check_parity(
    cuda_model: &mut OwnedQuantizedModelCuda,
    tokens: &[u32],
) -> Result<Vec<ParityResult>> {
    if tokens.is_empty() {
        return Ok(Vec::new());
    }

    let model = cuda_model.model();
    let config = &model.config;
    let kv_dim = config.kv_dim();
    let num_layers = config.num_layers;
    let max_seq = tokens.len() + 1;

    // Independent KV caches for CPU and GPU
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, max_seq);
    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, max_seq);
    cuda_model.executor_mut().reset_kv_cache_gpu();

    let mut results = Vec::new();

    for (pos, &token_id) in tokens.iter().enumerate() {
        // CPU forward — uses the CPU model reference
        let cpu_logits = cuda_model
            .model()
            .forward_single_with_cache(token_id, &mut cpu_cache, pos)
            .map_err(|e| {
                RealizarError::InferenceError(format!("CPU forward failed at pos {pos}: {e}"))
            })?;

        // GPU forward — same token, same position, independent KV cache
        let gpu_logits = cuda_model
            .forward_gpu_resident(token_id, &mut gpu_cache, pos)
            .map_err(|e| {
                RealizarError::InferenceError(format!("GPU forward failed at pos {pos}: {e}"))
            })?;

        // Compare
        let cpu_nan_count = cpu_logits.iter().filter(|x| x.is_nan()).count();
        let gpu_nan_count = gpu_logits.iter().filter(|x| x.is_nan()).count();

        let cpu_argmax = argmax(&cpu_logits);
        let gpu_argmax = argmax(&gpu_logits);

        let (max_abs_diff, max_diff_idx) = max_diff(&cpu_logits, &gpu_logits);

        results.push(ParityResult {
            token_id,
            position: pos,
            cpu_logits,
            gpu_logits,
            cpu_argmax,
            gpu_argmax,
            max_abs_diff,
            max_diff_idx,
            cpu_nan_count,
            gpu_nan_count,
        });
    }

    Ok(results)
}

/// Summarize parity results: print each token's status and the overall verdict.
pub fn print_parity_summary(results: &[ParityResult]) {
    let mut all_ok = true;
    for r in results {
        let status = if r.argmax_matches() && r.is_clean() {
            "OK"
        } else {
            all_ok = false;
            if !r.is_clean() {
                "FAIL(NaN)"
            } else {
                "FAIL(mismatch)"
            }
        };
        eprintln!(
            "  pos={:>3} token={:>6}  cpu_argmax={:>6} gpu_argmax={:>6}  max_diff={:.6}  {}",
            r.position, r.token_id, r.cpu_argmax, r.gpu_argmax, r.max_abs_diff, status,
        );
    }

    if all_ok {
        eprintln!("\nPARITY: ALL {} positions OK", results.len());
    } else {
        let failures: Vec<_> = results
            .iter()
            .filter(|r| !r.argmax_matches() || !r.is_clean())
            .collect();
        eprintln!(
            "\nPARITY: {} FAILURES out of {} positions",
            failures.len(),
            results.len()
        );

        // Show first divergence in detail
        if let Some(first) = failures.first() {
            eprintln!("\nFirst divergence at position {}:", first.position);
            eprintln!("{first}");
        }
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i as u32)
}

fn max_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    a.iter()
        .zip(b.iter())
        .enumerate()
        .map(|(i, (x, y))| {
            let diff = (x - y).abs();
            // NaN differences show as infinity
            let diff = if diff.is_nan() { f32::INFINITY } else { diff };
            (diff, i)
        })
        .max_by(|(d1, _), (d2, _)| d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0.0, 0))
}
