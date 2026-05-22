//! PMAT-701 Bug B: Q4K-native frozen-teacher path for `apr distill --backend cuda`.
//!
//! Contract: `contracts/cuda-q4k-frozen-teacher-v1.yaml`
//!
//! # What this fixes
//!
//! The legacy `CudaTrainerTeacher` (aprender-train-distill) dequantizes Q4K
//! teacher weights to F32 at GPU upload. A 7B Q4K teacher (4 GB on disk)
//! inflates to ~28 GB of F32 GPU memory, which trips the Linux OOM killer
//! once you add the student's F32 + grads + Adam + activations footprint.
//!
//! [`RealizarQ4KTeacher`] takes a different path: it wraps realizar's
//! existing inference-time `OwnedQuantizedModelCuda`, which keeps weights
//! in Q4K on the GPU and uses Q4K-native CUDA kernels for forward GEMM
//! (the same path validated by `apr run`). Teacher GPU footprint stays at
//! ~4 GB for the 7B model — within budget on Grace Blackwell GB10.
//!
//! # When this path is used
//!
//! [`run_cuda_backend`](super::distill::run_cuda_backend) inspects the
//! teacher .apr's tensor dtype histogram. Any presence of
//! [`TensorDType::Q4K`] or [`TensorDType::Q6K`] routes the teacher
//! provider here. F32/F16/BF16 teachers continue to use
//! `CudaTrainerTeacher` (the dequant path is harmless for those types
//! since they don't inflate).
//!
//! # Falsifier mapping
//!
//! - FT-Q4K-TEACHER-001: no `[PMAT-333] Dequantizing` log line in the
//!   teacher load path (the dequant happens inside realizar's
//!   `OwnedQuantizedModel::from_apr` only for non-quantized tensors,
//!   not for Q4K blocks).
//! - FT-Q4K-TEACHER-002: peak GPU memory for 7B Q4K teacher <= 6 GB.
//! - FT-Q4K-TEACHER-003: forward parity with `apr run` (same kernel path).
//! - FT-Q4K-TEACHER-005: `apr distill --epochs 1` completes on GB10.

#![cfg(all(feature = "cuda", feature = "training", feature = "inference"))]

use std::path::Path;

use entrenar_common::{EntrenarError, Result};
use entrenar_distill::teacher_provider::TeacherLogitsProvider;
use realizar::apr::MappedAprModel;
use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};

/// Teacher provider backed by realizar's CUDA inference path.
///
/// Constructed from a frozen-on-disk APR teacher (typically Q4K-quantized).
/// Holds an `OwnedQuantizedModelCuda` whose weights live on the GPU in their
/// native quantization format — no dequantization to F32 at upload, no
/// gradient/optimizer state.
///
/// # Memory budget
///
/// For a 7B Q4K teacher on Grace Blackwell GB10:
/// - On-disk: ~4 GB (Q4K super-blocks)
/// - GPU resident after `preload_weights_gpu()`: ~4 GB (same Q4K blocks)
/// - Forward activations per token: ~1-2 GB (transient)
///
/// vs. the legacy `CudaTrainerTeacher` which would consume ~28 GB F32.
pub struct RealizarQ4KTeacher {
    cuda_model: OwnedQuantizedModelCuda,
    vocab_size: usize,
}

impl RealizarQ4KTeacher {
    /// Load a Q4K teacher from an APR checkpoint and stage it on the GPU.
    ///
    /// # Errors
    ///
    /// Returns `EntrenarError::Internal` if the APR file cannot be mapped,
    /// the quantized model construction fails, or CUDA initialization fails.
    /// Per `cuda-q4k-frozen-teacher-v1.yaml` falsifier FT-Q4K-TEACHER-005,
    /// the load succeeds on Grace Blackwell GB10 for 7B Q4K teachers when
    /// `cuda-unified-memory-allocator-v1.yaml` (Bug A) is also in effect.
    pub fn from_apr_path(model_path: &Path) -> Result<Self> {
        let mapped =
            MappedAprModel::from_path(model_path).map_err(|e| EntrenarError::Internal {
                message: format!(
                    "RealizarQ4KTeacher: MappedAprModel::from_path({}): {e}",
                    model_path.display()
                ),
            })?;

        let quantized =
            OwnedQuantizedModel::from_apr(&mapped).map_err(|e| EntrenarError::Internal {
                message: format!("RealizarQ4KTeacher: OwnedQuantizedModel::from_apr: {e}"),
            })?;

        let vocab_size = quantized.config().vocab_size;

        let mut cuda_model =
            OwnedQuantizedModelCuda::new(quantized, 0).map_err(|e| EntrenarError::Internal {
                message: format!("RealizarQ4KTeacher: OwnedQuantizedModelCuda::new: {e}"),
            })?;

        // Pre-upload all Q4K weights to GPU. This is what makes the next
        // forward pass fast; without it, weights stream lazily and per-batch
        // latency dominates. Failure here is non-fatal — realizar falls back
        // to on-demand upload.
        match cuda_model.preload_weights_gpu() {
            Ok(bytes) => {
                eprintln!(
                    "[PMAT-701] RealizarQ4KTeacher: pre-uploaded {} MB to GPU (Q4K-native, no F32 dequant)",
                    bytes / (1024 * 1024)
                );
            }
            Err(e) => {
                eprintln!(
                    "[PMAT-701] RealizarQ4KTeacher: weight preload failed ({e}); falling back to on-demand upload"
                );
            }
        }

        Ok(Self {
            cuda_model,
            vocab_size,
        })
    }
}

impl TeacherLogitsProvider for RealizarQ4KTeacher {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn logits_for_batch(&mut self, input_ids: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(input_ids.len());
        for ids in input_ids {
            let logits =
                self.cuda_model
                    .forward_cuda(ids)
                    .map_err(|e| EntrenarError::Internal {
                        message: format!("RealizarQ4KTeacher.forward_cuda: {e}"),
                    })?;
            if logits.len() != self.vocab_size {
                return Err(EntrenarError::Internal {
                    message: format!(
                        "RealizarQ4KTeacher: forward_cuda returned {} logits, expected {} \
                         (vocab mismatch — teacher config does not match the on-disk checkpoint)",
                        logits.len(),
                        self.vocab_size
                    ),
                });
            }
            out.push(logits);
        }
        Ok(out)
    }
}
