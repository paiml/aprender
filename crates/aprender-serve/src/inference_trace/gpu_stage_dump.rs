//! SHIP-007 PR-B (PMAT-CODE-SHIP-007-GPU-CASCADE): GPU-side per-stage F32
//! tensor capture for CPU-vs-GPU parity bisection.
//!
//! Mirrors the CPU `apr trace --save-tensor` byte format (APRT header +
//! f32 LE body) so `apr diff --values` can compare GPU and CPU stage
//! tensors element-wise.
//!
//! Activated via the `APR_GPU_STAGE_DUMP=<dir>` environment variable.
//! When unset, all helpers are zero-cost no-ops (an `is_ok()` env check
//! per call).
//!
//! # Scope
//!
//! - This file ships the helper infrastructure (no GPU dispatch changes).
//! - Subsequent PR-Bn slices wire the helper into specific stages of
//!   `CudaExecutor::forward_all_layers_gpu_to_logits` and friends.
//!
//! # Contract reference
//!
//! `contracts/apr-ship-007-gpu-stage-bisection-v1.yaml` — proof
//! obligation PO-SHIP-007-001 (`gpu_stage_dump_byte_format_matches_cpu`).
//!
//! # Example
//!
//! ```ignore
//! use realizar::inference_trace::gpu_stage_dump::{maybe_dump_host_buffer, GpuStageDumpConfig};
//! use realizar::inference_trace::save_tensor_stage::SaveTensorStage;
//!
//! // At the embedding stage (host-side, no GPU<->host transfer required):
//! let config = GpuStageDumpConfig::from_env();
//! maybe_dump_host_buffer(config.as_ref(), SaveTensorStage::Embedding, 0, &embed_buf)?;
//! ```

/// L0-1b per-op tap (thread-local plan + GPU dump config + gate bypass).
pub mod per_op_tap;

use std::path::{Path, PathBuf};

use crate::inference_trace::save_tensor_emit::write_stage_file;
use crate::inference_trace::save_tensor_stage::SaveTensorStage;

/// Configuration for GPU stage dumps, read once from `APR_GPU_STAGE_DUMP`.
///
/// When the env var is unset, [`GpuStageDumpConfig::from_env`] returns
/// `None` and all callers short-circuit at zero cost.
#[derive(Debug, Clone)]
pub struct GpuStageDumpConfig {
    /// Root directory for stage dumps. Per-layer stages land at
    /// `<output_dir>/layer-N/<stage>.bin`; whole-model stages land at
    /// `<output_dir>/<stage>.bin`.
    pub output_dir: PathBuf,
}

impl GpuStageDumpConfig {
    /// Read configuration from `APR_GPU_STAGE_DUMP`. Returns `None` when
    /// the env var is unset, empty, or whitespace-only.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        if let Some(armed) = per_op_tap::gpu_dump() {
            return Some(armed);
        }
        let raw = std::env::var("APR_GPU_STAGE_DUMP").ok()?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        Some(Self {
            output_dir: PathBuf::from(trimmed),
        })
    }

    /// Create a config with an explicit output directory (for tests).
    #[must_use]
    pub fn with_output_dir(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
        }
    }

    /// Borrow the configured output directory.
    #[must_use]
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }
}

/// Dump a host-side f32 buffer to the configured stage path. No-op when
/// `config` is `None` (the common case when `APR_GPU_STAGE_DUMP` is
/// unset).
///
/// Use this at stages where the tensor already lives on the host (e.g.
/// the GPU forward path's embedding lookup, which goes through
/// `Model::embed_into` on the CPU before being uploaded).
///
/// For tensors that live on the GPU, call
/// [`maybe_dump_gpu_buffer`] (planned in PR-B's GPU-buffer slice).
///
/// # Errors
///
/// Forwards any I/O error from [`write_stage_file`].
pub fn maybe_dump_host_buffer(
    config: Option<&GpuStageDumpConfig>,
    stage: SaveTensorStage,
    layer: u32,
    values: &[f32],
) -> std::io::Result<()> {
    let Some(cfg) = config else {
        return Ok(());
    };
    write_stage_file(&cfg.output_dir, stage, layer, values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Mutex;

    /// `APR_GPU_STAGE_DUMP` is a process-global; serialize test access so
    /// concurrent tests don't race on env mutation.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_env_var<T>(name: &str, value: Option<&str>, f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let prior = std::env::var(name).ok();
        match value {
            Some(v) => std::env::set_var(name, v),
            None => std::env::remove_var(name),
        }
        let result = f();
        match prior {
            Some(p) => std::env::set_var(name, p),
            None => std::env::remove_var(name),
        }
        result
    }

    #[test]
    fn from_env_unset_returns_none() {
        with_env_var("APR_GPU_STAGE_DUMP", None, || {
            assert!(GpuStageDumpConfig::from_env().is_none());
        });
    }

    #[test]
    fn from_env_empty_returns_none() {
        with_env_var("APR_GPU_STAGE_DUMP", Some(""), || {
            assert!(GpuStageDumpConfig::from_env().is_none());
        });
        with_env_var("APR_GPU_STAGE_DUMP", Some("   "), || {
            assert!(GpuStageDumpConfig::from_env().is_none());
        });
    }

    #[test]
    fn from_env_set_returns_config() {
        with_env_var("APR_GPU_STAGE_DUMP", Some("/tmp/ship-007-test"), || {
            let cfg = GpuStageDumpConfig::from_env().expect("env set");
            assert_eq!(cfg.output_dir(), Path::new("/tmp/ship-007-test"));
        });
    }

    #[test]
    fn maybe_dump_host_buffer_no_config_is_noop() {
        // No config → no file, no error.
        let result = maybe_dump_host_buffer(None, SaveTensorStage::Embedding, 0, &[1.0, 2.0]);
        assert!(result.is_ok());
    }

    #[test]
    fn maybe_dump_host_buffer_writes_aprt_format() {
        let tmp = std::env::temp_dir().join(format!(
            "ship-007-pr-b-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let cfg = GpuStageDumpConfig::with_output_dir(&tmp);

        let values: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        maybe_dump_host_buffer(Some(&cfg), SaveTensorStage::Embedding, 0, &values)
            .expect("write_stage_file");

        let file_path = tmp.join("layer-0").join("embedding.bin");
        assert!(file_path.exists(), "expected dump at {:?}", file_path);

        let bytes = fs::read(&file_path).expect("read dumped file");
        // 12-byte APRT header (4-byte magic + 4-byte layer + 4-byte dim) + 16 × 4 bytes body.
        assert_eq!(bytes.len(), 12 + 16 * 4, "APRT header + 16 f32 values");

        // Body should round-trip as the original f32s (little-endian).
        let mut decoded = [0.0f32; 16];
        for (i, chunk) in bytes[12..].chunks_exact(4).enumerate() {
            let arr: [u8; 4] = chunk.try_into().expect("4-byte chunk");
            decoded[i] = f32::from_le_bytes(arr);
        }
        assert_eq!(decoded.as_slice(), values.as_slice());

        // Cleanup.
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn maybe_dump_host_buffer_per_layer_path_isolation() {
        // PO-SHIP-007-003: each (stage, layer) tuple → unique path.
        let tmp = std::env::temp_dir().join(format!(
            "ship-007-pr-b-iso-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let cfg = GpuStageDumpConfig::with_output_dir(&tmp);

        let v1 = vec![1.0f32; 4];
        let v2 = vec![2.0f32; 4];
        maybe_dump_host_buffer(Some(&cfg), SaveTensorStage::Embedding, 0, &v1).expect("dump 0");
        maybe_dump_host_buffer(Some(&cfg), SaveTensorStage::Embedding, 1, &v2).expect("dump 1");

        let p0 = tmp.join("layer-0").join("embedding.bin");
        let p1 = tmp.join("layer-1").join("embedding.bin");
        assert!(p0.exists());
        assert!(p1.exists());
        assert_ne!(fs::read(&p0).unwrap(), fs::read(&p1).unwrap());

        let _ = fs::remove_dir_all(&tmp);
    }
}
