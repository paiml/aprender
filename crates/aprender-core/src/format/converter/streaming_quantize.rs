// Streaming APR→Q4K quantization for large models (GH-434 / ALB-093).
//
// Avoids the ~3x file-size RAM requirement of the full-load path by iterating
// tensors via mmap + AprV2ReaderRef and emitting via AprV2StreamingWriter.
// Peak memory is bounded by the single largest dequantized tensor, not the
// whole model.
//
// Included via `include!()` into converter/mod.rs so it can use the module's
// private helpers (`should_quantize_tensor`, `validate_tensor_values`,
// `quantize_q4_k_matrix`).

use crate::format::v2::{AprV2ReaderRef, AprV2StreamingWriter};
// issue #2231: `get_tensor_as_f32` is the re-attached `AprV2DequantExt` method.
use crate::format::AprV2DequantExt;

/// Streaming threshold: inputs at or above this size take the streaming path.
///
/// Below 4 GiB the full-load path is faster (no mmap overhead, batching wins)
/// and every existing test depends on it. Above 4 GiB the full-load path
/// requires 12+ GiB RAM and starts to OOM on commodity boxes.
pub(crate) const STREAMING_THRESHOLD_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Stream-quantize an APR v2 input to a Q4K APR v2 output.
///
/// Peak memory ≈ (largest tensor as F32) + (Q4K output of that tensor). For a
/// 57 GiB F16 input with expert tensors ≤ 2 GiB F32, peak is ≤ ~2.5 GiB — down
/// from ~170 GiB for the full-load path.
///
/// # Arguments
/// * `input`  — APR v2 source (F16/F32/Q8 etc.)
/// * `output` — target APR v2 Q4K path
///
/// # Returns
/// Number of tensors written.
///
/// # Errors
/// Returns a `FormatError` if mmap, APR v2 parse, per-tensor dequantize, or
/// finalize fails.
pub(crate) fn streaming_quantize_apr_to_q4k(input: &Path, output: &Path) -> Result<usize> {
    use crate::bundle::MappedFile;

    let mapped = MappedFile::open(input).map_err(|e| AprenderError::FormatError {
        message: format!("mmap '{}' failed: {e}", input.display()),
    })?;

    #[cfg(unix)]
    {
        // Best effort — non-fatal if advise fails.
        let _ = mapped.advise_sequential();
    }

    let reader =
        AprV2ReaderRef::from_bytes(mapped.as_slice()).map_err(|e| AprenderError::FormatError {
            message: format!("APR v2 parse of '{}' failed: {e:?}", input.display()),
        })?;

    let names: Vec<String> = reader
        .tensor_names()
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    let param_count: u64 = names
        .iter()
        .filter_map(|n| reader.get_tensor(n))
        .map(|e| e.element_count() as u64)
        .sum();

    let metadata = build_streaming_q4k_metadata(reader.metadata(), param_count);

    let mut writer =
        AprV2StreamingWriter::new(metadata).map_err(|e| AprenderError::FormatError {
            message: format!("streaming writer init failed: {e:?}"),
        })?;

    for name in &names {
        let entry = reader
            .get_tensor(name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("tensor '{name}' missing from index"),
            })?;
        let shape = entry.shape.clone();

        // Dequantize to f32 (one tensor at a time — dropped before the next
        // iteration borrows new data).
        let f32_data =
            reader
                .get_tensor_as_f32(name)
                .ok_or_else(|| AprenderError::FormatError {
                    message: format!(
                        "failed to dequantize tensor '{name}' (dtype {:?})",
                        entry.dtype
                    ),
                })?;

        // Jidoka: validate before writing (catches upstream corruption).
        validate_tensor_values(name, &f32_data)?;

        if should_quantize_tensor(name, &shape, f32_data.len()) {
            let q4k_bytes = quantize_q4_k_matrix(&f32_data, &shape);
            drop(f32_data);
            writer
                .add_q4k_raw_tensor(name.clone(), shape, &q4k_bytes)
                .map_err(|e| AprenderError::FormatError {
                    message: format!("write q4k '{name}' failed: {e:?}"),
                })?;
        } else {
            writer
                .add_f32_tensor(name.clone(), shape, &f32_data)
                .map_err(|e| AprenderError::FormatError {
                    message: format!("write f32 '{name}' failed: {e:?}"),
                })?;
        }
    }

    writer
        .finalize(output)
        .map_err(|e| AprenderError::FormatError {
            message: format!("finalize '{}' failed: {e:?}", output.display()),
        })?;

    Ok(names.len())
}

/// Build Q4K metadata by cloning the source APR metadata and overriding the
/// quantization + param_count fields. Preserves tokenizer, chat template,
/// architecture, rope params, and all custom keys — the streaming path must
/// produce a fully-self-contained output (Jidoka: no silent data loss).
fn build_streaming_q4k_metadata(
    source: &crate::format::v2::AprV2Metadata,
    param_count: u64,
) -> crate::format::v2::AprV2Metadata {
    let mut meta = source.clone();
    meta.quantization = Some(QuantizationMetadata {
        quant_type: "q4_k".to_string(),
        bits: 4,
        block_size: Some(256),
        symmetric: false,
    });
    meta.param_count = param_count;
    // Reset total_size — the writer does not recompute it; leaving the source
    // value here would be stale after quantization shrinks the file.
    meta.total_size = 0;
    if meta.original_format.is_none() {
        meta.original_format = Some("apr".to_string());
    }
    meta
}

/// Check whether the input file qualifies for the streaming Q4K path.
///
/// Criteria (all must hold):
///   1. Magic bytes parse as APR v2.
///   2. File size ≥ effective threshold.
///
/// The effective threshold is the `APR_STREAMING_THRESHOLD` env var (bytes,
/// decimal) if set and parsable as `u64`, else `STREAMING_THRESHOLD_BYTES`.
/// The env override exists so integration tests can exercise the streaming
/// path on pygmy fixtures (the 4 GiB default is infeasible for CI) and so ops
/// can lower the bar on memory-constrained hosts. Production deployments that
/// do not set the variable see the compile-time default.
pub(crate) fn qualifies_for_streaming_q4k(path: &Path) -> bool {
    use crate::format::rosetta::FormatType;

    let threshold = effective_streaming_threshold();
    let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if size < threshold {
        return false;
    }
    matches!(FormatType::from_magic(path), Ok(FormatType::Apr))
}

/// Resolve the streaming threshold, honoring test overrides and env var.
///
/// Precedence: test override (cfg(test) only) > env var > compile-time default.
/// Test override is a `cfg(test)` static so it is compiled out of production
/// builds.
fn effective_streaming_threshold() -> u64 {
    #[cfg(test)]
    {
        let t = STREAMING_THRESHOLD_TEST_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
        if t != u64::MAX {
            return t;
        }
    }
    std::env::var("APR_STREAMING_THRESHOLD")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(STREAMING_THRESHOLD_BYTES)
}

/// Test-only threshold override. `u64::MAX` means "no override, use env/default".
/// Tests that mutate this MUST serialize via `STREAMING_THRESHOLD_TEST_MUTEX`
/// to avoid races with any concurrently-running test.
#[cfg(test)]
pub(crate) static STREAMING_THRESHOLD_TEST_OVERRIDE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(u64::MAX);

/// Serializes tests that set `STREAMING_THRESHOLD_TEST_OVERRIDE`.
#[cfg(test)]
pub(crate) static STREAMING_THRESHOLD_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Estimate the peak RAM the streaming Q4K path would require, scanning only
/// the APR v2 tensor index (no tensor data loaded).
///
/// Peak is bounded by a single tensor's working set: F32 dequant (`n * 4` bytes)
/// plus its Q4K output (`n * 4.5 / 8` bytes rounded up). Returns `None` if the
/// path is not an APR v2 file.
///
/// Used by `apr quantize --plan` for accurate memory reporting on ≥4 GiB inputs
/// where the full-load estimate (input + output) overstates RAM by ~20x.
pub fn streaming_quantize_peak_estimate(path: &Path) -> Option<u64> {
    use crate::bundle::MappedFile;
    use crate::format::v2::AprV2ReaderRef;

    let mapped = MappedFile::open(path).ok()?;
    let reader = AprV2ReaderRef::from_bytes(mapped.as_slice()).ok()?;
    reader
        .tensor_names()
        .iter()
        .filter_map(|n| reader.get_tensor(n))
        .map(|e| {
            let n = e.element_count() as u64;
            let f32_bytes = n.saturating_mul(4);
            let q4k_bytes = n.saturating_mul(9).div_ceil(16);
            f32_bytes.saturating_add(q4k_bytes)
        })
        .max()
}
