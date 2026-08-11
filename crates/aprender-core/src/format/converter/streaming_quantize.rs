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

/// Bytes a single tensor contributes to a Q4K APR, given only its name+shape.
///
/// Mirrors the two Q4K write paths exactly — `save_model_tensors_q4k` and
/// `streaming_quantize_apr_to_q4k` both gate on [`should_quantize_tensor`] and
/// then emit either `quantize_q4_k_matrix` blocks or a plain F32 tensor.
///
/// A Q4K super-block covers 256 elements in 144 bytes, and
/// `quantize_q4_k_matrix` blocks **per row**, zero-padding each row up to a
/// whole number of super-blocks. So a `[N, 384]` weight costs
/// `N * 2 * 144` bytes, not `N * 384 * 4.5 / 8`.
fn q4k_tensor_bytes(name: &str, shape: &[usize]) -> u64 {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: u64 = 144;

    let elements: usize = shape.iter().product();
    if !should_quantize_tensor(name, shape, elements) {
        // Not eligible → written verbatim as F32.
        return (elements as u64).saturating_mul(4);
    }
    if shape.len() == 2 {
        let rows = shape[0] as u64;
        let super_blocks_per_row = shape[1].div_ceil(SUPER_BLOCK_SIZE) as u64;
        return rows
            .saturating_mul(super_blocks_per_row)
            .saturating_mul(SUPER_BLOCK_BYTES);
    }
    // >2D falls through to flat `quantize_q4_k`, which blocks the whole buffer.
    (elements.div_ceil(SUPER_BLOCK_SIZE) as u64).saturating_mul(SUPER_BLOCK_BYTES)
}

/// Estimate the Q4K tensor payload `apr quantize -s q4k` would write, by
/// scanning only the input's tensor index (no tensor data is loaded).
///
/// #2392 (dogfood 0.63.0, finding 3): `apr quantize --plan -s q4k` used to apply
/// a flat "4.5 bits per weight against an assumed-F32 input", producing a
/// **constant** 7.111x reduction ratio for every model ever passed to it. That
/// ignores the two things that actually decide a Q4K model's size: which tensors
/// are eligible at all (embeddings, norms, biases, scales and anything under 256
/// elements stay F32 — 84.4% of one real model's bytes), and the per-row padding
/// Q4K applies to rows whose width is not a multiple of 256. On a real 87 MB
/// model the plan promised 12778677 bytes and quantization produced 55507012 —
/// 4.34x optimistic. On a small model whose weights Q4K actually *inflates*, the
/// plan still promised a 7x shrink.
///
/// Returns the summed tensor payload, or `None` when the input's tensor index
/// cannot be read (unknown format) so the caller can fall back to the flat
/// estimate. This counts tensor bytes only: the container's metadata — notably
/// an embedded tokenizer vocabulary, which can be megabytes for a 151k-token BPE
/// model — is not included, so the estimate is a lower bound on the file size.
pub fn q4k_output_size_estimate(path: &Path) -> Option<u64> {
    q4k_estimate_from_apr(path).or_else(|| q4k_estimate_from_safetensors(path))
}

/// Q4K payload estimate for an APR v2 source (both `apr quantize` APR paths).
fn q4k_estimate_from_apr(path: &Path) -> Option<u64> {
    use crate::bundle::MappedFile;

    let mapped = MappedFile::open(path).ok()?;
    let reader = AprV2ReaderRef::from_bytes(mapped.as_slice()).ok()?;
    let names = reader.tensor_names();
    if names.is_empty() {
        return None;
    }
    Some(
        names
            .iter()
            .filter_map(|n| reader.get_tensor(n).map(|e| (n, e)))
            .map(|(n, e)| q4k_tensor_bytes(n, &e.shape))
            .fold(0u64, u64::saturating_add),
    )
}

/// Q4K payload estimate for a SafeTensors source.
fn q4k_estimate_from_safetensors(path: &Path) -> Option<u64> {
    let mapped = crate::serialization::safetensors::MappedSafeTensors::open(path).ok()?;
    let names = mapped.tensor_names();
    if names.is_empty() {
        return None;
    }
    Some(
        names
            .iter()
            .filter_map(|n| mapped.get_metadata(n).map(|m| (n, m)))
            .map(|(n, m)| q4k_tensor_bytes(n, &m.shape))
            .fold(0u64, u64::saturating_add),
    )
}
