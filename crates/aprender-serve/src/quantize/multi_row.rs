//! #4228: K-quant matmul over `t` activation rows at once (CPU prefill).
//!
//! `output[r][o] = W[o] · input[r]` for `r < t`, where `input` is `[t][in_dim]`
//! and `output` is `[t][out_dim]`, both row-major and contiguous.
//!
//! The one-row path reads the whole weight matrix from DRAM once per token, so
//! a T-token prefill streams it T times. Here every rayon task owns a tile of
//! weight rows and applies each row to a tile of [`TOKEN_TILE`] tokens while it
//! is hot in L1. The weight matrix is then read about once per token tile, and
//! a matmul costs one rayon fan-out instead of `t`.
//!
//! **Bitwise the one-row path, per row.** For every `(r, o)` the value comes
//! from the same kernel and the same inputs as
//! `OwnedQuantizedModel::fused_matmul_into` on `input[r]`:
//! - **Q4_K:** `input[r]` is padded and quantized to Q8_K on its own, its bsums
//!   are computed on their own, and the row goes through
//!   `ggml_style_q4k_q8k_dot_avx2_raw`, as `fused_q4k_q8k_parallel_matvec_into`'s
//!   AVX2+FMA path does.
//! - **Q5_K / Q6_K:** the same `fused_q5k_dot_simd` / `fused_q6k_dot_simd` on
//!   `input[r]` padded to 256, as `generic_parallel_matvec_into` does.
//!
//! Neither path sums across rows or tokens, so a token's output does not depend
//! on the other tokens in the batch. A case this module does not reproduce
//! exactly returns `Ok(false)` so the caller takes the one-row path. That covers
//! other qtypes, a CPU without AVX2+FMA, and the FP32-activation scopes
//! (`DIRECT_FP32_GEMV=1`, [`super::parallel_k::with_fp32_activations`]).

use crate::error::{RealizarError, Result};
use crate::gguf::{GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K};

use super::QK_K;

/// Tokens a weight row is applied to while it is resident in L1.
pub const TOKEN_TILE: usize = 8;

/// Weight rows one rayon task owns, as the one-row path's midi-tile.
const ROW_TILE: usize = 64;

const Q4K_BYTES: usize = 144;
const Q5K_BYTES: usize = 176;
const Q6K_BYTES: usize = 210;

/// Does [`fused_k_rows_matmul_into`] compute `qtype` itself on this thread and CPU?
/// `false` means it would return `Ok(false)`.
#[must_use]
pub fn multi_row_supported(qtype: u32) -> bool {
    match qtype {
        GGUF_TYPE_Q4_K => q4k_lean_available(),
        GGUF_TYPE_Q5_K | GGUF_TYPE_Q6_K => true,
        _ => false,
    }
}

fn q4k_lean_available() -> bool {
    let fp32 = super::parallel_k::fp32_activations_scoped()
        || std::env::var("DIRECT_FP32_GEMV").as_deref() == Ok("1");
    #[cfg(target_arch = "x86_64")]
    let simd = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let simd = false;
    simd && !fp32
}

/// `output[r] = W · input[r]` for the `t` rows of `input`. Returns `Ok(false)`,
/// with `output` untouched, when `qtype` is not one this module reproduces
/// bitwise (see the module docs); the caller then runs the one-row path.
///
/// # Errors
/// A shape mismatch between `weight_data`, `input`, `output` and the dims.
pub fn fused_k_rows_matmul_into(
    qtype: u32,
    weight_data: &[u8],
    input: &[f32],
    t: usize,
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<bool> {
    if !multi_row_supported(qtype) || t == 0 {
        return Ok(false);
    }
    let sb_bytes = match qtype {
        GGUF_TYPE_Q4_K => Q4K_BYTES,
        GGUF_TYPE_Q5_K => Q5K_BYTES,
        _ => Q6K_BYTES,
    };
    let nsb = in_dim.div_ceil(QK_K);
    let bytes_per_row = nsb * sb_bytes;
    if weight_data.len() < out_dim * bytes_per_row
        || input.len() != t * in_dim
        || output.len() < t * out_dim
    {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "multi-row K-quant: weight {} B (need {}), input {} (need {}), output {} (need {})",
                weight_data.len(),
                out_dim * bytes_per_row,
                input.len(),
                t * in_dim,
                output.len(),
                t * out_dim
            ),
        });
    }
    let padded = nsb * QK_K;
    let mut acts = vec![0.0f32; t * padded];
    for (r, row) in input.chunks_exact(in_dim).enumerate() {
        acts[r * padded..r * padded + in_dim].copy_from_slice(row);
    }

    // `[out_dim][t]`: a rayon task owns ROW_TILE whole rows of it.
    let mut by_row = vec![0.0f32; out_dim * t];
    match qtype {
        GGUF_TYPE_Q4_K => q4k_rows(weight_data, &acts, t, nsb, bytes_per_row, &mut by_row)?,
        GGUF_TYPE_Q5_K => f32_rows(
            weight_data,
            &acts,
            t,
            padded,
            bytes_per_row,
            &mut by_row,
            super::fused_q5k_q6k::fused_q5k_dot_simd,
        ),
        _ => f32_rows(
            weight_data,
            &acts,
            t,
            padded,
            bytes_per_row,
            &mut by_row,
            super::fused_q5k_q6k::fused_q6k_dot_simd,
        ),
    }
    for (o, vals) in by_row.chunks_exact(t).enumerate() {
        for (r, &v) in vals.iter().enumerate() {
            output[r * out_dim + o] = v;
        }
    }
    Ok(true)
}

/// Q5_K / Q6_K: the one-row path's f32 dot on each padded token row.
fn f32_rows(
    weight_data: &[u8],
    acts: &[f32],
    t: usize,
    padded: usize,
    bytes_per_row: usize,
    by_row: &mut [f32],
    dot: fn(&[u8], &[f32]) -> Result<f32>,
) {
    use rayon::prelude::*;
    by_row
        .par_chunks_mut(ROW_TILE * t)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let row0 = ci * ROW_TILE;
            let rows = chunk.len() / t;
            for r0 in (0..t).step_by(TOKEN_TILE) {
                let r1 = (r0 + TOKEN_TILE).min(t);
                for i in 0..rows {
                    let w0 = (row0 + i) * bytes_per_row;
                    let w = &weight_data[w0..w0 + bytes_per_row];
                    for r in r0..r1 {
                        chunk[i * t + r] =
                            dot(w, &acts[r * padded..(r + 1) * padded]).unwrap_or(0.0);
                    }
                }
            }
        });
}

/// Q4_K: each token row quantized to Q8_K on its own, then the lean AVX2 dot.
#[allow(clippy::unnecessary_wraps)] // Err only on non-x86_64, where it is unreachable
fn q4k_rows(
    weight_data: &[u8],
    acts: &[f32],
    t: usize,
    nsb: usize,
    bytes_per_row: usize,
    by_row: &mut [f32],
) -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        use rayon::prelude::*;
        let padded = nsb * QK_K;
        let mut scales = vec![0.0f32; t * nsb];
        let mut quants = vec![0i8; t * padded];
        let mut bsums = Vec::with_capacity(t * nsb * 16);
        for r in 0..t {
            let q = &mut quants[r * padded..(r + 1) * padded];
            super::quantize_activations_q8k_into(
                &acts[r * padded..(r + 1) * padded],
                &mut scales[r * nsb..(r + 1) * nsb],
                q,
            )?;
            // SAFETY: q4k_lean_available() checked AVX2 before this function was
            // reached, and `q` holds nsb * 256 quants.
            bsums.extend(unsafe { super::fused_k::precompute_q8k_bsums_i16(q, nsb) });
        }
        by_row
            .par_chunks_mut(ROW_TILE * t)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let row0 = ci * ROW_TILE;
                let rows = chunk.len() / t;
                for r0 in (0..t).step_by(TOKEN_TILE) {
                    let r1 = (r0 + TOKEN_TILE).min(t);
                    for i in 0..rows {
                        let w0 = (row0 + i) * bytes_per_row;
                        let w = &weight_data[w0..w0 + bytes_per_row];
                        for r in r0..r1 {
                            // SAFETY: AVX2+FMA checked by q4k_lean_available(); `w` is one
                            // whole row of nsb super-blocks, and token r's scales, quants
                            // and bsums are nsb, nsb*256 and nsb*16 long.
                            chunk[i * t + r] = unsafe {
                                super::fused_k::ggml_style_q4k_q8k_dot_avx2_raw(
                                    w.as_ptr(),
                                    scales[r * nsb..].as_ptr(),
                                    quants[r * padded..].as_ptr(),
                                    bsums[r * nsb * 16..].as_ptr(),
                                    nsb,
                                )
                            };
                        }
                    }
                }
            });
        Ok(())
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (weight_data, acts, t, nsb, bytes_per_row, by_row);
        Err(RealizarError::InvalidShape {
            reason: "multi-row Q4_K needs x86_64 AVX2".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic, valid-looking K-quant weight: random bytes with the f16
    /// super-block scales forced small and finite.
    fn weight(qtype: u32, rows: usize, in_dim: usize, seed: u64) -> Vec<u8> {
        let (sb, d_off) = match qtype {
            GGUF_TYPE_Q4_K => (Q4K_BYTES, vec![0, 2]),
            GGUF_TYPE_Q5_K => (Q5K_BYTES, vec![0, 2]),
            _ => (Q6K_BYTES, vec![208]),
        };
        let nsb = in_dim.div_ceil(QK_K);
        let mut s = seed;
        let mut bytes: Vec<u8> = (0..rows * nsb * sb)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                (s >> 33) as u8
            })
            .collect();
        let d = half::f16::from_f32(0.01).to_bits().to_le_bytes();
        for blk in bytes.chunks_exact_mut(sb) {
            for &o in &d_off {
                blk[o..o + 2].copy_from_slice(&d);
            }
        }
        bytes
    }

    fn acts(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(7);
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn one_row(qtype: u32, w: &[u8], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut y = vec![0.0; out_dim];
        match qtype {
            GGUF_TYPE_Q4_K => {
                super::super::fused_q4k_parallel_matvec_into(w, x, in_dim, out_dim, &mut y)
            },
            GGUF_TYPE_Q5_K => {
                super::super::fused_q5k_parallel_matvec_into(w, x, in_dim, out_dim, &mut y)
            },
            _ => super::super::fused_q6k_parallel_matvec_into(w, x, in_dim, out_dim, &mut y),
        }
        .expect("one-row matvec");
        y
    }

    /// Every (token, row) output is bitwise the one-row kernel's, for widths that
    /// straddle TOKEN_TILE and ROW_TILE and an in_dim that needs padding.
    #[test]
    fn multi_row_is_bitwise_the_one_row_path_per_token() {
        for qtype in [GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K] {
            if !multi_row_supported(qtype) {
                continue;
            }
            for (in_dim, out_dim) in [(512, 300), (768 + 128, 65)] {
                let w = weight(qtype, out_dim, in_dim, 11 + u64::from(qtype));
                for t in [1, 7, 8, 9, 17] {
                    let x = acts(t * in_dim, 3 + t as u64);
                    let mut y = vec![f32::NAN; t * out_dim];
                    assert!(
                        fused_k_rows_matmul_into(qtype, &w, &x, t, in_dim, out_dim, &mut y)
                            .expect("multi-row"),
                        "qtype {qtype} must be handled"
                    );
                    for r in 0..t {
                        let want =
                            one_row(qtype, &w, &x[r * in_dim..(r + 1) * in_dim], in_dim, out_dim);
                        let got = &y[r * out_dim..(r + 1) * out_dim];
                        let same = want
                            .iter()
                            .zip(got)
                            .all(|(a, b)| a.to_bits() == b.to_bits());
                        assert!(
                            same,
                            "qtype {qtype} in {in_dim} out {out_dim} t {t} token {r}"
                        );
                        assert!(want.iter().any(|v| *v != 0.0), "vacuous: all-zero output");
                    }
                }
            }
        }
    }

    #[test]
    fn unsupported_qtype_is_declined_and_output_untouched() {
        let mut y = vec![1.5f32; 4];
        let done = fused_k_rows_matmul_into(
            crate::gguf::GGUF_TYPE_Q8_0,
            &[],
            &[0.0; 64],
            2,
            32,
            2,
            &mut y,
        )
        .expect("decline is not an error");
        assert!(!done);
        assert!(y.iter().all(|v| *v == 1.5));
    }
}
