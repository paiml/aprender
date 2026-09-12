//! Batched (multi-row) fused matmul for K-quantized weights — the CPU prefill GEMM.
//!
//! # Why this exists (#2787)
//!
//! Every CPU matmul in the quantized path was a **matvec**: one activation
//! vector against the whole weight tensor. Prefill therefore ran the decode
//! kernel once per prompt token, and a 7B Q4_K_M model streams ~4.4 GB of
//! weights from RAM on every one of those calls. Measured on `lambda`
//! (48-core x86_64, `qwen2.5-coder-7b-instruct-q4_k_m.gguf`, 513-token prompt),
//! prefill came out at the decode rate — i.e. prefill *was* decode.
//!
//! THE TWO RATES AND THEIR RATIO ARE IN THE RECEIPT, NOT HERE. They are the
//! `prefill_tok_s`, `decode_tok_s` and `ratio` columns of
//! `evidence/perf-2787/baseline-origin-main.csv` (row `rep=0`, taken on
//! origin/main a866988e4, before this change); the host, model, prompt shape
//! and the load the box was under are in `evidence/perf-2787/provenance.txt`.
//! A figure typed into a doc comment is a claim a `cargo doc` reader takes as a
//! measurement, and nothing can re-derive it or notice it going stale — which
//! is what `scripts/check_no_claim_literals.sh` bans. Citing the receipt keeps
//! the measurement and drops the unbacked literal.
//!
//! llama.cpp does not have this shape on the same class of box, because it
//! batches its prefill — which is what this module makes apr do. No llama.cpp
//! figure is quoted here: this branch measured none, and importing a
//! third-party number would be the same defect with a better provenance story.
//!
//! # What it does
//!
//! `n_rows` activation vectors are multiplied against the same weight tensor in
//! one pass. The parallel axis stays the OUTPUT dimension — exactly as in
//! [`generic_parallel_matvec_into`](super::generic_matvec::generic_parallel_matvec_into) —
//! so each rayon task loads a weight row once and reuses it across all `n_rows`
//! activation vectors. Weight bytes streamed from RAM drop by a factor of
//! `n_rows`; the arithmetic is unchanged.
//!
//! # Numerics: identical, not merely close
//!
//! The inner product is the SAME per-row dot function the matvec path calls,
//! given the SAME row bytes and the SAME activation vector. No accumulation
//! order changes, no new SIMD kernel, no re-association. Element `(s, o)` of
//! the batched result is bit-for-bit the element the matvec path produces for
//! activation row `s`. `falsify_batched_prefill.rs` asserts that bitwise.
//!
//! # Layout
//!
//! Activations and outputs are both **row-major**: `activations[s * in_dim + i]`,
//! `output[s * out_dim + o]`. Row-major output means a rayon task owning a set
//! of output columns writes a strided set of slots, which `slice::chunks_mut`
//! cannot express — hence the one raw-pointer scatter below, whose disjointness
//! argument is spelled out at the `unsafe` block.

use super::format_trait::{QuantBlockFormat, Q5K, Q6K};
use super::generic_matvec::FusedDotFn;
use crate::error::{RealizarError, Result};
use crate::gguf::{GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K};

/// TCB-style midi-tile: output rows per rayon task (matches the matvec path).
const MIDI_TILE_M: usize = 64;

/// Below this many output rows the rayon dispatch costs more than it saves
/// (PAR-126, same constant as the matvec path).
const PARALLEL_THRESHOLD: usize = 256;

/// A `*mut f32` that rayon may move into a task.
///
/// Only ever used to write slots that the surrounding loop proves are disjoint
/// across tasks; see the SAFETY comment at each use.
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);

// SAFETY: `OutPtr` carries no ownership and grants no shared access on its own.
// Every task that receives one writes only slots `s * out_dim + o` for output
// columns `o` in its own midi-tile, and midi-tiles partition `0..out_dim`, so no
// two tasks ever address the same slot. Reads never happen through it.
unsafe impl Send for OutPtr {}
// SAFETY: see the `Send` impl — the pointer is only ever used for disjoint writes.
unsafe impl Sync for OutPtr {}

/// Shape/bounds validation shared by both batched kernels.
fn validate(
    weight_len: usize,
    bytes_per_row: usize,
    acts_len: usize,
    in_dim: usize,
    out_dim: usize,
    n_rows: usize,
    out_len: usize,
    fmt: &str,
) -> Result<()> {
    if n_rows == 0 {
        return Err(RealizarError::InvalidShape {
            reason: format!("{fmt} batched matmul: n_rows must be > 0"),
        });
    }
    let need_w = out_dim * bytes_per_row;
    if weight_len < need_w {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "{fmt} weight data too small: need {need_w} bytes for {out_dim}x{in_dim}, have {weight_len}"
            ),
        });
    }
    if acts_len != n_rows * in_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "{fmt} batched matmul: activations {acts_len} != n_rows*in_dim = {}",
                n_rows * in_dim
            ),
        });
    }
    if out_len < n_rows * out_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "{fmt} batched matmul: output {out_len} < n_rows*out_dim = {}",
                n_rows * out_dim
            ),
        });
    }
    Ok(())
}

/// Pad each activation row to the super-block boundary, producing one flat
/// `n_rows * padded_in_dim` buffer.
///
/// GH-202: the fused dot kernels read a whole super-block, so a row whose
/// `in_dim` is not a multiple of `ELEMENTS_PER_SUPERBLOCK` must be zero-filled
/// to that boundary — the same padding the matvec path applies.
fn pad_rows(activations: &[f32], in_dim: usize, padded_in_dim: usize, n_rows: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_rows * padded_in_dim];
    for s in 0..n_rows {
        out[s * padded_in_dim..s * padded_in_dim + in_dim]
            .copy_from_slice(&activations[s * in_dim..(s + 1) * in_dim]);
    }
    out
}

/// Generic batched fused matmul for any blocked quantization format, using the
/// format's f32-activation dot kernel.
///
/// This is [`generic_parallel_matvec_into`](super::generic_matvec::generic_parallel_matvec_into)
/// with an inner loop over `n_rows` activation vectors. The dot function, its
/// arguments and its accumulation order are unchanged.
///
/// # Errors
///
/// Returns an error when the weight buffer is too small for `out_dim × in_dim`,
/// when `activations.len() != n_rows * in_dim`, or when `output` is shorter than
/// `n_rows * out_dim`.
pub fn generic_parallel_matmul_into<F: QuantBlockFormat>(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    n_rows: usize,
    output: &mut [f32],
    dot_fn: FusedDotFn,
) -> Result<()> {
    let super_blocks_per_row = in_dim.div_ceil(F::ELEMENTS_PER_SUPERBLOCK);
    let bytes_per_row = super_blocks_per_row * F::SUPERBLOCK_BYTES;
    validate(
        weight_data.len(),
        bytes_per_row,
        activations.len(),
        in_dim,
        out_dim,
        n_rows,
        output.len(),
        F::FORMAT_ID,
    )?;

    let padded_in_dim = super_blocks_per_row * F::ELEMENTS_PER_SUPERBLOCK;
    let acts = pad_rows(activations, in_dim, padded_in_dim, n_rows);

    let run_tile = |tile_start: usize, tile_len: usize, out_ptr: OutPtr| {
        for local in 0..tile_len {
            let row = tile_start + local;
            let row_data = &weight_data[row * bytes_per_row..(row + 1) * bytes_per_row];
            for s in 0..n_rows {
                let a = &acts[s * padded_in_dim..(s + 1) * padded_in_dim];
                let v = dot_fn(row_data, a).unwrap_or(0.0);
                // SAFETY: `row` lies in this task's midi-tile and midi-tiles
                // partition `0..out_dim`, so slot `s * out_dim + row` is written
                // by this task alone. `s < n_rows` and `row < out_dim`, and
                // `validate` proved `output.len() >= n_rows * out_dim`, so the
                // index is in bounds of the allocation `out_ptr` came from.
                unsafe { *out_ptr.0.add(s * out_dim + row) = v };
            }
        }
    };

    let out_ptr = OutPtr(output.as_mut_ptr());
    if out_dim < PARALLEL_THRESHOLD {
        run_tile(0, out_dim, out_ptr);
    } else {
        use rayon::prelude::*;
        let n_tiles = out_dim.div_ceil(MIDI_TILE_M);
        (0..n_tiles).into_par_iter().for_each(|t| {
            let start = t * MIDI_TILE_M;
            let len = MIDI_TILE_M.min(out_dim - start);
            run_tile(start, len, out_ptr);
        });
    }
    Ok(())
}

/// Batched Q4_K × Q8_K matmul: quantize each activation row to Q8_K once, then
/// reuse every weight row across all `n_rows` columns.
///
/// Mirrors `fused_q4k_parallel_matvec_into`'s Q8_K path exactly — same
/// `quantize_activations_q8k_into`, same `precompute_q8k_bsums`, same
/// `fused_q4k_q8k_dot_with_bsums_simd` — so element `(s, o)` equals what that
/// function writes for activation row `s`.
///
/// # Errors
///
/// Returns an error on the shape conditions listed by
/// [`generic_parallel_matmul_into`], or when Q8_K activation quantization fails.
pub fn fused_q4k_parallel_matmul_into(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    n_rows: usize,
    output: &mut [f32],
) -> Result<()> {
    const SB_BYTES: usize = 144;
    const QK_K: usize = 256;

    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * SB_BYTES;
    validate(
        weight_data.len(),
        bytes_per_row,
        activations.len(),
        in_dim,
        out_dim,
        n_rows,
        output.len(),
        "Q4_K",
    )?;

    let padded_in_dim = super_blocks_per_row * QK_K;
    let acts = pad_rows(activations, in_dim, padded_in_dim, n_rows);

    // One Q8_K quantization per activation row, hoisted out of the weight loop.
    // Same `quantize_activations_q8k_into` the matvec path calls, on the same
    // (padded) activation vector, so the quantized column for row `s` is
    // bit-identical to the one the matvec path builds for that token.
    let mut scales = vec![0.0f32; n_rows * super_blocks_per_row];
    let mut quants = vec![0i8; n_rows * padded_in_dim];
    for s in 0..n_rows {
        super::quantize_activations_q8k_into(
            &acts[s * padded_in_dim..(s + 1) * padded_in_dim],
            &mut scales[s * super_blocks_per_row..(s + 1) * super_blocks_per_row],
            &mut quants[s * padded_in_dim..(s + 1) * padded_in_dim],
        )?;
    }

    // Mirror `fused_q4k_q8k_parallel_matvec_into`'s dispatch (q5k_q6k_matvec.rs).
    // The row kernel MUST be the same one the matvec path picks on this CPU or
    // the batched result is merely close, not equal.
    #[cfg(target_arch = "x86_64")]
    let use_lean = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_lean = false;

    #[cfg(target_arch = "x86_64")]
    if use_lean {
        // SAFETY: AVX2 was detected above; `quants` holds `n_rows` blocks of
        // `super_blocks_per_row * 256` i8 values, which is the length
        // `precompute_q8k_bsums_i16` reads for `super_blocks_per_row` blocks.
        let bsums: Vec<i16> = (0..n_rows)
            .flat_map(|s| unsafe {
                super::fused_k::precompute_q8k_bsums_i16(
                    &quants[s * padded_in_dim..(s + 1) * padded_in_dim],
                    super_blocks_per_row,
                )
            })
            .collect();
        let bs_stride = super_blocks_per_row * 16;

        let w_addr = weight_data.as_ptr() as usize;
        let sc_addr = scales.as_ptr() as usize;
        let qq_addr = quants.as_ptr() as usize;
        let bs_addr = bsums.as_ptr() as usize;
        let out_ptr = OutPtr(output.as_mut_ptr());

        let run_tile = move |tile_start: usize, tile_len: usize| {
            // Bind the whole struct: edition-2021 disjoint capture would
            // otherwise capture the bare `*mut f32` field, which is not `Sync`.
            let out_ptr = out_ptr;
            // SAFETY: the four base addresses were captured as `usize` to cross
            // the rayon closure boundary and are rebuilt here as `*const`. Every
            // offset is inside the buffer it came from: `row < out_dim` and the
            // weight buffer holds `out_dim * bytes_per_row`; `s < n_rows` and the
            // scale/quant/bsum buffers each hold `n_rows` strides. AVX2+FMA were
            // detected above, which is what `ggml_style_q4k_q8k_dot_avx2_raw`
            // requires. Writes go to `s * out_dim + row`, and midi-tiles
            // partition `0..out_dim`, so tasks never share a slot.
            unsafe {
                let w = w_addr as *const u8;
                let sc = sc_addr as *const f32;
                let qq = qq_addr as *const i8;
                let bs = bs_addr as *const i16;
                for local in 0..tile_len {
                    let row = tile_start + local;
                    let w_row = w.add(row * bytes_per_row);
                    for s in 0..n_rows {
                        let v = super::fused_k::ggml_style_q4k_q8k_dot_avx2_raw(
                            w_row,
                            sc.add(s * super_blocks_per_row),
                            qq.add(s * padded_in_dim),
                            bs.add(s * bs_stride),
                            super_blocks_per_row,
                        );
                        *out_ptr.0.add(s * out_dim + row) = v;
                    }
                }
            }
        };

        if out_dim < PARALLEL_THRESHOLD {
            run_tile(0, out_dim);
        } else {
            use rayon::prelude::*;
            let n_tiles = out_dim.div_ceil(MIDI_TILE_M);
            (0..n_tiles).into_par_iter().for_each(|t| {
                let start = t * MIDI_TILE_M;
                run_tile(start, MIDI_TILE_M.min(out_dim - start));
            });
        }
        return Ok(());
    }

    // Portable path: same `fused_q4k_q8k_dot_with_bsums_simd` / `fused_q4k_q8k_dot_simd`
    // pair the matvec path falls back to when the lean dispatch is unavailable.
    let bsums: Vec<Option<Vec<[i32; 8]>>> = (0..n_rows)
        .map(|s| {
            super::precompute_q8k_bsums(
                &quants[s * padded_in_dim..(s + 1) * padded_in_dim],
                super_blocks_per_row,
            )
            .ok()
        })
        .collect();

    let run_tile = |tile_start: usize, tile_len: usize, out_ptr: OutPtr| {
        for local in 0..tile_len {
            let row = tile_start + local;
            let row_data = &weight_data[row * bytes_per_row..(row + 1) * bytes_per_row];
            for s in 0..n_rows {
                let sc = &scales[s * super_blocks_per_row..(s + 1) * super_blocks_per_row];
                let qq = &quants[s * padded_in_dim..(s + 1) * padded_in_dim];
                let v = match bsums[s] {
                    Some(ref bs) => super::bsum_precompute::fused_q4k_q8k_dot_with_bsums_simd(
                        row_data, sc, qq, bs,
                    ),
                    None => super::fused_k::fused_q4k_q8k_dot_simd(row_data, sc, qq),
                }
                .unwrap_or(0.0);
                // SAFETY: identical disjointness argument to
                // `generic_parallel_matmul_into` — midi-tiles partition
                // `0..out_dim`, `s < n_rows`, and `validate` proved the buffer
                // holds `n_rows * out_dim` elements.
                unsafe { *out_ptr.0.add(s * out_dim + row) = v };
            }
        }
    };

    let out_ptr = OutPtr(output.as_mut_ptr());
    if out_dim < PARALLEL_THRESHOLD {
        run_tile(0, out_dim, out_ptr);
    } else {
        use rayon::prelude::*;
        let n_tiles = out_dim.div_ceil(MIDI_TILE_M);
        (0..n_tiles).into_par_iter().for_each(|t| {
            let start = t * MIDI_TILE_M;
            let len = MIDI_TILE_M.min(out_dim - start);
            run_tile(start, len, out_ptr);
        });
    }
    Ok(())
}

/// True when [`quantized_matmul_batch_into`] has a batched kernel for `qtype`.
///
/// Deliberately narrow: a caller that cannot batch must fall back to the
/// per-token loop rather than silently take a different numerical path.
#[must_use]
pub fn batched_matmul_supports(qtype: u32) -> bool {
    matches!(qtype, GGUF_TYPE_Q4_K | GGUF_TYPE_Q5_K | GGUF_TYPE_Q6_K)
}

/// Dispatch a batched matmul by GGUF quantization type.
///
/// `activations` is `n_rows × in_dim` row-major; `output` is `n_rows × out_dim`
/// row-major.
///
/// # Errors
///
/// Returns [`RealizarError::UnsupportedOperation`] when `qtype` has no batched
/// kernel (check with [`batched_matmul_supports`] first), or the shape errors
/// listed by [`generic_parallel_matmul_into`].
pub fn quantized_matmul_batch_into(
    weight_data: &[u8],
    qtype: u32,
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    n_rows: usize,
    output: &mut [f32],
) -> Result<()> {
    match qtype {
        GGUF_TYPE_Q4_K => fused_q4k_parallel_matmul_into(
            weight_data,
            activations,
            in_dim,
            out_dim,
            n_rows,
            output,
        ),
        GGUF_TYPE_Q5_K => generic_parallel_matmul_into::<Q5K>(
            weight_data,
            activations,
            in_dim,
            out_dim,
            n_rows,
            output,
            super::fused_q5k_q6k::fused_q5k_dot_simd,
        ),
        GGUF_TYPE_Q6_K => generic_parallel_matmul_into::<Q6K>(
            weight_data,
            activations,
            in_dim,
            out_dim,
            n_rows,
            output,
            super::fused_q5k_q6k::fused_q6k_dot_simd,
        ),
        _ => Err(RealizarError::UnsupportedOperation {
            operation: "quantized_matmul_batch_into".to_string(),
            reason: format!("no batched CPU kernel for GGUF qtype {qtype}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::{
        encode::{quantize_q4_k, quantize_q5_k, quantize_q6_k},
        fused_q4k_parallel_matvec_into, fused_q5k_parallel_matvec_into,
        fused_q6k_parallel_matvec_into,
    };

    /// Deterministic pseudo-random floats — no rand dependency, no flake.
    fn prng(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                #[allow(clippy::cast_precision_loss)]
                let u = (x >> 40) as f32 / f32::from(1u16 << 12);
                u - 2.0
            })
            .collect()
    }

    /// THE FALSIFIER (#2787).
    ///
    /// The batched prefill exists to make prefill faster WITHOUT changing what
    /// the model computes. This asserts the strongest form of that: element
    /// `(s, o)` of the batched result is **bit-identical** to the element the
    /// per-token matvec — the function the old prefill loop called — produces
    /// for activation row `s`.
    ///
    /// Revert `batched_matmul.rs` (or break the `s * out_dim + row` scatter, or
    /// point every column at row 0's activations) and this goes RED.
    fn assert_batched_equals_matvec_bitwise(qtype: u32, in_dim: usize, out_dim: usize, n: usize) {
        let w_f32 = prng(
            in_dim * out_dim,
            0x51ED_u64.wrapping_mul(u64::from(qtype) + 1),
        );
        let weight = match qtype {
            GGUF_TYPE_Q4_K => quantize_q4_k(&w_f32),
            GGUF_TYPE_Q5_K => quantize_q5_k(&w_f32),
            GGUF_TYPE_Q6_K => quantize_q6_k(&w_f32),
            _ => unreachable!("test covers only the batched K-quant formats"),
        };
        let acts = prng(n * in_dim, 0xAC7_u64.wrapping_mul(u64::from(qtype) + 1));

        let mut batched = vec![0.0f32; n * out_dim];
        quantized_matmul_batch_into(&weight, qtype, &acts, in_dim, out_dim, n, &mut batched)
            .expect("batched matmul must succeed on valid shapes");

        for s in 0..n {
            let mut reference = vec![0.0f32; out_dim];
            let row = &acts[s * in_dim..(s + 1) * in_dim];
            match qtype {
                GGUF_TYPE_Q4_K => {
                    fused_q4k_parallel_matvec_into(&weight, row, in_dim, out_dim, &mut reference)
                },
                GGUF_TYPE_Q5_K => {
                    fused_q5k_parallel_matvec_into(&weight, row, in_dim, out_dim, &mut reference)
                },
                GGUF_TYPE_Q6_K => {
                    fused_q6k_parallel_matvec_into(&weight, row, in_dim, out_dim, &mut reference)
                },
                _ => unreachable!("test covers only the batched K-quant formats"),
            }
            .expect("reference matvec must succeed on valid shapes");

            for o in 0..out_dim {
                let got = batched[s * out_dim + o];
                let want = reference[o];
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "qtype {qtype} row {s} col {o}: batched {got} != matvec {want}"
                );
            }
        }
    }

    #[test]
    fn falsify_q4k_batched_equals_matvec_bitwise_parallel() {
        // out_dim above PARALLEL_THRESHOLD → rayon midi-tile path.
        assert_batched_equals_matvec_bitwise(GGUF_TYPE_Q4_K, 512, 512, 5);
    }

    #[test]
    fn falsify_q4k_batched_equals_matvec_bitwise_sequential() {
        // out_dim below PARALLEL_THRESHOLD → sequential path.
        assert_batched_equals_matvec_bitwise(GGUF_TYPE_Q4_K, 256, 128, 4);
    }

    #[test]
    fn falsify_q5k_batched_equals_matvec_bitwise() {
        assert_batched_equals_matvec_bitwise(GGUF_TYPE_Q5_K, 512, 320, 3);
    }

    #[test]
    fn falsify_q6k_batched_equals_matvec_bitwise() {
        assert_batched_equals_matvec_bitwise(GGUF_TYPE_Q6_K, 512, 320, 3);
    }

    /// DISCRIMINATION CASE — must stay GREEN when the falsifier is RED.
    ///
    /// The equality assertion above would also pass if every activation row
    /// produced the same output (e.g. a kernel that ignores `s`), because the
    /// reference would then be compared against itself. This proves the rows
    /// carry different data, so the equality above is a real constraint and not
    /// a tautology.
    #[test]
    fn discrimination_batched_rows_are_distinct() {
        let (in_dim, out_dim, n) = (512usize, 512usize, 3usize);
        let weight = quantize_q4_k(&prng(in_dim * out_dim, 0x51ED));
        let acts = prng(n * in_dim, 0xAC7);
        let mut batched = vec![0.0f32; n * out_dim];
        quantized_matmul_batch_into(
            &weight,
            GGUF_TYPE_Q4_K,
            &acts,
            in_dim,
            out_dim,
            n,
            &mut batched,
        )
        .expect("batched matmul must succeed on valid shapes");

        for s in 1..n {
            let a = &batched[..out_dim];
            let b = &batched[s * out_dim..(s + 1) * out_dim];
            assert_ne!(a, b, "rows 0 and {s} are identical — the inputs did not differ, so the equality falsifier would be vacuous");
        }
        assert!(
            batched.iter().any(|v| *v != 0.0),
            "all outputs zero — the falsifier would pass against a kernel that writes nothing"
        );
    }

    #[test]
    fn unsupported_qtype_is_an_error_not_a_wrong_answer() {
        assert!(!batched_matmul_supports(crate::gguf::GGUF_TYPE_Q4_0));
        let mut out = vec![0.0f32; 8];
        let err = quantized_matmul_batch_into(
            &[0u8; 4096],
            crate::gguf::GGUF_TYPE_Q4_0,
            &[0.0f32; 512],
            256,
            2,
            2,
            &mut out,
        );
        assert!(err.is_err(), "unsupported qtype must refuse, not guess");
    }

    #[test]
    fn shape_violations_are_rejected() {
        let weight = quantize_q4_k(&prng(256 * 64, 1));
        let mut out = vec![0.0f32; 2 * 64];
        // activations too short for n_rows * in_dim
        assert!(quantized_matmul_batch_into(
            &weight,
            GGUF_TYPE_Q4_K,
            &prng(256, 2),
            256,
            64,
            2,
            &mut out
        )
        .is_err());
        // output too small
        let mut small = vec![0.0f32; 64];
        assert!(quantized_matmul_batch_into(
            &weight,
            GGUF_TYPE_Q4_K,
            &prng(512, 2),
            256,
            64,
            2,
            &mut small
        )
        .is_err());
        // n_rows == 0
        assert!(
            quantized_matmul_batch_into(&weight, GGUF_TYPE_Q4_K, &[], 256, 64, 0, &mut out)
                .is_err()
        );
    }
}
