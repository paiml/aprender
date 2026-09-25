// Multi-row Q4_K × Q8_K GEMM (#4228, CRUX #4223 technique T1).
//
// Included into parallel_k.rs, so it shares that module's imports.

/// L2 budget for one weight tile. A tile of `tile_rows × bytes_per_row` Q4_K
/// bytes is streamed from DRAM once and then reused for every activation row.
/// 256 KiB leaves room for the Q8_K activations on a 1 MiB (Zen 4) L2.
const MULTIROW_L2_TILE_BYTES: usize = 256 * 1024;

/// Rows per weight tile: an L2-sized tile, clamped to [4, 64] and kept a
/// multiple of 4.
fn multirow_tile_rows(bytes_per_row: usize) -> usize {
    let rows = (MULTIROW_L2_TILE_BYTES / bytes_per_row.max(1)).clamp(4, 64);
    rows - rows % 4
}

/// Multiply `m` Q8_K activation rows by one Q4_K weight matrix. The Q4_K
/// weights are row-major (LAYOUT-001/002).
///
/// This is the batched-prefill counterpart of
/// [`fused_q4k_q8k_parallel_matvec_into`]. The per-token matvec re-streams
/// the whole weight matrix from DRAM for every prompt token. This kernel
/// splits the weights into L2-resident tiles and applies each tile to all
/// `m` activation rows before it moves on, so each weight byte is read from
/// DRAM once per call rather than once per token (llamafile
/// `iqk_mul_mat.inc` `mul_mat_qX_K_q8_K_T`).
///
/// Every (row, token) dot goes through the same per-row kernel the matvec
/// uses, so the result is **bit-identical** to calling
/// [`fused_q4k_q8k_parallel_matvec_into`] once per token.
///
/// # Arguments
///
/// * `q8k_scales`: `m × in_dim/256` scales, token-major.
/// * `q8k_quants`: `m × in_dim` quants, token-major.
/// * `output`: `m × out_dim`, token-major (`output[t * out_dim + row]`).
///
/// # Errors
///
/// Returns an error if `in_dim` is not a multiple of 256 (Q8_K quantizes
/// activations in 256-wide super-blocks), or if any buffer is too small.
pub fn fused_q4k_q8k_multirow_matmul_into(
    weight_data: &[u8],
    q8k_scales: &[f32],
    q8k_quants: &[i8],
    m: usize,
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    const SUPER_BLOCK_BYTES: usize = 144;

    if in_dim % QK_K != 0 {
        return Err(RealizarError::InvalidShape {
            reason: format!("multirow Q4_K×Q8_K: in_dim {in_dim} is not a multiple of {QK_K}"),
        });
    }
    let nsb = in_dim / QK_K;
    let bytes_per_row = nsb * SUPER_BLOCK_BYTES;
    let need = |what: &str, need: usize, have: usize| -> Result<()> {
        if have < need {
            return Err(RealizarError::InvalidShape {
                reason: format!("multirow Q4_K×Q8_K: {what} too small: need {need}, have {have}"),
            });
        }
        Ok(())
    };
    need("weight data", out_dim * bytes_per_row, weight_data.len())?;
    need("q8k scales", m * nsb, q8k_scales.len())?;
    need("q8k quants", m * in_dim, q8k_quants.len())?;
    need("output", m * out_dim, output.len())?;

    if m == 0 || out_dim == 0 {
        return Ok(());
    }

    #[cfg(target_arch = "x86_64")]
    {
        if m > 1 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            use rayon::prelude::*;

            // Per-token i16 bsums, the same values the matvec's lean path precomputes.
            let mut bsums = Vec::with_capacity(m * nsb * 16);
            for t in 0..m {
                // SAFETY: avx2 was detected just above; the slice is exactly one token's
                // `in_dim` quants, which the length check above guarantees.
                bsums.extend(unsafe {
                    super::fused_k::precompute_q8k_bsums_i16(
                        &q8k_quants[t * in_dim..(t + 1) * in_dim],
                        nsb,
                    )
                });
            }

            let tile_rows = multirow_tile_rows(bytes_per_row);
            let n_tiles = out_dim.div_ceil(tile_rows);
            let w_addr = weight_data.as_ptr() as usize;
            let sc_addr = q8k_scales.as_ptr() as usize;
            let qq_addr = q8k_quants.as_ptr() as usize;
            let bs_addr = bsums.as_ptr() as usize;
            let out_addr = output.as_mut_ptr() as usize;

            (0..n_tiles).into_par_iter().for_each(|tile| {
                let row0 = tile * tile_rows;
                let rows = tile_rows.min(out_dim - row0);
                // SAFETY: avx2+fma were detected before dispatch. Every pointer below
                // is a buffer base captured as `usize` and offset within the bounds
                // validated above: weight rows < out_dim, token t < m. Each tile writes
                // only `output[t * out_dim + row0 .. + rows]` for its own row range, so
                // no two rayon tasks touch the same element. `bsums` and every input
                // outlive the parallel region.
                unsafe {
                    let w = w_addr as *const u8;
                    let out = out_addr as *mut f32;
                    // Token-outer: the tile's weights stay in L2 across all m tokens.
                    for t in 0..m {
                        let sc = (sc_addr as *const f32).add(t * nsb);
                        let qq = (qq_addr as *const i8).add(t * in_dim);
                        let bs = (bs_addr as *const i16).add(t * nsb * 16);
                        let dst = out.add(t * out_dim + row0);
                        for i in 0..rows {
                            *dst.add(i) = super::fused_k::ggml_style_q4k_q8k_dot_avx2_raw(
                                w.add((row0 + i) * bytes_per_row),
                                sc,
                                qq,
                                bs,
                                nsb,
                            );
                        }
                    }
                }
            });
            return Ok(());
        }
    }

    // m == 1, or no AVX2: the per-token matvec is the definition of the result.
    for t in 0..m {
        fused_q4k_q8k_parallel_matvec_into(
            weight_data,
            &q8k_scales[t * nsb..(t + 1) * nsb],
            &q8k_quants[t * in_dim..(t + 1) * in_dim],
            in_dim,
            out_dim,
            &mut output[t * out_dim..(t + 1) * out_dim],
        )?;
    }
    Ok(())
}

/// f32-activation multi-row Q4_K matmul: `input` is `m` token rows of `in_dim`, `output` is `m`
/// rows of `out_dim` (token-major).
///
/// Each row is quantized to Q8_K exactly as [`fused_q4k_parallel_matvec_into`] quantizes its
/// single row, so the result is bit-identical to calling that function once per row (#4228).
/// Falls back to that per-row call when `m == 1`, `in_dim` is not a multiple of 256, or
/// `DIRECT_FP32_GEMV=1` selects the f32 path.
///
/// # Errors
/// Mis-sized buffers, or a failure in the underlying matvec.
pub fn fused_q4k_multirow_matmul_f32_into(
    weight_data: &[u8],
    input: &[f32],
    m: usize,
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    if input.len() != m * in_dim || output.len() < m * out_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4K multirow f32: input {} != {m}x{in_dim} or output {} < {m}x{out_dim}",
                input.len(),
                output.len()
            ),
        });
    }
    let direct_fp32 = std::env::var("DIRECT_FP32_GEMV").as_deref() == Ok("1");
    if m <= 1 || in_dim % QK_K != 0 || direct_fp32 {
        for t in 0..m {
            fused_q4k_parallel_matvec_into(
                weight_data,
                &input[t * in_dim..(t + 1) * in_dim],
                in_dim,
                out_dim,
                &mut output[t * out_dim..(t + 1) * out_dim],
            )?;
        }
        return Ok(());
    }
    let nsb = in_dim / QK_K;
    let mut scales = vec![0.0f32; m * nsb];
    let mut quants = vec![0i8; m * in_dim];
    for t in 0..m {
        super::quantize_activations_q8k_into(
            &input[t * in_dim..(t + 1) * in_dim],
            &mut scales[t * nsb..(t + 1) * nsb],
            &mut quants[t * in_dim..(t + 1) * in_dim],
        )?;
    }
    fused_q4k_q8k_multirow_matmul_into(
        weight_data,
        &scales,
        &quants,
        m,
        in_dim,
        out_dim,
        &mut output[..m * out_dim],
    )
}

const Q5K_SB_BYTES: usize = 176;
/// Tokens per SIMD accumulator group in the Q5_K multi-row kernel.
const Q5K_LANES: usize = 8;

/// Token-major `input` (`m` rows of `in_dim`) into `groups` lane blocks:
/// `act_t[g][i][lane]`. Tail lanes stay zero; their results are discarded.
fn transpose_token_lanes(input: &[f32], in_dim: usize, groups: usize) -> Vec<f32> {
    let mut act_t = vec![0.0f32; groups * in_dim * Q5K_LANES];
    for (t, row) in input.chunks_exact(in_dim).enumerate() {
        let base = (t / Q5K_LANES) * in_dim * Q5K_LANES + t % Q5K_LANES;
        for (i, &x) in row.iter().enumerate() {
            act_t[base + i * Q5K_LANES] = x;
        }
    }
    act_t
}

/// One Q5_K weight row against every token: dequantize the row into `w` once, then
/// replay `fused_q5k_dot`'s mul-then-add order per lane. `out_r[t]` is token `t`'s output.
fn q5k_row_all_tokens(row: &[u8], act_t: &[f32], w: &mut [f32], out_r: &mut [f32]) {
    let in_dim = w.len();
    for (sb_i, sb) in row.as_chunks::<Q5K_SB_BYTES>().0.iter().enumerate() {
        let wb = &mut w[sb_i * QK_K..(sb_i + 1) * QK_K];
        super::dequant::for_each_q5k_value(sb, |i, v| wb[i] = v);
    }
    let m = out_r.len();
    for (g, a) in act_t.chunks_exact(in_dim * Q5K_LANES).enumerate() {
        let mut acc = [0.0f32; Q5K_LANES];
        for (wi, ai) in w.iter().zip(a.as_chunks::<Q5K_LANES>().0) {
            for l in 0..Q5K_LANES {
                acc[l] += *wi * ai[l];
            }
        }
        let n = Q5K_LANES.min(m - g * Q5K_LANES);
        out_r[g * Q5K_LANES..g * Q5K_LANES + n].copy_from_slice(&acc[..n]);
    }
}

/// Multi-row Q5_K matmul over `m` token rows, bit-identical to [`fused_q5k_parallel_matvec_into`]
/// once per row (#4228). Token-major `input`/`output`.
///
/// # Errors
/// Mis-sized buffers.
pub fn fused_q5k_multirow_matmul_into(
    weight_data: &[u8],
    input: &[f32],
    m: usize,
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    use rayon::prelude::*;
    const SB_BYTES: usize = Q5K_SB_BYTES;

    // A padded tail would add w*0.0 terms whose sign can differ; keep the generic path there.
    if m <= 1 || in_dim % QK_K != 0 {
        return super::generic_matvec::generic_multirow_matmul_into::<Q5K>(
            weight_data,
            input,
            m,
            in_dim,
            out_dim,
            output,
            fused_q5k_dot_simd,
        );
    }
    let bytes_per_row = in_dim / QK_K * SB_BYTES;
    if weight_data.len() < out_dim * bytes_per_row
        || input.len() != m * in_dim
        || output.len() < m * out_dim
    {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q5_K multirow: weight {} < {out_dim}x{bytes_per_row}, input {} != {m}x{in_dim} \
                 or output {} < {m}x{out_dim}",
                weight_data.len(),
                input.len(),
                output.len()
            ),
        });
    }

    let groups = m.div_ceil(Q5K_LANES);
    let act_t = transpose_token_lanes(input, in_dim, groups);

    // `fused_q5k_dot` is `acc += v * act[i]` over i ascending, v from `for_each_q5k_value`
    // (which emits i = 0..256 in order per super-block). Each row is dequantized ONCE here and
    // every token replays that exact sequence in its own lane — separate mul and add, no FMA,
    // same order — so each output is bit-identical to the per-token dot.
    let rows_per_task = 8;
    let mut by_row = vec![0.0f32; out_dim * m];
    by_row
        .par_chunks_mut(rows_per_task * m)
        .enumerate()
        .for_each(|(ti, chunk)| {
            let mut w = vec![0.0f32; in_dim];
            for (r, out_r) in chunk.chunks_exact_mut(m).enumerate() {
                let at = (ti * rows_per_task + r) * bytes_per_row;
                q5k_row_all_tokens(&weight_data[at..at + bytes_per_row], &act_t, &mut w, out_r);
            }
        });
    for (row, vals) in by_row.chunks_exact(m).enumerate() {
        for (t, &v) in vals.iter().enumerate() {
            output[t * out_dim + row] = v;
        }
    }
    Ok(())
}

/// Multi-row Q6_K matmul over `m` token rows, bit-identical to [`fused_q6k_parallel_matvec_into`]
/// once per row (#4228). Token-major `input`/`output`.
///
/// # Errors
/// Mis-sized buffers.
pub fn fused_q6k_multirow_matmul_into(
    weight_data: &[u8],
    input: &[f32],
    m: usize,
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    super::generic_matvec::generic_multirow_matmul_into::<Q6K>(
        weight_data,
        input,
        m,
        in_dim,
        out_dim,
        output,
        fused_q6k_dot_simd,
    )
}

#[cfg(test)]
mod multirow_tests {
    use super::*;
    use crate::quantize::quantize_activations_q8k_into;

    /// Deterministic Q4_K weights with varied scale and quant bytes (LCG). d and
    /// dmin are fixed finite f16 values, so no super-block decodes to NaN or inf.
    fn q4k_weights(out_dim: usize, in_dim: usize, seed: u64) -> Vec<u8> {
        let nsb = in_dim / QK_K;
        let mut s = seed;
        let mut next = move || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (s >> 33) as u8
        };
        let mut w = vec![0u8; out_dim * nsb * 144];
        for sb in w.chunks_exact_mut(144) {
            sb[0..2].copy_from_slice(&0x2E66u16.to_le_bytes()); // d ≈ 0.1
            sb[2..4].copy_from_slice(&0x211Fu16.to_le_bytes()); // dmin ≈ 0.01
            for b in &mut sb[4..] {
                *b = next();
            }
        }
        w
    }

    fn q8k_rows(m: usize, in_dim: usize) -> (Vec<f32>, Vec<i8>) {
        let nsb = in_dim / QK_K;
        let mut scales = vec![0f32; m * nsb];
        let mut quants = vec![0i8; m * in_dim];
        for t in 0..m {
            let x: Vec<f32> = (0..in_dim)
                .map(|i| (((i * 7 + t * 13) % 29) as f32 - 14.0) * 0.07 + t as f32 * 0.01)
                .collect();
            quantize_activations_q8k_into(
                &x,
                &mut scales[t * nsb..(t + 1) * nsb],
                &mut quants[t * in_dim..(t + 1) * in_dim],
            )
            .expect("quantize q8k");
        }
        (scales, quants)
    }

    fn per_token(w: &[u8], sc: &[f32], qq: &[i8], m: usize, k: usize, n: usize) -> Vec<f32> {
        let nsb = k / QK_K;
        let mut out = vec![0f32; m * n];
        for t in 0..m {
            fused_q4k_q8k_parallel_matvec_into(
                w,
                &sc[t * nsb..(t + 1) * nsb],
                &qq[t * k..(t + 1) * k],
                k,
                n,
                &mut out[t * n..(t + 1) * n],
            )
            .expect("matvec");
        }
        out
    }

    /// FALSIFY-4228-001: multirow == per-token matvec, bit for bit. The shapes
    /// cover a partial tile, a single row, m = 1, and several tiles (out_dim 130 >
    /// 64). The two in_dims give tile_rows 64 and a multi-super-block row.
    #[test]
    fn falsify_4228_001_multirow_bit_identical_to_per_token_matvec() {
        for &(m, in_dim, out_dim) in &[
            (1, 256, 5),
            (2, 256, 1),
            (3, 512, 64),
            (8, 256, 130),
            (5, 1024, 67),
        ] {
            let w = q4k_weights(out_dim, in_dim, (m * 31 + out_dim) as u64);
            let (sc, qq) = q8k_rows(m, in_dim);
            let want = per_token(&w, &sc, &qq, m, in_dim, out_dim);
            let mut got = vec![f32::NAN; m * out_dim];
            fused_q4k_q8k_multirow_matmul_into(&w, &sc, &qq, m, in_dim, out_dim, &mut got)
                .expect("multirow");
            let diff = want
                .iter()
                .zip(&got)
                .position(|(a, b)| a.to_bits() != b.to_bits());
            assert_eq!(diff, None, "m={m} in={in_dim} out={out_dim}: first mismatch index");
            assert!(want.iter().any(|v| *v != 0.0), "fixture degenerate: all-zero output");
        }
    }

    /// FALSIFY-4228-002: a token's row lands at `output[t * out_dim ..]`. With
    /// two DIFFERENT token rows, a swapped or transposed store fails.
    #[test]
    fn falsify_4228_002_rows_are_token_major() {
        let (m, k, n) = (2, 256, 8);
        let w = q4k_weights(n, k, 7);
        let (sc, qq) = q8k_rows(m, k);
        let mut got = vec![0f32; m * n];
        fused_q4k_q8k_multirow_matmul_into(&w, &sc, &qq, m, k, n, &mut got).expect("multirow");
        let mut t1 = vec![0f32; n];
        fused_q4k_q8k_parallel_matvec_into(&w, &sc[1..2], &qq[k..2 * k], k, n, &mut t1)
            .expect("matvec");
        assert_eq!(&got[n..], &t1[..]);
        assert_ne!(&got[..n], &got[n..], "fixture degenerate: both tokens identical");
    }

    #[test]
    fn falsify_4228_003_rejects_bad_shapes() {
        let w = q4k_weights(2, 256, 1);
        let (sc, qq) = q8k_rows(2, 256);
        let mut out = vec![0f32; 4];
        // in_dim not a multiple of 256
        assert!(fused_q4k_q8k_multirow_matmul_into(&w, &sc, &qq, 2, 200, 2, &mut out).is_err());
        // weights too small for out_dim
        assert!(fused_q4k_q8k_multirow_matmul_into(&w, &sc, &qq, 2, 256, 3, &mut out).is_err());
        // activations too small for m
        assert!(fused_q4k_q8k_multirow_matmul_into(&w, &sc, &qq, 3, 256, 1, &mut out).is_err());
        // output too small
        assert!(fused_q4k_q8k_multirow_matmul_into(&w, &sc, &qq, 2, 256, 2, &mut out[..3]).is_err());
        // m == 0 is a no-op
        assert!(fused_q4k_q8k_multirow_matmul_into(&w, &[], &[], 0, 256, 2, &mut []).is_ok());
    }

    #[test]
    fn multirow_tile_rows_is_l2_sized_multiple_of_four() {
        assert_eq!(multirow_tile_rows(144), 64);
        assert_eq!(multirow_tile_rows(16 * 144), 64); // 4096-wide row: 2304 B
        assert_eq!(multirow_tile_rows(48 * 144), 36); // 12288-wide row: 6912 B → 37 → 36
        assert_eq!(multirow_tile_rows(1 << 20), 4);
        for b in [144, 1000, 6912, 50_000, 1 << 20] {
            assert_eq!(multirow_tile_rows(b) % 4, 0);
        }
    }

    /// Perf probe (not a gate): multirow vs m × per-token matvec on a 4096² Q4_K
    /// matrix. Run it with
    /// `cargo test -p aprender-serve --release --lib multirow_perf -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf probe; run explicitly"]
    fn multirow_perf_probe() {
        let (k, n) = (4096, 4096);
        let w = q4k_weights(n, k, 3);
        for m in [8usize, 32, 64] {
            let (sc, qq) = q8k_rows(m, k);
            let mut out = vec![0f32; m * n];
            let reps = 5;
            let mut best = [f64::MAX; 2];
            for _ in 0..reps {
                let t0 = std::time::Instant::now();
                let _ = per_token(&w, &sc, &qq, m, k, n);
                best[0] = best[0].min(t0.elapsed().as_secs_f64());
                let t0 = std::time::Instant::now();
                fused_q4k_q8k_multirow_matmul_into(&w, &sc, &qq, m, k, n, &mut out).expect("mr");
                best[1] = best[1].min(t0.elapsed().as_secs_f64());
            }
            eprintln!(
                "m={m:3} per-token {:.3} ms  multirow {:.3} ms  speedup {:.2}x",
                best[0] * 1e3,
                best[1] * 1e3,
                best[0] / best[1]
            );
        }
    }

    /// FALSIFY-4228-006: the Q5_K/Q6_K multi-row GEMMs equal the per-row matvec bit for bit,
    /// including a ragged last tile and an out_dim below the per-row parallel threshold.
    #[test]
    fn falsify_4228_006_q5k_q6k_multirow_bit_identical() {
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for &(sb_bytes, is_q5) in &[(176usize, true), (210usize, false)] {
            for &(m, in_dim, out_dim) in &[(3usize, 256usize, 70usize), (9, 512, 300), (2, 768, 257), (17, 1024, 33)] {
                let nsb = in_dim / 256;
                let w: Vec<u8> = (0..out_dim * nsb * sb_bytes).map(|_| next() as u8).collect();
                let x: Vec<f32> = (0..m * in_dim)
                    .map(|_| ((next() % 2001) as f32 - 1000.0) / 997.0)
                    .collect();
                let mut want = vec![0.0f32; m * out_dim];
                for t in 0..m {
                    let (xi, yo) = (&x[t * in_dim..(t + 1) * in_dim], &mut want[t * out_dim..(t + 1) * out_dim]);
                    if is_q5 {
                        fused_q5k_parallel_matvec_into(&w, xi, in_dim, out_dim, yo).expect("q5k");
                    } else {
                        fused_q6k_parallel_matvec_into(&w, xi, in_dim, out_dim, yo).expect("q6k");
                    }
                }
                let mut got = vec![0.0f32; m * out_dim];
                if is_q5 {
                    fused_q5k_multirow_matmul_into(&w, &x, m, in_dim, out_dim, &mut got).expect("q5k mr");
                } else {
                    fused_q6k_multirow_matmul_into(&w, &x, m, in_dim, out_dim, &mut got).expect("q6k mr");
                }
                // Random bytes make some f16 scales NaN/Inf. Which NaN payload an add
                // propagates depends on operand order in the SIMD lane, not on the math, so
                // every NaN compares as one canonical NaN; every other bit must match.
                let bits = |v: &[f32]| {
                    v.iter()
                        .map(|f| if f.is_nan() { f32::NAN.to_bits() } else { f.to_bits() })
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(&got), bits(&want), "q5={is_q5} m={m} in={in_dim} out={out_dim}");
            }
        }
    }
}
