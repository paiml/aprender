
/// Parallel fused Q5_K matrix-vector multiply - writes to pre-allocated buffer
///
/// IMP-131: Zero-allocation variant for hot-path inference.
/// Contract: quantized-dot-product-v1.yaml — delegates to generic matvec.
#[allow(clippy::similar_names)]
pub fn fused_q5k_parallel_matvec_into(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    generic_parallel_matvec_into::<Q5K>(
        weight_data,
        activations,
        in_dim,
        out_dim,
        output,
        fused_q5k_dot_simd,
    )
}

/// Parallel fused Q6_K matrix-vector multiply
///
/// Contract: quantized-dot-product-v1.yaml — delegates to generic matvec.
///
/// # Errors
///
/// Returns error if:
/// - Weight data is too small for the given dimensions
/// - Activation length doesn't match input dimension
#[allow(clippy::similar_names)]
pub fn fused_q6k_parallel_matvec(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    generic_parallel_matvec::<Q6K>(weight_data, activations, in_dim, out_dim, fused_q6k_dot_simd)
}

/// Parallel fused Q6_K matrix-vector multiply - writes to pre-allocated buffer
///
/// IMP-131: Zero-allocation variant for hot-path inference.
/// Contract: quantized-dot-product-v1.yaml — delegates to generic matvec.
#[allow(clippy::similar_names)]
pub fn fused_q6k_parallel_matvec_into(
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    generic_parallel_matvec_into::<Q6K>(
        weight_data,
        activations,
        in_dim,
        out_dim,
        output,
        fused_q6k_dot_simd,
    )
}

// ============================================================================

// LAYOUT-002: Legacy aliases DELETED (2026-02-03)
// - fused_q6k_colmajor_matvec: Misleading name, was just an alias for fused_q6k_parallel_matvec
// - fused_q4k_auto_matvec_into: Confusing name, was just an alias for fused_q4k_parallel_matvec_into
// ONE WAY ONLY: Use fused_q{4,5,6}k_parallel_matvec* functions directly

/// Parallel Q4_K × Q8_K matrix-vector multiply with TCB tiling
///
/// Uses rayon parallel iterators for multi-core acceleration and TCB tiling
/// pattern for cache optimization.
///
/// Contract: quantized-dot-product-v1.yaml — precomputes Q8K bsums once per
/// token for the fallback path (4-row AVX512 VNNI path has internal precomputation).
pub fn fused_q4k_q8k_parallel_matvec_into(
    weight_data: &[u8],
    q8k_scales: &[f32],
    q8k_quants: &[i8],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    use rayon::prelude::*;

    const SUPER_BLOCK_BYTES: usize = 144;

    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * SUPER_BLOCK_BYTES;

    let expected_weight_bytes = out_dim * bytes_per_row;
    if weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Weight data too small: need {} bytes, have {}",
                expected_weight_bytes,
                weight_data.len()
            ),
        });
    }

    if output.len() < out_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Output buffer too small: need {}, have {}",
                out_dim,
                output.len()
            ),
        });
    }

    // PMAT-306: Lean dispatch — raw pointer arithmetic, no per-row overhead.
    // PMAT-304 perf stat: our IPC 1.60 vs llama.cpp 1.01. The 0.59 IPC excess
    // is from bounds checks, slice operations, and closure dispatch between
    // memory loads. This path uses raw pointers like ggml (ggml-cpu.c:1214).
    #[cfg(target_arch = "x86_64")]
    let use_lean = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_lean = false;

    if use_lean {
        #[cfg(target_arch = "x86_64")]
        {
            let bsums_i16 = unsafe {
                super::fused_k::precompute_q8k_bsums_i16(q8k_quants, super_blocks_per_row)
            };

            // PMAT-306: Raw pointer parallel dispatch — minimal overhead.
            let bpr = bytes_per_row;
            let nsb = super_blocks_per_row;

            // PMAT-310: GEMV pool FALSIFIED (deadlock under concurrent requests).
            // Barrier-based pool + Mutex for work distribution = deadlock when
            // HTTP handler thread contends with pool threads on barriers.
            // Rayon's work-stealing, despite overhead, handles concurrency correctly.
            use rayon::prelude::*;
            let w_addr = weight_data.as_ptr() as usize;
            let sc_addr = q8k_scales.as_ptr() as usize;
            let qq_addr = q8k_quants.as_ptr() as usize;
            let bs_addr = bsums_i16.as_ptr() as usize;

            output[..out_dim]
                .par_chunks_mut(64)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let row_start = ci * 64;
                    unsafe {
                        let w = w_addr as *const u8;
                        let sc = sc_addr as *const f32;
                        let qq = qq_addr as *const i8;
                        let bs = bs_addr as *const i16;
                        for (i, out) in chunk.iter_mut().enumerate() {
                            let row = row_start + i;
                            *out = super::fused_k::ggml_style_q4k_q8k_dot_avx2_raw(
                                w.add(row * bpr), sc, qq, bs, nsb,
                            );
                        }
                    }
                });

            return Ok(());
        }
    }

    let bsums = precompute_q8k_bsums(q8k_quants, super_blocks_per_row).ok();

    // Fallback: original per-block hsum path
    const MIDI_TILE_M: usize = 64;
    const MICRO_TILE_M: usize = 4;

    #[cfg(target_arch = "x86_64")]
    let use_4row_kernel =
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vnni");
    #[cfg(not(target_arch = "x86_64"))]
    let use_4row_kernel = false;

    // PMAT-298: Use rayon for all matmuls above 1 midi-tile (64 rows).
    // Below that, sequential is faster (no thread dispatch overhead).
    // Combined with PMAT-297 thread pool = 16 physical cores.
    let use_parallel = out_dim > MIDI_TILE_M;

    if use_4row_kernel && out_dim >= MICRO_TILE_M {
        if use_parallel {
        // TCB-style tiled execution with 4-row micro-kernel + rayon parallelism
        output[..out_dim]
            .par_chunks_mut(MIDI_TILE_M)
            .enumerate()
            .for_each(|(midi_idx, midi_chunk)| {
                let midi_start = midi_idx * MIDI_TILE_M;
                let midi_rows = midi_chunk.len();

                // Process micro-tiles (4 rows at a time) within this midi-tile
                let full_micro_tiles = midi_rows / MICRO_TILE_M;
                let remainder = midi_rows % MICRO_TILE_M;

                for micro_idx in 0..full_micro_tiles {
                    let row_base = midi_start + micro_idx * MICRO_TILE_M;

                    // Build row pointers for 4-row kernel
                    let row_ptrs: [*const u8; 4] = [
                        weight_data.as_ptr().wrapping_add(row_base * bytes_per_row),
                        weight_data
                            .as_ptr()
                            .wrapping_add((row_base + 1) * bytes_per_row),
                        weight_data
                            .as_ptr()
                            .wrapping_add((row_base + 2) * bytes_per_row),
                        weight_data
                            .as_ptr()
                            .wrapping_add((row_base + 3) * bytes_per_row),
                    ];

                    // SAFETY: AVX-512 VNNI detected, pointers are within weight_data bounds
                    #[cfg(target_arch = "x86_64")]
                    let outputs = unsafe {
                        fused_q4k_q8k_dot_4rows_avx512vnni(
                            row_ptrs,
                            bytes_per_row,
                            q8k_scales,
                            q8k_quants,
                        )
                    };

                    #[cfg(not(target_arch = "x86_64"))]
                    let outputs = [0.0f32; 4];

                    let local_base = micro_idx * MICRO_TILE_M;
                    midi_chunk[local_base] = outputs[0];
                    midi_chunk[local_base + 1] = outputs[1];
                    midi_chunk[local_base + 2] = outputs[2];
                    midi_chunk[local_base + 3] = outputs[3];
                }

                // Handle remainder rows (< 4) with bsum-aware single-row kernel
                for r in 0..remainder {
                    let row = midi_start + full_micro_tiles * MICRO_TILE_M + r;
                    let row_start = row * bytes_per_row;
                    let row_data = &weight_data[row_start..row_start + bytes_per_row];
                    let local_idx = full_micro_tiles * MICRO_TILE_M + r;
                    midi_chunk[local_idx] = if let Some(ref bs) = bsums {
                        fused_q4k_q8k_dot_with_bsums_simd(
                            row_data, q8k_scales, q8k_quants, bs,
                        )
                        .unwrap_or(0.0)
                    } else {
                        fused_q4k_q8k_dot_simd(row_data, q8k_scales, q8k_quants).unwrap_or(0.0)
                    };
                }
            });
        } else {
            // PMAT-298: Sequential path for small matmuls (out_dim < 1024).
            // Avoids rayon thread dispatch overhead (~10us per task).
            for midi_idx in 0..(out_dim + MIDI_TILE_M - 1) / MIDI_TILE_M {
                let midi_start = midi_idx * MIDI_TILE_M;
                let midi_end = (midi_start + MIDI_TILE_M).min(out_dim);
                let midi_rows = midi_end - midi_start;

                let full_micro_tiles = midi_rows / MICRO_TILE_M;
                let remainder = midi_rows % MICRO_TILE_M;

                for micro_idx in 0..full_micro_tiles {
                    let row_base = midi_start + micro_idx * MICRO_TILE_M;

                    // PMAT-299: Prefetch next micro-tile's weight rows
                    #[cfg(target_arch = "x86_64")]
                    if micro_idx + 1 < full_micro_tiles {
                        let next_base = midi_start + (micro_idx + 1) * MICRO_TILE_M;
                        for r in 0..4 {
                            let p = weight_data.as_ptr().wrapping_add((next_base + r) * bytes_per_row);
                            unsafe {
                                std::arch::x86_64::_mm_prefetch(p.cast::<i8>(), std::arch::x86_64::_MM_HINT_T1);
                            }
                        }
                    }

                    let row_ptrs: [*const u8; 4] = [
                        weight_data.as_ptr().wrapping_add(row_base * bytes_per_row),
                        weight_data.as_ptr().wrapping_add((row_base + 1) * bytes_per_row),
                        weight_data.as_ptr().wrapping_add((row_base + 2) * bytes_per_row),
                        weight_data.as_ptr().wrapping_add((row_base + 3) * bytes_per_row),
                    ];

                    #[cfg(target_arch = "x86_64")]
                    let outputs = unsafe {
                        fused_q4k_q8k_dot_4rows_avx512vnni(
                            row_ptrs, bytes_per_row, q8k_scales, q8k_quants,
                        )
                    };
                    #[cfg(not(target_arch = "x86_64"))]
                    let outputs = [0.0f32; 4];

                    let local_base = midi_start + micro_idx * MICRO_TILE_M;
                    output[local_base] = outputs[0];
                    output[local_base + 1] = outputs[1];
                    output[local_base + 2] = outputs[2];
                    output[local_base + 3] = outputs[3];
                }

                for r in 0..remainder {
                    let row = midi_start + full_micro_tiles * MICRO_TILE_M + r;
                    let row_start = row * bytes_per_row;
                    let row_data = &weight_data[row_start..row_start + bytes_per_row];
                    output[row] = if let Some(ref bs) = bsums {
                        fused_q4k_q8k_dot_with_bsums_simd(row_data, q8k_scales, q8k_quants, bs)
                            .unwrap_or(0.0)
                    } else {
                        fused_q4k_q8k_dot_simd(row_data, q8k_scales, q8k_quants).unwrap_or(0.0)
                    };
                }
            }
        }
    } else if use_parallel {
        // Fallback: per-row execution with rayon + precomputed bsums
        output[..out_dim]
            .par_chunks_mut(MIDI_TILE_M)
            .enumerate()
            .for_each(|(midi_idx, midi_chunk)| {
                let midi_start = midi_idx * MIDI_TILE_M;

                for (local_idx, out) in midi_chunk.iter_mut().enumerate() {
                    let row = midi_start + local_idx;
                    let row_start = row * bytes_per_row;
                    let row_data = &weight_data[row_start..row_start + bytes_per_row];
                    *out = if let Some(ref bs) = bsums {
                        fused_q4k_q8k_dot_with_bsums_simd(
                            row_data, q8k_scales, q8k_quants, bs,
                        )
                        .unwrap_or(0.0)
                    } else {
                        fused_q4k_q8k_dot_simd(row_data, q8k_scales, q8k_quants).unwrap_or(0.0)
                    };
                }
            });
    } else {
        // PMAT-298: Sequential fallback for small matmuls without AVX-512
        for row in 0..out_dim {
            let row_start = row * bytes_per_row;
            let row_data = &weight_data[row_start..row_start + bytes_per_row];
            output[row] = if let Some(ref bs) = bsums {
                fused_q4k_q8k_dot_with_bsums_simd(row_data, q8k_scales, q8k_quants, bs)
                    .unwrap_or(0.0)
            } else {
                fused_q4k_q8k_dot_simd(row_data, q8k_scales, q8k_quants).unwrap_or(0.0)
            };
        }
    }

    Ok(())
}

/// Fused FFN up+gate projection in single parallel region
///
/// Eliminates rayon::join overhead by processing both up and gate weights
/// in a single par_chunks_mut call. Both projections share the same Q8K
/// quantized input, so we only load it once per midi-tile.
///
/// # Performance
///
/// Reduces parallel region spawns from 2 to 1 per FFN layer, saving ~10-50µs
/// per layer. For 28 layers, this is 280-1400µs per token.
///
/// # Arguments
///
/// * `up_weight` - Q4K weight data for FFN up projection
/// * `gate_weight` - Q4K weight data for FFN gate projection
/// * `q8k_scales` - Pre-quantized activation scales
/// * `q8k_quants` - Pre-quantized activation values
/// * `in_dim` - Input dimension (hidden_dim)
/// * `out_dim` - Output dimension (intermediate_dim)
/// * `up_output` - Output buffer for up projection
/// * `gate_output` - Output buffer for gate projection
#[allow(clippy::too_many_arguments)]
pub fn fused_q4k_q8k_ffn_up_gate_into(
    up_weight: &[u8],
    gate_weight: &[u8],
    q8k_scales: &[f32],
    q8k_quants: &[i8],
    in_dim: usize,
    out_dim: usize,
    up_output: &mut [f32],
    gate_output: &mut [f32],
) -> Result<()> {
    use rayon::prelude::*;

    const SUPER_BLOCK_BYTES: usize = 144;
    const MIDI_TILE_M: usize = 64;

    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * SUPER_BLOCK_BYTES;

    let expected_weight_bytes = out_dim * bytes_per_row;
    if up_weight.len() < expected_weight_bytes || gate_weight.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Weight data too small: need {} bytes",
                expected_weight_bytes
            ),
        });
    }

    if up_output.len() < out_dim || gate_output.len() < out_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!("Output buffers too small: need {}", out_dim),
        });
    }

    // CONTRACT-DERIVED P1: Precompute Q8K bsums ONCE (shared by up + gate)
    let bsums = precompute_q8k_bsums(q8k_quants, super_blocks_per_row).ok();

    // Process both up and gate in a single parallel region
    // Each thread handles a midi-tile of rows for BOTH projections
    // We use zip + par_chunks_mut to ensure thread-safe non-overlapping access
    up_output[..out_dim]
        .par_chunks_mut(MIDI_TILE_M)
        .zip(gate_output[..out_dim].par_chunks_mut(MIDI_TILE_M))
        .enumerate()
        .for_each(|(midi_idx, (up_chunk, gate_chunk))| {
            let midi_start = midi_idx * MIDI_TILE_M;

            for (local_row, (up_out, gate_out)) in
                up_chunk.iter_mut().zip(gate_chunk.iter_mut()).enumerate()
            {
                let row = midi_start + local_row;
                let row_start = row * bytes_per_row;

                // Compute up projection for this row (with precomputed bsums)
                let up_row = &up_weight[row_start..row_start + bytes_per_row];
                *up_out = if let Some(ref bs) = bsums {
                    fused_q4k_q8k_dot_with_bsums_simd(up_row, q8k_scales, q8k_quants, bs)
                        .unwrap_or(0.0)
                } else {
                    fused_q4k_q8k_dot_simd(up_row, q8k_scales, q8k_quants).unwrap_or(0.0)
                };

                // Compute gate projection for this row (with precomputed bsums)
                let gate_row = &gate_weight[row_start..row_start + bytes_per_row];
                *gate_out = if let Some(ref bs) = bsums {
                    fused_q4k_q8k_dot_with_bsums_simd(gate_row, q8k_scales, q8k_quants, bs)
                        .unwrap_or(0.0)
                } else {
                    fused_q4k_q8k_dot_simd(gate_row, q8k_scales, q8k_quants).unwrap_or(0.0)
                };
            }
        });

    Ok(())
}
