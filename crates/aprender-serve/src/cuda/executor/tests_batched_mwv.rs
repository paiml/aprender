//! #5303: the batched MWV Q4_K and Q6_K GEMVs against the single-row MWV
//! GEMVs, row by row. A batched decode step may replace `m` single steps only
//! if every row it produces is the single step's output to the bit.

use super::CudaExecutor;
use serial_test::serial;
use trueno_gpu::driver::GpuBuffer;
use trueno_gpu::GpuError;

/// Activation rows per launch, up to the scheduler's largest batch.
const ROWS: [u32; 6] = [1, 2, 3, 4, 8, 16];

/// Warps per block. The per-row sum order depends on it, so each count is
/// checked against the single-row kernel at the same count.
const WARPS: [u32; 3] = [2, 3, 4];

/// One quant type: its super-block, its two GEMVs, and the shapes to check.
struct Quant {
    name: &'static str,
    /// Bytes per 256-value super-block.
    block_bytes: usize,
    /// Bytes at the start of a block written as fixed f16 scales, then the
    /// rest pseudo-random; or, if `head` is false, the scales go at the end.
    scales: [u8; 4],
    head: bool,
    /// (n, k) pairs.
    shapes: &'static [(u32, u32)],
    single: fn(
        &mut CudaExecutor,
        u64,
        &GpuBuffer<f32>,
        &GpuBuffer<f32>,
        u32,
        u32,
    ) -> Result<(), GpuError>,
    batched: fn(
        &mut CudaExecutor,
        u64,
        &GpuBuffer<f32>,
        &GpuBuffer<f32>,
        u32,
        u32,
        u32,
    ) -> Result<(), GpuError>,
}

/// Q4_K: d and dmin (f16 0.0999 and 0.0125) lead the 144-byte block. The
/// last shape's k is not a multiple of 256, so its last super-block is part
/// padding.
const Q4K: Quant = Quant {
    name: "q4k",
    block_bytes: 144,
    scales: [0x66, 0x2E, 0x66, 0x22],
    head: true,
    shapes: &[(64, 256), (48, 1024), (17, 3584), (40, 1000)],
    single: CudaExecutor::mwv_q4k_gemv_into,
    batched: CudaExecutor::batched_mwv_q4k_gemv_into,
};

/// Q6_K: d (f16 0.0062) closes the 210-byte block. The single-row wrapper
/// asserts k % 256 == 0, so every shape is aligned.
const Q6K: Quant = Quant {
    name: "q6k",
    block_bytes: 210,
    scales: [0x66, 0x1E, 0, 0],
    head: false,
    shapes: &[(64, 256), (48, 1024), (17, 3584), (40, 512)],
    single: CudaExecutor::mwv_q6k_gemv_into,
    batched: CudaExecutor::batched_mwv_q6k_gemv_into,
};

/// xorshift32, never 0.
fn next(seed: &mut u32) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}

/// `n` rows of super-blocks for `k` inputs: fixed small finite scales,
/// pseudo-random everything else.
fn weights(q: &Quant, n: u32, k: u32, mut seed: u32) -> Vec<u8> {
    let blocks = (k.div_ceil(256) * n) as usize;
    let fixed = if q.head { 4 } else { 2 };
    let mut out = Vec::with_capacity(blocks * q.block_bytes);
    for _ in 0..blocks {
        if q.head {
            out.extend_from_slice(&q.scales[..fixed]);
        }
        for _ in 0..q.block_bytes - fixed {
            out.push(next(&mut seed) as u8);
        }
        if !q.head {
            out.extend_from_slice(&q.scales[..fixed]);
        }
    }
    out
}

/// `rows × k` activations in [-1, 1).
fn activations(rows: u32, k: u32, mut seed: u32) -> Vec<f32> {
    (0..rows * k)
        .map(|_| (next(&mut seed) as f32 / u32::MAX as f32) * 2.0 - 1.0)
        .collect()
}

/// `n` outputs of the single-row GEMV per row of `x`, concatenated.
fn single_rows(ex: &mut CudaExecutor, q: &Quant, w: u64, x: &[f32], n: u32, k: u32) -> Vec<f32> {
    let mut want = vec![0.0f32; x.len() / k as usize * n as usize];
    for (row, out_row) in x.chunks(k as usize).zip(want.chunks_mut(n as usize)) {
        let input = GpuBuffer::from_host(ex.context(), row).expect("row up");
        let out = GpuBuffer::new(ex.context(), n as usize).expect("row out");
        (q.single)(ex, w, &input, &out, n, k).expect("single gemv");
        ex.synchronize().expect("sync");
        out.copy_to_host(out_row).expect("row down");
    }
    want
}

/// All `m` rows of `x` in one batched launch.
fn batched(
    ex: &mut CudaExecutor,
    q: &Quant,
    w: u64,
    x: &[f32],
    m: u32,
    n: u32,
    k: u32,
) -> Vec<f32> {
    let input = GpuBuffer::from_host(ex.context(), x).expect("rows up");
    let out = GpuBuffer::new(ex.context(), (m * n) as usize).expect("rows out");
    (q.batched)(ex, w, &input, &out, m, n, k).expect("batched gemv");
    ex.synchronize().expect("sync");
    let mut got = vec![0.0f32; (m * n) as usize];
    out.copy_to_host(&mut got).expect("rows down");
    got
}

/// Row by row, `got` is bitwise `want`, and `want` is not all ~0.
fn assert_rows_bitwise(want: &[f32], got: &[f32], n: u32, what: &str) {
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    let rows = want.chunks(n as usize).zip(got.chunks(n as usize));
    for (r, (want, got)) in rows.enumerate() {
        assert_eq!(
            bits(got),
            bits(want),
            "{what}: row {r} is not the single-row GEMV"
        );
    }
    assert!(
        want.iter().any(|v| v.abs() > 1e-3),
        "{what}: the single-row outputs are all ~0, so equality proves nothing"
    );
}

/// Every shape, warp count and row count of one quant type; returns the
/// number of cases checked.
fn check_quant(ex: &mut CudaExecutor, q: &Quant) -> usize {
    let mut checked = 0usize;
    for (i, &(n, k)) in (0u32..).zip(q.shapes) {
        let name = format!("batched_mwv_{}_5303_{i}", q.name);
        ex.load_quantized_weights(&name, &weights(q, n, k, 0x9E37 + i))
            .expect("load weights");
        let w = ex.get_quantized_weight_ptr(&name).expect("weight ptr");
        for warps in WARPS {
            ex.gpu_profile.mwv_warps = warps;
            for m in ROWS {
                let x = activations(m, k, n ^ k ^ m ^ warps);
                let want = single_rows(ex, q, w, &x, n, k);
                let got = batched(ex, q, w, &x, m, n, k);
                let what = format!("{} n={n} k={k} warps={warps} m={m}", q.name);
                assert_rows_bitwise(&want, &got, n, &what);
                checked += 1;
            }
        }
    }
    checked
}

/// FALSIFY (#5303): for every shape, row count and warp count, each row of
/// the batched MWV Q4_K GEMV is bitwise the single-row MWV GEMV of that row.
#[test]
#[serial]
fn batched_mwv_q4k_rows_are_bitwise_the_single_row_gemv() {
    let mut ex = crate::cuda_executor_or_skip!(0);
    let checked = check_quant(&mut ex, &Q4K);
    assert_eq!(checked, Q4K.shapes.len() * WARPS.len() * ROWS.len());
}

/// FALSIFY (#5303): the same for the batched MWV Q6_K GEMV, which serves
/// lm_head and part of ffn_down and attn_v.
#[test]
#[serial]
fn batched_mwv_q6k_rows_are_bitwise_the_single_row_gemv() {
    let mut ex = crate::cuda_executor_or_skip!(0);
    let checked = check_quant(&mut ex, &Q6K);
    assert_eq!(checked, Q6K.shapes.len() * WARPS.len() * ROWS.len());
}

/// The rows of a batch are distinct inputs, so a kernel that read row 0 for
/// every row (or any one row for all) would fail the bitwise tests above.
#[test]
fn the_batch_rows_differ() {
    let x = activations(16, 256, 7);
    let rows: Vec<&[f32]> = x.chunks(256).collect();
    for (a, row_a) in rows.iter().enumerate() {
        for row_b in &rows[a + 1..] {
            assert_ne!(row_a, row_b);
        }
    }
}
