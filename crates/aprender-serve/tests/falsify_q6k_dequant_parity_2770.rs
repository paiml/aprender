//! FALSIFY-Q6K-DEQUANT-001 (aprender#2770): the Q6_K GPU dequant kernel that feeds
//! the cuBLAS GEMM route must agree with the CPU dequant that the rest of the
//! system already trusts.
//!
//! # What this catches
//!
//! `Q6KDequantKernel` (`aprender-gpu/src/kernels/quantize/q6k/dequant.rs`) is the
//! only Q6_K reader on the `cublas_prefill_gemm` path. Every GEMM precision on that
//! path is built from its FP32 output: FP8 E4M3 caches from it
//! (`get_or_cache_fp8_weight`), FP16 caches from it (`get_or_cache_fp16_weight`),
//! and the SGEMM fallback uses it directly. The GEMV path never calls it -- it reads
//! Q6_K in place.
//!
//! Q6_K packs each 128-value half as four groups of 32, and the two bits of the
//! group index mean DIFFERENT things:
//!
//! ```text
//!   group 0:  ql[l]      low nibble    qh[l] >> 0
//!   group 1:  ql[l + 32] low nibble    qh[l] >> 2
//!   group 2:  ql[l]      high nibble   qh[l] >> 4
//!   group 3:  ql[l + 32] high nibble   qh[l] >> 6
//! ```
//!
//! so the byte offset is driven by `group & 1` and the nibble by `group >= 2`. A
//! kernel that swaps those two roles is right for groups 0 and 3 (both bits equal)
//! and wrong for groups 1 and 2, which exchange their low four bits while keeping
//! the correct `qh` bits and the correct sub-block scale. Half of every super-block
//! is then subtly wrong rather than obviously broken -- the weights keep their
//! magnitude and sign structure, so the model still emits fluent text, just a
//! different continuation. That is what #2770 observed at m >= 4 and what a human
//! spot-check reads as sampling variation.
//!
//! # The fixture is the hard part
//!
//! `q5k_q6k_gemm_row_parity.rs` builds its super-block with `ql = 0x55` everywhere.
//! Both nibbles of every byte are then `0x5` and both 32-byte halves are identical,
//! so swapping byte offset with nibble select is UNOBSERVABLE and a comparison
//! against any correct reference still passes.
//!
//! That is not a trap only uniform fixtures fall into. The first version of THIS
//! fixture filled `ql` from a pattern linear in the byte index, which repeats with
//! period 16; the group-1/group-2 byte step is 32, a multiple of 16, so `ql[l]` and
//! `ql[l + 32]` were byte-identical and a mutation of the byte offset alone PASSED.
//! The fixture is therefore constructed lane by lane, and
//! `q6k_fixture_can_distinguish_every_part_of_the_group_mapping` asserts each
//! required inequality before any GPU output is compared.
//!
//! Run on a CUDA host:
//!   cargo test -p aprender-serve --features cuda --test falsify_q6k_dequant_parity_2770

#![cfg(feature = "cuda")]

use realizar::quantize::dequantize_q6_k;
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
use trueno_gpu::kernels::{Kernel, Q6KDequantKernel};

const SB_BYTES: usize = 210;
const SB_VALUES: usize = 256;

/// One Q6_K super-block whose `ql` bytes are CONSTRUCTED to make each part of the
/// group mapping separately observable.
///
/// A linear nibble pattern cannot do this. Any function of the byte index that is
/// linear mod 16 repeats with period 16, and the group-1/group-2 byte step is 32 --
/// a multiple of 16 -- so `ql[l]` and `ql[l + 32]` come out byte-identical and the
/// step is invisible. A first version of this fixture did exactly that and let a
/// mutation of the byte offset pass. The nibbles are therefore laid down per lane
/// with the three needed inequalities guaranteed by construction:
///
/// ```text
///   ql[64h + l]      = A | (B << 4)     A = (l + 8h + seed) & 0xF, B = A + 1
///   ql[64h + l + 32] = C | (D << 4)     C = A + 2,                 D = C + 5
/// ```
///
///   A != B and C != D  -> the low/high nibble select is observable
///   C != B             -> the group-1/group-2 exchange (#2770) is observable
///   (A,B) != (C,D)     -> the +32 byte step is observable
///
/// `A` depends on the half, so swapping the two 128-value halves is observable too.
/// The 16 sub-block scales are all different, so a mis-indexed scale also shows.
fn q6k_superblock_distinct(seed: u8) -> Vec<u8> {
    let mut b = vec![0u8; SB_BYTES];
    for half in 0..2usize {
        for l in 0..32usize {
            let a = ((l as u8).wrapping_add(8 * half as u8).wrapping_add(seed)) & 0x0F;
            let bb = (a + 1) & 0x0F;
            let c = (a + 2) & 0x0F;
            let d = (c + 5) & 0x0F;
            b[64 * half + l] = a | (bb << 4);
            b[64 * half + l + 32] = c | (d << 4);
        }
    }
    // qh[128..192): four 2-bit lanes per byte, all four different where possible.
    for (i, x) in b.iter_mut().take(192).skip(128).enumerate() {
        let i = i as u8;
        let l0 = i & 0x3;
        let l1 = (i.wrapping_add(1)) & 0x3;
        let l2 = (i.wrapping_add(2)) & 0x3;
        let l3 = (i.wrapping_add(3)) & 0x3;
        *x = l0 | (l1 << 2) | (l2 << 4) | (l3 << 6);
    }
    // scales[192..208): 16 distinct signed i8, mixed sign.
    for (i, x) in b.iter_mut().take(208).skip(192).enumerate() {
        let s: i8 = (i as i8) - 7; // -7 ..= 8, no zero-only block
        *x = if s == 0 { 5 } else { s } as u8;
    }
    // d = 0.125 in f16 (0x3000): exact in binary, so no rounding enters the compare.
    b[208] = 0x00;
    b[209] = 0x30;
    b
}

/// Run `q6k_dequant_to_f32` over `n` rows of `k` values and return the FP32 output.
///
/// Mirrors the production launch in
/// `cuda/executor/layers/cublas_prefill/gemm.rs::launch_dequant_q6k`:
/// grid (n, ceil(k/256)), block 32, args (out, weights, k, n).
fn gpu_dequant_q6k(ctx: &CudaContext, weights: &[u8], n: u32, k: u32) -> Vec<f32> {
    let kernel = Q6KDequantKernel::new(k, n);
    let ptx = kernel.emit_ptx();
    let mut module = CudaModule::from_ptx(ctx, &ptx).expect("Q6K dequant PTX failed to compile");
    let stream = CudaStream::new(ctx).expect("stream");

    let w_buf = GpuBuffer::from_host(ctx, weights).expect("weight upload");
    let out_len = (n as usize) * (k as usize);
    let out_buf: GpuBuffer<f32> = GpuBuffer::new(ctx, out_len).expect("output alloc");

    let num_sb = k.div_ceil(256);
    let config = LaunchConfig::grid_2d(n, num_sb, 32, 1);

    let out_ptr = out_buf.as_ptr();
    let w_ptr = w_buf.as_ptr();
    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(out_ptr) as *mut _,
        std::ptr::addr_of!(w_ptr) as *mut _,
        std::ptr::addr_of!(k) as *mut _,
        std::ptr::addr_of!(n) as *mut _,
    ];
    unsafe {
        stream
            .launch_kernel(&mut module, kernel.name(), &config, &mut args)
            .expect("q6k_dequant_to_f32 launch");
    }
    stream.synchronize().expect("sync");

    let mut out = vec![0.0f32; out_len];
    out_buf.copy_to_host(&mut out).expect("download");
    out
}

/// FIXTURE VALIDITY. The kernel derives the `ql` byte offset from one bit of the group index and the
/// nibble from the other. Three separate inequalities are needed before a GPU-vs-CPU
/// comparison can see a mistake in either, and they are asserted here on the fixture
/// BYTES -- not on dequantized values, since two different 6-bit codes can still
/// multiply to the same float through different sub-block scales.
///
/// This check is not decoration. It is the direct product of a mutation that PASSED:
/// with a linear nibble pattern, `ql[l]` and `ql[l + 32]` were byte-identical, so
/// reverting the byte offset alone left every output unchanged and the falsifier
/// reported green on a kernel that was still wrong.
#[test]
fn q6k_fixture_can_distinguish_every_part_of_the_group_mapping() {
    let sb = q6k_superblock_distinct(0);

    let (mut swap_ok, mut step_ok, mut nibble_ok) = (0usize, 0usize, 0usize);
    for half in 0..2usize {
        for l in 0..32usize {
            let lo_byte = sb[64 * half + l];
            let hi_byte = sb[64 * half + l + 32];
            // (a) group-1 low nibble vs group-2 high nibble: the #2770 exchange.
            if (hi_byte & 0x0F) != (lo_byte >> 4) {
                swap_ok += 1;
            }
            // (b) the +32 byte step itself.
            if hi_byte != lo_byte {
                step_ok += 1;
            }
            // (c) low-vs-high nibble select, in both bytes of the lane.
            if (lo_byte & 0x0F) != (lo_byte >> 4) && (hi_byte & 0x0F) != (hi_byte >> 4) {
                nibble_ok += 1;
            }
        }
    }
    assert_eq!(
        swap_ok,
        64,
        "fixture blind to the group-1/group-2 ql exchange on {} of 64 lanes",
        64 - swap_ok
    );
    assert_eq!(
        step_ok,
        64,
        "fixture blind to the +32 ql byte step on {} of 64 lanes -- this is the hole \
         that let a byte-offset mutation pass",
        64 - step_ok
    );
    assert_eq!(
        nibble_ok,
        64,
        "fixture blind to the low/high nibble select on {} of 64 lanes",
        64 - nibble_ok
    );

    // And the block must not be degenerate in the trivial ways.
    let cpu = dequantize_q6_k(&sb).expect("CPU dequant");
    assert_eq!(cpu.len(), SB_VALUES);
    assert!(cpu.iter().any(|v| *v != 0.0), "all-zero fixture");
    assert!(
        cpu.iter().all(|v: &f32| v.is_finite()),
        "non-finite fixture"
    );
}

/// FALSIFY-Q6K-DEQUANT-001: one super-block, GPU kernel vs the production CPU dequant.
#[test]
fn falsify_q6k_dequant_matches_cpu_single_superblock() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!(
            "FALSIFY-Q6K-DEQUANT-001 SKIPPED (ENV): no CUDA device. This is an \
             environment result, not a pass."
        );
        return;
    };

    let sb = q6k_superblock_distinct(0);
    let cpu = dequantize_q6_k(&sb).expect("CPU dequant");
    let gpu = gpu_dequant_q6k(&ctx, &sb, 1, SB_VALUES as u32);

    let mismatches: Vec<usize> = (0..SB_VALUES).filter(|&i| gpu[i] != cpu[i]).collect();
    assert!(
        mismatches.is_empty(),
        "Q6_K GPU dequant disagrees with the CPU dequant at {}/{} elements. \
         First 8: {:?}. Element {} : GPU {} vs CPU {}. \
         Every cuBLAS GEMM precision on the decode path is built from this buffer, \
         so this corrupts Q4_K_M's Q6_K tensors (attn_v, ffn_down, output) at m >= 4.",
        mismatches.len(),
        SB_VALUES,
        &mismatches[..8.min(mismatches.len())],
        mismatches[0],
        gpu[mismatches[0]],
        cpu[mismatches[0]],
    );
}

/// Multi-row, multi-super-block: also pins the row stride and the super-block stride,
/// which a single 1x256 case cannot see.
#[test]
fn falsify_q6k_dequant_matches_cpu_multi_row() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!(
            "FALSIFY-Q6K-DEQUANT-001 (multi-row) SKIPPED (ENV): no CUDA device. \
             This is an environment result, not a pass."
        );
        return;
    };

    let n: u32 = 3;
    let k: u32 = 512; // 2 super-blocks per row
    let sb_per_row = (k as usize) / SB_VALUES;

    let mut weights = Vec::with_capacity(n as usize * sb_per_row * SB_BYTES);
    for i in 0..(n as usize * sb_per_row) {
        weights.extend_from_slice(&q6k_superblock_distinct(i as u8 * 5));
    }

    let cpu = dequantize_q6_k(&weights).expect("CPU dequant");
    let gpu = gpu_dequant_q6k(&ctx, &weights, n, k);

    assert_eq!(cpu.len(), gpu.len());
    let mismatches: Vec<usize> = (0..cpu.len()).filter(|&i| gpu[i] != cpu[i]).collect();
    assert!(
        mismatches.is_empty(),
        "Q6_K GPU dequant disagrees with CPU at {}/{} elements (n={}, k={}). \
         First 8: {:?}. Element {} : GPU {} vs CPU {}.",
        mismatches.len(),
        cpu.len(),
        n,
        k,
        &mismatches[..8.min(mismatches.len())],
        mismatches[0],
        gpu[mismatches[0]],
        cpu[mismatches[0]],
    );
}
