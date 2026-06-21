// PMAT-882 — pure-Rust cuda-oxide port of the incremental (KV-cache) attention kernel.
//
// Target = hand-PTX MultiWarpIncrementalAttentionKernel
// (crates/aprender-gpu/src/kernels/attention/paged/multi_warp/build_ptx.rs), whose
// design intent is "multiple warps per head to parallelize across KV positions"
// (more blocks => more SM occupancy => long-context throughput). The production
// long-context path is the same idea taken further: Flash-Decoding split-K
// (crates/aprender-gpu/src/kernels/attention/paged/flash_decoding/), one block
// per (head, chunk) + a reduction.
//
// CPU reference = `causal_attention_cached`
// (crates/aprender-serve/src/apr_transformer/attention_kernels.rs).
//
// Computes, for ONE decode token (new_seq_len == 1) and each query head h:
//     score[j] = dot(Q[h], K[j, kv_head(h)]) * scale          for j in 0..kv_len
//     w        = softmax(score)                                 (all cached j valid)
//     out[h]   = sum_j w[j] * V[j, kv_head(h)]
// where kv_head(h) = h / (n_heads / n_kv_heads)  (GQA).
//
// Layout (matches the live serve `causal_attention_cached` [seq, kv_dim] cache):
//   q   : [n_heads * head_dim]                f32
//   k   : [kv_len * kv_dim]                   f32   (kv_dim = n_kv_heads * head_dim)
//   v   : [kv_len * kv_dim]                   f32
//   out : [n_heads * head_dim]               f32
//   K[j, kv_head] at j*kv_dim + kv_head*head_dim ; same for V.
//
// Two kernels are provided:
//   (A) incremental_attention  — one block per head, T threads, online softmax +
//       cooperative merge. The direct single-launch MultiWarp analog. Good for
//       short ctx, occupancy-starved for long ctx (only n_heads blocks).
//   (B) attn_chunk + attn_reduce — Flash-Decoding split-K. Block per (head,chunk)
//       writes a partial (max,sum,out[head_dim]); attn_reduce merges the chunks.
//       This is the occupancy fix for long context (n_heads*n_chunks blocks).
//
// PARITY is vs the CPU `causal_attention_cached` math. A true hand-PTX A/B needs
// the full aprender-gpu CUDA executor on gx10 (the GH-480 Blackwell JIT path this
// north-star escapes); per the PMAT-881 process we report CPU-ref parity + the
// documented hand-PTX baseline (multi_warp/mod.rs: ~10us target, 81us single-warp).

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::shared::SharedArray;
use cuda_device::{kernel, thread, warp};
use cuda_host::cuda_module;

const MAX_HD: usize = 128; // head_dim cap (hand-PTX caps at 128)
const T: usize = 128; // threads/block for kernel (A)
const TC: usize = 64; // threads/block for chunk kernel (B)
const CHUNK: usize = 64; // KV positions per chunk (B)
const NW: usize = 32; // warps/head for kernel (C) — the hand-PTX num_warps_per_head.
// Swept on GB10 {4,8,16,32}: 32 is best (32*32=1024 = max threads/block); more
// warps/head = more KV-position parallelism within the single block-per-head.
// kernel (C) shared layout: [NW max][NW sum][NW * MAX_HD partial-out]
const SMEM_C: usize = 2 * NW + NW * MAX_HD;

// kernel (A) shared layout: [T max][T sum][T*MAX_HD acc]
const SMEM_A: usize = 2 * T + T * MAX_HD;
// chunk kernel (B) shared layout: [TC max][TC sum][TC*MAX_HD acc]
const SMEM_B: usize = 2 * TC + TC * MAX_HD;

#[cuda_module]
mod kernels {
    use super::*;

    // ---- (A) single-block-per-head online-softmax attention ----
    #[kernel]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    #[allow(clippy::too_many_arguments)]
    pub fn incremental_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &[f32],
        kv_len: u32,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        scale: f32,
    ) {
        static mut SMEM: SharedArray<f32, SMEM_A> = SharedArray::UNINIT;
        unsafe {
            let head = thread::blockIdx_x();
            if head >= n_heads {
                return;
            }
            let lane = thread::threadIdx_x() as usize;
            let hd = head_dim as usize;
            let group_size = n_heads / n_kv_heads;
            let kv_head = head / group_size;
            let kv_dim = (n_kv_heads * head_dim) as usize;
            let q_off = (head * head_dim) as usize;
            let kv_head_off = (kv_head * head_dim) as usize;

            let acc_base = 2 * T + lane * MAX_HD;
            let mut d = 0usize;
            while d < hd {
                SMEM[acc_base + d] = 0.0;
                d += 1;
            }
            let mut m = f32::NEG_INFINITY;
            let mut s = 0.0f32;

            let klen = kv_len as usize;
            let mut j = lane;
            while j < klen {
                let k_start = j * kv_dim + kv_head_off;
                let mut dot = 0.0f32;
                let mut dd = 0usize;
                while dd < hd {
                    dot += q[q_off + dd] * k[k_start + dd];
                    dd += 1;
                }
                let score = dot * scale;
                let new_m = if score > m { score } else { m };
                let corr = (m - new_m).exp();
                let p = (score - new_m).exp();
                s = s * corr + p;
                let v_start = j * kv_dim + kv_head_off;
                let mut e = 0usize;
                while e < hd {
                    SMEM[acc_base + e] = SMEM[acc_base + e] * corr + p * v[v_start + e];
                    e += 1;
                }
                m = new_m;
                j += T;
            }

            SMEM[lane] = m;
            SMEM[lane + T] = s;
            thread::sync_threads();

            if lane == 0 {
                let mut gmax = f32::NEG_INFINITY;
                let mut t = 0usize;
                while t < T {
                    let lm = SMEM[t];
                    if lm > gmax {
                        gmax = lm;
                    }
                    t += 1;
                }
                let mut gsum = 0.0f32;
                let mut t2 = 0usize;
                while t2 < T {
                    gsum += SMEM[t2 + T] * (SMEM[t2] - gmax).exp();
                    t2 += 1;
                }
                SMEM[0] = gmax;
                SMEM[1] = gsum;
            }
            thread::sync_threads();

            let gmax = SMEM[0];
            let inv = 1.0f32 / SMEM[1];
            let my_corr = (m - gmax).exp() * inv;
            let mut e2 = 0usize;
            while e2 < hd {
                SMEM[acc_base + e2] *= my_corr;
                e2 += 1;
            }
            thread::sync_threads();

            let out_off = (head * head_dim) as usize;
            let mut e = lane;
            while e < hd {
                let mut sum = 0.0f32;
                let mut t = 0usize;
                while t < T {
                    sum += SMEM[2 * T + t * MAX_HD + e];
                    t += 1;
                }
                let op = &mut *(out.as_ptr().add(out_off + e) as *mut f32);
                *op = sum;
                e += T;
            }
        }
    }

    // ---- (C) warp-coalesced multi-warp attention (faithful hand-PTX analog) ----
    // Grid = (n_heads,1,1). Block = (32*NW,1,1). Each warp processes a chunk of
    // KV positions; within a warp the 32 lanes COOPERATIVELY compute each
    // position's Q.K dot (lane l holds dims l, l+32, l+64, l+96 => coalesced K/V
    // loads), warp-reduce via shfl_xor, online-softmax, then cross-warp merge in
    // shared memory. This matches MultiWarpIncrementalAttentionKernel exactly and
    // fixes the un-coalesced memory pattern of kernels (A)/(B).
    #[kernel]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    #[allow(clippy::too_many_arguments)]
    pub fn attn_warp(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &[f32],
        kv_len: u32,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        scale: f32,
    ) {
        static mut SMEM: SharedArray<f32, SMEM_C> = SharedArray::UNINIT;
        unsafe {
            let head = thread::blockIdx_x();
            if head >= n_heads {
                return;
            }
            let tid = thread::threadIdx_x();
            let widx = (tid / 32) as usize;
            let lane = warp::lane_id();

            let hd = head_dim;
            let group_size = n_heads / n_kv_heads;
            let kv_head = head / group_size;
            let kv_dim = n_kv_heads * head_dim;
            let q_base = head * head_dim;
            let kv_base = kv_head * head_dim;

            // Lane l holds head-dim slots l, l+32, l+64, l+96 (up to head_dim<=128).
            let i0 = lane;
            let i1 = lane + 32;
            let i2 = lane + 64;
            let i3 = lane + 96;
            let b0 = i0 < hd;
            let b1 = i1 < hd;
            let b2 = i2 < hd;
            let b3 = i3 < hd;
            let q0 = if b0 { q[(q_base + i0) as usize] } else { 0.0 };
            let q1 = if b1 { q[(q_base + i1) as usize] } else { 0.0 };
            let q2 = if b2 { q[(q_base + i2) as usize] } else { 0.0 };
            let q3 = if b3 { q[(q_base + i3) as usize] } else { 0.0 };

            // chunk boundaries for this warp (ceil split, like the hand-PTX kernel)
            let nw = NW as u32;
            let chunk = (kv_len + nw - 1) / nw;
            let start = (widx as u32) * chunk;
            let mut end = start + chunk;
            if end > kv_len {
                end = kv_len;
            }

            let mut m = f32::NEG_INFINITY;
            let mut s = 0.0f32;
            let mut o0 = 0.0f32;
            let mut o1 = 0.0f32;
            let mut o2 = 0.0f32;
            let mut o3 = 0.0f32;

            let mut pos = start;
            while pos < end {
                let krow = pos * kv_dim + kv_base;
                let k0 = if b0 { k[(krow + i0) as usize] } else { 0.0 };
                let k1 = if b1 { k[(krow + i1) as usize] } else { 0.0 };
                let k2 = if b2 { k[(krow + i2) as usize] } else { 0.0 };
                let k3 = if b3 { k[(krow + i3) as usize] } else { 0.0 };
                // partial dot in this lane, then warp-reduce (xor butterfly).
                let mut dot = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
                dot += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, dot, 16);
                dot += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, dot, 8);
                dot += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, dot, 4);
                dot += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, dot, 2);
                dot += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, dot, 1);
                let score = dot * scale; // all lanes hold the full score now

                let new_m = if score > m { score } else { m };
                let corr = (m - new_m).exp();
                let p = (score - new_m).exp();
                s = s * corr + p;

                let v0 = if b0 { v[(krow + i0) as usize] } else { 0.0 };
                let v1 = if b1 { v[(krow + i1) as usize] } else { 0.0 };
                let v2 = if b2 { v[(krow + i2) as usize] } else { 0.0 };
                let v3 = if b3 { v[(krow + i3) as usize] } else { 0.0 };
                o0 = o0 * corr + p * v0;
                o1 = o1 * corr + p * v1;
                o2 = o2 * corr + p * v2;
                o3 = o3 * corr + p * v3;
                m = new_m;
                pos += 1;
            }

            // publish per-warp (max,sum) to shared (lane 0 of each warp).
            if lane == 0 {
                SMEM[widx] = m;
                SMEM[NW + widx] = s;
            }
            thread::sync_threads();

            // warp 0 lane 0 reduces global (max, sum) across warps. It reads every
            // warp's (max, sum) into registers FIRST, then writes gmax/gsum into
            // SMEM[0]/SMEM[1] (the max slots of warps 0/1, already consumed) — no
            // live data is clobbered.
            if widx == 0 && lane == 0 {
                let mut gmax = f32::NEG_INFINITY;
                let mut w = 0usize;
                while w < NW {
                    let lm = SMEM[w];
                    if lm > gmax {
                        gmax = lm;
                    }
                    w += 1;
                }
                let mut gsum = 0.0f32;
                let mut w2 = 0usize;
                while w2 < NW {
                    gsum += SMEM[NW + w2] * (SMEM[w2] - gmax).exp();
                    w2 += 1;
                }
                SMEM[0] = gmax;
                SMEM[1] = gsum;
            }
            thread::sync_threads();

            // Each warp scales its own (o0..o3) by exp(m - gmax) and stores into
            // the shared out area; warp 0 then sums across warps and normalizes.
            let gmax = SMEM[0];
            let gsum = SMEM[1];
            let inv = 1.0f32 / gsum;
            let cf = (m - gmax).exp();
            let ob = 2 * NW + widx * MAX_HD;
            if b0 {
                SMEM[ob + i0 as usize] = o0 * cf;
            }
            if b1 {
                SMEM[ob + i1 as usize] = o1 * cf;
            }
            if b2 {
                SMEM[ob + i2 as usize] = o2 * cf;
            }
            if b3 {
                SMEM[ob + i3 as usize] = o3 * cf;
            }
            thread::sync_threads();

            // warp 0 sums each element across warps, normalizes by gsum, writes out.
            if widx == 0 {
                let out_base = head * head_dim;
                let mut e = lane;
                while e < hd {
                    let mut acc = 0.0f32;
                    let mut w = 0usize;
                    while w < NW {
                        acc += SMEM[2 * NW + w * MAX_HD + e as usize];
                        w += 1;
                    }
                    let op = &mut *(out.as_ptr().add((out_base + e) as usize) as *mut f32);
                    *op = acc * inv;
                    e += 32;
                }
            }
        }
    }

    // ---- (C-rawptr) PMAT-883: raw-pointer C-style ABI variant of attn_warp ----
    //
    // Bit-identical compute to `attn_warp` but with a stable C-style ABI suitable
    // for `include_str!` -> CudaModule::from_ptx -> cuLaunchKernel (mirrors the
    // q4k-matvec raw-ptr promotion path). No fat slice pointers: the four buffers
    // are plain device pointers and the four dims + scale are scalars.
    //
    // Entry: attn_warp_rawptr(
    //   q:*const f32, k:*const f32, v:*const f32, out:*mut f32,
    //   kv_len:u32, head_dim:u32, n_heads:u32, n_kv_heads:u32, scale:f32)
    //
    // Layout = interleaved [seq, kv_dim] cache (kv_dim = n_kv_heads*head_dim),
    // EXACTLY the live serve `causal_attention_cached` layout. GQA mapping is
    // RUNTIME (kv_head = head/(n_heads/n_kv_heads)) — nothing is baked in, so the
    // single emitted PTX serves every (head_dim<=128, n_heads, n_kv_heads) shape.
    // Launch: grid=(n_heads,1,1), block=(32*NW,1,1) (NW=32 warps/head); out zeroed
    // is NOT required (the kernel writes every out[head*head_dim + e]).
    #[kernel]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    #[allow(clippy::too_many_arguments)]
    pub fn attn_warp_rawptr(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        out: *mut f32,
        kv_len: u32,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        scale: f32,
    ) {
        static mut SMEM2: SharedArray<f32, SMEM_C> = SharedArray::UNINIT;
        unsafe {
            let head = thread::blockIdx_x();
            if head >= n_heads {
                return;
            }
            let tid = thread::threadIdx_x();
            let widx = (tid / 32) as usize;
            let lane = warp::lane_id();

            let hd = head_dim;
            let group_size = n_heads / n_kv_heads;
            let kv_head = head / group_size;
            let kv_dim = n_kv_heads * head_dim;
            let q_base = head * head_dim;
            let kv_base = kv_head * head_dim;

            let i0 = lane;
            let i1 = lane + 32;
            let i2 = lane + 64;
            let i3 = lane + 96;
            let b0 = i0 < hd;
            let b1 = i1 < hd;
            let b2 = i2 < hd;
            let b3 = i3 < hd;
            let rdq = |o: u32| *q.add(o as usize);
            let q0 = if b0 { rdq(q_base + i0) } else { 0.0 };
            let q1 = if b1 { rdq(q_base + i1) } else { 0.0 };
            let q2 = if b2 { rdq(q_base + i2) } else { 0.0 };
            let q3 = if b3 { rdq(q_base + i3) } else { 0.0 };

            let nw = NW as u32;
            let chunk = (kv_len + nw - 1) / nw;
            let start = (widx as u32) * chunk;
            let mut end = start + chunk;
            if end > kv_len {
                end = kv_len;
            }

            let mut m = f32::NEG_INFINITY;
            let mut s = 0.0f32;
            let mut o0 = 0.0f32;
            let mut o1 = 0.0f32;
            let mut o2 = 0.0f32;
            let mut o3 = 0.0f32;

            let rdk = |o: u32| *k.add(o as usize);
            let rdv = |o: u32| *v.add(o as usize);
            let mut pos = start;
            while pos < end {
                let krow = pos * kv_dim + kv_base;
                let k0 = if b0 { rdk(krow + i0) } else { 0.0 };
                let k1 = if b1 { rdk(krow + i1) } else { 0.0 };
                let k2 = if b2 { rdk(krow + i2) } else { 0.0 };
                let k3 = if b3 { rdk(krow + i3) } else { 0.0 };
                let mut dot = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
                dot += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, dot, 16);
                dot += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, dot, 8);
                dot += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, dot, 4);
                dot += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, dot, 2);
                dot += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, dot, 1);
                let score = dot * scale;

                let new_m = if score > m { score } else { m };
                let corr = (m - new_m).exp();
                let p = (score - new_m).exp();
                s = s * corr + p;

                let v0 = if b0 { rdv(krow + i0) } else { 0.0 };
                let v1 = if b1 { rdv(krow + i1) } else { 0.0 };
                let v2 = if b2 { rdv(krow + i2) } else { 0.0 };
                let v3 = if b3 { rdv(krow + i3) } else { 0.0 };
                o0 = o0 * corr + p * v0;
                o1 = o1 * corr + p * v1;
                o2 = o2 * corr + p * v2;
                o3 = o3 * corr + p * v3;
                m = new_m;
                pos += 1;
            }

            if lane == 0 {
                SMEM2[widx] = m;
                SMEM2[NW + widx] = s;
            }
            thread::sync_threads();

            if widx == 0 && lane == 0 {
                let mut gmax = f32::NEG_INFINITY;
                let mut w = 0usize;
                while w < NW {
                    let lm = SMEM2[w];
                    if lm > gmax {
                        gmax = lm;
                    }
                    w += 1;
                }
                let mut gsum = 0.0f32;
                let mut w2 = 0usize;
                while w2 < NW {
                    gsum += SMEM2[NW + w2] * (SMEM2[w2] - gmax).exp();
                    w2 += 1;
                }
                SMEM2[0] = gmax;
                SMEM2[1] = gsum;
            }
            thread::sync_threads();

            let gmax = SMEM2[0];
            let gsum = SMEM2[1];
            let inv = 1.0f32 / gsum;
            let cf = (m - gmax).exp();
            let ob = 2 * NW + widx * MAX_HD;
            if b0 {
                SMEM2[ob + i0 as usize] = o0 * cf;
            }
            if b1 {
                SMEM2[ob + i1 as usize] = o1 * cf;
            }
            if b2 {
                SMEM2[ob + i2 as usize] = o2 * cf;
            }
            if b3 {
                SMEM2[ob + i3 as usize] = o3 * cf;
            }
            thread::sync_threads();

            if widx == 0 {
                let out_base = head * head_dim;
                let mut e = lane;
                while e < hd {
                    let mut acc = 0.0f32;
                    let mut w = 0usize;
                    while w < NW {
                        acc += SMEM2[2 * NW + w * MAX_HD + e as usize];
                        w += 1;
                    }
                    *out.add((out_base + e) as usize) = acc * inv;
                    e += 32;
                }
            }
        }
    }

    // ---- (C2) PMAT-884 SEPARATE-HEAD warp-coalesced attention (LIVE cache) ----
    //
    // Identical compute to `attn_warp_rawptr` (kernel C), but indexes the LIVE
    // serve KV cache layout `[num_kv_heads, max_len, head_dim]` directly — NO
    // interleave/gather adapter. This is PMAT-883 promotion option (b): the only
    // change vs attn_warp_rawptr is the K/V row address:
    //
    //   interleaved (rawptr): krow = pos*kv_dim       + kv_head*head_dim
    //   separate-head (this):  krow = kv_head*kv_stride + pos*head_dim
    //
    // where kv_stride = max_len*head_dim is the per-kv-head stride of the live
    // cache (`CudaExecutor::incremental_attention_async`). Adding the explicit
    // `kv_stride` param means this ONE PTX serves any (max_len, head_dim<=128,
    // n_heads, n_kv_heads) decode shape with the GQA mapping still RUNTIME.
    //
    // Entry: attn_warp_sephead_rawptr(
    //   q:*const f32, k:*const f32, v:*const f32, out:*mut f32,
    //   kv_len:u32, head_dim:u32, n_heads:u32, n_kv_heads:u32,
    //   kv_stride:u32, scale:f32)
    // Launch: grid=(n_heads,1,1), block=(32*NW,1,1).
    #[kernel]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    #[allow(clippy::too_many_arguments)]
    pub fn attn_warp_sephead_rawptr(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        out: *mut f32,
        kv_len: u32,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        kv_stride: u32,
        scale: f32,
    ) {
        static mut SMEM3: SharedArray<f32, SMEM_C> = SharedArray::UNINIT;
        unsafe {
            let head = thread::blockIdx_x();
            if head >= n_heads {
                return;
            }
            let tid = thread::threadIdx_x();
            let widx = (tid / 32) as usize;
            let lane = warp::lane_id();

            let hd = head_dim;
            let group_size = n_heads / n_kv_heads;
            let kv_head = head / group_size;
            let q_base = head * head_dim;
            // SEPARATE-HEAD: base of this kv_head's slab = kv_head * kv_stride.
            let kv_base = kv_head * kv_stride;

            let i0 = lane;
            let i1 = lane + 32;
            let i2 = lane + 64;
            let i3 = lane + 96;
            let b0 = i0 < hd;
            let b1 = i1 < hd;
            let b2 = i2 < hd;
            let b3 = i3 < hd;
            let rdq = |o: u32| *q.add(o as usize);
            let q0 = if b0 { rdq(q_base + i0) } else { 0.0 };
            let q1 = if b1 { rdq(q_base + i1) } else { 0.0 };
            let q2 = if b2 { rdq(q_base + i2) } else { 0.0 };
            let q3 = if b3 { rdq(q_base + i3) } else { 0.0 };

            let nw = NW as u32;
            let chunk = (kv_len + nw - 1) / nw;
            let start = (widx as u32) * chunk;
            let mut end = start + chunk;
            if end > kv_len {
                end = kv_len;
            }

            let mut m = f32::NEG_INFINITY;
            let mut s = 0.0f32;
            let mut o0 = 0.0f32;
            let mut o1 = 0.0f32;
            let mut o2 = 0.0f32;
            let mut o3 = 0.0f32;

            let rdk = |o: u32| *k.add(o as usize);
            let rdv = |o: u32| *v.add(o as usize);
            let mut pos = start;
            while pos < end {
                // SEPARATE-HEAD: row = kv_head_slab + pos*head_dim.
                let krow = kv_base + pos * head_dim;
                let k0 = if b0 { rdk(krow + i0) } else { 0.0 };
                let k1 = if b1 { rdk(krow + i1) } else { 0.0 };
                let k2 = if b2 { rdk(krow + i2) } else { 0.0 };
                let k3 = if b3 { rdk(krow + i3) } else { 0.0 };
                let mut dot = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
                dot += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, dot, 16);
                dot += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, dot, 8);
                dot += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, dot, 4);
                dot += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, dot, 2);
                dot += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, dot, 1);
                let score = dot * scale;

                let new_m = if score > m { score } else { m };
                let corr = (m - new_m).exp();
                let p = (score - new_m).exp();
                s = s * corr + p;

                let v0 = if b0 { rdv(krow + i0) } else { 0.0 };
                let v1 = if b1 { rdv(krow + i1) } else { 0.0 };
                let v2 = if b2 { rdv(krow + i2) } else { 0.0 };
                let v3 = if b3 { rdv(krow + i3) } else { 0.0 };
                o0 = o0 * corr + p * v0;
                o1 = o1 * corr + p * v1;
                o2 = o2 * corr + p * v2;
                o3 = o3 * corr + p * v3;
                m = new_m;
                pos += 1;
            }

            if lane == 0 {
                SMEM3[widx] = m;
                SMEM3[NW + widx] = s;
            }
            thread::sync_threads();

            if widx == 0 && lane == 0 {
                let mut gmax = f32::NEG_INFINITY;
                let mut w = 0usize;
                while w < NW {
                    let lm = SMEM3[w];
                    if lm > gmax {
                        gmax = lm;
                    }
                    w += 1;
                }
                let mut gsum = 0.0f32;
                let mut w2 = 0usize;
                while w2 < NW {
                    gsum += SMEM3[NW + w2] * (SMEM3[w2] - gmax).exp();
                    w2 += 1;
                }
                SMEM3[0] = gmax;
                SMEM3[1] = gsum;
            }
            thread::sync_threads();

            let gmax = SMEM3[0];
            let gsum = SMEM3[1];
            let inv = 1.0f32 / gsum;
            let cf = (m - gmax).exp();
            let ob = 2 * NW + widx * MAX_HD;
            if b0 {
                SMEM3[ob + i0 as usize] = o0 * cf;
            }
            if b1 {
                SMEM3[ob + i1 as usize] = o1 * cf;
            }
            if b2 {
                SMEM3[ob + i2 as usize] = o2 * cf;
            }
            if b3 {
                SMEM3[ob + i3 as usize] = o3 * cf;
            }
            thread::sync_threads();

            if widx == 0 {
                let out_base = head * head_dim;
                let mut e = lane;
                while e < hd {
                    let mut acc = 0.0f32;
                    let mut w = 0usize;
                    while w < NW {
                        acc += SMEM3[2 * NW + w * MAX_HD + e as usize];
                        w += 1;
                    }
                    *out.add((out_base + e) as usize) = acc * inv;
                    e += 32;
                }
            }
        }
    }

    // ---- (B1) Flash-Decoding chunk kernel ----
    // grid = (n_heads * n_chunks, 1, 1), block = (TC,1,1).
    // Each block reduces one CHUNK of KV positions for one head -> partial
    // (max, sum, out[head_dim]) written to global partials buffers.
    //   p_max : [n_heads * n_chunks]
    //   p_sum : [n_heads * n_chunks]
    //   p_out : [n_heads * n_chunks * head_dim]
    #[kernel]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    #[allow(clippy::too_many_arguments)]
    pub fn attn_chunk(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        p_max: &[f32],
        p_sum: &[f32],
        p_out: &[f32],
        kv_len: u32,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        n_chunks: u32,
        scale: f32,
    ) {
        static mut SMEM: SharedArray<f32, SMEM_B> = SharedArray::UNINIT;
        unsafe {
            let bid = thread::blockIdx_x();
            let head = bid / n_chunks;
            let chunk = bid % n_chunks;
            if head >= n_heads {
                return;
            }
            let lane = thread::threadIdx_x() as usize;
            let hd = head_dim as usize;
            let group_size = n_heads / n_kv_heads;
            let kv_head = head / group_size;
            let kv_dim = (n_kv_heads * head_dim) as usize;
            let q_off = (head * head_dim) as usize;
            let kv_head_off = (kv_head * head_dim) as usize;

            let chunk_start = (chunk as usize) * CHUNK;
            let mut chunk_end = chunk_start + CHUNK;
            let klen = kv_len as usize;
            if chunk_end > klen {
                chunk_end = klen;
            }

            let acc_base = 2 * TC + lane * MAX_HD;
            let mut d = 0usize;
            while d < hd {
                SMEM[acc_base + d] = 0.0;
                d += 1;
            }
            let mut m = f32::NEG_INFINITY;
            let mut s = 0.0f32;

            // thread strides positions WITHIN this chunk
            let mut j = chunk_start + lane;
            while j < chunk_end {
                let k_start = j * kv_dim + kv_head_off;
                let mut dot = 0.0f32;
                let mut dd = 0usize;
                while dd < hd {
                    dot += q[q_off + dd] * k[k_start + dd];
                    dd += 1;
                }
                let score = dot * scale;
                let new_m = if score > m { score } else { m };
                let corr = (m - new_m).exp();
                let p = (score - new_m).exp();
                s = s * corr + p;
                let v_start = j * kv_dim + kv_head_off;
                let mut e = 0usize;
                while e < hd {
                    SMEM[acc_base + e] = SMEM[acc_base + e] * corr + p * v[v_start + e];
                    e += 1;
                }
                m = new_m;
                j += TC;
            }

            SMEM[lane] = m;
            SMEM[lane + TC] = s;
            thread::sync_threads();

            // reduce within block to chunk-partial (max, sum) and rescale accs
            if lane == 0 {
                let mut gmax = f32::NEG_INFINITY;
                let mut t = 0usize;
                while t < TC {
                    let lm = SMEM[t];
                    if lm > gmax {
                        gmax = lm;
                    }
                    t += 1;
                }
                let mut gsum = 0.0f32;
                let mut t2 = 0usize;
                while t2 < TC {
                    gsum += SMEM[t2 + TC] * (SMEM[t2] - gmax).exp();
                    t2 += 1;
                }
                SMEM[0] = gmax;
                SMEM[1] = gsum;
            }
            thread::sync_threads();

            let gmax = SMEM[0];
            let gsum = SMEM[1];
            // rescale each thread's acc to the chunk's gmax (NOT normalized yet;
            // normalization happens in the reduce kernel after cross-chunk merge).
            let my_corr = (m - gmax).exp();
            let mut e2 = 0usize;
            while e2 < hd {
                SMEM[acc_base + e2] *= my_corr;
                e2 += 1;
            }
            thread::sync_threads();

            // parallel write of the chunk's UNNORMALIZED out[head_dim] (= sum over
            // positions of exp(score-gmax)*V) plus (gmax, gsum) for this chunk.
            let pbase = (bid as usize) * hd;
            let mut e = lane;
            while e < hd {
                let mut sum = 0.0f32;
                let mut t = 0usize;
                while t < TC {
                    sum += SMEM[2 * TC + t * MAX_HD + e];
                    t += 1;
                }
                let op = &mut *(p_out.as_ptr().add(pbase + e) as *mut f32);
                *op = sum;
                e += TC;
            }
            if lane == 0 {
                let pm = &mut *(p_max.as_ptr().add(bid as usize) as *mut f32);
                *pm = gmax;
                let ps = &mut *(p_sum.as_ptr().add(bid as usize) as *mut f32);
                *ps = gsum;
            }
        }
    }

    // ---- (B2) Flash-Decoding reduce kernel ----
    // grid = (n_heads, 1, 1), block = (head_dim, 1, 1).
    // Merges the n_chunks partials for one head into final out[head_dim].
    #[kernel]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    #[allow(clippy::too_many_arguments)]
    pub fn attn_reduce(
        p_max: &[f32],
        p_sum: &[f32],
        p_out: &[f32],
        out: &[f32],
        head_dim: u32,
        n_chunks: u32,
    ) {
        unsafe {
            let head = thread::blockIdx_x();
            let e = thread::threadIdx_x() as usize;
            let hd = head_dim as usize;
            if e >= hd {
                return;
            }
            let nc = n_chunks as usize;
            let cbase = (head as usize) * nc;

            // global max over chunks
            let mut gmax = f32::NEG_INFINITY;
            let mut c = 0usize;
            while c < nc {
                let cm = p_max[cbase + c];
                if cm > gmax {
                    gmax = cm;
                }
                c += 1;
            }
            // global sum + merged numerator for element e
            let mut gsum = 0.0f32;
            let mut acc = 0.0f32;
            let mut c2 = 0usize;
            while c2 < nc {
                let cm = p_max[cbase + c2];
                let cs = p_sum[cbase + c2];
                let scale_c = (cm - gmax).exp();
                gsum += cs * scale_c;
                let po = p_out[(cbase + c2) * hd + e];
                acc += po * scale_c;
                c2 += 1;
            }
            let out_off = (head as usize) * hd + e;
            let op = &mut *(out.as_ptr().add(out_off) as *mut f32);
            *op = acc / gsum;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference — `causal_attention_cached` decode math (single new token).
// ---------------------------------------------------------------------------
fn cpu_incremental_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kv_len: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    scale: f32,
) -> Vec<f32> {
    let group_size = n_heads / n_kv_heads;
    let kv_dim = n_kv_heads * head_dim;
    let q_dim = n_heads * head_dim;
    let mut out = vec![0.0f32; q_dim];
    for head in 0..n_heads {
        let kv_head = head / group_size;
        let q_off = head * head_dim;
        let kv_head_off = kv_head * head_dim;
        let mut scores = vec![0.0f32; kv_len];
        for (j, sc) in scores.iter_mut().enumerate() {
            let k_start = j * kv_dim + kv_head_off;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[q_off + d] * k[k_start + d];
            }
            *sc = dot * scale;
        }
        let mut maxs = f32::NEG_INFINITY;
        for &x in &scores {
            if x > maxs {
                maxs = x;
            }
        }
        let mut esum = 0.0f32;
        for x in scores.iter_mut() {
            *x = (*x - maxs).exp();
            esum += *x;
        }
        for x in scores.iter_mut() {
            *x /= esum;
        }
        let out_off = head * head_dim;
        for (j, &w) in scores.iter().enumerate() {
            let v_start = j * kv_dim + kv_head_off;
            for d in 0..head_dim {
                out[out_off + d] += w * v[v_start + d];
            }
        }
    }
    out
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn make_qkv(
    kv_len: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    seed: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let g = |i: usize| -> f32 {
        let x = (i
            .wrapping_mul(2654435761)
            .wrapping_add(seed.wrapping_mul(40503)))
            & 0xFFFF;
        (x as f32 / 32768.0) - 1.0
    };
    let q: Vec<f32> = (0..q_dim).map(g).collect();
    let k: Vec<f32> = (0..kv_len * kv_dim).map(|i| g(i + 7)).collect();
    let v: Vec<f32> = (0..kv_len * kv_dim).map(|i| g(i + 9973)).collect();
    (q, k, v)
}

struct Devs {
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
}

#[allow(clippy::too_many_arguments)]
fn parity_and_perf(
    ctx: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    kv_len: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    seed: usize,
    do_perf: bool,
    variant: u8, // 0 = single-block (A), 1 = split-K (B), 2 = warp-coalesced (C)
) -> (f32, f32, f32, f64) {
    let stream = ctx.default_stream();
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let (q, k, v) = make_qkv(kv_len, head_dim, n_heads, n_kv_heads, seed);
    let q_dim = n_heads * head_dim;

    let d = Devs {
        q: DeviceBuffer::from_host(&stream, &q).unwrap(),
        k: DeviceBuffer::from_host(&stream, &k).unwrap(),
        v: DeviceBuffer::from_host(&stream, &v).unwrap(),
        out: DeviceBuffer::<f32>::zeroed(&stream, q_dim).unwrap(),
    };

    let n_chunks = kv_len.div_ceil(CHUNK);
    let p_max = DeviceBuffer::<f32>::zeroed(&stream, n_heads * n_chunks).unwrap();
    let p_sum = DeviceBuffer::<f32>::zeroed(&stream, n_heads * n_chunks).unwrap();
    let p_out = DeviceBuffer::<f32>::zeroed(&stream, n_heads * n_chunks * head_dim).unwrap();

    let launch = |d: &Devs| {
        if variant == 2 {
            // (C) warp-coalesced: block = 32*NW threads, grid = n_heads.
            let cfg = LaunchConfig {
                grid_dim: (n_heads as u32, 1, 1),
                block_dim: ((32 * NW) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            module
                .attn_warp(
                    &stream,
                    cfg,
                    &d.q,
                    &d.k,
                    &d.v,
                    &d.out,
                    kv_len as u32,
                    head_dim as u32,
                    n_heads as u32,
                    n_kv_heads as u32,
                    scale,
                )
                .expect("attn_warp");
        } else if variant == 1 {
            let cfg_c = LaunchConfig {
                grid_dim: ((n_heads * n_chunks) as u32, 1, 1),
                block_dim: (TC as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            module
                .attn_chunk(
                    &stream,
                    cfg_c,
                    &d.q,
                    &d.k,
                    &d.v,
                    &p_max,
                    &p_sum,
                    &p_out,
                    kv_len as u32,
                    head_dim as u32,
                    n_heads as u32,
                    n_kv_heads as u32,
                    n_chunks as u32,
                    scale,
                )
                .expect("chunk");
            let cfg_r = LaunchConfig {
                grid_dim: (n_heads as u32, 1, 1),
                block_dim: (head_dim as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            module
                .attn_reduce(
                    &stream,
                    cfg_r,
                    &p_max,
                    &p_sum,
                    &p_out,
                    &d.out,
                    head_dim as u32,
                    n_chunks as u32,
                )
                .expect("reduce");
        } else {
            let cfg = LaunchConfig {
                grid_dim: (n_heads as u32, 1, 1),
                block_dim: (T as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            module
                .incremental_attention(
                    &stream,
                    cfg,
                    &d.q,
                    &d.k,
                    &d.v,
                    &d.out,
                    kv_len as u32,
                    head_dim as u32,
                    n_heads as u32,
                    n_kv_heads as u32,
                    scale,
                )
                .expect("attn");
        }
    };

    launch(&d);
    let got = d.out.to_host_vec(&stream).unwrap();
    let want = cpu_incremental_attention(&q, &k, &v, kv_len, head_dim, n_heads, n_kv_heads, scale);

    let cos = cosine_similarity(&got, &want);
    let mut maxabs_ref = 0.0f32;
    for &x in &want {
        if x.abs() > maxabs_ref {
            maxabs_ref = x.abs();
        }
    }
    let mut maxdiff = 0.0f32;
    for i in 0..q_dim {
        let dd = (got[i] - want[i]).abs();
        if dd > maxdiff {
            maxdiff = dd;
        }
    }
    let tol = 1e-4f32 * maxabs_ref.max(1e-6);

    let mut med = 0.0f64;
    if do_perf {
        // GPU-event timing: measures true on-device kernel time, not host launch
        // overhead. 50 timed launches per rep, median of 5 reps.
        for _ in 0..20 {
            launch(&d);
        }
        let _ = d.out.to_host_vec(&stream).unwrap();
        let iters = 50usize;
        let reps = 5;
        let mut times = Vec::new();
        for _ in 0..reps {
            let start = stream
                .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
                .unwrap();
            for _ in 0..iters {
                launch(&d);
            }
            let end = stream
                .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
                .unwrap();
            let ms = start.elapsed_ms(&end).unwrap() as f64;
            times.push(ms * 1000.0 / iters as f64); // us/launch
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        med = times[times.len() / 2];
    }
    (cos, maxdiff, tol, med)
}

// ---------------------------------------------------------------------------
// TRUE hand-PTX A/B: load the emitted multi_warp_attention PTX and launch it via
// the raw driver path. The hand-PTX uses a SEPARATE-HEAD K/V layout
// [kv_head, max_seq_len, head_dim] (kv_stride = max_seq_len*head_dim), Q/out
// [n_heads, head_dim]. We rebuild that layout from the same logical Q/K/V, run
// the kernel, verify vs the CPU reference, and time it with GPU events.
// Returns (cos, maxdiff, tol, us/launch).
#[allow(clippy::too_many_arguments)]
fn handptx_ab(
    ctx: &std::sync::Arc<CudaContext>,
    ptx_path: &str,
    max_seq_len: usize,
    kv_len: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    num_warps: usize,
    seed: usize,
) -> Option<(f32, f32, f32, f64)> {
    let stream = ctx.default_stream();
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let (q, k_il, v_il) = make_qkv(kv_len, head_dim, n_heads, n_kv_heads, seed);

    // Re-pack interleaved [seq, kv_dim] -> separate-head [kv_head, max_seq_len, head_dim].
    let kv_dim = n_kv_heads * head_dim;
    let kv_stride = max_seq_len * head_dim;
    let mut k_sep = vec![0.0f32; n_kv_heads * kv_stride];
    let mut v_sep = vec![0.0f32; n_kv_heads * kv_stride];
    for j in 0..kv_len {
        for h in 0..n_kv_heads {
            let src = j * kv_dim + h * head_dim;
            let dst = h * kv_stride + j * head_dim;
            k_sep[dst..dst + head_dim].copy_from_slice(&k_il[src..src + head_dim]);
            v_sep[dst..dst + head_dim].copy_from_slice(&v_il[src..src + head_dim]);
        }
    }

    let ptx = std::fs::read_to_string(ptx_path).ok()?;
    let module = ctx.load_module_from_ptx_src(&ptx).ok()?;
    let func = module.load_function("multi_warp_attention").ok()?;

    let q_dev = DeviceBuffer::from_host(&stream, &q).ok()?;
    let k_dev = DeviceBuffer::from_host(&stream, &k_sep).ok()?;
    let v_dev = DeviceBuffer::from_host(&stream, &v_sep).ok()?;
    let out_dev = DeviceBuffer::<f32>::zeroed(&stream, n_heads * head_dim).ok()?;

    let mut q_ptr = q_dev.cu_deviceptr();
    let mut k_ptr = k_dev.cu_deviceptr();
    let mut v_ptr = v_dev.cu_deviceptr();
    let mut o_ptr = out_dev.cu_deviceptr();
    let mut seq_len = kv_len as u32;
    let grid = (n_heads as u32, 1u32, 1u32);
    let block = ((32 * num_warps) as u32, 1u32, 1u32);

    let mut do_launch = || unsafe {
        let mut params: [*mut std::ffi::c_void; 5] = [
            &mut q_ptr as *mut _ as *mut std::ffi::c_void,
            &mut k_ptr as *mut _ as *mut std::ffi::c_void,
            &mut v_ptr as *mut _ as *mut std::ffi::c_void,
            &mut o_ptr as *mut _ as *mut std::ffi::c_void,
            &mut seq_len as *mut _ as *mut std::ffi::c_void,
        ];
        cuda_core::launch_kernel_on_stream(&func, grid, block, 0, &stream, &mut params)
    };

    do_launch().ok()?;
    let got = out_dev.to_host_vec(&stream).ok()?;
    let want = cpu_incremental_attention(
        &q, &k_il, &v_il, kv_len, head_dim, n_heads, n_kv_heads, scale,
    );
    let cos = cosine_similarity(&got, &want);
    let mut maxabs_ref = 0.0f32;
    for &x in &want {
        if x.abs() > maxabs_ref {
            maxabs_ref = x.abs();
        }
    }
    let mut maxdiff = 0.0f32;
    for i in 0..n_heads * head_dim {
        let dd = (got[i] - want[i]).abs();
        if dd > maxdiff {
            maxdiff = dd;
        }
    }
    let tol = 1e-4f32 * maxabs_ref.max(1e-6);

    for _ in 0..20 {
        do_launch().ok()?;
    }
    let _ = out_dev.to_host_vec(&stream).ok()?;
    let iters = 50usize;
    let reps = 5;
    let mut times = Vec::new();
    for _ in 0..reps {
        let start = stream
            .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
            .ok()?;
        for _ in 0..iters {
            do_launch().ok()?;
        }
        let end = stream
            .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
            .ok()?;
        let ms = start.elapsed_ms(&end).ok()? as f64;
        times.push(ms * 1000.0 / iters as f64);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some((cos, maxdiff, tol, times[times.len() / 2]))
}

// ---------------------------------------------------------------------------
// PMAT-883: standalone embeddable PTX for `attn_warp_rawptr`.
//
// Kernel C uses f32 `exp()` (softmax) which cuda-oxide lowers to a libdevice
// `__nv_expf` call; the pipeline therefore emits NVVM IR (.ll) and skips llc,
// leaving libNVVM lowering to the consumer. Producing a SELF-CONTAINED `.ptx`
// (no extern `__nv_*`) for `include_str!` -> CudaModule::from_ptx requires the
// libdevice-link + llc lowering done by `emit_ptx.sh` (the source-of-record emit
// path). This Rust `emit-ptx` mode just points at that script — it does not
// reimplement the LLVM toolchain orchestration in Rust (muda).
// ---------------------------------------------------------------------------
fn emit_ptx_artifact(out_path: &str) {
    eprintln!(
        "[PMAT-883] The standalone PTX for `attn_warp_rawptr` is emitted by the\n\
         documented source-of-record script (kernel C uses libdevice __nv_expf, so\n\
         cargo oxide emits NVVM IR and skips llc):\n\n\
         \x20   ./emit_ptx.sh {out_path}\n\n\
         which runs: cargo oxide pipeline -> llvm-link libdevice ->\n\
         opt internalize/nvvm-reflect/O3 -> llc nvptx64 sm_121 -> trim -> ptxas.\n\
         Then re-run `cargo oxide run` to execute the 3-way parity gate against it."
    );
    std::process::exit(0);
}

// ---------------------------------------------------------------------------
// PMAT-883: 3-way parity gate.
//
// Loads the EMITTED standalone oxide PTX (the source-of-record artifact) via the
// exact aprender consumption path (`load_module_from_ptx_src` -> resolve entry ->
// `cuLaunchKernel`), the hand-PTX `multi_warp_attention` baseline, and the CPU
// `causal_attention_cached` reference; runs all three on identical Q/K/V across
// the 9 decode configs and asserts oxide-PTX == hand-PTX == CPU within
// cos>=0.99 AND maxdiff < 1e-4*max|ref|.
// ---------------------------------------------------------------------------

/// Launch the EMITTED oxide PTX (raw-ptr `attn_warp_rawptr` entry, interleaved
/// [seq, kv_dim] layout) and return (cos, maxdiff, tol) vs the CPU reference.
#[allow(clippy::too_many_arguments)]
fn oxide_ptx_parity(
    ctx: &std::sync::Arc<CudaContext>,
    ptx_path: &str,
    kv_len: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    seed: usize,
) -> Option<(f32, f32, f32)> {
    let stream = ctx.default_stream();
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let (q, k, v) = make_qkv(kv_len, head_dim, n_heads, n_kv_heads, seed);
    let q_dim = n_heads * head_dim;

    let ptx = std::fs::read_to_string(ptx_path).ok()?;
    let module = ctx.load_module_from_ptx_src(&ptx).ok()?;
    let func = module.load_function("attn_warp_rawptr").ok()?;

    let q_dev = DeviceBuffer::from_host(&stream, &q).ok()?;
    let k_dev = DeviceBuffer::from_host(&stream, &k).ok()?;
    let v_dev = DeviceBuffer::from_host(&stream, &v).ok()?;
    let out_dev = DeviceBuffer::<f32>::zeroed(&stream, q_dim).ok()?;

    let mut q_ptr = q_dev.cu_deviceptr();
    let mut k_ptr = k_dev.cu_deviceptr();
    let mut v_ptr = v_dev.cu_deviceptr();
    let mut o_ptr = out_dev.cu_deviceptr();
    let mut kv_len_u = kv_len as u32;
    let mut head_dim_u = head_dim as u32;
    let mut n_heads_u = n_heads as u32;
    let mut n_kv_heads_u = n_kv_heads as u32;
    let mut scale_v = scale;
    let grid = (n_heads as u32, 1u32, 1u32);
    let block = ((32 * NW) as u32, 1u32, 1u32);

    unsafe {
        let mut params: [*mut std::ffi::c_void; 9] = [
            &mut q_ptr as *mut _ as *mut std::ffi::c_void,
            &mut k_ptr as *mut _ as *mut std::ffi::c_void,
            &mut v_ptr as *mut _ as *mut std::ffi::c_void,
            &mut o_ptr as *mut _ as *mut std::ffi::c_void,
            &mut kv_len_u as *mut _ as *mut std::ffi::c_void,
            &mut head_dim_u as *mut _ as *mut std::ffi::c_void,
            &mut n_heads_u as *mut _ as *mut std::ffi::c_void,
            &mut n_kv_heads_u as *mut _ as *mut std::ffi::c_void,
            &mut scale_v as *mut _ as *mut std::ffi::c_void,
        ];
        cuda_core::launch_kernel_on_stream(&func, grid, block, 0, &stream, &mut params).ok()?;
    }
    let got = out_dev.to_host_vec(&stream).ok()?;
    let want = cpu_incremental_attention(&q, &k, &v, kv_len, head_dim, n_heads, n_kv_heads, scale);
    let cos = cosine_similarity(&got, &want);
    let mut maxabs_ref = 0.0f32;
    for &x in &want {
        if x.abs() > maxabs_ref {
            maxabs_ref = x.abs();
        }
    }
    let mut maxdiff = 0.0f32;
    for i in 0..q_dim {
        let dd = (got[i] - want[i]).abs();
        if dd > maxdiff {
            maxdiff = dd;
        }
    }
    let tol = 1e-4f32 * maxabs_ref.max(1e-6);
    Some((cos, maxdiff, tol))
}

/// PMAT-884: launch the EMITTED oxide separate-head PTX
/// (`attn_warp_sephead_rawptr`, 10-param ABI) against the LIVE serve KV-cache
/// layout `[num_kv_heads, max_len, head_dim]` and return (cos, maxdiff, tol) vs
/// the CPU reference.
///
/// This is the on-device parity check against the REAL cache layout (PMAT-883
/// promotion criterion 1): the same Q/K/V logical data is re-packed into the
/// separate-head slab layout (kv_stride = max_len*head_dim) — EXACTLY how
/// `CudaExecutor::incremental_attention_async` stores the cache — and the kernel
/// indexes it directly (no interleave/gather adapter). Compares vs the CPU
/// `causal_attention_cached` reference (which consumes the interleaved logical
/// data), so a PASS proves the sephead kernel ≡ CPU on the live layout.
#[allow(clippy::too_many_arguments)]
fn sephead_ptx_parity(
    ctx: &std::sync::Arc<CudaContext>,
    ptx_path: &str,
    max_len: usize,
    kv_len: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    seed: usize,
) -> Option<(f32, f32, f32)> {
    let stream = ctx.default_stream();
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    // Logical Q/K/V in the interleaved [seq, kv_dim] layout (CPU-ref layout).
    let (q, k_il, v_il) = make_qkv(kv_len, head_dim, n_heads, n_kv_heads, seed);
    let q_dim = n_heads * head_dim;

    // Re-pack interleaved [seq, kv_dim] -> separate-head [kv_head, max_len, head_dim]
    // (kv_stride = max_len*head_dim). EXACTLY the live `incremental_attention_async`
    // cache slab. Positions [kv_len..max_len) are never read (kernel loops to kv_len),
    // so the unfilled tail is harmless (matches the live cache, which only fills up
    // to the current decode length).
    let kv_dim = n_kv_heads * head_dim;
    let kv_stride = max_len * head_dim;
    let mut k_sep = vec![0.0f32; n_kv_heads * kv_stride];
    let mut v_sep = vec![0.0f32; n_kv_heads * kv_stride];
    for j in 0..kv_len {
        for h in 0..n_kv_heads {
            let src = j * kv_dim + h * head_dim;
            let dst = h * kv_stride + j * head_dim;
            k_sep[dst..dst + head_dim].copy_from_slice(&k_il[src..src + head_dim]);
            v_sep[dst..dst + head_dim].copy_from_slice(&v_il[src..src + head_dim]);
        }
    }

    let ptx = std::fs::read_to_string(ptx_path).ok()?;
    let module = ctx.load_module_from_ptx_src(&ptx).ok()?;
    let func = module.load_function("attn_warp_sephead_rawptr").ok()?;

    let q_dev = DeviceBuffer::from_host(&stream, &q).ok()?;
    let k_dev = DeviceBuffer::from_host(&stream, &k_sep).ok()?;
    let v_dev = DeviceBuffer::from_host(&stream, &v_sep).ok()?;
    let out_dev = DeviceBuffer::<f32>::zeroed(&stream, q_dim).ok()?;

    let mut q_ptr = q_dev.cu_deviceptr();
    let mut k_ptr = k_dev.cu_deviceptr();
    let mut v_ptr = v_dev.cu_deviceptr();
    let mut o_ptr = out_dev.cu_deviceptr();
    let mut kv_len_u = kv_len as u32;
    let mut head_dim_u = head_dim as u32;
    let mut n_heads_u = n_heads as u32;
    let mut n_kv_heads_u = n_kv_heads as u32;
    let mut kv_stride_u = kv_stride as u32;
    let mut scale_v = scale;
    let grid = (n_heads as u32, 1u32, 1u32);
    let block = ((32 * NW) as u32, 1u32, 1u32);

    unsafe {
        let mut params: [*mut std::ffi::c_void; 10] = [
            &mut q_ptr as *mut _ as *mut std::ffi::c_void,
            &mut k_ptr as *mut _ as *mut std::ffi::c_void,
            &mut v_ptr as *mut _ as *mut std::ffi::c_void,
            &mut o_ptr as *mut _ as *mut std::ffi::c_void,
            &mut kv_len_u as *mut _ as *mut std::ffi::c_void,
            &mut head_dim_u as *mut _ as *mut std::ffi::c_void,
            &mut n_heads_u as *mut _ as *mut std::ffi::c_void,
            &mut n_kv_heads_u as *mut _ as *mut std::ffi::c_void,
            &mut kv_stride_u as *mut _ as *mut std::ffi::c_void,
            &mut scale_v as *mut _ as *mut std::ffi::c_void,
        ];
        cuda_core::launch_kernel_on_stream(&func, grid, block, 0, &stream, &mut params).ok()?;
    }
    let got = out_dev.to_host_vec(&stream).ok()?;
    let want =
        cpu_incremental_attention(&q, &k_il, &v_il, kv_len, head_dim, n_heads, n_kv_heads, scale);
    let cos = cosine_similarity(&got, &want);
    let mut maxabs_ref = 0.0f32;
    for &x in &want {
        if x.abs() > maxabs_ref {
            maxabs_ref = x.abs();
        }
    }
    let mut maxdiff = 0.0f32;
    for i in 0..q_dim {
        let dd = (got[i] - want[i]).abs();
        if dd > maxdiff {
            maxdiff = dd;
        }
    }
    let tol = 1e-4f32 * maxabs_ref.max(1e-6);
    Some((cos, maxdiff, tol))
}

/// PMAT-883 3-way parity gate over the full 9-config decode grid.
/// Returns true iff oxide-PTX, hand-PTX, and (loaded-module) raw-ptr kernel all
/// pass vs the CPU reference at every config.
fn run_3way_gate(ctx: &std::sync::Arc<CudaContext>, oxide_ptx_path: &str) -> bool {
    let head_dim = 128usize;
    let n_kv_heads = 8usize;
    let seqs = [128usize, 1024, 4096];
    let heads = [8usize, 16, 32];
    let handptx_msl = 4096usize;
    let nw32 = "baseline-ptx/multiwarp_msl4096_nw32.sm121.ptx";
    let nw8 = "baseline-ptx/multiwarp_msl4096_nw8.sm121.ptx";

    println!("\n== PMAT-883 3-WAY PARITY GATE (oxide-PTX == hand-PTX == CPU) ==");
    println!("   oxide PTX = {oxide_ptx_path}");
    println!("   gate: cos>=0.99 AND maxdiff < 1e-4*max|ref|, all 3 ways, all 9 configs\n");
    println!(
        "  {:>5} {:>4} {:>3} | {:<28} | {:<28} | {}",
        "seq", "head", "kv", "oxide-PTX (raw-ptr entry)", "hand-PTX multi_warp", "verdict"
    );

    let mut all_pass = true;
    for &nh in &heads {
        for &sl in &seqs {
            let nkv = if nh < n_kv_heads { nh } else { n_kv_heads };
            let seed = 4242 + sl + nh;

            // way 1: emitted oxide PTX, raw-ptr entry, interleaved layout
            let ox = oxide_ptx_parity(ctx, oxide_ptx_path, sl, head_dim, nh, nkv, seed);
            let ox_pass = ox.map(|(c, d, t)| c >= 0.99 && d < t).unwrap_or(false);

            // way 2: hand-PTX multi_warp_attention. The committed PTX bakes in
            // n_heads=32/n_kv_heads=8, so it is parity-valid only at heads=32;
            // for heads!=32 the hand-PTX way is reported as "n/a (baked 32h)".
            let hp = if nh == 32 {
                let f = if std::path::Path::new(nw32).exists() {
                    nw32
                } else {
                    nw8
                };
                handptx_ab(ctx, f, handptx_msl, sl, head_dim, nh, nkv, 32, seed)
                    .map(|(c, d, t, _)| (c, d, t))
            } else {
                None
            };
            let hp_pass = match (nh, hp) {
                (32, Some((c, d, t))) => c >= 0.99 && d < t,
                (32, None) => false, // hand-PTX SHOULD work at 32h; missing = fail
                (_, _) => true,      // n/a at non-32h is not a gate failure
            };

            let ox_str = match ox {
                Some((c, d, _)) => format!("cos={c:.6} md={d:.2e}"),
                None => "LOAD/LAUNCH FAIL".to_string(),
            };
            let hp_str = match (nh, hp) {
                (32, Some((c, d, _))) => format!("cos={c:.6} md={d:.2e}"),
                (32, None) => "LOAD/LAUNCH FAIL".to_string(),
                _ => "n/a (baked 32h)".to_string(),
            };

            let cfg_pass = ox_pass && hp_pass;
            if !cfg_pass {
                all_pass = false;
            }
            println!(
                "  {:>5} {:>4} {:>3} | {:<28} | {:<28} | {}",
                sl,
                nh,
                nkv,
                ox_str,
                hp_str,
                if cfg_pass { "PASS" } else { "FAIL" }
            );
        }
    }
    println!(
        "\nPMAT-883 3-WAY PARITY GATE: {}",
        if all_pass { "PASS" } else { "FAIL" }
    );
    all_pass
}

/// PMAT-884 LIVE-CACHE 3-way parity gate.
///
/// Promotion criterion 1: re-pass the 3-way gate against the LIVE separate-head
/// KV-cache layout `[num_kv_heads, max_len, head_dim]` (NOT the interleaved gate
/// inputs the 883 gate used). For each of the 9 decode configs it compares:
///   way 1 — oxide sephead PTX (`attn_warp_sephead_rawptr`, 10-param ABI),
///           K/V packed into the live slab layout, kernel indexes it directly,
///   way 2 — hand-PTX `multi_warp_attention` (already uses separate-head layout;
///           parity-valid only at heads=32 because the committed PTX bakes 32h),
///   way 3 — CPU `causal_attention_cached` reference,
/// and asserts cos >= 0.99 AND maxdiff < 1e-4*max|ref| for every way/config.
fn run_sephead_live_gate(ctx: &std::sync::Arc<CudaContext>, sephead_ptx_path: &str) -> bool {
    let head_dim = 128usize;
    let n_kv_heads = 8usize;
    let seqs = [128usize, 1024, 4096];
    let heads = [8usize, 16, 32];
    let max_len = 4096usize; // live cache slab stride = max_len*head_dim
    let handptx_msl = 4096usize;
    let nw32 = "baseline-ptx/multiwarp_msl4096_nw32.sm121.ptx";
    let nw8 = "baseline-ptx/multiwarp_msl4096_nw8.sm121.ptx";

    println!("\n== PMAT-884 LIVE-CACHE 3-WAY PARITY GATE (oxide sephead == hand-PTX == CPU) ==");
    println!("   sephead PTX = {sephead_ptx_path}");
    println!("   K/V layout  = SEPARATE-HEAD [num_kv_heads, max_len={max_len}, head_dim] (LIVE cache)");
    println!("   gate: cos>=0.99 AND maxdiff < 1e-4*max|ref|, all 3 ways, all 9 configs\n");
    println!(
        "  {:>5} {:>4} {:>3} | {:<30} | {:<28} | {}",
        "seq", "head", "kv", "oxide sephead (live layout)", "hand-PTX multi_warp", "verdict"
    );

    let mut all_pass = true;
    for &nh in &heads {
        for &sl in &seqs {
            let nkv = if nh < n_kv_heads { nh } else { n_kv_heads };
            let seed = 8484 + sl + nh;

            // way 1: oxide sephead PTX against the LIVE separate-head cache slab.
            let ox = sephead_ptx_parity(ctx, sephead_ptx_path, max_len, sl, head_dim, nh, nkv, seed);
            let ox_pass = ox.map(|(c, d, t)| c >= 0.99 && d < t).unwrap_or(false);

            // way 2: hand-PTX (already separate-head); parity-valid only at heads=32.
            let hp = if nh == 32 {
                let f = if std::path::Path::new(nw32).exists() {
                    nw32
                } else {
                    nw8
                };
                handptx_ab(ctx, f, handptx_msl, sl, head_dim, nh, nkv, 32, seed)
                    .map(|(c, d, t, _)| (c, d, t))
            } else {
                None
            };
            let hp_pass = match (nh, hp) {
                (32, Some((c, d, t))) => c >= 0.99 && d < t,
                (32, None) => false,
                (_, _) => true,
            };

            let ox_str = match ox {
                Some((c, d, _)) => format!("cos={c:.6} md={d:.2e}"),
                None => "LOAD/LAUNCH FAIL".to_string(),
            };
            let hp_str = match (nh, hp) {
                (32, Some((c, d, _))) => format!("cos={c:.6} md={d:.2e}"),
                (32, None) => "LOAD/LAUNCH FAIL".to_string(),
                _ => "n/a (baked 32h)".to_string(),
            };

            let cfg_pass = ox_pass && hp_pass;
            if !cfg_pass {
                all_pass = false;
            }
            println!(
                "  {:>5} {:>4} {:>3} | {:<30} | {:<28} | {}",
                sl,
                nh,
                nkv,
                ox_str,
                hp_str,
                if cfg_pass { "PASS" } else { "FAIL" }
            );
        }
    }
    println!(
        "\nPMAT-884 LIVE-CACHE 3-WAY PARITY GATE: {}",
        if all_pass { "PASS" } else { "FAIL" }
    );
    all_pass
}

fn main() {
    // PMAT-883 emit mode: dump the standalone embeddable PTX for attn_warp_rawptr.
    //   cargo oxide run -- emit-ptx [out_path]
    // The oxide toolchain compiles the #[kernel] fns to a PTX module; we resolve
    // the cuda-oxide-emitted .ptx for THIS crate and copy it to out_path so it can
    // be committed as the source-of-record (include_str! -> CudaModule::from_ptx).
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 && args[1] == "emit-ptx" {
        let out = args
            .get(2)
            .cloned()
            .unwrap_or_else(|| "generated/attn_warp.sm121.ptx".to_string());
        emit_ptx_artifact(&out);
        return;
    }

    let ctx = CudaContext::new(0).expect("ctx");
    let module = kernels::load(&ctx).expect("load");

    // PMAT-883: run the 3-way parity gate when the emitted PTX is present.
    let gate_ptx = [
        "generated/attn_warp.sm121.ptx",
        "/tmp/incattn_spike/attn_warp.sm121.ptx",
    ]
    .into_iter()
    .find(|p| std::path::Path::new(p).exists());
    if let Some(p) = gate_ptx {
        let ok = run_3way_gate(&ctx, p);
        if !ok {
            eprintln!("PMAT-883 3-WAY PARITY GATE FAILED");
            std::process::exit(2);
        }
    } else {
        println!(
            "\n[PMAT-883] emitted oxide PTX not found (run `cargo oxide run -- emit-ptx generated/attn_warp.sm121.ptx` first) — skipping 3-way gate, running 882 parity/perf only."
        );
    }

    // PMAT-884: run the LIVE-CACHE 3-way gate when the sephead PTX is present.
    let sephead_ptx = [
        "generated/attn_warp_sephead.sm121.ptx",
        "/tmp/incattn_spike/attn_warp_sephead.sm121.ptx",
    ]
    .into_iter()
    .find(|p| std::path::Path::new(p).exists());
    if let Some(p) = sephead_ptx {
        let ok = run_sephead_live_gate(&ctx, p);
        if !ok {
            eprintln!("PMAT-884 LIVE-CACHE 3-WAY PARITY GATE FAILED");
            std::process::exit(3);
        }
    } else {
        println!(
            "\n[PMAT-884] sephead oxide PTX not found (run `./emit_ptx_sephead.sh generated/attn_warp_sephead.sm121.ptx` first) — skipping live-cache gate."
        );
    }

    println!("== PMAT-882 cuda-oxide incremental (KV-cache) attention — parity + perf (GB10) ==");
    println!("   single-block T={T}  split-K TC={TC} CHUNK={CHUNK}  MAX_HD={MAX_HD}");

    let head_dim = 128usize;
    let n_kv_heads = 8usize;
    let seqs = [128usize, 1024, 4096];
    let heads = [8usize, 16, 32];

    // PARITY for ALL three kernels across the full grid.
    for &(var, name) in &[
        (0u8, "single-block (A)"),
        (1u8, "split-K (B)"),
        (2u8, "warp-coalesced (C)"),
    ] {
        let mut all_pass = true;
        println!("\n-- PARITY {name} (head_dim={head_dim}, n_kv_heads={n_kv_heads}) --");
        for &nh in &heads {
            for &sl in &seqs {
                let nkv = if nh < n_kv_heads { nh } else { n_kv_heads };
                let (cos, maxdiff, tol, _m) = parity_and_perf(
                    &ctx,
                    &module,
                    sl,
                    head_dim,
                    nh,
                    nkv,
                    1234 + sl + nh,
                    false,
                    var,
                );
                let pass = cos >= 0.99 && maxdiff < tol;
                if !pass {
                    all_pass = false;
                }
                println!(
                    "  seq={:>5} heads={:>3} kv={:>2} : cos={:.6} maxdiff={:.3e} tol={:.3e} -> {}",
                    sl,
                    nh,
                    nkv,
                    cos,
                    maxdiff,
                    tol,
                    if pass { "PASS" } else { "FAIL" }
                );
            }
        }
        println!("PARITY {name}: {}", if all_pass { "PASS" } else { "FAIL" });
        if !all_pass {
            eprintln!("PMAT-882 PARITY FAILED ({name})");
            std::process::exit(1);
        }
    }

    // PERF A/B/C at representative decode shapes.
    println!("\n-- PERF (us/launch, GPU-event, median of 5x50) — A vs B(split) vs C(warp) --");
    let perf_shapes: &[(usize, usize, usize, &str)] = &[
        (128, 16, 8, "short ctx kv=128 heads=16"),
        (128, 32, 8, "short ctx kv=128 heads=32"),
        (1024, 32, 8, "Qwen-7B-like kv=1024 heads=32"),
        (4096, 32, 8, "long ctx kv=4096 heads=32"),
    ];
    for &(sl, nh, nkv, label) in perf_shapes {
        let (_c1, _d1, _t1, med_a) =
            parity_and_perf(&ctx, &module, sl, head_dim, nh, nkv, 77, true, 0);
        let (_c2, _d2, _t2, med_b) =
            parity_and_perf(&ctx, &module, sl, head_dim, nh, nkv, 77, true, 1);
        let (_c3, _d3, _t3, med_c) =
            parity_and_perf(&ctx, &module, sl, head_dim, nh, nkv, 77, true, 2);
        let best = med_a.min(med_b).min(med_c);
        println!(
            "  kv={:>5} heads={:>3} : A={:>7.2}us  B={:>7.2}us  C={:>7.2}us  best={:>7.2}us  [{}]",
            sl, nh, med_a, med_b, med_c, best, label
        );
    }

    // TRUE hand-PTX A/B: oxide kernel C vs the emitted multi_warp_attention PTX
    // on the SAME data + GB10, same GPU-event timing. The PTX was emitted at
    // max_seq_len=4096, num_warps=8 (the documented default).
    let handptx_msl = 4096usize;
    // The emitted PTX bakes in n_heads=32 / n_kv_heads=8 (GQA mapping is compile-
    // time), so the A/B uses heads=32 shapes only for a fair, parity-valid compare.
    let ab_shapes: &[(usize, usize, usize, &str)] = &[
        (128, 32, 8, "short kv=128 heads=32"),
        (1024, 32, 8, "kv=1024 heads=32"),
        (4096, 32, 8, "long kv=4096 heads=32"),
    ];
    // Compare oxide C against the hand-PTX at both its default NW=8 and a matched
    // NW=32 (best-vs-best). Whichever hand-PTX file is present is used.
    // Hand-PTX baselines: look first in the run dir, then the committed
    // baseline-ptx/ (the source-of-record). See STATUS for the regen command.
    let find_ptx = |names: &[&str]| -> Option<String> {
        for n in names {
            if std::path::Path::new(n).exists() {
                return Some((*n).to_string());
            }
        }
        None
    };
    let nw8 = find_ptx(&[
        "/tmp/incattn_spike/multiwarp_msl4096_nw8.ptx",
        "baseline-ptx/multiwarp_msl4096_nw8.sm121.ptx",
    ]);
    let nw32 = find_ptx(&[
        "/tmp/incattn_spike/multiwarp_msl4096_nw32.ptx",
        "baseline-ptx/multiwarp_msl4096_nw32.sm121.ptx",
    ]);
    let baselines: Vec<(usize, String)> = [(8usize, nw8), (32usize, nw32)]
        .into_iter()
        .filter_map(|(nw, p)| p.map(|p| (nw, p)))
        .collect();
    for (hpx_nw, hpx_file) in &baselines {
        let hpx_nw = *hpx_nw;
        let hpx_file = hpx_file.as_str();
        println!(
            "\n-- TRUE hand-PTX A/B (oxide C NW={NW} vs multi_warp_attention NW={hpx_nw}, GB10) --"
        );
        for &(sl, nh, nkv, label) in ab_shapes {
            let (_c, _d, _t, oxide_us) =
                parity_and_perf(&ctx, &module, sl, head_dim, nh, nkv, 77, true, 2);
            match handptx_ab(
                &ctx,
                hpx_file,
                handptx_msl,
                sl,
                head_dim,
                nh,
                nkv,
                hpx_nw,
                77,
            ) {
                Some((hcos, hmd, htol, hpx_us)) => {
                    let hpx_ok = hcos >= 0.99 && hmd < htol;
                    let ratio = oxide_us / hpx_us;
                    println!(
                        "  {:<22}: oxide={:>7.2}us  handPTX={:>7.2}us  ratio={:>5.2}x  (handPTX parity {}: cos={:.4} maxdiff={:.2e})  {}",
                        label,
                        oxide_us,
                        hpx_us,
                        ratio,
                        if hpx_ok { "PASS" } else { "FAIL" },
                        hcos,
                        hmd,
                        if ratio <= 1.2 { "GO" } else { "no-go" }
                    );
                }
                None => println!("  {label:<22}: handPTX launch FAILED (see stderr)"),
            }
        }
    }

    println!("\nPMAT-882 DONE");
}
