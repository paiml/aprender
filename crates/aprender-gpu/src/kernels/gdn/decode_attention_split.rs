//! aprender#4273: decode attention split over the sequence (flash decoding).
//!
//! [`DecodeAttention256Kernel`](super::DecodeAttention256Kernel) launches one
//! block per query head — 16 blocks on Qwen3.5-4B, on a 128-SM 4090 — and each
//! block walks every cached position serially. Its cost is therefore linear in
//! the context on 16 SMs: measured on 0.69.3 RC1, apr decode fell from 70 tok/s
//! at 850 tokens of context to 7.8 tok/s at 32k while llama.cpp held ~150.
//!
//! This pair splits the positions of every head into slices of `split_len`:
//!
//! ```text
//! split kernel,  grid (num_heads, n_splits):  block (h, s) runs the unsplit
//!     kernel's pass loop over positions [s*split_len, min(seq_len, (s+1)*split_len))
//!     and stores its UNNORMALISED state: acc_s[h][0..head_dim], m_s[h], l_s[h]
//! reduce kernel, grid (num_heads):
//!     M   = max_s m_s
//!     out = sum_s acc_s * exp(m_s - M)  /  sum_s l_s * exp(m_s - M)
//! ```
//!
//! which is the same softmax the running max/sum of the unsplit kernel computes.
//! `n_splits = ceil(seq_len / split_len)` is derived from `seq_len` by BOTH
//! kernels and by [`DecodeAttentionSplitKernel::n_splits`] on the host, never
//! passed separately, so the grid and the partial layout cannot disagree.
//!
//! With one split (`seq_len <= split_len`) the reduce multiplies by
//! `exp(0) = 1` and adds to zero, so the output is bit-for-bit the unsplit
//! kernel's — asserted by a device test, not assumed.
//!
//! ## Partial layout
//!
//! `part_acc` is `[num_heads][n_splits][head_dim]` and `part_ml` is
//! `[num_heads][n_splits][2]` (`max`, `sum`), both f32; see
//! [`DecodeAttentionSplitKernel::partial_floats`].

use super::decode_attention::{emit_passes, PassRegs, PassShape, BLOCK, WARPS};
use crate::kernels::gdn::emit_exp_f32;
use crate::kernels::Kernel;
use crate::ptx::builder::{KernelBuilder, PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType, VirtualReg};

/// Positions per split. One per thread in the score phase; 16 splits per head
/// at 4k and 128 at 32k, so a 32k decode is 2048 blocks on Qwen3.5-4B.
pub const DEFAULT_SPLIT_LEN: u32 = 256;

/// `n_splits = ceil(seq_len / split_len)`, on the device.
fn emit_n_splits(ctx: &mut KernelBuilder<'_>, seq_len: VirtualReg, split_len: u32) -> VirtualReg {
    let padded = ctx.add_u32(seq_len, split_len - 1);
    ctx.div_u32(padded, split_len)
}

/// First kernel of the pair: one block per (query head, slice of positions).
#[derive(Debug, Clone, Copy)]
pub struct DecodeAttentionSplitKernel {
    /// Query heads.
    pub num_heads: u32,
    /// Key/value heads; `num_heads` must be a multiple of it.
    pub num_kv_heads: u32,
    /// Width of one head; must be `<= 256`.
    pub head_dim: u32,
    /// Positions per split; one pass, so also the shared scores buffer.
    pub split_len: u32,
}

impl DecodeAttentionSplitKernel {
    /// Create the kernel with [`DEFAULT_SPLIT_LEN`].
    ///
    /// # Panics
    /// If `head_dim` exceeds the 256-thread block or `num_kv_heads` does not
    /// divide `num_heads`.
    #[must_use]
    pub fn new(num_heads: u32, num_kv_heads: u32, head_dim: u32) -> Self {
        assert!(
            head_dim <= BLOCK,
            "head_dim {head_dim} exceeds the {BLOCK}-thread block; one thread owns one output element"
        );
        assert!(
            head_dim % 4 == 0,
            "head_dim {head_dim} must be a multiple of 4: the score dot reads 16-byte vectors"
        );
        assert!(
            num_kv_heads > 0 && num_heads % num_kv_heads == 0,
            "num_heads {num_heads} must be a multiple of num_kv_heads {num_kv_heads}"
        );
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            split_len: DEFAULT_SPLIT_LEN,
        }
    }

    /// Override the positions per split.
    ///
    /// # Panics
    /// If `split_len` is 0.
    #[must_use]
    pub const fn with_split_len(mut self, split_len: u32) -> Self {
        assert!(split_len > 0, "split_len must be positive");
        self.split_len = split_len;
        self
    }

    /// Splits for `seq_len` positions — the launch grid's `y`.
    #[must_use]
    pub const fn n_splits(&self, seq_len: u32) -> u32 {
        seq_len.div_ceil(self.split_len)
    }

    /// Floats of `(part_acc, part_ml)` a launch at `seq_len` writes.
    #[must_use]
    pub const fn partial_floats(&self, seq_len: u32) -> (usize, usize) {
        let rows = (self.num_heads * self.n_splits(seq_len)) as usize;
        (rows * self.head_dim as usize, rows * 2)
    }

    /// Static shared memory: the scores of one split plus two reduction rows.
    #[must_use]
    pub const fn shared_bytes(&self) -> usize {
        (self.split_len * 4 + WARPS * 4 * 2) as usize
    }

    /// Launch grid for `seq_len`.
    #[must_use]
    pub const fn grid(&self, seq_len: u32) -> (u32, u32, u32) {
        (self.num_heads, self.n_splits(seq_len), 1)
    }

    /// Launch block — 256 threads.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (BLOCK, 1, 1)
    }

    /// The reduce kernel matching this split.
    #[must_use]
    pub const fn reduce(&self) -> DecodeAttentionReduceKernel {
        DecodeAttentionReduceKernel {
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            split_len: self.split_len,
        }
    }
}

impl Kernel for DecodeAttentionSplitKernel {
    fn name(&self) -> &str {
        "gdn_decode_attention_split"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let group_size = self.num_heads / self.num_kv_heads;
        let split_len = self.split_len;
        let shape = PassShape {
            head_dim,
            cap: split_len,
            row_stride_bytes: self.num_kv_heads * head_dim * 4,
        };

        PtxKernel::new(self.name())
            .param(PtxType::U64, "q_ptr") // [num_heads * head_dim]
            .param(PtxType::U64, "k_cache_ptr") // [max_len][num_kv_heads * head_dim]
            .param(PtxType::U64, "v_cache_ptr") // [max_len][num_kv_heads * head_dim]
            .param(PtxType::U64, "part_acc_ptr") // [num_heads][n_splits][head_dim]
            .param(PtxType::U64, "part_ml_ptr") // [num_heads][n_splits][2]
            .param(PtxType::U32, "seq_len") // valid positions, = position + 1
            .shared_memory(self.shared_bytes())
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let h = ctx.special_reg(PtxReg::CtaIdX);
                let split = ctx.special_reg(PtxReg::CtaIdY);
                let lane = ctx.and_u32_imm(tid, 31);
                let warp = ctx.shr_u32_imm(tid, 5);

                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptr = ctx.load_param_u64("k_cache_ptr");
                let v_ptr = ctx.load_param_u64("v_cache_ptr");
                let acc_ptr = ctx.load_param_u64("part_acc_ptr");
                let ml_ptr = ctx.load_param_u64("part_ml_ptr");
                let seq_len = ctx.load_param_u32("seq_len");

                // This block's slice; a block past the end has nothing to store.
                let begin = ctx.mul_u32(split, split_len);
                let in_range = ctx.setp_lt_u32(begin, seq_len);
                ctx.branch_if_not(in_range, "gdn_attn_split_exit");
                let span_end = ctx.add_u32(begin, split_len);
                let end = ctx.min_u32(span_end, seq_len);

                let kv_h = ctx.div_u32(h, group_size);
                let kv_off = ctx.mul_wide_u32(kv_h, head_dim * 4);
                let head_off = ctx.mul_wide_u32(h, head_dim * 4);
                let q_base = ctx.add_u64(q_ptr, head_off);
                let k_head = ctx.add_u64(k_ptr, kv_off);
                let v_head = ctx.add_u64(v_ptr, kv_off);

                let st = emit_passes(
                    ctx,
                    shape,
                    PassRegs { tid, lane, warp, q_base, k_head, v_head, begin, end },
                );

                // row = h * n_splits + split
                let n_splits = emit_n_splits(ctx, seq_len, split_len);
                let h_rows = ctx.mul_u32_reg(h, n_splits);
                let row = ctx.add_u32_reg(h_rows, split);

                // Thread 0 stores (max, sum); every thread in the head its acc.
                let zero = ctx.mov_u32_imm(0);
                let is_t0 = ctx.setp_eq_u32(tid, zero);
                ctx.branch_if_not(is_t0, "gdn_attn_split_skip_ml");
                let ml_off = ctx.mul_wide_u32(row, 8);
                let ml_addr = ctx.add_u64(ml_ptr, ml_off);
                ctx.st_global_f32(ml_addr, st.running_max);
                let four_b = ctx.mov_u32_imm(4);
                let four_b64 = ctx.cvt_u64_u32(four_b);
                let sum_addr = ctx.add_u64(ml_addr, four_b64);
                ctx.st_global_f32(sum_addr, st.running_sum);
                ctx.label("gdn_attn_split_skip_ml");

                let in_head = ctx.setp_lt_u32(tid, st.head_dim_r);
                ctx.branch_if_not(in_head, "gdn_attn_split_exit");
                let row_off = ctx.mul_wide_u32(row, head_dim * 4);
                let row_base = ctx.add_u64(acc_ptr, row_off);
                let acc_addr = ctx.add_u64(row_base, st.out_elem_off);
                ctx.st_global_f32(acc_addr, st.acc);

                ctx.label("gdn_attn_split_exit");
                ctx.ret();
            })
    }
}

/// Second kernel of the pair: combine the splits of each head.
#[derive(Debug, Clone, Copy)]
pub struct DecodeAttentionReduceKernel {
    /// Query heads.
    pub num_heads: u32,
    /// Width of one head; must be `<= 256`.
    pub head_dim: u32,
    /// Positions per split — must equal the split kernel's.
    pub split_len: u32,
}

impl DecodeAttentionReduceKernel {
    /// Launch grid — one block per query head.
    #[must_use]
    pub const fn grid(&self) -> (u32, u32, u32) {
        (self.num_heads, 1, 1)
    }

    /// Launch block — 256 threads, one per output element.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (BLOCK, 1, 1)
    }
}

impl Kernel for DecodeAttentionReduceKernel {
    fn name(&self) -> &str {
        "gdn_decode_attention_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let split_len = self.split_len;

        PtxKernel::new(self.name())
            .param(PtxType::U64, "part_acc_ptr") // [num_heads][n_splits][head_dim]
            .param(PtxType::U64, "part_ml_ptr") // [num_heads][n_splits][2]
            .param(PtxType::U64, "out_ptr") // [num_heads * head_dim]
            .param(PtxType::U32, "seq_len")
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let h = ctx.special_reg(PtxReg::CtaIdX);

                let acc_ptr = ctx.load_param_u64("part_acc_ptr");
                let ml_ptr = ctx.load_param_u64("part_ml_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let seq_len = ctx.load_param_u32("seq_len");

                // An empty cache has no softmax: leave the output alone, as the
                // unsplit kernel does.
                let zero = ctx.mov_u32_imm(0);
                let has_positions = ctx.setp_lt_u32(zero, seq_len);
                ctx.branch_if_not(has_positions, "gdn_attn_reduce_exit");
                let head_dim_r = ctx.mov_u32_imm(head_dim);
                let in_head = ctx.setp_lt_u32(tid, head_dim_r);
                ctx.branch_if_not(in_head, "gdn_attn_reduce_exit");

                let n_splits = emit_n_splits(ctx, seq_len, split_len);
                let first_row = ctx.mul_u32_reg(h, n_splits);
                let ml_off = ctx.mul_wide_u32(first_row, 8);
                let ml_head = ctx.add_u64(ml_ptr, ml_off);
                let acc_off = ctx.mul_wide_u32(first_row, head_dim * 4);
                let acc_head = ctx.add_u64(acc_ptr, acc_off);
                let four = ctx.mov_u32_imm(4);
                let elem_off = ctx.mul_wide_u32_reg(tid, four);
                let four64 = ctx.cvt_u64_u32(four);
                let acc_elem = ctx.add_u64(acc_head, elem_off);

                // M = max_s m_s
                let global_max = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let s = ctx.mov_u32_imm(0);
                ctx.label("gdn_attn_reduce_max_loop");
                let max_go = ctx.setp_lt_u32(s, n_splits);
                ctx.branch_if_not(max_go, "gdn_attn_reduce_max_end");
                let m_off = ctx.mul_wide_u32(s, 8);
                let m_addr = ctx.add_u64(ml_head, m_off);
                let m_s = ctx.ld_global_f32(m_addr);
                ctx.max_f32_inplace(global_max, m_s);
                ctx.add_u32_inplace(s, 1);
                ctx.branch("gdn_attn_reduce_max_loop");
                ctx.label("gdn_attn_reduce_max_end");

                // splits ascending: sum += l_s * f_s; acc += acc_s * f_s
                let sum = ctx.mov_f32_imm(0.0);
                let acc = ctx.mov_f32_imm(0.0);
                let t = ctx.mov_u32_imm(0);
                ctx.label("gdn_attn_reduce_loop");
                let go = ctx.setp_lt_u32(t, n_splits);
                ctx.branch_if_not(go, "gdn_attn_reduce_end");
                let t_ml_off = ctx.mul_wide_u32(t, 8);
                let t_ml = ctx.add_u64(ml_head, t_ml_off);
                let m_t = ctx.ld_global_f32(t_ml);
                let l_addr = ctx.add_u64(t_ml, four64);
                let l_t = ctx.ld_global_f32(l_addr);
                let delta = ctx.sub_f32(m_t, global_max);
                let factor = emit_exp_f32(ctx, delta);
                let l_scaled = ctx.mul_f32(l_t, factor);
                ctx.add_f32_inplace(sum, l_scaled);
                let t_acc_off = ctx.mul_wide_u32(t, head_dim * 4);
                let t_acc = ctx.add_u64(acc_elem, t_acc_off);
                let a_t = ctx.ld_global_f32(t_acc);
                let a_scaled = ctx.mul_f32(a_t, factor);
                ctx.add_f32_inplace(acc, a_scaled);
                ctx.add_u32_inplace(t, 1);
                ctx.branch("gdn_attn_reduce_loop");
                ctx.label("gdn_attn_reduce_end");

                let head_off = ctx.mul_wide_u32(h, head_dim * 4);
                let out_head = ctx.add_u64(out_ptr, head_off);
                let out_addr = ctx.add_u64(out_head, elem_off);
                let result = ctx.div_f32(acc, sum);
                ctx.st_global_f32(out_addr, result);

                ctx.label("gdn_attn_reduce_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod ptx_tests {
    use super::*;

    #[test]
    fn gdn_decode_attention_split_ptx_shape() {
        let split = DecodeAttentionSplitKernel::new(16, 4, 256);
        let ptx = split.emit_ptx();
        assert!(ptx.contains(".entry gdn_decode_attention_split"), "{ptx}");
        assert!(ptx.contains("%ctaid.y"), "the split index is the grid's y:\n{ptx}");
        assert!(ptx.contains("gdn_attn_dot_loop"), "{ptx}");
        let reduce = split.reduce().emit_ptx();
        assert!(reduce.contains(".entry gdn_decode_attention_reduce"), "{reduce}");
        assert_eq!(split.shared_bytes(), 256 * 4 + 64);
    }

    #[test]
    fn gdn_decode_attention_split_grid_covers_every_position() {
        let k = DecodeAttentionSplitKernel::new(16, 4, 256);
        // ceil, never floor: a floor would silently drop the tail positions.
        assert_eq!(k.grid(1), (16, 1, 1));
        assert_eq!(k.grid(256), (16, 1, 1));
        assert_eq!(k.grid(257), (16, 2, 1));
        assert_eq!(k.grid(32_768), (16, 128, 1));
        assert_eq!(k.partial_floats(257), (16 * 2 * 256, 16 * 2 * 2));
    }
}

/// Device checks: parity with the unsplit kernel (itself CPU-parity tested), and
/// the aprender#4273 bandwidth gate.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod gdn_decode_attention_split_device_tests {
    use super::DecodeAttentionSplitKernel;
    use crate::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use crate::kernels::gdn::test_support::{assert_close, Lcg};
    use crate::kernels::gdn::DecodeAttention256Kernel;
    use crate::kernels::Kernel;
    use std::time::Instant;

    /// Qwen3.5-4B attention: 16 query heads, 4 KV heads, 256-wide.
    const NUM_HEADS: u32 = 16;
    const NUM_KV_HEADS: u32 = 4;
    const HEAD_DIM: u32 = 256;

    /// Device buffers for one attention call; the cache holds `max_len` rows.
    struct Fixture {
        q: GpuBuffer<f32>,
        k: GpuBuffer<f32>,
        v: GpuBuffer<f32>,
        out: GpuBuffer<f32>,
        part_acc: GpuBuffer<f32>,
        part_ml: GpuBuffer<f32>,
    }

    impl Fixture {
        fn new(ctx: &CudaContext, max_len: usize, max_splits: usize, seed: u32) -> Self {
            let row = (NUM_KV_HEADS * HEAD_DIM) as usize;
            let mut rng = Lcg::new(seed);
            let q = rng.vec((NUM_HEADS * HEAD_DIM) as usize, 0.1);
            let k = rng.vec(max_len * row, 0.5);
            let v = rng.vec(max_len * row, 0.5);
            let rows = NUM_HEADS as usize * max_splits;
            Self {
                q: GpuBuffer::from_host(ctx, &q).expect("q"),
                k: GpuBuffer::from_host(ctx, &k).expect("k"),
                v: GpuBuffer::from_host(ctx, &v).expect("v"),
                out: GpuBuffer::<f32>::new(ctx, (NUM_HEADS * HEAD_DIM) as usize).expect("out"),
                part_acc: GpuBuffer::<f32>::new(ctx, rows * HEAD_DIM as usize).expect("acc"),
                part_ml: GpuBuffer::<f32>::new(ctx, rows * 2).expect("ml"),
            }
        }

        fn download(&self) -> Vec<f32> {
            let mut got = vec![0.0f32; (NUM_HEADS * HEAD_DIM) as usize];
            self.out.copy_to_host(&mut got).expect("download");
            got
        }
    }

    /// A compiled kernel, launched without recompiling (the timing loop needs it).
    struct Loaded {
        module: CudaModule,
        name: String,
    }

    impl Loaded {
        fn new<K: Kernel>(ctx: &CudaContext, k: &K) -> Self {
            Self {
                module: CudaModule::from_ptx(ctx, &k.emit_ptx()).expect("module"),
                name: k.name().to_string(),
            }
        }

        fn launch(&mut self, stream: &CudaStream, grid: (u32, u32, u32), args: &mut [u64]) {
            let config = LaunchConfig {
                grid,
                block: (256, 1, 1),
                shared_mem: 0,
            };
            let mut raw: Vec<*mut std::ffi::c_void> = args
                .iter_mut()
                .map(|a| std::ptr::from_mut(a).cast())
                .collect();
            // SAFETY: the Fixture sizes every buffer to the extent the kernels
            // index at the seq_len passed, and grid/block are the kernels' own.
            unsafe {
                stream
                    .launch_kernel(&mut self.module, &self.name, &config, &mut raw)
                    .expect("launch");
            }
        }
    }

    /// The two ways to compute one decode attention.
    struct Paths {
        unsplit: (DecodeAttention256Kernel, Loaded),
        split: (DecodeAttentionSplitKernel, Loaded, Loaded),
    }

    impl Paths {
        fn new(ctx: &CudaContext, split_len: u32) -> Self {
            let u = DecodeAttention256Kernel::new(NUM_HEADS, NUM_KV_HEADS, HEAD_DIM);
            let s = DecodeAttentionSplitKernel::new(NUM_HEADS, NUM_KV_HEADS, HEAD_DIM)
                .with_split_len(split_len);
            Self {
                unsplit: (u, Loaded::new(ctx, &u)),
                split: (s, Loaded::new(ctx, &s), Loaded::new(ctx, &s.reduce())),
            }
        }

        fn run_unsplit(&mut self, stream: &CudaStream, f: &Fixture, seq_len: u32) {
            let (k, l) = &mut self.unsplit;
            let mut args = [f.q.as_ptr(), f.k.as_ptr(), f.v.as_ptr(), f.out.as_ptr(), u64::from(seq_len)];
            l.launch(stream, k.grid(), &mut args);
        }

        fn run_split(&mut self, stream: &CudaStream, f: &Fixture, seq_len: u32) {
            let (k, split, reduce) = &mut self.split;
            let mut a = [
                f.q.as_ptr(),
                f.k.as_ptr(),
                f.v.as_ptr(),
                f.part_acc.as_ptr(),
                f.part_ml.as_ptr(),
                u64::from(seq_len),
            ];
            split.launch(stream, k.grid(seq_len), &mut a);
            let mut b = [f.part_acc.as_ptr(), f.part_ml.as_ptr(), f.out.as_ptr(), u64::from(seq_len)];
            reduce.launch(stream, k.reduce().grid(), &mut b);
        }
    }

    #[test]
    fn gdn_decode_attention_split_matches_unsplit() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn_decode_attention_split: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        // split_len 8 so that 37 positions are 5 splits with a ragged tail.
        let f = Fixture::new(&ctx, 64, 8, 0x4273_0001);
        let mut paths = Paths::new(&ctx, 8);
        for seq_len in [1u32, 7, 8, 9, 37, 64] {
            paths.run_unsplit(&stream, &f, seq_len);
            stream.synchronize().expect("sync");
            let want = f.download();
            paths.run_split(&stream, &f, seq_len);
            stream.synchronize().expect("sync");
            let got = f.download();
            if seq_len <= 8 {
                // One split: the reduce scales by exp(0) = 1 and adds to zero.
                let same = got.iter().zip(&want).all(|(g, w)| g.to_bits() == w.to_bits());
                assert!(same, "seq_len {seq_len}: a single split must be bit-identical to the unsplit kernel");
            } else {
                assert_close(&got, &want, 1e-5, &format!("split vs unsplit, seq_len {seq_len}"));
            }
        }
    }

    /// Wall time per call over `n` calls, after warm-up, fully synchronised.
    fn time_ms(stream: &CudaStream, n: u32, mut f: impl FnMut()) -> f64 {
        for _ in 0..3 {
            f();
        }
        stream.synchronize().expect("sync");
        let t = Instant::now();
        for _ in 0..n {
            f();
        }
        stream.synchronize().expect("sync");
        t.elapsed().as_secs_f64() * 1e3 / f64::from(n)
    }

    /// aprender#4273 red test. At 32k the decode attention must be bound by reading
    /// the KV cache: at least HALF the bandwidth a device-to-device copy reaches on
    /// the same device, measured in the same process (no hard-coded peak).
    ///
    /// The unsplit kernel is timed too and must FAIL the same bound — the gate is
    /// shown capable of going red, on every run. The 32k/850 time ratio of both is
    /// printed for the receipt.
    #[test]
    #[ignore = "timing: run alone on the device via gpu-q"]
    fn gdn_decode_attention_32k_is_bound_by_the_kv_read() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn_decode_attention 32k timing: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        let long: u32 = 32_768;
        let short: u32 = 850;
        let split_len = super::DEFAULT_SPLIT_LEN;
        let f = Fixture::new(&ctx, long as usize, long.div_ceil(split_len) as usize, 0x4273_0002);
        let mut paths = Paths::new(&ctx, split_len);

        // Copy bandwidth: a D2D copy of the K cache reads and writes its bytes once.
        let kv_row_bytes = f64::from(NUM_KV_HEADS * HEAD_DIM * 4);
        let cache_bytes = kv_row_bytes * f64::from(long);
        let mut scratch = GpuBuffer::<f32>::new(&ctx, (long * NUM_KV_HEADS * HEAD_DIM) as usize).expect("scratch");
        let copy_ms = {
            for _ in 0..3 {
                scratch.copy_from_buffer(&f.k).expect("copy");
            }
            ctx.synchronize().expect("sync");
            let t = Instant::now();
            for _ in 0..10 {
                scratch.copy_from_buffer(&f.k).expect("copy");
            }
            ctx.synchronize().expect("sync");
            t.elapsed().as_secs_f64() * 1e3 / 10.0
        };
        let copy_gbs = 2.0 * cache_bytes / copy_ms / 1e6;

        // One decode attention reads K and V for every valid position.
        let gbs = |seq_len: u32, ms: f64| 2.0 * kv_row_bytes * f64::from(seq_len) / ms / 1e6;
        let mut report = Vec::new();
        let mut verdict = |name: &str, t_short: f64, t_long: f64| {
            let bw = gbs(long, t_long);
            let pass = bw >= 0.5 * copy_gbs;
            report.push(format!(
                "{name}: {short}={t_short:.4} ms  {long}={t_long:.4} ms  ratio={:.1}x  32k={bw:.0} GB/s ({:.0}% of copy) -> {}",
                t_long / t_short,
                100.0 * bw / copy_gbs,
                if pass { "PASS" } else { "FAIL" }
            ));
            pass
        };
        let us = time_ms(&stream, 5, || paths.run_unsplit(&stream, &f, short));
        let ul = time_ms(&stream, 5, || paths.run_unsplit(&stream, &f, long));
        let unsplit_pass = verdict("unsplit", us, ul);
        let ss = time_ms(&stream, 50, || paths.run_split(&stream, &f, short));
        let sl = time_ms(&stream, 50, || paths.run_split(&stream, &f, long));
        let split_pass = verdict("split", ss, sl);
        println!("copy: {copy_gbs:.0} GB/s over {:.0} MB", cache_bytes / 1e6);
        for line in &report {
            println!("{line}");
        }
        assert!(
            !unsplit_pass,
            "negative control: the unsplit kernel passed the bound, so it discriminates nothing\n{report:#?}"
        );
        assert!(split_pass, "split decode attention at 32k is not KV-read bound\n{report:#?}");
    }

    /// Sweep of `split_len` at 32k: prints bandwidth per split length. A probe,
    /// not a gate (aprender#4273).
    #[test]
    #[ignore = "timing probe; run on the GPU through gpu-q"]
    fn gdn_decode_attention_split_len_sweep() {
        let Ok(ctx) = CudaContext::new(0) else {
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        let long: u32 = 32_768;
        let f = Fixture::new(&ctx, long as usize, long.div_ceil(16) as usize, 0x4273_0003);
        let kv_row_bytes = f64::from(NUM_KV_HEADS * HEAD_DIM * 4);
        for split_len in [16, 32, 64, 128, 256, 512] {
            let mut paths = Paths::new(&ctx, split_len);
            let ms = time_ms(&stream, 50, || paths.run_split(&stream, &f, long));
            let bw = 2.0 * kv_row_bytes * f64::from(long) / ms / 1e6;
            println!("sweep split_len={split_len}: {ms:.4} ms  {bw:.0} GB/s");
        }
    }
}
