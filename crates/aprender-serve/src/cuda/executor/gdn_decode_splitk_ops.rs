//! PMAT-3725 / aprender#3725: executor wrapper for the split-K decode attention
//! pair in `aprender-gpu/src/kernels/gdn/decode_attention_splitk.rs`.
//!
//! Replaces [`CudaExecutor::gdn_decode_attention_into`] on the Qwen3.5 decode path:
//! the same attention block, spread over `(num_kv_heads, n_splits)` blocks instead
//! of `num_heads`, and able to read an f16 KV cache (the #3596 storage split: #3596
//! owns the storage and the row writes, this reads them).
//!
//! Same mechanics as the other GDN wrappers (`gdn_ops.rs`): `KernelType` arm ->
//! PTX compiled once per shape and cached by key -> launch -> the
//! `graph_recording` push. The split count is a runtime argument (from
//! [`SplitKPlan`]), so one module pair serves the whole decode.

use super::*;
use trueno_gpu::kernels::gdn::{
    splitk_partial_acc_len, splitk_partial_ml_len, DecodeAttentionSplitKKernel,
    DecodeAttentionSplitKReduceKernel, KvStorage, SplitKPlan,
};

impl CudaExecutor {
    /// Single-query decode attention over a resident KV cache, split-K.
    ///
    /// `k_cache_ptr` / `v_cache_ptr` point at `[max_len][num_kv_heads * head_dim]`
    /// caches stored as `kv` (f32, or f16 packed the same way); `seq_len` is the
    /// number of valid positions, i.e. `position + 1`. `partial_acc` and
    /// `partial_ml` are scratch sized by [`splitk_partial_acc_len`] /
    /// [`splitk_partial_ml_len`] for at least the plan's split count — see
    /// [`Self::gdn_decode_attention_splitk_scratch_splits`].
    ///
    /// # Errors
    /// `seq_len == 0`, scratch smaller than the plan needs, a null device pointer,
    /// or PTX compilation / launch failure.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_decode_attention_splitk_into(
        &mut self,
        q: &GpuBuffer<f32>,
        k_cache_ptr: u64,
        v_cache_ptr: u64,
        kv: KvStorage,
        partial_acc: &GpuBuffer<f32>,
        partial_ml: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
    ) -> Result<(), GpuError> {
        if seq_len == 0 {
            return Err(GpuError::InvalidParameter(
                "split-K decode attention over an empty KV cache has no softmax".into(),
            ));
        }
        let plan = SplitKPlan::for_seq_len(seq_len);
        let need_acc = splitk_partial_acc_len(num_heads, head_dim, plan.n_splits);
        let need_ml = splitk_partial_ml_len(num_heads, plan.n_splits);
        if partial_acc.len() < need_acc || partial_ml.len() < need_ml {
            return Err(GpuError::InvalidParameter(format!(
                "split-K scratch too small for seq_len {seq_len} ({} splits): \
                 partial_acc {} < {need_acc} or partial_ml {} < {need_ml}",
                plan.n_splits,
                partial_acc.len(),
                partial_ml.len(),
            )));
        }

        let a = DecodeAttentionSplitKKernel::new(num_heads, num_kv_heads, head_dim, kv);
        let a_type = KernelType::GdnDecodeAttentionSplitK {
            num_heads,
            num_kv_heads,
            head_dim,
            kv_f16: kv == KvStorage::F16,
        };
        let a_key = format!(
            "gdn_decode_attention_splitk_{}_{num_heads}_{num_kv_heads}_{head_dim}",
            kv.tag()
        );
        let a_name = self.gdn_prepare(&a_type, &a_key)?;
        let (gx, gy, _) = a.grid(plan);
        let (bx, _, _) = a.block();
        self.gdn_launch_mixed(
            &a_key,
            a_name,
            LaunchConfig::grid_2d(gx, gy, bx, 1),
            &[
                q.as_ptr(),
                k_cache_ptr,
                v_cache_ptr,
                partial_acc.as_ptr(),
                partial_ml.as_ptr(),
            ],
            &[
                u64::from(seq_len),
                u64::from(plan.chunk),
                u64::from(plan.n_splits),
            ],
        )?;

        let b = DecodeAttentionSplitKReduceKernel::new(num_heads, head_dim);
        let b_type = KernelType::GdnDecodeAttentionSplitKReduce {
            num_heads,
            head_dim,
        };
        let b_key = format!("gdn_decode_attention_splitk_reduce_{num_heads}_{head_dim}");
        let b_name = self.gdn_prepare(&b_type, &b_key)?;
        let (gx, _, _) = b.grid();
        let (bx, _, _) = b.block();
        self.gdn_launch_mixed(
            &b_key,
            b_name,
            LaunchConfig::grid_2d(gx, 1, bx, 1),
            &[partial_acc.as_ptr(), partial_ml.as_ptr(), output.as_ptr()],
            &[u64::from(plan.n_splits)],
        )
    }

    /// The split count the scratch must hold for every `seq_len` the default plan
    /// can produce: [`SplitKPlan`] never exceeds its target, so this is the target.
    #[must_use]
    pub const fn gdn_decode_attention_splitk_scratch_splits() -> u32 {
        trueno_gpu::kernels::gdn::SPLITK_DEFAULT_TARGET_SPLITS
    }
}

/// Through the executor (KernelType arms, PTX cache, `grid_2d`, scratch sizing),
/// at a context long enough for many splits — the Qwen3.5 forward tests use short
/// prompts, so on their own they only ever launch one split.
#[cfg(test)]
mod tests {
    use super::*;
    use trueno_gpu::kernels::gdn::decode_attention_reference_f64;

    const HEAD_DIM: u32 = 256;

    /// A seeded LCG in `[-scale, scale)`.
    fn lcg(seed: &mut u32, scale: f32) -> f32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (f32::from(((*seed >> 16) & 0xFFFF) as u16) / 32768.0 - 1.0) * scale
    }

    fn scratch(ex: &CudaExecutor, num_heads: u32) -> (GpuBuffer<f32>, GpuBuffer<f32>) {
        let splits = CudaExecutor::gdn_decode_attention_splitk_scratch_splits();
        (
            GpuBuffer::new(
                &ex.context,
                splitk_partial_acc_len(num_heads, HEAD_DIM, splits),
            )
            .expect("partial acc"),
            GpuBuffer::new(&ex.context, splitk_partial_ml_len(num_heads, splits))
                .expect("partial ml"),
        )
    }

    #[test]
    #[serial_test::serial]
    fn gdn_splitk_through_the_executor_matches_the_16_block_kernel_at_many_splits() {
        let mut ex = crate::cuda_executor_or_skip!(0);
        // The 9B's attention geometry; 5000 positions is 79 splits of 64.
        let (num_heads, num_kv_heads, seq_len) = (16u32, 4u32, 5000u32);
        assert!(SplitKPlan::for_seq_len(seq_len).n_splits > 1);
        let row = (num_kv_heads * HEAD_DIM) as usize;
        let mut seed = 0x3725_5E4Eu32;
        let q: Vec<f32> = (0..(num_heads * HEAD_DIM) as usize)
            .map(|_| lcg(&mut seed, 0.3))
            .collect();
        let k: Vec<f32> = (0..seq_len as usize * row)
            .map(|_| lcg(&mut seed, 0.8))
            .collect();
        let v: Vec<f32> = (0..seq_len as usize * row)
            .map(|_| lcg(&mut seed, 0.8))
            .collect();
        let q_buf = GpuBuffer::from_host(&ex.context, &q).expect("q");
        let k_buf = GpuBuffer::from_host(&ex.context, &k).expect("k");
        let v_buf = GpuBuffer::from_host(&ex.context, &v).expect("v");
        let old = GpuBuffer::<f32>::new(&ex.context, q.len()).expect("old out");
        let new = GpuBuffer::<f32>::new(&ex.context, q.len()).expect("new out");
        let (pacc, pml) = scratch(&ex, num_heads);

        ex.gdn_decode_attention_into(
            &q_buf,
            &k_buf,
            &v_buf,
            &old,
            num_heads,
            num_kv_heads,
            HEAD_DIM,
            seq_len,
        )
        .expect("16-block kernel");
        ex.gdn_decode_attention_splitk_into(
            &q_buf,
            k_buf.as_ptr(),
            v_buf.as_ptr(),
            KvStorage::F32,
            &pacc,
            &pml,
            &new,
            num_heads,
            num_kv_heads,
            HEAD_DIM,
            seq_len,
        )
        .expect("split-K");
        ex.stream.synchronize().expect("sync");
        let mut got_old = vec![0.0f32; q.len()];
        let mut got_new = vec![0.0f32; q.len()];
        old.copy_to_host(&mut got_old).expect("download old");
        new.copy_to_host(&mut got_new).expect("download new");

        let reference = decode_attention_reference_f64(
            &q,
            &k,
            &v,
            num_heads as usize,
            num_kv_heads as usize,
            HEAD_DIM as usize,
            seq_len as usize,
        );
        let scale = reference.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(
            scale > 1e-4,
            "the reference is all ~zero; nothing would be checked"
        );
        for (i, want) in reference.iter().enumerate() {
            let d_new = (f64::from(got_new[i]) - want).abs();
            let d_old = (f64::from(got_old[i]) - want).abs();
            assert!(
                d_new <= 1e-4 * scale && d_old <= 1e-4 * scale,
                "element {i}: split-K {} / 16-block {} / f64 {want} (limit {})",
                got_new[i],
                got_old[i],
                1e-4 * scale
            );
        }
    }

    #[test]
    #[serial_test::serial]
    fn gdn_splitk_refuses_an_empty_cache_and_short_scratch() {
        let mut ex = crate::cuda_executor_or_skip!(0);
        let (num_heads, num_kv_heads) = (16u32, 4u32);
        let q = GpuBuffer::<f32>::new(&ex.context, (num_heads * HEAD_DIM) as usize).expect("q");
        let kv = GpuBuffer::<f32>::new(&ex.context, (num_kv_heads * HEAD_DIM) as usize * 4096)
            .expect("kv");
        let out = GpuBuffer::<f32>::new(&ex.context, (num_heads * HEAD_DIM) as usize).expect("o");
        let (pacc, pml) = scratch(&ex, num_heads);
        let empty = ex.gdn_decode_attention_splitk_into(
            &q,
            kv.as_ptr(),
            kv.as_ptr(),
            KvStorage::F32,
            &pacc,
            &pml,
            &out,
            num_heads,
            num_kv_heads,
            HEAD_DIM,
            0,
        );
        assert!(
            empty.is_err(),
            "seq_len 0 has no softmax and must be refused"
        );

        // Scratch for ONE split, asked to serve 4096 positions (64 splits).
        let small_acc =
            GpuBuffer::<f32>::new(&ex.context, splitk_partial_acc_len(num_heads, HEAD_DIM, 1))
                .expect("small acc");
        let small_ml =
            GpuBuffer::<f32>::new(&ex.context, splitk_partial_ml_len(num_heads, 1)).expect("ml");
        let short = ex.gdn_decode_attention_splitk_into(
            &q,
            kv.as_ptr(),
            kv.as_ptr(),
            KvStorage::F32,
            &small_acc,
            &small_ml,
            &out,
            num_heads,
            num_kv_heads,
            HEAD_DIM,
            4096,
        );
        let msg = format!("{:?}", short.expect_err("short scratch must be refused"));
        assert!(msg.contains("scratch too small"), "{msg}");
    }
}
