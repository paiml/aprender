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
