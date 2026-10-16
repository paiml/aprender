impl CudaExecutor {
    /// #5303: [`Self::gemv_dispatch`] over `m` stacked activation rows:
    /// `output[r] = W · input[r]`, `input` `m × k` and `output` `m × n`.
    ///
    /// Row `r` is bitwise what [`Self::gemv_dispatch`] gives for `input[r]`
    /// alone. Where the single-row kernel has a batched twin (Q4_K on `Mwv`,
    /// Q5_K, Q8_0, and aligned Q6_K on `Mwv` or `HwDp4a`), the twin reads each
    /// weight once for all rows; every other type and variant runs the
    /// single-row kernel once per row. One row is always the single-row call.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_gemv_dispatch(
        &mut self,
        qtype: WeightQuantType,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        use crate::cuda::gpu_profile::{Q4kVariant, Q6kVariant};
        if m == 1 {
            return self.gemv_dispatch(qtype, weight_ptr, input, output, n, k);
        }
        match qtype {
            WeightQuantType::Q4K if self.gpu_profile.q4k == Q4kVariant::Mwv => {
                self.batched_mwv_q4k_gemv_into(weight_ptr, input, output, m, n, k)
            },
            WeightQuantType::Q6K
                if k.is_multiple_of(256) && self.gpu_profile.q6k == Q6kVariant::Mwv =>
            {
                self.batched_mwv_q6k_gemv_into(weight_ptr, input, output, m, n, k)
            },
            WeightQuantType::Q5K => self.batched_q5k_gemv_into(weight_ptr, input, output, m, n, k),
            WeightQuantType::Q8_0 => {
                self.batched_q8_0_gemv_into(weight_ptr, input, output, m, n, k)
            },
            WeightQuantType::Q6K
                if k.is_multiple_of(256) && self.gpu_profile.q6k == Q6kVariant::HwDp4a =>
            {
                self.batched_hw_dp4a_q6k_gemv_into(weight_ptr, input, output, m, n, k)
            },
            _ => {
                for r in 0..m {
                    // SAFETY: rows r < m of an `m × k` input and an `m × n`
                    // output (the caller's contract); the views are forgotten
                    // before they could free memory they do not own.
                    let x = unsafe {
                        GpuBuffer::<f32>::from_raw_parts(
                            input.as_ptr() + u64::from(r * k) * 4,
                            k as usize,
                        )
                    };
                    // SAFETY: as for `x`, row r of the `m × n` output.
                    let y = unsafe {
                        GpuBuffer::<f32>::from_raw_parts(
                            output.as_ptr() + u64::from(r * n) * 4,
                            n as usize,
                        )
                    };
                    // #4258: the Q8 activation cache is keyed on a buffer's
                    // base pointer, and a whole-batch write at the batch's
                    // base never invalidates a row view; quantize afresh.
                    self.q8_activation_valid = false;
                    let run = self.gemv_dispatch(qtype, weight_ptr, &x, &y, n, k);
                    std::mem::forget(x);
                    std::mem::forget(y);
                    run?;
                }
                Ok(())
            },
        }
    }
}
