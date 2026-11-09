//! APR-EMBED-001 EG-4 (#4903): executor ops for the EmbeddingGemma 2 encoder forward.
//!
//! Every op takes device pointers and runtime shapes, so one module per kernel serves every
//! input length. The kernels are the shared `aprender-gpu` ones (I-10); the only new one is
//! [`BandSoftmaxKernel`], the bidirectional (optionally windowed) twin of the causal mask.
//!
//! I-5: an encoder activation never runs in float16. [`CudaExecutor::eg2_project`] is the one
//! GEMM this forward dispatches: weights are dequantized to f32 and multiplied by the
//! pedantic-math cuBLAS handle (fp32 in, out and accumulate, no TF32). It never takes the
//! Qwen3.5 prefill's `APR_QWEN35_PREFILL_GEMM` legs, and it refuses, naming the kernel, on an
//! executor whose fp16 GEMM (`gemm_f16_to_f32`) is armed.

use super::*;
use trueno_gpu::kernels::{
    BandSoftmaxKernel, ElementwiseMulKernel, GeluKernel, PerHeadRmsNormKernel, ResidualAddKernel,
    ScaleKernel, BAND_UNBOUNDED,
};

/// Threads per block for the elementwise kernels.
const EW_THREADS: u32 = 256;

/// The refusal I-5 names: an fp16 GEMM on an executor this encoder would dispatch on.
pub(crate) const EG2_F16_REFUSAL: &str =
    "gemm_f16_to_f32 is armed on this executor: EmbeddingGemma 2 activations never run in \
     float16 (I-5); use an executor without the fp16 prefill GEMM";

/// The key range `[lo, hi)` the query rows `r0..r0 + n` of a `seq`-long input attend to
/// under half-window `w`.
fn key_range(r0: u32, n: u32, seq: u32, w: u32) -> (u32, u32) {
    let lo = r0.saturating_sub(w);
    let hi = (u64::from(r0) + u64::from(n) + u64::from(w)).min(u64::from(seq));
    (lo, hi as u32)
}

/// The most query rows per pass whose scores (`hpk * n * keys` floats) fit in `cap`;
/// `None` when not even one row fits.
fn pass_rows(seq: u32, hpk: u32, w: u32, cap: usize) -> Option<u32> {
    let cost = |n: u32| {
        let keys = (u64::from(n) + 2 * u64::from(w)).min(u64::from(seq));
        u64::from(hpk) * u64::from(n) * keys
    };
    let cap = cap as u64;
    if cost(1) > cap {
        return None;
    }
    let (mut lo, mut hi) = (1u32, seq);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if cost(mid) <= cap {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Some(lo)
}

impl CudaExecutor {
    /// `Y[rows × n] = alpha · X[rows × k] · Wᵀ` for the `[n × k]` weight at `w_ptr`, f32 end to
    /// end (I-5).
    ///
    /// # Errors
    /// The fp16 GEMM is armed on this executor ([`EG2_F16_REFUSAL`]), no f32 dequant exists for
    /// `qtype`, or cuBLAS fails.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn eg2_project(
        &mut self,
        qtype: WeightQuantType,
        w_ptr: u64,
        x: u64,
        y: u64,
        rows: u32,
        n: u32,
        k: u32,
        alpha: f32,
    ) -> Result<(), GpuError> {
        if self.qwen35_prefill_f16 {
            return Err(GpuError::InvalidParameter(EG2_F16_REFUSAL.to_string()));
        }
        validate_device_ptr(x, "eg2_project x")?;
        validate_device_ptr(y, "eg2_project y")?;
        let w = self.qwen35_dequant_f32(qtype, w_ptr, n, k)?;
        self.ensure_cublas()?;
        let handle = self.cublas_handle.as_ref().expect("cublas initialized");
        // Column-major view: Yᵀ (n × rows) = W (k × n col-major, op T) · Xᵀ (k × rows).
        handle.gemm_f32(
            trueno_gpu::driver::GemmOp::Trans,
            trueno_gpu::driver::GemmOp::NoTrans,
            n as i32,
            rows as i32,
            k as i32,
            alpha,
            w,
            k as i32,
            x,
            k as i32,
            0.0,
            y,
            n as i32,
        )
    }

    /// Whether `qtype` has the device f32 dequant [`Self::eg2_project`] needs, proven by running
    /// it on the `[n × k]` weight at `w_ptr`.
    pub(crate) fn eg2_has_dequant(
        &mut self,
        qtype: WeightQuantType,
        w_ptr: u64,
        n: u32,
        k: u32,
    ) -> bool {
        self.qwen35_dequant_f32(qtype, w_ptr, n, k).is_ok()
    }

    /// RMSNorm of each of `rows` rows of `width` floats with the shared `[width]` `gamma`.
    ///
    /// # Errors
    /// Compile/launch failure or a null pointer.
    pub(crate) fn eg2_rms_rows(
        &mut self,
        input: u64,
        gamma: u64,
        output: u64,
        width: u32,
        rows: u32,
        eps: f32,
    ) -> Result<(), GpuError> {
        // The single-row PTX bakes only the width and epsilon; the row count is the grid.
        let kern = PerHeadRmsNormKernel::new(width, 1).with_epsilon(eps);
        let key = format!("eg2_rms_{width}_{:08x}", eps.to_bits());
        self.qp_prepare(&key, &kern)?;
        let mut args = [input, output, gamma];
        self.qp_launch(
            &key,
            kern.name(),
            LaunchConfig::grid_2d(rows, 1, 32, 1),
            &mut args,
            3,
        )
    }

    /// `y = x · s` over `n` floats (`y` may be `x`).
    ///
    /// # Errors
    /// Compile/launch failure or a null pointer.
    pub(crate) fn eg2_scale(&mut self, x: u64, y: u64, s: f32, n: u32) -> Result<(), GpuError> {
        let kern = ScaleKernel::new(n);
        self.qp_prepare("eg2_scale", &kern)?;
        let mut args = [x, y, u64::from(s.to_bits()), u64::from(n)];
        let config = LaunchConfig::grid_2d(n.div_ceil(EW_THREADS), 1, EW_THREADS, 1);
        self.qp_launch("eg2_scale", kern.name(), config, &mut args, 2)
    }

    /// `y = gelu(x)` (tanh form) over `n` floats (`y` may be `x`).
    ///
    /// # Errors
    /// Compile/launch failure or a null pointer.
    pub(crate) fn eg2_gelu(&mut self, x: u64, y: u64, n: u32) -> Result<(), GpuError> {
        let kern = GeluKernel::new(n);
        self.qp_prepare("eg2_gelu", &kern)?;
        let mut args = [x, y, u64::from(n)];
        let config = LaunchConfig::grid_2d(n.div_ceil(EW_THREADS), 1, EW_THREADS, 1);
        self.qp_launch("eg2_gelu", kern.name(), config, &mut args, 2)
    }

    /// `out = a ⊙ b` over `n` floats.
    ///
    /// # Errors
    /// Compile/launch failure or a null pointer.
    pub(crate) fn eg2_mul(&mut self, a: u64, b: u64, out: u64, n: u32) -> Result<(), GpuError> {
        let kern = ElementwiseMulKernel::new(n);
        self.qp_prepare("eg2_mul", &kern)?;
        let mut args = [a, b, out, u64::from(n)];
        let config = LaunchConfig::grid_2d(n.div_ceil(EW_THREADS), 1, EW_THREADS, 1);
        self.qp_launch("eg2_mul", kern.name(), config, &mut args, 3)
    }

    /// `out = a + b` over `n` floats.
    ///
    /// # Errors
    /// Compile/launch failure or a null pointer.
    pub(crate) fn eg2_add(&mut self, a: u64, b: u64, out: u64, n: u32) -> Result<(), GpuError> {
        let kern = ResidualAddKernel::new(n);
        self.qp_prepare("eg2_add", &kern)?;
        let mut args = [a, b, out, u64::from(n)];
        let config = LaunchConfig::grid_2d(n.div_ceil(EW_THREADS), 1, EW_THREADS, 1);
        self.qp_launch("eg2_add", kern.name(), config, &mut args, 3)
    }

    /// The mean of `rows` rows of `width` floats: `out = (1/rows) · Xᵀ · ones`.
    ///
    /// `ones` holds at least `rows` ones.
    ///
    /// # Errors
    /// cuBLAS failure or a null pointer.
    pub(crate) fn eg2_mean_rows(
        &mut self,
        x: u64,
        ones: u64,
        out: u64,
        rows: u32,
        width: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(x, "eg2_mean_rows x")?;
        validate_device_ptr(ones, "eg2_mean_rows ones")?;
        validate_device_ptr(out, "eg2_mean_rows out")?;
        self.ensure_cublas()?;
        let handle = self.cublas_handle.as_ref().expect("cublas initialized");
        // Column-major: X is width × rows (ld width); out (width × 1) = X · ones (rows × 1).
        handle.gemm_f32(
            trueno_gpu::driver::GemmOp::NoTrans,
            trueno_gpu::driver::GemmOp::NoTrans,
            width as i32,
            1,
            rows as i32,
            1.0 / rows as f32,
            x,
            width as i32,
            ones,
            rows as i32,
            0.0,
            out,
            width as i32,
        )
    }

    /// Bidirectional grouped-query attention over a whole `seq`-long input, the device twin of
    /// `OwnedQuantizedModel::parallel_multihead_attention_windowed`.
    ///
    /// `q`/`out` are `[seq][heads * head_dim]`, `k`/`v` `[seq][kv_heads * head_dim]`, all
    /// row-major. Query `i` attends to the keys `j` with `|i - j| <= half_window`, or to every
    /// key when `half_window` is `None`. Scores are `scale · q·k`. The query rows run in passes
    /// whose `[heads/kv_heads][rows][keys]` scores fit in the `scores_cap` floats at `scores`.
    ///
    /// # Errors
    /// `heads` is not a multiple of `kv_heads`, one query row's scores exceed `scores_cap`, or a
    /// cuBLAS / launch failure.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn eg2_attention(
        &mut self,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        scores: u64,
        scores_cap: usize,
        seq: u32,
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
        scale: f32,
        half_window: Option<u32>,
    ) -> Result<(), GpuError> {
        if kv_heads == 0 || heads % kv_heads != 0 {
            return Err(GpuError::InvalidParameter(format!(
                "eg2_attention: {heads} query heads over {kv_heads} KV heads"
            )));
        }
        for (p, name) in [
            (q, "q"),
            (k, "k"),
            (v, "v"),
            (out, "out"),
            (scores, "scores"),
        ] {
            validate_device_ptr(p, &format!("eg2_attention {name}"))?;
        }
        let hpk = heads / kv_heads;
        let w = half_window.map_or(BAND_UNBOUNDED, |w| w.min(BAND_UNBOUNDED));
        let rows = pass_rows(seq, hpk, w, scores_cap).ok_or_else(|| {
            GpuError::InvalidParameter(format!(
                "eg2_attention: one query row's scores exceed the {scores_cap}-float buffer"
            ))
        })?;
        self.ensure_cublas()?;
        self.qp_prepare("eg2_band_softmax", &BandSoftmaxKernel)?;
        let shape = PassShape {
            heads,
            kv_heads,
            head_dim,
            scale,
            w,
        };
        let mut r0 = 0;
        while r0 < seq {
            let n = rows.min(seq - r0);
            let (lo, hi) = key_range(r0, n, seq, w);
            self.eg2_attention_pass(&shape, [q, k, v, out, scores], r0, n, lo, hi)?;
            r0 += n;
        }
        Ok(())
    }

    /// Query rows `r0..r0 + n` over keys `lo..hi`, for every KV group.
    fn eg2_attention_pass(
        &mut self,
        s: &PassShape,
        [q, k, v, out, scores]: [u64; 5],
        r0: u32,
        n: u32,
        lo: u32,
        hi: u32,
    ) -> Result<(), GpuError> {
        let hpk = s.heads / s.kv_heads;
        let q_dim = s.heads * s.head_dim;
        let kv_dim = s.kv_heads * s.head_dim;
        let total = hi - lo;
        let per_head = i64::from(n) * i64::from(total);
        let f = std::mem::size_of::<f32>() as u64;
        for g in 0..s.kv_heads {
            let kv_off = (u64::from(lo) * u64::from(kv_dim) + u64::from(g * s.head_dim)) * f;
            let q_off = (u64::from(r0) * u64::from(q_dim) + u64::from(g * hpk * s.head_dim)) * f;
            let handle = self.cublas_handle.as_ref().expect("cublas initialized");
            // scores[h][t][j] = scale · q_h[t] · k[j]: column-major C (total × n).
            handle.gemm_f32_strided_batched(
                trueno_gpu::driver::GemmOp::Trans,
                trueno_gpu::driver::GemmOp::NoTrans,
                total as i32,
                n as i32,
                s.head_dim as i32,
                s.scale,
                k + kv_off,
                kv_dim as i32,
                0,
                q + q_off,
                q_dim as i32,
                i64::from(s.head_dim),
                0.0,
                scores,
                total as i32,
                per_head,
                hpk as i32,
            )?;
            let mut args = [
                scores,
                u64::from(n),
                u64::from(total),
                u64::from(r0 - lo),
                u64::from(s.w),
            ];
            self.qp_launch(
                "eg2_band_softmax",
                "band_softmax",
                LaunchConfig::grid_2d(hpk, n, 32, 1),
                &mut args,
                1,
            )?;
            let handle = self.cublas_handle.as_ref().expect("cublas initialized");
            // out_h[t] = Σ_j p[h][t][j] v[j]: column-major C (head_dim × n, ld q_dim).
            handle.gemm_f32_strided_batched(
                trueno_gpu::driver::GemmOp::NoTrans,
                trueno_gpu::driver::GemmOp::NoTrans,
                s.head_dim as i32,
                n as i32,
                total as i32,
                1.0,
                v + kv_off,
                kv_dim as i32,
                0,
                scores,
                total as i32,
                per_head,
                0.0,
                out + q_off,
                q_dim as i32,
                i64::from(s.head_dim),
                hpk as i32,
            )?;
        }
        Ok(())
    }
}

/// The shape an attention pass runs at.
struct PassShape {
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    scale: f32,
    w: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_range_clips_to_the_input_and_the_window() {
        assert_eq!(key_range(0, 4, 10, 2), (0, 6));
        assert_eq!(key_range(5, 4, 10, 2), (3, 10));
        assert_eq!(key_range(3, 2, 10, BAND_UNBOUNDED), (0, 10));
        assert_eq!(key_range(0, 10, 10, 0), (0, 10));
    }

    #[test]
    fn pass_rows_is_the_largest_pass_that_fits() {
        // Global: 3 heads per group, 8000 keys per row.
        assert_eq!(
            pass_rows(8000, 3, BAND_UNBOUNDED, 3 * 8000 * 100),
            Some(100)
        );
        assert_eq!(pass_rows(8000, 3, BAND_UNBOUNDED, 3 * 8000 - 1), None);
        // The whole input fits.
        assert_eq!(pass_rows(16, 1, BAND_UNBOUNDED, 1 << 20), Some(16));
        // Windowed: n rows see n + 2w keys.
        let n = pass_rows(8000, 2, 256, 2 * 1000 * (1000 + 512)).expect("fits");
        assert_eq!(n, 1000);
    }
}
