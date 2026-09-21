//! #3728: the FP8 prefill GEMM's intermediate must not overflow when its activation is small.
//!
//! `cublas_prefill_fp8_gemm` quantizes the activation with `448/act_absmax`, folds the weight
//! dequant into cuBLASLt's alpha, and writes `D = true_result × 448/act_absmax` before step 4
//! multiplies `act_absmax/448` back out. With an FP16 D that gain saturates at 65,504:
//! qwen2.5-coder-7b's layer-1 QKV (input absmax 0.0602, output up to 11.54) reached 85,807 and
//! `apr run --gpu` fell back at cosine 0.5668. This row rebuilds that range on the real kernel.
//! It needs a CUDA device and skips, loudly, without one.

use crate::cuda::executor::CudaExecutor;
use trueno_gpu::driver::GpuBuffer;

/// E4M3 `0x7E` is +448 (S=0, E=1111, M=110: 1.75 × 2^8), the format's largest finite value, so a
/// weight of 1.0 with `w_absmax = 1.0` quantizes to it exactly and dequantizes with 1/448.
const E4M3_448: u8 = 0x7E;

#[test]
fn an_fp8_gemm_whose_activation_gain_overflows_fp16_still_matches_f32() {
    let Ok(mut exec) = CudaExecutor::new(0) else {
        eprintln!("SKIP #3728 fp8 range row: no CUDA device");
        return;
    };
    // One tensor-core tile of rows (m_padded == m, so no padded tail is involved).
    let (m, n, k) = (16u32, 64u32, 256u32);

    // Activation: 0.05 everywhere, absmax 0.0602 (the 7b's layer-1 attn_norm output absmax).
    let act_absmax = 0.0602f32;
    let mut x = vec![0.05f32; (m * k) as usize];
    x[0] = act_absmax;
    let weights = vec![E4M3_448; (n * k) as usize];

    // f32 reference: every weight is 1.0, so each output is its row's sum.
    let reference: Vec<f32> = (0..m as usize)
        .flat_map(|row| {
            let sum: f32 = x[row * k as usize..(row + 1) * k as usize].iter().sum();
            std::iter::repeat_n(sum, n as usize)
        })
        .collect();

    // Not vacuous: at this range the pre-fix FP16 D must overflow. D = true × 448/act_absmax.
    let peak = reference.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let d_peak = peak * 448.0 / act_absmax;
    assert!(
        d_peak > 65_504.0,
        "fixture must drive D past FP16's 65,504 (got {d_peak}); otherwise this row proves nothing"
    );

    let x_buf = GpuBuffer::from_host(&exec.context, &x).expect("upload activation");
    let w_buf = GpuBuffer::from_host(&exec.context, &weights).expect("upload FP8 weight");
    let out = GpuBuffer::<f32>::new(&exec.context, (m * n) as usize).expect("alloc output");
    let weight_key = w_buf.as_ptr();
    exec.fp8_weight_scales.insert(weight_key, 1.0 / 448.0);

    exec.cublas_prefill_fp8_gemm(
        w_buf.as_ptr(),
        weight_key,
        x_buf.as_ptr(),
        out.as_ptr(),
        m,
        n,
        k,
    )
    .expect("FP8 GEMM");
    exec.stream.synchronize().expect("sync");
    let mut got = vec![0.0f32; (m * n) as usize];
    out.copy_to_host(&mut got).expect("download");

    let non_finite = got.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite,
        0,
        "D saturated: {non_finite} of {} outputs are inf/NaN",
        got.len()
    );
    // FP8's own error on this input: 0.05 × 448/0.0602 = 372.1 rounds to the E4M3 grid point
    // 384 (+3.2%), so 5% bounds the quantization and nothing else.
    let worst = got
        .iter()
        .zip(&reference)
        .map(|(g, r)| ((g - r) / r).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 0.05,
        "FP8 GEMM off the f32 reference by {:.2}% (bound 5%)",
        worst * 100.0
    );
}
