//! #3807: the FP8 prefill recipe on the real kernels — E4M3 subnormals are encoded, and one
//! token's outlier no longer sets every token's scale.
//!
//! The old recipe quantized the whole activation batch with one `448/absmax` and flushed E4M3
//! subnormals, so qwen2.5-coder-7b's down-projection outlier (absmax 871) zeroed every other
//! token's inputs below 0.03, and `apr run --chat` failed the F2 gate at position 1 (cosine
//! 0.8134). These rows need a CUDA device and skip, loudly, without one.

use crate::cuda::executor::CudaExecutor;
use trueno_gpu::driver::GpuBuffer;

/// E4M3FN decode (bias 7, 3 mantissa bits, exponent 0 = subnormal `m × 2^-9`).
fn e4m3_decode(code: u8) -> f64 {
    let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
    let e = i32::from((code >> 3) & 0x0F);
    let m = f64::from(code & 0x07);
    let mag = if e == 0 {
        m * 2f64.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    sign * mag
}

/// Reference encoder: nearest finite E4M3 value, ties to the even code, saturating at 448.
fn e4m3_encode_ref(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80 } else { 0x00 };
    let a = f64::from(x.abs());
    let mut best = 0u8;
    for code in 1u8..=0x7E {
        let (d_new, d_best) = ((e4m3_decode(code) - a).abs(), (e4m3_decode(best) - a).abs());
        if d_new < d_best || (d_new == d_best && code % 2 == 0) {
            best = code;
        }
    }
    sign | best
}

/// Every finite E4M3 magnitude, the midpoints between neighbours (the ties), values just off
/// each midpoint, values past 448, values too small to round up, and all of it negated.
fn quantizer_inputs() -> Vec<f32> {
    let grid: Vec<f64> = (0u8..=0x7E).map(e4m3_decode).collect();
    let mut xs = Vec::new();
    for w in grid.windows(2) {
        let mid = (w[0] + w[1]) / 2.0;
        let eps = (w[1] - w[0]) * 1e-3;
        xs.extend([w[0], mid, mid - eps, mid + eps]);
    }
    xs.extend([
        448.0,
        450.0,
        463.9,
        500.0,
        1e6,
        2f64.powi(-11),
        2f64.powi(-10) * 0.9,
    ]);
    let pos: Vec<f32> = xs.iter().map(|&v| v as f32).collect();
    let mut all = pos.clone();
    all.extend(pos.iter().map(|v| -v));
    all
}

#[test]
fn the_quantizer_encodes_every_e4m3_value_including_subnormals() {
    let Ok(mut exec) = CudaExecutor::new(0) else {
        eprintln!("SKIP #3807 quantizer row: no CUDA device");
        return;
    };
    let xs = quantizer_inputs();
    let want: Vec<u8> = xs.iter().map(|&x| e4m3_encode_ref(x)).collect();
    // Not vacuous: the subnormal range must be exercised with non-zero codes (the old
    // converters wrote 0 for every one of these).
    let subnormal_nonzero = xs
        .iter()
        .zip(&want)
        .filter(|(x, c)| x.abs() < 2f32.powi(-6) && **c & 0x7F != 0)
        .count();
    assert!(
        subnormal_nonzero >= 40,
        "only {subnormal_nonzero} subnormal cases"
    );

    // One row whose absmax is 448, so the quantizer's scale is exactly 1.
    let cols = xs.len() as u32;
    let src = GpuBuffer::from_host(&exec.context, &xs).expect("upload");
    let absmax = GpuBuffer::from_host(&exec.context, &[448.0f32]).expect("absmax");
    let dst = GpuBuffer::<u8>::new(&exec.context, xs.len()).expect("dst");
    exec.fp8_quantize_rows(src.as_ptr(), dst.as_ptr(), 1, cols, absmax.as_ptr())
        .expect("quantize");
    exec.stream.synchronize().expect("sync");
    let mut got = vec![0u8; xs.len()];
    dst.copy_to_host(&mut got).expect("download");

    let bad: Vec<String> = xs
        .iter()
        .zip(got.iter().zip(&want))
        .filter(|(_, (g, w))| g != w)
        .take(8)
        .map(|(x, (g, w))| format!("x={x:e}: got {g:#04x}, want {w:#04x}"))
        .collect();
    assert!(
        bad.is_empty(),
        "E4M3 encoding differs from the reference (first 8): {bad:?}"
    );
}

/// Deterministic values in [-1, 1).
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

#[test]
fn an_outlier_token_does_not_zero_the_other_tokens() {
    let Ok(mut exec) = CudaExecutor::new(0) else {
        eprintln!("SKIP #3807 outlier row: no CUDA device");
        return;
    };
    let (m, n, k) = (16usize, 64usize, 256usize);
    let mut seed = 3807u64;
    // Token 0 carries qwen2.5-coder-7b's layer-27 down-projection outlier; the rest are small
    // activations. Under one batch scale (448/871) they land at most 0.0051, a few 2^-9
    // subnormal steps: wrong by tens of percent even with subnormals kept, and zero with FTZ.
    let mut x: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed) * 0.01).collect();
    x[0] = 871.0;
    // Weights in [-1, 1). Channel 3 carries an 8.0 outlier and channel 7 is ~2.7e5x smaller: a
    // scale shared by the whole tensor (448/8) puts channel 7 below E4M3's smallest subnormal.
    let mut w: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed)).collect();
    w[3 * k + 5] = 8.0;
    for v in &mut w[7 * k..8 * k] {
        *v *= 3e-5;
    }

    // Not vacuous: under ONE batch scale, tokens 1..m sit below E4M3's smallest normal.
    let below = x[k..]
        .iter()
        .filter(|v| (v.abs() * 448.0 / 871.0) < 2f32.powi(-6))
        .count();
    assert!(
        below * 2 > x.len() - k,
        "fixture: only {below} values under the batch-scale threshold"
    );

    // The weight goes through the same quantizer `get_or_cache_fp8_weight` uses.
    let w_f32 = GpuBuffer::from_host(&exec.context, &w).expect("upload w");
    let (w_fp8, w_absmax) = exec
        .fp8_quantize_weight(w_f32.as_ptr(), n as u32, k as u32)
        .expect("quantize weight");
    let weight_key = w_fp8.as_ptr();
    exec.fp8_weight_row_absmax.insert(weight_key, w_absmax);

    let x_buf = GpuBuffer::from_host(&exec.context, &x).expect("upload x");
    let out = GpuBuffer::<f32>::new(&exec.context, m * n).expect("out");
    exec.cublas_prefill_fp8_gemm(
        w_fp8.as_ptr(),
        weight_key,
        x_buf.as_ptr(),
        out.as_ptr(),
        m as u32,
        n as u32,
        k as u32,
    )
    .expect("FP8 GEMM");
    exec.stream.synchronize().expect("sync");
    let mut y = vec![0.0f32; m * n];
    out.copy_to_host(&mut y).expect("download");

    // f64 reference, then relative L2 errors two ways. Per token (over its n channels): the old
    // recipe lost most of tokens 1..m's inputs. Per channel (over tokens 1..m, so token 0's
    // outlier does not dominate): a tensor-wide weight scale loses channel 7. E4M3's 3-bit
    // mantissa alone gives a few percent.
    let reference: Vec<f64> = (0..m * n)
        .map(|i| {
            let (t, c) = (i / n, i % n);
            (0..k)
                .map(|j| f64::from(x[t * k + j]) * f64::from(w[c * k + j]))
                .sum()
        })
        .collect();
    let rel = |cells: &mut dyn Iterator<Item = usize>| {
        let (mut err, mut norm) = (0.0f64, 0.0f64);
        for i in cells {
            err += (f64::from(y[i]) - reference[i]).powi(2);
            norm += reference[i].powi(2);
        }
        (err / norm).sqrt()
    };
    // Bounds: a token's error averages over 64 channels (8%); a channel's over 15 tokens is
    // noisier (20%). The failures they exist for are 19% (shared token scale) and ~100% (shared
    // weight scale, channel 7).
    let worst_token = (0..m)
        .map(|t| (rel(&mut (0..n).map(|c| t * n + c)), t))
        .fold((0.0f64, 0usize), |a, b| if b.0 > a.0 { b } else { a });
    let worst_channel = (0..n)
        .map(|c| (rel(&mut (1..m).map(|t| t * n + c)), c))
        .fold((0.0f64, 0usize), |a, b| if b.0 > a.0 { b } else { a });
    eprintln!(
        "#3807 outlier row: worst token {} at {:.2}%, worst channel {} at {:.2}%",
        worst_token.1,
        worst_token.0 * 100.0,
        worst_channel.1,
        worst_channel.0 * 100.0
    );
    assert!(
        worst_token.0 < 0.08 && worst_channel.0 < 0.20,
        "FP8 GEMM off the f64 reference (token {} {:.1}% vs 8%, channel {} {:.1}% vs 20%): a scale \
         shared across tokens or channels, or a flush-to-zero, is back (#3807)",
        worst_token.1,
        worst_token.0 * 100.0,
        worst_channel.1,
        worst_channel.0 * 100.0
    );
}
