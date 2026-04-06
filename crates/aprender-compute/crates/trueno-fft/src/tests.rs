//! Tests for FFT — provable contracts and falsification tests.

use crate::bluestein::bluestein_fft;
use crate::complex::Complex;
use crate::stockham::{fft_2d, fft_c2r, fft_r2c, FftPlan};

// ============================================================================
// FALSIFY-FFT-001: Parseval energy conservation
// ============================================================================

#[test]
fn test_parseval_energy_conservation_size_4() {
    let plan = FftPlan::new(4).expect("valid plan");
    let input = [
        Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(4.0, 0.0),
    ];
    let mut output = [Complex::ZERO; 4];
    plan.forward(&input, &mut output).expect("fft ok");

    let energy_time: f32 = input.iter().map(|x| x.norm_sq()).sum();
    let energy_freq: f32 = output.iter().map(|x| x.norm_sq()).sum::<f32>() / 4.0;
    assert!(
        (energy_time - energy_freq).abs() < 1e-4,
        "Parseval violated: time={energy_time}, freq={energy_freq}"
    );
}

#[test]
fn test_parseval_energy_conservation_size_8() {
    let plan = FftPlan::new(8).expect("valid plan");
    let input: Vec<Complex> = (0..8).map(|i| Complex::new(i as f32, 0.0)).collect();
    let mut output = vec![Complex::ZERO; 8];
    plan.forward(&input, &mut output).expect("fft ok");

    let energy_time: f32 = input.iter().map(|x| x.norm_sq()).sum();
    let energy_freq: f32 = output.iter().map(|x| x.norm_sq()).sum::<f32>() / 8.0;
    assert!(
        (energy_time - energy_freq).abs() < 1e-3,
        "Parseval violated: time={energy_time}, freq={energy_freq}"
    );
}

// ============================================================================
// FALSIFY-FFT-002: Inverse roundtrip
// ============================================================================

#[test]
fn test_inverse_roundtrip_size_4() {
    let plan = FftPlan::new(4).expect("valid plan");
    let input = [
        Complex::new(1.0, 2.0),
        Complex::new(3.0, -1.0),
        Complex::new(0.5, 0.5),
        Complex::new(-2.0, 1.0),
    ];
    let mut freq = [Complex::ZERO; 4];
    let mut recovered = [Complex::ZERO; 4];

    plan.forward(&input, &mut freq).expect("fft ok");
    plan.inverse(&freq, &mut recovered).expect("ifft ok");

    for (i, (orig, rec)) in input.iter().zip(recovered.iter()).enumerate() {
        let err = (*orig - *rec).abs();
        assert!(
            err < 1e-5,
            "Roundtrip failed at index {i}: orig={orig:?}, recovered={rec:?}, err={err}"
        );
    }
}

#[test]
fn test_inverse_roundtrip_size_16() {
    let plan = FftPlan::new(16).expect("valid plan");
    let input: Vec<Complex> =
        (0..16).map(|i| Complex::new((i as f32).sin(), (i as f32).cos())).collect();
    let mut freq = vec![Complex::ZERO; 16];
    let mut recovered = vec![Complex::ZERO; 16];

    plan.forward(&input, &mut freq).expect("fft ok");
    plan.inverse(&freq, &mut recovered).expect("ifft ok");

    for (i, (orig, rec)) in input.iter().zip(recovered.iter()).enumerate() {
        let err = (*orig - *rec).abs();
        assert!(err < 1e-4, "Roundtrip failed at index {i}: err={err}");
    }
}

// ============================================================================
// FALSIFY-FFT-003: Known value — impulse response
// ============================================================================

#[test]
fn test_impulse_response() {
    let plan = FftPlan::new(8).expect("valid plan");
    let mut input = vec![Complex::ZERO; 8];
    input[0] = Complex::new(1.0, 0.0); // Delta function

    let mut output = vec![Complex::ZERO; 8];
    plan.forward(&input, &mut output).expect("fft ok");

    // FFT of impulse = all ones
    for (k, x) in output.iter().enumerate() {
        assert!(
            (x.re - 1.0).abs() < 1e-6 && x.im.abs() < 1e-6,
            "Impulse response wrong at k={k}: {x:?}"
        );
    }
}

#[test]
fn test_dc_signal() {
    let plan = FftPlan::new(4).expect("valid plan");
    let input = [Complex::new(3.0, 0.0); 4]; // Constant signal

    let mut output = [Complex::ZERO; 4];
    plan.forward(&input, &mut output).expect("fft ok");

    // FFT of constant = N*constant at k=0, zero elsewhere
    assert!((output[0].re - 12.0).abs() < 1e-5, "DC component wrong");
    for k in 1..4 {
        assert!(output[k].abs() < 1e-5, "Non-DC component non-zero at k={k}: {:?}", output[k]);
    }
}

// ============================================================================
// Linearity tests
// ============================================================================

#[test]
fn test_linearity() {
    let plan = FftPlan::new(8).expect("valid plan");
    let x: Vec<Complex> = (0..8).map(|i| Complex::new(i as f32, 0.0)).collect();
    let y: Vec<Complex> = (0..8).map(|i| Complex::new(0.0, (i as f32).sin())).collect();

    let alpha = Complex::new(2.0, 0.0);
    let beta = Complex::new(0.5, 0.0);

    // FFT(α*x + β*y)
    let combined: Vec<Complex> =
        x.iter().zip(y.iter()).map(|(&xi, &yi)| alpha * xi + beta * yi).collect();
    let mut fft_combined = vec![Complex::ZERO; 8];
    plan.forward(&combined, &mut fft_combined).expect("ok");

    // α*FFT(x) + β*FFT(y)
    let mut fft_x = vec![Complex::ZERO; 8];
    let mut fft_y = vec![Complex::ZERO; 8];
    plan.forward(&x, &mut fft_x).expect("ok");
    plan.forward(&y, &mut fft_y).expect("ok");

    for k in 0..8 {
        let expected = alpha * fft_x[k] + beta * fft_y[k];
        let err = (fft_combined[k] - expected).abs();
        assert!(err < 1e-4, "Linearity violated at k={k}: err={err}");
    }
}

// ============================================================================
// R2C tests
// ============================================================================

#[test]
fn test_r2c_basic() {
    let plan = FftPlan::new(8).expect("valid plan");
    let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let mut output = vec![Complex::ZERO; 5]; // N/2 + 1

    plan.forward_r2c(&input, &mut output).expect("r2c ok");

    // DC component = sum of input
    let expected_dc: f32 = input.iter().sum();
    assert!(
        (output[0].re - expected_dc).abs() < 1e-4,
        "DC wrong: got {}, expected {expected_dc}",
        output[0].re
    );
    // DC imaginary should be ~0 for real input
    assert!(output[0].im.abs() < 1e-5);
}

// ============================================================================
// 2D FFT tests
// ============================================================================

#[test]
fn test_fft_2d_impulse() {
    let mut input = vec![Complex::ZERO; 4 * 4];
    input[0] = Complex::new(1.0, 0.0);
    let mut output = vec![Complex::ZERO; 16];

    fft_2d(&input, &mut output, 4, 4).expect("2d fft ok");

    // 2D impulse → all ones
    for (k, x) in output.iter().enumerate() {
        assert!((x.re - 1.0).abs() < 1e-5 && x.im.abs() < 1e-5, "2D impulse wrong at k={k}: {x:?}");
    }
}

// ============================================================================
// Error handling tests
// ============================================================================

#[test]
fn test_zero_length_rejected() {
    assert!(FftPlan::new(0).is_err());
}

#[test]
fn test_non_power_of_two_rejected() {
    assert!(FftPlan::new(3).is_err());
    assert!(FftPlan::new(5).is_err());
    assert!(FftPlan::new(6).is_err());
}

#[test]
fn test_dimension_mismatch_rejected() {
    let plan = FftPlan::new(4).expect("valid plan");
    let input = [Complex::ZERO; 4];
    let mut output = [Complex::ZERO; 8]; // Wrong size

    assert!(plan.forward(&input, &mut output).is_err());
}

#[test]
fn test_size_1_fft() {
    let plan = FftPlan::new(1).expect("valid plan");
    let input = [Complex::new(42.0, -7.0)];
    let mut output = [Complex::ZERO; 1];

    plan.forward(&input, &mut output).expect("fft ok");
    assert!((output[0].re - 42.0).abs() < 1e-6);
    assert!((output[0].im - (-7.0)).abs() < 1e-6);
}

// ============================================================================
// Property-based tests
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_complex_vec(n: usize) -> impl Strategy<Value = Vec<Complex>> {
        proptest::collection::vec((-100.0f32..100.0, -100.0f32..100.0), n)
            .prop_map(|v| v.into_iter().map(|(re, im)| Complex::new(re, im)).collect())
    }

    proptest! {
        #[test]
        fn prop_parseval_conservation(input in arb_complex_vec(16)) {
            let plan = FftPlan::new(16).expect("valid");
            let mut output = vec![Complex::ZERO; 16];
            plan.forward(&input, &mut output).expect("ok");

            let energy_time: f32 = input.iter().map(|x| x.norm_sq()).sum();
            let energy_freq: f32 = output.iter().map(|x| x.norm_sq()).sum::<f32>() / 16.0;

            let rel_err = if energy_time.abs() > 1e-10 {
                (energy_time - energy_freq).abs() / energy_time
            } else {
                (energy_time - energy_freq).abs()
            };
            prop_assert!(rel_err < 1e-3, "Parseval: time={energy_time}, freq={energy_freq}, rel_err={rel_err}");
        }

        #[test]
        fn prop_inverse_roundtrip(input in arb_complex_vec(16)) {
            let plan = FftPlan::new(16).expect("valid");
            let mut freq = vec![Complex::ZERO; 16];
            let mut recovered = vec![Complex::ZERO; 16];

            plan.forward(&input, &mut freq).expect("ok");
            plan.inverse(&freq, &mut recovered).expect("ok");

            for (orig, rec) in input.iter().zip(recovered.iter()) {
                let err = (*orig - *rec).abs();
                prop_assert!(err < 1e-3, "Roundtrip err={err}");
            }
        }
    }
}

// ============================================================================
// Bluestein (arbitrary-length FFT)
// ============================================================================

#[test]
fn test_bluestein_size_3() -> Result<(), Box<dyn std::error::Error>> {
    // DFT of [1, 1, 1] size 3: X[0]=3, X[1]=X[2]=0
    let input = [Complex::new(1.0, 0.0), Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
    let mut output = [Complex::ZERO; 3];
    bluestein_fft(&input, &mut output, false)?;
    assert!((output[0].re - 3.0).abs() < 1e-3);
    assert!(output[1].abs() < 1e-3);
    assert!(output[2].abs() < 1e-3);
    Ok(())
}

#[test]
fn test_bluestein_size_5_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let input = [
        Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(4.0, 0.0),
        Complex::new(5.0, 0.0),
    ];
    let mut freq = [Complex::ZERO; 5];
    let mut recovered = [Complex::ZERO; 5];

    bluestein_fft(&input, &mut freq, false)?;
    bluestein_fft(&freq, &mut recovered, true)?;

    for (orig, rec) in input.iter().zip(recovered.iter()) {
        let err = (*orig - *rec).abs();
        assert!(err < 1e-3, "Bluestein roundtrip error: {err}");
    }
    Ok(())
}

#[test]
fn test_bluestein_size_7_parseval() -> Result<(), Box<dyn std::error::Error>> {
    let input: Vec<Complex> = (0..7).map(|i| Complex::new(i as f32, 0.0)).collect();
    let mut output = vec![Complex::ZERO; 7];
    bluestein_fft(&input, &mut output, false)?;

    // Parseval: Σ|x|² = (1/N)·Σ|X|²
    let energy_time: f32 = input.iter().map(|x| x.norm_sq()).sum();
    let energy_freq: f32 = output.iter().map(|x| x.norm_sq()).sum::<f32>() / 7.0;
    assert!(
        (energy_time - energy_freq).abs() < 0.5,
        "Parseval violated: time={energy_time}, freq={energy_freq}"
    );
    Ok(())
}

#[test]
fn test_bluestein_size_6_impulse() -> Result<(), Box<dyn std::error::Error>> {
    // Impulse: [1, 0, 0, 0, 0, 0] -> all X[k] = 1
    let mut input = vec![Complex::ZERO; 6];
    input[0] = Complex::new(1.0, 0.0);
    let mut output = vec![Complex::ZERO; 6];
    bluestein_fft(&input, &mut output, false)?;

    for k in 0..6 {
        assert!((output[k].re - 1.0).abs() < 1e-3, "Impulse X[{k}].re = {}", output[k].re);
        assert!(output[k].im.abs() < 1e-3, "Impulse X[{k}].im = {}", output[k].im);
    }
    Ok(())
}

#[test]
fn test_bluestein_power_of_two_matches_stockham() -> Result<(), Box<dyn std::error::Error>> {
    // For power-of-two sizes, Bluestein should match Stockham
    let input = [
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(-1.0, 0.0),
        Complex::new(0.0, -1.0),
    ];
    let mut blue_out = [Complex::ZERO; 4];
    let mut stock_out = [Complex::ZERO; 4];

    bluestein_fft(&input, &mut blue_out, false)?;
    let plan = FftPlan::new(4)?;
    plan.forward(&input, &mut stock_out)?;

    for k in 0..4 {
        let diff = (blue_out[k] - stock_out[k]).abs();
        assert!(diff < 1e-3, "Bluestein vs Stockham diff at {k}: {diff}");
    }
    Ok(())
}

#[test]
fn test_bluestein_size_1() -> Result<(), Box<dyn std::error::Error>> {
    let input = [Complex::new(42.0, 7.0)];
    let mut output = [Complex::ZERO; 1];
    bluestein_fft(&input, &mut output, false)?;
    assert!((output[0].re - 42.0).abs() < 1e-6);
    assert!((output[0].im - 7.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_bluestein_empty() {
    let input: &[Complex] = &[];
    let output: &mut [Complex] = &mut [];
    assert!(bluestein_fft(input, output, false).is_err());
}

// ============================================================================
// 3D FFT tests (Contract: fft-3d-v1.yaml)
// ============================================================================

use crate::fft3d::{fft_3d, fft_batched, ifft_3d};

#[test]
fn test_fft_3d_impulse() -> Result<(), Box<dyn std::error::Error>> {
    let n = 2;
    let total = n * n * n;
    let mut input = vec![Complex::ZERO; total];
    input[0] = Complex::new(1.0, 0.0);
    let mut output = vec![Complex::ZERO; total];

    fft_3d(&input, &mut output, n, n, n)?;

    // 3D impulse → all ones
    for (k, x) in output.iter().enumerate() {
        assert!((x.re - 1.0).abs() < 1e-4 && x.im.abs() < 1e-4, "3D impulse wrong at k={k}: {x:?}");
    }
    Ok(())
}

#[test]
fn test_fft_3d_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let (nx, ny, nz) = (2, 4, 2);
    let total = nx * ny * nz;
    let input: Vec<Complex> =
        (0..total).map(|i| Complex::new((i as f32).sin(), (i as f32).cos())).collect();
    let mut freq = vec![Complex::ZERO; total];
    let mut recovered = vec![Complex::ZERO; total];

    fft_3d(&input, &mut freq, nx, ny, nz)?;
    ifft_3d(&freq, &mut recovered, nx, ny, nz)?;

    for (i, (orig, rec)) in input.iter().zip(recovered.iter()).enumerate() {
        let err = (*orig - *rec).abs();
        assert!(err < 1e-3, "3D roundtrip failed at {i}: err={err}");
    }
    Ok(())
}

#[test]
fn test_fft_3d_parseval() -> Result<(), Box<dyn std::error::Error>> {
    let (nx, ny, nz) = (2, 2, 4);
    let total = nx * ny * nz;
    let input: Vec<Complex> = (0..total).map(|i| Complex::new(i as f32, 0.0)).collect();
    let mut output = vec![Complex::ZERO; total];

    fft_3d(&input, &mut output, nx, ny, nz)?;

    let energy_time: f32 = input.iter().map(|x| x.norm_sq()).sum();
    let energy_freq: f32 = output.iter().map(|x| x.norm_sq()).sum::<f32>() / total as f32;
    assert!(
        (energy_time - energy_freq).abs() < 1.0,
        "3D Parseval violated: time={energy_time}, freq={energy_freq}"
    );
    Ok(())
}

#[test]
fn test_fft_3d_zero_dim() {
    let input = vec![Complex::ZERO; 0];
    let mut output = vec![];
    assert!(fft_3d(&input, &mut output, 0, 2, 2).is_err());
}

#[test]
fn test_fft_3d_size_mismatch() {
    let input = vec![Complex::ZERO; 8];
    let mut output = vec![Complex::ZERO; 16];
    assert!(fft_3d(&input, &mut output, 2, 2, 2).is_err());
}

// ============================================================================
// Batched FFT tests
// ============================================================================

#[test]
fn test_fft_batched_impulse() -> Result<(), Box<dyn std::error::Error>> {
    let n = 4;
    let batch = 3;
    let total = n * batch;
    let mut input = vec![Complex::ZERO; total];
    // Each batch has impulse at position 0
    for b in 0..batch {
        input[b * n] = Complex::new(1.0, 0.0);
    }
    let mut output = vec![Complex::ZERO; total];

    fft_batched(&input, &mut output, n, batch, false)?;

    // Each batch should produce all-ones
    for b in 0..batch {
        for k in 0..n {
            let x = output[b * n + k];
            assert!(
                (x.re - 1.0).abs() < 1e-5 && x.im.abs() < 1e-5,
                "Batched impulse wrong at batch={b}, k={k}: {x:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn test_fft_batched_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let n = 8;
    let batch = 2;
    let total = n * batch;
    let input: Vec<Complex> = (0..total).map(|i| Complex::new((i as f32).sin(), 0.0)).collect();
    let mut freq = vec![Complex::ZERO; total];
    let mut recovered = vec![Complex::ZERO; total];

    fft_batched(&input, &mut freq, n, batch, false)?;
    fft_batched(&freq, &mut recovered, n, batch, true)?;

    for (i, (orig, rec)) in input.iter().zip(recovered.iter()).enumerate() {
        let err = (*orig - *rec).abs();
        assert!(err < 1e-4, "Batched roundtrip err at {i}: {err}");
    }
    Ok(())
}

#[test]
fn test_fft_batched_size_mismatch() {
    let input = vec![Complex::ZERO; 10]; // not a multiple of n=4
    let mut output = vec![Complex::ZERO; 8];
    assert!(fft_batched(&input, &mut output, 4, 2, false).is_err());
}

// ============================================================================
// C2R (complex-to-real inverse) tests
// ============================================================================

#[test]
fn test_c2r_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let n = 8;
    let plan = FftPlan::new(n)?;
    let real_input: Vec<f32> = (0..n).map(|i| i as f32).collect();

    // R2C forward
    let mut freq = vec![Complex::ZERO; n / 2 + 1];
    plan.forward_r2c(&real_input, &mut freq)?;

    // C2R inverse
    let mut recovered = vec![0.0f32; n];
    plan.inverse_c2r(&freq, &mut recovered)?;

    for (i, (&orig, &rec)) in real_input.iter().zip(recovered.iter()).enumerate() {
        let err = (orig - rec).abs();
        assert!(err < 1e-4, "C2R roundtrip failed at {i}: orig={orig}, rec={rec}, err={err}");
    }
    Ok(())
}

#[test]
fn test_c2r_dc_signal() -> Result<(), Box<dyn std::error::Error>> {
    let n = 4;
    let plan = FftPlan::new(n)?;
    let real_input = vec![5.0f32; n];

    let mut freq = vec![Complex::ZERO; n / 2 + 1];
    plan.forward_r2c(&real_input, &mut freq)?;

    let mut recovered = vec![0.0f32; n];
    plan.inverse_c2r(&freq, &mut recovered)?;

    for (i, &v) in recovered.iter().enumerate() {
        assert!((v - 5.0).abs() < 1e-4, "C2R DC signal wrong at {i}: {v}");
    }
    Ok(())
}

#[test]
fn test_fft_r2c_free_function() -> Result<(), Box<dyn std::error::Error>> {
    let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let mut output = vec![Complex::ZERO; 5];
    fft_r2c(&input, &mut output)?;

    let expected_dc: f32 = input.iter().sum();
    assert!((output[0].re - expected_dc).abs() < 1e-4);
    Ok(())
}

#[test]
fn test_fft_c2r_free_function() -> Result<(), Box<dyn std::error::Error>> {
    let n = 8;
    let real_input: Vec<f32> = (0..n).map(|i| i as f32).collect();

    let mut freq = vec![Complex::ZERO; n / 2 + 1];
    fft_r2c(&real_input, &mut freq)?;

    let mut recovered = vec![0.0f32; n];
    fft_c2r(&freq, &mut recovered, n)?;

    for (i, (&orig, &rec)) in real_input.iter().zip(recovered.iter()).enumerate() {
        assert!((orig - rec).abs() < 1e-4, "C2R free fn failed at {i}");
    }
    Ok(())
}

#[test]
fn test_c2r_dimension_mismatch() {
    let plan = FftPlan::new(8).expect("valid plan");
    let freq = vec![Complex::ZERO; 3]; // Wrong: should be 5
    let mut output = vec![0.0f32; 8];
    assert!(plan.inverse_c2r(&freq, &mut output).is_err());
}

#[test]
fn test_c2r_output_dimension_mismatch() {
    let plan = FftPlan::new(8).expect("valid plan");
    let freq = vec![Complex::ZERO; 5];
    let mut output = vec![0.0f32; 4]; // Wrong: should be 8
    assert!(plan.inverse_c2r(&freq, &mut output).is_err());
}

// ── Fft trait tests ──────────────────────────────────────────────

use crate::Fft;

#[test]
fn test_fft_trait_roundtrip() {
    let plan = FftPlan::new(4).expect("valid plan");
    let fft: &dyn Fft = &plan;

    let input = [
        Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(4.0, 0.0),
    ];
    let mut freq = vec![Complex::ZERO; 4];
    let mut roundtrip = vec![Complex::ZERO; 4];

    fft.fft_1d(&input, &mut freq).expect("forward ok");
    fft.ifft_1d(&freq, &mut roundtrip).expect("inverse ok");

    for (i, (a, b)) in input.iter().zip(roundtrip.iter()).enumerate() {
        assert!((a.re - b.re).abs() < 1e-5, "roundtrip mismatch at {i}");
    }
}

#[test]
fn test_fft_trait_len() {
    let plan = FftPlan::new(16).expect("valid plan");
    let fft: &dyn Fft = &plan;
    assert_eq!(fft.len(), 16);
    assert!(!fft.is_empty());
}

#[test]
fn test_fft_trait_r2c_c2r() {
    let plan = FftPlan::new(8).expect("valid plan");
    let fft: &dyn Fft = &plan;

    let input = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0_f32];
    let mut freq = vec![Complex::ZERO; 5]; // N/2+1
    let mut output = vec![0.0_f32; 8];

    fft.fft_r2c(&input, &mut freq).expect("r2c ok");
    fft.fft_c2r(&freq, &mut output).expect("c2r ok");

    for (i, (&a, &b)) in input.iter().zip(output.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "r2c/c2r roundtrip mismatch at {i}: {a} vs {b}");
    }
}

// ── Fft trait fft_2d (§3C parity) ──────────────────────────────

#[test]
fn test_fft_trait_2d_impulse() {
    let plan = FftPlan::new(4).expect("valid plan");
    let fft: &dyn Fft = &plan;

    // 4×4 impulse → all ones in frequency domain
    let mut input = vec![Complex::ZERO; 16];
    input[0] = Complex::new(1.0, 0.0);
    let mut output = vec![Complex::ZERO; 16];

    fft.fft_2d(&input, &mut output, 4, 4).expect("2d ok");

    // All frequency bins should be 1.0 + 0i for an impulse at (0,0)
    for (i, &v) in output.iter().enumerate() {
        assert!(
            (v.re - 1.0).abs() < 1e-4 && v.im.abs() < 1e-4,
            "2D impulse mismatch at {i}: ({}, {})",
            v.re,
            v.im
        );
    }
}

#[test]
fn test_fft_trait_2d_parseval() {
    let plan = FftPlan::new(4).expect("valid plan");
    let fft: &dyn Fft = &plan;

    let input: Vec<Complex> =
        (0..16).map(|i| Complex::new((i as f32).sin(), (i as f32).cos())).collect();
    let mut output = vec![Complex::ZERO; 16];

    fft.fft_2d(&input, &mut output, 4, 4).expect("2d ok");

    let energy_time: f32 = input.iter().map(|c| c.norm_sq()).sum();
    let energy_freq: f32 = output.iter().map(|c| c.norm_sq()).sum::<f32>() / 16.0;

    assert!(
        (energy_time - energy_freq).abs() < 0.1,
        "2D Parseval: time={energy_time}, freq={energy_freq}"
    );
}

// ============================================================================
// POPPERIAN FALSIFICATION: Adversarial edge cases (GH #152)
// ============================================================================

#[test]
fn test_falsify_fft_all_zeros() {
    let plan = FftPlan::new(8).expect("valid plan");
    let input = vec![Complex::ZERO; 8];
    let mut output = vec![Complex::ZERO; 8];
    plan.forward(&input, &mut output).expect("fft ok");
    for (k, x) in output.iter().enumerate() {
        assert!(x.abs() < 1e-7, "FFT of all zeros should be all zeros, got {x:?} at k={k}");
    }
}

#[test]
fn test_falsify_fft_nyquist_only() {
    // Signal that alternates +1, -1, +1, -1 (Nyquist frequency)
    let plan = FftPlan::new(4).expect("valid plan");
    let input = [
        Complex::new(1.0, 0.0),
        Complex::new(-1.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(-1.0, 0.0),
    ];
    let mut output = [Complex::ZERO; 4];
    plan.forward(&input, &mut output).expect("fft ok");

    // All energy should be at Nyquist (k=N/2=2)
    assert!(output[0].abs() < 1e-5, "DC should be zero: {:?}", output[0]);
    assert!(output[2].abs() > 3.9, "Nyquist should have all energy: {:?}", output[2]);
}

#[test]
fn test_falsify_fft_large_power_of_two() {
    let n = 1024;
    let plan = FftPlan::new(n).expect("valid plan");
    let mut input = vec![Complex::ZERO; n];
    input[0] = Complex::new(1.0, 0.0);
    let mut output = vec![Complex::ZERO; n];
    plan.forward(&input, &mut output).expect("fft ok");

    // Impulse → all ones
    for (k, x) in output.iter().enumerate() {
        assert!(
            (x.re - 1.0).abs() < 1e-4 && x.im.abs() < 1e-4,
            "Large FFT impulse wrong at k={k}: {x:?}"
        );
    }
}

#[test]
fn test_falsify_fft_roundtrip_large() {
    let n = 256;
    let plan = FftPlan::new(n).expect("valid plan");
    let input: Vec<Complex> =
        (0..n).map(|i| Complex::new((i as f32 * 0.1).sin(), (i as f32 * 0.07).cos())).collect();
    let mut freq = vec![Complex::ZERO; n];
    let mut recovered = vec![Complex::ZERO; n];

    plan.forward(&input, &mut freq).expect("ok");
    plan.inverse(&freq, &mut recovered).expect("ok");

    let max_err =
        input.iter().zip(recovered.iter()).map(|(a, b)| (*a - *b).abs()).fold(0.0_f32, f32::max);
    assert!(max_err < 1e-3, "Large roundtrip max error: {max_err}");
}

#[test]
fn test_falsify_bluestein_large_prime() -> Result<(), Box<dyn std::error::Error>> {
    // Large prime: 127
    let n = 127;
    let mut input = vec![Complex::ZERO; n];
    input[0] = Complex::new(1.0, 0.0);
    let mut output = vec![Complex::ZERO; n];
    bluestein_fft(&input, &mut output, false)?;

    // Impulse → all ones
    for k in 0..n {
        assert!(
            (output[k].re - 1.0).abs() < 0.05 && output[k].im.abs() < 0.05,
            "Bluestein prime=127 impulse wrong at k={k}: {:?}",
            output[k]
        );
    }
    Ok(())
}

#[test]
fn test_falsify_bluestein_roundtrip_prime_23() -> Result<(), Box<dyn std::error::Error>> {
    let n = 23;
    let input: Vec<Complex> =
        (0..n).map(|i| Complex::new((i as f32).sin(), (i as f32).cos())).collect();
    let mut freq = vec![Complex::ZERO; n];
    let mut recovered = vec![Complex::ZERO; n];

    bluestein_fft(&input, &mut freq, false)?;
    bluestein_fft(&freq, &mut recovered, true)?;

    let max_err =
        input.iter().zip(recovered.iter()).map(|(a, b)| (*a - *b).abs()).fold(0.0_f32, f32::max);
    assert!(max_err < 0.05, "Bluestein prime=23 roundtrip max error: {max_err}");
    Ok(())
}

#[test]
fn test_falsify_fft_2d_constant() {
    // Constant 4×4 input → energy only at DC
    let val = Complex::new(3.0, 0.0);
    let input = vec![val; 16];
    let mut output = vec![Complex::ZERO; 16];
    fft_2d(&input, &mut output, 4, 4).expect("ok");

    // DC = N*M*val = 16*3 = 48
    assert!((output[0].re - 48.0).abs() < 1e-3, "2D constant DC: {:?}", output[0]);
    // All other bins should be ~0
    for k in 1..16 {
        assert!(output[k].abs() < 1e-3, "2D constant non-DC at k={k}: {:?}", output[k]);
    }
}

#[test]
fn test_falsify_r2c_c2r_roundtrip_large() -> Result<(), Box<dyn std::error::Error>> {
    let n = 128;
    let plan = FftPlan::new(n)?;
    let real_input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
    let mut freq = vec![Complex::ZERO; n / 2 + 1];
    let mut recovered = vec![0.0f32; n];

    plan.forward_r2c(&real_input, &mut freq)?;
    plan.inverse_c2r(&freq, &mut recovered)?;

    let max_err =
        real_input.iter().zip(recovered.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    assert!(max_err < 1e-3, "R2C/C2R large roundtrip max error: {max_err}");
    Ok(())
}
