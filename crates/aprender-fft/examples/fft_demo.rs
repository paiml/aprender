//! FFT demonstration: Stockham, Bluestein, 3D, batched.
//!
//! ```sh
//! cargo run --example fft_demo -p trueno-fft
//! ```

use trueno_fft::{bluestein_fft, fft_3d, fft_batched, fft_c2r, fft_r2c, ifft_3d, Complex, FftPlan};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== trueno-fft: Full FFT Demo ===\n");

    // ── 1D Stockham FFT ────────────────────────────────────
    let plan = FftPlan::new(8)?;
    let input: Vec<Complex> = (0..8).map(|i| Complex::new(i as f32, 0.0)).collect();
    let mut output = vec![Complex::ZERO; 8];
    plan.forward(&input, &mut output)?;
    println!("1D FFT of [0..8]:");
    for (k, x) in output.iter().enumerate() {
        println!("  X[{k}] = {:.3} + {:.3}i", x.re, x.im);
    }

    // Parseval check
    let e_time: f32 = input.iter().map(|x| x.norm_sq()).sum();
    let e_freq: f32 = output.iter().map(|x| x.norm_sq()).sum::<f32>() / 8.0;
    println!("Parseval: time={e_time:.4}, freq={e_freq:.4}, err={:.2e}", (e_time - e_freq).abs());

    // Roundtrip
    let mut recovered = vec![Complex::ZERO; 8];
    plan.inverse(&output, &mut recovered)?;
    let max_err: f32 =
        input.iter().zip(recovered.iter()).map(|(a, b)| (*a - *b).abs()).fold(0.0f32, f32::max);
    println!("Roundtrip max error: {max_err:.2e}");

    // ── Bluestein (arbitrary length) ───────────────────────
    println!("\n--- Bluestein FFT (non-power-of-two) ---");
    let input5: Vec<Complex> = (0..5).map(|i| Complex::new(i as f32, 0.0)).collect();
    let mut output5 = vec![Complex::ZERO; 5];
    bluestein_fft(&input5, &mut output5, false)?;
    println!("Bluestein FFT of [0..5] (size 5):");
    for (k, x) in output5.iter().enumerate() {
        println!("  X[{k}] = {:.3} + {:.3}i", x.re, x.im);
    }

    // Bluestein roundtrip
    let mut rec5 = vec![Complex::ZERO; 5];
    bluestein_fft(&output5, &mut rec5, true)?;
    let err5: f32 =
        input5.iter().zip(rec5.iter()).map(|(a, b)| (*a - *b).abs()).fold(0.0f32, f32::max);
    println!("Bluestein roundtrip error: {err5:.2e}");

    // ── 3D FFT ─────────────────────────────────────────────
    println!("\n--- 3D FFT ---");
    let (nx, ny, nz) = (2, 4, 2);
    let total = nx * ny * nz;
    let input3d: Vec<Complex> =
        (0..total).map(|i| Complex::new((i as f32).sin(), (i as f32).cos())).collect();
    let mut freq3d = vec![Complex::ZERO; total];
    let mut rec3d = vec![Complex::ZERO; total];
    fft_3d(&input3d, &mut freq3d, nx, ny, nz)?;
    ifft_3d(&freq3d, &mut rec3d, nx, ny, nz)?;
    let err3d: f32 =
        input3d.iter().zip(rec3d.iter()).map(|(a, b)| (*a - *b).abs()).fold(0.0f32, f32::max);
    println!("3D FFT {nx}×{ny}×{nz} roundtrip error: {err3d:.2e}");

    // ── Batched FFT ────────────────────────────────────────
    println!("\n--- Batched FFT ---");
    let n = 4;
    let batch = 3;
    let batch_input: Vec<Complex> = (0..n * batch).map(|i| Complex::new(i as f32, 0.0)).collect();
    let mut batch_freq = vec![Complex::ZERO; n * batch];
    let mut batch_rec = vec![Complex::ZERO; n * batch];
    fft_batched(&batch_input, &mut batch_freq, n, batch, false)?;
    fft_batched(&batch_freq, &mut batch_rec, n, batch, true)?;
    let err_batch: f32 = batch_input
        .iter()
        .zip(batch_rec.iter())
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0f32, f32::max);
    println!("{batch} batches of size {n}: roundtrip error = {err_batch:.2e}");

    // ── R2C + C2R ─────────────────────────────────────────
    println!("\n--- R2C + C2R ---");
    let real_input: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let mut r2c_out = vec![Complex::ZERO; 5]; // N/2+1
    fft_r2c(&real_input, &mut r2c_out)?;
    println!("R2C DC component: {:.3} (expected: 28.0)", r2c_out[0].re);

    // C2R roundtrip
    let mut c2r_out = vec![0.0f32; 8];
    fft_c2r(&r2c_out, &mut c2r_out, 8)?;
    let c2r_err: f32 =
        real_input.iter().zip(c2r_out.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("R2C → C2R roundtrip error: {c2r_err:.2e}");

    println!("\n=== All FFT demos passed ===");
    Ok(())
}
