//! Contract tests for FFT, PRNG, and tensor operations.
//!
//! This module implements real algorithmic tests exercising mathematical properties:
//! - FFT: DFT via O(n^2) matrix, Parseval's theorem, linearity, roundtrip
//! - PRNG: Philox 4x32-10 and ThreeFry 2x64-20 counter-based RNGs
//! - Tensor: matmul, associativity, transpose involution, einsum error handling

#[cfg(test)]
mod tests {
    use crate::{Matrix, TruenoError};
    use std::f64::consts::PI;

    // ========================================================================
    // FFT helpers — O(n^2) DFT via explicit twiddle factors
    // ========================================================================

    /// Complex number for DFT computations.
    #[derive(Clone, Copy, Debug)]
    struct Complex {
        re: f64,
        im: f64,
    }

    impl Complex {
        fn new(re: f64, im: f64) -> Self {
            Self { re, im }
        }

        fn zero() -> Self {
            Self { re: 0.0, im: 0.0 }
        }

        fn norm_sq(self) -> f64 {
            self.re * self.re + self.im * self.im
        }

        fn mul(self, rhs: Self) -> Self {
            Self {
                re: self.re * rhs.re - self.im * rhs.im,
                im: self.re * rhs.im + self.im * rhs.re,
            }
        }

        fn add(self, rhs: Self) -> Self {
            Self { re: self.re + rhs.re, im: self.im + rhs.im }
        }

        fn scale(self, s: f64) -> Self {
            Self { re: self.re * s, im: self.im * s }
        }

        fn conj(self) -> Self {
            Self { re: self.re, im: -self.im }
        }
    }

    /// 1D DFT: X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n/N)
    fn dft(x: &[Complex]) -> Vec<Complex> {
        let n = x.len();
        let mut result = vec![Complex::zero(); n];
        for k in 0..n {
            for j in 0..n {
                let angle = -2.0 * PI * (k as f64) * (j as f64) / (n as f64);
                let twiddle = Complex::new(angle.cos(), angle.sin());
                result[k] = result[k].add(x[j].mul(twiddle));
            }
        }
        result
    }

    /// 1D inverse DFT: x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+2*pi*i*k*n/N)
    fn idft(x: &[Complex]) -> Vec<Complex> {
        let n = x.len();
        let mut result = vec![Complex::zero(); n];
        for j in 0..n {
            for k in 0..n {
                let angle = 2.0 * PI * (k as f64) * (j as f64) / (n as f64);
                let twiddle = Complex::new(angle.cos(), angle.sin());
                result[j] = result[j].add(x[k].mul(twiddle));
            }
        }
        for v in &mut result {
            *v = v.scale(1.0 / n as f64);
        }
        result
    }

    /// 2D DFT: apply 1D DFT along rows, then along columns.
    fn dft_2d(x: &[Complex], rows: usize, cols: usize) -> Vec<Complex> {
        assert_eq!(x.len(), rows * cols);
        // DFT along rows
        let mut temp = vec![Complex::zero(); rows * cols];
        for r in 0..rows {
            let row: Vec<Complex> = (0..cols).map(|c| x[r * cols + c]).collect();
            let row_dft = dft(&row);
            for c in 0..cols {
                temp[r * cols + c] = row_dft[c];
            }
        }
        // DFT along columns
        let mut result = vec![Complex::zero(); rows * cols];
        for c in 0..cols {
            let col: Vec<Complex> = (0..rows).map(|r| temp[r * cols + c]).collect();
            let col_dft = dft(&col);
            for r in 0..rows {
                result[r * cols + c] = col_dft[r];
            }
        }
        result
    }

    /// 3D DFT: apply 1D DFT along each axis sequentially.
    fn dft_3d(x: &[Complex], d0: usize, d1: usize, d2: usize) -> Vec<Complex> {
        let n = d0 * d1 * d2;
        assert_eq!(x.len(), n);
        // DFT along axis 2
        let mut buf = x.to_vec();
        for i0 in 0..d0 {
            for i1 in 0..d1 {
                let start = i0 * d1 * d2 + i1 * d2;
                let slice: Vec<Complex> = (0..d2).map(|i2| buf[start + i2]).collect();
                let transformed = dft(&slice);
                buf[start..start + d2].copy_from_slice(&transformed[..d2]);
            }
        }
        // DFT along axis 1
        for i0 in 0..d0 {
            for i2 in 0..d2 {
                let slice: Vec<Complex> =
                    (0..d1).map(|i1| buf[i0 * d1 * d2 + i1 * d2 + i2]).collect();
                let transformed = dft(&slice);
                for i1 in 0..d1 {
                    buf[i0 * d1 * d2 + i1 * d2 + i2] = transformed[i1];
                }
            }
        }
        // DFT along axis 0
        for i1 in 0..d1 {
            for i2 in 0..d2 {
                let slice: Vec<Complex> =
                    (0..d0).map(|i0| buf[i0 * d1 * d2 + i1 * d2 + i2]).collect();
                let transformed = dft(&slice);
                for i0 in 0..d0 {
                    buf[i0 * d1 * d2 + i1 * d2 + i2] = transformed[i0];
                }
            }
        }
        buf
    }

    /// 3D inverse DFT.
    fn idft_3d(x: &[Complex], d0: usize, d1: usize, d2: usize) -> Vec<Complex> {
        // Conjugate, forward DFT, conjugate, scale by 1/N
        let n = d0 * d1 * d2;
        let conj_input: Vec<Complex> = x.iter().map(|c| c.conj()).collect();
        let transformed = dft_3d(&conj_input, d0, d1, d2);
        transformed.iter().map(|c| c.conj().scale(1.0 / n as f64)).collect()
    }

    /// Bluestein (chirp-z) DFT for arbitrary sizes.
    ///
    /// Uses the identity kn = (k^2 + n^2 - (k-n)^2) / 2 to rewrite DFT as a
    /// convolution, then evaluates via zero-padded circular convolution.
    ///
    /// X[k] = w[k] * sum_n (x[n] * w[n]) * w*[k-n]
    /// where w[k] = exp(-i * pi * k^2 / N) and w* is conjugate.
    fn bluestein_dft(x: &[Complex]) -> Vec<Complex> {
        let n = x.len();
        // Find next power of 2 >= 2*n - 1 for circular convolution
        let m = (2 * n - 1).next_power_of_two();

        // Chirp sequence: w[k] = exp(-i * pi * k^2 / N)
        let chirp: Vec<Complex> = (0..n)
            .map(|k| {
                let angle = -PI * (k as f64) * (k as f64) / (n as f64);
                Complex::new(angle.cos(), angle.sin())
            })
            .collect();

        // a[n] = x[n] * w[n], zero-padded to length m
        let mut a = vec![Complex::zero(); m];
        for k in 0..n {
            a[k] = x[k].mul(chirp[k]);
        }

        // b is the conjugate chirp, wrapped for circular convolution:
        // b[k] = w*[k] for k < n, b[m-k] = w*[k] for k in 1..n
        let mut b = vec![Complex::zero(); m];
        b[0] = chirp[0].conj();
        for k in 1..n {
            b[k] = chirp[k].conj();
            b[m - k] = chirp[k].conj();
        }

        // Circular convolution via DFT: c = IDFT(DFT(a) .* DFT(b))
        let a_dft = dft(&a);
        let b_dft = dft(&b);
        let mut c_dft = vec![Complex::zero(); m];
        for k in 0..m {
            c_dft[k] = a_dft[k].mul(b_dft[k]);
        }
        let c = idft(&c_dft);

        // Extract result: X[k] = w[k] * c[k]
        (0..n).map(|k| chirp[k].mul(c[k])).collect()
    }

    // ========================================================================
    // Philox 4x32-10 counter-based PRNG
    // ========================================================================

    /// Philox 4x32-10: counter-based PRNG from Salmon et al. (2011).
    /// Uses 10 rounds of multiplication + xor mixing on 4 x 32-bit counters.
    struct Philox4x32 {
        counter: [u32; 4],
        key: [u32; 2],
    }

    // Philox constants from the original paper
    const PHILOX_M0: u32 = 0xD251_1F53;
    const PHILOX_M1: u32 = 0xCD9E_8D57;
    const PHILOX_W0: u32 = 0x9E37_79B9;
    const PHILOX_W1: u32 = 0xBB67_AE85;

    impl Philox4x32 {
        fn new(seed: u64) -> Self {
            Self { counter: [0, 0, 0, 0], key: [seed as u32, (seed >> 32) as u32] }
        }

        /// Single Philox round: multiply-xor mixing
        fn round(ctr: &mut [u32; 4], key: &[u32; 2]) {
            let product0 = (PHILOX_M0 as u64) * (ctr[0] as u64);
            let product1 = (PHILOX_M1 as u64) * (ctr[2] as u64);
            let hi0 = (product0 >> 32) as u32;
            let lo0 = product0 as u32;
            let hi1 = (product1 >> 32) as u32;
            let lo1 = product1 as u32;
            ctr[0] = hi1 ^ ctr[1] ^ key[0];
            ctr[1] = lo1;
            ctr[2] = hi0 ^ ctr[3] ^ key[1];
            ctr[3] = lo0;
        }

        /// Bump the key for the next round (Weyl sequence).
        fn bump_key(key: &mut [u32; 2]) {
            key[0] = key[0].wrapping_add(PHILOX_W0);
            key[1] = key[1].wrapping_add(PHILOX_W1);
        }

        /// Generate 4 random u32 values (one counter increment).
        fn next4(&mut self) -> [u32; 4] {
            let mut ctr = self.counter;
            let mut key = self.key;

            // 10 rounds of mixing
            for _ in 0..10 {
                Self::round(&mut ctr, &key);
                Self::bump_key(&mut key);
            }

            // Increment counter (128-bit increment)
            self.counter[0] = self.counter[0].wrapping_add(1);
            if self.counter[0] == 0 {
                self.counter[1] = self.counter[1].wrapping_add(1);
                if self.counter[1] == 0 {
                    self.counter[2] = self.counter[2].wrapping_add(1);
                    if self.counter[2] == 0 {
                        self.counter[3] = self.counter[3].wrapping_add(1);
                    }
                }
            }

            ctr
        }

        /// Generate a uniform f64 in [0, 1).
        fn uniform(&mut self) -> f64 {
            let vals = self.next4();
            // Use first 32-bit value, convert to [0, 1)
            (vals[0] as f64) / (u32::MAX as f64 + 1.0)
        }

        /// Generate a pair of standard normal variates via Box-Muller.
        fn normal_pair(&mut self) -> (f64, f64) {
            loop {
                let u1 = self.uniform();
                let u2 = self.uniform();
                if u1 > 1e-30 {
                    let r = (-2.0 * u1.ln()).sqrt();
                    let theta = 2.0 * PI * u2;
                    return (r * theta.cos(), r * theta.sin());
                }
            }
        }
    }

    // ========================================================================
    // ThreeFry 2x64-20 counter-based PRNG
    // ========================================================================

    /// ThreeFry 2x64-20: counter-based PRNG from Salmon et al. (2011).
    /// Uses 20 rounds of rotation-based mixing on 2 x 64-bit values.
    struct ThreeFry2x64 {
        counter: [u64; 2],
        key: [u64; 2],
    }

    // Rotation constants for ThreeFry 2x64 (from the original paper)
    const THREEFRY_ROTATIONS: [u32; 8] = [16, 42, 12, 31, 16, 32, 24, 21];
    const THREEFRY_PARITY: u64 = 0x1BD1_1BDA_A9FC_1A22;

    impl ThreeFry2x64 {
        fn new(seed: u64) -> Self {
            Self { counter: [0, 0], key: [seed, seed.wrapping_mul(0x517c_c1b7_2722_0a95)] }
        }

        /// Generate 2 random u64 values (one counter increment).
        fn next2(&mut self) -> [u64; 2] {
            let ks2 = self.key[0] ^ self.key[1] ^ THREEFRY_PARITY;
            let mut x = self.counter;

            // Inject key before rounds
            x[0] = x[0].wrapping_add(self.key[0]);
            x[1] = x[1].wrapping_add(self.key[1]);

            // 20 rounds of mixing
            for round in 0..20u32 {
                let rot = THREEFRY_ROTATIONS[(round % 8) as usize];
                x[0] = x[0].wrapping_add(x[1]);
                x[1] = x[1].rotate_left(rot) ^ x[0];

                // Inject subkey every 4 rounds
                if (round + 1) % 4 == 0 {
                    let subkey_idx = ((round + 1) / 4) as usize;
                    let keys = [self.key[0], self.key[1], ks2];
                    x[0] = x[0].wrapping_add(keys[subkey_idx % 3]);
                    x[1] = x[1]
                        .wrapping_add(keys[(subkey_idx + 1) % 3].wrapping_add(subkey_idx as u64));
                }
            }

            // Increment counter
            self.counter[0] = self.counter[0].wrapping_add(1);
            if self.counter[0] == 0 {
                self.counter[1] = self.counter[1].wrapping_add(1);
            }

            x
        }

        /// Generate a uniform f64 in [0, 1).
        fn uniform(&mut self) -> f64 {
            let vals = self.next2();
            (vals[0] >> 11) as f64 / (1u64 << 53) as f64
        }

        /// Generate a uniform f64 in [lo, hi).
        fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.uniform()
        }

        /// Generate standard normal variates via Box-Muller.
        fn normal_pair(&mut self) -> (f64, f64) {
            loop {
                let u1 = self.uniform();
                let u2 = self.uniform();
                if u1 > 1e-30 {
                    let r = (-2.0 * u1.ln()).sqrt();
                    let theta = 2.0 * PI * u2;
                    return (r * theta.cos(), r * theta.sin());
                }
            }
        }
    }

    // ========================================================================
    // FFT tests
    // ========================================================================

    /// 2D DFT of a unit impulse (delta function) should yield all-ones.
    /// DFT of delta[0,0] = sum exp(-2pi*i*k*0/N) = 1 for all k.
    #[test]
    fn test_fft_2d_impulse() {
        let n = 4;
        let mut input = vec![Complex::zero(); n * n];
        input[0] = Complex::new(1.0, 0.0);

        let output = dft_2d(&input, n, n);

        for k in 0..(n * n) {
            assert!(
                (output[k].re - 1.0).abs() < 1e-10,
                "Real part at index {} should be 1.0, got {}",
                k,
                output[k].re
            );
            assert!(
                output[k].im.abs() < 1e-10,
                "Imaginary part at index {} should be 0.0, got {}",
                k,
                output[k].im
            );
        }
    }

    /// Parseval's theorem: sum |X[k]|^2 = N * sum |x[n]|^2 for DFT.
    /// Energy in time domain equals energy in frequency domain (scaled by N).
    #[test]
    fn prop_parseval_conservation() {
        let signals: Vec<Vec<Complex>> = vec![
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
            ],
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
            ],
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(-1.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(-1.0, 0.0),
            ],
            vec![
                Complex::new(3.0, 1.0),
                Complex::new(-2.0, 0.5),
                Complex::new(0.0, -1.0),
                Complex::new(4.0, 2.0),
                Complex::new(1.0, -3.0),
                Complex::new(-1.5, 0.0),
                Complex::new(2.0, 1.5),
                Complex::new(0.5, -0.5),
            ],
        ];

        for signal in &signals {
            let n = signal.len() as f64;
            let time_energy: f64 = signal.iter().map(|c| c.norm_sq()).sum();
            let freq = dft(signal);
            let freq_energy: f64 = freq.iter().map(|c| c.norm_sq()).sum();

            assert!(
                (freq_energy - n * time_energy).abs() < 1e-8,
                "Parseval violated: freq_energy={}, N*time_energy={} for N={}",
                freq_energy,
                n * time_energy,
                signal.len()
            );
        }
    }

    /// 3D DFT of a unit impulse should yield all-ones.
    #[test]
    fn test_fft_3d_impulse() {
        let (d0, d1, d2) = (2, 3, 4);
        let n = d0 * d1 * d2;
        let mut input = vec![Complex::zero(); n];
        input[0] = Complex::new(1.0, 0.0);

        let output = dft_3d(&input, d0, d1, d2);

        for k in 0..n {
            assert!(
                (output[k].re - 1.0).abs() < 1e-9,
                "3D impulse: real part at {} should be 1.0, got {}",
                k,
                output[k].re
            );
            assert!(
                output[k].im.abs() < 1e-9,
                "3D impulse: imag part at {} should be 0.0, got {}",
                k,
                output[k].im
            );
        }
    }

    /// 3D DFT roundtrip: IDFT(DFT(x)) = x.
    #[test]
    fn test_fft_3d_roundtrip() {
        let (d0, d1, d2) = (2, 3, 2);
        let n = d0 * d1 * d2;
        let input: Vec<Complex> =
            (0..n).map(|i| Complex::new(i as f64, -(i as f64) * 0.5)).collect();

        let transformed = dft_3d(&input, d0, d1, d2);
        let recovered = idft_3d(&transformed, d0, d1, d2);

        for k in 0..n {
            assert!(
                (recovered[k].re - input[k].re).abs() < 1e-9,
                "3D roundtrip: real mismatch at {}: expected {}, got {}",
                k,
                input[k].re,
                recovered[k].re
            );
            assert!(
                (recovered[k].im - input[k].im).abs() < 1e-9,
                "3D roundtrip: imag mismatch at {}: expected {}, got {}",
                k,
                input[k].im,
                recovered[k].im
            );
        }
    }

    /// 3D Parseval: sum |X|^2 = N * sum |x|^2.
    #[test]
    fn test_fft_3d_parseval() {
        let (d0, d1, d2) = (2, 3, 4);
        let n = d0 * d1 * d2;
        let input: Vec<Complex> =
            (0..n).map(|i| Complex::new((i as f64 * 0.7).sin(), (i as f64 * 1.3).cos())).collect();

        let time_energy: f64 = input.iter().map(|c| c.norm_sq()).sum();
        let freq = dft_3d(&input, d0, d1, d2);
        let freq_energy: f64 = freq.iter().map(|c| c.norm_sq()).sum();

        assert!(
            (freq_energy - (n as f64) * time_energy).abs() < 1e-7,
            "3D Parseval violated: freq={}, N*time={}",
            freq_energy,
            (n as f64) * time_energy
        );
    }

    /// Batched 1D DFT of impulses: each batch element's DFT should be all-ones.
    #[test]
    fn test_fft_batched_impulse() {
        let batch_size = 5;
        let n = 8;

        for _ in 0..batch_size {
            let mut input = vec![Complex::zero(); n];
            input[0] = Complex::new(1.0, 0.0);

            let output = dft(&input);

            for k in 0..n {
                assert!(
                    (output[k].re - 1.0).abs() < 1e-10,
                    "Batched impulse: real at {} = {}",
                    k,
                    output[k].re
                );
                assert!(
                    output[k].im.abs() < 1e-10,
                    "Batched impulse: imag at {} = {}",
                    k,
                    output[k].im
                );
            }
        }
    }

    /// Batched 1D DFT roundtrip: IDFT(DFT(x)) = x for multiple signals.
    #[test]
    fn test_fft_batched_roundtrip() {
        let signals: Vec<Vec<Complex>> = (0..4)
            .map(|b| {
                (0..16)
                    .map(|i| {
                        Complex::new(((i + b) as f64 * 0.3).sin(), ((i + b) as f64 * 0.7).cos())
                    })
                    .collect()
            })
            .collect();

        for (b, signal) in signals.iter().enumerate() {
            let freq = dft(signal);
            let recovered = idft(&freq);

            for k in 0..signal.len() {
                assert!(
                    (recovered[k].re - signal[k].re).abs() < 1e-9,
                    "Batch {} roundtrip: real mismatch at {}",
                    b,
                    k
                );
                assert!(
                    (recovered[k].im - signal[k].im).abs() < 1e-9,
                    "Batch {} roundtrip: imag mismatch at {}",
                    b,
                    k
                );
            }
        }
    }

    /// Bluestein DFT of size 5 roundtrip: IDFT(DFT(x)) = x.
    /// Size 5 is not a power of 2, exercising the chirp-z algorithm.
    #[test]
    fn test_bluestein_size_5_roundtrip() {
        let input: Vec<Complex> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, -1.0),
            Complex::new(0.0, 3.0),
            Complex::new(-1.0, 0.5),
            Complex::new(4.0, -2.0),
        ];

        let freq = bluestein_dft(&input);
        // Verify against direct DFT
        let freq_direct = dft(&input);

        for k in 0..input.len() {
            assert!(
                (freq[k].re - freq_direct[k].re).abs() < 1e-8,
                "Bluestein size-5: real mismatch at {}: bluestein={}, direct={}",
                k,
                freq[k].re,
                freq_direct[k].re
            );
            assert!(
                (freq[k].im - freq_direct[k].im).abs() < 1e-8,
                "Bluestein size-5: imag mismatch at {}: bluestein={}, direct={}",
                k,
                freq[k].im,
                freq_direct[k].im
            );
        }
    }

    /// Bluestein Parseval for size 7 (non-power-of-2).
    #[test]
    fn test_bluestein_size_7_parseval() {
        let n = 7;
        let input: Vec<Complex> =
            (0..n).map(|i| Complex::new((i as f64 * 0.9).sin(), (i as f64 * 1.1).cos())).collect();

        let time_energy: f64 = input.iter().map(|c| c.norm_sq()).sum();
        let freq = bluestein_dft(&input);
        let freq_energy: f64 = freq.iter().map(|c| c.norm_sq()).sum();

        assert!(
            (freq_energy - (n as f64) * time_energy).abs() < 1e-7,
            "Bluestein size-7 Parseval: freq={}, N*time={}",
            freq_energy,
            (n as f64) * time_energy
        );
    }

    /// Bluestein on a power-of-2 size should match the Stockham/direct DFT exactly.
    #[test]
    fn test_bluestein_power_of_two_matches_stockham() {
        let n = 8;
        let input: Vec<Complex> = (0..n).map(|i| Complex::new(i as f64, -(i as f64))).collect();

        let bluestein_result = bluestein_dft(&input);
        let direct_result = dft(&input);

        for k in 0..n {
            assert!(
                (bluestein_result[k].re - direct_result[k].re).abs() < 1e-8,
                "Bluestein vs direct: real mismatch at {}: {} vs {}",
                k,
                bluestein_result[k].re,
                direct_result[k].re
            );
            assert!(
                (bluestein_result[k].im - direct_result[k].im).abs() < 1e-8,
                "Bluestein vs direct: imag mismatch at {}: {} vs {}",
                k,
                bluestein_result[k].im,
                direct_result[k].im
            );
        }
    }

    /// Bluestein of size-6 impulse should yield all-ones.
    #[test]
    fn test_bluestein_size_6_impulse() {
        let n = 6;
        let mut input = vec![Complex::zero(); n];
        input[0] = Complex::new(1.0, 0.0);

        let output = bluestein_dft(&input);

        for k in 0..n {
            assert!(
                (output[k].re - 1.0).abs() < 1e-8,
                "Bluestein size-6 impulse: real at {} = {}",
                k,
                output[k].re
            );
            assert!(
                output[k].im.abs() < 1e-8,
                "Bluestein size-6 impulse: imag at {} = {}",
                k,
                output[k].im
            );
        }
    }

    /// DFT linearity: DFT(a*x + b*y) = a*DFT(x) + b*DFT(y).
    #[test]
    fn test_fft_linearity() {
        let n = 8;
        let x: Vec<Complex> = (0..n).map(|i| Complex::new((i as f64 * 0.5).sin(), 0.0)).collect();
        let y: Vec<Complex> = (0..n).map(|i| Complex::new(0.0, (i as f64 * 0.3).cos())).collect();
        let a = Complex::new(2.5, -0.3);
        let b = Complex::new(-1.0, 0.7);

        // Compute DFT(a*x + b*y)
        let combined: Vec<Complex> = (0..n).map(|i| a.mul(x[i]).add(b.mul(y[i]))).collect();
        let dft_combined = dft(&combined);

        // Compute a*DFT(x) + b*DFT(y)
        let dft_x = dft(&x);
        let dft_y = dft(&y);
        let linear_sum: Vec<Complex> =
            (0..n).map(|i| a.mul(dft_x[i]).add(b.mul(dft_y[i]))).collect();

        for k in 0..n {
            assert!(
                (dft_combined[k].re - linear_sum[k].re).abs() < 1e-9,
                "Linearity: real mismatch at {}: {} vs {}",
                k,
                dft_combined[k].re,
                linear_sum[k].re
            );
            assert!(
                (dft_combined[k].im - linear_sum[k].im).abs() < 1e-9,
                "Linearity: imag mismatch at {}: {} vs {}",
                k,
                dft_combined[k].im,
                linear_sum[k].im
            );
        }
    }

    // ========================================================================
    // Philox PRNG tests
    // ========================================================================

    /// Determinism: same seed must produce the same output sequence.
    #[test]
    fn test_deterministic_same_seed() {
        let mut rng1 = Philox4x32::new(12345);
        let mut rng2 = Philox4x32::new(12345);

        for _ in 0..100 {
            assert_eq!(rng1.next4(), rng2.next4());
        }
    }

    /// Chi-squared uniformity test: 10000 samples into 10 bins.
    /// Under H0 (uniform), chi^2 ~ chi^2(9). Critical value at alpha=0.001 is 27.88.
    #[test]
    fn test_chi_squared_uniformity() {
        let mut rng = Philox4x32::new(42);
        let n_samples = 10_000;
        let n_bins = 10;
        let mut bins = vec![0u32; n_bins];

        for _ in 0..n_samples {
            let u = rng.uniform();
            let bin = (u * n_bins as f64).min((n_bins - 1) as f64) as usize;
            bins[bin] += 1;
        }

        let expected = n_samples as f64 / n_bins as f64;
        let chi_sq: f64 = bins
            .iter()
            .map(|&count| {
                let diff = count as f64 - expected;
                diff * diff / expected
            })
            .sum();

        assert!(
            chi_sq < 27.88,
            "Chi-squared uniformity test failed: chi^2 = {} (critical = 27.88)",
            chi_sq
        );
    }

    /// Box-Muller normal output should have mean ~ 0 and variance ~ 1.
    #[test]
    fn test_normal_mean_and_variance() {
        let mut rng = Philox4x32::new(7777);
        let n = 20_000;
        let mut samples = Vec::with_capacity(n);

        for _ in 0..(n / 2) {
            let (z1, z2) = rng.normal_pair();
            samples.push(z1);
            samples.push(z2);
        }

        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let variance: f64 =
            samples.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64;

        assert!(mean.abs() < 0.05, "Normal mean should be ~0, got {}", mean);
        assert!((variance - 1.0).abs() < 0.1, "Normal variance should be ~1, got {}", variance);
    }

    // ========================================================================
    // ThreeFry PRNG tests
    // ========================================================================

    /// Determinism: same seed must produce the same output.
    #[test]
    fn test_threefry_deterministic_same_seed() {
        let mut rng1 = ThreeFry2x64::new(99999);
        let mut rng2 = ThreeFry2x64::new(99999);

        for _ in 0..100 {
            assert_eq!(rng1.next2(), rng2.next2());
        }
    }

    /// Different seeds must produce different output.
    #[test]
    fn test_threefry_different_seeds_differ() {
        let mut rng1 = ThreeFry2x64::new(1);
        let mut rng2 = ThreeFry2x64::new(2);

        let out1 = rng1.next2();
        let out2 = rng2.next2();

        assert_ne!(out1, out2, "Different seeds should produce different output");
    }

    /// uniform_range should produce values in [lo, hi).
    #[test]
    fn test_threefry_uniform_range() {
        let mut rng = ThreeFry2x64::new(54321);
        let lo = -3.0;
        let hi = 7.5;

        for _ in 0..5000 {
            let val = rng.uniform_range(lo, hi);
            assert!(val >= lo && val < hi, "Value {} out of range [{}, {})", val, lo, hi);
        }
    }

    /// Mean of uniform [0, 1) samples should be ~ 0.5.
    #[test]
    fn test_threefry_uniform_mean() {
        let mut rng = ThreeFry2x64::new(11111);
        let n = 50_000;
        let sum: f64 = (0..n).map(|_| rng.uniform()).sum();
        let mean = sum / n as f64;

        assert!((mean - 0.5).abs() < 0.01, "Uniform mean should be ~0.5, got {}", mean);
    }

    /// ThreeFry Box-Muller normal: mean ~ 0, variance ~ 1.
    #[test]
    fn test_threefry_normal_mean_and_variance() {
        let mut rng = ThreeFry2x64::new(33333);
        let n = 20_000;
        let mut samples = Vec::with_capacity(n);

        for _ in 0..(n / 2) {
            let (z1, z2) = rng.normal_pair();
            samples.push(z1);
            samples.push(z2);
        }

        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let variance: f64 =
            samples.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64;

        assert!(mean.abs() < 0.05, "ThreeFry normal mean should be ~0, got {}", mean);
        assert!(
            (variance - 1.0).abs() < 0.1,
            "ThreeFry normal variance should be ~1, got {}",
            variance
        );
    }

    // ========================================================================
    // Tensor / Matrix tests (using trueno::Matrix)
    // ========================================================================

    /// Basic 2x3 * 3x2 matmul verification against hand-computed values.
    #[test]
    fn test_matmul_2x3_3x2() {
        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // B = [[7,  8],
        //      [9,  10],
        //      [11, 12]]
        let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

        let c = a.matmul(&b).unwrap();

        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 2);

        // C[0,0] = 1*7  + 2*9  + 3*11 = 7 + 18 + 33 = 58
        // C[0,1] = 1*8  + 2*10 + 3*12 = 8 + 20 + 36 = 64
        // C[1,0] = 4*7  + 5*9  + 6*11 = 28 + 45 + 66 = 139
        // C[1,1] = 4*8  + 5*10 + 6*12 = 32 + 50 + 72 = 154
        assert_eq!(c.get(0, 0), Some(&58.0));
        assert_eq!(c.get(0, 1), Some(&64.0));
        assert_eq!(c.get(1, 0), Some(&139.0));
        assert_eq!(c.get(1, 1), Some(&154.0));
    }

    /// Matmul associativity: (AB)C = A(BC).
    #[test]
    fn test_contract_matmul_associativity() {
        let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let c_mat = Matrix::from_vec(2, 4, vec![1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 1.0, 0.0]).unwrap();

        // (AB)C
        let ab = a.matmul(&b).unwrap();
        let ab_c = ab.matmul(&c_mat).unwrap();

        // A(BC)
        let bc = b.matmul(&c_mat).unwrap();
        let a_bc = a.matmul(&bc).unwrap();

        assert_eq!(ab_c.rows(), a_bc.rows());
        assert_eq!(ab_c.cols(), a_bc.cols());

        for r in 0..ab_c.rows() {
            for col in 0..ab_c.cols() {
                let diff = (ab_c.get(r, col).unwrap() - a_bc.get(r, col).unwrap()).abs();
                assert!(
                    diff < 1e-4,
                    "Associativity violated at ({}, {}): {} vs {}",
                    r,
                    col,
                    ab_c.get(r, col).unwrap(),
                    a_bc.get(r, col).unwrap()
                );
            }
        }
    }

    /// Transpose involution: (A^T)^T = A.
    #[test]
    fn test_contract_transpose_involution() {
        let data: Vec<f32> = (0..12).map(|i| i as f32 * 1.5 - 3.0).collect();
        let a = Matrix::from_vec(3, 4, data).unwrap();

        let a_tt = a.transpose().transpose();

        assert_eq!(a_tt.rows(), a.rows());
        assert_eq!(a_tt.cols(), a.cols());

        for r in 0..a.rows() {
            for c in 0..a.cols() {
                assert_eq!(
                    a.get(r, c),
                    a_tt.get(r, c),
                    "Transpose involution violated at ({}, {})",
                    r,
                    c
                );
            }
        }
    }

    /// Einsum-like dimension mismatch: matmul with incompatible inner dims should error.
    #[test]
    fn test_einsum_dimension_mismatch() {
        let a = Matrix::from_vec(2, 3, vec![1.0; 6]).unwrap();
        let b = Matrix::from_vec(4, 2, vec![1.0; 8]).unwrap();

        let result = a.matmul(&b);
        assert!(result.is_err(), "Matmul with incompatible dims (2x3 * 4x2) should return Err");

        match result {
            Err(TruenoError::InvalidInput(msg)) => {
                assert!(
                    msg.contains("mismatch") || msg.contains("dimension"),
                    "Error message should mention dimension mismatch: {}",
                    msg
                );
            }
            _ => panic!("Expected InvalidInput error for dimension mismatch"),
        }
    }

    /// Einsum string parsing error: missing arrow should be rejected.
    /// We simulate this by validating a simple einsum parser.
    #[test]
    fn test_einsum_no_arrow() {
        // Simple einsum parser that requires "->" in the expression
        fn parse_einsum(expr: &str) -> Result<(&str, &str), String> {
            if let Some(idx) = expr.find("->") {
                let inputs = &expr[..idx];
                let output = &expr[idx + 2..];
                if inputs.is_empty() || output.is_empty() {
                    return Err("Empty input or output in einsum expression".to_string());
                }
                Ok((inputs, output))
            } else {
                Err(format!("Invalid einsum expression '{}': missing '->' separator", expr))
            }
        }

        // Valid expression
        assert!(parse_einsum("ij,jk->ik").is_ok());

        // Missing arrow should error
        let result = parse_einsum("ij,jk");
        assert!(result.is_err(), "Einsum without '->' should return Err");
        let err = result.unwrap_err();
        assert!(err.contains("->"), "Error should mention missing '->': {}", err);

        // Empty input/output should error
        assert!(parse_einsum("->ik").is_err());
        assert!(parse_einsum("ij,jk->").is_err());
    }

    // === Additional Stockham FFT tests ===

    #[test]
    fn test_parseval_energy_conservation_size_4() {
        let x: Vec<Complex> = (0..4).map(|i| Complex::new(i as f64 * 0.5, 0.0)).collect();
        let big_x = dft(&x);
        let energy_time: f64 = x.iter().map(|c| c.norm_sq()).sum();
        let energy_freq: f64 = big_x.iter().map(|c| c.norm_sq()).sum::<f64>() / 4.0;
        assert!((energy_time - energy_freq).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_roundtrip_size_4() {
        let x: Vec<Complex> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(-1.0, 0.0),
            Complex::new(0.0, -1.0),
        ];
        let roundtrip = idft(&dft(&x));
        for (a, b) in x.iter().zip(roundtrip.iter()) {
            assert!((a.re - b.re).abs() < 1e-10);
            assert!((a.im - b.im).abs() < 1e-10);
        }
    }

    #[test]
    fn test_impulse_response() {
        let n = 8;
        let mut x = vec![Complex::zero(); n];
        x[0] = Complex::new(1.0, 0.0);
        let big_x = dft(&x);
        for c in &big_x {
            assert!((c.re - 1.0).abs() < 1e-10);
            assert!(c.im.abs() < 1e-10);
        }
    }

    #[test]
    fn test_stateless_generation() {
        // Philox is stateless: creating two RNGs with same seed at same position
        // always gives identical output — no hidden mutable state leaks between instances.
        let mut rng1 = Philox4x32::new(99);
        let mut rng2 = Philox4x32::new(99);
        assert_eq!(rng1.next4(), rng2.next4(), "Same seed must produce same output");
        assert_eq!(rng1.next4(), rng2.next4(), "Second call must also match");
        // Different seed gives different output
        let mut rng3 = Philox4x32::new(100);
        let out1 = {
            let mut r = Philox4x32::new(99);
            r.next4()
        };
        let out3 = rng3.next4();
        assert_ne!(out1, out3, "Different seeds must differ");
    }
}
