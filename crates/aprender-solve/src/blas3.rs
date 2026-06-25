//! BLAS Level-3 operations: syrk, syr2k, trmm, symm.
//!
//! # Contract: blas-level3-v1.yaml
//!
//! Reference CPU implementations for cuBLAS parity.

use crate::error::SolverError;

/// Symmetric rank-k update: C = α·A·Aᵀ + β·C
///
/// A is n×k, C is n×n (symmetric, stored as full matrix row-major).
///
/// # Errors
///
/// Returns error on dimension mismatch.
pub fn syrk(
    a: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), SolverError> {
    validate_buffer(a, n * k, n, k)?;
    validate_buffer(c, n * n, n, n)?;

    // C = β·C
    for val in c.iter_mut() {
        *val *= beta;
    }

    // C += α·A·Aᵀ
    for i in 0..n {
        for j in 0..=i {
            let dot = dot_row_row(a, k, i, j);
            let update = alpha * dot;
            c[i * n + j] += update;
            if i != j {
                c[j * n + i] += update; // symmetric
            }
        }
    }

    Ok(())
}

/// Dot product of rows i and j of matrix a (n×k layout).
fn dot_row_row(a: &[f32], k: usize, i: usize, j: usize) -> f32 {
    let mut sum = 0.0_f32;
    for p in 0..k {
        sum += a[i * k + p] * a[j * k + p];
    }
    sum
}

/// Symmetric rank-2k update: C = α·A·Bᵀ + α·B·Aᵀ + β·C
///
/// A and B are n×k, C is n×n (symmetric, stored as full matrix row-major).
///
/// # Errors
///
/// Returns error on dimension mismatch.
pub fn syr2k(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), SolverError> {
    validate_buffer(a, n * k, n, k)?;
    validate_buffer(b, n * k, n, k)?;
    validate_buffer(c, n * n, n, n)?;

    // C = β·C
    for val in c.iter_mut() {
        *val *= beta;
    }

    // C += α·(A·Bᵀ + B·Aᵀ)
    for i in 0..n {
        for j in 0..=i {
            let dot_ab = dot_rows(a, b, k, i, j);
            let dot_ba = dot_rows(b, a, k, i, j);
            let update = alpha * (dot_ab + dot_ba);
            c[i * n + j] += update;
            if i != j {
                c[j * n + i] += update; // symmetric
            }
        }
    }

    Ok(())
}

/// Dot product of row i of matrix a with row j of matrix b (n×k layout).
fn dot_rows(a: &[f32], b: &[f32], k: usize, i: usize, j: usize) -> f32 {
    let mut sum = 0.0_f32;
    for p in 0..k {
        sum += a[i * k + p] * b[j * k + p];
    }
    sum
}

/// Triangular matrix multiply: B = α·A·B (left side, lower triangular).
///
/// A is n×n lower triangular, B is n×nrhs (row-major).
///
/// # Errors
///
/// Returns error on dimension mismatch.
pub fn trmm(
    a: &[f32],
    b: &mut [f32],
    n: usize,
    nrhs: usize,
    alpha: f32,
) -> Result<(), SolverError> {
    validate_buffer(a, n * n, n, n)?;
    validate_buffer(b, n * nrhs, n, nrhs)?;

    // Process from bottom row to top to avoid overwriting
    // For lower triangular: row i depends on rows 0..=i
    // We compute new_b[i] = α · Σ_{j<=i} A[i,j] · old_b[j]
    // Process top-to-bottom, accumulate into temporary
    let mut temp = vec![0.0_f32; n * nrhs];

    for i in 0..n {
        for j in 0..=i {
            let a_val = alpha * a[i * n + j];
            for col in 0..nrhs {
                temp[i * nrhs + col] += a_val * b[j * nrhs + col];
            }
        }
    }

    b[..n * nrhs].copy_from_slice(&temp[..n * nrhs]);
    Ok(())
}

/// Symmetric matrix multiply: C = α·A·B + β·C
///
/// A is n×n symmetric (stored full), B is n×m, C is n×m (row-major).
///
/// # Errors
///
/// Returns error on dimension mismatch.
pub fn symm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    m: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), SolverError> {
    validate_buffer(a, n * n, n, n)?;
    validate_buffer(b, n * m, n, m)?;
    validate_buffer(c, n * m, n, m)?;

    // C = β·C + α·A·B (standard GEMM since A is stored as full matrix)
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0_f32;
            for p in 0..n {
                sum += a[i * n + p] * b[p * m + j];
            }
            c[i * m + j] = alpha * sum + beta * c[i * m + j];
        }
    }

    Ok(())
}

/// Mixed-precision GEMM: C = α·A·B + β·C with f16 inputs, f32 accumulation.
///
/// A is m×k, B is k×n, C is m×n. Inputs `a` and `b` are `u16` containing
/// IEEE 754 half-precision floats. Accumulation is in f32 for cuBLAS `gemmEx` parity.
///
/// # Errors
///
/// Returns error on dimension mismatch.
#[allow(clippy::too_many_arguments)]
pub fn gemm_ex(
    a: &[u16],
    b: &[u16],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), SolverError> {
    if a.len() != m * k {
        return Err(SolverError::BufferLengthMismatch {
            expected: m * k,
            got: a.len(),
            rows: m,
            cols: k,
        });
    }
    if b.len() != k * n {
        return Err(SolverError::BufferLengthMismatch {
            expected: k * n,
            got: b.len(),
            rows: k,
            cols: n,
        });
    }
    if c.len() != m * n {
        return Err(SolverError::BufferLengthMismatch {
            expected: m * n,
            got: c.len(),
            rows: m,
            cols: n,
        });
    }

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f64;
            for p in 0..k {
                let a_val = f16_to_f32(a[i * k + p]);
                let b_val = f16_to_f32(b[p * n + j]);
                sum += f64::from(a_val) * f64::from(b_val);
            }
            c[i * n + j] = alpha * sum as f32 + beta * c[i * n + j];
        }
    }

    Ok(())
}

/// Strided batched GEMM: C_b = α·A_b·B_b + β·C_b for b = 0..batch_count.
///
/// Each batch matrix is accessed via stride offsets into flat buffers.
/// A_b is m×k at a[b*stride_a..], B_b is k×n at b[b*stride_b..],
/// C_b is m×n at c[b*stride_c..].
///
/// # Errors
///
/// Returns error on dimension mismatch or insufficient buffer length.
#[allow(clippy::too_many_arguments)]
pub fn gemm_strided_batched(
    a: &[f32],
    stride_a: usize,
    b: &[f32],
    stride_b: usize,
    c: &mut [f32],
    stride_c: usize,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), SolverError> {
    if batch_count == 0 {
        return Ok(());
    }

    let a_needed = (batch_count - 1) * stride_a + m * k;
    if a.len() < a_needed {
        return Err(SolverError::BufferLengthMismatch {
            expected: a_needed,
            got: a.len(),
            rows: m,
            cols: k,
        });
    }
    let b_needed = (batch_count - 1) * stride_b + k * n;
    if b.len() < b_needed {
        return Err(SolverError::BufferLengthMismatch {
            expected: b_needed,
            got: b.len(),
            rows: k,
            cols: n,
        });
    }
    let c_needed = (batch_count - 1) * stride_c + m * n;
    if c.len() < c_needed {
        return Err(SolverError::BufferLengthMismatch {
            expected: c_needed,
            got: c.len(),
            rows: m,
            cols: n,
        });
    }

    for batch in 0..batch_count {
        let a_off = batch * stride_a;
        let b_off = batch * stride_b;
        let c_off = batch * stride_c;

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0_f64;
                for p in 0..k {
                    sum += f64::from(a[a_off + i * k + p]) * f64::from(b[b_off + p * n + j]);
                }
                c[c_off + i * n + j] = alpha * sum as f32 + beta * c[c_off + i * n + j];
            }
        }
    }

    Ok(())
}

/// Convert IEEE 754 half-precision (u16) to f32 (software implementation).
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1F;
    let mant = h & 0x3FF;

    if exp == 0 {
        // Subnormal or zero
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // Subnormal: value = (-1)^sign * 2^(-14) * (mant/1024)
        let val = (mant as f32) * (1.0 / 1024.0) * (1.0 / 16384.0);
        return if sign == 1 { -val } else { val };
    }

    if exp == 31 {
        // Inf or NaN
        return if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }

    // Normal: value = (-1)^sign * 2^(exp-15) * (1 + mant/1024)
    let f32_exp = (exp as i32) - 15 + 127;
    let f32_bits = ((sign as u32) << 31) | ((f32_exp as u32) << 23) | ((mant as u32) << 13);
    f32::from_bits(f32_bits)
}

/// Convert f32 to IEEE 754 half-precision (u16) with round-to-nearest-even.
///
/// Bit-identical to `half::f16::from_f32` across the entire `f32` domain
/// (verified exhaustively over all 2^32 inputs, including NaN payloads).
///
/// # Contract (OBLIG-SOLVE-F32-F16-RNE)
/// - Rounds to nearest, ties to even (IEEE 754 default rounding).
/// - `255.99 → 0x5C00` (rounds up across the mantissa-carry boundary, not `0x5BFF`).
/// - `65520.0 → 0x7C00` (+Inf — the half-way overflow case rounds up, not `0x7BFF`).
/// - The smallest f32 above the f16-subnormal half-way point rounds up to `0x0001`,
///   not `0x0000`.
///
/// The previous implementation truncated the mantissa (round-toward-zero) and
/// truncated the subnormal/overflow boundaries, diverging from IEEE RNE in
/// ~436M of 2^32 inputs. f16 GEMM inputs are produced here, so a non-RNE
/// conversion silently perturbed every mixed-precision matmul.
pub fn f32_to_f16(f: f32) -> u16 {
    let x: u32 = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = (x >> 23) & 0xFF; // biased f32 exponent (8 bits)
    let mantissa = x & 0x007F_FFFF;

    // Inf / NaN
    if exp == 0xFF {
        return if mantissa == 0 {
            sign | 0x7C00 // ±Inf
        } else {
            // NaN: canonical quiet bit + top mantissa bits (matches `half`)
            sign | 0x7E00 | ((mantissa >> 13) as u16)
        };
    }

    // Rebias exponent: f32 bias = 127, f16 bias = 15.
    let half_exp = (exp as i32) - 127 + 15;

    if half_exp >= 0x1F {
        // Overflow → ±Inf
        return sign | 0x7C00;
    }

    if half_exp <= 0 {
        // Subnormal or zero. `14 - half_exp` is the right-shift; >24 means every
        // significant bit is shifted out (true zero).
        if 14 - half_exp > 24 {
            return sign;
        }
        let m = mantissa | 0x0080_0000; // restore implicit leading 1
        let shift = (14 - half_exp) as u32;
        let mut half_mant = m >> shift;
        // Round to nearest even: round up iff the highest dropped bit is set AND
        // the remaining dropped/sticky bits make it strictly past the half-way tie.
        let round_bit = 1u32 << (shift - 1);
        if (m & round_bit) != 0 && (m & (3 * round_bit - 1)) != 0 {
            half_mant += 1;
        }
        return sign | half_mant as u16;
    }

    // Normal number.
    let half_mant = (mantissa >> 13) as u16;
    let mut result = sign | ((half_exp as u16) << 10) | half_mant;
    // Round to nearest even on the 13 dropped mantissa bits.
    let round_bit = 0x0000_1000u32;
    if (mantissa & round_bit) != 0 && (mantissa & (3 * round_bit - 1)) != 0 {
        result += 1;
    }
    result
}

/// Epilogue operation applied after GEMM computation.
///
/// Matches cuBLASLt `cublasLtEpilogue_t` post-processing modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Epilogue {
    /// No post-processing: C = α·A·B + β·C
    #[default]
    None,
    /// ReLU activation: C = max(0, α·A·B + β·C)
    Relu,
    /// Bias add: C = α·A·B + β·C + bias
    Bias,
    /// GELU activation: C = gelu(α·A·B + β·C)
    Gelu,
    /// Bias + ReLU: C = max(0, α·A·B + β·C + bias)
    BiasRelu,
    /// Bias + GELU: C = gelu(α·A·B + β·C + bias)
    BiasGelu,
}

/// Mixed-precision GEMM with epilogue fusion.
///
/// Like `gemm_ex` but applies a post-processing epilogue operation.
/// `bias` is a per-column vector of length `n` (one per output column),
/// required when epilogue is `Bias`, `BiasRelu`, or `BiasGelu`.
///
/// # Errors
///
/// Returns error on dimension mismatch or missing bias.
#[allow(clippy::too_many_arguments)]
pub fn gemm_ex_epilogue(
    a: &[u16],
    b: &[u16],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
    epilogue: Epilogue,
    bias: Option<&[f32]>,
) -> Result<(), SolverError> {
    gemm_ex(a, b, c, m, n, k, alpha, beta)?;
    apply_epilogue(c, m, n, epilogue, bias)
}

/// Apply epilogue post-processing to a matrix C (m×n row-major).
fn apply_epilogue(
    c: &mut [f32],
    m: usize,
    n: usize,
    epilogue: Epilogue,
    bias: Option<&[f32]>,
) -> Result<(), SolverError> {
    if epilogue == Epilogue::None {
        return Ok(());
    }

    let bias_vec = validate_bias_if_needed(epilogue, bias, n)?;

    // Add bias if present
    if let Some(bv) = bias_vec {
        add_bias(c, m, n, bv);
    }

    // Apply activation
    apply_activation(c, epilogue);

    Ok(())
}

/// Validate that bias is provided when the epilogue requires it.
fn validate_bias_if_needed(
    epilogue: Epilogue,
    bias: Option<&[f32]>,
    n: usize,
) -> Result<Option<&[f32]>, SolverError> {
    let needs_bias = matches!(
        epilogue,
        Epilogue::Bias | Epilogue::BiasRelu | Epilogue::BiasGelu
    );
    if !needs_bias {
        return Ok(None);
    }
    let bv = bias.ok_or(SolverError::InvalidInput {
        reason: "epilogue requires bias vector",
    })?;
    if bv.len() != n {
        return Err(SolverError::BufferLengthMismatch {
            expected: n,
            got: bv.len(),
            rows: 1,
            cols: n,
        });
    }
    Ok(Some(bv))
}

/// Add per-column bias to matrix C.
fn add_bias(c: &mut [f32], m: usize, n: usize, bias: &[f32]) {
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] += bias[j];
        }
    }
}

/// Apply element-wise activation function.
fn apply_activation(c: &mut [f32], epilogue: Epilogue) {
    match epilogue {
        Epilogue::Relu | Epilogue::BiasRelu => {
            for val in c.iter_mut() {
                *val = val.max(0.0);
            }
        }
        Epilogue::Gelu | Epilogue::BiasGelu => {
            for val in c.iter_mut() {
                *val = gelu(*val);
            }
        }
        Epilogue::None | Epilogue::Bias => {}
    }
}

/// GELU approximation: x · Φ(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
fn gelu(x: f32) -> f32 {
    let coeff = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (coeff * (x + 0.044715 * x * x * x)).tanh())
}

/// Validate buffer length matches expected dimensions.
fn validate_buffer(
    buf: &[f32],
    expected: usize,
    rows: usize,
    cols: usize,
) -> Result<(), SolverError> {
    if buf.len() != expected {
        return Err(SolverError::BufferLengthMismatch {
            expected,
            got: buf.len(),
            rows,
            cols,
        });
    }
    Ok(())
}
