
/// Continued fraction for incomplete beta (Lentz's algorithm).
fn beta_continued_fraction(a: f32, b: f32, x: f32) -> f32 {
    let max_iter = 100;
    let eps = 1e-7;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iter {
        let m_f = m as f32;
        let m2 = 2.0 * m_f;

        // Even step
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }

    h
}

/// Log-gamma function `ln Γ(z)` (Lanczos approximation).
///
/// PMAT-904: the raw-space `gamma()` below evaluates `(...).exp()` directly,
/// which overflows f32 (Γ(36) ≈ 1e40, Inf at z ≳ 36). p-value paths for
/// degrees-of-freedom df ≳ 72 must combine gamma terms in LOG space (divide
/// gammas as a subtraction of `ln_gamma`, then a single final `exp`) so no
/// intermediate ever overflows. `ln Γ` grows only ~ z ln z, never overflowing
/// f32 for any df reachable here.
fn ln_gamma(z: f32) -> f32 {
    if z < 0.5 {
        // Reflection: ln Γ(z) = ln(π / sin(πz)) − ln Γ(1−z).
        let sin_pz = (PI * z).sin();
        (PI / sin_pz).ln() - ln_gamma(1.0 - z)
    } else {
        // Lanczos series, evaluated entirely in log space.
        let z = z - 1.0;
        let tmp = z + 5.5;
        let tmp = (z + 0.5) * tmp.ln() - tmp;
        let ser = 1.0 + 76.180_09_f32 / (z + 1.0) - 86.505_32_f32 / (z + 2.0)
            + 24.014_1_f32 / (z + 3.0)
            - 1.231_739_5_f32 / (z + 4.0)
            + 0.001_208_58_f32 / (z + 5.0)
            - 0.000_005_363_82_f32 / (z + 6.0);
        tmp + (ser * (2.0 * PI).sqrt()).ln()
    }
}

/// Gamma function approximation (Stirling's approximation).
///
/// NOTE: this raw-space form overflows f32 for z ≳ 36. Production p-value code
/// uses `ln_gamma` and combines in log space (PMAT-904); this raw form is kept
/// only for the small-argument unit tests.
#[cfg(test)]
fn gamma(z: f32) -> f32 {
    if z < 0.5 {
        // Reflection formula: Γ(z) = π / (sin(πz) * Γ(1-z))
        PI / ((PI * z).sin() * gamma(1.0 - z))
    } else {
        // Stirling's approximation
        let z = z - 1.0;
        let tmp = z + 5.5;
        let tmp = (z + 0.5) * tmp.ln() - tmp;
        let ser = 1.0 + 76.180_09_f32 / (z + 1.0) - 86.505_32_f32 / (z + 2.0)
            + 24.014_1_f32 / (z + 3.0)
            - 1.231_739_5_f32 / (z + 4.0)
            + 0.001_208_58_f32 / (z + 5.0)
            - 0.000_005_363_82_f32 / (z + 6.0);
        (tmp + ser.ln()).exp() * (2.0 * PI).sqrt()
    }
}
