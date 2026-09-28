//! Error models (KTEST-001 §3.1): the bound Bᵢ on |ŷᵢ − yᵢ| for one output element.
//!
//! Only the models with a proven bound are implemented. Every other declared id is refused with
//! [`Refusal::NotImplemented`], because a tolerance without an error-model citation is STOP S-1:
//! the harness never falls back to a guessed or measured tolerance.

/// A floating-point type a kernel accumulates in or stores to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F64,
    F32,
    /// TF32 tensor-core mode: f32 range, 10 explicit significand bits (p = 11). Carried as its own
    /// `dtype_acc`, never folded into F32 (K1b).
    Tf32,
    F16,
    Bf16,
}

impl Dtype {
    /// Significand precision p, including the implicit bit.
    #[must_use]
    pub const fn precision(self) -> i32 {
        match self {
            Self::F64 => 53,
            Self::F32 => 24,
            Self::Tf32 | Self::F16 => 11,
            Self::Bf16 => 8,
        }
    }

    /// The unit roundoff u = 2^−p (round to nearest).
    #[must_use]
    pub fn unit_roundoff(self) -> f64 {
        2f64.powi(-self.precision())
    }

    /// The smallest positive normal number: the most a flush-to-zero can lose per operation.
    #[must_use]
    pub fn min_normal(self) -> f64 {
        match self {
            Self::F64 => f64::MIN_POSITIVE,
            Self::F32 | Self::Tf32 | Self::Bf16 => 2f64.powi(-126),
            Self::F16 => 2f64.powi(-14),
        }
    }

    /// γ_n = n·u / (1 − n·u), or `None` when n·u ≥ 1 and the bound is vacuous (Theorem K1b).
    #[must_use]
    pub fn gamma(self, n: usize) -> Option<f64> {
        #[allow(clippy::cast_precision_loss)] // a reduction length, far below 2^53
        let nu = n as f64 * self.unit_roundoff();
        (nu < 1.0).then(|| nu / (1.0 - nu))
    }
}

/// The error model a kernel declares in the registry (`error_model` field).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorModel {
    /// `EM-DOT`: dot product, GEMV, GEMM in any summation order.
    /// |ŷᵢ − yᵢ| ≤ γ_K(acc)·(|A|·|B|)ᵢ + η_FTZ (Higham Thm 3.1; order-free by Theorem K1a).
    Dot {
        /// Products summed per output element.
        k: usize,
        /// The type the kernel accumulates in.
        acc: Dtype,
        /// The type the output is stored in. A coarser output adds one rounding, u_out·|y|.
        out: Dtype,
        /// The kernel may flush subnormals to zero (WGSL, CUDA/Metal fast-math): adds K·min_normal.
        ftz: bool,
    },
    /// `EM-RED` for a sum of n terms: `EM-DOT` with K = n and |B| = 1.
    Sum {
        n: usize,
        acc: Dtype,
        out: Dtype,
        ftz: bool,
    },
    /// `EM-RED` for max: exact, so Bᵢ = 0.
    Max,
    /// `EM-DEQ`: bit-exact (0 ULP) against pinned llama.cpp fixtures, so Bᵢ = 0.
    Dequant,
    /// `EM-ELEM`: needs the backend's published builtin ULP table (§3.1); not implemented yet.
    Elementwise,
    /// `EM-SMX`: composed from EM-ELEM(exp) and EM-RED; not implemented yet.
    Softmax,
    /// `EM-ATT`: composed from EM-DOT, EM-SMX, EM-DOT; not implemented yet.
    Attention,
    /// `EM-ROPE`: EM-ELEM(sin, cos) + 2 ULP; not implemented yet.
    Rope,
}

/// Why the harness refuses to produce a bound. A refusal is a RED cell, never a skip.
#[derive(Debug, Clone, PartialEq)]
pub enum Refusal {
    /// K·u ≥ 1: the accumulator gives no guarantee at this length (K1b). The kernel must
    /// accumulate in a wider type; registering it is refused.
    Vacuous {
        id: &'static str,
        n: usize,
        acc: Dtype,
    },
    /// The declared model has no implemented bound (STOP S-1: never guess a tolerance).
    NotImplemented { id: &'static str },
    /// Output, oracle and bound buffers of different lengths: a harness bug.
    LengthMismatch {
        yhat: usize,
        oracle: usize,
        bound: usize,
    },
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vacuous { id, n, acc } => write!(
                f,
                "{id}: n = {n} with a {acc:?} accumulator gives n·u ≥ 1, so no bound holds (K1b)"
            ),
            Self::NotImplemented { id } => {
                write!(
                    f,
                    "{id}: no implemented error bound; refusing to guess one (S-1)"
                )
            }
            Self::LengthMismatch {
                yhat,
                oracle,
                bound,
            } => write!(
                f,
                "length mismatch: {yhat} outputs, {oracle} oracle values, {bound} bounds"
            ),
        }
    }
}

impl std::error::Error for Refusal {}

/// The per-element bound of one error model: Bᵢ = `scale`·magnitudeᵢ + `absolute`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bound {
    /// Multiplies the element's magnitude (Σ|a||b| for a dot, Σ|x| for a sum).
    pub scale: f64,
    /// Added once per element (the flush-to-zero term η_FTZ).
    pub absolute: f64,
}

impl Bound {
    /// Bᵢ for an element of the given magnitude.
    #[must_use]
    pub fn at(self, magnitude: f64) -> f64 {
        self.scale * magnitude + self.absolute
    }

    /// Bᵢ for every element.
    #[must_use]
    pub fn over(self, magnitudes: &[f64]) -> Vec<f64> {
        magnitudes.iter().map(|&m| self.at(m)).collect()
    }
}

/// The oracle is f64, not exact: its own error, γ_n(f64)·magnitude, is added to every bound so a
/// correct kernel is never judged RED by the oracle's rounding (it is ~1e-16 relative per term).
fn oracle_slack(n: usize) -> f64 {
    Dtype::F64.gamma(n.max(1)).unwrap_or(f64::INFINITY)
}

fn dot_like(
    id: &'static str,
    n: usize,
    acc: Dtype,
    out: Dtype,
    ftz: bool,
) -> Result<Bound, Refusal> {
    let gamma = acc.gamma(n).ok_or(Refusal::Vacuous { id, n, acc })?;
    // A store to a coarser type rounds once more; |y| ≤ magnitude, so u_out·magnitude covers it.
    let store = if out.precision() < acc.precision() {
        out.unit_roundoff()
    } else {
        0.0
    };
    #[allow(clippy::cast_precision_loss)] // a reduction length, far below 2^53
    let absolute = if ftz {
        n as f64 * acc.min_normal()
    } else {
        0.0
    };
    Ok(Bound {
        scale: gamma + store + oracle_slack(n),
        absolute,
    })
}

impl ErrorModel {
    /// The registry id, as KTEST-001 §3.1 spells it.
    #[must_use]
    pub const fn id(&self) -> &'static str {
        match self {
            Self::Dot { .. } => "EM-DOT",
            Self::Sum { .. } | Self::Max => "EM-RED",
            Self::Dequant => "EM-DEQ",
            Self::Elementwise => "EM-ELEM",
            Self::Softmax => "EM-SMX",
            Self::Attention => "EM-ATT",
            Self::Rope => "EM-ROPE",
        }
    }

    /// The per-element bound, or why none can be given.
    ///
    /// # Errors
    /// [`Refusal::Vacuous`] when the accumulator is too narrow for the length (K1b), and
    /// [`Refusal::NotImplemented`] for a model whose bound is not implemented yet (S-1).
    pub fn bound(&self) -> Result<Bound, Refusal> {
        let id = self.id();
        match *self {
            Self::Dot { k, acc, out, ftz } => dot_like(id, k, acc, out, ftz),
            Self::Sum { n, acc, out, ftz } => dot_like(id, n, acc, out, ftz),
            Self::Max | Self::Dequant => Ok(Bound {
                scale: 0.0,
                absolute: 0.0,
            }),
            Self::Elementwise | Self::Softmax | Self::Attention | Self::Rope => {
                Err(Refusal::NotImplemented { id })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors the Lean lemmas `f16_acc_k4096_vacuous`, `bf16_acc_k256_vacuous` and
    /// `f32_needed_for_4096` in KernelTesting.lean (Theorem K1b).
    #[test]
    fn k1b_vacuity_matches_the_lean_lemmas() {
        assert_eq!(Dtype::F16.gamma(4096), None);
        assert_eq!(Dtype::F16.gamma(2048), None, "K·u = 1 is already vacuous");
        assert!(Dtype::F16.gamma(2047).is_some());
        assert_eq!(Dtype::Bf16.gamma(256), None);
        assert!(Dtype::Bf16.gamma(255).is_some());
        assert_eq!(Dtype::Tf32.gamma(4096), None, "TF32 is not f32 for K1b");
        let g = Dtype::F32
            .gamma(1 << 20)
            .expect("f32 is meaningful up to 2^20");
        assert!((g - 1.0 / 15.0).abs() < 1e-15, "γ_(2^20) = 1/15, got {g}");
        assert!(Dtype::F32.gamma(4096).is_some());
    }

    #[test]
    fn f16_ftz_drift_quarter() {
        // [Lean] f16_ftz_drift_quarter: η_FTZ at K = 4096 with f16 min_normal = 4096·2^-14 = 0.25.
        let eta = 4096.0 * Dtype::F16.min_normal();
        assert!((eta - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn a_vacuous_accumulator_is_refused_not_loosened() {
        let m = ErrorModel::Dot {
            k: 4096,
            acc: Dtype::F16,
            out: Dtype::F16,
            ftz: false,
        };
        assert_eq!(
            m.bound(),
            Err(Refusal::Vacuous {
                id: "EM-DOT",
                n: 4096,
                acc: Dtype::F16
            })
        );
    }

    #[test]
    fn unimplemented_models_refuse_s1() {
        for m in [
            ErrorModel::Elementwise,
            ErrorModel::Softmax,
            ErrorModel::Attention,
            ErrorModel::Rope,
        ] {
            assert!(
                matches!(m.bound(), Err(Refusal::NotImplemented { id }) if id == m.id()),
                "{} must refuse, not guess",
                m.id()
            );
        }
    }

    #[test]
    fn dot_bound_terms() {
        let f32_dot = |out, ftz| {
            ErrorModel::Dot {
                k: 1024,
                acc: Dtype::F32,
                out,
                ftz,
            }
            .bound()
            .expect("f32 at K = 1024 has a bound")
        };
        let g = Dtype::F32.gamma(1024).expect("finite");
        let plain = f32_dot(Dtype::F32, false);
        assert!(plain.scale >= g && plain.scale < g * 1.000_001);
        assert!(plain.absolute.abs() < f64::MIN_POSITIVE);
        let to_f16 = f32_dot(Dtype::F16, false);
        assert!((to_f16.scale - plain.scale - 2f64.powi(-11)).abs() < 1e-18);
        let ftz = f32_dot(Dtype::F32, true);
        assert!((ftz.absolute - 1024.0 * 2f64.powi(-126)).abs() < 1e-45);
        assert!((plain.at(2.0) - 2.0 * plain.scale).abs() < f64::MIN_POSITIVE);
        let exact = ErrorModel::Dequant.bound().expect("bit-exact bound");
        assert!(exact.at(123.0).abs() < f64::MIN_POSITIVE);
    }
}
