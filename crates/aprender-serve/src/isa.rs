//! KTEST-04 (KTEST-001 §4 "dispatch-path classes", §6.1): `APR_FORCE_ISA` caps which runtime
//! CPU ISA paths this process may take, so every path the binary contains can be forced and
//! receipted on a host that has it — `intel` runs scalar, AVX2 and AVX-512 from one build.
//!
//! Every runtime ISA decision in this crate goes through [`cpu_feature!`], which is
//! `is_x86_feature_detected!(f) && isa::gate(f)`. Forcing only ever *narrows* what runs: a
//! gated-off feature falls through to the next path down, and every `SAFETY:` argument that
//! cites the detection still holds, because the detection is still evaluated first.
//!
//! | `APR_FORCE_ISA` | x86_64 paths allowed | refused when |
//! |---|---|---|
//! | unset, `native` | every detected feature | never |
//! | `avx512` | every detected feature | the host lacks `avx512f` |
//! | `avx2` | SSE*/AVX/AVX2/FMA/F16C — no `avx512*` | the host lacks `avx2` or `fma` |
//! | `scalar` | none — only the portable fallbacks | never |
//!
//! On aarch64 the accepted values are `native`, `neon` and `scalar` (this crate has no runtime
//! aarch64 dispatch; trueno's NEON backend reads the same variable).
//!
//! A value that names a level this host cannot run is refused, not quietly lowered to
//! `native`: a receipt stamped `avx512` that ran AVX2 is exactly what KTEST-001 STOP S-2/S-3
//! forbid. The refusal is a panic on first use, naming the accepted values.
//!
//! [`refused`] counts gate refusals of features the host *does* have — the proof, in a receipt,
//! that the force engaged at a live dispatch site rather than being read and ignored.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// The environment variable that caps runtime ISA dispatch.
pub const ENV: &str = "APR_FORCE_ISA";

/// How far up the ISA ladder this process may dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Ceiling {
    /// Portable code only.
    Scalar,
    /// x86_64: up to AVX2/FMA/F16C. aarch64: NEON.
    Simd,
    /// Every detected feature (the default).
    Native,
}

impl Ceiling {
    /// The name a receipt records: `scalar`, `avx2`/`neon`, or `native`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Simd if cfg!(target_arch = "aarch64") => "neon",
            Self::Simd => "avx2",
            Self::Native => "native",
        }
    }
}

static REFUSED: AtomicU64 = AtomicU64::new(0);

/// Parse an `APR_FORCE_ISA` value against the features `host_has` reports.
///
/// # Errors
/// An unknown value, or one naming a level the host cannot run (S-3).
pub fn parse(value: Option<&str>, host_has: impl Fn(&str) -> bool) -> Result<Ceiling, String> {
    let need = |c: Ceiling, v: &str, feats: &[&str]| match feats.iter().find(|f| !host_has(f)) {
        Some(f) => Err(format!(
            "{ENV}={v}: this host lacks `{f}`, so the {v} path cannot run here (KTEST-001 S-3)"
        )),
        None => Ok(c),
    };
    let v = value.map(str::trim).unwrap_or("");
    match (v, cfg!(target_arch = "aarch64")) {
        ("" | "native", _) => Ok(Ceiling::Native),
        ("scalar", _) => Ok(Ceiling::Scalar),
        ("avx512", false) => need(Ceiling::Native, v, &["avx512f"]),
        ("avx2", false) => need(Ceiling::Simd, v, &["avx2", "fma"]),
        ("neon", true) => need(Ceiling::Simd, v, &["neon"]),
        _ => Err(format!(
            "{ENV}={v:?} is not one of: native, scalar, {}",
            if cfg!(target_arch = "aarch64") {
                "neon"
            } else {
                "avx2, avx512"
            }
        )),
    }
}

/// Whether `feature` runs at or below `ceiling`.
#[must_use]
pub fn within(ceiling: Ceiling, feature: &str) -> bool {
    match ceiling {
        Ceiling::Native => true,
        Ceiling::Scalar => false,
        // Unknown names sit above SIMD, so a new feature string is off until classified.
        Ceiling::Simd => matches!(
            feature,
            "sse"
                | "sse2"
                | "sse3"
                | "ssse3"
                | "sse4.1"
                | "sse4.2"
                | "avx"
                | "avx2"
                | "fma"
                | "f16c"
                | "neon"
        ),
    }
}

/// This process's ceiling, read from [`ENV`] once.
///
/// # Panics
/// If [`ENV`] is set to a value [`parse`] refuses.
#[must_use]
pub fn ceiling() -> Ceiling {
    static CEILING: OnceLock<Ceiling> = OnceLock::new();
    *CEILING.get_or_init(|| {
        parse(std::env::var(ENV).ok().as_deref(), host_has).unwrap_or_else(|e| panic!("{e}"))
    })
}

/// Whether the ceiling permits `feature`, without counting — for reporting what is in force.
#[must_use]
pub fn permits(feature: &str) -> bool {
    within(ceiling(), feature)
}

/// The dispatch-site check behind [`cpu_feature!`]: call only after the host reported
/// `feature`, so a refusal here always means a live path was forced off.
#[must_use]
pub fn gate(feature: &str) -> bool {
    let ok = permits(feature);
    if !ok {
        REFUSED.fetch_add(1, Ordering::Relaxed);
    }
    ok
}

/// Dispatch-site refusals so far in this process (see the module doc).
#[must_use]
pub fn refused() -> u64 {
    REFUSED.load(Ordering::Relaxed)
}

#[allow(unused_variables)]
fn host_has(feature: &str) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        match feature {
            "avx2" => std::arch::is_x86_feature_detected!("avx2"),
            "fma" => std::arch::is_x86_feature_detected!("fma"),
            "avx512f" => std::arch::is_x86_feature_detected!("avx512f"),
            _ => false,
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        feature == "neon" && std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

/// `is_x86_feature_detected!(f) && isa::gate(f)` — the only way this crate may ask for a
/// runtime x86 feature outside tests (FALSIFY-KTEST-04-GUARD). Call it as
/// `crate::isa::cpu_feature!("avx2")`.
macro_rules! cpu_feature {
    ($f:tt) => {
        (std::arch::is_x86_feature_detected!($f) && $crate::isa::gate($f))
    };
}
#[allow(unused_imports)] // only x86_64 dispatch sites expand it
pub(crate) use cpu_feature;

#[cfg(test)]
#[path = "isa_tests.rs"]
mod tests;
