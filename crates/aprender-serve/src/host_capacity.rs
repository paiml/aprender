//! #4947 F2: will the Qwen3.5 host build fit in host RAM — decided BEFORE building it.
//!
//! `apr run` builds the Qwen3.5 host model ([`Qwen35Forward::leak_host`]) before any
//! device is touched, on the GPU path too. On a small unified-memory device (a Jetson
//! Orin: 7.6 GiB that the CPU and the GPU share) that build ran straight into earlyoom
//! and the kernel OOM killer, and the user got a killed process instead of a reason.
//! [`host_build_verdict`] is the admission rule as a pure function of numbers, so a
//! case table pins it; [`admit_qwen35_host_build`] measures the host and applies it.
//!
//! The rule, quorum-ruled on #4947 (Q8b, 3/3), refusing when
//!
//! ```text
//! need = 2.75 × file_bytes + f32 token-embedding bytes  >  90% × MemAvailable
//! ```
//!
//! The host build holds the mapped file, an owned quantized copy of the weights and
//! `token_embd.weight` dequantized to f32 (`vocab × hidden × 4` bytes, whatever the
//! file's quantization). That last term is fixed per architecture, so `k × file`
//! alone cannot fit: a 0.8B IQ2_XXS file peaks at 5.47× its size and a 27B Q4_K_M at
//! 2.59×. Subtracting the embedding leaves a tight band. Passing-run peaks
//! (`/usr/bin/time -v`, `apr run --no-gpu`, apr 0.70.3, x86_64, MiB):
//!
//! | model | file | peak | f32 embd | (peak − embd) / file | need | margin |
//! |-------|------|------|----------|----------------------|------|--------|
//! | 0.8B Q4_K_M | 507.8 | 2120.8 | 970 | 2.27 | 2366.6 | +11.6% |
//! | 0.8B UD-IQ2_XXS | 322.6 | 1764.6 | 970 | 2.46 | 1857.0 | +5.2% |
//! | 2B Q4_K_M | 1221.5 | 4518.4 | 1940 | 2.11 | 5299.1 | +17.3% |
//! | 4B Q4_K_M | 2614.0 | 7872.1 | 2425 | 2.08 | 9613.4 | +22.1% |
//! | 4B UD-Q4_K_XL | 2777.2 | 8199.0 | 2425 | 2.08 | 10062.3 | +22.7% |
//! | 9B Q4_K_M | 5417.4 | 17280.3 | 3880 | 2.47 | 18777.8 | +8.7% |
//! | 27B Q4_K_M | 15965.3 | 41331.0 | 4850 | 2.29 | 48754.5 | +18.0% |
//!
//! 2.75 covers the worst residual (9B, whose Q4_K embedding is briefly held twice by
//! the dequant's collect) and every row by at least 5.2%. The 90% keeps a tenth of
//! `MemAvailable` free at the predicted need, above earlyoom's 5%-of-total floor on the
//! Jetson. On GB10 (`MemAvailable` 90.3 GiB) the 27B needs 47.6 GiB and is admitted
//! as before; on the Jetson (3910 MiB available) the 0.8B is admitted and the 2B, whose
//! real peak of 4518 MiB is more than the device has, is refused.
//!
//! [`Qwen35Forward::leak_host`]: crate::gguf::qwen35_session::Qwen35Forward::leak_host

use crate::error::{RealizarError, Result};
use crate::gguf::GGUFModel;

const MIB: u64 = 1024 * 1024;

/// The need's multiple of the GGUF file size, in percent (2.75×).
pub const FILE_FACTOR_PERCENT: u64 = 275;

/// The share of the measured memory the need may take, in percent.
pub const AVAILABLE_PERCENT: u64 = 90;

/// What this host's RAM was measured as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostMemory {
    /// `MemAvailable` from `/proc/meminfo`: what can be allocated without swapping.
    Available(u64),
    /// No `MemAvailable` here (macOS, Windows): total RAM, a looser bound.
    TotalOnly(u64),
    /// Nothing could be measured. The verdict refuses (#2568: an unknown memory
    /// limit is not an unlimited one).
    Unknown,
}

impl HostMemory {
    /// Measure this host: `MemAvailable` where `/proc/meminfo` has it, else total RAM
    /// from [`crate::contract_gate::system_memory_bytes`], else [`Self::Unknown`].
    #[must_use]
    pub fn measure() -> Self {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok();
        if let Some((available, _total)) =
            meminfo.as_deref().and_then(crate::capacity::parse_meminfo)
        {
            return Self::Available(available);
        }
        crate::contract_gate::system_memory_bytes().map_or(Self::Unknown, Self::TotalOnly)
    }
}

/// Bytes the host build is predicted to need: 2.75 × the file plus the f32 embedding.
#[must_use]
pub fn host_build_need(file_bytes: u64, embedding_f32_bytes: u64) -> u64 {
    let scaled = u128::from(file_bytes) * u128::from(FILE_FACTOR_PERCENT) / 100;
    u64::try_from(scaled)
        .unwrap_or(u64::MAX)
        .saturating_add(embedding_f32_bytes)
}

/// Bytes of `token_embd.weight` once dequantized to f32. A model without one has
/// nothing to dequantize (its build fails on the missing tensor), and dimensions
/// whose product overflows count as `u64::MAX`, which the verdict refuses.
#[must_use]
pub fn qwen35_embedding_f32_bytes(model: &GGUFModel) -> u64 {
    model
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .map_or(0, |t| {
            t.dims
                .iter()
                .try_fold(4u64, |acc, &d| acc.checked_mul(d))
                .unwrap_or(u64::MAX)
        })
}

/// The admission rule itself, isolated from how memory was measured.
///
/// # Errors
/// [`RealizarError::HostRamRefused`] with the arithmetic when the need exceeds 90%
/// of the measured memory, or when nothing could be measured.
pub fn host_build_verdict(
    file_bytes: u64,
    embedding_f32_bytes: u64,
    memory: HostMemory,
) -> Result<()> {
    let need = host_build_need(file_bytes, embedding_f32_bytes);
    let terms = format!(
        "~{} MiB (2.75 x file {} MiB + f32 token embedding {} MiB)",
        need / MIB,
        file_bytes / MIB,
        embedding_f32_bytes / MIB
    );
    let (measured, what) = match memory {
        HostMemory::Available(bytes) => (bytes, "available"),
        HostMemory::TotalOnly(bytes) => (
            bytes,
            "of total RAM (this platform reports no MemAvailable)",
        ),
        HostMemory::Unknown => {
            return Err(RealizarError::HostRamRefused(format!(
                "cannot measure host RAM on this platform ({}), so the guard cannot be \
                 evaluated; refusing to build {terms}. An unknown memory limit is not an \
                 unlimited one (#2568, #4947)",
                std::env::consts::OS
            )));
        },
    };
    // Exact in u128, and never above `measured`, so it fits back in a u64.
    let limit = u128::from(measured) * u128::from(AVAILABLE_PERCENT) / 100;
    if u128::from(need) > limit {
        return Err(RealizarError::HostRamRefused(format!(
            "the Qwen3.5 host build needs {terms}, more than 90% of the {} MiB {what} \
             ({} MiB), and would be OOM-killed. Free memory or use a smaller model (#4947)",
            measured / MIB,
            limit / u128::from(MIB)
        )));
    }
    Ok(())
}

/// Refuse a Qwen3.5 host build that `memory` cannot hold, before a byte of it is built.
///
/// # Errors
/// As [`host_build_verdict`].
pub fn admit_qwen35_host_build(
    model: &GGUFModel,
    file_bytes: u64,
    memory: HostMemory,
) -> Result<()> {
    host_build_verdict(file_bytes, qwen35_embedding_f32_bytes(model), memory)
}

#[cfg(test)]
#[path = "host_capacity_tests.rs"]
mod tests;
