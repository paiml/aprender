//! #4947 F2: will the Qwen3.5 host build fit in host RAM — decided BEFORE building it.
//!
//! `apr run` builds the Qwen3.5 host model ([`Qwen35Forward::leak_host`]) before any
//! device is touched, on the GPU path too. On a small unified-memory device (a Jetson
//! Orin: 7.6 GiB that the CPU and the GPU share) that build ran straight into earlyoom
//! and the kernel OOM killer, and the user got a killed process instead of a reason.
//! [`host_build_verdict`] is the admission rule as a pure function of numbers, so a
//! case table pins it; [`admit_qwen35_host_build`] measures the host and applies it.
//!
//! The rule refuses when
//!
//! ```text
//! need = 1.10 × file_bytes + 0.60 × vocab × hidden × 4  >  90% × MemAvailable
//! ```
//!
//! where `MemAvailable` is lowered to `memory.max − memory.current` of the tightest
//! cgroup v2 limit on this process's path when that is smaller (#4992): in a capped
//! container or systemd slice the cgroup OOM killer fires at its own limit. Both are
//! read before the file is mapped ([`MappedGGUFModel::host_memory_before_map`]): the
//! map is populated and, where `RLIMIT_MEMLOCK` allows, locked at open, so a reading
//! taken after it has already spent the file that the need counts in full. Under a
//! 6 GiB cgroup on GB10 a post-map reading left 3432 MiB and refused the 4B, whose
//! build peaks at 3708 MiB.
//!
//! Since #4991 the host build holds one owned quantized copy of the weights (the map's
//! pages are dropped as each tensor is copied out) and keeps `token_embd.weight` in its
//! GGUF encoding. What remains above the file is per-architecture state that scales
//! with `vocab × hidden`, not with the file's quantization, so `k × file` alone cannot
//! fit: a 0.8B IQ2_XXS peaks at 2.35× its file and a 27B Q4_K_M at 1.11×. The second
//! term is sized as a share of the f32 token-embedding size, the `vocab × hidden`
//! measure the model already carries. Passing-run peaks after #4991 (`/usr/bin/time -v`
//! maximum resident set, `apr run <gguf> --no-gpu -p "What is 2+2?" -n 8`, apr 0.70.3
//! @ 1ec5fe3b31; x86_64 is noah-Lambda-Vector, aarch64 is GB10; MiB). A dash is a run
//! not taken there: the 2.75× rule this replaces refused the 4B rows under the dev
//! host's agent cgroup, and the 27B runs on GB10 only. The 9B x86_64 peak is the
//! 16-token run #4991 recorded:
//!
//! | model | file | x86_64 peak | aarch64 peak | f32 embd | peak / file | need | margin |
//! |-------|------|-------------|--------------|----------|-------------|------|--------|
//! | 0.8B Q4_K_M | 507.8 | 1007.0 | 1005.0 | 970 | 1.98 | 1140.6 | +13.3% |
//! | 0.8B UD-IQ2_XXS | 322.6 | 758.5 | 755.3 | 970 | 2.35 | 936.8 | +23.5% |
//! | 2B Q4_K_M | 1221.5 | 2120.0 | 2115.1 | 1940 | 1.74 | 2507.7 | +18.3% |
//! | 4B Q4_K_M | 2614.0 | — | 3707.7 | 2425 | 1.42 | 4330.4 | +16.8% |
//! | 4B UD-Q4_K_XL | 2777.2 | — | 3870.3 | 2425 | 1.39 | 4509.9 | +16.5% |
//! | 9B Q4_K_M | 5417.4 | 6855.2 | 6148.1 | 3880 | 1.27 | 8287.1 | +20.9% |
//! | 27B Q4_K_M | 15965.3 | — | 17741.3 | 4850 | 1.11 | 20471.8 | +15.4% |
//!
//! The margin is over the larger peak where both hosts ran, else over the one peak;
//! every row clears 13%. The 27B was not used to set 1.10 and 0.60: it was measured
//! afterwards, under this rule.
//! The 90% keeps a tenth of `MemAvailable` free at the predicted need, above
//! earlyoom's 5%-of-total floor on the Jetson. On GB10 (`MemAvailable` 90.3 GiB)
//! every row is admitted; on the Jetson (3910 MiB available) the 0.8B and the 2B are
//! admitted and the 4B is refused. The 2.75 × file + f32 embedding rule this replaces
//! fitted the pre-#4991 build, which held the f32 embedding and every page of the
//! map. After #4991 it predicted 2.3–3.1× the real peak and refused the 9B on hosts
//! with 8–20 GiB free.
//!
//! [`Qwen35Forward::leak_host`]: crate::gguf::qwen35_session::Qwen35Forward::leak_host
//! [`MappedGGUFModel::host_memory_before_map`]: crate::gguf::MappedGGUFModel::host_memory_before_map

use crate::error::{RealizarError, Result};
use crate::gguf::GGUFModel;

const MIB: u64 = 1024 * 1024;

/// The need's multiple of the GGUF file size, in percent (1.10×).
pub const FILE_FACTOR_PERCENT: u64 = 110;

/// The need's share of the f32 token-embedding size (`vocab × hidden × 4`), in percent.
pub const EMBEDDING_FACTOR_PERCENT: u64 = 60;

/// The share of the measured memory the need may take, in percent.
pub const AVAILABLE_PERCENT: u64 = 90;

/// What this host's RAM was measured as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostMemory {
    /// `MemAvailable` from `/proc/meminfo`: what can be allocated without swapping.
    Available(u64),
    /// `memory.max − memory.current` of the tightest cgroup v2 limit this process
    /// runs under, where that is below `MemAvailable` (#4992): a container or a
    /// systemd slice with a cap is OOM-killed at its own limit, not the host's.
    CgroupHeadroom(u64),
    /// No `MemAvailable` here (macOS, Windows): total RAM, a looser bound.
    TotalOnly(u64),
    /// Nothing could be measured. The verdict refuses (#2568: an unknown memory
    /// limit is not an unlimited one).
    Unknown,
}

impl HostMemory {
    /// Measure this host: the tighter of `MemAvailable` and this process's cgroup
    /// headroom where `/proc/meminfo` has it, else total RAM
    /// from [`crate::contract_gate::system_memory_bytes`], else [`Self::Unknown`].
    #[must_use]
    pub fn measure() -> Self {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok();
        if let Some((available, _total)) =
            meminfo.as_deref().and_then(crate::capacity::parse_meminfo)
        {
            return Self::bound(available, cgroup_headroom_here());
        }
        crate::contract_gate::system_memory_bytes().map_or(Self::Unknown, Self::TotalOnly)
    }

    /// `min(MemAvailable, cgroup headroom)`, naming which of the two bounds the build.
    #[must_use]
    pub fn bound(available: u64, cgroup_headroom: Option<u64>) -> Self {
        match cgroup_headroom {
            Some(headroom) if headroom < available => Self::CgroupHeadroom(headroom),
            _ => Self::Available(available),
        }
    }
}

/// Headroom of one cgroup v2 level: `memory.max − memory.current`, or `None` when
/// the level is unlimited (`max`) or its files cannot be read or parsed.
#[must_use]
pub fn cgroup_level_headroom(max: &str, current: &str) -> Option<u64> {
    let max = max.trim();
    if max == "max" {
        return None;
    }
    let current = current.trim().parse::<u64>().ok()?;
    Some(max.parse::<u64>().ok()?.saturating_sub(current))
}

/// The tightest headroom on the path from `root` down to `root/relative` (each
/// ancestor's limit caps everything under it, so the leaf alone is not enough).
/// `None` when no level on the path carries a limit.
#[must_use]
pub fn cgroup_headroom_under(root: &std::path::Path, relative: &str) -> Option<u64> {
    let mut dir = root.to_path_buf();
    let levels = std::iter::once(String::new()).chain(
        relative
            .split('/')
            .filter(|s| !s.is_empty())
            .map(str::to_owned),
    );
    levels
        .filter_map(|part| {
            dir.push(part);
            let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok();
            cgroup_level_headroom(&read("memory.max")?, &read("memory.current")?)
        })
        .min()
}

/// [`cgroup_headroom_under`] for this process: its cgroup v2 path from the `0::`
/// line of `/proc/self/cgroup`, under `/sys/fs/cgroup`. `None` on cgroup v1 or
/// off Linux, where `MemAvailable` alone bounds the build as before.
fn cgroup_headroom_here() -> Option<u64> {
    let cgroup = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let relative = cgroup.lines().find_map(|l| l.strip_prefix("0::"))?;
    cgroup_headroom_under(std::path::Path::new("/sys/fs/cgroup"), relative)
}

/// Bytes the host build is predicted to need: 1.10 × the file plus 0.60 × the f32
/// embedding size, each term rounded down.
#[must_use]
pub fn host_build_need(file_bytes: u64, embedding_f32_bytes: u64) -> u64 {
    let percent_of = |bytes: u64, percent: u64| {
        u64::try_from(u128::from(bytes) * u128::from(percent) / 100).unwrap_or(u64::MAX)
    };
    percent_of(file_bytes, FILE_FACTOR_PERCENT)
        .saturating_add(percent_of(embedding_f32_bytes, EMBEDDING_FACTOR_PERCENT))
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
        "~{} MiB (1.10 x file {} MiB + 0.60 x f32 token-embedding size {} MiB)",
        need / MIB,
        file_bytes / MIB,
        embedding_f32_bytes / MIB
    );
    let (measured, what) = match memory {
        HostMemory::Available(bytes) => (bytes, "available"),
        HostMemory::CgroupHeadroom(bytes) => (
            bytes,
            "left under this process's cgroup memory.max (below MemAvailable)",
        ),
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
