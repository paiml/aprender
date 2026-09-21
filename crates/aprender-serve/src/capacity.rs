//! PMAT-3596 (#3596): will a request fit on the GPU — decided BEFORE loading.
//!
//! A long-context request that does not fit must be refused before a single weight
//! is uploaded, with the arithmetic that says why, and never discovered as an OOM an
//! hour into a prefill. [`plan`] is that arithmetic, as a pure function of numbers, so
//! the two places that need it compute the same thing: the load path (which refuses)
//! and `apr serve plan` / the release gate (which predicts). The budget terms and their
//! names are `apr serve plan --json`'s (`weights_mb`, `kv_cache_mb`, `activations_mb`,
//! `overhead_mb`, `total_mb`, `gpu_total_mb`), plus the two a refusal needs that a plan
//! does not: the MEASURED `gpu_free_mb` it compared against, and the KV dtype chosen.
//!
//! The rule, operator-ruled on #3596:
//!
//! 1. f32 KV when it fits the device's free memory;
//! 2. otherwise f16 KV when THAT fits — halving the KV term — and the f16 decode read
//!    exists (#3725); while it does not, refuse and name the blocker;
//! 3. otherwise refuse.
//!
//! A refusal says whether the request would fit the device's TOTAL memory: if it
//! would, the refusal is caused by another tenant ([`RefusalKind::CoTenant`]) and a
//! release gate must read it as RED, not as an acceptable "this host cannot" — or a
//! busy GPU would silently shrink the set of owed cells (aprender-62, #3596).

use serde::{Deserialize, Serialize};

const MIB: f64 = 1024.0 * 1024.0;

/// Bytes the plan holds back for the CUDA context, cuBLAS, JIT-compiled modules and
/// allocator fragmentation — the constant `apr serve plan` already charges.
pub const OVERHEAD_BYTES: u64 = 512 * 1024 * 1024;

/// On a unified-memory device (GB10), bytes of `MemAvailable` the plan leaves to
/// everything else on the host — the CI build containers above all: at 15:56Z on
/// 2026-09-21 three concurrent interactive runs on gx10 made the kernel's global
/// OOM kill 18 of them. NOT a measured peak of those containers — a named reserve
/// until one is measured; shared with the qwen3moe resident path (#3714).
pub const UNIFIED_HEADROOM_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// How a device's memory is shaped, as a plan needs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "class", rename_all = "snake_case")]
pub enum DeviceMemory {
    /// A discrete GPU (RTX 4090): its own VRAM, `cuMemGetInfo` is the truth.
    Discrete {
        /// `cuMemGetInfo` free.
        free: u64,
        /// `cuMemGetInfo` total.
        total: u64,
    },
    /// An integrated GPU sharing host memory (GB10): allocations are managed memory
    /// from the host pool, and `cuMemGetInfo`'s free EXCLUDES reclaimable page cache —
    /// measured on gx10 by aprender-eb: 16,056 MiB "free" beside 92,443 MiB
    /// `MemAvailable`. The host's `MemAvailable` is the truth.
    Unified {
        /// `/proc/meminfo` `MemAvailable`.
        available: u64,
        /// `/proc/meminfo` `MemTotal`.
        total: u64,
        /// `cuMemGetInfo` free, recorded but not planned against.
        cuda_free: u64,
    },
}

impl DeviceMemory {
    /// `(free, total)` a plan compares against: a unified device keeps
    /// [`UNIFIED_HEADROOM_BYTES`] of `MemAvailable` back for the rest of the host.
    #[must_use]
    pub const fn plan_free_total(self) -> (u64, u64) {
        match self {
            Self::Discrete { free, total } => (free, total),
            Self::Unified {
                available, total, ..
            } => (available.saturating_sub(UNIFIED_HEADROOM_BYTES), total),
        }
    }
}

/// Parse `MemAvailable` and `MemTotal` (bytes) out of `/proc/meminfo` text.
#[must_use]
pub fn parse_meminfo(text: &str) -> Option<(u64, u64)> {
    let field = |name: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(name))
            .and_then(|rest| rest.trim().strip_suffix("kB"))
            .and_then(|kb| kb.trim().parse::<u64>().ok())
            .map(|kb| kb * 1024)
    };
    Some((field("MemAvailable:")?, field("MemTotal:")?))
}

/// Measure the device the executor is bound to (#3596, #3714): discrete →
/// `cuMemGetInfo`; integrated → the host's `/proc/meminfo`.
///
/// # Errors
/// The driver refuses the attribute or memory query, or an integrated device's host
/// has no readable `/proc/meminfo` (never guessed).
#[cfg(feature = "cuda")]
pub fn measure_device_memory(
    executor: &crate::cuda::CudaExecutor,
) -> std::result::Result<DeviceMemory, String> {
    use trueno_gpu::driver::{classify_device_memory, DeviceMemoryClass};
    let (cuda_free, cuda_total) = executor
        .memory_info()
        .map_err(|e| format!("cuMemGetInfo failed: {e}"))?;
    let class = classify_device_memory(executor.context())
        .map_err(|e| format!("the device class could not be read: {e}"))?;
    match class {
        DeviceMemoryClass::ClassicDevice => Ok(DeviceMemory::Discrete {
            free: cuda_free as u64,
            total: cuda_total as u64,
        }),
        DeviceMemoryClass::UnifiedMemory => {
            let text = std::fs::read_to_string("/proc/meminfo")
                .map_err(|e| format!("a unified-memory device needs /proc/meminfo: {e}"))?;
            let (available, total) = parse_meminfo(&text)
                .ok_or_else(|| "/proc/meminfo has no MemAvailable/MemTotal".to_string())?;
            Ok(DeviceMemory::Unified {
                available,
                total,
                cuda_free: cuda_free as u64,
            })
        },
    }
}

/// The numbers a plan is made from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapacityInputs {
    /// Weights the device will hold, as uploaded (quantized bytes, f32 vectors).
    pub weights_bytes: u64,
    /// KV bytes per position at f32, summed over the layers that HAVE a KV cache (for
    /// a hybrid, the full-attention layers only).
    pub kv_bytes_per_token_f32: u64,
    /// Positions the KV cache must hold: prompt + generated + 1.
    pub seq_len: u64,
    /// Everything else the run allocates: prefill/decode workspace, recurrent state,
    /// scratch.
    pub workspace_bytes: u64,
    /// See [`OVERHEAD_BYTES`].
    pub overhead_bytes: u64,
    /// Free device memory, measured after the context exists.
    pub gpu_free_bytes: u64,
    /// Total device memory.
    pub gpu_total_bytes: u64,
    /// Can decode attention read an f16 KV cache (#3725)?
    pub f16_kv_decode_available: bool,
    /// Where `gpu_free_bytes`/`gpu_total_bytes` came from, when a device was measured
    /// — so a refusal can say "MemAvailable less the headroom" instead of "free".
    pub memory: Option<DeviceMemory>,
}

/// The KV cache element type a plan chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KvDtype {
    /// 4 bytes per element.
    F32,
    /// 2 bytes per element.
    F16,
}

impl KvDtype {
    /// Bytes per KV element.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

/// The budget of one plan, in `apr serve plan`'s terms.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CapacityBudget {
    /// Weights, MiB.
    pub weights_mb: f64,
    /// KV bytes per position at the chosen dtype.
    pub kv_bytes_per_token: u64,
    /// The chosen (or, in a refusal, the last tried) KV dtype.
    pub kv_dtype: KvDtype,
    /// Positions the KV cache holds.
    pub seq_len: u64,
    /// `kv_bytes_per_token * seq_len`, MiB.
    pub kv_cache_mb: f64,
    /// Workspace, MiB.
    pub activations_mb: f64,
    /// Overhead, MiB.
    pub overhead_mb: f64,
    /// Sum of the four terms above, MiB.
    pub total_mb: f64,
    /// Measured free device memory, MiB.
    pub gpu_free_mb: f64,
    /// Device total, MiB.
    pub gpu_total_mb: f64,
    /// The measurement behind the two figures above, if one was made.
    pub memory: Option<DeviceMemory>,
}

/// Why a plan refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefusalKind {
    /// Does not fit the device's TOTAL memory even with an f16 KV: this host cannot
    /// serve the request.
    ExceedsDevice,
    /// Would fit the device's total, not its free memory: another process holds the
    /// difference. Not a property of the host — a gate reads it as RED.
    CoTenant,
    /// Fits only with an f16 KV, and decode cannot read one yet (#3725).
    F16DecodeUnavailable,
}

/// A refused plan: the budget, the kind, and the one-line reason.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacityRefusal {
    /// Why.
    pub kind: RefusalKind,
    /// The arithmetic, at the dtype the refusal is about.
    pub budget: CapacityBudget,
    /// One line, with the numbers.
    pub reason: String,
}

impl std::fmt::Display for CapacityRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.reason)
    }
}

/// What a plan concluded.
#[derive(Debug, Clone, PartialEq)]
pub enum CapacityVerdict {
    /// The request fits, with this budget (and its KV dtype).
    Fits(CapacityBudget),
    /// It does not.
    Refused(CapacityRefusal),
}

fn budget(i: &CapacityInputs, dtype: KvDtype) -> (CapacityBudget, u64) {
    let per_token = i.kv_bytes_per_token_f32 / 4 * dtype.bytes();
    let kv = per_token.saturating_mul(i.seq_len);
    let total = i
        .weights_bytes
        .saturating_add(kv)
        .saturating_add(i.workspace_bytes)
        .saturating_add(i.overhead_bytes);
    let mb = |b: u64| b as f64 / MIB;
    (
        CapacityBudget {
            weights_mb: mb(i.weights_bytes),
            kv_bytes_per_token: per_token,
            kv_dtype: dtype,
            seq_len: i.seq_len,
            kv_cache_mb: mb(kv),
            activations_mb: mb(i.workspace_bytes),
            overhead_mb: mb(i.overhead_bytes),
            total_mb: mb(total),
            gpu_free_mb: mb(i.gpu_free_bytes),
            gpu_total_mb: mb(i.gpu_total_bytes),
            memory: i.memory,
        },
        total,
    )
}

fn arithmetic(b: &CapacityBudget) -> String {
    let source = match b.memory {
        Some(DeviceMemory::Unified { available, cuda_free, .. }) => format!(
            " (unified memory: MemAvailable {:.0} MiB less a {:.0} MiB headroom; cuMemGetInfo said {:.0} MiB free)",
            available as f64 / MIB,
            UNIFIED_HEADROOM_BYTES as f64 / MIB,
            cuda_free as f64 / MIB
        ),
        Some(DeviceMemory::Discrete { .. }) => " (discrete GPU: cuMemGetInfo)".to_string(),
        None => String::new(),
    };
    format!(
        "weights {:.0} MiB + KV {:.0} MiB ({} positions x {} B at {:?}) + workspace {:.0} MiB \
         + overhead {:.0} MiB = {:.0} MiB, against {:.0} MiB free of {:.0} MiB{source}",
        b.weights_mb,
        b.kv_cache_mb,
        b.seq_len,
        b.kv_bytes_per_token,
        b.kv_dtype,
        b.activations_mb,
        b.overhead_mb,
        b.total_mb,
        b.gpu_free_mb,
        b.gpu_total_mb
    )
}

/// Decide f32 / f16 / refuse. See the module docs for the rule.
///
/// A refusal is classified by what an EMPTY device of this size could do, because
/// that is what a release gate owes (aprender-62, #3596):
///
/// | f32 fits total | f16 fits total | refusal |
/// |---|---|---|
/// | yes | — | [`RefusalKind::CoTenant`] — an empty card holds it at f32 |
/// | no | yes, f16 decode missing | [`RefusalKind::F16DecodeUnavailable`] (#3725) |
/// | no | yes, f16 decode present | [`RefusalKind::CoTenant`] — an empty card holds it at f16 |
/// | no | no | [`RefusalKind::ExceedsDevice`] |
#[must_use]
pub fn plan(i: &CapacityInputs) -> CapacityVerdict {
    let (b32, need32) = budget(i, KvDtype::F32);
    if need32 <= i.gpu_free_bytes {
        return CapacityVerdict::Fits(b32);
    }
    let (b16, need16) = budget(i, KvDtype::F16);
    if i.f16_kv_decode_available && need16 <= i.gpu_free_bytes {
        return CapacityVerdict::Fits(b16);
    }
    let held = i.gpu_total_bytes.saturating_sub(i.gpu_free_bytes) as f64 / MIB;
    let co_tenant = |b: CapacityBudget| {
        CapacityVerdict::Refused(CapacityRefusal {
            kind: RefusalKind::CoTenant,
            reason: format!(
                "an empty GPU of this size could hold this context at {:?}, but other processes                  hold {held:.0} MiB of it (free < need <= total): {}",
                b.kv_dtype,
                arithmetic(&b)
            ),
            budget: b,
        })
    };
    if need32 <= i.gpu_total_bytes {
        return co_tenant(b32);
    }
    if need16 <= i.gpu_total_bytes {
        if i.f16_kv_decode_available {
            return co_tenant(b16);
        }
        return CapacityVerdict::Refused(CapacityRefusal {
            kind: RefusalKind::F16DecodeUnavailable,
            reason: format!(
                "this GPU can hold the context only with an f16 KV cache, and f16 KV decode read \
                 not available (#3725): {}",
                arithmetic(&b16)
            ),
            budget: b16,
        });
    }
    CapacityVerdict::Refused(CapacityRefusal {
        kind: RefusalKind::ExceedsDevice,
        reason: format!(
            "this GPU cannot hold the context even with an f16 KV cache: {}",
            arithmetic(&b16)
        ),
        budget: b16,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB_U: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB_U;
    /// An RTX 4090 as `cuMemGetInfo` reports it (24,035 MiB), and a realistic idle
    /// free figure once a context exists.
    const CARD_4090: u64 = 24_035 * MIB_U;
    const IDLE_4090: u64 = 23_300 * MIB_U;
    /// GB10 (119 GiB unified).
    const GB10: u64 = 119 * GIB;

    /// Qwen3.5-9B: 4,873 MiB of weights, 64 KiB/position at f32, 1.5 GiB workspace.
    fn nine_b(seq_len: u64, free: u64, total: u64, f16: bool) -> CapacityInputs {
        CapacityInputs {
            weights_bytes: 4_873 * MIB_U,
            kv_bytes_per_token_f32: 64 * 1024,
            seq_len,
            workspace_bytes: 1_500 * MIB_U,
            overhead_bytes: OVERHEAD_BYTES,
            gpu_free_bytes: free,
            gpu_total_bytes: total,
            f16_kv_decode_available: f16,
            memory: None,
        }
    }

    /// Qwen3.5-27B: 15,287 MiB of weights, 128 KiB/position at f32.
    fn twenty_seven_b(seq_len: u64, free: u64, total: u64, f16: bool) -> CapacityInputs {
        CapacityInputs {
            weights_bytes: 15_287 * MIB_U,
            kv_bytes_per_token_f32: 128 * 1024,
            workspace_bytes: 2_000 * MIB_U,
            ..nine_b(seq_len, free, total, f16)
        }
    }

    fn kind(v: &CapacityVerdict) -> &'static str {
        match v {
            CapacityVerdict::Fits(b) => match b.kv_dtype {
                KvDtype::F32 => "f32",
                KvDtype::F16 => "f16",
            },
            CapacityVerdict::Refused(r) => match r.kind {
                RefusalKind::ExceedsDevice => "exceeds_device",
                RefusalKind::CoTenant => "co_tenant",
                RefusalKind::F16DecodeUnavailable => "f16_decode_unavailable",
            },
        }
    }

    /// The case table. Each row states the arithmetic that decides it (MiB).
    #[test]
    fn capacity_plan_case_table() {
        let rows: Vec<(&str, CapacityInputs, &str)> = vec![
            // 4873 + 1250 + 1500 + 512 = 8135 <= 23300
            (
                "9B@20k on an idle 4090",
                nine_b(20_001, IDLE_4090, CARD_4090, false),
                "f32",
            ),
            // 4873 + 9250 + 1500 + 512 = 16135 <= 23300
            (
                "9B@148k on an idle 4090",
                nine_b(148_001, IDLE_4090, CARD_4090, false),
                "f32",
            ),
            // 4873 + 16384 + 1500 + 512 = 23269 <= 23300: f32, by 31 MiB
            (
                "9B@262k on an idle 4090",
                nine_b(262_145, IDLE_4090, CARD_4090, false),
                "f32",
            ),
            // f32 23269 > 22000; f16 4873 + 8192 + 2012 = 15077 <= 22000
            (
                "9B@262k, 22000 free, f16 decode",
                nine_b(262_145, 22_000 * MIB_U, CARD_4090, true),
                "f16",
            ),
            // f32 23269 > 22000 but <= 24035 total: an empty card holds it at f32
            (
                "9B@262k, 22000 free, no f16",
                nine_b(262_145, 22_000 * MIB_U, CARD_4090, false),
                "co_tenant",
            ),
            // 9B@148k while another process holds 12 GiB: 16135 > 11747 free, <= total
            (
                "9B@148k beside a 12 GiB co-tenant",
                nine_b(148_001, CARD_4090 - 12 * GIB, CARD_4090, false),
                "co_tenant",
            ),
            // f32 4873 + 18750 + 2012 = 25635 > 24035 total; f16 16260 <= total, no f16
            (
                "9B@300k on a 4090, no f16",
                nine_b(300_000, IDLE_4090, CARD_4090, false),
                "f16_decode_unavailable",
            ),
            (
                "9B@300k on a 4090, f16",
                nine_b(300_000, IDLE_4090, CARD_4090, true),
                "f16",
            ),
            // f16 4873 + 62500 + 2012 > 24035
            (
                "9B@2M positions",
                nine_b(2_000_000, IDLE_4090, CARD_4090, true),
                "exceeds_device",
            ),
            // aprender-62's measured cell: f16 15287 + 9250 + 2512 = 27049 > 24035
            (
                "27B@148k on a 4090",
                twenty_seven_b(148_001, IDLE_4090, CARD_4090, true),
                "exceeds_device",
            ),
            (
                "27B@262k on a 4090",
                twenty_seven_b(262_145, IDLE_4090, CARD_4090, true),
                "exceeds_device",
            ),
            // 15287 + 32768 + 2512 = 50567 <= 110 GiB
            (
                "27B@262k on GB10",
                twenty_seven_b(262_145, 110 * GIB, GB10, false),
                "f32",
            ),
        ];
        for (what, inputs, want) in rows {
            assert_eq!(kind(&plan(&inputs)), want, "{what}");
        }
    }

    #[test]
    fn capacity_budget_terms_add_up_and_name_the_numbers() {
        let CapacityVerdict::Refused(r) = plan(&nine_b(300_000, IDLE_4090, CARD_4090, false))
        else {
            panic!("9B@300k on a 4090 without f16 decode must refuse")
        };
        let b = r.budget;
        let sum = b.weights_mb + b.kv_cache_mb + b.activations_mb + b.overhead_mb;
        assert!((sum - b.total_mb).abs() < 1e-6, "{b:?}");
        assert_eq!(b.kv_dtype, KvDtype::F16);
        assert_eq!(b.kv_bytes_per_token, 32 * 1024);
        assert!(r.reason.contains("#3725"), "{}", r.reason);
        assert!(r.reason.contains("MiB free of"), "{}", r.reason);
        // The refusal serialises with serve plan's term names.
        let v = serde_json::to_value(&r).expect("json");
        for key in [
            "weights_mb",
            "kv_cache_mb",
            "activations_mb",
            "overhead_mb",
            "total_mb",
            "gpu_free_mb",
            "gpu_total_mb",
            "kv_bytes_per_token",
            "kv_dtype",
            "seq_len",
        ] {
            assert!(v["budget"].get(key).is_some(), "budget lacks {key}: {v}");
        }
        assert_eq!(v["kind"], "f16_decode_unavailable");
        assert_eq!(v["budget"]["kv_dtype"], "f16");
    }

    #[test]
    fn capacity_unified_memory_plans_against_mem_available_less_the_headroom() {
        let text =
            "MemTotal:       124958720 kB\nMemFree:  16000000 kB\nMemAvailable:   94661632 kB\n";
        let (available, total) = parse_meminfo(text).expect("parse");
        assert_eq!(available, 94_661_632 * 1024);
        assert_eq!(total, 124_958_720 * 1024);
        let unified = DeviceMemory::Unified {
            available,
            total,
            cuda_free: 16_056 * MIB_U,
        };
        let (free, t) = unified.plan_free_total();
        assert_eq!(
            free,
            available - UNIFIED_HEADROOM_BYTES,
            "headroom is held back"
        );
        assert_eq!(t, total);
        // The 27B at 262,144 fits GB10 by MemAvailable, and would be REFUSED by the
        // 16 GiB cuMemGetInfo figure — the bug aprender-eb measured.
        let mut i = twenty_seven_b(262_145, free, t, false);
        assert_eq!(kind(&plan(&i)), "f32");
        i.gpu_free_bytes = 16_056 * MIB_U;
        assert_ne!(kind(&plan(&i)), "f32");
        assert!(
            parse_meminfo("MemTotal: 1 kB\n").is_none(),
            "no MemAvailable: refuse to guess"
        );
        // A refusal on unified memory names where "free" came from.
        let mut big = twenty_seven_b(2_000_000, free, t, false);
        big.memory = Some(unified);
        let CapacityVerdict::Refused(r) = plan(&big) else {
            panic!("the 27B at 2M positions cannot fit GB10")
        };
        assert!(r.reason.contains("MemAvailable"), "{}", r.reason);
        assert!(r.reason.contains("cuMemGetInfo said"), "{}", r.reason);
        let discrete = DeviceMemory::Discrete { free: 5, total: 9 };
        assert_eq!(discrete.plan_free_total(), (5, 9));
    }

    #[test]
    fn capacity_the_boundary_is_inclusive() {
        let mut i = nine_b(1000, 0, 24 * GIB, false);
        let (_, need) = budget(&i, KvDtype::F32);
        i.gpu_free_bytes = need;
        assert!(matches!(plan(&i), CapacityVerdict::Fits(b) if b.kv_dtype == KvDtype::F32));
        i.gpu_free_bytes = need - 1;
        assert!(!matches!(plan(&i), CapacityVerdict::Fits(b) if b.kv_dtype == KvDtype::F32));
    }
}
