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
        },
        total,
    )
}

fn arithmetic(b: &CapacityBudget) -> String {
    format!(
        "weights {:.0} MiB + KV {:.0} MiB ({} positions x {} B at {:?}) + workspace {:.0} MiB \
         + overhead {:.0} MiB = {:.0} MiB, against {:.0} MiB free of {:.0} MiB",
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
    fn capacity_the_boundary_is_inclusive() {
        let mut i = nine_b(1000, 0, 24 * GIB, false);
        let (_, need) = budget(&i, KvDtype::F32);
        i.gpu_free_bytes = need;
        assert!(matches!(plan(&i), CapacityVerdict::Fits(b) if b.kv_dtype == KvDtype::F32));
        i.gpu_free_bytes = need - 1;
        assert!(!matches!(plan(&i), CapacityVerdict::Fits(b) if b.kv_dtype == KvDtype::F32));
    }
}
