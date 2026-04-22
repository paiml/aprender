//! `Device` — selector for the training backend (`apr pretrain`).
//!
//! Contract binding: `contracts/entrenar/gpu-training-backend-v1.yaml`
//! §device_dispatch.
//!
//! The string grammar accepted by `resolve_device` is fixed by
//! INV-GPUTRAIN-001 / §device_dispatch.requested_device.grammar:
//!
//! ```text
//! ^(cpu|cuda(:[0-9]|:1[0-5])?|auto)$
//! ```
//!
//! - `cpu`            — force the CPU (trueno SIMD) training path.
//! - `cuda`           — alias for `cuda:0`.
//! - `cuda:N` (0..=15)— explicit CUDA device index.
//! - `auto`           — `cuda:0` if `cuda_training_available()`, else `cpu`.
//!
//! The `auto` resolution is NOT a silent fallback: callers are obliged
//! by GATE-GPUTRAIN-002 to print the resolved `Device` before starting
//! training so the operator sees which backend was actually selected.
//!
//! Explicit `cuda` / `cuda:N` on a host without a usable CUDA runtime
//! MUST return `DeviceError::CudaNotAvailable`. FALSIFY-GPUTRAIN-002
//! binds this invariant.

use std::fmt;

use crate::autograd::cuda_training_available;

/// Training backend selection.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Device {
    /// CPU (trueno SIMD) — `TransformerTrainer`.
    Cpu,
    /// CUDA device `index` — `CudaTransformerTrainer`.
    Cuda { index: u8 },
}

impl Device {
    /// Short human-readable tag used in CLI banners and run-dir metadata.
    #[must_use]
    pub fn tag(&self) -> String {
        match self {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda { index } => format!("cuda:{index}"),
        }
    }

    /// Is this device a CUDA device (any index)?
    #[must_use]
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda { .. })
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.tag())
    }
}

/// Failure modes for `resolve_device`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceError {
    /// Input string did not match
    /// `^(cpu|cuda(:[0-9]|:1[0-5])?|auto)$`.
    InvalidSpec(String),
    /// Caller explicitly requested CUDA (or `auto` resolved to CUDA on a
    /// host advertising CUDA) but `cuda_training_available()` returned
    /// false. GATE-GPUTRAIN-002 forbids silent CPU fallback on explicit
    /// CUDA requests — this variant IS the hard failure.
    CudaNotAvailable { requested: String },
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceError::InvalidSpec(s) => write!(
                f,
                "--device `{s}` does not match grammar \
                 ^(cpu|cuda(:[0-9]|:1[0-5])?|auto)$ \
                 (contract gpu-training-backend-v1 INV-GPUTRAIN-001)",
            ),
            DeviceError::CudaNotAvailable { requested } => write!(
                f,
                "--device `{requested}` requested but CUDA runtime is \
                 not available on this host \
                 (contract gpu-training-backend-v1 GATE-GPUTRAIN-002: \
                 no silent CPU fallback). Rebuild with `--features cuda` \
                 or pass `--device cpu` to opt in to the CPU path.",
            ),
        }
    }
}

impl std::error::Error for DeviceError {}

/// One row of `nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv,noheader`.
///
/// Contract binding: `gpu-training-backend-v1` INV-GPUTRAIN-003. The live
/// smoke run records these rows to a CSV; the discharge probe below
/// collapses the trace into a single `residency_discharge` verdict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuSample {
    pub timestamp: String,
    pub used_mib: u64,
    pub free_mib: u64,
    pub util_pct: u32,
}

/// Parse the `--query-gpu` CSV body (no header). Silently drops rows that
/// fail to parse; returning an empty `Vec` is a legitimate "no evidence"
/// signal that the discharge probe below rejects.
#[must_use]
pub fn parse_nvidia_smi_gpu_trace(csv: &str) -> Vec<GpuSample> {
    csv.lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| {
            let cols: Vec<&str> = line.split(',').map(str::trim).collect();
            if cols.len() != 4 {
                return None;
            }
            let used_mib = cols[1].strip_suffix("MiB").unwrap_or(cols[1]).trim().parse().ok()?;
            let free_mib = cols[2].strip_suffix("MiB").unwrap_or(cols[2]).trim().parse().ok()?;
            let util_pct = cols[3].strip_suffix('%').unwrap_or(cols[3]).trim().parse().ok()?;
            Some(GpuSample { timestamp: cols[0].to_string(), used_mib, free_mib, util_pct })
        })
        .collect()
}

/// One row of `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`.
///
/// Contract binding: `gpu-training-backend-v1` INV-GPUTRAIN-003 *ACTIVE*
/// discharge. This per-PID format is strictly stronger than the global
/// `--query-gpu` delta (`GpuSample`): the latter can be polluted by any
/// other process on the device, whereas this one reports exactly what the
/// training PID allocated. Contract v1.2.0 promotes GATE-GPUTRAIN-003 to
/// ACTIVE on this format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputeAppsSample {
    pub pid: u32,
    pub process_name: String,
    pub used_mib: u64,
}

/// Parse the `--query-compute-apps` CSV body. The first row is the header
/// `pid, process_name, used_gpu_memory [MiB]` emitted by `nvidia-smi` when
/// the header suffix is not suppressed; we detect and skip it. Silently
/// drops rows that fail to parse.
#[must_use]
pub fn parse_nvidia_smi_compute_apps_csv(csv: &str) -> Vec<ComputeAppsSample> {
    csv.lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| {
            let cols: Vec<&str> = line.split(',').map(str::trim).collect();
            if cols.len() != 3 {
                return None;
            }
            // Header row carries literal "pid" in col 0 — skip it.
            if cols[0].eq_ignore_ascii_case("pid") {
                return None;
            }
            let pid: u32 = cols[0].parse().ok()?;
            let process_name = cols[1].to_string();
            let used_mib = cols[2].strip_suffix("MiB").unwrap_or(cols[2]).trim().parse().ok()?;
            Some(ComputeAppsSample { pid, process_name, used_mib })
        })
        .collect()
}

/// Per-PID residency discharge for GATE-GPUTRAIN-003 ACTIVE promotion.
///
/// Returns the first sample matching `expected_pid` with at least
/// `min_mib` of memory. Failure modes are enumerated so the operator can
/// tell whether the probe ran, found the PID but saw no memory, or never
/// saw the PID at all. This is the binding the contract's
/// `blocks_active_promotion_on` clause names.
///
/// # Errors
/// Returns `Err(&'static str)` diagnosing the first reason the trace does
/// NOT constitute PID-level residency evidence.
pub fn assert_pid_residency_discharge(
    samples: &[ComputeAppsSample],
    expected_pid: u32,
    min_mib: u64,
) -> Result<&ComputeAppsSample, &'static str> {
    if samples.is_empty() {
        return Err("empty trace — nvidia-smi --query-compute-apps captured no samples");
    }
    let pid_match = samples.iter().find(|s| s.pid == expected_pid).ok_or(
        "expected PID not present — training process never appeared in compute-apps trace",
    )?;
    if pid_match.used_mib < min_mib {
        return Err("training PID present but reported memory below threshold — \
             weights likely never left CPU; check CudaTransformerTrainer init order");
    }
    Ok(pid_match)
}

/// Residency discharge for GATE-GPUTRAIN-003 / FALSIFY-GPUTRAIN-003.
///
/// Returns the peak `GpuSample` iff the trace records a memory rise of
/// at least `min_delta_mib` above the first-sample baseline. A flat trace
/// (all samples at baseline) is a FAILURE — it proves CUDA was compiled
/// and nvidia-smi saw the card, but also proves weights never made it to
/// device memory.
///
/// # Errors
/// Returns `Err(&'static str)` diagnosing the first reason the trace does
/// NOT constitute residency evidence.
pub fn assert_residency_discharge(
    samples: &[GpuSample],
    min_delta_mib: u64,
) -> Result<&GpuSample, &'static str> {
    let baseline = samples.first().ok_or("empty trace — nvidia-smi captured no samples")?;
    let peak = samples
        .iter()
        .max_by_key(|s| s.used_mib)
        .expect("non-empty iter: baseline already unwrapped");
    let delta = peak.used_mib.saturating_sub(baseline.used_mib);
    if delta < min_delta_mib {
        return Err("peak memory did not rise above baseline — weights likely never \
             left CPU; check CudaTransformerTrainer init order");
    }
    Ok(peak)
}

/// Resolve a CLI `--device` string into a concrete `Device`.
///
/// Contract: this function is THE single binding point for
/// INV-GPUTRAIN-001 (grammar) and GATE-GPUTRAIN-002 (no silent CPU
/// fallback on explicit CUDA request).
///
/// # Errors
/// - [`DeviceError::InvalidSpec`] — `spec` is not one of `cpu`,
///   `cuda`, `cuda:N` (0..=15), or `auto`.
/// - [`DeviceError::CudaNotAvailable`] — `spec` explicitly asked for
///   CUDA (or `auto` chose CUDA) but `cuda_training_available()`
///   returned `false`.
pub fn resolve_device(spec: &str) -> Result<Device, DeviceError> {
    let parsed =
        parse_device_spec(spec).ok_or_else(|| DeviceError::InvalidSpec(spec.to_string()))?;

    match parsed {
        ParsedSpec::Cpu => Ok(Device::Cpu),
        ParsedSpec::Cuda(index) => {
            if cuda_training_available() {
                Ok(Device::Cuda { index })
            } else {
                Err(DeviceError::CudaNotAvailable { requested: spec.to_string() })
            }
        }
        ParsedSpec::Auto => {
            if cuda_training_available() {
                Ok(Device::Cuda { index: 0 })
            } else {
                Ok(Device::Cpu)
            }
        }
    }
}

/// Pure-function parser: string → `ParsedSpec`. Separated from the
/// availability probe so FALSIFY-GPUTRAIN-001 (grammar) can be
/// exercised deterministically regardless of whether the host has CUDA.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ParsedSpec {
    Cpu,
    Cuda(u8),
    Auto,
}

fn parse_device_spec(spec: &str) -> Option<ParsedSpec> {
    match spec {
        "cpu" => Some(ParsedSpec::Cpu),
        "auto" => Some(ParsedSpec::Auto),
        "cuda" => Some(ParsedSpec::Cuda(0)),
        other => {
            let rest = other.strip_prefix("cuda:")?;
            // Grammar `:[0-9]|:1[0-5]` — one digit 0-9 OR "1" then 0-5.
            // `u8::from_str` rejects leading zeros ("cuda:01") by parsing
            // them, but the grammar does not: "01" is NOT in
            // `[0-9]|1[0-5]`. We therefore reject any multi-char string
            // whose first char is `0` or whose value is outside [0, 15].
            let idx: u8 = rest.parse().ok()?;
            if idx > 15 {
                return None;
            }
            // Reject leading-zero spellings that happen to parse
            // (e.g. "cuda:01"). Grammar allows only 1-2 chars AND
            // 2-char forms must start with '1'.
            match rest.len() {
                1 => {}
                2 if rest.starts_with('1') => {}
                _ => return None,
            }
            Some(ParsedSpec::Cuda(idx))
        }
    }
}

#[cfg(test)]
#[path = "device_tests.rs"]
mod tests;
