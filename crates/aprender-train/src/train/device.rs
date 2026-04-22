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
    let parsed = parse_device_spec(spec).ok_or_else(|| DeviceError::InvalidSpec(spec.to_string()))?;

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
mod tests {
    use super::*;

    // ─── FALSIFY-GPUTRAIN-001: grammar ──────────────────────────────────
    //
    // Binds contract `gpu-training-backend-v1` INV-GPUTRAIN-001. Any
    // string that does NOT match
    // `^(cpu|cuda(:[0-9]|:1[0-5])?|auto)$` MUST be rejected with
    // `DeviceError::InvalidSpec`; any string that DOES match MUST parse.

    #[test]
    fn falsify_gputrain_001_accepts_cpu() {
        assert_eq!(parse_device_spec("cpu"), Some(ParsedSpec::Cpu));
    }

    #[test]
    fn falsify_gputrain_001_accepts_auto() {
        assert_eq!(parse_device_spec("auto"), Some(ParsedSpec::Auto));
    }

    #[test]
    fn falsify_gputrain_001_accepts_cuda_alias() {
        assert_eq!(parse_device_spec("cuda"), Some(ParsedSpec::Cuda(0)));
    }

    #[test]
    fn falsify_gputrain_001_accepts_cuda_single_digit() {
        for i in 0..=9u8 {
            let spec = format!("cuda:{i}");
            assert_eq!(
                parse_device_spec(&spec),
                Some(ParsedSpec::Cuda(i)),
                "grammar must accept {spec}",
            );
        }
    }

    #[test]
    fn falsify_gputrain_001_accepts_cuda_10_through_15() {
        for i in 10..=15u8 {
            let spec = format!("cuda:{i}");
            assert_eq!(
                parse_device_spec(&spec),
                Some(ParsedSpec::Cuda(i)),
                "grammar must accept {spec}",
            );
        }
    }

    #[test]
    fn falsify_gputrain_001_rejects_index_16() {
        assert_eq!(parse_device_spec("cuda:16"), None);
    }

    #[test]
    fn falsify_gputrain_001_rejects_index_99() {
        assert_eq!(parse_device_spec("cuda:99"), None);
    }

    #[test]
    fn falsify_gputrain_001_rejects_leading_zero() {
        // Grammar allows one digit [0-9] or two chars 1[0-5]; "01"
        // matches neither.
        assert_eq!(parse_device_spec("cuda:01"), None);
    }

    #[test]
    fn falsify_gputrain_001_rejects_empty_index() {
        assert_eq!(parse_device_spec("cuda:"), None);
    }

    #[test]
    fn falsify_gputrain_001_rejects_negative_index() {
        assert_eq!(parse_device_spec("cuda:-1"), None);
    }

    #[test]
    fn falsify_gputrain_001_rejects_typo() {
        assert_eq!(parse_device_spec("gpu"), None);
        assert_eq!(parse_device_spec("CUDA"), None);
        assert_eq!(parse_device_spec("cudaa"), None);
        assert_eq!(parse_device_spec(""), None);
        assert_eq!(parse_device_spec(" cpu"), None);
    }

    #[test]
    fn falsify_gputrain_001_resolve_wraps_invalid_as_device_error() {
        let err = resolve_device("gpu").unwrap_err();
        assert!(matches!(err, DeviceError::InvalidSpec(ref s) if s == "gpu"));
    }

    // ─── FALSIFY-GPUTRAIN-002: no silent CPU fallback ──────────────────
    //
    // Binds contract `gpu-training-backend-v1` INV-GPUTRAIN-002 /
    // GATE-GPUTRAIN-002. Explicit `--device cuda` / `cuda:N` MUST hard-
    // fail when the host has no CUDA runtime. `auto` is the ONLY spec
    // allowed to fall back.

    #[test]
    fn falsify_gputrain_002_explicit_cuda_without_runtime_errors() {
        if cuda_training_available() {
            // On a CUDA host this branch is a positive assertion:
            // explicit `cuda:0` must resolve successfully, and `auto`
            // must choose CUDA (not silently downgrade).
            assert_eq!(resolve_device("cuda:0"), Ok(Device::Cuda { index: 0 }));
            assert_eq!(resolve_device("auto"), Ok(Device::Cuda { index: 0 }));
        } else {
            // On a CPU-only host:
            // - explicit `cuda:0` MUST hard-fail (no silent fallback)
            // - explicit `cuda` MUST hard-fail (alias for `cuda:0`)
            // - `auto` MAY fall back to CPU (this is the documented
            //   safe-default escape hatch)
            let err = resolve_device("cuda:0").unwrap_err();
            assert!(matches!(err, DeviceError::CudaNotAvailable { .. }));
            let err = resolve_device("cuda").unwrap_err();
            assert!(matches!(err, DeviceError::CudaNotAvailable { .. }));
            assert_eq!(resolve_device("auto"), Ok(Device::Cpu));
        }
    }

    #[test]
    fn falsify_gputrain_002_cpu_always_resolves() {
        // `--device cpu` must always return `Device::Cpu`, regardless of
        // whether CUDA is available — it is an explicit opt-in to the
        // CPU path (for falsification parity runs, reproducibility, or
        // hosts without a usable GPU).
        assert_eq!(resolve_device("cpu"), Ok(Device::Cpu));
    }

    #[test]
    fn device_tag_round_trips() {
        assert_eq!(Device::Cpu.tag(), "cpu");
        assert_eq!(Device::Cuda { index: 0 }.tag(), "cuda:0");
        assert_eq!(Device::Cuda { index: 7 }.tag(), "cuda:7");
        assert_eq!(Device::Cuda { index: 15 }.tag(), "cuda:15");
    }

    #[test]
    fn device_is_cuda_discriminator() {
        assert!(!Device::Cpu.is_cuda());
        assert!(Device::Cuda { index: 0 }.is_cuda());
    }

    #[test]
    fn device_error_display_mentions_contract() {
        let invalid = DeviceError::InvalidSpec("bogus".into()).to_string();
        assert!(invalid.contains("INV-GPUTRAIN-001"));
        assert!(invalid.contains("bogus"));
        let unavail =
            DeviceError::CudaNotAvailable { requested: "cuda:0".into() }.to_string();
        assert!(unavail.contains("GATE-GPUTRAIN-002"));
        assert!(unavail.contains("cuda:0"));
    }
}
