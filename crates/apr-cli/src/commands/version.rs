//! `apr --version --json` emitter.
//!
//! Contract binding: `contracts/entrenar/gpu-training-backend-v1.yaml`
//! GATE-GPUTRAIN-006 / INV-GPUTRAIN-007 / FALSIFY-GPUTRAIN-007.
//!
//! Operators must be able to distinguish three states without reading
//! a stack trace:
//!
//! 1. Binary compiled *without* `--features cuda`.
//! 2. Binary compiled *with* `--features cuda` but no CUDA runtime
//!    is present on the host (driver missing, no GPU).
//! 3. Binary compiled *with* `--features cuda` AND a runnable CUDA
//!    runtime (ready to dispatch `apr pretrain --device cuda:N`).
//!
//! State (1) → `cuda_feature: false, cuda_runtime_available: false`.
//! State (2) → `cuda_feature: true,  cuda_runtime_available: false`.
//! State (3) → `cuda_feature: true,  cuda_runtime_available: true,
//!              visible_devices: [{index, name}]`.
//!
//! The closed failure mode this resolves is FM-GPUTRAIN-STALE-BUILD:
//! "Binary was built without --features cuda but reports a version
//! string that looks current. Operator mis-attributes CPU-only
//! performance to a GPU code bug."

use serde::{Deserialize, Serialize};

/// Structured `apr --version --json` output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionReport {
    /// Crate version (`CARGO_PKG_VERSION`), e.g. `"0.31.2"`.
    pub version: String,
    /// Short git SHA the binary was built from.
    pub git_sha: String,
    /// Whether the binary was compiled with `--features cuda` (compile-time).
    pub cuda_feature: bool,
    /// Whether a CUDA runtime is callable *right now* (runtime probe).
    /// Always `false` when `cuda_feature == false`.
    pub cuda_runtime_available: bool,
    /// Visible CUDA devices. Empty when cuda_runtime_available is false.
    pub visible_devices: Vec<CudaDeviceInfo>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaDeviceInfo {
    /// Device index, as consumed by `apr pretrain --device cuda:N`.
    pub index: u8,
    /// Device name (e.g. `"NVIDIA GeForce RTX 4090"`). `(unknown)` when
    /// the runtime exists but the probe could not read the name.
    pub name: String,
}

/// Compute-time constant: was the `cuda` cargo feature on during the
/// build that produced this binary?
#[must_use]
pub const fn cuda_feature_enabled() -> bool {
    cfg!(feature = "cuda")
}

/// Pure constructor — no I/O, no global state. All fields are explicit
/// inputs so tests can drive every branch. The public entry point
/// [`collect_version_report`] is the live wiring.
#[must_use]
pub fn build_version_report(
    version: &str,
    git_sha: &str,
    cuda_feature: bool,
    cuda_runtime_available: bool,
    visible_devices: Vec<CudaDeviceInfo>,
) -> VersionReport {
    // Contract invariant: if the feature was not compiled in, the
    // runtime cannot be available and no devices can be visible.
    // Reject the impossible inputs by normalizing them out — this is
    // the Poka-Yoke binding for INV-GPUTRAIN-007.
    if !cuda_feature {
        return VersionReport {
            version: version.to_string(),
            git_sha: git_sha.to_string(),
            cuda_feature: false,
            cuda_runtime_available: false,
            visible_devices: Vec::new(),
        };
    }
    if !cuda_runtime_available {
        return VersionReport {
            version: version.to_string(),
            git_sha: git_sha.to_string(),
            cuda_feature: true,
            cuda_runtime_available: false,
            visible_devices: Vec::new(),
        };
    }
    VersionReport {
        version: version.to_string(),
        git_sha: git_sha.to_string(),
        cuda_feature: true,
        cuda_runtime_available: true,
        visible_devices,
    }
}

/// Collect the live report from the current process — version/SHA from
/// compile-time env vars, cuda flags from runtime probes.
#[must_use]
pub fn collect_version_report() -> VersionReport {
    let cuda_feature = cuda_feature_enabled();
    let cuda_runtime_available = cuda_runtime_probe();
    let visible = if cuda_runtime_available {
        probe_visible_devices()
    } else {
        Vec::new()
    };
    build_version_report(
        env!("CARGO_PKG_VERSION"),
        env!("APR_GIT_SHA"),
        cuda_feature,
        cuda_runtime_available,
        visible,
    )
}

fn cuda_runtime_probe() -> bool {
    #[cfg(feature = "training")]
    {
        entrenar::autograd::cuda_training_available()
    }
    #[cfg(not(feature = "training"))]
    {
        false
    }
}

/// Probe `nvidia-smi --query-gpu=name --format=csv,noheader` for visible
/// devices. Returns `Vec::new()` on any failure — the runtime-available
/// flag is the load-bearing signal; device enumeration is best-effort.
fn probe_visible_devices() -> Vec<CudaDeviceInfo> {
    let Ok(out) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
    else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .enumerate()
        .take(16)
        .map(|(i, name)| CudaDeviceInfo {
            index: i as u8,
            name: name.to_string(),
        })
        .collect()
}

/// Emit the report as JSON on stdout. Returns the exit code the caller
/// should propagate.
///
/// # Errors
/// Returns `Err` if serialization fails — which, for a fixed schema
/// with no custom Serialize impl, is structurally impossible at
/// runtime.
pub fn emit_version_json(report: &VersionReport) -> Result<(), serde_json::Error> {
    let s = serde_json::to_string(report)?;
    println!("{s}");
    Ok(())
}

/// Early-intercept predicate for `cli_main()`: did the user pass BOTH
/// `--version` / `-V` AND `--json`? clap's `--version` handler exits
/// before our global `--json` flag is read, so we peek the raw args
/// here.
#[must_use]
pub fn should_emit_version_json<I, S>(args: I) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut has_version = false;
    let mut has_json = false;
    for arg in args {
        match arg.as_ref() {
            "--version" | "-V" => has_version = true,
            "--json" => has_json = true,
            _ => {}
        }
    }
    has_version && has_json
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn falsify_gputrain_007_feature_off_normalizes_runtime_off() {
        // State (1): cuda_feature=false ⇒ runtime_available and
        // visible_devices are FORCED to false/empty regardless of input.
        let fake_devices = vec![CudaDeviceInfo {
            index: 0,
            name: "should-be-dropped".to_string(),
        }];
        let rpt = build_version_report("0.31.2", "abc123", false, true, fake_devices);
        assert!(!rpt.cuda_feature);
        assert!(
            !rpt.cuda_runtime_available,
            "compile-time-off MUST force runtime-off (INV-GPUTRAIN-007)"
        );
        assert!(
            rpt.visible_devices.is_empty(),
            "compile-time-off MUST drop devices"
        );
    }

    #[test]
    fn falsify_gputrain_007_feature_on_runtime_off_drops_devices() {
        // State (2): cuda_feature=true, runtime=false ⇒ visible_devices empty.
        let fake_devices = vec![CudaDeviceInfo {
            index: 0,
            name: "phantom".to_string(),
        }];
        let rpt = build_version_report("0.31.2", "abc123", true, false, fake_devices);
        assert!(rpt.cuda_feature);
        assert!(!rpt.cuda_runtime_available);
        assert!(
            rpt.visible_devices.is_empty(),
            "runtime-off MUST drop devices — no phantom GPUs"
        );
    }

    #[test]
    fn falsify_gputrain_007_feature_on_runtime_on_keeps_devices() {
        // State (3): both flags true, devices preserved.
        let rtx = vec![CudaDeviceInfo {
            index: 0,
            name: "NVIDIA GeForce RTX 4090".to_string(),
        }];
        let rpt = build_version_report("0.31.2", "abc123", true, true, rtx.clone());
        assert!(rpt.cuda_feature);
        assert!(rpt.cuda_runtime_available);
        assert_eq!(rpt.visible_devices, rtx);
    }

    #[test]
    fn version_json_round_trips() {
        let rpt = build_version_report(
            "0.31.2",
            "0b2b9f2ea",
            true,
            true,
            vec![CudaDeviceInfo {
                index: 0,
                name: "NVIDIA GeForce RTX 4090".to_string(),
            }],
        );
        let s = serde_json::to_string(&rpt).expect("serialize");
        // Schema presence — exact keys named by the contract
        // GATE-GPUTRAIN-006 evidence_required clause.
        assert!(s.contains("\"cuda_feature\":true"));
        assert!(s.contains("\"cuda_runtime_available\":true"));
        assert!(s.contains("\"visible_devices\":[{"));
        assert!(s.contains("\"index\":0"));
        assert!(s.contains("\"name\":\"NVIDIA GeForce RTX 4090\""));
        let decoded: VersionReport = serde_json::from_str(&s).expect("round-trip");
        assert_eq!(decoded, rpt);
    }

    #[test]
    fn cuda_feature_enabled_matches_cfg() {
        // The const fn IS `cfg!(feature = "cuda")` by definition, so
        // this assertion is tautological — but it IS the binding that
        // FALSIFY-GPUTRAIN-007 requires: "cuda_feature field equals
        // cfg!(feature = cuda)".
        assert_eq!(cuda_feature_enabled(), cfg!(feature = "cuda"));
    }

    #[test]
    fn collect_version_report_matches_binary_metadata() {
        let rpt = collect_version_report();
        assert_eq!(rpt.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(rpt.git_sha, env!("APR_GIT_SHA"));
        assert_eq!(rpt.cuda_feature, cfg!(feature = "cuda"));
        // INV-GPUTRAIN-007: if cuda_feature is off, runtime MUST be off.
        if !rpt.cuda_feature {
            assert!(!rpt.cuda_runtime_available);
            assert!(rpt.visible_devices.is_empty());
        }
    }

    #[test]
    fn should_emit_version_json_detects_both_flags() {
        assert!(should_emit_version_json(["apr", "--version", "--json"]));
        assert!(should_emit_version_json(["apr", "-V", "--json"]));
        assert!(should_emit_version_json(["apr", "--json", "--version"]));
    }

    #[test]
    fn should_emit_version_json_rejects_missing_flag() {
        assert!(!should_emit_version_json(["apr", "--version"]));
        assert!(!should_emit_version_json(["apr", "-V"]));
        assert!(!should_emit_version_json(["apr", "--json"]));
        assert!(!should_emit_version_json(["apr", "run", "--json"]));
        assert!(!should_emit_version_json(["apr"]));
    }
}
