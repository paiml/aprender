//! Unit tests for `device` (extracted from `device.rs` to keep file-size invariant).
//!
//! Included via `#[cfg(test)] #[path = "device_tests.rs"] mod tests;` in the parent.

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
    let unavail = DeviceError::CudaNotAvailable { requested: "cuda:0".into() }.to_string();
    assert!(unavail.contains("GATE-GPUTRAIN-002"));
    assert!(unavail.contains("cuda:0"));
}

// ─── FALSIFY-GPUTRAIN-003: residency probe ─────────────────────────
//
// Binds contract `gpu-training-backend-v1` INV-GPUTRAIN-003 /
// GATE-GPUTRAIN-003. The three cases below discharge the probe at
// algorithm-level: empty trace, flat trace (all-baseline), and a
// trace with a real memory rise. The live evidence CSV at
// `evidence/gpu-training-backend/smoke-gpu-trace-2026-04-22.csv`
// exercises the third case against the lambda-labs RTX 4090 dispatch
// (PMAT-679, 903 MiB → 6400 MiB peak).
//
// The stronger --query-compute-apps per-PID proof is NOT yet wired —
// this probe accepts the weaker --query-gpu delta evidence; a full
// ACTIVE discharge still blocks on PID-level binding.

const MIN_RESIDENCY_DELTA_MIB: u64 = 1000;

#[test]
fn residency_probe_empty() {
    let samples = parse_nvidia_smi_gpu_trace("");
    assert!(samples.is_empty(), "empty CSV must yield empty trace");
    let err = assert_residency_discharge(&samples, MIN_RESIDENCY_DELTA_MIB).unwrap_err();
    assert!(err.contains("empty trace"));
}

#[test]
fn residency_probe_zero_mem() {
    // Baseline-only trace — all samples at the same low watermark.
    // Proves nvidia-smi works AND CUDA was visible, but training never
    // populated device memory. FAILURE case.
    let csv = "\
2026/04/22 07:42:57.308, 903 MiB, 23144 MiB, 5 %
2026/04/22 07:42:59.314, 903 MiB, 23144 MiB, 4 %
2026/04/22 07:43:01.315, 903 MiB, 23144 MiB, 3 %
";
    let samples = parse_nvidia_smi_gpu_trace(csv);
    assert_eq!(samples.len(), 3);
    assert_eq!(samples[0].used_mib, 903);
    let err = assert_residency_discharge(&samples, MIN_RESIDENCY_DELTA_MIB).unwrap_err();
    assert!(err.contains("peak memory did not rise"));
}

#[test]
fn residency_probe_nonzero_mem() {
    // Real smoke-run evidence shape: 903 MiB baseline spikes to 6400
    // MiB during training, then settles back near 6400 MiB for
    // checkpoint write. This is the PMAT-679 signal.
    let csv = "\
2026/04/22 07:42:57.308, 903 MiB, 23144 MiB, 39 %
2026/04/22 07:42:59.314, 903 MiB, 23144 MiB, 27 %
2026/04/22 07:43:03.315, 6400 MiB, 17647 MiB, 99 %
2026/04/22 07:43:05.315, 6400 MiB, 17647 MiB, 6 %
";
    let samples = parse_nvidia_smi_gpu_trace(csv);
    assert_eq!(samples.len(), 4);
    let peak = assert_residency_discharge(&samples, MIN_RESIDENCY_DELTA_MIB)
        .expect("5497 MiB delta far exceeds 1000 MiB threshold");
    assert_eq!(peak.used_mib, 6400, "peak must reflect the spike, not the baseline");
    // Utilization peak is a secondary signal — under ties on used_mib
    // (spike and post-spike rows both read 6400 MiB) the probe is
    // free to return either, so assert on the whole trace instead.
    let max_util = samples.iter().map(|s| s.util_pct).max().unwrap();
    assert_eq!(max_util, 99, "trace must record the 99%% training spike");
}

// ─── FALSIFY-GPUTRAIN-003 ACTIVE: per-PID residency probe ──────────
//
// Binds contract `gpu-training-backend-v1` INV-GPUTRAIN-003 at the
// level demanded by `blocks_active_promotion_on`: the training PID
// itself must appear in nvidia-smi's compute-apps table with
// non-trivial memory. Live evidence at
// `evidence/gpu-training-backend/smoke-compute-apps-2026-04-22.csv`
// (PMAT-680, pid 2467054 → 5492 MiB stable across the dispatch).

const PMAT_680_MIN_PID_MIB: u64 = 1000;

#[test]
fn pid_residency_probe_empty() {
    let samples = parse_nvidia_smi_compute_apps_csv("");
    assert!(samples.is_empty());
    let err =
        assert_pid_residency_discharge(&samples, 2_467_054, PMAT_680_MIN_PID_MIB).unwrap_err();
    assert!(err.contains("empty trace"));
}

#[test]
fn pid_residency_probe_no_matching_pid() {
    // Some other process owns the GPU but the training PID never
    // appears — e.g. CPU path silently taken, or the smoke script
    // captured compute-apps before `apr` started.
    let csv = "\
pid, process_name, used_gpu_memory [MiB]
9999, /usr/bin/other-gpu-app, 4096 MiB
";
    let samples = parse_nvidia_smi_compute_apps_csv(csv);
    assert_eq!(samples.len(), 1, "header row must be skipped");
    assert_eq!(samples[0].pid, 9999);
    let err =
        assert_pid_residency_discharge(&samples, 2_467_054, PMAT_680_MIN_PID_MIB).unwrap_err();
    assert!(err.contains("expected PID not present"));
}

#[test]
fn pid_residency_probe_matching_pid_with_mem() {
    // Real lambda-labs dispatch: `apr pretrain --device cuda:0` with
    // MODEL-2 370M from-scratch. PID 2467054 reports 5492 MiB steady-
    // state for the duration of training.
    let csv = "\
pid, process_name, used_gpu_memory [MiB]
2467054, /mnt/nvme-raid0/targets/aprender/release/apr, 5492 MiB
2467054, /mnt/nvme-raid0/targets/aprender/release/apr, 5492 MiB
2467054, /mnt/nvme-raid0/targets/aprender/release/apr, 5492 MiB
";
    let samples = parse_nvidia_smi_compute_apps_csv(csv);
    assert_eq!(samples.len(), 3, "3 data rows after header");
    let hit = assert_pid_residency_discharge(&samples, 2_467_054, PMAT_680_MIN_PID_MIB)
        .expect("5492 MiB far exceeds 1000 MiB PID threshold");
    assert_eq!(hit.pid, 2_467_054);
    assert_eq!(hit.used_mib, 5492);
    assert!(hit.process_name.ends_with("apr"));
}

#[test]
fn residency_probe_rejects_malformed_row() {
    // Parser must silently skip garbage rows (wrong column count,
    // bogus numbers) without panicking — the real smoke log mixes
    // ERR rows in when nvidia-smi transiently fails.
    let csv = "\
2026/04/22 07:42:57.308, 903 MiB, 23144 MiB, 39 %
BAD ROW, NO COMMAS
2026/04/22 07:43:03.315, 6400 MiB, 17647 MiB, 99 %
";
    let samples = parse_nvidia_smi_gpu_trace(csv);
    assert_eq!(samples.len(), 2, "malformed row must be dropped, not panic");
}
