#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! TUI Compute Monitor Falsification Tests (TRUENO-SPEC-020)
//!
//! Implementation of the 100-point Popperian Falsification Suite.
//! Section 1: Hardware Detection (H001-H010)

use std::process::Command;

/// H003: CPU core count matches physical reality
/// Hypothesis: "CPU core count matches physical reality"
/// Test: "Compare num_cpus::get() with /proc/cpuinfo (Linux)"
/// Falsification: "Count mismatch (hyperthreading confusion)"
#[test]
fn test_h003_cpu_core_count() {
    let detected_cores = num_cpus::get();

    // Linux-specific verification via lscpu or /proc/cpuinfo
    #[cfg(target_os = "linux")]
    {
        let output = Command::new("grep")
            .args(["-c", "^processor", "/proc/cpuinfo"])
            .output()
            .expect("Failed to execute grep");

        let proc_string = String::from_utf8(output.stdout).expect("Invalid UTF-8");
        let proc_cores: usize = proc_string.trim().parse().expect("Failed to parse core count");

        assert_eq!(
            detected_cores, proc_cores,
            "H003 FALSIFIED: num_cpus ({}) != /proc/cpuinfo ({})",
            detected_cores, proc_cores
        );
    }
}

/// H009: Unified device trait works for all backends (Mocked for CI)
/// Hypothesis: "Unified device trait works for all backends"
/// Test: "Call all ComputeDevice methods on CPU/NVIDIA/AMD"
#[test]
fn test_h009_unified_device_trait_structure() {
    // This test verifies the trait definition exists and is implementable
    // We define a MockDevice to prove the trait is sound

    use anyhow::Result;
    use trueno::monitor::{ComputeDevice, DeviceId, DeviceType};

    struct MockCpu;
    impl ComputeDevice for MockCpu {
        fn device_id(&self) -> DeviceId {
            DeviceId(0)
        }
        fn device_name(&self) -> &'static str {
            "Mock CPU"
        }
        fn device_type(&self) -> DeviceType {
            DeviceType::Cpu
        }

        fn compute_utilization(&self) -> Result<f64> {
            Ok(0.5)
        }
        fn compute_clock_mhz(&self) -> Result<u32> {
            Ok(3000)
        }
        fn compute_temperature_c(&self) -> Result<f64> {
            Ok(45.0)
        }
        fn compute_power_watts(&self) -> Result<f64> {
            Ok(65.0)
        }
        fn compute_power_limit_watts(&self) -> Result<f64> {
            Ok(105.0)
        }

        fn memory_used_bytes(&self) -> Result<u64> {
            Ok(1024)
        }
        fn memory_total_bytes(&self) -> Result<u64> {
            Ok(2048)
        }
        fn memory_bandwidth_gbps(&self) -> Result<f64> {
            Ok(50.0)
        }

        fn sm_count(&self) -> u32 {
            8
        }
        fn active_sm_count(&self) -> Result<u32> {
            Ok(4)
        }

        fn pcie_tx_bytes_per_sec(&self) -> Result<u64> {
            Ok(0)
        }
        fn pcie_rx_bytes_per_sec(&self) -> Result<u64> {
            Ok(0)
        }
        fn pcie_generation(&self) -> u8 {
            0
        }
        fn pcie_width(&self) -> u8 {
            0
        }
    }

    let device = MockCpu;
    assert_eq!(device.device_name(), "Mock CPU");
    assert!(device.compute_utilization().is_ok());
}
