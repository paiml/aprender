use super::*;

// =========================================================================
// H021: Compute Metrics Tests
// =========================================================================

#[test]
fn h021_compute_metrics_default() {
    let metrics = ComputeMetrics::default();
    assert!(metrics.devices.is_empty());
    assert!(metrics.active_kernels.is_empty());
    assert_eq!(metrics.input_latency_ms, 0.0);
}

#[test]
fn h021_compute_metrics_total_latency() {
    let mut metrics = ComputeMetrics::default();
    metrics.input_latency_ms = 1.0;
    metrics.compute_latency_ms = 5.0;
    metrics.reduce_latency_ms = 0.5;
    metrics.output_latency_ms = 1.0;

    assert!((metrics.total_latency_ms() - 7.5).abs() < 0.01);
}

#[test]
fn h021_compute_metrics_throughput() {
    let mut metrics = ComputeMetrics::default();
    metrics.input_latency_ms = 10.0; // 10ms total = 100 ops/s

    assert!((metrics.throughput_ops() - 100.0).abs() < 1.0);
}

#[test]
fn h021_compute_metrics_throughput_zero_latency() {
    let metrics = ComputeMetrics::default();
    assert_eq!(metrics.throughput_ops(), 0.0);
}

#[test]
fn h021_compute_metrics_efficiency() {
    let mut metrics = ComputeMetrics::default();
    metrics.flops_achieved = 500.0;
    metrics.flops_theoretical = 1000.0;

    assert!((metrics.efficiency_percent() - 50.0).abs() < 0.01);
}

#[test]
fn h021_compute_metrics_efficiency_zero_theoretical() {
    let metrics = ComputeMetrics::default();
    assert_eq!(metrics.efficiency_percent(), 0.0);
}

// =========================================================================
// H022: Device Compute Metrics Tests
// =========================================================================

#[test]
fn h022_device_metrics_new() {
    let metrics = DeviceComputeMetrics::new(DeviceId::nvidia(0));
    assert_eq!(metrics.device_id, DeviceId::nvidia(0));
    assert_eq!(metrics.utilization_pct, 0.0);
}

#[test]
fn h022_device_metrics_update_history() {
    let mut metrics = DeviceComputeMetrics::new(DeviceId::nvidia(0));

    for i in 0..100 {
        metrics.update_utilization(i as f64);
    }

    assert_eq!(metrics.history.len(), DeviceComputeMetrics::MAX_HISTORY_POINTS);
    // Last value should be 99
    assert!((metrics.history.back().expect("test") - 99.0).abs() < 0.01);
}

#[test]
fn h022_device_metrics_warp_occupancy() {
    let mut metrics = DeviceComputeMetrics::new(DeviceId::nvidia(0));
    metrics.warps_active = 48;
    metrics.warps_max = 64;

    assert!((metrics.warp_occupancy_pct() - 75.0).abs() < 0.01);
}

#[test]
fn h022_device_metrics_warp_occupancy_zero() {
    let metrics = DeviceComputeMetrics::new(DeviceId::nvidia(0));
    assert_eq!(metrics.warp_occupancy_pct(), 0.0);
}

#[test]
fn h022_device_metrics_clock_ratio() {
    let mut metrics = DeviceComputeMetrics::new(DeviceId::nvidia(0));
    metrics.clock_mhz = 1800;
    metrics.clock_max_mhz = 2400;

    assert!((metrics.clock_ratio() - 0.75).abs() < 0.01);
}

#[test]
fn h022_device_metrics_power_ratio() {
    let mut metrics = DeviceComputeMetrics::new(DeviceId::nvidia(0));
    metrics.power_watts = 300.0;
    metrics.power_limit_watts = 450.0;

    assert!((metrics.power_ratio() - 0.666).abs() < 0.01);
}

#[test]
fn h022_device_metrics_throttling() {
    let mut metrics = DeviceComputeMetrics::new(DeviceId::nvidia(0));
    assert!(!metrics.is_throttling());

    metrics.throttle_reason = Some(ThrottleReason::Thermal);
    assert!(metrics.is_throttling());

    metrics.throttle_reason = Some(ThrottleReason::None);
    assert!(!metrics.is_throttling());
}

// =========================================================================
// H023: Kernel Execution Tests
// =========================================================================

#[test]
fn h023_kernel_execution_new() {
    let kernel = KernelExecution::new("test_kernel", DeviceId::nvidia(0));
    assert_eq!(kernel.name, "test_kernel");
    assert_eq!(kernel.status, KernelStatus::Pending);
}

#[test]
fn h023_kernel_execution_builder() {
    let kernel = KernelExecution::new("gemm", DeviceId::nvidia(0))
        .with_dims((128, 128, 1), (16, 16, 1))
        .with_shared_mem(4096)
        .with_registers(32);

    assert_eq!(kernel.grid_dim, (128, 128, 1));
    assert_eq!(kernel.block_dim, (16, 16, 1));
    assert_eq!(kernel.shared_mem_bytes, 4096);
    assert_eq!(kernel.registers_per_thread, 32);
}

#[test]
fn h023_kernel_execution_total_threads() {
    let kernel =
        KernelExecution::new("test", DeviceId::nvidia(0)).with_dims((128, 64, 1), (16, 16, 1));

    // 128 * 64 * 1 blocks * 16 * 16 * 1 threads/block = 2,097,152
    assert_eq!(kernel.total_threads(), 128 * 64 * 16 * 16);
}

#[test]
fn h023_kernel_execution_total_blocks() {
    let kernel =
        KernelExecution::new("test", DeviceId::nvidia(0)).with_dims((128, 64, 2), (16, 16, 1));

    assert_eq!(kernel.total_blocks(), 128 * 64 * 2);
}

#[test]
fn h023_kernel_execution_threads_per_block() {
    let kernel =
        KernelExecution::new("test", DeviceId::nvidia(0)).with_dims((1, 1, 1), (16, 16, 4));

    assert_eq!(kernel.threads_per_block(), 16 * 16 * 4);
}

#[test]
fn h023_kernel_execution_lifecycle() {
    let mut kernel = KernelExecution::new("test", DeviceId::nvidia(0));
    assert_eq!(kernel.status, KernelStatus::Pending);
    assert_eq!(kernel.progress_pct(), 0.0);

    kernel.start();
    assert_eq!(kernel.status, KernelStatus::Running);

    std::thread::sleep(std::time::Duration::from_millis(10));
    kernel.update_elapsed();
    assert!(kernel.elapsed_ms > 0.0);

    kernel.complete();
    assert_eq!(kernel.status, KernelStatus::Completed);
    assert_eq!(kernel.progress_pct(), 100.0);
}

// =========================================================================
// H024: FLOPS Calculation Tests
// =========================================================================

#[test]
fn h024_gemm_flops() {
    // GEMM 1024x1024x1024 = 2 * 1024^3 = 2,147,483,648 FLOPs
    let flops = gemm_flops(1024, 1024, 1024);
    assert!((flops - 2.0 * 1024.0 * 1024.0 * 1024.0).abs() < 1.0);
}

#[test]
fn h024_achieved_gflops() {
    // 2e9 FLOPs in 1 second = 2 GFLOPS
    let gflops = achieved_gflops(2e9, Duration::from_secs(1));
    assert!((gflops - 2.0).abs() < 0.01);
}

#[test]
fn h024_achieved_gflops_zero_time() {
    let gflops = achieved_gflops(1e9, Duration::ZERO);
    assert_eq!(gflops, 0.0);
}

#[test]
fn h024_compute_efficiency_calculation() {
    // 75 GFLOPS achieved out of 100 GFLOPS theoretical = 75%
    let eff = compute_efficiency(75.0, 100.0);
    assert!((eff - 75.0).abs() < 0.01);
}

#[test]
fn h024_compute_efficiency_zero_theoretical() {
    let eff = compute_efficiency(100.0, 0.0);
    assert_eq!(eff, 0.0);
}

// =========================================================================
// H025: Kernel Tracking in Metrics
// =========================================================================

#[test]
fn h025_metrics_track_kernel() {
    let mut metrics = ComputeMetrics::new();
    let kernel = KernelExecution::new("test", DeviceId::nvidia(0));

    metrics.track_kernel(kernel);
    assert_eq!(metrics.active_kernels.len(), 1);
}

#[test]
fn h025_metrics_clear_completed() {
    let mut metrics = ComputeMetrics::new();

    let mut k1 = KernelExecution::new("running", DeviceId::nvidia(0));
    k1.status = KernelStatus::Running;

    let mut k2 = KernelExecution::new("completed", DeviceId::nvidia(0));
    k2.status = KernelStatus::Completed;

    metrics.track_kernel(k1);
    metrics.track_kernel(k2);

    assert_eq!(metrics.active_kernels.len(), 2);

    metrics.clear_completed_kernels();
    assert_eq!(metrics.active_kernels.len(), 1);
    assert_eq!(metrics.active_kernels[0].name, "running");
}
