use super::*;

// =========================================================================
// H031: PCIe Bandwidth Tests
// =========================================================================

#[test]
fn h031_pcie_bandwidth_gen4_x16() {
    // PCIe 4.0 x16 = ~31.5 GB/s
    let bw = DataFlowMetrics::calculate_pcie_bandwidth(4, 16);
    assert!((bw - 31.5).abs() < 0.5, "Expected ~31.5 GB/s, got {}", bw);
}

#[test]
fn h031_pcie_bandwidth_gen5_x16() {
    // PCIe 5.0 x16 = ~63 GB/s
    let bw = DataFlowMetrics::calculate_pcie_bandwidth(5, 16);
    assert!((bw - 63.0).abs() < 1.0, "Expected ~63 GB/s, got {}", bw);
}

#[test]
fn h031_pcie_bandwidth_gen3_x16() {
    // PCIe 3.0 x16 = ~15.75 GB/s
    let bw = DataFlowMetrics::calculate_pcie_bandwidth(3, 16);
    assert!((bw - 15.75).abs() < 0.5, "Expected ~15.75 GB/s, got {}", bw);
}

#[test]
fn h031_pcie_bandwidth_gen4_x8() {
    // PCIe 4.0 x8 = ~15.75 GB/s
    let bw = DataFlowMetrics::calculate_pcie_bandwidth(4, 8);
    assert!((bw - 15.75).abs() < 0.5, "Expected ~15.75 GB/s, got {}", bw);
}

#[test]
fn h031_pcie_bandwidth_unknown_gen() {
    let bw = DataFlowMetrics::calculate_pcie_bandwidth(99, 16);
    assert_eq!(bw, 0.0);
}

// =========================================================================
// H032: Data Flow Metrics Tests
// =========================================================================

#[test]
fn h032_data_flow_default() {
    let metrics = DataFlowMetrics::default();
    assert_eq!(metrics.pcie_generation, 4);
    assert_eq!(metrics.pcie_width, 16);
    assert!(metrics.pcie_theoretical_gbps > 30.0);
}

#[test]
fn h032_data_flow_set_pcie_config() {
    let mut metrics = DataFlowMetrics::new();
    metrics.set_pcie_config(5, 16);

    assert_eq!(metrics.pcie_generation, 5);
    assert_eq!(metrics.pcie_width, 16);
    assert!(metrics.pcie_theoretical_gbps > 60.0);
}

#[test]
fn h032_data_flow_tx_utilization() {
    let mut metrics = DataFlowMetrics::new();
    metrics.pcie_theoretical_gbps = 31.5;
    metrics.pcie_tx_gbps = 15.75;

    assert!((metrics.pcie_tx_utilization_pct() - 50.0).abs() < 1.0);
}

#[test]
fn h032_data_flow_rx_utilization() {
    let mut metrics = DataFlowMetrics::new();
    metrics.pcie_theoretical_gbps = 31.5;
    metrics.pcie_rx_gbps = 7.875;

    assert!((metrics.pcie_rx_utilization_pct() - 25.0).abs() < 1.0);
}

#[test]
fn h032_data_flow_utilization_zero_theoretical() {
    let mut metrics = DataFlowMetrics::new();
    metrics.pcie_theoretical_gbps = 0.0;

    assert_eq!(metrics.pcie_tx_utilization_pct(), 0.0);
    assert_eq!(metrics.pcie_rx_utilization_pct(), 0.0);
}

// =========================================================================
// H033: Transfer Tracking Tests
// =========================================================================

#[test]
fn h033_transfer_id_unique() {
    let id1 = TransferId::new();
    let id2 = TransferId::new();
    assert_ne!(id1, id2);
}

#[test]
fn h033_transfer_new() {
    let transfer = Transfer::new(
        TransferDirection::HostToDevice,
        MemoryLocation::SystemRam,
        MemoryLocation::GpuVram(DeviceId::nvidia(0)),
        1024 * 1024,
    );

    assert_eq!(transfer.direction, TransferDirection::HostToDevice);
    assert_eq!(transfer.size_bytes, 1024 * 1024);
    assert_eq!(transfer.transferred_bytes, 0);
    assert_eq!(transfer.status, TransferStatus::Pending);
}

#[test]
fn h033_transfer_h2d_helper() {
    let transfer = Transfer::host_to_device(1024 * 1024, DeviceId::nvidia(0));
    assert_eq!(transfer.direction, TransferDirection::HostToDevice);
    assert_eq!(transfer.source, MemoryLocation::SystemRam);
}

#[test]
fn h033_transfer_d2h_helper() {
    let transfer = Transfer::device_to_host(1024 * 1024, DeviceId::nvidia(0));
    assert_eq!(transfer.direction, TransferDirection::DeviceToHost);
    assert_eq!(transfer.destination, MemoryLocation::SystemRam);
}

#[test]
fn h033_transfer_with_label() {
    let transfer = Transfer::host_to_device(1024, DeviceId::nvidia(0)).with_label("tensor_a");
    assert_eq!(transfer.label, "tensor_a");
}

#[test]
fn h033_transfer_progress() {
    let mut transfer = Transfer::host_to_device(1000, DeviceId::nvidia(0));
    assert_eq!(transfer.progress_pct(), 0.0);

    transfer.update_progress(500);
    assert!((transfer.progress_pct() - 50.0).abs() < 0.01);
    assert_eq!(transfer.status, TransferStatus::InProgress);

    transfer.complete();
    assert_eq!(transfer.progress_pct(), 100.0);
    assert_eq!(transfer.status, TransferStatus::Completed);
}

#[test]
fn h033_transfer_progress_zero_size() {
    let transfer = Transfer::host_to_device(0, DeviceId::nvidia(0));
    assert_eq!(transfer.progress_pct(), 100.0);
}

#[test]
fn h033_transfer_elapsed() {
    let transfer = Transfer::host_to_device(1024, DeviceId::nvidia(0));
    std::thread::sleep(std::time::Duration::from_millis(10));
    assert!(transfer.elapsed_ms() >= 10.0);
}

#[test]
fn h033_transfer_bandwidth() {
    let mut transfer = Transfer::host_to_device(1_000_000_000, DeviceId::nvidia(0)); // 1 GB
    std::thread::sleep(std::time::Duration::from_millis(100));
    transfer.transferred_bytes = 100_000_000; // 100 MB in ~100ms = ~1 GB/s

    let bw = transfer.bandwidth_gbps();
    // Should be roughly 1 GB/s (but timing is imprecise in tests)
    assert!(bw > 0.5 && bw < 2.0, "Bandwidth {} GB/s unexpected", bw);
}

// =========================================================================
// H034: Transfer Direction Display Tests
// =========================================================================

#[test]
fn h034_transfer_direction_display() {
    assert_eq!(format!("{}", TransferDirection::HostToDevice), "H→D");
    assert_eq!(format!("{}", TransferDirection::DeviceToHost), "D→H");
    assert_eq!(format!("{}", TransferDirection::DeviceToDevice), "D→D");
    assert_eq!(format!("{}", TransferDirection::PeerToPeer), "P2P");
}

// =========================================================================
// H035: Memory Location Display Tests
// =========================================================================

#[test]
fn h035_memory_location_display() {
    assert_eq!(format!("{}", MemoryLocation::SystemRam), "RAM");
    assert_eq!(format!("{}", MemoryLocation::PinnedMemory), "Pinned");
    assert_eq!(format!("{}", MemoryLocation::UnifiedMemory), "Unified");
    assert!(format!("{}", MemoryLocation::GpuVram(DeviceId::nvidia(0))).contains("NVIDIA"));
}

// =========================================================================
// H036: History Tracking Tests
// =========================================================================

#[test]
fn h036_history_update() {
    let mut metrics = DataFlowMetrics::new();

    for i in 0..100 {
        metrics.pcie_tx_gbps = i as f64;
        metrics.pcie_rx_gbps = i as f64 * 0.5;
        metrics.memory_bus_utilization_pct = i as f64;
        metrics.update_history();
    }

    assert_eq!(
        metrics.pcie_tx_history.len(),
        DataFlowMetrics::MAX_HISTORY_POINTS
    );
    assert_eq!(
        metrics.pcie_rx_history.len(),
        DataFlowMetrics::MAX_HISTORY_POINTS
    );
    assert_eq!(
        metrics.memory_bus_history.len(),
        DataFlowMetrics::MAX_HISTORY_POINTS
    );
}

// =========================================================================
// H037: Transfer Management Tests
// =========================================================================

#[test]
fn h037_start_and_complete_transfer() {
    let mut metrics = DataFlowMetrics::new();
    let transfer = Transfer::host_to_device(1024, DeviceId::nvidia(0));
    let id = transfer.id;

    metrics.start_transfer(transfer);
    assert_eq!(metrics.active_transfers.len(), 1);

    metrics.complete_transfer(id);
    assert_eq!(metrics.active_transfers.len(), 0);
    assert_eq!(metrics.completed_transfers.len(), 1);
}

#[test]
fn h037_completed_transfer_limit() {
    let mut metrics = DataFlowMetrics::new();

    for _ in 0..150 {
        let transfer = Transfer::host_to_device(1024, DeviceId::nvidia(0));
        let id = transfer.id;
        metrics.start_transfer(transfer);
        metrics.complete_transfer(id);
    }

    assert_eq!(
        metrics.completed_transfers.len(),
        DataFlowMetrics::MAX_COMPLETED_TRANSFERS
    );
}

#[test]
fn h037_bytes_in_flight() {
    let mut metrics = DataFlowMetrics::new();

    let mut t1 = Transfer::host_to_device(1000, DeviceId::nvidia(0));
    t1.transferred_bytes = 400;

    let mut t2 = Transfer::host_to_device(2000, DeviceId::nvidia(0));
    t2.transferred_bytes = 500;

    metrics.start_transfer(t1);
    metrics.start_transfer(t2);

    // (1000-400) + (2000-500) = 600 + 1500 = 2100
    assert_eq!(metrics.bytes_in_flight(), 2100);
}

// =========================================================================
// H038: Pinned Memory Tests
// =========================================================================

#[test]
fn h038_pinned_memory_utilization() {
    let mut metrics = DataFlowMetrics::new();
    metrics.pinned_memory_used_bytes = 512 * 1024 * 1024; // 512 MB
    metrics.pinned_memory_total_bytes = 1024 * 1024 * 1024; // 1 GB

    assert!((metrics.pinned_memory_utilization_pct() - 50.0).abs() < 0.01);
}

#[test]
fn h038_pinned_memory_utilization_zero_total() {
    let metrics = DataFlowMetrics::new();
    assert_eq!(metrics.pinned_memory_utilization_pct(), 0.0);
}
