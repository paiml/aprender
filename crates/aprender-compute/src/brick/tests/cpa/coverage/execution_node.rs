use super::super::super::super::*;

// ========================
// Coverage Tests (C006-C011): ExecutionNode methods
// ========================

/// C006: ExecutionNode::name() all variants
#[test]
fn test_c006_execution_node_name() {
    let layer = ExecutionNode::Layer { index: 5 };
    assert_eq!(layer.name(), "Layer5");

    let brick = ExecutionNode::Brick { id: BrickId::GateProjection, timing_ns: 100, elements: 10 };
    assert_eq!(brick.name(), "GateProjection");

    let kernel = ExecutionNode::Kernel {
        name: "my_kernel".into(),
        ptx_hash: 0x123,
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    };
    assert_eq!(kernel.name(), "my_kernel");

    let func = ExecutionNode::Function {
        name: "my_func".into(),
        file: Some("test.rs".into()),
        line: Some(42),
    };
    assert_eq!(func.name(), "my_func");

    // Transfer variants
    let h2d = ExecutionNode::Transfer {
        src: "host".into(),
        dst: "device".into(),
        bytes: 1024,
        direction: TransferDirection::H2D,
        timing_ns: None,
    };
    assert_eq!(h2d.name(), "H2D:host->device");

    let d2h = ExecutionNode::Transfer {
        src: "device".into(),
        dst: "host".into(),
        bytes: 1024,
        direction: TransferDirection::D2H,
        timing_ns: None,
    };
    assert_eq!(d2h.name(), "D2H:device->host");

    let d2d = ExecutionNode::Transfer {
        src: "dev0".into(),
        dst: "dev1".into(),
        bytes: 1024,
        direction: TransferDirection::D2D,
        timing_ns: None,
    };
    assert_eq!(d2d.name(), "D2D:dev0->dev1");
}

/// C007: ExecutionNode::is_transfer()
#[test]
fn test_c007_execution_node_is_transfer() {
    let transfer = ExecutionNode::Transfer {
        src: "a".into(),
        dst: "b".into(),
        bytes: 100,
        direction: TransferDirection::H2D,
        timing_ns: None,
    };
    assert!(transfer.is_transfer());

    let brick = ExecutionNode::Brick { id: BrickId::RmsNorm, timing_ns: 100, elements: 10 };
    assert!(!brick.is_transfer());
}

/// C008: ExecutionNode::timing_ns() all variants
#[test]
fn test_c008_execution_node_timing_ns() {
    let brick = ExecutionNode::Brick { id: BrickId::RmsNorm, timing_ns: 12345, elements: 10 };
    assert_eq!(brick.timing_ns(), Some(12345));

    let kernel = ExecutionNode::Kernel {
        name: "k".into(),
        ptx_hash: 0,
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
        timing_ns: Some(67890),
        arithmetic_intensity: None,
        achieved_tflops: None,
    };
    assert_eq!(kernel.timing_ns(), Some(67890));

    let transfer = ExecutionNode::Transfer {
        src: "a".into(),
        dst: "b".into(),
        bytes: 100,
        direction: TransferDirection::H2D,
        timing_ns: Some(11111),
    };
    assert_eq!(transfer.timing_ns(), Some(11111));

    let layer = ExecutionNode::Layer { index: 0 };
    assert_eq!(layer.timing_ns(), None);

    let func = ExecutionNode::Function { name: "f".into(), file: None, line: None };
    assert_eq!(func.timing_ns(), None);
}

/// C009: ExecutionNode::ptx_hash()
#[test]
fn test_c009_execution_node_ptx_hash() {
    let kernel = ExecutionNode::Kernel {
        name: "k".into(),
        ptx_hash: 0xDEADBEEF,
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    };
    assert_eq!(kernel.ptx_hash(), Some(0xDEADBEEF));

    let brick = ExecutionNode::Brick { id: BrickId::RmsNorm, timing_ns: 100, elements: 10 };
    assert_eq!(brick.ptx_hash(), None);
}

/// C010: ExecutionNode::arithmetic_intensity() and achieved_tflops()
#[test]
fn test_c010_execution_node_roofline_accessors() {
    let kernel = ExecutionNode::Kernel {
        name: "k".into(),
        ptx_hash: 0,
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
        timing_ns: Some(1000),
        arithmetic_intensity: Some(50.0),
        achieved_tflops: Some(10.5),
    };
    assert_eq!(kernel.arithmetic_intensity(), Some(50.0));
    assert_eq!(kernel.achieved_tflops(), Some(10.5));

    let brick = ExecutionNode::Brick { id: BrickId::RmsNorm, timing_ns: 100, elements: 10 };
    assert_eq!(brick.arithmetic_intensity(), None);
    assert_eq!(brick.achieved_tflops(), None);
}

/// C011: ExecutionNode::transfer_bytes()
#[test]
fn test_c011_execution_node_transfer_bytes() {
    let transfer = ExecutionNode::Transfer {
        src: "a".into(),
        dst: "b".into(),
        bytes: 1024 * 1024,
        direction: TransferDirection::H2D,
        timing_ns: None,
    };
    assert_eq!(transfer.transfer_bytes(), Some(1024 * 1024));

    let brick = ExecutionNode::Brick { id: BrickId::RmsNorm, timing_ns: 100, elements: 10 };
    assert_eq!(brick.transfer_bytes(), None);
}
