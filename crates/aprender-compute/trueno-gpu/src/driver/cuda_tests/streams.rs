//! CUDA stream tests

use super::*;

#[test]
fn test_stream_creation() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let stream = CudaStream::new(&ctx).expect("Stream creation MUST succeed");
    assert!(!stream.raw().is_null());
}

#[test]
fn test_stream_synchronize() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let stream = CudaStream::new(&ctx).expect("Stream creation MUST succeed");
    stream.synchronize().expect("Stream sync MUST succeed");
}

#[test]
fn test_multiple_streams() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");

    let mut streams = Vec::new();
    for _ in 0..4 {
        streams.push(CudaStream::new(&ctx).expect("Stream creation MUST succeed"));
    }

    // Sync all streams
    for stream in &streams {
        stream.synchronize().expect("Stream sync MUST succeed");
    }
}

// ============================================================================
// Capture Mode Tests (Forces coverage of graph.rs CaptureMode)
// ============================================================================

#[test]
fn test_capture_mode_to_cuda_mode() {
    assert_eq!(CaptureMode::Global.to_cuda_mode(), 0);
    assert_eq!(CaptureMode::ThreadLocal.to_cuda_mode(), 1);
    assert_eq!(CaptureMode::Relaxed.to_cuda_mode(), 2);
}

#[test]
fn test_capture_mode_default() {
    assert_eq!(CaptureMode::default(), CaptureMode::Global);
}

#[test]
fn test_capture_mode_debug_and_clone() {
    let mode = CaptureMode::Global;
    let cloned = mode.clone();
    assert_eq!(mode, cloned);

    // Test Debug impl
    let debug_str = format!("{:?}", CaptureMode::ThreadLocal);
    assert!(debug_str.contains("ThreadLocal"));
}
