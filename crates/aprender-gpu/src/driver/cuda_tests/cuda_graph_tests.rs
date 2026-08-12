//! CUDA graph capture, instantiate, and replay tests

use super::*;

#[test]
fn test_cuda_graph_new() {
    let _ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let graph = CudaGraph::new().expect("Graph creation MUST succeed");
    assert!(!graph.raw().is_null());
}

#[test]
fn test_cuda_graph_default() {
    let _ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let graph = CudaGraph::default();
    assert!(!graph.raw().is_null());
}

#[test]
fn test_cuda_graph_instantiate_empty() {
    let _ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let graph = CudaGraph::new().expect("Graph creation MUST succeed");
    let exec = graph.instantiate().expect("Graph instantiate MUST succeed");
    assert!(!exec.raw().is_null());
}

#[test]
fn test_cuda_graph_capture_and_replay() {
    let _capture_lock = super::capture_vs_ctx_sync();
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let stream = CudaStream::new(&ctx).expect("Stream creation MUST succeed");

    // GPU-ORD-4: ThreadLocal, not Global. `CaptureMode::Global` makes every
    // legacy-stream synchronous API illegal in the WHOLE PROCESS for as long as
    // the capture is open, and an unsafe call from any other thread invalidates
    // the capture. Neither this test nor the crate needs that: it captures its
    // own stream from its own thread. Leaving it Global made unrelated
    // concurrent tests fail with CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED (900) —
    // `test_context_synchronize` panicking on a plain `ctx.synchronize()`.
    stream
        .begin_capture(CaptureMode::ThreadLocal)
        .expect("Begin capture MUST succeed");

    // Simulate some work (empty capture is valid)
    // In a real scenario, we'd launch kernels here

    // End capture
    let graph = stream.end_capture().expect("End capture MUST succeed");
    assert!(!graph.raw().is_null());

    // Instantiate
    let exec = graph.instantiate().expect("Instantiate MUST succeed");
    assert!(!exec.raw().is_null());

    // Launch the graph 10 times to verify replay works
    for _ in 0..10 {
        stream
            .launch_graph(&exec)
            .expect("Graph launch MUST succeed");
    }

    stream.synchronize().expect("Final sync MUST succeed");
}

#[test]
fn test_cuda_graph_capture_modes() {
    let _capture_lock = super::capture_vs_ctx_sync();
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let stream = CudaStream::new(&ctx).expect("Stream creation MUST succeed");

    // GPU-ORD-4: `CaptureMode::Global` is deliberately absent. Opening a Global
    // capture makes every legacy-stream synchronous API illegal across the
    // whole process until it closes, and an "unsafe" call from any other thread
    // invalidates the capture — so this test cannot open one without breaking
    // whatever else is running. The value it added here was "the driver accepts
    // mode 0", and the enum-to-constant mapping is already asserted directly by
    // `driver::graph::tests` (`CaptureMode::Global.to_cuda_mode() == 0`), which
    // costs nothing and cannot disturb a neighbour.
    for mode in [CaptureMode::ThreadLocal, CaptureMode::Relaxed] {
        stream
            .begin_capture(mode)
            .expect("Begin capture MUST succeed");
        let graph = stream.end_capture().expect("End capture MUST succeed");
        let exec = graph.instantiate().expect("Instantiate MUST succeed");
        stream.launch_graph(&exec).expect("Launch MUST succeed");
        stream.synchronize().expect("Sync MUST succeed");
    }
}

#[test]
fn test_cuda_graph_with_kernel() {
    let _capture_lock = super::capture_vs_ctx_sync();
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let stream = CudaStream::new(&ctx).expect("Stream creation MUST succeed");

    // Create a simple add kernel PTX
    let ptx = r#".version 8.0
.target sm_80
.address_size 64

.visible .entry add_one(
    .param .u64 ptr,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;
    .reg .f32 %f<2>;

    ld.param.u64 %rd1, [ptr];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mad.lo.u32 %r2, %r3, 256, %r2;
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $done;

    cvt.u64.u32 %rd2, %r2;
    shl.b64 %rd3, %rd2, 2;
    add.u64 %rd3, %rd1, %rd3;

    ld.global.f32 %f1, [%rd3];
    add.f32 %f1, %f1, 1.0;
    st.global.f32 [%rd3], %f1;

$done:
    ret;
}
"#;

    let mut module = CudaModule::from_ptx(&ctx, ptx).expect("Module load MUST succeed");

    // Allocate buffer
    let data = vec![1.0f32; 256];
    let mut buffer = GpuBuffer::from_host(&ctx, &data).expect("Buffer MUST succeed");

    let n: u32 = 256;
    let config = LaunchConfig::linear(256, 256);

    // GPU-ORD-4: ThreadLocal, not Global. `CaptureMode::Global` makes every
    // legacy-stream synchronous API illegal in the WHOLE PROCESS for as long as
    // the capture is open, and an unsafe call from any other thread invalidates
    // the capture. Neither this test nor the crate needs that: it captures its
    // own stream from its own thread. Leaving it Global made unrelated
    // concurrent tests fail with CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED (900) —
    // `test_context_synchronize` panicking on a plain `ctx.synchronize()`.
    stream
        .begin_capture(CaptureMode::ThreadLocal)
        .expect("Begin capture MUST succeed");

    // Launch kernel (will be captured)
    let mut ptr_arg = buffer.as_ptr() as *mut c_void;
    let mut n_arg = (&n as *const u32) as *mut c_void;
    let mut args = [
        &mut ptr_arg as *mut *mut c_void as *mut c_void,
        &mut n_arg as *mut *mut c_void as *mut c_void,
    ];

    unsafe {
        stream
            .launch_kernel(&mut module, "add_one", &config, &mut args)
            .expect("Kernel launch MUST succeed");
    }

    // End capture
    let graph = stream.end_capture().expect("End capture MUST succeed");
    let exec = graph.instantiate().expect("Instantiate MUST succeed");

    // Replay the graph 100 times
    for _ in 0..100 {
        stream
            .launch_graph(&exec)
            .expect("Graph launch MUST succeed");
    }

    stream.synchronize().expect("Sync MUST succeed");

    // Verify the result: each element should be original + 100
    // Note: Graph capture does NOT execute the kernel - it only records it
    // So: initial 1.0 + (100 replays * 1.0) = 101.0
    let mut result = vec![0.0f32; 256];
    buffer
        .copy_to_host(&mut result)
        .expect("copy_to_host MUST succeed");
    for (i, &val) in result.iter().enumerate() {
        assert!(
            (val - 101.0).abs() < 0.01,
            "Element {} should be 101.0, got {}",
            i,
            val
        );
    }
}

#[test]
fn test_cuda_graph_drop_cleanup() {
    let _ctx = CudaContext::new(0).expect("Context creation MUST succeed");

    // Create and drop multiple graphs
    for _ in 0..10 {
        let graph = CudaGraph::new().expect("Graph creation MUST succeed");
        let _exec = graph.instantiate().expect("Instantiate MUST succeed");
        // Graph and exec are dropped here
    }
    // If we get here, cleanup is working
}
