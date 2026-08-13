//! Basic memory operation tests: zero-sized buffer, unaligned copy, OOM resilience

use super::*;

#[test]
fn test_zero_sized_buffer() {
    let ctx = CudaContext::new(0).expect("Context");

    // 0-sized allocation should either succeed (ptr=null/special) or fail gracefully
    // It should NOT panic or crash CUDA
    let buf_result = GpuBuffer::<f32>::new(&ctx, 0);

    // We accept either behavior, but it must be robust
    if let Ok(mut buf) = buf_result {
        assert_eq!(buf.len(), 0);
        // Copying 0 bytes should be a no-op
        let src: Vec<f32> = vec![];
        buf.copy_from_host(&src)
            .expect("Zero-byte copy should succeed");
    }
}

#[test]
fn test_unaligned_byte_copy() {
    let ctx = CudaContext::new(0).expect("Context");
    let len = 1024;
    let mut buf = GpuBuffer::<u8>::new(&ctx, len).expect("Alloc");

    let data: Vec<u8> = (0..len).map(|i| (i % 255) as u8).collect();
    buf.copy_from_host(&data).expect("Copy");

    let mut out = vec![0u8; len];
    buf.copy_to_host(&mut out).expect("Download");

    assert_eq!(data, out);
}

/// GPU-ORD-4: this test deliberately consumes the whole device, so it must not
/// share it — while it held ~24GB, `test_cuda_stress_memory_pressure` reported
/// "Memory exhausted after 2 chunks (512MB)" on a 24GB card. Its own leak check
/// now reads this thread's accounting instead of device-global free memory,
/// which a neighbour moves (it once failed with "268435456 bytes missing"
/// having leaked nothing).
#[test]
fn test_oom_resilience() {
    let _exclusive = device_memory_exclusive();
    let ctx = CudaContext::new(0).expect("Context");
    let outstanding_start = device_bytes_outstanding();

    // Allocate 1GB chunks until failure
    let mut allocations = Vec::new();
    let chunk_size = 1024 * 1024 * 1024 / 4; // 1GB of f32 (256M elements)

    // RTX 4090 has 24GB. 30 chunks * 1GB = 30GB -> Must OOM.
    // Limit to 20 to avoid freezing system if driver is aggressive
    let mut hit_oom = false;

    for i in 0..30 {
        match GpuBuffer::<f32>::new(&ctx, chunk_size) {
            Ok(buf) => allocations.push(buf),
            Err(GpuError::OutOfMemory { .. }) => {
                hit_oom = true;
                println!("Hit OOM at chunk {}", i);
                break;
            }
            Err(GpuError::MemoryAllocation(msg)) if msg.contains("OUT_OF_MEMORY") => {
                // Also acceptable - OOM wrapped in MemoryAllocation
                hit_oom = true;
                println!("Hit OOM (MemoryAllocation) at chunk {}", i);
                break;
            }
            Err(e) => panic!("Unexpected error during OOM stress: {:?}", e),
        }
    }

    // If we didn't hit OOM, we either have >30GB RAM or something is wrong
    // But we don't assert(hit_oom) to avoid flaky fails on 80GB A100s if running elsewhere

    // Drop all allocations
    drop(allocations);

    // Verify every chunk we managed to allocate was returned
    assert_eq!(
        device_bytes_outstanding(),
        outstanding_start,
        "Memory leak after allocating to OOM and dropping (hit_oom={hit_oom})"
    );
}
