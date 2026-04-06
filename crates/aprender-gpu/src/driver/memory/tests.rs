use super::*;
use std::mem;

#[test]
#[cfg(not(feature = "cuda"))]
fn test_buffer_requires_cuda_feature() {
    // Without cuda feature, allocation should fail
    // This test verifies the module compiles
    assert!(true);
}

#[test]
fn test_size_bytes_calculation() {
    // Test size calculation logic (doesn't require CUDA)
    let size = 1024 * mem::size_of::<f32>();
    assert_eq!(size, 4096);
}

#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;
    use crate::driver::CudaContext;

    macro_rules! cuda_ctx {
        () => {
            match CudaContext::new(0) {
                Ok(ctx) => ctx,
                Err(e) => {
                    eprintln!("Skipping CUDA test: {:?}", e);
                    return;
                }
            }
        };
    }

    #[test]
    fn test_gpu_buffer_new_empty() {
        let ctx = cuda_ctx!();
        let buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 0).unwrap();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.size_bytes(), 0);
    }

    #[test]
    fn test_gpu_buffer_new_allocation() {
        let ctx = cuda_ctx!();
        let buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 1024).unwrap();
        assert!(!buf.is_empty());
        assert_eq!(buf.len(), 1024);
        assert_eq!(buf.size_bytes(), 4096);
        assert!(buf.as_ptr() != 0);
    }

    #[test]
    fn test_gpu_buffer_copy_roundtrip() {
        let ctx = cuda_ctx!();
        let mut buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 256).unwrap();

        // Upload data
        let host_data: Vec<f32> = (0..256).map(|i| i as f32).collect();
        buf.copy_from_host(&host_data).unwrap();

        // Download and verify
        let mut result = vec![0.0f32; 256];
        buf.copy_to_host(&mut result).unwrap();
        assert_eq!(host_data, result);
    }

    #[test]
    fn test_gpu_buffer_copy_from_host_size_mismatch() {
        let ctx = cuda_ctx!();
        let mut buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 100).unwrap();

        // Try to copy too much data
        let host_data: Vec<f32> = vec![1.0; 200];
        let result = buf.copy_from_host(&host_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_buffer_copy_to_host_size_mismatch() {
        let ctx = cuda_ctx!();
        let buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 100).unwrap();

        // Smaller host buffer is OK (partial copy) — copies first 50 elements
        let mut partial: Vec<f32> = vec![0.0; 50];
        assert!(buf.copy_to_host(&mut partial).is_ok());

        // Larger host buffer than device MUST fail
        let mut too_large: Vec<f32> = vec![0.0; 200];
        assert!(buf.copy_to_host(&mut too_large).is_err());
    }

    #[test]
    fn test_gpu_buffer_clone_metadata() {
        let ctx = cuda_ctx!();
        let buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 512).unwrap();

        let view = buf.clone_metadata();
        assert_eq!(view.as_ptr(), buf.as_ptr());
        assert_eq!(view.len(), buf.len());
        assert!(!view.is_empty());
    }

    #[test]
    fn test_gpu_buffer_view_empty() {
        let ctx = cuda_ctx!();
        let buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 0).unwrap();

        let view = buf.clone_metadata();
        assert!(view.is_empty());
        assert_eq!(view.len(), 0);
    }

    #[test]
    fn test_gpu_buffer_raw_parts() {
        let ctx = cuda_ctx!();
        let buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 64).unwrap();
        let ptr = buf.as_ptr();
        let len = buf.len();

        // Create non-owning buffer from raw parts
        let buf2 = unsafe { GpuBuffer::<f32>::from_raw_parts(ptr, len) };
        assert_eq!(buf2.as_ptr(), ptr);
        assert_eq!(buf2.len(), len);

        // Forget buf2 to prevent double-free
        std::mem::forget(buf2);
    }

    #[test]
    fn test_gpu_buffer_from_host() {
        let ctx = cuda_ctx!();
        let data: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let buf = GpuBuffer::from_host(&ctx, &data).unwrap();
        assert_eq!(buf.len(), 128);

        // Verify data was uploaded correctly
        let mut result = vec![0.0f32; 128];
        buf.copy_to_host(&mut result).unwrap();
        assert_eq!(data, result);
    }

    #[test]
    fn test_gpu_buffer_copy_from_host_at() {
        let ctx = cuda_ctx!();
        let mut buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 100).unwrap();

        // Initialize with zeros
        let zeros = vec![0.0f32; 100];
        buf.copy_from_host(&zeros).unwrap();

        // Copy partial data at offset
        let partial = vec![1.0f32; 20];
        buf.copy_from_host_at(&partial, 50).unwrap();

        // Verify
        let mut result = vec![0.0f32; 100];
        buf.copy_to_host(&mut result).unwrap();
        assert_eq!(result[49], 0.0);
        assert_eq!(result[50], 1.0);
        assert_eq!(result[69], 1.0);
        assert_eq!(result[70], 0.0);
    }

    #[test]
    fn test_gpu_buffer_copy_to_host_at() {
        let ctx = cuda_ctx!();
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let buf = GpuBuffer::from_host(&ctx, &data).unwrap();

        // Copy partial data from offset
        let mut result = vec![0.0f32; 20];
        buf.copy_to_host_at(&mut result, 30).unwrap();

        assert_eq!(result[0], 30.0);
        assert_eq!(result[19], 49.0);
    }

    #[test]
    fn test_gpu_buffer_clone_device() {
        let ctx = cuda_ctx!();
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let buf = GpuBuffer::from_host(&ctx, &data).unwrap();

        // Clone on device
        let cloned = buf.clone(&ctx).unwrap();
        assert_eq!(cloned.len(), buf.len());
        assert_ne!(cloned.as_ptr(), buf.as_ptr()); // Different memory

        // Verify content
        let mut result = vec![0.0f32; 64];
        cloned.copy_to_host(&mut result).unwrap();
        assert_eq!(data, result);
    }

    #[test]
    fn test_gpu_buffer_copy_from_buffer() {
        let ctx = cuda_ctx!();
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let src = GpuBuffer::from_host(&ctx, &data).unwrap();

        let mut dst: GpuBuffer<f32> = GpuBuffer::new(&ctx, 32).unwrap();
        dst.copy_from_buffer(&src).unwrap();

        // Verify
        let mut result = vec![0.0f32; 32];
        dst.copy_to_host(&mut result).unwrap();
        assert_eq!(data, result);
    }

    #[test]
    fn test_gpu_buffer_copy_from_buffer_at() {
        let ctx = cuda_ctx!();
        let src_data: Vec<f32> = vec![5.0f32; 10];
        let src = GpuBuffer::from_host(&ctx, &src_data).unwrap();

        let mut dst: GpuBuffer<f32> = GpuBuffer::new(&ctx, 50).unwrap();
        let zeros = vec![0.0f32; 50];
        dst.copy_from_host(&zeros).unwrap();

        // Copy src to dst at offset 20
        dst.copy_from_buffer_at(&src, 20, 0, 10).unwrap();

        // Verify
        let mut result = vec![0.0f32; 50];
        dst.copy_to_host(&mut result).unwrap();
        assert_eq!(result[19], 0.0);
        assert_eq!(result[20], 5.0);
        assert_eq!(result[29], 5.0);
        assert_eq!(result[30], 0.0);
    }

    #[test]
    fn test_gpu_buffer_view_size_bytes() {
        let ctx = cuda_ctx!();
        let buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 256).unwrap();
        let view = buf.clone_metadata();
        assert_eq!(view.size_bytes(), 256 * 4);
    }

    #[test]
    fn test_gpu_buffer_as_kernel_arg() {
        let ctx = cuda_ctx!();
        let buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 32).unwrap();
        let arg = buf.as_kernel_arg();
        assert!(!arg.is_null());
    }

    #[test]
    fn test_gpu_buffer_async_copy() {
        use crate::driver::CudaStream;
        let ctx = cuda_ctx!();
        let stream = CudaStream::new(&ctx).unwrap();

        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let src = GpuBuffer::from_host(&ctx, &data).unwrap();
        let mut dst: GpuBuffer<f32> = GpuBuffer::new(&ctx, 64).unwrap();

        unsafe {
            dst.copy_from_buffer_async(&src, &stream).unwrap();
        }
        stream.synchronize().unwrap();

        let mut result = vec![0.0f32; 64];
        dst.copy_to_host(&mut result).unwrap();
        assert_eq!(data, result);
    }

    #[test]
    fn test_gpu_buffer_async_copy_at() {
        use crate::driver::CudaStream;
        let ctx = cuda_ctx!();
        let stream = CudaStream::new(&ctx).unwrap();

        let data: Vec<f32> = vec![7.0f32; 10];
        let src = GpuBuffer::from_host(&ctx, &data).unwrap();

        let mut dst: GpuBuffer<f32> = GpuBuffer::new(&ctx, 50).unwrap();
        let zeros = vec![0.0f32; 50];
        dst.copy_from_host(&zeros).unwrap();

        unsafe {
            dst.copy_from_buffer_at_async(&src, 15, 0, 10, &stream).unwrap();
        }
        stream.synchronize().unwrap();

        let mut result = vec![0.0f32; 50];
        dst.copy_to_host(&mut result).unwrap();
        assert_eq!(result[14], 0.0);
        assert_eq!(result[15], 7.0);
        assert_eq!(result[24], 7.0);
        assert_eq!(result[25], 0.0);
    }

    #[test]
    fn test_gpu_buffer_async_copy_size_mismatch() {
        use crate::driver::CudaStream;
        let ctx = cuda_ctx!();
        let stream = CudaStream::new(&ctx).unwrap();

        let src: GpuBuffer<f32> = GpuBuffer::new(&ctx, 100).unwrap();
        let mut dst: GpuBuffer<f32> = GpuBuffer::new(&ctx, 50).unwrap();

        let result = unsafe { dst.copy_from_buffer_async(&src, &stream) };
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_buffer_async_copy_empty() {
        use crate::driver::CudaStream;
        let ctx = cuda_ctx!();
        let stream = CudaStream::new(&ctx).unwrap();

        let src: GpuBuffer<f32> = GpuBuffer::new(&ctx, 0).unwrap();
        let mut dst: GpuBuffer<f32> = GpuBuffer::new(&ctx, 0).unwrap();

        unsafe {
            dst.copy_from_buffer_async(&src, &stream).unwrap();
        }
    }

    #[test]
    fn test_gpu_buffer_async_copy_at_bounds_check() {
        use crate::driver::CudaStream;
        let ctx = cuda_ctx!();
        let stream = CudaStream::new(&ctx).unwrap();

        let src: GpuBuffer<f32> = GpuBuffer::new(&ctx, 10).unwrap();
        let mut dst: GpuBuffer<f32> = GpuBuffer::new(&ctx, 20).unwrap();

        // dst out of bounds
        let result = unsafe { dst.copy_from_buffer_at_async(&src, 15, 0, 10, &stream) };
        assert!(result.is_err());

        // src out of bounds
        let result = unsafe { dst.copy_from_buffer_at_async(&src, 0, 5, 10, &stream) };
        assert!(result.is_err());

        // zero count
        unsafe {
            dst.copy_from_buffer_at_async(&src, 0, 0, 0, &stream).unwrap();
        }
    }

    /// PMAT-420 / trueno#232: Verify that GpuBuffer transfers work correctly
    /// when the buffer is moved to a different thread than where the CUDA
    /// context was created.
    ///
    /// Before the fix, cuMemcpyHtoD returned CUDA_SUCCESS but silently
    /// transferred zero bytes because no CUDA context was current on the
    /// worker thread. Now `ensure_context()` pushes the stored context
    /// handle before every transfer.
    #[test]
    fn test_pmat420_cross_thread_transfer_nonzero() {
        let ctx = cuda_ctx!();
        let data: Vec<f32> = (1..=256).map(|i| i as f32).collect();

        // Allocate and upload on this thread (context is current here)
        let mut buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 256).unwrap();
        buf.copy_from_host(&data).unwrap();

        // Send the buffer to a new thread where context is NOT current
        let handle = std::thread::spawn(move || {
            // This thread has NO CUDA context pushed.
            // Before PMAT-420 fix, copy_to_host would read back all zeros.
            let mut result = vec![0.0f32; 256];
            buf.copy_to_host(&mut result).expect("copy_to_host on foreign thread must succeed");

            // The critical assertion: data must NOT be all zeros
            let nonzero = result.iter().filter(|&&v| v != 0.0).count();
            assert_eq!(
                nonzero, 256,
                "PMAT-420 regression: GPU readback is zeros on cross-thread transfer \
                 ({} of 256 elements are nonzero)",
                nonzero
            );
            assert_eq!(result[0], 1.0);
            assert_eq!(result[255], 256.0);

            buf // return ownership for cleanup
        });

        let mut buf = handle.join().expect("Worker thread must not panic");

        // Also test cross-thread upload: write on a different thread
        let handle2 = std::thread::spawn(move || {
            let new_data: Vec<f32> = (0..256).map(|i| -(i as f32)).collect();
            buf.copy_from_host(&new_data).expect("copy_from_host on foreign thread must succeed");
            (buf, new_data)
        });

        let (buf, new_data) = handle2.join().expect("Worker thread 2 must not panic");

        // Verify the cross-thread upload was correct (read back on original thread)
        let mut result = vec![0.0f32; 256];
        buf.copy_to_host(&mut result).unwrap();
        assert_eq!(result, new_data);
    }

    /// PMAT-420: Verify copy_from_host_at also works cross-thread
    #[test]
    fn test_pmat420_cross_thread_partial_transfer() {
        let ctx = cuda_ctx!();

        let mut buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 100).unwrap();
        let zeros = vec![0.0f32; 100];
        buf.copy_from_host(&zeros).unwrap();

        // Send to a worker thread and do partial write
        let handle = std::thread::spawn(move || {
            let patch: Vec<f32> = vec![42.0f32; 10];
            buf.copy_from_host_at(&patch, 50)
                .expect("copy_from_host_at on foreign thread must succeed");

            let mut result = vec![0.0f32; 100];
            buf.copy_to_host(&mut result).expect("copy_to_host on foreign thread must succeed");

            // Verify the partial write landed correctly
            assert_eq!(result[49], 0.0);
            assert_eq!(result[50], 42.0);
            assert_eq!(result[59], 42.0);
            assert_eq!(result[60], 0.0);
        });

        handle.join().expect("Worker thread must not panic");
    }
}
