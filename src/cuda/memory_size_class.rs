
    // =========================================================================
    // SizeClass Tests
    // =========================================================================

    #[test]
    fn test_size_class_constants() {
        assert_eq!(SizeClass::CLASSES.len(), 9);
        assert_eq!(SizeClass::CLASSES[0], 4096); // 4 KB
        assert_eq!(SizeClass::CLASSES[8], 268_435_456); // 256 MB
    }

    #[test]
    fn test_size_class_for_size_exact_match() {
        let class = SizeClass::for_size(4096).expect("class");
        assert_eq!(class.bytes(), 4096);
    }

    #[test]
    fn test_size_class_for_size_rounds_up() {
        let class = SizeClass::for_size(5000).expect("class");
        assert_eq!(class.bytes(), 16384); // 16 KB
    }

    #[test]
    fn test_size_class_for_size_smallest() {
        let class = SizeClass::for_size(1).expect("class");
        assert_eq!(class.bytes(), 4096); // 4 KB minimum
    }

    #[test]
    fn test_size_class_for_size_too_large() {
        let result = SizeClass::for_size(300_000_000); // > 256 MB
        assert!(result.is_none());
    }

    #[test]
    fn test_size_class_for_size_zero() {
        let class = SizeClass::for_size(0).expect("class");
        assert_eq!(class.bytes(), 4096); // 4 KB minimum
    }

    #[test]
    fn test_size_class_ord() {
        let small = SizeClass(4096);
        let large = SizeClass(65536);
        assert!(small < large);
    }

    #[test]
    fn test_size_class_eq() {
        let a = SizeClass(4096);
        let b = SizeClass(4096);
        assert_eq!(a, b);
    }

    #[test]
    fn test_size_class_clone_copy() {
        let class = SizeClass(4096);
        let cloned = class;
        assert_eq!(cloned.bytes(), 4096);
    }

    #[test]
    fn test_size_class_debug() {
        let class = SizeClass(4096);
        let debug_str = format!("{:?}", class);
        assert!(debug_str.contains("SizeClass"));
        assert!(debug_str.contains("4096"));
    }

    // =========================================================================
    // GpuBufferHandle Tests
    // =========================================================================

    #[test]
    fn test_gpu_buffer_handle_new() {
        let handle = GpuBufferHandle {
            size: 1024,
            in_use: false,
        };
        assert_eq!(handle.size, 1024);
        assert!(!handle.in_use);
    }

    #[test]
    fn test_gpu_buffer_handle_debug() {
        let handle = GpuBufferHandle {
            size: 1024,
            in_use: true,
        };
        let debug_str = format!("{:?}", handle);
        assert!(debug_str.contains("GpuBufferHandle"));
        assert!(debug_str.contains("1024"));
        assert!(debug_str.contains("true"));
    }

    // =========================================================================
    // GpuMemoryPool Tests
    // =========================================================================

    #[test]
    fn test_gpu_memory_pool_default() {
        let pool = GpuMemoryPool::default();
        assert_eq!(pool.max_size(), 2 * 1024 * 1024 * 1024); // 2 GB
    }

    #[test]
    fn test_gpu_memory_pool_new() {
        let pool = GpuMemoryPool::new();
        let stats = pool.stats();
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.peak_usage, 0);
        assert_eq!(stats.pool_hits, 0);
        assert_eq!(stats.pool_misses, 0);
    }

    #[test]
    fn test_gpu_memory_pool_with_max_size() {
        let pool = GpuMemoryPool::with_max_size(1024 * 1024);
        assert_eq!(pool.max_size(), 1024 * 1024);
    }

    #[test]
    fn test_gpu_memory_pool_try_get_empty() {
        let mut pool = GpuMemoryPool::new();
        let result = pool.try_get(4096);
        assert!(result.is_none());
        assert_eq!(pool.stats().pool_misses, 1);
    }

    #[test]
    fn test_gpu_memory_pool_return_and_get() {
        let mut pool = GpuMemoryPool::new();
        let handle = GpuBufferHandle {
            size: 4096,
            in_use: true,
        };
        pool.return_buffer(handle);

        let result = pool.try_get(4096);
        assert!(result.is_some());
        let retrieved = result.expect("retrieved");
        assert_eq!(retrieved.size, 4096);
        assert!(retrieved.in_use);
        assert_eq!(pool.stats().pool_hits, 1);
    }

    #[test]
    fn test_gpu_memory_pool_record_allocation() {
        let mut pool = GpuMemoryPool::new();
        pool.record_allocation(1000);
        pool.record_allocation(2000);

        let stats = pool.stats();
        assert_eq!(stats.total_allocated, 3000);
        assert_eq!(stats.peak_usage, 3000);
    }

    #[test]
    fn test_gpu_memory_pool_record_deallocation() {
        let mut pool = GpuMemoryPool::new();
        pool.record_allocation(5000);
        pool.record_deallocation(2000);

        assert_eq!(pool.stats().total_allocated, 3000);
        assert_eq!(pool.stats().peak_usage, 5000);
    }

    #[test]
    fn test_gpu_memory_pool_deallocation_underflow() {
        let mut pool = GpuMemoryPool::new();
        pool.record_allocation(1000);
        pool.record_deallocation(5000); // More than allocated
        assert_eq!(pool.stats().total_allocated, 0); // Saturating sub
    }

    #[test]
    fn test_gpu_memory_pool_has_capacity() {
        let pool = GpuMemoryPool::with_max_size(1000);
        assert!(pool.has_capacity(500));
        assert!(pool.has_capacity(1000));
        assert!(!pool.has_capacity(1001));
    }

    #[test]
    fn test_gpu_memory_pool_clear() {
        let mut pool = GpuMemoryPool::new();
        let handle = GpuBufferHandle {
            size: 4096,
            in_use: false,
        };
        pool.return_buffer(handle);
        assert_eq!(pool.stats().free_buffers, 1);

        pool.clear();
        assert_eq!(pool.stats().free_buffers, 0);
    }

    #[test]
    fn test_gpu_memory_pool_debug() {
        let pool = GpuMemoryPool::new();
        let debug_str = format!("{:?}", pool);
        assert!(debug_str.contains("GpuMemoryPool"));
    }

    // =========================================================================
    // PoolStats Tests
    // =========================================================================

    #[test]
    fn test_pool_stats_clone() {
        let stats = PoolStats {
            total_allocated: 1000,
            peak_usage: 2000,
            pool_hits: 10,
            pool_misses: 5,
            hit_rate: 0.667,
            free_buffers: 3,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.total_allocated, 1000);
        assert_eq!(cloned.pool_hits, 10);
    }

    #[test]
    fn test_pool_stats_estimated_savings_with_hits() {
        let stats = PoolStats {
            total_allocated: 0,
            peak_usage: 0,
            pool_hits: 100,
            pool_misses: 50,
            hit_rate: 0.667,
            free_buffers: 0,
        };
        let savings = stats.estimated_savings_bytes();
        assert_eq!(savings, 100 * 1024 * 1024); // 100 MB
    }

    #[test]
    fn test_pool_stats_estimated_savings_no_hits() {
        let stats = PoolStats {
            total_allocated: 0,
            peak_usage: 0,
            pool_hits: 0,
            pool_misses: 50,
            hit_rate: 0.0,
            free_buffers: 0,
        };
        assert_eq!(stats.estimated_savings_bytes(), 0);
    }

    #[test]
    fn test_pool_stats_debug() {
        let stats = PoolStats {
            total_allocated: 0,
            peak_usage: 0,
            pool_hits: 0,
            pool_misses: 0,
            hit_rate: 0.0,
            free_buffers: 0,
        };
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("PoolStats"));
    }

    // =========================================================================
    // PinnedHostBuffer Tests
    // =========================================================================

    #[test]
    fn test_pinned_host_buffer_new() {
        let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(100);
        assert_eq!(buf.len(), 100);
        assert!(!buf.is_empty());
        assert!(!buf.is_pinned()); // Currently not truly pinned
    }

    #[test]
    fn test_pinned_host_buffer_empty() {
        let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(0);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_pinned_host_buffer_as_slice() {
        let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(10);
        let slice = buf.as_slice();
        assert_eq!(slice.len(), 10);
        assert_eq!(slice[0], 0.0); // Default f32
    }

    #[test]
    fn test_pinned_host_buffer_as_mut_slice() {
        let mut buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(10);
        buf.as_mut_slice()[0] = 42.0;
        assert_eq!(buf.as_slice()[0], 42.0);
    }

    #[test]
    fn test_pinned_host_buffer_size_bytes() {
        let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(100);
        assert_eq!(buf.size_bytes(), 100 * std::mem::size_of::<f32>());
    }

    #[test]
    fn test_pinned_host_buffer_copy_from_slice() {
        let mut buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(3);
        buf.copy_from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(buf.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_pinned_host_buffer_debug() {
        let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(10);
        let debug_str = format!("{:?}", buf);
        assert!(debug_str.contains("PinnedHostBuffer"));
    }

    #[test]
    fn test_pinned_host_buffer_different_types() {
        let buf_i32: PinnedHostBuffer<i32> = PinnedHostBuffer::new(5);
        assert_eq!(buf_i32.len(), 5);
        assert_eq!(buf_i32.size_bytes(), 5 * std::mem::size_of::<i32>());

        let buf_u8: PinnedHostBuffer<u8> = PinnedHostBuffer::new(100);
        assert_eq!(buf_u8.size_bytes(), 100);
    }

    // =========================================================================
    // StagingBufferPool Tests
    // =========================================================================

    #[test]
    fn test_staging_buffer_pool_default() {
        let pool = StagingBufferPool::default();
        let stats = pool.stats();
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.pool_hits, 0);
        assert_eq!(stats.pool_misses, 0);
    }

    #[test]
    fn test_staging_buffer_pool_new() {
        let pool = StagingBufferPool::new();
        let stats = pool.stats();
        assert_eq!(stats.free_buffers, 0);
    }

    #[test]
    fn test_staging_buffer_pool_with_max_size() {
        let pool = StagingBufferPool::with_max_size(1024 * 1024);
        let stats = pool.stats();
        assert_eq!(stats.total_allocated, 0);
    }

    #[test]
    fn test_staging_buffer_pool_get_allocates() {
        let mut pool = StagingBufferPool::new();
        let buf = pool.get(100);
        assert!(buf.len() >= 100);
        assert_eq!(pool.stats().pool_misses, 1);
    }

    #[test]
    fn test_staging_buffer_pool_put_and_get() {
        let mut pool = StagingBufferPool::new();
        let buf = pool.get(1000);
        let size = buf.size_bytes();
        pool.put(buf);

        assert_eq!(pool.stats().free_buffers, 1);

        let buf2 = pool.get(1000);
        assert_eq!(buf2.size_bytes(), size);
        assert_eq!(pool.stats().pool_hits, 1);
    }

    #[test]
    fn test_staging_buffer_pool_clear() {
        let mut pool = StagingBufferPool::new();
        let buf = pool.get(1000);
        pool.put(buf);
        assert_eq!(pool.stats().free_buffers, 1);

        pool.clear();
        assert_eq!(pool.stats().free_buffers, 0);
        assert_eq!(pool.stats().total_allocated, 0);
    }

    #[test]
    fn test_staging_buffer_pool_debug() {
        let pool = StagingBufferPool::new();
        let debug_str = format!("{:?}", pool);
        assert!(debug_str.contains("StagingBufferPool"));
    }

    // =========================================================================
    // StagingPoolStats Tests
    // =========================================================================

    #[test]
    fn test_staging_pool_stats_clone() {
        let stats = StagingPoolStats {
            total_allocated: 1000,
            peak_usage: 2000,
            pool_hits: 10,
            pool_misses: 5,
            free_buffers: 3,
            hit_rate: 0.667,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.total_allocated, 1000);
        assert!((cloned.hit_rate - 0.667).abs() < 0.001);
    }

    #[test]
    fn test_staging_pool_stats_debug() {
        let stats = StagingPoolStats {
            total_allocated: 0,
            peak_usage: 0,
            pool_hits: 0,
            pool_misses: 0,
            free_buffers: 0,
            hit_rate: 0.0,
        };
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("StagingPoolStats"));
    }

    // =========================================================================
    // TransferMode Tests
    // =========================================================================

    #[test]
    fn test_transfer_mode_default() {
        let mode = TransferMode::default();
        assert_eq!(mode, TransferMode::Pageable);
    }

    #[test]
    fn test_transfer_mode_requires_pinned() {
        assert!(!TransferMode::Pageable.requires_pinned());
        assert!(TransferMode::Pinned.requires_pinned());
        assert!(TransferMode::ZeroCopy.requires_pinned());
        assert!(TransferMode::Async.requires_pinned());
    }
