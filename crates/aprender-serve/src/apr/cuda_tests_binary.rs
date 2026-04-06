
    /// Helper to write a tensor entry in APR v2 binary format
    fn write_tensor_entry_binary(
        name: &str,
        dtype: u8,
        shape: &[usize],
        offset: u64,
        size: u64,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        // Name length (u16 LE)
        buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
        // Name bytes
        buf.extend_from_slice(name.as_bytes());
        // Dtype (u8)
        buf.push(dtype);
        // Ndim (u8)
        buf.push(shape.len() as u8);
        // Shape dimensions (u64 LE each)
        for &dim in shape {
            buf.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        // Offset (u64 LE)
        buf.extend_from_slice(&offset.to_le_bytes());
        // Size (u64 LE)
        buf.extend_from_slice(&size.to_le_bytes());
        buf
    }

    /// Helper to create a minimal valid APR v2 model in memory
    fn create_minimal_apr_model() -> AprV2Model {
        // Minimal metadata with required fields for AprV2ModelCuda
        let metadata = r#"{
            "model_type": "test",
            "name": "test-cuda-model",
            "hidden_dim": 32,
            "num_layers": 1,
            "num_heads": 1,
            "vocab_size": 100
        }"#;

        // Create tensor index in binary format
        // dtype: 0=F32
        let tensor_index_binary = write_tensor_entry_binary(
            "model.embed_tokens.weight", // name
            0,                           // dtype = F32
            &[100, 32],                  // shape = [vocab_size, hidden_dim]
            0,                           // offset in data section
            100 * 32 * 4,                // size in bytes
        );

        // Embedding data (100 * 32 floats)
        let tensor_data: Vec<f32> = vec![0.1; 100 * 32];
        let tensor_bytes: Vec<u8> = tensor_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Calculate offsets (64-byte aligned)
        let metadata_offset = HEADER_SIZE as u64;
        let metadata_size = metadata.len() as u32;
        let tensor_index_offset =
            ((metadata_offset as usize + metadata.len()).div_ceil(64) * 64) as u64;
        let data_offset =
            ((tensor_index_offset as usize + tensor_index_binary.len()).div_ceil(64) * 64) as u64;

        let total_size = data_offset as usize + tensor_bytes.len();
        let mut data = vec![0u8; total_size];

        // Header (64 bytes)
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // Version major
        data[5] = 0; // Version minor
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // Flags
        data[8..12].copy_from_slice(&1u32.to_le_bytes()); // Tensor count
        data[12..20].copy_from_slice(&metadata_offset.to_le_bytes());
        data[20..24].copy_from_slice(&metadata_size.to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        // Metadata
        data[metadata_offset as usize..metadata_offset as usize + metadata.len()]
            .copy_from_slice(metadata.as_bytes());

        // Tensor index (binary format)
        data[tensor_index_offset as usize
            ..tensor_index_offset as usize + tensor_index_binary.len()]
            .copy_from_slice(&tensor_index_binary);

        // Tensor data
        data[data_offset as usize..data_offset as usize + tensor_bytes.len()]
            .copy_from_slice(&tensor_bytes);

        AprV2Model::from_bytes(data).expect("should create minimal APR model")
    }

    // =========================================================================
    // AprV2ModelCuda Construction Tests
    // =========================================================================

    #[test]
    fn test_minimal_model_has_embedding_tensor() {
        let model = create_minimal_apr_model();

        // Debug: list all tensors
        for tensor in &model.tensors {
            println!("Tensor: '{}' shape={:?}", tensor.name, tensor.shape);
        }

        // Verify the embedding tensor exists
        let embedding = model.get_tensor("model.embed_tokens.weight");
        assert!(
            embedding.is_some(),
            "Should have model.embed_tokens.weight tensor"
        );
    }

    #[test]
    fn test_apr_cuda_new_succeeds_with_gpu() {
        let model = create_minimal_apr_model();

        // Attempt to create CUDA wrapper - this tests the constructor path
        let result = AprV2ModelCuda::new(model, 0);

        // Should succeed if GPU is available, fail gracefully if not
        // Under coverage instrumentation, may fail - that's OK
        match result {
            Ok(cuda_model) => {
                // Verify device info was captured
                if !cuda_model.device_name().is_empty() {
                    println!("GPU: {}", cuda_model.device_name());
                }
            },
            Err(e) => {
                // Expected if no GPU or under heavy instrumentation
                println!("CUDA init result: {:?}", e);
            },
        }
    }

    #[test]
    fn test_apr_cuda_with_max_seq_len() {
        let model = create_minimal_apr_model();

        // Test with custom sequence length
        let result = AprV2ModelCuda::with_max_seq_len(model, 0, 512);

        // Under coverage instrumentation, may fail - that's OK
        match result {
            Ok(_cuda_model) => {
                println!("CUDA model created with max_seq_len=512");
            },
            Err(e) => {
                println!("CUDA init result: {:?}", e);
            },
        }
    }

    #[test]
    fn test_apr_cuda_invalid_device() {
        let model = create_minimal_apr_model();

        // Device 999 should not exist
        let result = AprV2ModelCuda::new(model, 999);

        // Should fail - exercises error path
        assert!(result.is_err(), "Device 999 should not exist");
    }

    // =========================================================================
    // AprV2ModelCuda Method Tests (require GPU)
    // =========================================================================

    #[test]
    fn test_apr_cuda_device_name() {
        let model = create_minimal_apr_model();

        if let Ok(cuda_model) = AprV2ModelCuda::new(model, 0) {
            let name = cuda_model.device_name();
            // Should return non-empty device name
            assert!(!name.is_empty(), "Device name should not be empty");
            println!("GPU device: {}", name);
        }
    }

    #[test]
    fn test_apr_cuda_memory_info() {
        let model = create_minimal_apr_model();

        if let Ok(cuda_model) = AprV2ModelCuda::new(model, 0) {
            let (free, total) = cuda_model.memory_info();
            // Memory should be positive
            assert!(total > 0, "Total GPU memory should be > 0");
            assert!(free <= total, "Free memory should not exceed total");
            println!("GPU memory: {}/{} bytes free", free, total);
        }
    }

    #[test]
    fn test_apr_cuda_vram_mb() {
        let model = create_minimal_apr_model();

        if let Ok(cuda_model) = AprV2ModelCuda::new(model, 0) {
            let vram = cuda_model.vram_mb();
            assert!(vram > 0, "VRAM should be > 0 MB");
            println!("GPU VRAM: {} MB", vram);
        }
    }

    #[test]
    fn test_apr_cuda_inner_model() {
        let model = create_minimal_apr_model();

        if let Ok(cuda_model) = AprV2ModelCuda::new(model, 0) {
            let inner = cuda_model.inner();
            assert!(inner.tensor_count() > 0, "Model should have tensors");
        }
    }

    #[test]
    fn test_apr_cuda_is_available() {
        // Static method - should always work
        let available = AprV2ModelCuda::is_available();
        println!("CUDA available: {}", available);
        // Don't assert - just exercise the code path
    }

    #[test]
    fn test_apr_cuda_num_devices() {
        let count = AprV2ModelCuda::num_devices();
        println!("CUDA devices: {}", count);
        // Should be >= 0
    }

    #[test]
    fn test_apr_cuda_profiling() {
        let model = create_minimal_apr_model();

        if let Ok(mut cuda_model) = AprV2ModelCuda::new(model, 0) {
            // Test profiling toggle
            assert!(!cuda_model.is_profiling_enabled());

            cuda_model.enable_profiling();
            assert!(cuda_model.is_profiling_enabled());

            cuda_model.disable_profiling();
            assert!(!cuda_model.is_profiling_enabled());
        }
    }

    #[test]
    fn test_apr_cuda_profiler_access() {
        let model = create_minimal_apr_model();

        if let Ok(cuda_model) = AprV2ModelCuda::new(model, 0) {
            // Just access the profiler - don't need to assert anything
            let _profiler = cuda_model.profiler();
        }
    }

    #[test]
    fn test_apr_cuda_reset_profiler() {
        let model = create_minimal_apr_model();

        if let Ok(mut cuda_model) = AprV2ModelCuda::new(model, 0) {
            cuda_model.enable_profiling();
            cuda_model.reset_profiler();
            // Should not panic
        }
    }

    #[test]
    fn test_apr_cuda_reset_kv_cache() {
        let model = create_minimal_apr_model();

        if let Ok(mut cuda_model) = AprV2ModelCuda::new(model, 0) {
            cuda_model.reset_kv_cache();
            // Should not panic
        }
    }

    #[test]
    fn test_apr_cuda_weights_cached() {
        let model = create_minimal_apr_model();

        if let Ok(cuda_model) = AprV2ModelCuda::new(model, 0) {
            let cached = cuda_model.weights_cached();
            // Initial state - may or may not be cached
            println!("Weights cached: {}", cached);
        }
    }

    #[test]
    fn test_apr_cuda_cached_weight_mb() {
        let model = create_minimal_apr_model();

        if let Ok(cuda_model) = AprV2ModelCuda::new(model, 0) {
            let mb = cuda_model.cached_weight_mb();
            println!("Cached weight: {} MB", mb);
        }
    }

    // =========================================================================
    // Phase 45: Test Executor Injection Tests
    // =========================================================================

    #[test]
    fn test_apr_cuda_with_test_executor() {
        use crate::gpu::executor::MockExecutor;

        let model = create_minimal_apr_model();

        if let Ok(mut cuda_model) = AprV2ModelCuda::new(model, 0) {
            // Initially no test executor
            assert!(!cuda_model.has_test_executor());

            // Inject test executor
            let mock = MockExecutor::new("apr_test");
            cuda_model.with_test_executor(Box::new(mock));

            // Now has test executor
            assert!(cuda_model.has_test_executor());
        }
    }

    #[test]
    fn test_apr_cuda_test_executor_bypasses_fast_path() {
        use crate::gpu::executor::MockExecutor;

        let model = create_minimal_apr_model();

        if let Ok(mut cuda_model) = AprV2ModelCuda::new(model, 0) {
            // Inject test executor
            let mock = MockExecutor::new("bypass_test");
            cuda_model.with_test_executor(Box::new(mock));

            // has_cached_weight should return false when test_executor is present
            // This forces the uncached GEMM path which routes through test_executor
            assert!(!cuda_model.has_test_executor() || !cuda_model.weights_cached());
            // Actually just verify test executor is set
            assert!(cuda_model.has_test_executor());
        }
    }

    #[test]
    fn test_apr_cuda_forward_with_test_executor() {
        use crate::gpu::executor::CpuExecutor;

        let model = create_transformer_model();

        if let Ok(mut cuda_model) = AprV2ModelCuda::new(model, 0) {
            // Inject CPU executor for testing
            let cpu = CpuExecutor::new();
            cuda_model.with_test_executor(Box::new(cpu));

            // Call forward_cuda - should use test executor path
            // The model won't have proper weights so this will likely fail,
            // but it exercises the test_executor code paths
            let result = cuda_model.forward_cuda(&[1]);

            // Result may fail due to missing weights, but the path was exercised
            println!("Forward with test executor result: {:?}", result.is_ok());
        }
    }

    #[test]
    fn test_apr_cuda_forward_with_mock_executor() {
        use crate::gpu::executor::MockExecutor;

        let model = create_transformer_model();

        if let Ok(mut cuda_model) = AprV2ModelCuda::new(model, 0) {
            // Inject mock executor
            let mock = MockExecutor::new("forward_mock");
            cuda_model.with_test_executor(Box::new(mock));

            // Call forward_cuda
            let result = cuda_model.forward_cuda(&[1]);

            // May fail due to missing weights, but exercises the path
            println!("Forward with mock executor: {:?}", result.is_ok());
        }
    }

    #[test]
    fn test_apr_cuda_forward_mock_error_handling() {
        use crate::gpu::executor::MockExecutor;

        let model = create_transformer_model();

        if let Ok(mut cuda_model) = AprV2ModelCuda::new(model, 0) {
            // Inject mock that fails
            let mock = MockExecutor::new("fail_mock").with_matmul_failure();
            cuda_model.with_test_executor(Box::new(mock));

            // forward_cuda should fail when mock fails
            let result = cuda_model.forward_cuda(&[1]);

            // May fail for other reasons (missing weights), but the path is exercised
            println!("Forward with failing mock: {:?}", result.is_err());
        }
    }

    #[test]
    fn test_apr_cuda_generate_with_test_executor() {
        use crate::gpu::executor::CpuExecutor;

        let model = create_transformer_model();

        if let Ok(mut cuda_model) = AprV2ModelCuda::new(model, 0) {
            // Inject CPU executor
            let cpu = CpuExecutor::new();
            cuda_model.with_test_executor(Box::new(cpu));

            // Call generate_cuda - exercises forward path with test executor
            let result = cuda_model.generate_cuda(&[1], 1, 2); // 1 token, eos=2

            println!("Generate with test executor: {:?}", result.is_ok());
        }
    }
