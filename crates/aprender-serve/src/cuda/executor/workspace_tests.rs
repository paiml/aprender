
#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    /// Helper to create CudaExecutor for tests
    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // Workspace Initialization Tests
    // ========================================================================

    #[test]
    fn test_has_workspace_initial_false() {
        let Some(exec) = create_executor() else {
            return;
        };
        assert!(!exec.has_workspace());
    }

    #[test]
    fn test_init_workspace_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let result = exec.init_workspace(256, 1024);
        assert!(result.is_ok());
        assert!(exec.has_workspace());
    }

    #[test]
    fn test_init_workspace_dimensions() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let hidden_dim = 512;
        let intermediate_dim = 2048;

        exec.init_workspace(hidden_dim, intermediate_dim).unwrap();

        assert_eq!(exec.workspace.hidden_dim, hidden_dim);
        assert_eq!(exec.workspace.intermediate_dim, intermediate_dim);
        assert_eq!(exec.workspace.batch_size, 1);
    }

    #[test]
    fn test_clear_workspace() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        exec.init_workspace(256, 1024).unwrap();
        assert!(exec.has_workspace());

        exec.clear_workspace();
        assert!(!exec.has_workspace());
    }

    // ========================================================================
    // Batched Workspace Tests
    // ========================================================================

    #[test]
    fn test_init_batched_workspace_m4() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let result = exec.init_batched_workspace(256, 1024, 4);
        assert!(result.is_ok());
        assert!(exec.has_workspace());
        assert_eq!(exec.workspace_batch_size(), 4);
    }

    #[test]
    fn test_init_batched_workspace_m8() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let result = exec.init_batched_workspace(512, 2048, 8);
        assert!(result.is_ok());
        assert_eq!(exec.workspace_batch_size(), 8);
    }

    #[test]
    fn test_init_batched_workspace_m32() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let result = exec.init_batched_workspace(256, 1024, 32);
        assert!(result.is_ok());
        assert_eq!(exec.workspace_batch_size(), 32);
    }

    #[test]
    fn test_init_batched_workspace_invalid_zero() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let result = exec.init_batched_workspace(256, 1024, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_init_batched_workspace_invalid_too_large() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let result = exec.init_batched_workspace(256, 1024, 33);
        assert!(result.is_err());
    }

    // ========================================================================
    // Decode Graph Tests
    // ========================================================================

    #[test]
    fn test_has_decode_graph_initial_false() {
        let Some(exec) = create_executor() else {
            return;
        };
        assert!(!exec.has_decode_graph());
    }

    #[test]
    fn test_clear_decode_graph() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        exec.clear_decode_graph();
        assert!(!exec.has_decode_graph());
        assert_eq!(exec.decode_token_count, 0);
    }

    // ========================================================================
    // GEMV Buffer Pool Tests
    // ========================================================================

    #[test]
    fn test_gemv_buffer_stats_initial() {
        let Some(exec) = create_executor() else {
            return;
        };

        let (input_bytes, output_bytes) = exec.gemv_buffer_stats();
        assert_eq!(input_bytes, 0);
        assert_eq!(output_bytes, 0);
    }

    #[test]
    fn test_ensure_gemv_input_buffer() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let size = 256;
        let result = exec.ensure_gemv_input_buffer(size);
        assert!(result.is_ok());

        let (input_bytes, _) = exec.gemv_buffer_stats();
        assert_eq!(input_bytes, size * 4);
    }

    #[test]
    fn test_ensure_gemv_output_buffer() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let size = 128;
        let result = exec.ensure_gemv_output_buffer(size);
        assert!(result.is_ok());

        let (_, output_bytes) = exec.gemv_buffer_stats();
        assert_eq!(output_bytes, size * 4);
    }

    #[test]
    fn test_gemv_buffer_reuse() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // First allocation
        let ptr1 = exec.ensure_gemv_input_buffer(256).unwrap();
        // Same size should reuse buffer
        let ptr2 = exec.ensure_gemv_input_buffer(256).unwrap();
        assert_eq!(ptr1, ptr2);

        // Different size should reallocate
        let ptr3 = exec.ensure_gemv_input_buffer(512).unwrap();
        // Pointer may differ after reallocation
        let _ = ptr3;

        let (input_bytes, _) = exec.gemv_buffer_stats();
        assert_eq!(input_bytes, 512 * 4);
    }

    #[test]
    fn test_clear_gemv_buffers() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        exec.ensure_gemv_input_buffer(256).unwrap();
        exec.ensure_gemv_output_buffer(128).unwrap();

        exec.clear_gemv_buffers();

        let (input_bytes, output_bytes) = exec.gemv_buffer_stats();
        assert_eq!(input_bytes, 0);
        assert_eq!(output_bytes, 0);
    }

    #[test]
    fn test_copy_to_gemv_input() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let input: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();

        exec.ensure_gemv_input_buffer(256).unwrap();
        let result = exec.copy_to_gemv_input(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_copy_from_gemv_output() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let size = 128;
        exec.ensure_gemv_output_buffer(size).unwrap();

        let mut output = vec![0.0f32; size];
        let result = exec.copy_from_gemv_output(&mut output);
        assert!(result.is_ok());
    }
}
