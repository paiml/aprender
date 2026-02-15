
    #[test]
    fn test_print_q6k_blocks_synthetic() {
        let bytes = vec![0u8; Q6K_BLOCK_SIZE * 2];
        print_q6k_blocks(&bytes, 0, 1);
    }

    #[test]
    fn test_print_q8_0_blocks_synthetic() {
        let bytes = vec![0u8; Q8_0_BLOCK_SIZE * 2];
        print_q8_0_blocks(&bytes, 0, 1);
    }

    #[test]
    fn test_print_blocks_exceeds_bounds() {
        // Should print warning, not panic
        let bytes = vec![0u8; 10];
        print_q4k_blocks(&bytes, 0, 1);
    }
