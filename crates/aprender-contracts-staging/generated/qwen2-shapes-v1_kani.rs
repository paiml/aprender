#[cfg(kani)]
mod verification {
    /// KANI-QW2-001: Shape consistency for Qwen2.5-7B config
    /// Obligation: QW2-INV-001
    /// Strategy: exhaustive
    /// Verifies: For any valid GQA config, Q dim >= KV dim and Q dim is a multiple of KV dim
    #[kani::proof]
    #[kani::unwind(2)]
    #[kani::solver(cadical)]
    fn verify_qwen2_shapes() {
        let n_h: usize = kani::any();
        let n_kv: usize = kani::any();
        let d_k: usize = kani::any();

        kani::assume(n_h >= 1 && n_h <= 64);
        kani::assume(n_kv >= 1 && n_kv <= n_h);
        kani::assume(d_k >= 1 && d_k <= 256);
        kani::assume(n_h % n_kv == 0); // GQA divisibility

        let q_dim = n_h * d_k;
        let kv_dim = n_kv * d_k;

        // Q dimension must be >= KV dimension
        assert!(q_dim >= kv_dim, "Q dim must be >= KV dim");
        // Q dimension must be an exact multiple of KV dimension
        assert!(q_dim % kv_dim == 0, "Q dim must be a multiple of KV dim");
        // The GQA ratio from dims must match the head ratio
        let gqa_ratio = n_h / n_kv;
        assert_eq!(q_dim / kv_dim, gqa_ratio);
        // Hidden size (q_dim) must be divisible by n_h (head dim consistency)
        assert_eq!(q_dim / n_h, d_k);
    }
}
