#[cfg(kani)]
mod verification {

    /// KANI-Q35-001: Shape consistency for Qwen3.5 config
    /// Obligation: Q35-INV-001
    /// Strategy: exhaustive
    /// Verifies GQA shape algebra holds for all valid configs
    #[kani::proof]
    #[kani::unwind(2)]
    #[kani::solver(cadical)]
    fn verify_qwen35_shapes() {
        let n_h: usize = kani::any();
        let n_kv: usize = kani::any();
        let d_k: usize = kani::any();

        kani::assume(n_h >= 1 && n_h <= 64);
        kani::assume(n_kv >= 1 && n_kv <= n_h);
        kani::assume(d_k >= 1 && d_k <= 512);
        kani::assume(n_h % n_kv == 0); // GQA divisibility

        let q_dim = n_h * d_k;
        let kv_dim = n_kv * d_k;

        // Q dimension must be >= KV dimension
        assert!(q_dim >= kv_dim, "Q dim must be >= KV dim");
        // Q dimension must be exact multiple of KV dimension
        assert!(q_dim % kv_dim == 0, "Q dim must be multiple of KV dim");

        // GQA ratio consistency
        let gqa_ratio = n_h / n_kv;
        assert!(gqa_ratio >= 1);
        assert_eq!(gqa_ratio * n_kv, n_h);
        assert_eq!(gqa_ratio * kv_dim, q_dim);
    }
}
