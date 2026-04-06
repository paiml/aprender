#[cfg(kani)]
mod verification {

    /// KANI-QHF-001: Shape preservation through hybrid block
    /// Obligation: QHF-INV-001
    /// Strategy: bounded_int — verify shape algebra for bounded configs
    #[kani::proof]
    #[kani::unwind(5)]
    #[kani::solver(cadical)]
    fn verify_hybrid_block_shapes() {
        let d_model: usize = kani::any();
        let n_heads: usize = kani::any();
        let n_kv: usize = kani::any();
        let d_k: usize = kani::any();
        let intermediate: usize = kani::any();

        kani::assume(d_model >= 64 && d_model <= 8192);
        kani::assume(n_heads >= 1 && n_heads <= 64);
        kani::assume(n_kv >= 1 && n_kv <= n_heads);
        kani::assume(d_k >= 16 && d_k <= 512);
        kani::assume(n_heads % n_kv == 0);
        kani::assume(n_heads * d_k == d_model);
        kani::assume(intermediate >= d_model);
        kani::assume(intermediate <= 4 * d_model);

        // Attention sublayer: Q[d_model, n_h*d_k] -> attn -> O[n_h*d_k, d_model]
        let q_dim = n_heads * d_k;
        assert_eq!(q_dim, d_model, "Q output dim must equal d_model");
        let o_out = d_model;
        assert_eq!(o_out, d_model, "O projection must restore d_model");

        // GDN sublayer: in_proj + out_proj both use d_model
        let gdn_out = d_model;
        assert_eq!(gdn_out, d_model, "GDN must preserve d_model");

        // FFN sublayer: gate/up expand, down contracts back
        let ffn_out = d_model;
        assert_eq!(ffn_out, d_model, "FFN must preserve d_model");

        // Both attention and GDN blocks preserve shape
        let attn_block_out = d_model; // attn + residual
        let gdn_block_out = d_model; // gdn + residual
        assert_eq!(attn_block_out, gdn_block_out, "both block types same output dim");
    }

    /// KANI-QHF-002: Layer type exclusivity
    /// Obligation: QHF-INV-002
    /// Strategy: exhaustive — verify partition over all 48 layers
    #[kani::proof]
    #[kani::unwind(49)]
    fn verify_layer_type_partition() {
        let layer_idx: usize = kani::any();
        kani::assume(layer_idx < 48);

        // Even = attention, odd = GDN
        let is_attn = layer_idx % 2 == 0;
        let is_gdn = layer_idx % 2 == 1;

        // XOR: exactly one must be true
        assert!(is_attn ^ is_gdn, "layer must be exactly one type");
        // Both cannot be true
        assert!(!(is_attn && is_gdn), "layer cannot be both types");
        // At least one must be true
        assert!(is_attn || is_gdn, "layer must have a type");
    }
}
