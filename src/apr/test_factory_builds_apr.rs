
#[cfg(test)]
mod tests {
    use super::*;
    use crate::apr::AprV2Model;

    #[test]
    fn test_factory_builds_valid_apr() {
        let data = build_executable_pygmy_apr();
        let model = AprV2Model::from_bytes(data);
        assert!(model.is_ok());
    }

    #[test]
    fn test_factory_apr_forward_works() {
        let data = build_executable_pygmy_apr();
        let model = AprV2Model::from_bytes(data).unwrap();
        let logits = model.forward(&[1]);
        assert!(logits.is_ok());
    }

    // =========================================================================
    // GH-194: Weight Tying Tests (token_embd.weight as lm_head)
    // =========================================================================

    /// GH-194: GGUF naming convention model must load successfully
    #[test]
    fn test_gh194_gguf_names_model_loads() {
        let data = build_executable_pygmy_apr_gguf_names();
        let model = AprV2Model::from_bytes(data);
        assert!(model.is_ok(), "GH-194: GGUF-named model must load");
    }

    /// GH-194: GGUF model with token_embd.weight (no lm_head) must find lm_head tensor
    #[test]
    fn test_gh194_gguf_names_finds_lm_head_via_token_embd() {
        let data = build_executable_pygmy_apr_gguf_names();
        let model = AprV2Model::from_bytes(data).unwrap();

        // The model should find token_embd.weight when looking for lm_head
        let lm_head_candidates = [
            "lm_head.weight",
            "output.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "token_embd.weight",
        ];

        let found = lm_head_candidates
            .iter()
            .any(|&name| model.get_tensor(name).is_some());

        assert!(
            found,
            "GH-194: Model must find at least one lm_head candidate. \
             token_embd.weight should be found for weight tying."
        );
    }

    /// GH-194: GGUF model forward pass must work with weight tying
    #[test]
    fn test_gh194_gguf_names_forward_works() {
        let data = build_executable_pygmy_apr_gguf_names();
        let model = AprV2Model::from_bytes(data).unwrap();

        // Forward should work even without explicit lm_head.weight
        let result = model.forward(&[1]);
        assert!(
            result.is_ok(),
            "GH-194: Forward must work with token_embd.weight as lm_head. Error: {:?}",
            result.err()
        );

        // Logits should have vocab_size elements
        let logits = result.unwrap();
        assert_eq!(
            logits.len(),
            10,
            "GH-194: Logits must have vocab_size=10 elements"
        );
    }

    /// GH-194: embed_tokens.weight tied model must work
    #[test]
    fn test_gh194_embed_tied_forward_works() {
        let data = build_executable_pygmy_apr_embed_tied();
        let model = AprV2Model::from_bytes(data).unwrap();

        let result = model.forward(&[1]);
        assert!(
            result.is_ok(),
            "GH-194: Forward must work with embed_tokens.weight as lm_head. Error: {:?}",
            result.err()
        );
    }

    /// GH-194: Tensor count must match expected count (no silent drops)
    #[test]
    fn test_gh194_tensor_count_preserved() {
        // HuggingFace naming: 12 tensors
        let hf_data = build_executable_pygmy_apr();
        let hf_model = AprV2Model::from_bytes(hf_data).unwrap();
        assert_eq!(
            hf_model.tensor_count(),
            12,
            "HuggingFace model must have 12 tensors"
        );

        // GGUF naming: 11 tensors (no lm_head due to weight tying)
        let gguf_data = build_executable_pygmy_apr_gguf_names();
        let gguf_model = AprV2Model::from_bytes(gguf_data).unwrap();
        assert_eq!(
            gguf_model.tensor_count(),
            11,
            "GGUF model must have 11 tensors (weight tying)"
        );

        // Embed-tied naming: 11 tensors (no lm_head)
        let tied_data = build_executable_pygmy_apr_embed_tied();
        let tied_model = AprV2Model::from_bytes(tied_data).unwrap();
        assert_eq!(
            tied_model.tensor_count(),
            11,
            "Embed-tied model must have 11 tensors"
        );
    }

    /// GH-194: All tensor naming conventions must produce valid logits
    #[test]
    fn test_gh194_all_naming_conventions_produce_valid_logits() {
        let models = vec![
            ("HuggingFace", build_executable_pygmy_apr()),
            ("GGUF", build_executable_pygmy_apr_gguf_names()),
            ("EmbedTied", build_executable_pygmy_apr_embed_tied()),
        ];

        for (name, data) in models {
            let model =
                AprV2Model::from_bytes(data).unwrap_or_else(|_| panic!("{name} model must load"));
            let logits = model
                .forward(&[1])
                .unwrap_or_else(|_| panic!("{name} forward must work"));

            // Logits must be valid floats (no NaN/Inf)
            assert!(
                logits.iter().all(|&x| x.is_finite()),
                "{name}: Logits must be finite (no NaN/Inf)"
            );

            // Logits must have correct size
            assert_eq!(logits.len(), 10, "{name}: Logits must have vocab_size=10");
        }
    }
}
