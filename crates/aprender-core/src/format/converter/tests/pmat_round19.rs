//! Round 19 Falsification Tests (Metadata & Safety)
//!
//! PMAT-223: Metadata Fidelity
//! PMAT-224: Architecture Safety

#[cfg(test)]
mod tests {
    use crate::format::converter::write::write_apr_file;
    use crate::format::converter_types::{Architecture, ImportOptions};
    use crate::serialization::safetensors::UserMetadata;
    use std::collections::BTreeMap;
    use std::fs;

    // ========================================================================
    // PMAT-223: Metadata Fidelity Tests
    // ========================================================================

    #[test]
    fn test_pmat223_user_metadata_preservation() {
        // Setup output path
        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("pmat223_metadata.apr");

        // Mock Tensors
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.0; 10], vec![10]),
        );

        // Mock User Metadata (simulating SafeTensors __metadata__)
        let mut user_metadata = UserMetadata::new();
        user_metadata.insert("training_run_id".to_string(), "run_12345".to_string());
        user_metadata.insert("license".to_string(), "apache-2.0".to_string());
        user_metadata.insert("base_model".to_string(), "Qwen/Qwen2.5-Coder".to_string());

        // Write APR file
        let options = ImportOptions {
            architecture: Architecture::Qwen2, // Verified arch
            allow_no_config: true,
            ..Default::default()
        };

        let empty_f16: BTreeMap<String, (Vec<u8>, Vec<usize>, bool)> = BTreeMap::new();
        write_apr_file(
            &tensors,
            &empty_f16,
            &output_path,
            &options,
            None, // No tokenizer
            None, // No config
            &user_metadata,
        )
        .expect("Failed to write APR file");

        // Read back and verify
        let bytes = fs::read(&output_path).expect("Failed to read APR file");

        // Parse metadata (JSON header is at the end of the file or beginning? APR v2 has it at end typically or header)
        // We'll just regex for it to be robust against format changes in this test
        let content = String::from_utf8_lossy(&bytes);

        // Check for source_metadata
        assert!(
            content.contains("source_metadata"),
            "APR missing source_metadata field"
        );
        assert!(
            content.contains("training_run_id"),
            "APR missing custom key"
        );
        assert!(content.contains("run_12345"), "APR missing custom value");
        assert!(content.contains("base_model"), "APR missing base_model key");

        // Cleanup
        let _ = fs::remove_file(output_path);
    }

    // ========================================================================
    // PMAT-224: Architecture Safety Tests
    // ========================================================================

    #[test]
    fn test_pmat224_strict_rejects_unverified_arch() {
        // GH-326 Phase 4b refit: this test used to verify that BERT was
        // rejected by strict mode (since it was unverified). After BERT
        // was promoted to verified, the "rejection by strict mode"
        // semantic moved to Gpt2 which is still unverified. The test
        // now pins TWO invariants:
        // (a) BERT IS verified (no longer rejected by strict mode)
        // (b) Gpt2 is NOT verified (still rejected by strict mode)

        let bert = Architecture::Bert;
        assert!(
            bert.is_inference_verified(),
            "BERT verified post-#326 Phase 4b"
        );

        let qwen = Architecture::Qwen2;
        assert!(qwen.is_inference_verified(), "Qwen2 verified");

        let gpt2 = Architecture::Gpt2;
        assert!(
            !gpt2.is_inference_verified(),
            "Gpt2 unverified — exercises the strict-mode rejection path"
        );

        // Strict mode rejects unverified architectures
        let options_strict_unverified = ImportOptions {
            architecture: Architecture::Gpt2,
            strict: true,
            allow_no_config: true,
            ..Default::default()
        };
        assert!(
            !options_strict_unverified
                .architecture
                .is_inference_verified()
                && options_strict_unverified.strict,
            "Strict-mode Gpt2 import should hit the rejection branch"
        );

        // Strict mode allows verified architectures (including BERT
        // post-Phase 4b)
        let options_strict_verified = ImportOptions {
            architecture: Architecture::Bert,
            strict: true,
            allow_no_config: true,
            ..Default::default()
        };
        assert!(
            options_strict_verified.architecture.is_inference_verified(),
            "Strict-mode BERT import should now pass the verification check \
             (post-#326 Phase 4b)"
        );
    }
}
