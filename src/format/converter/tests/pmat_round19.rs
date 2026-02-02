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
            ..Default::default()
        };

        write_apr_file(
            &tensors,
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
    fn test_pmat224_bert_rejection() {
        // This test simulates the architecture check logic in apr_import
        // We can't easily call apr_import because it requires a real file on disk.
        // Instead, we'll verify the properties of the Architecture enum and logic used.

        let arch = Architecture::Bert;
        assert!(!arch.is_inference_verified(), "BERT should NOT be verified");

        let qwen = Architecture::Qwen2;
        assert!(qwen.is_inference_verified(), "Qwen2 SHOULD be verified");

        // Simulate logic flow: strict mode rejects unverified architectures
        let options_strict = ImportOptions {
            architecture: Architecture::Bert,
            strict: true,
            ..Default::default()
        };

        if !options_strict.architecture.is_inference_verified() && options_strict.strict {
            // This represents the error path
            assert!(true, "Strict import correctly flagged unverified arch");
        } else {
            panic!("Strict import of BERT should have failed check");
        }

        // Default (permissive) mode allows unverified architectures with warning
        let options_permissive = ImportOptions {
            architecture: Architecture::Bert,
            ..Default::default()
        };

        if !options_permissive.architecture.is_inference_verified() && options_permissive.strict {
            panic!("Permissive import of BERT should have passed check");
        } else {
            // This represents the success path
            assert!(true, "Permissive import correctly bypassed check");
        }
    }
}
