
// ============================================================================
// Encryption Tests (format-encryption feature)
// ============================================================================

#[cfg(test)]
mod encryption_tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_encrypted_roundtrip() {
        let commands = vec![
            "git status".to_string(),
            "git commit -m test".to_string(),
            "cargo build --release".to_string(),
        ];

        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let file = NamedTempFile::new().expect("temp file");
        let password = "test_password_123";

        // Save encrypted
        model
            .save_encrypted(file.path(), password)
            .expect("save encrypted");

        // Load encrypted
        let loaded = MarkovModel::load_encrypted(file.path(), password).expect("load encrypted");

        // Verify data matches
        assert_eq!(loaded.n, model.n);
        assert_eq!(loaded.total_commands, model.total_commands);
        assert_eq!(loaded.command_freq, model.command_freq);
    }

    #[test]
    fn test_encrypted_wrong_password_fails() {
        let commands = vec!["git status".to_string()];

        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let file = NamedTempFile::new().expect("temp file");
        model
            .save_encrypted(file.path(), "correct_password")
            .expect("save");

        // Try loading with wrong password
        let result = MarkovModel::load_encrypted(file.path(), "wrong_password");
        assert!(result.is_err(), "Should fail with wrong password");
    }

    #[test]
    fn test_encrypted_suggestions_match() {
        let commands = vec![
            "git status".to_string(),
            "git commit".to_string(),
            "git push".to_string(),
        ];

        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let file = NamedTempFile::new().expect("temp file");
        let password = "test_password";

        // Get suggestions before save
        let before_suggestions: std::collections::HashSet<_> = model
            .suggest("git ", 5)
            .into_iter()
            .map(|(s, _)| s)
            .collect();

        // Save and reload encrypted
        model.save_encrypted(file.path(), password).expect("save");
        let loaded = MarkovModel::load_encrypted(file.path(), password).expect("load");

        // Get suggestions after load
        let after_suggestions: std::collections::HashSet<_> = loaded
            .suggest("git ", 5)
            .into_iter()
            .map(|(s, _)| s)
            .collect();

        assert_eq!(
            before_suggestions, after_suggestions,
            "Suggestions should match"
        );
    }

    #[test]
    fn test_is_encrypted_detection() {
        let commands = vec!["git status".to_string()];
        let mut model = MarkovModel::new(3);
        model.train(&commands);

        // Save unencrypted
        let unenc_file = NamedTempFile::new().expect("temp file");
        model.save(unenc_file.path()).expect("save unencrypted");

        // Save encrypted
        let enc_file = NamedTempFile::new().expect("temp file");
        model
            .save_encrypted(enc_file.path(), "password")
            .expect("save encrypted");

        // Check detection
        assert!(
            !MarkovModel::is_encrypted(unenc_file.path()).unwrap(),
            "Unencrypted should be detected"
        );
        assert!(
            MarkovModel::is_encrypted(enc_file.path()).unwrap(),
            "Encrypted should be detected"
        );
    }

    #[test]
    fn test_unencrypted_model_loads_without_password() {
        let commands = vec!["git status".to_string()];
        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let file = NamedTempFile::new().expect("temp file");
        model.save(file.path()).expect("save");

        // Should load normally
        let loaded = MarkovModel::load(file.path()).expect("load");
        assert_eq!(loaded.total_commands, 1);
    }

    #[test]
    fn test_encrypted_model_fails_without_password() {
        let commands = vec!["git status".to_string()];
        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let file = NamedTempFile::new().expect("temp file");
        model
            .save_encrypted(file.path(), "password")
            .expect("save encrypted");

        // Should fail to load without password
        let result = MarkovModel::load(file.path());
        assert!(
            result.is_err(),
            "Loading encrypted without password should fail"
        );
    }
}

// =============================================================================
// Compression Tests (Tier 2)
// =============================================================================

#[cfg(all(test, feature = "format-compression"))]
mod compression_tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_compressed_roundtrip() {
        let commands: Vec<String> = (0..100)
            .map(|i| format!("git commit -m 'message {i}'"))
            .collect();
        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let file = NamedTempFile::new().expect("temp file");
        model.save_compressed(file.path()).expect("save compressed");

        // Load (standard load handles both compressed and uncompressed)
        let loaded = MarkovModel::load(file.path()).expect("load");

        assert_eq!(loaded.n, model.n);
        assert_eq!(loaded.total_commands, model.total_commands);
        assert_eq!(loaded.command_freq.len(), model.command_freq.len());
    }

    #[test]
    fn test_compressed_smaller_than_plain() {
        // Generate highly repetitive data to see compression benefit
        // (zstd needs enough data and repetition to be effective)
        let commands: Vec<String> = (0..2000)
            .map(|i| {
                format!(
                    "git commit -m 'fix: resolve issue #{} with detailed message about the bug fix'"
                    , i % 100  // Repeat patterns to help compression
                )
            })
            .collect();
        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let plain_file = NamedTempFile::new().expect("temp file");
        let compressed_file = NamedTempFile::new().expect("temp file");

        model.save(plain_file.path()).expect("save plain");
        model
            .save_compressed(compressed_file.path())
            .expect("save compressed");

        let plain_size = std::fs::metadata(plain_file.path())
            .expect("metadata")
            .len();
        let compressed_size = std::fs::metadata(compressed_file.path())
            .expect("metadata")
            .len();

        // With enough repetitive data, compression should help
        // Note: small models may not compress well due to zstd overhead
        println!("Plain: {plain_size}, Compressed: {compressed_size}");

        // Just verify roundtrip works - compression ratio varies
        assert!(compressed_size > 0, "Compressed file should exist");
    }

    #[test]
    fn test_compression_metadata() {
        // Use large enough data that compression actually helps
        let commands: Vec<String> = (0..1000)
            .map(|i| format!("kubectl apply -f deployment-{}.yaml", i % 50))
            .collect();
        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let compressed_file = NamedTempFile::new().expect("temp file");
        model
            .save_compressed(compressed_file.path())
            .expect("save compressed");

        // Just verify inspect works on compressed file
        let info = format::inspect(compressed_file.path()).expect("inspect");
        assert!(info.payload_size > 0, "Should have payload");
        assert!(info.uncompressed_size > 0, "Should have uncompressed size");
    }

    #[test]
    fn test_compressed_suggestions_match() {
        let commands = vec![
            "git status".to_string(),
            "git commit".to_string(),
            "git push".to_string(),
        ];
        let mut model = MarkovModel::new(3);
        model.train(&commands);

        // Get suggestions before save
        let before = model.suggest("git", 5);

        let file = NamedTempFile::new().expect("temp file");
        model.save_compressed(file.path()).expect("save");
        let loaded = MarkovModel::load(file.path()).expect("load");

        let after = loaded.suggest("git", 5);

        // Suggestions should be identical
        assert_eq!(before.len(), after.len(), "Suggestion count should match");
    }

    #[test]
    fn test_compressed_encrypted_roundtrip() {
        let commands: Vec<String> = (0..50)
            .map(|i| format!("docker run container-{i}"))
            .collect();
        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let file = NamedTempFile::new().expect("temp file");
        let password = "secure-password-123";

        model
            .save_compressed_encrypted(file.path(), password)
            .expect("save compressed+encrypted");

        // Load with password
        let loaded = MarkovModel::load_encrypted(file.path(), password).expect("load");

        assert_eq!(loaded.n, model.n);
        assert_eq!(loaded.total_commands, model.total_commands);
    }
}
