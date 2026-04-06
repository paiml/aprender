
// =============================================================================
// Falsification Tests (Format×Device Matrix)
// =============================================================================

pub mod falsification_tests;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixture_gguf_tiny() {
        let fixture = ModelFixture::gguf("test_tiny", ModelConfig::tiny());
        assert!(fixture.path().exists());
        assert_eq!(fixture.format(), ModelFormat::Gguf);
        assert_eq!(fixture.config().hidden_dim, 64);

        let bytes = fixture.read_bytes().expect("read");
        assert!(!bytes.is_empty());
        // Verify GGUF magic
        assert_eq!(&bytes[0..4], &0x46554747u32.to_le_bytes());
    }

    #[test]
    fn test_fixture_gguf_gqa() {
        let fixture = ModelFixture::gguf("test_gqa", ModelConfig::gqa());
        assert!(fixture.path().exists());
        assert_eq!(fixture.config().num_heads, 8);
        assert_eq!(fixture.config().num_kv_heads, 2);
    }

    #[test]
    fn test_fixture_gguf_invalid_magic() {
        let fixture = ModelFixture::gguf_invalid_magic("bad_magic");
        assert!(fixture.path().exists());

        let bytes = fixture.read_bytes().expect("read");
        // Verify invalid magic
        assert_eq!(&bytes[0..4], &0xDEADBEEFu32.to_le_bytes());
    }

    #[test]
    fn test_fixture_gguf_invalid_version() {
        let fixture = ModelFixture::gguf_invalid_version("bad_version");
        let bytes = fixture.read_bytes().expect("read");
        // Verify version 999
        assert_eq!(&bytes[4..8], &999u32.to_le_bytes());
    }

    #[test]
    fn test_fixture_safetensors() {
        let fixture = ModelFixture::safetensors("test_st", ModelConfig::tiny());
        assert!(fixture.path().exists());
        assert_eq!(fixture.format(), ModelFormat::SafeTensors);

        let bytes = fixture.read_bytes().expect("read");
        assert!(!bytes.is_empty());
        // SafeTensors starts with JSON header length (u64 LE)
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().expect("header_len"));
        assert!(header_len > 0 && header_len < 1_000_000);
    }

    #[test]
    fn test_fixture_apr() {
        let fixture = ModelFixture::apr("test_apr", ModelConfig::tiny());
        assert!(fixture.path().exists());
        assert_eq!(fixture.format(), ModelFormat::Apr);

        let bytes = fixture.read_bytes().expect("read");
        // Verify APR magic
        assert_eq!(&bytes[0..4], b"APR\x00");
    }

    #[test]
    fn test_fixture_cleanup_on_drop() {
        let path = {
            let fixture = ModelFixture::gguf("test_cleanup", ModelConfig::tiny());
            let p = fixture.path().to_path_buf();
            assert!(p.exists());
            p
        };
        // After fixture is dropped, file should be cleaned up
        assert!(!path.exists());
    }

    #[test]
    fn test_model_config_builder() {
        let config = ModelConfig::tiny()
            .with_architecture("qwen2")
            .with_hidden_dim(256)
            .with_layers(4)
            .with_vocab_size(50000)
            .with_gqa(16, 4);

        assert_eq!(config.architecture, "qwen2");
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.num_layers, 4);
        assert_eq!(config.vocab_size, 50000);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.num_kv_heads, 4);
    }

    #[test]
    fn test_format_extension() {
        assert_eq!(ModelFormat::Gguf.extension(), "gguf");
        assert_eq!(ModelFormat::Apr.extension(), "apr");
        assert_eq!(ModelFormat::SafeTensors.extension(), "safetensors");
    }
}
