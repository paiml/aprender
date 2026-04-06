
impl ModelFixture for SafetensorsFixture {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn format(&self) -> ModelFormat {
        ModelFormat::Safetensors
    }

    fn quant_type(&self) -> QuantType {
        self.weights.quant_type
    }

    fn forward(&self, _device: Device, _tokens: &[u32]) -> Result<Vec<f32>> {
        let output_size = self.config.vocab_size;
        let mut logits = vec![0.0f32; output_size];

        for (i, logit) in logits.iter_mut().enumerate() {
            *logit = ((i as f32) / (output_size as f32) - 0.5) * 2.0;
        }

        Ok(logits)
    }

    fn embed(&self, _device: Device, token: u32) -> Result<Vec<f32>> {
        let mut embedding = vec![0.0f32; self.config.hidden_dim];
        let scale = 1.0 / (self.config.hidden_dim as f32).sqrt();

        for (i, val) in embedding.iter_mut().enumerate() {
            *val = ((token as usize * 17 + i * 13) % 1000) as f32 / 1000.0 * scale;
        }

        Ok(embedding)
    }

    fn to_bytes(&self) -> Result<Vec<u8>> {
        // Simplified safetensors format
        let header = serde_json::json!({
            "__metadata__": {
                "format": "pt",
                "num_heads": self.config.num_heads,
                "num_kv_heads": self.config.num_kv_heads,
            },
            "model.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [self.config.vocab_size, self.config.hidden_dim],
                "data_offsets": [0, self.weights.embed_weights.len()]
            }
        });

        let header_bytes = serde_json::to_vec(&header).map_err(|e| RealizarError::FormatError {
            reason: format!("Safetensors header serialization failed: {}", e),
        })?;
        let header_len = header_bytes.len() as u64;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&header_len.to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&self.weights.embed_weights);

        Ok(bytes)
    }

    fn convert_to(&self, target: ModelFormat) -> Result<Box<dyn ModelFixture>> {
        match target {
            ModelFormat::GGUF => Ok(Box::new(GgufFixture {
                config: self.config.clone(),
                weights: self.weights.clone(),
            })),
            ModelFormat::APR => Ok(Box::new(AprFixture {
                config: self.config.clone(),
                weights: self.weights.clone(),
            })),
            ModelFormat::Safetensors => Ok(Box::new(SafetensorsFixture {
                config: self.config.clone(),
                weights: self.weights.clone(),
            })),
            ModelFormat::PyTorch => Err(RealizarError::UnsupportedOperation {
                operation: "convert_to".to_string(),
                reason: "Safetensors to PyTorch conversion not supported".to_string(),
            }),
        }
    }

    fn memory_bytes(&self) -> usize {
        self.weights.total_bytes()
    }
}

/// Create fixture from format and constructor input
pub fn create_fixture(
    format: ModelFormat,
    input: &ConstructorInput,
) -> Result<Box<dyn ModelFixture>> {
    match format {
        ModelFormat::GGUF => Ok(Box::new(GgufFixture::from_constructor(input))),
        ModelFormat::APR => Ok(Box::new(AprFixture::from_constructor(input))),
        ModelFormat::Safetensors => Ok(Box::new(SafetensorsFixture::from_constructor(input))),
        ModelFormat::PyTorch => Err(RealizarError::UnsupportedOperation {
            operation: "create_fixture".to_string(),
            reason: "PyTorch fixtures not yet implemented".to_string(),
        }),
    }
}

/// Helper to write GGUF key-value pair
fn write_gguf_kv(bytes: &mut Vec<u8>, key: &str, value: u32) {
    // Key length
    bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
    // Key bytes
    bytes.extend_from_slice(key.as_bytes());
    // Value type (4 = U32)
    bytes.extend_from_slice(&4u32.to_le_bytes());
    // Value
    bytes.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_fixture_creation() {
        let fixture = GgufFixture::tiny_gqa();
        assert_eq!(fixture.config().num_heads, 4);
        assert_eq!(fixture.config().num_kv_heads, 2);
        assert!(fixture.config().is_gqa());
    }

    #[test]
    fn test_gguf_to_apr_conversion() {
        let gguf = GgufFixture::tiny_gqa();
        let apr = gguf.convert_to(ModelFormat::APR).expect("apr");

        assert_eq!(apr.format(), ModelFormat::APR);
        assert_eq!(apr.config().num_heads, gguf.config().num_heads);
        assert_eq!(apr.config().num_kv_heads, gguf.config().num_kv_heads);
    }

    #[test]
    fn test_gguf_round_trip() {
        let original = GgufFixture::tiny_gqa();
        let apr = original.convert_to(ModelFormat::APR).expect("apr");
        let back = apr.convert_to(ModelFormat::GGUF).expect("back");

        assert_eq!(back.config().num_heads, original.config().num_heads);
        assert_eq!(back.config().num_kv_heads, original.config().num_kv_heads);
        assert_eq!(back.config().hidden_dim, original.config().hidden_dim);
    }

    #[test]
    fn test_safetensors_quant_fallback() {
        // Q4_K is not supported by Safetensors, should fallback to F32
        let fixture = SafetensorsFixture::new(ModelConfig::tiny(), QuantType::Q4_K, 42);
        assert_eq!(fixture.quant_type(), QuantType::F32);
    }

    #[test]
    fn test_fixture_serialization() {
        let gguf = GgufFixture::tiny_gqa();
        let bytes = gguf.to_bytes().expect("bytes");

        // Should start with GGUF magic
        assert_eq!(&bytes[0..4], b"GGUF");
    }

    #[test]
    fn test_apr_serialization() {
        let apr = AprFixture::tiny_gqa();
        let bytes = apr.to_bytes().expect("bytes");

        // Should start with APR magic
        assert_eq!(&bytes[0..4], b"APR\x02");
    }

    #[test]
    fn test_forward_output_size() {
        let fixture = GgufFixture::tiny_gqa();
        let tokens = vec![1, 2, 3];
        let output = fixture.forward(Device::Cpu, &tokens).expect("output");

        assert_eq!(output.len(), fixture.config().vocab_size);
    }

    #[test]
    fn test_embed_output_size() {
        let fixture = GgufFixture::tiny_gqa();
        let embedding = fixture.embed(Device::Cpu, 42).expect("embedding");

        assert_eq!(embedding.len(), fixture.config().hidden_dim);
    }

    #[test]
    fn test_create_fixture_dispatch() {
        let input = ConstructorInput::new(ModelConfig::tiny());

        let gguf = create_fixture(ModelFormat::GGUF, &input).expect("gguf");
        assert_eq!(gguf.format(), ModelFormat::GGUF);

        let apr = create_fixture(ModelFormat::APR, &input).expect("apr");
        assert_eq!(apr.format(), ModelFormat::APR);
    }

    #[test]
    fn test_memory_bytes() {
        let fixture = GgufFixture::tiny_gqa();
        let bytes = fixture.memory_bytes();

        // Should be non-zero
        assert!(bytes > 0);

        // Larger config should use more memory
        let large = GgufFixture::new(ModelConfig::small(), QuantType::Q4_0, 42);
        assert!(large.memory_bytes() > bytes);
    }

    #[test]
    fn test_gguf_tiny_mha() {
        let fixture = GgufFixture::tiny_mha();
        assert!(!fixture.config().is_gqa());
        assert_eq!(fixture.config().num_kv_heads, fixture.config().num_heads);
    }

    #[test]
    fn test_gguf_small() {
        let fixture = GgufFixture::small();
        assert_eq!(fixture.config().hidden_dim, 256);
    }

    #[test]
    fn test_gguf_to_safetensors_conversion() {
        let gguf = GgufFixture::tiny_gqa();
        let st = gguf.convert_to(ModelFormat::Safetensors).expect("st");
        assert_eq!(st.format(), ModelFormat::Safetensors);
    }

    #[test]
    fn test_unsupported_conversions() {
        let gguf = GgufFixture::tiny_gqa();
        assert!(gguf.convert_to(ModelFormat::PyTorch).is_err());

        let apr = AprFixture::tiny_gqa();
        assert!(apr.convert_to(ModelFormat::PyTorch).is_err());

        let st = SafetensorsFixture::tiny();
        assert!(st.convert_to(ModelFormat::PyTorch).is_err());
    }

    #[test]
    fn test_clone_conversions() {
        let gguf = GgufFixture::tiny_gqa();
        let gguf2 = gguf.convert_to(ModelFormat::GGUF).expect("gguf2");
        assert_eq!(gguf2.format(), ModelFormat::GGUF);

        let apr = AprFixture::tiny_gqa();
        let apr2 = apr.convert_to(ModelFormat::APR).expect("apr2");
        assert_eq!(apr2.format(), ModelFormat::APR);

        let st = SafetensorsFixture::tiny();
        let st2 = st.convert_to(ModelFormat::Safetensors).expect("st2");
        assert_eq!(st2.format(), ModelFormat::Safetensors);
    }

    #[test]
    fn test_safetensors_serialization() {
        let st = SafetensorsFixture::tiny();
        let bytes = st.to_bytes().expect("bytes");
        assert!(bytes.len() > 8);
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().expect("header_len"));
        assert!(header_len > 0);
    }

    #[test]
    fn test_create_fixture_unsupported() {
        let input = ConstructorInput::new(ModelConfig::tiny());
        assert!(create_fixture(ModelFormat::PyTorch, &input).is_err());
    }

    #[test]
    fn test_apr_to_safetensors_conversion() {
        let apr = AprFixture::tiny_gqa();
        let st = apr.convert_to(ModelFormat::Safetensors).expect("st");
        assert_eq!(st.format(), ModelFormat::Safetensors);
    }
}
