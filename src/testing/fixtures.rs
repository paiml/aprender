//! Model Fixtures for Testing
//!
//! Concrete fixture implementations for each model format.
//! Inspired by PyTorch's `ModuleInfo` / `ModuleInput` pattern.

use super::{
    generators::{ModelWeights, SyntheticWeightGenerator},
    ConstructorInput, Device, ModelConfig, ModelFormat, QuantType,
};
use crate::error::RealizarError;
use crate::Result;

/// Trait for model fixtures that can be tested
pub trait ModelFixture: Send + Sync {
    /// Get model configuration
    fn config(&self) -> &ModelConfig;

    /// Get model format
    fn format(&self) -> ModelFormat;

    /// Get quantization type
    fn quant_type(&self) -> QuantType;

    /// Run forward pass on specified device
    fn forward(&self, device: Device, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Get embedding for a token
    fn embed(&self, device: Device, token: u32) -> Result<Vec<f32>>;

    /// Serialize to bytes
    fn to_bytes(&self) -> Result<Vec<u8>>;

    /// Convert to another format
    fn convert_to(&self, target: ModelFormat) -> Result<Box<dyn ModelFixture>>;

    /// Memory footprint in bytes
    fn memory_bytes(&self) -> usize;
}

/// GGUF format fixture
pub struct GgufFixture {
    config: ModelConfig,
    weights: ModelWeights,
}

impl GgufFixture {
    /// Create from configuration and seed
    pub fn new(config: ModelConfig, quant: QuantType, seed: u64) -> Self {
        let gen = SyntheticWeightGenerator::new(seed);
        let weights = gen.generate_model_weights(&config, quant);
        Self { config, weights }
    }

    /// Create tiny GQA model for fast tests
    pub fn tiny_gqa() -> Self {
        Self::new(ModelConfig::tiny(), QuantType::Q4_0, 42)
    }

    /// Create tiny MHA model (no GQA)
    pub fn tiny_mha() -> Self {
        let mut config = ModelConfig::tiny();
        config.num_kv_heads = config.num_heads; // MHA
        Self::new(config, QuantType::Q4_0, 42)
    }

    /// Create small model for integration tests
    pub fn small() -> Self {
        Self::new(ModelConfig::small(), QuantType::Q4_K, 42)
    }

    /// Create from constructor input
    pub fn from_constructor(input: &ConstructorInput) -> Self {
        Self::new(
            input.config.clone(),
            input.quantization.unwrap_or(QuantType::F32),
            input.weights_seed,
        )
    }
}

impl ModelFixture for GgufFixture {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn format(&self) -> ModelFormat {
        ModelFormat::GGUF
    }

    fn quant_type(&self) -> QuantType {
        self.weights.quant_type
    }

    fn forward(&self, _device: Device, _tokens: &[u32]) -> Result<Vec<f32>> {
        // Placeholder - actual implementation would use GGUFTransformer
        // For now, return deterministic output based on config
        let output_size = self.config.vocab_size;
        let mut logits = vec![0.0f32; output_size];

        // Simple deterministic logits based on token values
        for (i, logit) in logits.iter_mut().enumerate() {
            *logit = ((i as f32) / (output_size as f32) - 0.5) * 2.0;
        }

        Ok(logits)
    }

    fn embed(&self, _device: Device, token: u32) -> Result<Vec<f32>> {
        // Return deterministic embedding
        let mut embedding = vec![0.0f32; self.config.hidden_dim];
        let scale = 1.0 / (self.config.hidden_dim as f32).sqrt();

        for (i, val) in embedding.iter_mut().enumerate() {
            *val = ((token as usize * 17 + i * 13) % 1000) as f32 / 1000.0 * scale;
        }

        Ok(embedding)
    }

    fn to_bytes(&self) -> Result<Vec<u8>> {
        // Build minimal GGUF format
        let mut bytes = Vec::new();

        // GGUF magic: "GGUF"
        bytes.extend_from_slice(b"GGUF");

        // Version (3)
        bytes.extend_from_slice(&3u32.to_le_bytes());

        // Tensor count
        let tensor_count = 1 + self.config.num_layers * 9 + 2; // embed + layers + output
        bytes.extend_from_slice(&(tensor_count as u64).to_le_bytes());

        // Metadata KV count (simplified)
        bytes.extend_from_slice(&10u64.to_le_bytes());

        // Add key metadata
        // num_heads
        write_gguf_kv(
            &mut bytes,
            "llama.attention.head_count",
            self.config.num_heads as u32,
        );
        // num_kv_heads
        write_gguf_kv(
            &mut bytes,
            "llama.attention.head_count_kv",
            self.config.num_kv_heads as u32,
        );
        // hidden_dim
        write_gguf_kv(
            &mut bytes,
            "llama.embedding_length",
            self.config.hidden_dim as u32,
        );
        // num_layers
        write_gguf_kv(
            &mut bytes,
            "llama.block_count",
            self.config.num_layers as u32,
        );
        // vocab_size
        write_gguf_kv(
            &mut bytes,
            "llama.vocab_size",
            self.config.vocab_size as u32,
        );

        Ok(bytes)
    }

    fn convert_to(&self, target: ModelFormat) -> Result<Box<dyn ModelFixture>> {
        match target {
            ModelFormat::APR => Ok(Box::new(AprFixture::from_gguf(self)?)),
            ModelFormat::Safetensors => Ok(Box::new(SafetensorsFixture::from_gguf(self)?)),
            ModelFormat::GGUF => {
                // Clone self
                Ok(Box::new(GgufFixture {
                    config: self.config.clone(),
                    weights: self.weights.clone(),
                }))
            },
            ModelFormat::PyTorch => Err(RealizarError::UnsupportedOperation {
                operation: "convert_to".to_string(),
                reason: "GGUF to PyTorch conversion not supported".to_string(),
            }),
        }
    }

    fn memory_bytes(&self) -> usize {
        self.weights.total_bytes()
    }
}

/// APR format fixture
pub struct AprFixture {
    config: ModelConfig,
    weights: ModelWeights,
}

impl AprFixture {
    /// Create from configuration
    pub fn new(config: ModelConfig, quant: QuantType, seed: u64) -> Self {
        let gen = SyntheticWeightGenerator::new(seed);
        let weights = gen.generate_model_weights(&config, quant);
        Self { config, weights }
    }

    /// Create tiny GQA model
    pub fn tiny_gqa() -> Self {
        Self::new(ModelConfig::tiny(), QuantType::Q4_0, 42)
    }

    /// Create from GGUF fixture
    pub fn from_gguf(gguf: &GgufFixture) -> Result<Self> {
        Ok(Self {
            config: gguf.config.clone(),
            weights: gguf.weights.clone(),
        })
    }

    /// Create from constructor input
    pub fn from_constructor(input: &ConstructorInput) -> Self {
        Self::new(
            input.config.clone(),
            input.quantization.unwrap_or(QuantType::F32),
            input.weights_seed,
        )
    }
}

impl ModelFixture for AprFixture {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn format(&self) -> ModelFormat {
        ModelFormat::APR
    }

    fn quant_type(&self) -> QuantType {
        self.weights.quant_type
    }

    fn forward(&self, _device: Device, _tokens: &[u32]) -> Result<Vec<f32>> {
        // Same placeholder as GGUF
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
        let mut bytes = Vec::new();

        // APR magic: "APR\x02" (version 2)
        bytes.extend_from_slice(b"APR\x02");

        // Header size
        bytes.extend_from_slice(&64u32.to_le_bytes());

        // Metadata as JSON
        let metadata = serde_json::json!({
            "num_heads": self.config.num_heads,
            "num_kv_heads": self.config.num_kv_heads,
            "hidden_size": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "vocab_size": self.config.vocab_size,
            "intermediate_size": self.config.intermediate_dim,
            "rope_theta": self.config.rope_theta,
        });
        let metadata_bytes =
            serde_json::to_vec(&metadata).map_err(|e| RealizarError::FormatError {
                reason: format!("APR metadata serialization failed: {}", e),
            })?;

        // Metadata offset and size
        bytes.extend_from_slice(&64u64.to_le_bytes()); // offset
        bytes.extend_from_slice(&(metadata_bytes.len() as u32).to_le_bytes()); // size

        // Pad header to 64 bytes
        while bytes.len() < 64 {
            bytes.push(0);
        }

        // Append metadata
        bytes.extend_from_slice(&metadata_bytes);

        Ok(bytes)
    }

    fn convert_to(&self, target: ModelFormat) -> Result<Box<dyn ModelFixture>> {
        match target {
            ModelFormat::GGUF => Ok(Box::new(GgufFixture {
                config: self.config.clone(),
                weights: self.weights.clone(),
            })),
            ModelFormat::Safetensors => Ok(Box::new(SafetensorsFixture {
                config: self.config.clone(),
                weights: self.weights.clone(),
            })),
            ModelFormat::APR => Ok(Box::new(AprFixture {
                config: self.config.clone(),
                weights: self.weights.clone(),
            })),
            ModelFormat::PyTorch => Err(RealizarError::UnsupportedOperation {
                operation: "convert_to".to_string(),
                reason: "APR to PyTorch conversion not supported".to_string(),
            }),
        }
    }

    fn memory_bytes(&self) -> usize {
        self.weights.total_bytes()
    }
}

/// Safetensors format fixture
pub struct SafetensorsFixture {
    config: ModelConfig,
    weights: ModelWeights,
}

impl SafetensorsFixture {
    /// Create from configuration
    pub fn new(config: ModelConfig, quant: QuantType, seed: u64) -> Self {
        // Safetensors only supports F32/F16/BF16
        let actual_quant = if quant.supported_by(ModelFormat::Safetensors) {
            quant
        } else {
            QuantType::F32
        };

        let gen = SyntheticWeightGenerator::new(seed);
        let weights = gen.generate_model_weights(&config, actual_quant);
        Self { config, weights }
    }

    /// Create tiny model
    pub fn tiny() -> Self {
        Self::new(ModelConfig::tiny(), QuantType::F32, 42)
    }

    /// Create from GGUF fixture
    pub fn from_gguf(gguf: &GgufFixture) -> Result<Self> {
        // Dequantize if needed
        let quant = if gguf.quant_type().supported_by(ModelFormat::Safetensors) {
            gguf.quant_type()
        } else {
            QuantType::F32
        };

        let gen = SyntheticWeightGenerator::new(42);
        let weights = gen.generate_model_weights(&gguf.config, quant);

        Ok(Self {
            config: gguf.config.clone(),
            weights,
        })
    }

    /// Create from constructor input
    pub fn from_constructor(input: &ConstructorInput) -> Self {
        Self::new(
            input.config.clone(),
            input.quantization.unwrap_or(QuantType::F32),
            input.weights_seed,
        )
    }
}

include!("safetensors_fixture.rs");
