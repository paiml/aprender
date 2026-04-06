//! Architecture detection from model weights.

use std::collections::HashMap;

/// Detected model architecture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Architecture {
    /// LLaMA family
    Llama,
    /// Mistral family
    Mistral,
    /// GPT-2/GPT-NeoX family
    Gpt,
    /// BERT family
    Bert,
    /// T5 family
    T5,
    /// Falcon family
    Falcon,
    /// Unknown architecture
    Unknown,
}

impl Architecture {
    /// Get architecture name as string.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::Gpt => "gpt",
            Self::Bert => "bert",
            Self::T5 => "t5",
            Self::Falcon => "falcon",
            Self::Unknown => "unknown",
        }
    }

    /// Check if architecture supports attention distillation.
    pub fn supports_attention_distill(&self) -> bool {
        matches!(
            self,
            Self::Llama | Self::Mistral | Self::Gpt | Self::Bert | Self::T5
        )
    }
}

/// Architecture detector from tensor names.
#[derive(Debug, Default)]
pub struct ArchitectureDetector {
    tensor_names: Vec<String>,
}

impl ArchitectureDetector {
    /// Create a new detector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add tensor names for detection.
    pub fn with_tensors(mut self, names: Vec<String>) -> Self {
        self.tensor_names = names;
        self
    }

    /// Detect architecture from tensor names.
    pub fn detect(&self) -> Architecture {
        let names_lower: Vec<String> = self.tensor_names.iter().map(|n| n.to_lowercase()).collect();

        // LLaMA patterns
        if names_lower
            .iter()
            .any(|n| n.contains("model.layers") && n.contains("self_attn"))
        {
            return Architecture::Llama;
        }

        // Mistral patterns (similar to LLaMA but with sliding window)
        if names_lower
            .iter()
            .any(|n| n.contains("mistral") || n.contains("sliding_window"))
        {
            return Architecture::Mistral;
        }

        // GPT patterns
        if names_lower
            .iter()
            .any(|n| n.contains("transformer.h") || n.contains("attn.c_attn"))
        {
            return Architecture::Gpt;
        }

        // BERT patterns
        if names_lower
            .iter()
            .any(|n| n.contains("bert.encoder") || n.contains("bert.pooler"))
        {
            return Architecture::Bert;
        }

        // T5 patterns
        if names_lower
            .iter()
            .any(|n| n.contains("encoder.block") && n.contains("decoder.block"))
        {
            return Architecture::T5;
        }

        // Falcon patterns
        if names_lower
            .iter()
            .any(|n| n.contains("transformer.h") && n.contains("self_attention.dense"))
        {
            return Architecture::Falcon;
        }

        Architecture::Unknown
    }

    /// Detect from a map of tensor name to shape.
    pub fn detect_from_shapes(&self, shapes: &HashMap<String, Vec<usize>>) -> ArchitectureInfo {
        let architecture = self.detect();

        // Extract dimensions from common tensors
        let hidden_dim = shapes
            .iter()
            .find(|(name, _)| name.contains("embed") || name.contains("wte"))
            // N-05 (Meyer DbC): 0 when no embedding tensor found, not hardcoded 4096.
            .map_or(0, |(_, shape)| shape.last().copied().unwrap_or(0));

        let num_layers = shapes
            .keys()
            .filter(|name| name.contains(".layers.") || name.contains(".h."))
            .filter_map(|name| name.split('.').find_map(|part| part.parse::<u32>().ok()))
            .max()
            // N-05: 0 when no layer tensors found, not hardcoded 32.
            .map_or(0, |n| n + 1);

        let vocab_size = shapes
            .iter()
            .find(|(name, _)| name.contains("embed_tokens") || name.contains("wte"))
            // N-05: 0 when no embedding tensor found, not hardcoded 32000.
            .map_or(0, |(_, shape)| shape.first().copied().unwrap_or(0));

        let num_heads = estimate_num_heads(hidden_dim);

        ArchitectureInfo {
            architecture,
            hidden_dim,
            num_layers,
            vocab_size,
            num_heads,
        }
    }
}

fn estimate_num_heads(hidden_dim: usize) -> u32 {
    // Common head dimensions: 64, 128
    let head_dim_64 = hidden_dim / 64;
    let head_dim_128 = hidden_dim / 128;

    // Prefer 64-dim heads for standard models
    if hidden_dim.is_multiple_of(64) && head_dim_64 <= 64 {
        head_dim_64 as u32
    } else if hidden_dim.is_multiple_of(128) {
        head_dim_128 as u32
    } else {
        // N-05 (Meyer DbC): return 0 (unknown) instead of hardcoded 32.
        // Callers must handle 0 as "could not estimate".
        0
    }
}

/// Detected architecture information.
#[derive(Debug, Clone)]
pub struct ArchitectureInfo {
    /// Detected architecture family
    pub architecture: Architecture,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of layers
    pub num_layers: u32,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Number of attention heads
    pub num_heads: u32,
}

impl ArchitectureInfo {
    /// Estimate total parameters.
    pub fn estimate_params(&self) -> u64 {
        // Simplified estimation
        let embed_params = self.vocab_size as u64 * self.hidden_dim as u64 * 2; // input + output
        let layer_params =
            u64::from(self.num_layers) * self.hidden_dim as u64 * self.hidden_dim as u64 * 12; // Q,K,V,O + MLP
        embed_params + layer_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_llama() {
        let detector = ArchitectureDetector::new().with_tensors(vec![
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "model.layers.0.mlp.gate_proj.weight".to_string(),
        ]);

        assert_eq!(detector.detect(), Architecture::Llama);
    }

    #[test]
    fn test_detect_bert() {
        let detector = ArchitectureDetector::new().with_tensors(vec![
            "bert.encoder.layer.0.attention.self.query.weight".to_string(),
            "bert.pooler.dense.weight".to_string(),
        ]);

        assert_eq!(detector.detect(), Architecture::Bert);
    }

    #[test]
    fn test_detect_gpt() {
        let detector = ArchitectureDetector::new().with_tensors(vec![
            "transformer.h.0.attn.c_attn.weight".to_string(),
            "transformer.wte.weight".to_string(),
        ]);

        assert_eq!(detector.detect(), Architecture::Gpt);
    }

    #[test]
    fn test_detect_unknown() {
        let detector =
            ArchitectureDetector::new().with_tensors(vec!["some.random.tensor.weight".to_string()]);

        assert_eq!(detector.detect(), Architecture::Unknown);
    }

    #[test]
    fn test_architecture_supports_distill() {
        assert!(Architecture::Llama.supports_attention_distill());
        assert!(Architecture::Bert.supports_attention_distill());
        assert!(!Architecture::Unknown.supports_attention_distill());
    }

    #[test]
    fn test_estimate_num_heads() {
        assert_eq!(estimate_num_heads(4096), 64);
        assert_eq!(estimate_num_heads(2048), 32);
        assert_eq!(estimate_num_heads(768), 12);
    }

    #[test]
    fn test_estimate_num_heads_128_dim() {
        // Test 128-dim head preference for large hidden dims
        assert_eq!(estimate_num_heads(8192), 64); // 8192/128 = 64 (uses 128-dim)
    }

    #[test]
    fn test_estimate_num_heads_default() {
        // N-05: Non-divisible hidden dim returns 0 (unknown), not hardcoded 32
        assert_eq!(estimate_num_heads(1000), 0);
    }

    #[test]
    fn test_architecture_name() {
        assert_eq!(Architecture::Llama.name(), "llama");
        assert_eq!(Architecture::Mistral.name(), "mistral");
        assert_eq!(Architecture::Gpt.name(), "gpt");
        assert_eq!(Architecture::Bert.name(), "bert");
        assert_eq!(Architecture::T5.name(), "t5");
        assert_eq!(Architecture::Falcon.name(), "falcon");
        assert_eq!(Architecture::Unknown.name(), "unknown");
    }

    #[test]
    fn test_detect_mistral() {
        let detector = ArchitectureDetector::new()
            .with_tensors(vec!["model.mistral.layers.0.weight".to_string()]);
        assert_eq!(detector.detect(), Architecture::Mistral);
    }

    #[test]
    fn test_gpt_takes_precedence_over_falcon() {
        // GPT is checked before Falcon in detection order
        // transformer.h matches GPT, even with self_attention.dense
        let detector = ArchitectureDetector::new().with_tensors(vec![
            "transformer.h.0.self_attention.dense.weight".to_string(),
        ]);
        // GPT is detected first due to matching transformer.h
        assert_eq!(detector.detect(), Architecture::Gpt);
    }

    #[test]
    fn test_architecture_info_estimate_params() {
        let info = ArchitectureInfo {
            architecture: Architecture::Llama,
            hidden_dim: 4096,
            num_layers: 32,
            vocab_size: 32000,
            num_heads: 32,
        };
        let params = info.estimate_params();
        // Should be in billions for 7B-class model
        assert!(params > 1_000_000_000);
        assert!(params < 20_000_000_000);
    }

    #[test]
    fn test_detector_detect_from_shapes() {
        let mut shapes = HashMap::new();
        shapes.insert("model.embed_tokens.weight".to_string(), vec![32000, 4096]);
        shapes.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            vec![4096, 4096],
        );
        shapes.insert(
            "model.layers.31.mlp.gate_proj.weight".to_string(),
            vec![11008, 4096],
        );

        let detector = ArchitectureDetector::new().with_tensors(shapes.keys().cloned().collect());
        let info = detector.detect_from_shapes(&shapes);

        assert_eq!(info.architecture, Architecture::Llama);
        assert_eq!(info.hidden_dim, 4096);
        assert_eq!(info.num_layers, 32);
        assert_eq!(info.vocab_size, 32000);
    }

    #[test]
    fn test_architecture_default_new() {
        let detector = ArchitectureDetector::new();
        // Empty detector should return Unknown
        assert_eq!(detector.detect(), Architecture::Unknown);
    }

    // =========================================================================
    // FALSIFY tests — contract violation sweep (N-05)
    // =========================================================================

    #[test]
    fn test_falsify_n05_estimate_num_heads_non_divisible_returns_zero() {
        // N-05: estimate_num_heads must return 0 (unknown) for non-standard
        // hidden dims, never a hardcoded guess like 32.
        for dim in [7, 13, 97, 1000, 1001, 3] {
            assert_eq!(
                estimate_num_heads(dim),
                0,
                "estimate_num_heads({dim}) must be 0 (unknown), not a hardcoded fallback"
            );
        }
    }

    #[test]
    fn test_falsify_n05_detect_from_shapes_missing_tensors_returns_zero_dims() {
        // N-05: When no embedding/layer tensors are found, dimensions must be 0,
        // not hardcoded architecture-specific defaults.
        let shapes = HashMap::new();
        let detector = ArchitectureDetector::new().with_tensors(vec![]);
        let info = detector.detect_from_shapes(&shapes);

        assert_eq!(
            info.hidden_dim, 0,
            "hidden_dim must be 0 when no embedding tensor found"
        );
        assert_eq!(
            info.num_layers, 0,
            "num_layers must be 0 when no layer tensors found"
        );
        assert_eq!(
            info.vocab_size, 0,
            "vocab_size must be 0 when no embedding tensor found"
        );
        assert_eq!(
            info.num_heads, 0,
            "num_heads must be 0 when hidden_dim is 0"
        );
    }

    #[test]
    fn test_t5_not_detected_without_both() {
        // T5 needs both encoder.block AND decoder.block
        let detector =
            ArchitectureDetector::new().with_tensors(vec!["encoder.block.0.weight".to_string()]);
        // Should not detect as T5 with only encoder
        assert_ne!(detector.detect(), Architecture::T5);
    }
}
