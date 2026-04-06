//! Model Fixture Pattern for Test Standardization (Phase 54)
//!
//! Provides RAII-based model fixtures that:
//! - Create valid (or invalid) model files in temporary directories
//! - Provide paths and metadata to tests
//! - Automatically clean up on Drop
//!
//! # Example
//!
//! ```rust,ignore
//! use realizar::fixtures::{ModelFixture, ModelConfig};
//!
//! #[test]
//! fn test_model_loading() {
//!     // Setup: creates temp/tiny_llama.gguf
//!     let fixture = ModelFixture::gguf("tiny_llama", ModelConfig::tiny());
//!
//!     // Use the fixture
//!     let model = GGUFModel::from_file(fixture.path()).expect("model");
//!     assert_eq!(model.architecture(), Some("llama"));
//!
//!     // Teardown: automatic on Drop
//! }
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use tempfile::TempDir;

use crate::error::{RealizarError, Result};

// =============================================================================
// Model Configuration
// =============================================================================

/// Configuration for generating test models
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model architecture name
    pub architecture: String,
    /// Hidden dimension size
    pub hidden_dim: usize,
    /// Intermediate dimension (FFN)
    pub intermediate_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Context length
    pub context_length: usize,
    /// RoPE theta
    pub rope_theta: f32,
    /// Layer norm epsilon
    pub eps: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::tiny()
    }
}

impl ModelConfig {
    /// Tiny model for fast unit tests (64 hidden, 1 layer)
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            architecture: "llama".to_string(),
            hidden_dim: 64,
            intermediate_dim: 128,
            num_heads: 4,
            num_kv_heads: 4,
            num_layers: 1,
            vocab_size: 100,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-5,
        }
    }

    /// Small model for integration tests (256 hidden, 2 layers)
    #[must_use]
    pub fn small() -> Self {
        Self {
            architecture: "llama".to_string(),
            hidden_dim: 256,
            intermediate_dim: 512,
            num_heads: 8,
            num_kv_heads: 8,
            num_layers: 2,
            vocab_size: 1000,
            context_length: 1024,
            rope_theta: 10000.0,
            eps: 1e-5,
        }
    }

    /// GQA model (different num_heads vs num_kv_heads)
    #[must_use]
    pub fn gqa() -> Self {
        Self {
            architecture: "llama".to_string(),
            hidden_dim: 128,
            intermediate_dim: 256,
            num_heads: 8,
            num_kv_heads: 2, // GQA: 4 Q heads per KV head
            num_layers: 1,
            vocab_size: 100,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-5,
        }
    }

    /// Phi-2 style model (LayerNorm + GELU)
    #[must_use]
    pub fn phi() -> Self {
        Self {
            architecture: "phi2".to_string(),
            hidden_dim: 64,
            intermediate_dim: 128,
            num_heads: 4,
            num_kv_heads: 4,
            num_layers: 1,
            vocab_size: 100,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-5,
        }
    }

    /// Qwen style model
    #[must_use]
    pub fn qwen() -> Self {
        Self {
            architecture: "qwen2".to_string(),
            hidden_dim: 128,
            intermediate_dim: 256,
            num_heads: 8,
            num_kv_heads: 4,
            num_layers: 1,
            vocab_size: 100,
            context_length: 512,
            rope_theta: 1000000.0,
            eps: 1e-6,
        }
    }

    /// Builder: set architecture
    #[must_use]
    pub fn with_architecture(mut self, arch: &str) -> Self {
        self.architecture = arch.to_string();
        self
    }

    /// Builder: set hidden dimension
    #[must_use]
    pub fn with_hidden_dim(mut self, dim: usize) -> Self {
        self.hidden_dim = dim;
        self
    }

    /// Builder: set number of layers
    #[must_use]
    pub fn with_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    /// Builder: set vocab size
    #[must_use]
    pub fn with_vocab_size(mut self, n: usize) -> Self {
        self.vocab_size = n;
        self
    }

    /// Builder: set GQA config
    #[must_use]
    pub fn with_gqa(mut self, num_heads: usize, num_kv_heads: usize) -> Self {
        self.num_heads = num_heads;
        self.num_kv_heads = num_kv_heads;
        self
    }
}

// =============================================================================
// Model Format
// =============================================================================

/// Supported model formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// GGUF format (llama.cpp compatible)
    Gguf,
    /// APR format (Aprender native)
    Apr,
    /// SafeTensors format (HuggingFace)
    SafeTensors,
}

impl ModelFormat {
    /// Get file extension for format
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
            Self::Apr => "apr",
            Self::SafeTensors => "safetensors",
        }
    }
}

impl std::fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gguf => write!(f, "GGUF"),
            Self::Apr => write!(f, "APR"),
            Self::SafeTensors => write!(f, "SafeTensors"),
        }
    }
}

// =============================================================================
// Model Fixture
// =============================================================================

/// RAII-based model fixture for testing
///
/// Creates a model file in a temporary directory and automatically
/// cleans up when dropped.
pub struct ModelFixture {
    /// Path to the generated model file
    path: PathBuf,
    /// Temporary directory (keeps files alive until Drop)
    _temp_dir: TempDir,
    /// Model format
    format: ModelFormat,
    /// Configuration used to generate the model
    config: ModelConfig,
}

include!("mod_generate_apr_model.rs");
include!("mod_fixture_gguf.rs");
