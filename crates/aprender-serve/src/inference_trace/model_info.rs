
/// Model information for trace header
#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    /// Model name/path
    pub name: String,
    /// Number of layers
    pub num_layers: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Quantization type (e.g., "Q4_K_M")
    pub quant_type: Option<String>,
}
