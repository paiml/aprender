//! LoRA adapter metadata

/// Adapter metadata (without weights)
#[derive(Debug, Clone)]
pub struct AdapterMetadata {
    pub version: String,
    pub rank: usize,
    pub alpha: f32,
    pub d_out: usize,
    pub d_in: usize,
    pub scale: f32,
    pub num_params: usize,
}
