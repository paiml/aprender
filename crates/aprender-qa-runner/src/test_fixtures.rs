//! Test fixtures for apr-qa-runner integration tests
//!
//! Provides minimal "pygmy" model files that can be used for integration testing
//! without requiring real model files.

use std::io::Write;
use std::path::PathBuf;
use tempfile::TempDir;

/// GGUF Magic number
const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"
/// GGUF Version 3
const GGUF_VERSION_V3: u32 = 3;
/// GGUF alignment
const GGUF_ALIGNMENT: usize = 32;
/// F32 tensor type
const GGUF_TYPE_F32: u32 = 0;
/// Q4_0 tensor type
const GGUF_TYPE_Q4_0: u32 = 2;

/// Builder for creating valid GGUF v3 files in memory
pub struct GgufBuilder {
    metadata: Vec<(String, u32, Vec<u8>)>,
    tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
}

impl Default for GgufBuilder {
    /// Create an empty GGUF builder with no metadata or tensors
    fn default() -> Self {
        Self::new()
    }
}

/// Builder methods for constructing GGUF v3 files with metadata and tensors
impl GgufBuilder {
    /// Create a new GGUF builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
        }
    }

    /// Add a string metadata value
    #[must_use]
    pub fn add_string(mut self, key: &str, value: &str) -> Self {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
        self.metadata.push((key.to_string(), 8, bytes));
        self
    }

    /// Add a u32 metadata value
    #[must_use]
    pub fn add_u32(mut self, key: &str, value: u32) -> Self {
        self.metadata
            .push((key.to_string(), 4, value.to_le_bytes().to_vec()));
        self
    }

    /// Add a f32 metadata value
    #[must_use]
    pub fn add_f32(mut self, key: &str, value: f32) -> Self {
        self.metadata
            .push((key.to_string(), 6, value.to_le_bytes().to_vec()));
        self
    }

    /// Set architecture
    #[must_use]
    pub fn architecture(self, arch: &str) -> Self {
        self.add_string("general.architecture", arch)
    }

    /// Set hidden dimension
    #[must_use]
    pub fn hidden_dim(self, arch: &str, dim: u32) -> Self {
        self.add_u32(&format!("{arch}.embedding_length"), dim)
    }

    /// Set number of layers
    #[must_use]
    pub fn num_layers(self, arch: &str, count: u32) -> Self {
        self.add_u32(&format!("{arch}.block_count"), count)
    }

    /// Set number of attention heads
    #[must_use]
    pub fn num_heads(self, arch: &str, count: u32) -> Self {
        self.add_u32(&format!("{arch}.attention.head_count"), count)
    }

    /// Set number of KV heads
    #[must_use]
    pub fn num_kv_heads(self, arch: &str, count: u32) -> Self {
        self.add_u32(&format!("{arch}.attention.head_count_kv"), count)
    }

    /// Set context length
    #[must_use]
    pub fn context_length(self, arch: &str, len: u32) -> Self {
        self.add_u32(&format!("{arch}.context_length"), len)
    }

    /// Set RoPE frequency base
    #[must_use]
    pub fn rope_freq_base(self, arch: &str, base: f32) -> Self {
        self.add_f32(&format!("{arch}.rope.freq_base"), base)
    }

    /// Set RMS epsilon
    #[must_use]
    pub fn rms_epsilon(self, arch: &str, eps: f32) -> Self {
        self.add_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"), eps)
    }

    /// Set feed-forward hidden dimension
    #[must_use]
    pub fn ffn_hidden_dim(self, arch: &str, dim: u32) -> Self {
        self.add_u32(&format!("{arch}.feed_forward_length"), dim)
    }

    /// Add an F32 tensor
    #[must_use]
    pub fn add_f32_tensor(mut self, name: &str, dims: &[u64], data: &[f32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.tensors
            .push((name.to_string(), dims.to_vec(), GGUF_TYPE_F32, bytes));
        self
    }

    /// Add a Q4_0 tensor (18 bytes per 32 elements)
    #[must_use]
    pub fn add_q4_0_tensor(mut self, name: &str, dims: &[u64], data: &[u8]) -> Self {
        self.tensors.push((
            name.to_string(),
            dims.to_vec(),
            GGUF_TYPE_Q4_0,
            data.to_vec(),
        ));
        self
    }

    /// Build the GGUF file as a byte vector
    #[must_use]
    pub fn build(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        data.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        data.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());

        // Metadata
        for (key, value_type, value_bytes) in &self.metadata {
            data.extend_from_slice(&(key.len() as u64).to_le_bytes());
            data.extend_from_slice(key.as_bytes());
            data.extend_from_slice(&value_type.to_le_bytes());
            data.extend_from_slice(value_bytes);
        }

        // Tensor info
        let mut tensor_data_offset = 0u64;
        for (name, dims, qtype, tensor_bytes) in &self.tensors {
            data.extend_from_slice(&(name.len() as u64).to_le_bytes());
            data.extend_from_slice(name.as_bytes());
            data.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for dim in dims.iter().rev() {
                data.extend_from_slice(&dim.to_le_bytes());
            }
            data.extend_from_slice(&qtype.to_le_bytes());
            data.extend_from_slice(&tensor_data_offset.to_le_bytes());
            tensor_data_offset += tensor_bytes.len() as u64;
        }

        // Align to GGUF_ALIGNMENT (32 bytes)
        let current_len = data.len();
        let aligned = current_len.div_ceil(GGUF_ALIGNMENT) * GGUF_ALIGNMENT;
        data.resize(aligned, 0);

        // Tensor data
        for (_, _, _, tensor_bytes) in &self.tensors {
            data.extend_from_slice(tensor_bytes);
        }

        data
    }
}

/// Create valid Q4_0 data for a tensor
fn create_q4_0_data(num_elements: usize) -> Vec<u8> {
    let num_blocks = num_elements.div_ceil(32);
    let mut data = Vec::with_capacity(num_blocks * 18);
    for _ in 0..num_blocks {
        let scale = half::f16::from_f32(0.1);
        data.extend_from_slice(&scale.to_le_bytes());
        data.extend([0x88u8; 16]);
    }
    data
}

/// Build an executable "pygmy" GGUF model for testing
///
/// This creates a minimal valid GGUF model that can pass structure validation
/// but produces garbage output. Perfect for testing the QA pipeline without
/// needing real model files.
///
/// Dimensions chosen to be small but valid:
/// - vocab_size: 32
/// - hidden_dim: 32 (aligns with Q4_0 block size)
/// - intermediate_dim: 64
/// - num_heads: 4
/// - num_kv_heads: 4
/// - num_layers: 1
#[must_use]
pub fn build_pygmy_gguf() -> Vec<u8> {
    const VOCAB_SIZE: usize = 32;
    const HIDDEN_DIM: usize = 32;
    const INTERMEDIATE_DIM: usize = 64;
    const NUM_HEADS: usize = 4;
    const NUM_KV_HEADS: usize = 4;
    const CONTEXT_LENGTH: usize = 32;

    let kv_dim = NUM_KV_HEADS * (HIDDEN_DIM / NUM_HEADS);

    // F32 data
    let embed_data: Vec<f32> = (0..VOCAB_SIZE * HIDDEN_DIM)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect();
    let norm_data: Vec<f32> = vec![1.0; HIDDEN_DIM];

    // Q4_0 data
    let q_data = create_q4_0_data(HIDDEN_DIM * HIDDEN_DIM);
    let k_data = create_q4_0_data(HIDDEN_DIM * kv_dim);
    let v_data = create_q4_0_data(HIDDEN_DIM * kv_dim);
    let attn_out_data = create_q4_0_data(HIDDEN_DIM * HIDDEN_DIM);
    let ffn_gate_data = create_q4_0_data(HIDDEN_DIM * INTERMEDIATE_DIM);
    let ffn_up_data = create_q4_0_data(HIDDEN_DIM * INTERMEDIATE_DIM);
    let ffn_down_data = create_q4_0_data(INTERMEDIATE_DIM * HIDDEN_DIM);
    let lm_head_data = create_q4_0_data(HIDDEN_DIM * VOCAB_SIZE);

    GgufBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", HIDDEN_DIM as u32)
        .num_layers("llama", 1)
        .num_heads("llama", NUM_HEADS as u32)
        .num_kv_heads("llama", NUM_KV_HEADS as u32)
        .context_length("llama", CONTEXT_LENGTH as u32)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", INTERMEDIATE_DIM as u32)
        .add_f32_tensor(
            "token_embd.weight",
            &[VOCAB_SIZE as u64, HIDDEN_DIM as u64],
            &embed_data,
        )
        .add_f32_tensor("blk.0.attn_norm.weight", &[HIDDEN_DIM as u64], &norm_data)
        .add_q4_0_tensor(
            "blk.0.attn_q.weight",
            &[HIDDEN_DIM as u64, HIDDEN_DIM as u64],
            &q_data,
        )
        .add_q4_0_tensor(
            "blk.0.attn_k.weight",
            &[HIDDEN_DIM as u64, kv_dim as u64],
            &k_data,
        )
        .add_q4_0_tensor(
            "blk.0.attn_v.weight",
            &[HIDDEN_DIM as u64, kv_dim as u64],
            &v_data,
        )
        .add_q4_0_tensor(
            "blk.0.attn_output.weight",
            &[HIDDEN_DIM as u64, HIDDEN_DIM as u64],
            &attn_out_data,
        )
        .add_f32_tensor("blk.0.ffn_norm.weight", &[HIDDEN_DIM as u64], &norm_data)
        .add_q4_0_tensor(
            "blk.0.ffn_gate.weight",
            &[HIDDEN_DIM as u64, INTERMEDIATE_DIM as u64],
            &ffn_gate_data,
        )
        .add_q4_0_tensor(
            "blk.0.ffn_up.weight",
            &[HIDDEN_DIM as u64, INTERMEDIATE_DIM as u64],
            &ffn_up_data,
        )
        .add_q4_0_tensor(
            "blk.0.ffn_down.weight",
            &[INTERMEDIATE_DIM as u64, HIDDEN_DIM as u64],
            &ffn_down_data,
        )
        .add_f32_tensor("output_norm.weight", &[HIDDEN_DIM as u64], &norm_data)
        .add_q4_0_tensor(
            "output.weight",
            &[HIDDEN_DIM as u64, VOCAB_SIZE as u64],
            &lm_head_data,
        )
        .build()
}

/// A temporary model directory with pygmy model files for all formats
///
/// Creates:
/// - `<temp>/gguf/model.gguf` - Pygmy GGUF model
/// - `<temp>/apr/model.apr` - Empty placeholder (format testing only)
/// - `<temp>/safetensors/model.safetensors` - Empty placeholder
pub struct PygmyModelDir {
    /// Temporary directory handle (kept alive for cleanup on drop)
    _temp_dir: TempDir,
    /// Path to the root directory containing format subdirs
    pub root: PathBuf,
    /// Path to the GGUF model file
    pub gguf_path: PathBuf,
    /// Path to the APR model file (placeholder)
    pub apr_path: PathBuf,
    /// Path to the SafeTensors model file (placeholder)
    pub st_path: PathBuf,
}

/// Creation and path access for temporary pygmy model directories
impl PygmyModelDir {
    /// Create a new pygmy model directory with all format files
    ///
    /// # Errors
    ///
    /// Returns an error if the temporary directory cannot be created or files
    /// cannot be written.
    pub fn new() -> std::io::Result<Self> {
        let temp_dir = TempDir::new()?;
        let root = temp_dir.path().to_path_buf();

        // Create format subdirectories
        let gguf_dir = root.join("gguf");
        let apr_dir = root.join("apr");
        let st_dir = root.join("safetensors");

        std::fs::create_dir_all(&gguf_dir)?;
        std::fs::create_dir_all(&apr_dir)?;
        std::fs::create_dir_all(&st_dir)?;

        // Write GGUF model
        let gguf_path = gguf_dir.join("model.gguf");
        let gguf_data = build_pygmy_gguf();
        let mut file = std::fs::File::create(&gguf_path)?;
        file.write_all(&gguf_data)?;

        // Write APR placeholder (minimal valid header)
        let apr_path = apr_dir.join("model.apr");
        let mut file = std::fs::File::create(&apr_path)?;
        // APR magic + minimal header (will fail actual loading but tests path resolution)
        file.write_all(b"APR\x00")?;
        file.write_all(&[0u8; 60])?;

        // Write SafeTensors placeholder
        let st_path = st_dir.join("model.safetensors");
        let mut file = std::fs::File::create(&st_path)?;
        // Minimal SafeTensors header (will fail actual loading but tests path resolution)
        file.write_all(&8u64.to_le_bytes())?; // header size
        file.write_all(b"{}")?; // empty JSON
        file.write_all(&[0u8; 2])?; // padding

        Ok(Self {
            _temp_dir: temp_dir,
            root,
            gguf_path,
            apr_path,
            st_path,
        })
    }

    /// Get the root path as a string
    #[must_use]
    pub fn root_str(&self) -> &str {
        self.root.to_str().unwrap_or("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify pygmy GGUF has correct magic and version in header
    #[test]
    fn test_build_pygmy_gguf_valid_header() {
        let data = build_pygmy_gguf();

        // Check magic
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(magic, GGUF_MAGIC);

        // Check version
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(version, GGUF_VERSION_V3);
    }

    /// Verify pygmy GGUF size is between 1KB and 50KB
    #[test]
    fn test_build_pygmy_gguf_size() {
        let data = build_pygmy_gguf();

        // Should be small (< 50KB)
        assert!(
            data.len() < 50_000,
            "Pygmy should be < 50KB, got {}",
            data.len()
        );

        // But should be non-trivial
        assert!(
            data.len() > 1_000,
            "Pygmy should be > 1KB, got {}",
            data.len()
        );
    }

    /// Verify PygmyModelDir creates all format files on disk
    #[test]
    fn test_pygmy_model_dir_creates_files() {
        let dir = PygmyModelDir::new().expect("Should create temp dir");

        assert!(dir.gguf_path.exists(), "GGUF file should exist");
        assert!(dir.apr_path.exists(), "APR file should exist");
        assert!(dir.st_path.exists(), "SafeTensors file should exist");
    }

    /// Verify PygmyModelDir GGUF file has valid magic bytes
    #[test]
    fn test_pygmy_model_dir_gguf_valid() {
        let dir = PygmyModelDir::new().expect("Should create temp dir");

        let data = std::fs::read(&dir.gguf_path).expect("Should read GGUF");
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(magic, GGUF_MAGIC);
    }

    /// Verify empty GgufBuilder produces valid header
    #[test]
    fn test_gguf_builder_empty() {
        let data = GgufBuilder::new().build();

        // Should have valid header
        assert!(data.len() >= 24);
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(magic, GGUF_MAGIC);
    }

    /// Verify GgufBuilder encodes metadata count correctly in header
    #[test]
    fn test_gguf_builder_with_metadata() {
        let data = GgufBuilder::new()
            .architecture("llama")
            .add_u32("test.value", 42)
            .build();

        // Should have valid header with 2 metadata entries
        let n_metadata = u64::from_le_bytes([
            data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
        ]);
        assert_eq!(n_metadata, 2);
    }

    /// Verify GgufBuilder default trait produces valid header
    #[test]
    fn test_gguf_builder_default() {
        let builder = GgufBuilder::default();
        let data = builder.build();

        // Should have valid header
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(magic, GGUF_MAGIC);
    }

    /// Verify PygmyModelDir root_str returns non-empty path
    #[test]
    fn test_pygmy_model_dir_root_str() {
        let dir = PygmyModelDir::new().expect("Should create temp dir");
        let root_str = dir.root_str();

        // Should return a non-empty path string
        assert!(!root_str.is_empty());
    }
}
