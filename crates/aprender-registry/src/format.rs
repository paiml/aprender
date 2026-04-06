//! Model Format Detection and Metadata
//!
//! Detects model file formats and extracts metadata from model files.
//!
//! ## Supported Formats
//!
//! - **GGUF**: GGML Universal Format (llama.cpp, ollama)
//! - **SafeTensors**: HuggingFace safe tensor format
//! - **APR**: Aprender native format
//! - **ONNX**: Open Neural Network Exchange
//! - **PyTorch**: `.pt`/`.pth` files (detection only)
//!
//! ## Example
//!
//! ```rust,ignore
//! use pacha::format::{detect_format, ModelFormat};
//!
//! let format = detect_format(&data)?;
//! match format {
//!     ModelFormat::Gguf(info) => println!("GGUF: {} params", info.parameters),
//!     ModelFormat::SafeTensors(info) => println!("SafeTensors: {} tensors", info.tensor_count),
//!     _ => println!("Other format"),
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// FMT-001: Model Format Enum
// ============================================================================

/// Detected model format
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelFormat {
    /// GGUF format (llama.cpp)
    Gguf(GgufInfo),
    /// SafeTensors format (HuggingFace)
    SafeTensors(SafeTensorsInfo),
    /// Aprender native format
    Apr(AprInfo),
    /// ONNX format
    Onnx(OnnxInfo),
    /// PyTorch format (limited detection)
    PyTorch,
    /// Unknown format
    Unknown,
}

impl ModelFormat {
    /// Get format name
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gguf(_) => "GGUF",
            Self::SafeTensors(_) => "SafeTensors",
            Self::Apr(_) => "APR",
            Self::Onnx(_) => "ONNX",
            Self::PyTorch => "PyTorch",
            Self::Unknown => "Unknown",
        }
    }

    /// Get file extension
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Gguf(_) => ".gguf",
            Self::SafeTensors(_) => ".safetensors",
            Self::Apr(_) => ".apr",
            Self::Onnx(_) => ".onnx",
            Self::PyTorch => ".pt",
            Self::Unknown => "",
        }
    }

    /// Check if format is quantized
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        match self {
            Self::Gguf(info) => info.quantization.is_some(),
            Self::Apr(info) => info.quantization.is_some(),
            _ => false,
        }
    }

    /// Get quantization type if available
    #[must_use]
    pub fn quantization(&self) -> Option<&str> {
        match self {
            Self::Gguf(info) => info.quantization.as_deref(),
            Self::Apr(info) => info.quantization.as_deref(),
            _ => None,
        }
    }

    /// Get parameter count if available
    #[must_use]
    pub fn parameters(&self) -> Option<u64> {
        match self {
            Self::Gguf(info) => info.parameters,
            Self::SafeTensors(info) => info.parameters,
            Self::Apr(info) => info.parameters,
            Self::Onnx(info) => info.parameters,
            Self::PyTorch | Self::Unknown => None,
        }
    }
}

// ============================================================================
// FMT-002: Format-Specific Info
// ============================================================================

/// GGUF file information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GgufInfo {
    /// GGUF version
    pub version: u32,
    /// Number of tensors
    pub tensor_count: u64,
    /// Number of metadata key-value pairs
    pub metadata_count: u64,
    /// Model architecture (e.g., "llama", "mistral")
    pub architecture: Option<String>,
    /// Quantization type (e.g., "Q4_K_M", "Q8_0")
    pub quantization: Option<String>,
    /// Context length
    pub context_length: Option<u32>,
    /// Embedding dimension
    pub embedding_dim: Option<u32>,
    /// Number of layers
    pub num_layers: Option<u32>,
    /// Number of attention heads
    pub num_heads: Option<u32>,
    /// Vocabulary size
    pub vocab_size: Option<u32>,
    /// Estimated parameter count
    pub parameters: Option<u64>,
    /// Model name from metadata
    pub name: Option<String>,
    /// Author from metadata
    pub author: Option<String>,
    /// License from metadata
    pub license: Option<String>,
}

impl Default for GgufInfo {
    fn default() -> Self {
        Self {
            version: 0,
            tensor_count: 0,
            metadata_count: 0,
            architecture: None,
            quantization: None,
            context_length: None,
            embedding_dim: None,
            num_layers: None,
            num_heads: None,
            vocab_size: None,
            parameters: None,
            name: None,
            author: None,
            license: None,
        }
    }
}

/// SafeTensors file information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafeTensorsInfo {
    /// Number of tensors
    pub tensor_count: usize,
    /// Tensor names and shapes
    pub tensors: HashMap<String, TensorInfo>,
    /// Metadata from header
    pub metadata: HashMap<String, String>,
    /// Estimated parameter count
    pub parameters: Option<u64>,
    /// Data type
    pub dtype: Option<String>,
}

impl Default for SafeTensorsInfo {
    fn default() -> Self {
        Self {
            tensor_count: 0,
            tensors: HashMap::new(),
            metadata: HashMap::new(),
            parameters: None,
            dtype: None,
        }
    }
}

/// Tensor information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: String,
    /// Offset in file
    pub offset: usize,
}

/// APR (Aprender) file information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AprInfo {
    /// APR version
    pub version: u32,
    /// Model type (e.g., "LogisticRegression")
    pub model_type: String,
    /// Quantization type
    pub quantization: Option<String>,
    /// Compressed
    pub compressed: bool,
    /// Encrypted
    pub encrypted: bool,
    /// Signed
    pub signed: bool,
    /// Parameter count
    pub parameters: Option<u64>,
    /// CRC32 checksum
    pub checksum: Option<u32>,
}

impl Default for AprInfo {
    fn default() -> Self {
        Self {
            version: 0,
            model_type: String::new(),
            quantization: None,
            compressed: false,
            encrypted: false,
            signed: false,
            parameters: None,
            checksum: None,
        }
    }
}

/// ONNX file information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnnxInfo {
    /// ONNX IR version
    pub ir_version: u64,
    /// Producer name
    pub producer_name: Option<String>,
    /// Producer version
    pub producer_version: Option<String>,
    /// Model description
    pub description: Option<String>,
    /// Number of nodes
    pub node_count: usize,
    /// Estimated parameters
    pub parameters: Option<u64>,
}

impl Default for OnnxInfo {
    fn default() -> Self {
        Self {
            ir_version: 0,
            producer_name: None,
            producer_version: None,
            description: None,
            node_count: 0,
            parameters: None,
        }
    }
}

// ============================================================================
// FMT-003: Format Detection
// ============================================================================

/// Magic bytes for format detection
mod magic {
    /// GGUF magic bytes ("GGUF")
    pub(super) const GGUF: [u8; 4] = [0x47, 0x47, 0x55, 0x46];
    /// SafeTensors starts with JSON header size (little-endian u64)
    pub(super) const _SAFETENSORS_MIN_HEADER: u64 = 8;
    /// APR magic bytes ("APR\0")
    pub(super) const APR: [u8; 4] = [0x41, 0x50, 0x52, 0x00];
    /// ONNX (protobuf) magic
    pub(super) const ONNX: [u8; 2] = [0x08, 0x00]; // Protobuf field 1, varint
    /// PyTorch magic (PK zip for newer, 0x80 for older pickle)
    pub(super) const PYTORCH_ZIP: [u8; 2] = [0x50, 0x4B];
    pub(super) const PYTORCH_PICKLE: u8 = 0x80;
}

/// Detect model format from bytes
///
/// # Arguments
///
/// * `data` - At least first 1KB of the file
///
/// # Returns
///
/// Detected `ModelFormat` with extracted metadata
#[must_use]
pub fn detect_format(data: &[u8]) -> ModelFormat {
    if data.len() < 8 {
        return ModelFormat::Unknown;
    }

    // Check GGUF magic
    if data[..4] == magic::GGUF {
        return parse_gguf_header(data);
    }

    // Check APR magic
    if data[..4] == magic::APR {
        return parse_apr_header(data);
    }

    // Try SafeTensors BEFORE PyTorch: SafeTensors has a more specific signature
    // (u64 header size + valid JSON), while PyTorch pickle is just data[0]==0x80.
    // SafeTensors files whose header_size has low byte 0x80 would otherwise be
    // misidentified as PyTorch pickle. (Fixes pacha#4)
    if let Some(info) = try_parse_safetensors(data) {
        return ModelFormat::SafeTensors(info);
    }

    // Check PyTorch (zip or pickle) — AFTER SafeTensors to avoid false positives
    if data[..2] == magic::PYTORCH_ZIP || data[0] == magic::PYTORCH_PICKLE {
        return ModelFormat::PyTorch;
    }

    // Try ONNX (protobuf)
    if data[0] == magic::ONNX[0] {
        if let Some(info) = try_parse_onnx(data) {
            return ModelFormat::Onnx(info);
        }
    }

    ModelFormat::Unknown
}

/// Known format extensions and their names
const FORMAT_EXTENSIONS: &[(&str, &str)] = &[
    (".gguf", "GGUF"),
    (".safetensors", "SafeTensors"),
    (".apr", "APR"),
    (".onnx", "ONNX"),
    (".pt", "PyTorch"),
    (".pth", "PyTorch"),
    (".bin", "Binary"),
];

/// Detect format from file path extension
#[must_use]
pub fn detect_format_from_path(path: &str) -> Option<&'static str> {
    let path_lower = path.to_lowercase();
    FORMAT_EXTENSIONS.iter().find(|(ext, _)| path_lower.ends_with(ext)).map(|(_, name)| *name)
}

/// Parse GGUF header
fn parse_gguf_header(data: &[u8]) -> ModelFormat {
    if data.len() < 24 {
        return ModelFormat::Gguf(GgufInfo::default());
    }

    // GGUF header format:
    // 0-3: magic "GGUF"
    // 4-7: version (u32 LE)
    // 8-15: tensor_count (u64 LE)
    // 16-23: metadata_kv_count (u64 LE)

    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let tensor_count = u64::from_le_bytes([
        data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
    ]);
    let metadata_count = u64::from_le_bytes([
        data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
    ]);

    // For full metadata parsing, we'd need to parse the key-value pairs
    // This is a simplified version that just extracts the header info

    ModelFormat::Gguf(GgufInfo { version, tensor_count, metadata_count, ..Default::default() })
}

/// Parse APR header
fn parse_apr_header(data: &[u8]) -> ModelFormat {
    if data.len() < 16 {
        return ModelFormat::Apr(AprInfo::default());
    }

    // APR header format:
    // 0-3: magic "APR\0"
    // 4-7: version (u32 LE)
    // 8-11: flags (u32 LE)
    // 12-15: model_type_len (u32 LE)

    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let flags = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);

    let compressed = (flags & 0x01) != 0;
    let encrypted = (flags & 0x02) != 0;
    let signed = (flags & 0x04) != 0;

    // Extract model type if we have enough data
    let model_type_len = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let model_type = if data.len() >= 16 + model_type_len {
        String::from_utf8_lossy(&data[16..16 + model_type_len]).to_string()
    } else {
        String::new()
    };

    ModelFormat::Apr(AprInfo {
        version,
        model_type,
        compressed,
        encrypted,
        signed,
        ..Default::default()
    })
}

/// Read SafeTensors header JSON from raw bytes
fn read_safetensors_header(data: &[u8]) -> Option<&[u8]> {
    if data.len() < 8 {
        return None;
    }

    let header_size = u64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]) as usize;

    if header_size == 0 || header_size > 100_000_000 {
        return None;
    }

    if data.len() < 8 + header_size {
        return None;
    }

    let header_json = &data[8..8 + header_size];
    if header_json.first() != Some(&b'{') {
        return None;
    }
    Some(header_json)
}

/// Extract __metadata__ string values from a parsed SafeTensors header
fn extract_metadata(header: &HashMap<String, serde_json::Value>, info: &mut SafeTensorsInfo) {
    let Some(meta) = header.get("__metadata__") else { return };
    let Some(obj) = meta.as_object() else { return };
    for (k, v) in obj {
        if let Some(s) = v.as_str() {
            info.metadata.insert(k.clone(), s.to_string());
        }
    }
}

/// Try to parse SafeTensors header
fn try_parse_safetensors(data: &[u8]) -> Option<SafeTensorsInfo> {
    // Check for partial header (enough to identify format, not enough to parse)
    if data.len() >= 8 {
        let header_size = u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]) as usize;
        if header_size > 0
            && header_size <= 100_000_000
            && data.len() < 8 + header_size
            && data.get(8) == Some(&b'{')
        {
            return Some(SafeTensorsInfo { tensor_count: 0, ..Default::default() });
        }
    }

    let header_json = read_safetensors_header(data)?;
    let header: HashMap<String, serde_json::Value> = serde_json::from_slice(header_json).ok()?;

    let mut info = SafeTensorsInfo::default();
    info.tensor_count = header.keys().filter(|k| *k != "__metadata__").count();
    extract_metadata(&header, &mut info);
    info.parameters = Some(extract_tensor_info(&header, &mut info));
    Some(info)
}

/// Extract tensor info from a SafeTensors JSON header
fn extract_tensor_info(
    header: &HashMap<String, serde_json::Value>,
    info: &mut SafeTensorsInfo,
) -> u64 {
    let mut total_params: u64 = 0;
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let Some(obj) = value.as_object() else {
            continue;
        };
        let (Some(dtype), Some(shape)) = (obj.get("dtype"), obj.get("shape")) else {
            continue;
        };
        let dtype_str = dtype.as_str().unwrap_or("F32").to_string();
        let shape_vec: Vec<usize> = shape
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
            .unwrap_or_default();

        let elements: u64 = shape_vec.iter().map(|&s| s as u64).product();
        total_params += elements;

        info.tensors.insert(
            name.clone(),
            TensorInfo { shape: shape_vec, dtype: dtype_str.clone(), offset: 0 },
        );

        if info.dtype.is_none() {
            info.dtype = Some(dtype_str);
        }
    }
    total_params
}

/// Try to parse ONNX header (simplified)
fn try_parse_onnx(data: &[u8]) -> Option<OnnxInfo> {
    // ONNX uses protobuf format
    // This is a simplified detection that just checks for valid protobuf structure

    if data.len() < 16 {
        return None;
    }

    // Very basic protobuf field detection
    // Field 1 (ir_version) should be present at the start
    if data[0] != 0x08 {
        return None;
    }

    // Read varint for ir_version
    let (ir_version, _) = read_varint(&data[1..])?;

    Some(OnnxInfo { ir_version, ..Default::default() })
}

/// Read a protobuf varint
fn read_varint(data: &[u8]) -> Option<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift = 0;

    for (i, &byte) in data.iter().enumerate() {
        if i >= 10 {
            return None; // Varint too long
        }

        result |= ((byte & 0x7F) as u64) << shift;
        shift += 7;

        if byte & 0x80 == 0 {
            return Some((result, i + 1));
        }
    }

    None
}

// ============================================================================
// FMT-004: Quantization Types
// ============================================================================

/// Common quantization types (GGUF spec naming convention)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum QuantType {
    /// Full precision (FP32)
    F32,
    /// Half precision (FP16)
    F16,
    /// Brain floating point (BF16)
    BF16,
    /// 8-bit integer
    Q8_0,
    /// 8-bit with K-quants
    Q8_K,
    /// 6-bit K-quants
    Q6_K,
    /// 5-bit K-quants (small)
    Q5_K_S,
    /// 5-bit K-quants (medium)
    Q5_K_M,
    /// 5-bit (legacy)
    Q5_0,
    /// 5-bit with 1 (legacy)
    Q5_1,
    /// 4-bit K-quants (small)
    Q4_K_S,
    /// 4-bit K-quants (medium)
    Q4_K_M,
    /// 4-bit (legacy)
    Q4_0,
    /// 4-bit with 1 (legacy)
    Q4_1,
    /// 3-bit K-quants (small)
    Q3_K_S,
    /// 3-bit K-quants (medium)
    Q3_K_M,
    /// 3-bit K-quants (large)
    Q3_K_L,
    /// 2-bit K-quants (small)
    Q2_K_S,
    /// 2-bit K-quants
    Q2_K,
    /// Importance-weighted 4-bit (non-linear)
    IQ4_NL,
    /// Importance-weighted 4-bit (extra small)
    IQ4_XS,
    /// Importance-weighted 3-bit (small)
    IQ3_S,
    /// Importance-weighted 3-bit (medium)
    IQ3_M,
    /// Importance-weighted 3-bit (extra small)
    IQ3_XS,
    /// Importance-weighted 3-bit (extra extra small)
    IQ3_XXS,
    /// Importance-weighted 2-bit (small)
    IQ2_S,
    /// Importance-weighted 2-bit (extra small)
    IQ2_XS,
    /// Importance-weighted 2-bit (extra extra small)
    IQ2_XXS,
    /// Importance-weighted 1-bit (small)
    IQ1_S,
    /// Importance-weighted 1-bit (medium)
    IQ1_M,
}

impl QuantType {
    /// Parse quantization type from string
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        let s = s.to_uppercase();
        match s.as_str() {
            "F32" | "FP32" => Some(Self::F32),
            "F16" | "FP16" => Some(Self::F16),
            "BF16" => Some(Self::BF16),
            "Q8_0" => Some(Self::Q8_0),
            "Q8_K" => Some(Self::Q8_K),
            "Q6_K" => Some(Self::Q6_K),
            "Q5_K_S" => Some(Self::Q5_K_S),
            "Q5_K_M" => Some(Self::Q5_K_M),
            "Q5_0" => Some(Self::Q5_0),
            "Q5_1" => Some(Self::Q5_1),
            "Q4_K_S" => Some(Self::Q4_K_S),
            "Q4_K_M" => Some(Self::Q4_K_M),
            "Q4_0" => Some(Self::Q4_0),
            "Q4_1" => Some(Self::Q4_1),
            "Q3_K_S" => Some(Self::Q3_K_S),
            "Q3_K_M" => Some(Self::Q3_K_M),
            "Q3_K_L" => Some(Self::Q3_K_L),
            "Q2_K_S" => Some(Self::Q2_K_S),
            "Q2_K" => Some(Self::Q2_K),
            "IQ4_NL" => Some(Self::IQ4_NL),
            "IQ4_XS" => Some(Self::IQ4_XS),
            "IQ3_S" => Some(Self::IQ3_S),
            "IQ3_M" => Some(Self::IQ3_M),
            "IQ3_XS" => Some(Self::IQ3_XS),
            "IQ3_XXS" => Some(Self::IQ3_XXS),
            "IQ2_S" => Some(Self::IQ2_S),
            "IQ2_XS" => Some(Self::IQ2_XS),
            "IQ2_XXS" => Some(Self::IQ2_XXS),
            "IQ1_S" => Some(Self::IQ1_S),
            "IQ1_M" => Some(Self::IQ1_M),
            _ => None,
        }
    }

    /// Get bits per weight
    #[must_use]
    pub const fn bits_per_weight(&self) -> f32 {
        match self {
            Self::F32 => 32.0,
            Self::F16 | Self::BF16 => 16.0,
            Self::Q8_0 | Self::Q8_K => 8.0,
            Self::Q6_K => 6.5,
            Self::Q5_K_S | Self::Q5_K_M | Self::Q5_0 | Self::Q5_1 => 5.5,
            Self::Q4_K_S | Self::Q4_K_M | Self::Q4_0 | Self::Q4_1 => 4.5,
            Self::Q3_K_S | Self::Q3_K_M | Self::Q3_K_L => 3.5,
            Self::Q2_K_S | Self::Q2_K => 2.5,
            Self::IQ4_NL | Self::IQ4_XS => 4.25,
            Self::IQ3_S | Self::IQ3_M | Self::IQ3_XS | Self::IQ3_XXS => 3.0,
            Self::IQ2_S | Self::IQ2_XS | Self::IQ2_XXS => 2.0,
            Self::IQ1_S | Self::IQ1_M => 1.5,
        }
    }

    /// Estimate file size for given parameter count
    #[must_use]
    pub fn estimate_size(&self, parameters: u64) -> u64 {
        let bits = self.bits_per_weight() as f64;
        let bytes = (parameters as f64 * bits) / 8.0;
        // Add ~10% overhead for metadata
        (bytes * 1.1) as u64
    }

    /// Get quality tier (1-5, higher is better quality)
    #[must_use]
    pub const fn quality_tier(&self) -> u8 {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 5,
            Self::Q8_0 | Self::Q8_K => 5,
            Self::Q6_K => 4,
            Self::Q5_K_S | Self::Q5_K_M | Self::Q5_0 | Self::Q5_1 => 4,
            Self::Q4_K_S | Self::Q4_K_M | Self::Q4_0 | Self::Q4_1 => 3,
            Self::IQ4_NL | Self::IQ4_XS => 3,
            Self::Q3_K_S | Self::Q3_K_M | Self::Q3_K_L => 2,
            Self::IQ3_S | Self::IQ3_M | Self::IQ3_XS | Self::IQ3_XXS => 2,
            Self::Q2_K_S | Self::Q2_K => 1,
            Self::IQ2_S | Self::IQ2_XS | Self::IQ2_XXS => 1,
            Self::IQ1_S | Self::IQ1_M => 1,
        }
    }

    /// Get recommended VRAM in GB for given parameter count
    #[must_use]
    pub fn vram_requirement(&self, parameters: u64) -> f64 {
        // Base model size
        let model_size = self.estimate_size(parameters) as f64;
        // Context cache (rough estimate: 2GB per 4K context)
        let context_overhead = 2.0 * 1024.0 * 1024.0 * 1024.0;
        // Total in GB
        (model_size + context_overhead) / (1024.0 * 1024.0 * 1024.0)
    }
}

impl std::fmt::Display for QuantType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::Q8_0 => "Q8_0",
            Self::Q8_K => "Q8_K",
            Self::Q6_K => "Q6_K",
            Self::Q5_K_S => "Q5_K_S",
            Self::Q5_K_M => "Q5_K_M",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q4_K_S => "Q4_K_S",
            Self::Q4_K_M => "Q4_K_M",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q3_K_S => "Q3_K_S",
            Self::Q3_K_M => "Q3_K_M",
            Self::Q3_K_L => "Q3_K_L",
            Self::Q2_K_S => "Q2_K_S",
            Self::Q2_K => "Q2_K",
            Self::IQ4_NL => "IQ4_NL",
            Self::IQ4_XS => "IQ4_XS",
            Self::IQ3_S => "IQ3_S",
            Self::IQ3_M => "IQ3_M",
            Self::IQ3_XS => "IQ3_XS",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ2_S => "IQ2_S",
            Self::IQ2_XS => "IQ2_XS",
            Self::IQ2_XXS => "IQ2_XXS",
            Self::IQ1_S => "IQ1_S",
            Self::IQ1_M => "IQ1_M",
        };
        write!(f, "{s}")
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // FMT-001: Format Detection Tests
    // ========================================================================

    #[test]
    fn test_detect_gguf_format() {
        // GGUF magic + version 3 + 100 tensors + 50 metadata
        let mut data = vec![0u8; 100];
        data[0..4].copy_from_slice(&magic::GGUF);
        data[4..8].copy_from_slice(&3u32.to_le_bytes());
        data[8..16].copy_from_slice(&100u64.to_le_bytes());
        data[16..24].copy_from_slice(&50u64.to_le_bytes());

        let format = detect_format(&data);
        assert!(matches!(format, ModelFormat::Gguf(_)));

        if let ModelFormat::Gguf(info) = format {
            assert_eq!(info.version, 3);
            assert_eq!(info.tensor_count, 100);
            assert_eq!(info.metadata_count, 50);
        }
    }

    #[test]
    fn test_detect_apr_format() {
        // APR magic + version 1 + flags (compressed + signed)
        let mut data = vec![0u8; 100];
        data[0..4].copy_from_slice(&magic::APR);
        data[4..8].copy_from_slice(&1u32.to_le_bytes());
        data[8..12].copy_from_slice(&0x05u32.to_le_bytes()); // compressed + signed
        data[12..16].copy_from_slice(&4u32.to_le_bytes()); // model type len
        data[16..20].copy_from_slice(b"Test");

        let format = detect_format(&data);
        assert!(matches!(format, ModelFormat::Apr(_)));

        if let ModelFormat::Apr(info) = format {
            assert_eq!(info.version, 1);
            assert!(info.compressed);
            assert!(!info.encrypted);
            assert!(info.signed);
            assert_eq!(info.model_type, "Test");
        }
    }

    #[test]
    fn test_detect_pytorch_zip_format() {
        let mut data = vec![0u8; 100];
        data[0..2].copy_from_slice(&magic::PYTORCH_ZIP);

        let format = detect_format(&data);
        assert!(matches!(format, ModelFormat::PyTorch));
    }

    #[test]
    fn test_detect_pytorch_pickle_format() {
        let mut data = vec![0u8; 100];
        data[0] = magic::PYTORCH_PICKLE;

        let format = detect_format(&data);
        assert!(matches!(format, ModelFormat::PyTorch));
    }

    #[test]
    fn test_detect_safetensors_format() {
        // SafeTensors: header_size (u64 LE) + JSON header
        let header = r#"{"tensor1":{"dtype":"F32","shape":[100,100]}}"#;
        let header_bytes = header.as_bytes();
        let header_size = header_bytes.len() as u64;

        let mut data = Vec::new();
        data.extend_from_slice(&header_size.to_le_bytes());
        data.extend_from_slice(header_bytes);

        let format = detect_format(&data);
        assert!(matches!(format, ModelFormat::SafeTensors(_)));

        if let ModelFormat::SafeTensors(info) = format {
            assert_eq!(info.tensor_count, 1);
            assert!(info.tensors.contains_key("tensor1"));
            assert_eq!(info.parameters, Some(10000)); // 100 * 100
        }
    }

    #[test]
    fn test_detect_unknown_format() {
        let data = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
        let format = detect_format(&data);
        assert!(matches!(format, ModelFormat::Unknown));
    }

    #[test]
    fn test_detect_empty_data() {
        let data = vec![];
        let format = detect_format(&data);
        assert!(matches!(format, ModelFormat::Unknown));
    }

    #[test]
    fn test_detect_short_data() {
        let data = vec![0x47, 0x47, 0x55]; // Incomplete GGUF magic
        let format = detect_format(&data);
        assert!(matches!(format, ModelFormat::Unknown));
    }

    // ========================================================================
    // FMT-002: Format Info Tests
    // ========================================================================

    #[test]
    fn test_model_format_name() {
        assert_eq!(ModelFormat::Gguf(GgufInfo::default()).name(), "GGUF");
        assert_eq!(ModelFormat::SafeTensors(SafeTensorsInfo::default()).name(), "SafeTensors");
        assert_eq!(ModelFormat::Apr(AprInfo::default()).name(), "APR");
        assert_eq!(ModelFormat::Onnx(OnnxInfo::default()).name(), "ONNX");
        assert_eq!(ModelFormat::PyTorch.name(), "PyTorch");
        assert_eq!(ModelFormat::Unknown.name(), "Unknown");
    }

    #[test]
    fn test_model_format_extension() {
        assert_eq!(ModelFormat::Gguf(GgufInfo::default()).extension(), ".gguf");
        assert_eq!(
            ModelFormat::SafeTensors(SafeTensorsInfo::default()).extension(),
            ".safetensors"
        );
        assert_eq!(ModelFormat::Apr(AprInfo::default()).extension(), ".apr");
    }

    #[test]
    fn test_model_format_is_quantized() {
        let gguf_quant = ModelFormat::Gguf(GgufInfo {
            quantization: Some("Q4_K_M".to_string()),
            ..Default::default()
        });
        assert!(gguf_quant.is_quantized());

        let gguf_no_quant = ModelFormat::Gguf(GgufInfo::default());
        assert!(!gguf_no_quant.is_quantized());
    }

    #[test]
    fn test_model_format_quantization() {
        let format = ModelFormat::Gguf(GgufInfo {
            quantization: Some("Q8_0".to_string()),
            ..Default::default()
        });
        assert_eq!(format.quantization(), Some("Q8_0"));
    }

    #[test]
    fn test_model_format_parameters() {
        let format =
            ModelFormat::Gguf(GgufInfo { parameters: Some(7_000_000_000), ..Default::default() });
        assert_eq!(format.parameters(), Some(7_000_000_000));

        assert_eq!(ModelFormat::PyTorch.parameters(), None);
    }

    // ========================================================================
    // FMT-003: Path Detection Tests
    // ========================================================================

    #[test]
    fn test_detect_format_from_path() {
        assert_eq!(detect_format_from_path("model.gguf"), Some("GGUF"));
        assert_eq!(detect_format_from_path("model.GGUF"), Some("GGUF"));
        assert_eq!(detect_format_from_path("model.safetensors"), Some("SafeTensors"));
        assert_eq!(detect_format_from_path("model.apr"), Some("APR"));
        assert_eq!(detect_format_from_path("model.onnx"), Some("ONNX"));
        assert_eq!(detect_format_from_path("model.pt"), Some("PyTorch"));
        assert_eq!(detect_format_from_path("model.pth"), Some("PyTorch"));
        assert_eq!(detect_format_from_path("model.bin"), Some("Binary"));
        assert_eq!(detect_format_from_path("model.txt"), None);
    }

    // ========================================================================
    // FMT-004: Quantization Tests
    // ========================================================================

    #[test]
    fn test_quant_type_from_str() {
        assert_eq!(QuantType::from_str("Q4_K_M"), Some(QuantType::Q4_K_M));
        assert_eq!(QuantType::from_str("q4_k_m"), Some(QuantType::Q4_K_M));
        assert_eq!(QuantType::from_str("F16"), Some(QuantType::F16));
        assert_eq!(QuantType::from_str("fp16"), Some(QuantType::F16));
        assert_eq!(QuantType::from_str("invalid"), None);
    }

    #[test]
    fn test_quant_type_bits_per_weight() {
        assert!((QuantType::F32.bits_per_weight() - 32.0).abs() < f32::EPSILON);
        assert!((QuantType::F16.bits_per_weight() - 16.0).abs() < f32::EPSILON);
        assert!((QuantType::Q8_0.bits_per_weight() - 8.0).abs() < f32::EPSILON);
        assert!((QuantType::Q4_K_M.bits_per_weight() - 4.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quant_type_estimate_size() {
        let params = 7_000_000_000u64; // 7B parameters

        // F32: 7B * 32 bits / 8 * 1.1 = ~30.8 GB
        let f32_size = QuantType::F32.estimate_size(params);
        assert!(f32_size > 28_000_000_000 && f32_size < 32_000_000_000);

        // Q4_K_M: 7B * 4.5 bits / 8 * 1.1 = ~4.3 GB
        let q4_size = QuantType::Q4_K_M.estimate_size(params);
        assert!(q4_size > 4_000_000_000 && q4_size < 5_000_000_000);
    }

    #[test]
    fn test_quant_type_quality_tier() {
        assert_eq!(QuantType::F32.quality_tier(), 5);
        assert_eq!(QuantType::Q8_0.quality_tier(), 5);
        assert_eq!(QuantType::Q4_K_M.quality_tier(), 3);
        assert_eq!(QuantType::Q2_K.quality_tier(), 1);
    }

    #[test]
    fn test_quant_type_vram_requirement() {
        let params = 7_000_000_000u64;
        let vram_f32 = QuantType::F32.vram_requirement(params);
        let vram_q4 = QuantType::Q4_K_M.vram_requirement(params);

        // F32 should require more VRAM than Q4
        assert!(vram_f32 > vram_q4);
        // Both should be positive
        assert!(vram_f32 > 0.0);
        assert!(vram_q4 > 0.0);
    }

    #[test]
    fn test_quant_type_display() {
        assert_eq!(format!("{}", QuantType::Q4_K_M), "Q4_K_M");
        assert_eq!(format!("{}", QuantType::F16), "F16");
        assert_eq!(format!("{}", QuantType::IQ3_XXS), "IQ3_XXS");
    }

    // ========================================================================
    // Serialization Tests
    // ========================================================================

    #[test]
    fn test_gguf_info_serialization() {
        let info = GgufInfo {
            version: 3,
            tensor_count: 100,
            metadata_count: 50,
            architecture: Some("llama".to_string()),
            quantization: Some("Q4_K_M".to_string()),
            ..Default::default()
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("llama"));
        assert!(json.contains("Q4_K_M"));

        let parsed: GgufInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.architecture, Some("llama".to_string()));
    }

    #[test]
    fn test_safetensors_info_serialization() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            TensorInfo { shape: vec![100, 100], dtype: "F32".to_string(), offset: 0 },
        );

        let info = SafeTensorsInfo {
            tensor_count: 1,
            tensors,
            parameters: Some(10000),
            ..Default::default()
        };

        let json = serde_json::to_string(&info).unwrap();
        let parsed: SafeTensorsInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.tensor_count, 1);
        assert_eq!(parsed.parameters, Some(10000));
    }

    #[test]
    fn test_model_format_serialization() {
        let format = ModelFormat::Gguf(GgufInfo { version: 3, ..Default::default() });

        let json = serde_json::to_string(&format).unwrap();
        let parsed: ModelFormat = serde_json::from_str(&json).unwrap();

        assert!(matches!(parsed, ModelFormat::Gguf(_)));
    }

    #[test]
    fn test_quant_type_serialization() {
        let qt = QuantType::Q4_K_M;
        let json = serde_json::to_string(&qt).unwrap();
        let parsed: QuantType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, QuantType::Q4_K_M);
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    #[test]
    fn test_safetensors_with_metadata() {
        let header = r#"{"__metadata__":{"format":"pt"},"tensor1":{"dtype":"F16","shape":[512]}}"#;
        let header_bytes = header.as_bytes();
        let header_size = header_bytes.len() as u64;

        let mut data = Vec::new();
        data.extend_from_slice(&header_size.to_le_bytes());
        data.extend_from_slice(header_bytes);

        let format = detect_format(&data);
        if let ModelFormat::SafeTensors(info) = format {
            assert_eq!(info.tensor_count, 1); // Excludes __metadata__
            assert_eq!(info.metadata.get("format"), Some(&"pt".to_string()));
            assert_eq!(info.dtype, Some("F16".to_string()));
        } else {
            panic!("Expected SafeTensors format");
        }
    }

    #[test]
    fn test_pacha4_safetensors_header_size_0x80_not_pytorch() {
        // Regression test for pacha#4: SafeTensors file whose header_size
        // has low byte 0x80 was misidentified as PyTorch pickle because
        // detect_format checked data[0]==0x80 before trying SafeTensors.
        //
        // Real-world case: Qwen2.5-Coder-1.5B-Instruct model.safetensors
        // has header_size=38528 → first byte is 0x80.
        let header = r#"{"__metadata__":{"format":"pt"},"tensor1":{"dtype":"F32","shape":[32],"data_offsets":[0,128]}}"#;
        let header_bytes = header.as_bytes();
        // Force header_size to have low byte 0x80 (= 128)
        // We need header_size = header_bytes.len(), and pad to make it end in 0x80
        let target_size = 128usize; // 0x80
        assert!(header_bytes.len() <= target_size, "header too large for test");
        let padding = target_size - header_bytes.len();

        let mut data = Vec::new();
        data.extend_from_slice(&(target_size as u64).to_le_bytes());
        data.extend_from_slice(header_bytes);
        // Pad JSON with spaces before closing brace
        // Actually, we need to pad the header itself. Let's build a padded header.
        let padded_header = format!(
            r#"{{"__metadata__":{{"format":"pt"}},"tensor1":{{"dtype":"F32","shape":[32],"data_offsets":[0,128]}}{}}}"#,
            " ".repeat(padding)
        );
        let padded_bytes = padded_header.as_bytes();

        let mut data2 = Vec::new();
        data2.extend_from_slice(&(padded_bytes.len() as u64).to_le_bytes());
        data2.extend_from_slice(padded_bytes);
        // Add some fake tensor data
        data2.extend_from_slice(&[0u8; 128]);

        // First byte should be 0x80 (the pickle magic)
        assert_eq!(data2[0], 0x80, "Test setup: first byte must be 0x80");

        let format = detect_format(&data2);
        match format {
            ModelFormat::SafeTensors(info) => {
                assert_eq!(info.tensor_count, 1);
                assert_eq!(info.metadata.get("format"), Some(&"pt".to_string()));
            }
            other => panic!("Expected SafeTensors but got {:?} — pacha#4 regression", other),
        }
    }

    #[test]
    fn test_safetensors_invalid_header_size() {
        // Header size larger than file
        let header_size = 1_000_000u64;
        let mut data = Vec::new();
        data.extend_from_slice(&header_size.to_le_bytes());
        data.extend_from_slice(b"{}");

        let format = detect_format(&data);
        // Should still identify as SafeTensors due to JSON structure
        assert!(matches!(format, ModelFormat::SafeTensors(_)));
    }

    #[test]
    fn test_gguf_info_default() {
        let info = GgufInfo::default();
        assert_eq!(info.version, 0);
        assert!(info.architecture.is_none());
    }

    #[test]
    fn test_apr_info_default() {
        let info = AprInfo::default();
        assert_eq!(info.version, 0);
        assert!(!info.compressed);
        assert!(!info.encrypted);
    }
}
