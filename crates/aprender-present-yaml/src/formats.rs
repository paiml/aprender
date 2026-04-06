//! File format loaders for Aprender (.apr) and Alimentar (.ald) files.
//!
//! # File Formats
//!
//! ## Alimentar Dataset (.ald)
//!
//! Binary format for tensor datasets:
//! ```text
//! [4 bytes] Magic: "ALD\0"
//! [4 bytes] Version (u32 LE)
//! [4 bytes] Num tensors (u32 LE)
//! For each tensor:
//!   [4 bytes] Name length (u32 LE)
//!   [N bytes] Name (UTF-8)
//!   [4 bytes] Dtype (u32 LE): 0=f32, 1=f64, 2=i32, 3=i64, 4=u8
//!   [4 bytes] Num dimensions (u32 LE)
//!   [4*D bytes] Shape (D x u32 LE)
//!   [N bytes] Data (raw bytes)
//! ```
//!
//! ## Aprender Model (.apr)
//!
//! Binary format for trained models:
//! ```text
//! [4 bytes] Magic: "APR\0"
//! [4 bytes] Version (u32 LE)
//! [4 bytes] Model type length (u32 LE)
//! [N bytes] Model type (UTF-8)
//! [4 bytes] Num layers (u32 LE)
//! For each layer:
//!   [4 bytes] Layer type length (u32 LE)
//!   [N bytes] Layer type (UTF-8)
//!   [4 bytes] Num parameters (u32 LE)
//!   For each parameter:
//!     [Tensor data as in ALD]
//! [Metadata section]
//! ```

use std::io::{self, Read, Write};

/// Data type for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DType {
    /// 32-bit float
    F32 = 0,
    /// 64-bit float
    F64 = 1,
    /// 32-bit signed integer
    I32 = 2,
    /// 64-bit signed integer
    I64 = 3,
    /// 8-bit unsigned integer
    U8 = 4,
}

impl DType {
    /// Get byte size of one element.
    #[must_use]
    pub const fn size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::U8 => 1,
        }
    }

    /// Parse from u32.
    #[must_use]
    pub const fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F64),
            2 => Some(Self::I32),
            3 => Some(Self::I64),
            4 => Some(Self::U8),
            _ => None,
        }
    }
}

/// A tensor with shape and data.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor name
    pub name: String,
    /// Data type
    pub dtype: DType,
    /// Shape dimensions
    pub shape: Vec<u32>,
    /// Raw data bytes
    pub data: Vec<u8>,
}

impl Tensor {
    /// Create a new tensor.
    #[must_use]
    pub fn new(name: impl Into<String>, dtype: DType, shape: Vec<u32>, data: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape,
            data,
        }
    }

    /// Get number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// Get expected data size in bytes.
    #[must_use]
    pub fn expected_size(&self) -> usize {
        self.numel() * self.dtype.size()
    }

    /// Validate tensor data size.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.data.len() == self.expected_size()
    }

    /// Get data as f32 vector (if dtype is F32).
    pub fn to_f32_vec(&self) -> Option<Vec<f32>> {
        if self.dtype != DType::F32 {
            return None;
        }
        let floats: Vec<f32> = self
            .data
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk.try_into().expect("chunk size");
                f32::from_le_bytes(arr)
            })
            .collect();
        Some(floats)
    }

    /// Create f32 tensor from slice.
    #[must_use]
    pub fn from_f32(name: impl Into<String>, shape: Vec<u32>, data: &[f32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        Self::new(name, DType::F32, shape, bytes)
    }
}

/// Alimentar dataset (.ald file).
#[derive(Debug, Clone)]
pub struct AldDataset {
    /// Format version
    pub version: u32,
    /// Tensors in the dataset
    pub tensors: Vec<Tensor>,
}

/// Magic bytes for .ald files.
const ALD_MAGIC: &[u8; 4] = b"ALD\0";

/// Current .ald format version.
const ALD_VERSION: u32 = 1;

impl AldDataset {
    /// Create a new empty dataset.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: ALD_VERSION,
            tensors: Vec::new(),
        }
    }

    /// Add a tensor to the dataset.
    pub fn add_tensor(&mut self, tensor: Tensor) {
        self.tensors.push(tensor);
    }

    /// Get tensor by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Load from bytes.
    ///
    /// # Errors
    ///
    /// Returns error if the format is invalid.
    pub fn load(data: &[u8]) -> Result<Self, FormatError> {
        let mut cursor = io::Cursor::new(data);
        Self::read_from(&mut cursor)
    }

    /// Read from a reader.
    ///
    /// # Errors
    ///
    /// Returns error if the format is invalid.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self, FormatError> {
        // Read magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != ALD_MAGIC {
            return Err(FormatError::InvalidMagic);
        }

        // Read version
        let version = read_u32(reader)?;
        if version > ALD_VERSION {
            return Err(FormatError::UnsupportedVersion(version));
        }

        // Read tensor count
        let num_tensors = read_u32(reader)?;
        let mut tensors = Vec::with_capacity(num_tensors as usize);

        for _ in 0..num_tensors {
            let tensor = read_tensor(reader)?;
            tensors.push(tensor);
        }

        Ok(Self { version, tensors })
    }

    /// Write to bytes.
    #[must_use]
    pub fn save(&self) -> Vec<u8> {
        let mut data = Vec::new();
        self.write_to(&mut data).expect("write to vec");
        data
    }

    /// Write to a writer.
    ///
    /// # Errors
    ///
    /// Returns error if writing fails.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // Write magic
        writer.write_all(ALD_MAGIC)?;

        // Write version
        write_u32(writer, self.version)?;

        // Write tensor count
        write_u32(writer, self.tensors.len() as u32)?;

        // Write each tensor
        for tensor in &self.tensors {
            write_tensor(writer, tensor)?;
        }

        Ok(())
    }
}

impl Default for AldDataset {
    fn default() -> Self {
        Self::new()
    }
}

/// Aprender model (.apr file).
#[derive(Debug, Clone)]
pub struct AprModel {
    /// Format version
    pub version: u32,
    /// Model type (e.g., "linear", "mlp", "transformer")
    pub model_type: String,
    /// Model layers
    pub layers: Vec<ModelLayer>,
    /// Model metadata
    pub metadata: std::collections::HashMap<String, String>,
}

/// A model layer with parameters.
#[derive(Debug, Clone)]
pub struct ModelLayer {
    /// Layer type (e.g., "dense", "conv2d", "attention")
    pub layer_type: String,
    /// Layer parameters (weights, biases)
    pub parameters: Vec<Tensor>,
}

/// Magic bytes for .apr files.
const APR_MAGIC: &[u8; 4] = b"APR\0";

/// Current .apr format version.
const APR_VERSION: u32 = 1;

impl AprModel {
    /// Create a new model.
    #[must_use]
    pub fn new(model_type: impl Into<String>) -> Self {
        Self {
            version: APR_VERSION,
            model_type: model_type.into(),
            layers: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add a layer to the model.
    pub fn add_layer(&mut self, layer: ModelLayer) {
        self.layers.push(layer);
    }

    /// Get total parameter count.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|l| &l.parameters)
            .map(Tensor::numel)
            .sum()
    }

    /// Load from bytes.
    ///
    /// # Errors
    ///
    /// Returns error if the format is invalid.
    pub fn load(data: &[u8]) -> Result<Self, FormatError> {
        let mut cursor = io::Cursor::new(data);
        Self::read_from(&mut cursor)
    }

    /// Read from a reader.
    ///
    /// # Errors
    ///
    /// Returns error if the format is invalid.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self, FormatError> {
        // Read magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != APR_MAGIC {
            return Err(FormatError::InvalidMagic);
        }

        // Read version
        let version = read_u32(reader)?;
        if version > APR_VERSION {
            return Err(FormatError::UnsupportedVersion(version));
        }

        // Read model type
        let model_type = read_string(reader)?;

        // Read layers
        let num_layers = read_u32(reader)?;
        let mut layers = Vec::with_capacity(num_layers as usize);

        for _ in 0..num_layers {
            let layer_type = read_string(reader)?;
            let num_params = read_u32(reader)?;
            let mut parameters = Vec::with_capacity(num_params as usize);

            for _ in 0..num_params {
                let tensor = read_tensor(reader)?;
                parameters.push(tensor);
            }

            layers.push(ModelLayer {
                layer_type,
                parameters,
            });
        }

        // Read metadata (optional, remaining bytes)
        let mut metadata = std::collections::HashMap::new();
        while let Ok(key) = read_string(reader) {
            if let Ok(value) = read_string(reader) {
                metadata.insert(key, value);
            } else {
                break;
            }
        }

        Ok(Self {
            version,
            model_type,
            layers,
            metadata,
        })
    }

    /// Write to bytes.
    #[must_use]
    pub fn save(&self) -> Vec<u8> {
        let mut data = Vec::new();
        self.write_to(&mut data).expect("write to vec");
        data
    }

    /// Write to a writer.
    ///
    /// # Errors
    ///
    /// Returns error if writing fails.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // Write magic
        writer.write_all(APR_MAGIC)?;

        // Write version
        write_u32(writer, self.version)?;

        // Write model type
        write_string(writer, &self.model_type)?;

        // Write layers
        write_u32(writer, self.layers.len() as u32)?;
        for layer in &self.layers {
            write_string(writer, &layer.layer_type)?;
            write_u32(writer, layer.parameters.len() as u32)?;
            for param in &layer.parameters {
                write_tensor(writer, param)?;
            }
        }

        // Write metadata
        for (key, value) in &self.metadata {
            write_string(writer, key)?;
            write_string(writer, value)?;
        }

        Ok(())
    }
}

/// Format parsing error.
#[derive(Debug, Clone, PartialEq)]
pub enum FormatError {
    /// Invalid magic bytes
    InvalidMagic,
    /// Unsupported format version
    UnsupportedVersion(u32),
    /// Invalid data type
    InvalidDType(u32),
    /// Truncated data
    TruncatedData,
    /// IO error
    IoError(String),
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMagic => write!(f, "Invalid file magic bytes"),
            Self::UnsupportedVersion(v) => write!(f, "Unsupported format version: {v}"),
            Self::InvalidDType(d) => write!(f, "Invalid dtype: {d}"),
            Self::TruncatedData => write!(f, "Truncated data"),
            Self::IoError(e) => write!(f, "IO error: {e}"),
        }
    }
}

impl std::error::Error for FormatError {}

impl From<io::Error> for FormatError {
    fn from(e: io::Error) -> Self {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            Self::TruncatedData
        } else {
            Self::IoError(e.to_string())
        }
    }
}

// =============================================================================
// Helper functions
// =============================================================================

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, FormatError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn write_u32<W: Write>(writer: &mut W, v: u32) -> io::Result<()> {
    writer.write_all(&v.to_le_bytes())
}

fn read_string<R: Read>(reader: &mut R) -> Result<String, FormatError> {
    let len = read_u32(reader)? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| FormatError::IoError(e.to_string()))
}

fn write_string<W: Write>(writer: &mut W, s: &str) -> io::Result<()> {
    write_u32(writer, s.len() as u32)?;
    writer.write_all(s.as_bytes())
}

fn read_tensor<R: Read>(reader: &mut R) -> Result<Tensor, FormatError> {
    let name = read_string(reader)?;
    let dtype_u32 = read_u32(reader)?;
    let dtype = DType::from_u32(dtype_u32).ok_or(FormatError::InvalidDType(dtype_u32))?;

    let num_dims = read_u32(reader)? as usize;
    let mut shape = Vec::with_capacity(num_dims);
    for _ in 0..num_dims {
        shape.push(read_u32(reader)?);
    }

    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let data_size = numel * dtype.size();
    let mut data = vec![0u8; data_size];
    reader.read_exact(&mut data)?;

    Ok(Tensor {
        name,
        dtype,
        shape,
        data,
    })
}

fn write_tensor<W: Write>(writer: &mut W, tensor: &Tensor) -> io::Result<()> {
    write_string(writer, &tensor.name)?;
    write_u32(writer, tensor.dtype as u32)?;
    write_u32(writer, tensor.shape.len() as u32)?;
    for &dim in &tensor.shape {
        write_u32(writer, dim)?;
    }
    writer.write_all(&tensor.data)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // DType tests
    // =========================================================================

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::F32.size(), 4);
        assert_eq!(DType::F64.size(), 8);
        assert_eq!(DType::I32.size(), 4);
        assert_eq!(DType::I64.size(), 8);
        assert_eq!(DType::U8.size(), 1);
    }

    #[test]
    fn test_dtype_from_u32() {
        assert_eq!(DType::from_u32(0), Some(DType::F32));
        assert_eq!(DType::from_u32(1), Some(DType::F64));
        assert_eq!(DType::from_u32(2), Some(DType::I32));
        assert_eq!(DType::from_u32(3), Some(DType::I64));
        assert_eq!(DType::from_u32(4), Some(DType::U8));
        assert_eq!(DType::from_u32(5), None);
    }

    // =========================================================================
    // Tensor tests
    // =========================================================================

    #[test]
    fn test_tensor_numel() {
        let t = Tensor::new("test", DType::F32, vec![2, 3, 4], vec![0; 96]);
        assert_eq!(t.numel(), 24);
    }

    #[test]
    fn test_tensor_expected_size() {
        let t = Tensor::new("test", DType::F32, vec![2, 3], vec![]);
        assert_eq!(t.expected_size(), 24); // 6 elements * 4 bytes
    }

    #[test]
    fn test_tensor_is_valid() {
        let valid = Tensor::new("test", DType::F32, vec![2, 3], vec![0; 24]);
        assert!(valid.is_valid());

        let invalid = Tensor::new("test", DType::F32, vec![2, 3], vec![0; 10]);
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_tensor_from_f32() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let t = Tensor::from_f32("weights", vec![2, 2], &data);

        assert_eq!(t.name, "weights");
        assert_eq!(t.dtype, DType::F32);
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.data.len(), 16);

        let vec = t.to_f32_vec().unwrap();
        assert_eq!(vec, data.to_vec());
    }

    // =========================================================================
    // AldDataset tests
    // =========================================================================

    #[test]
    fn test_ald_new() {
        let ds = AldDataset::new();
        assert_eq!(ds.version, ALD_VERSION);
        assert!(ds.tensors.is_empty());
    }

    #[test]
    fn test_ald_add_get() {
        let mut ds = AldDataset::new();
        ds.add_tensor(Tensor::from_f32("x", vec![10], &[0.0; 10]));
        ds.add_tensor(Tensor::from_f32("y", vec![5], &[0.0; 5]));

        assert!(ds.get("x").is_some());
        assert!(ds.get("y").is_some());
        assert!(ds.get("z").is_none());
    }

    #[test]
    fn test_ald_roundtrip() {
        let mut ds = AldDataset::new();
        ds.add_tensor(Tensor::from_f32("weights", vec![3, 3], &[1.0; 9]));
        ds.add_tensor(Tensor::from_f32("bias", vec![3], &[0.5; 3]));

        let bytes = ds.save();
        let loaded = AldDataset::load(&bytes).unwrap();

        assert_eq!(loaded.version, ds.version);
        assert_eq!(loaded.tensors.len(), 2);
        assert_eq!(loaded.get("weights").unwrap().shape, vec![3, 3]);
        assert_eq!(loaded.get("bias").unwrap().shape, vec![3]);
    }

    #[test]
    fn test_ald_invalid_magic() {
        let result = AldDataset::load(b"BAAD");
        assert!(matches!(result, Err(FormatError::InvalidMagic)));
    }

    #[test]
    fn test_ald_truncated() {
        let result = AldDataset::load(b"ALD\0");
        assert!(matches!(result, Err(FormatError::TruncatedData)));
    }

    // =========================================================================
    // AprModel tests
    // =========================================================================

    #[test]
    fn test_apr_new() {
        let model = AprModel::new("mlp");
        assert_eq!(model.version, APR_VERSION);
        assert_eq!(model.model_type, "mlp");
        assert!(model.layers.is_empty());
    }

    #[test]
    fn test_apr_param_count() {
        let mut model = AprModel::new("test");
        model.add_layer(ModelLayer {
            layer_type: "dense".to_string(),
            parameters: vec![
                Tensor::from_f32("w", vec![10, 5], &[0.0; 50]),
                Tensor::from_f32("b", vec![5], &[0.0; 5]),
            ],
        });

        assert_eq!(model.param_count(), 55);
    }

    #[test]
    fn test_apr_roundtrip() {
        let mut model = AprModel::new("classifier");
        model.add_layer(ModelLayer {
            layer_type: "dense".to_string(),
            parameters: vec![
                Tensor::from_f32("weight", vec![4, 2], &[1.0; 8]),
                Tensor::from_f32("bias", vec![2], &[0.1, 0.2]),
            ],
        });
        model
            .metadata
            .insert("trained_epochs".to_string(), "100".to_string());

        let bytes = model.save();
        let loaded = AprModel::load(&bytes).unwrap();

        assert_eq!(loaded.model_type, "classifier");
        assert_eq!(loaded.layers.len(), 1);
        assert_eq!(loaded.layers[0].layer_type, "dense");
        assert_eq!(loaded.layers[0].parameters.len(), 2);
    }

    #[test]
    fn test_apr_invalid_magic() {
        let result = AprModel::load(b"NOPE");
        assert!(matches!(result, Err(FormatError::InvalidMagic)));
    }

    // =========================================================================
    // FormatError tests
    // =========================================================================

    #[test]
    fn test_format_error_display() {
        assert!(FormatError::InvalidMagic.to_string().contains("magic"));
        assert!(FormatError::UnsupportedVersion(99)
            .to_string()
            .contains("99"));
        assert!(FormatError::InvalidDType(255).to_string().contains("255"));
        assert!(FormatError::TruncatedData.to_string().contains("Truncated"));
    }
}
