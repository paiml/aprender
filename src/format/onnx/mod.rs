//! ONNX format reader (GH-238)
//!
//! Lightweight ONNX protobuf parser that extracts tensor initializers
//! (weights) from `.onnx` files without requiring the `prost` crate.
//!
//! # ONNX Protobuf Layout (simplified)
//!
//! ```text
//! ModelProto {
//!   ir_version: int64        (field 1)
//!   graph: GraphProto        (field 7)
//!     initializer: [TensorProto]  (field 5, repeated)
//!       dims: [int64]        (field 1, repeated/packed)
//!       data_type: int32     (field 2)
//!       name: string         (field 8)
//!       raw_data: bytes      (field 13)
//!       float_data: [float]  (field 4, packed)
//! }
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use aprender::format::onnx::OnnxReader;
//!
//! let reader = OnnxReader::from_file("model.onnx")?;
//! for tensor in reader.tensors() {
//!     println!("{}: {:?} ({:?})", tensor.name, tensor.shape, tensor.data_type);
//! }
//! ```

use crate::error::{AprenderError, Result};
use std::collections::BTreeMap;
use std::path::Path;

/// ONNX data types (from onnx.proto3 TensorProto.DataType)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxDataType {
    Float,
    Uint8,
    Int8,
    Uint16,
    Int16,
    Int32,
    Int64,
    String,
    Bool,
    Float16,
    Double,
    Uint32,
    Uint64,
    BFloat16,
    Unknown(i32),
}

impl OnnxDataType {
    fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::Float,
            2 => Self::Uint8,
            3 => Self::Int8,
            4 => Self::Uint16,
            5 => Self::Int16,
            6 => Self::Int32,
            7 => Self::Int64,
            8 => Self::String,
            9 => Self::Bool,
            10 => Self::Float16,
            11 => Self::Double,
            12 => Self::Uint32,
            13 => Self::Uint64,
            16 => Self::BFloat16,
            other => Self::Unknown(other),
        }
    }

    /// Bytes per element for this data type
    pub fn element_size(&self) -> usize {
        match self {
            Self::Float | Self::Int32 | Self::Uint32 => 4,
            Self::Double | Self::Int64 | Self::Uint64 => 8,
            Self::Float16 | Self::BFloat16 | Self::Int16 | Self::Uint16 => 2,
            Self::Uint8 | Self::Int8 | Self::Bool => 1,
            Self::String | Self::Unknown(_) => 0,
        }
    }
}

/// A tensor extracted from an ONNX file
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    /// Tensor name
    pub name: String,
    /// Tensor shape (dimensions)
    pub shape: Vec<usize>,
    /// Data type
    pub data_type: OnnxDataType,
    /// Raw bytes of tensor data
    pub raw_data: Vec<u8>,
}

impl OnnxTensor {
    /// Convert tensor data to f32 values
    pub fn to_f32(&self) -> Vec<f32> {
        match self.data_type {
            OnnxDataType::Float => self
                .raw_data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            OnnxDataType::Float16 => self
                .raw_data
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    f16_to_f32(bits)
                })
                .collect(),
            OnnxDataType::Double => self
                .raw_data
                .chunks_exact(8)
                .map(|b| {
                    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect(),
            OnnxDataType::Int8 => self.raw_data.iter().map(|&b| (b as i8) as f32).collect(),
            OnnxDataType::Uint8 => self.raw_data.iter().map(|&b| b as f32).collect(),
            OnnxDataType::Int32 => self
                .raw_data
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
                .collect(),
            OnnxDataType::Int64 => self
                .raw_data
                .chunks_exact(8)
                .map(|b| {
                    i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect(),
            _ => Vec::new(),
        }
    }
}

/// Convert IEEE 754 half-precision to single-precision
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut m = mantissa;
            let mut e = 0u32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            let f32_exp = 127 - 15 - e;
            let f32_mant = (m & 0x3FF) << 13;
            f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant)
        }
    } else if exponent == 31 {
        // Inf/NaN
        let f32_mant = mantissa << 13;
        f32::from_bits((sign << 31) | (0xFF << 23) | f32_mant)
    } else {
        // Normalized
        let f32_exp = exponent + 127 - 15;
        let f32_mant = mantissa << 13;
        f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant)
    }
}

/// ONNX model metadata
#[derive(Debug, Clone, Default)]
pub struct OnnxMetadata {
    /// IR version
    pub ir_version: i64,
    /// Producer name
    pub producer_name: String,
    /// Producer version
    pub producer_version: String,
    /// Domain
    pub domain: String,
    /// Model version
    pub model_version: i64,
    /// Doc string
    pub doc_string: String,
    /// Opset imports
    pub opset_versions: Vec<(String, i64)>,
}

/// ONNX file reader
#[derive(Debug)]
pub struct OnnxReader {
    /// Extracted tensors
    tensors: Vec<OnnxTensor>,
    /// Model metadata
    metadata: OnnxMetadata,
}

impl OnnxReader {
    /// Read an ONNX file from disk
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let data = std::fs::read(path.as_ref()).map_err(|e| AprenderError::FormatError {
            message: format!("Failed to read ONNX file: {e}"),
        })?;
        Self::from_bytes(&data)
    }

    /// Parse ONNX data from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut reader = ProtobufReader::new(data);
        let mut metadata = OnnxMetadata::default();
        let mut tensors = Vec::new();

        // Parse ModelProto fields
        while reader.has_more() {
            let (field_num, wire_type) = reader.read_tag()?;
            match (field_num, wire_type) {
                // ir_version (field 1, varint)
                (1, 0) => {
                    metadata.ir_version = reader.read_varint()? as i64;
                }
                // producer_name (field 2, length-delimited)
                (2, 2) => {
                    metadata.producer_name = reader.read_string()?;
                }
                // producer_version (field 3, length-delimited)
                (3, 2) => {
                    metadata.producer_version = reader.read_string()?;
                }
                // domain (field 4, length-delimited)
                (4, 2) => {
                    metadata.domain = reader.read_string()?;
                }
                // model_version (field 5, varint)
                (5, 0) => {
                    metadata.model_version = reader.read_varint()? as i64;
                }
                // doc_string (field 6, length-delimited)
                (6, 2) => {
                    metadata.doc_string = reader.read_string()?;
                }
                // graph (field 7, length-delimited)
                (7, 2) => {
                    let graph_data = reader.read_bytes()?;
                    tensors = Self::parse_graph(graph_data)?;
                }
                // opset_import (field 8, length-delimited)
                (8, 2) => {
                    let opset_data = reader.read_bytes()?;
                    if let Ok((domain, version)) = Self::parse_opset_import(opset_data) {
                        metadata.opset_versions.push((domain, version));
                    }
                }
                // Skip unknown fields
                (_, 0) => {
                    reader.read_varint()?;
                }
                (_, 1) => {
                    reader.skip(8)?;
                }
                (_, 2) => {
                    let len = reader.read_varint()? as usize;
                    reader.skip(len)?;
                }
                (_, 5) => {
                    reader.skip(4)?;
                }
                _ => {
                    return Err(AprenderError::FormatError {
                        message: format!(
                            "Unknown protobuf wire type {wire_type} for field {field_num}"
                        ),
                    });
                }
            }
        }

        Ok(Self { tensors, metadata })
    }

    /// Get extracted tensors
    pub fn tensors(&self) -> &[OnnxTensor] {
        &self.tensors
    }

    /// Get model metadata
    pub fn metadata(&self) -> &OnnxMetadata {
        &self.metadata
    }

    /// Convert all tensors to F32 and return as BTreeMap
    pub fn to_f32_tensors(&self) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        let mut result = BTreeMap::new();
        for tensor in &self.tensors {
            let f32_data = tensor.to_f32();
            if !f32_data.is_empty() {
                result.insert(tensor.name.clone(), (f32_data, tensor.shape.clone()));
            }
        }
        result
    }

    /// Parse GraphProto to extract initializer tensors
    fn parse_graph(data: &[u8]) -> Result<Vec<OnnxTensor>> {
        let mut reader = ProtobufReader::new(data);
        let mut tensors = Vec::new();

        while reader.has_more() {
            let (field_num, wire_type) = reader.read_tag()?;
            match (field_num, wire_type) {
                // initializer (field 5, repeated length-delimited TensorProto)
                (5, 2) => {
                    let tensor_data = reader.read_bytes()?;
                    if let Ok(tensor) = Self::parse_tensor_proto(tensor_data) {
                        tensors.push(tensor);
                    }
                }
                // Skip other fields
                (_, 0) => {
                    reader.read_varint()?;
                }
                (_, 1) => {
                    reader.skip(8)?;
                }
                (_, 2) => {
                    let len = reader.read_varint()? as usize;
                    reader.skip(len)?;
                }
                (_, 5) => {
                    reader.skip(4)?;
                }
                _ => {
                    reader.read_varint()?;
                }
            }
        }

        Ok(tensors)
    }

    /// Parse OperatorSetIdProto
    fn parse_opset_import(data: &[u8]) -> Result<(String, i64)> {
        let mut reader = ProtobufReader::new(data);
        let mut domain = String::new();
        let mut version = 0i64;

        while reader.has_more() {
            let (field_num, wire_type) = reader.read_tag()?;
            match (field_num, wire_type) {
                (1, 2) => domain = reader.read_string()?,
                (2, 0) => version = reader.read_varint()? as i64,
                (_, 0) => {
                    reader.read_varint()?;
                }
                (_, 2) => {
                    let len = reader.read_varint()? as usize;
                    reader.skip(len)?;
                }
                _ => break,
            }
        }

        Ok((domain, version))
    }

    /// Parse TensorProto
    fn parse_tensor_proto(data: &[u8]) -> Result<OnnxTensor> {
        let mut reader = ProtobufReader::new(data);
        let mut name = String::new();
        let mut dims: Vec<usize> = Vec::new();
        let mut data_type = OnnxDataType::Float;
        let mut raw_data: Vec<u8> = Vec::new();
        let mut float_data: Vec<f32> = Vec::new();
        let mut int32_data: Vec<i32> = Vec::new();
        let mut int64_data: Vec<i64> = Vec::new();
        let mut double_data: Vec<f64> = Vec::new();

        while reader.has_more() {
            let (field_num, wire_type) = reader.read_tag()?;
            match (field_num, wire_type) {
                // dims (field 1, repeated int64 - packed or unpacked)
                (1, 0) => {
                    dims.push(reader.read_varint()? as usize);
                }
                (1, 2) => {
                    // Packed repeated int64
                    let packed = reader.read_bytes()?;
                    let mut pr = ProtobufReader::new(packed);
                    while pr.has_more() {
                        dims.push(pr.read_varint()? as usize);
                    }
                }
                // data_type (field 2, varint)
                (2, 0) => {
                    data_type = OnnxDataType::from_i32(reader.read_varint()? as i32);
                }
                // float_data (field 4, packed repeated float)
                (4, 2) => {
                    let packed = reader.read_bytes()?;
                    for chunk in packed.chunks_exact(4) {
                        float_data
                            .push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    }
                }
                (4, 5) => {
                    float_data.push(reader.read_f32()?);
                }
                // int32_data (field 5, packed repeated int32)
                (5, 2) => {
                    let packed = reader.read_bytes()?;
                    let mut pr = ProtobufReader::new(packed);
                    while pr.has_more() {
                        int32_data.push(pr.read_varint()? as i32);
                    }
                }
                (5, 0) => {
                    int32_data.push(reader.read_varint()? as i32);
                }
                // int64_data (field 7, packed repeated int64)
                (7, 2) => {
                    let packed = reader.read_bytes()?;
                    let mut pr = ProtobufReader::new(packed);
                    while pr.has_more() {
                        int64_data.push(pr.read_varint()? as i64);
                    }
                }
                (7, 0) => {
                    int64_data.push(reader.read_varint()? as i64);
                }
                // name (field 8, string)
                (8, 2) => {
                    name = reader.read_string()?;
                }
                // double_data (field 10, packed repeated double)
                (10, 2) => {
                    let packed = reader.read_bytes()?;
                    for chunk in packed.chunks_exact(8) {
                        double_data.push(f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]));
                    }
                }
                (10, 1) => {
                    double_data.push(reader.read_f64()?);
                }
                // raw_data (field 13, bytes)
                (13, 2) => {
                    raw_data = reader.read_bytes()?.to_vec();
                }
                // Skip unknown fields
                (_, 0) => {
                    reader.read_varint()?;
                }
                (_, 1) => {
                    reader.skip(8)?;
                }
                (_, 2) => {
                    let len = reader.read_varint()? as usize;
                    reader.skip(len)?;
                }
                (_, 5) => {
                    reader.skip(4)?;
                }
                _ => {
                    reader.read_varint()?;
                }
            }
        }

        // If raw_data is empty, reconstruct from typed arrays
        if raw_data.is_empty() {
            if !float_data.is_empty() {
                raw_data = float_data.iter().flat_map(|f| f.to_le_bytes()).collect();
            } else if !int32_data.is_empty() {
                raw_data = int32_data.iter().flat_map(|i| i.to_le_bytes()).collect();
            } else if !int64_data.is_empty() {
                raw_data = int64_data.iter().flat_map(|i| i.to_le_bytes()).collect();
            } else if !double_data.is_empty() {
                raw_data = double_data.iter().flat_map(|d| d.to_le_bytes()).collect();
            }
        }

        Ok(OnnxTensor {
            name,
            shape: dims,
            data_type,
            raw_data,
        })
    }
}

/// Minimal protobuf wire format reader
struct ProtobufReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ProtobufReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn has_more(&self) -> bool {
        self.pos < self.data.len()
    }

    fn read_tag(&mut self) -> Result<(u32, u32)> {
        let varint = self.read_varint()?;
        let field_num = (varint >> 3) as u32;
        let wire_type = (varint & 0x7) as u32;
        Ok((field_num, wire_type))
    }

    fn read_varint(&mut self) -> Result<u64> {
        let mut result: u64 = 0;
        let mut shift = 0;
        loop {
            if self.pos >= self.data.len() {
                return Err(AprenderError::FormatError {
                    message: "Unexpected end of protobuf data".to_string(),
                });
            }
            let byte = self.data[self.pos];
            self.pos += 1;
            result |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
            if shift >= 64 {
                return Err(AprenderError::FormatError {
                    message: "Varint overflow".to_string(),
                });
            }
        }
    }

    fn read_bytes(&mut self) -> Result<&'a [u8]> {
        let len = self.read_varint()? as usize;
        if self.pos + len > self.data.len() {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Protobuf length-delimited field extends past data ({} + {} > {})",
                    self.pos,
                    len,
                    self.data.len()
                ),
            });
        }
        let result = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(result)
    }

    fn read_string(&mut self) -> Result<String> {
        let bytes = self.read_bytes()?;
        String::from_utf8(bytes.to_vec()).map_err(|_| AprenderError::FormatError {
            message: "Invalid UTF-8 in protobuf string".to_string(),
        })
    }

    fn read_f32(&mut self) -> Result<f32> {
        if self.pos + 4 > self.data.len() {
            return Err(AprenderError::FormatError {
                message: "Unexpected end reading f32".to_string(),
            });
        }
        let bytes = [
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ];
        self.pos += 4;
        Ok(f32::from_le_bytes(bytes))
    }

    fn read_f64(&mut self) -> Result<f64> {
        if self.pos + 8 > self.data.len() {
            return Err(AprenderError::FormatError {
                message: "Unexpected end reading f64".to_string(),
            });
        }
        let bytes = [
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
            self.data[self.pos + 4],
            self.data[self.pos + 5],
            self.data[self.pos + 6],
            self.data[self.pos + 7],
        ];
        self.pos += 8;
        Ok(f64::from_le_bytes(bytes))
    }

    fn skip(&mut self, n: usize) -> Result<()> {
        if self.pos + n > self.data.len() {
            return Err(AprenderError::FormatError {
                message: "Unexpected end skipping protobuf data".to_string(),
            });
        }
        self.pos += n;
        Ok(())
    }
}

/// Check if a file is an ONNX model by reading the first few bytes
pub fn is_onnx_file(path: &Path) -> bool {
    // Check extension first
    if path.extension().and_then(|e| e.to_str()) == Some("onnx") {
        return true;
    }
    // Check protobuf magic (ONNX starts with varint tag for field 1, wire type 0)
    // Field 1 (ir_version) with varint wire type = tag byte 0x08
    std::fs::read(path)
        .ok()
        .is_some_and(|data| data.len() > 4 && data[0] == 0x08)
}

/// Check if a file is a NeMo archive (.nemo = tar.gz)
pub fn is_nemo_file(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some("nemo")
}

#[cfg(test)]
mod tests;
