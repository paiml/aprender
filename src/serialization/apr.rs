//! APR (Aprender) binary format with JSON metadata.
//!
//! A compact binary format optimized for WASM deployment with:
//! - LZ4 compression support
//! - JSON metadata section (vocab, config, tokenizer, etc.)
//! - Streaming decompression capability
//!
//! Format:
//! ```text
//! [4-byte magic: "APR1"]
//! [4-byte metadata_len: u32 little-endian]
//! [JSON metadata: arbitrary key-value pairs]
//! [4-byte n_tensors: u32 little-endian]
//! [Tensor index: name, dtype, shape, offset, size per tensor]
//! [Raw tensor data: values in little-endian]
//! [4-byte CRC32: checksum of all preceding bytes]
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// Magic bytes for APR format
pub const APR_MAGIC: [u8; 4] = [b'A', b'P', b'R', b'1'];

/// Tensor descriptor in the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AprTensorDescriptor {
    /// Tensor name
    pub name: String,
    /// Data type (e.g., "F32", "I8")
    pub dtype: String,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Byte offset in data section
    pub offset: usize,
    /// Byte size
    pub size: usize,
}

/// APR file metadata - arbitrary JSON
pub type AprMetadata = BTreeMap<String, JsonValue>;

/// APR format reader
#[derive(Debug)]
pub struct AprReader {
    /// Parsed metadata
    pub metadata: AprMetadata,
    /// Tensor descriptors
    pub tensors: Vec<AprTensorDescriptor>,
    /// Raw file data
    data: Vec<u8>,
    /// Offset to tensor data section
    tensor_data_offset: usize,
}

impl AprReader {
    /// Load APR file from path
    ///
    /// # Errors
    /// Returns error if file is invalid or cannot be read
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let data = fs::read(path).map_err(|e| format!("Failed to read file: {e}"))?;
        Self::from_bytes(data)
    }

    /// Parse APR format from bytes
    ///
    /// # Errors
    /// Returns error if format is invalid
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, String> {
        // Validate magic
        if data.len() < 8 {
            return Err("File too short".to_string());
        }
        if data[0..4] != APR_MAGIC {
            return Err(format!(
                "Invalid magic: expected APR1, got {:?}",
                &data[0..4]
            ));
        }

        // Read metadata length
        let metadata_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

        if data.len() < 8 + metadata_len + 4 {
            return Err("File too short for metadata".to_string());
        }

        // Parse metadata JSON
        let metadata_json = &data[8..8 + metadata_len];
        let metadata: AprMetadata = if metadata_len > 0 {
            serde_json::from_slice(metadata_json)
                .map_err(|e| format!("Invalid metadata JSON: {e}"))?
        } else {
            BTreeMap::new()
        };

        // Read tensor count
        let tensor_count_offset = 8 + metadata_len;
        let n_tensors = u32::from_le_bytes([
            data[tensor_count_offset],
            data[tensor_count_offset + 1],
            data[tensor_count_offset + 2],
            data[tensor_count_offset + 3],
        ]) as usize;

        // Read tensor index (JSON array)
        let index_offset = tensor_count_offset + 4;

        // Find end of tensor index by reading the index length
        let index_len = u32::from_le_bytes([
            data[index_offset],
            data[index_offset + 1],
            data[index_offset + 2],
            data[index_offset + 3],
        ]) as usize;

        let index_data = &data[index_offset + 4..index_offset + 4 + index_len];
        let tensors: Vec<AprTensorDescriptor> = if n_tensors > 0 {
            serde_json::from_slice(index_data).map_err(|e| format!("Invalid tensor index: {e}"))?
        } else {
            Vec::new()
        };

        let tensor_data_offset = index_offset + 4 + index_len;

        Ok(Self {
            metadata,
            tensors,
            data,
            tensor_data_offset,
        })
    }

    /// Get metadata value by key
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&JsonValue> {
        self.metadata.get(key)
    }

    /// Read tensor data as f32 values
    ///
    /// # Errors
    /// Returns error if tensor not found or data invalid
    pub fn read_tensor_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let desc = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("Tensor not found: {name}"))?;

        let start = self.tensor_data_offset + desc.offset;
        let end = start + desc.size;

        if end > self.data.len() {
            return Err("Tensor data out of bounds".to_string());
        }

        let bytes = &self.data[start..end];
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(values)
    }
}

/// APR format writer
#[derive(Debug, Default)]
pub struct AprWriter {
    /// Metadata to write
    metadata: AprMetadata,
    /// Tensors to write
    tensors: Vec<(AprTensorDescriptor, Vec<u8>)>,
}

impl AprWriter {
    /// Create new writer
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set metadata key-value pair
    pub fn set_metadata(&mut self, key: impl Into<String>, value: JsonValue) {
        self.metadata.insert(key.into(), value);
    }

    /// Add tensor with f32 data
    pub fn add_tensor_f32(&mut self, name: impl Into<String>, shape: Vec<usize>, data: &[f32]) {
        let name = name.into();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let size = bytes.len();

        // Calculate offset based on existing tensors
        let offset: usize = self.tensors.iter().map(|(_, d)| d.len()).sum();

        let desc = AprTensorDescriptor {
            name,
            dtype: "F32".to_string(),
            shape,
            offset,
            size,
        };

        self.tensors.push((desc, bytes));
    }

    /// Write to bytes
    ///
    /// # Errors
    /// Returns error if serialization fails
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let mut output = Vec::new();

        // 1. Magic
        output.extend_from_slice(&APR_MAGIC);

        // 2. Metadata
        let metadata_json = serde_json::to_string(&self.metadata)
            .map_err(|e| format!("Metadata serialization failed: {e}"))?;
        let metadata_bytes = metadata_json.as_bytes();
        output.extend_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        output.extend_from_slice(metadata_bytes);

        // 3. Tensor count
        output.extend_from_slice(&(self.tensors.len() as u32).to_le_bytes());

        // 4. Tensor index
        let descriptors: Vec<_> = self.tensors.iter().map(|(d, _)| d).collect();
        let index_json = serde_json::to_string(&descriptors)
            .map_err(|e| format!("Index serialization failed: {e}"))?;
        let index_bytes = index_json.as_bytes();
        output.extend_from_slice(&(index_bytes.len() as u32).to_le_bytes());
        output.extend_from_slice(index_bytes);

        // 5. Tensor data
        for (_, data) in &self.tensors {
            output.extend_from_slice(data);
        }

        // 6. CRC32
        let crc = crc32(&output);
        output.extend_from_slice(&crc.to_le_bytes());

        Ok(output)
    }

    /// Write to file
    ///
    /// # Errors
    /// Returns error if write fails
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let bytes = self.to_bytes()?;
        fs::write(path, bytes).map_err(|e| format!("Write failed: {e}"))
    }
}

/// Simple CRC32 implementation
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        let idx = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = CRC32_TABLE[idx] ^ (crc >> 8);
    }
    !crc
}

/// CRC32 lookup table (IEEE polynomial)
const CRC32_TABLE: [u32; 256] = [
    0x0000_0000,
    0x7707_3096,
    0xEE0E_612C,
    0x9909_51BA,
    0x076D_C419,
    0x706A_F48F,
    0xE963_A535,
    0x9E64_95A3,
    0x0EDB_8832,
    0x79DC_B8A4,
    0xE0D5_E91E,
    0x97D2_D988,
    0x09B6_4C2B,
    0x7EB1_7CBD,
    0xE7B8_2D07,
    0x90BF_1D91,
    0x1DB7_1064,
    0x6AB0_20F2,
    0xF3B9_7148,
    0x84BE_41DE,
    0x1ADA_D47D,
    0x6DDD_E4EB,
    0xF4D4_B551,
    0x83D3_85C7,
    0x136C_9856,
    0x646B_A8C0,
    0xFD62_F97A,
    0x8A65_C9EC,
    0x1401_5C4F,
    0x6306_6CD9,
    0xFA0F_3D63,
    0x8D08_0DF5,
    0x3B6E_20C8,
    0x4C69_105E,
    0xD560_41E4,
    0xA267_7172,
    0x3C03_E4D1,
    0x4B04_D447,
    0xD20D_85FD,
    0xA50A_B56B,
    0x35B5_A8FA,
    0x42B2_986C,
    0xDBBB_C9D6,
    0xACBC_F940,
    0x32D8_6CE3,
    0x45DF_5C75,
    0xDCD6_0DCF,
    0xABD1_3D59,
    0x26D9_30AC,
    0x51DE_003A,
    0xC8D7_5180,
    0xBFD0_6116,
    0x21B4_F4B5,
    0x56B3_C423,
    0xCFBA_9599,
    0xB8BD_A50F,
    0x2802_B89E,
    0x5F05_8808,
    0xC60C_D9B2,
    0xB10B_E924,
    0x2F6F_7C87,
    0x5868_4C11,
    0xC161_1DAB,
    0xB666_2D3D,
    0x76DC_4190,
    0x01DB_7106,
    0x98D2_20BC,
    0xEFD5_102A,
    0x71B1_8589,
    0x06B6_B51F,
    0x9FBF_E4A5,
    0xE8B8_D433,
    0x7807_C9A2,
    0x0F00_F934,
    0x9609_A88E,
    0xE10E_9818,
    0x7F6A_0DBB,
    0x086D_3D2D,
    0x9164_6C97,
    0xE663_5C01,
    0x6B6B_51F4,
    0x1C6C_6162,
    0x8565_30D8,
    0xF262_004E,
    0x6C06_95ED,
    0x1B01_A57B,
    0x8208_F4C1,
    0xF50F_C457,
    0x65B0_D9C6,
    0x12B7_E950,
    0x8BBE_B8EA,
    0xFCB9_887C,
    0x62DD_1DDF,
    0x15DA_2D49,
    0x8CD3_7CF3,
    0xFBD4_4C65,
    0x4DB2_6158,
    0x3AB5_51CE,
    0xA3BC_0074,
    0xD4BB_30E2,
    0x4ADF_A541,
    0x3DD8_95D7,
    0xA4D1_C46D,
    0xD3D6_F4FB,
    0x4369_E96A,
    0x346E_D9FC,
    0xAD67_8846,
    0xDA60_B8D0,
    0x4404_2D73,
    0x3303_1DE5,
    0xAA0A_4C5F,
    0xDD0D_7CC9,
    0x5005_713C,
    0x2702_41AA,
    0xBE0B_1010,
    0xC90C_2086,
    0x5768_B525,
    0x206F_85B3,
    0xB966_D409,
    0xCE61_E49F,
    0x5EDE_F90E,
    0x29D9_C998,
    0xB0D0_9822,
    0xC7D7_A8B4,
    0x59B3_3D17,
    0x2EB4_0D81,
    0xB7BD_5C3B,
    0xC0BA_6CAD,
    0xEDB8_8320,
    0x9ABF_B3B6,
    0x03B6_E20C,
    0x74B1_D29A,
    0xEAD5_4739,
    0x9DD2_77AF,
    0x04DB_2615,
    0x73DC_1683,
    0xE363_0B12,
    0x9464_3B84,
    0x0D6D_6A3E,
    0x7A6A_5AA8,
    0xE40E_CF0B,
    0x9309_FF9D,
    0x0A00_AE27,
    0x7D07_9EB1,
    0xF00F_9344,
    0x8708_A3D2,
    0x1E01_F268,
    0x6906_C2FE,
    0xF762_575D,
    0x8065_67CB,
    0x196C_3671,
    0x6E6B_06E7,
    0xFED4_1B76,
    0x89D3_2BE0,
    0x10DA_7A5A,
    0x67DD_4ACC,
    0xF9B9_DF6F,
    0x8EBE_EFF9,
    0x17B7_BE43,
    0x60B0_8ED5,
    0xD6D6_A3E8,
    0xA1D1_937E,
    0x38D8_C2C4,
    0x4FDF_F252,
    0xD1BB_67F1,
    0xA6BC_5767,
    0x3FB5_06DD,
    0x48B2_364B,
    0xD80D_2BDA,
    0xAF0A_1B4C,
    0x3603_4AF6,
    0x4104_7A60,
    0xDF60_EFC3,
    0xA867_DF55,
    0x316E_8EEF,
    0x4669_BE79,
    0xCB61_B38C,
    0xBC66_831A,
    0x256F_D2A0,
    0x5268_E236,
    0xCC0C_7795,
    0xBB0B_4703,
    0x2202_16B9,
    0x5505_262F,
    0xC5BA_3BBE,
    0xB2BD_0B28,
    0x2BB4_5A92,
    0x5CB3_6A04,
    0xC2D7_FFA7,
    0xB5D0_CF31,
    0x2CD9_9E8B,
    0x5BDE_AE1D,
    0x9B64_C2B0,
    0xEC63_F226,
    0x756A_A39C,
    0x026D_930A,
    0x9C09_06A9,
    0xEB0E_363F,
    0x7207_6785,
    0x0500_5713,
    0x95BF_4A82,
    0xE2B8_7A14,
    0x7BB1_2BAE,
    0x0CB6_1B38,
    0x92D2_8E9B,
    0xE5D5_BE0D,
    0x7CDC_EFB7,
    0x0BDB_DF21,
    0x86D3_D2D4,
    0xF1D4_E242,
    0x68DD_B3F8,
    0x1FDA_836E,
    0x81BE_16CD,
    0xF6B9_265B,
    0x6FB0_77E1,
    0x18B7_4777,
    0x8808_5AE6,
    0xFF0F_6A70,
    0x6606_3BCA,
    0x1101_0B5C,
    0x8F65_9EFF,
    0xF862_AE69,
    0x616B_FFD3,
    0x166C_CF45,
    0xA00A_E278,
    0xD70D_D2EE,
    0x4E04_8354,
    0x3903_B3C2,
    0xA767_2661,
    0xD060_16F7,
    0x4969_474D,
    0x3E6E_77DB,
    0xAED1_6A4A,
    0xD9D6_5ADC,
    0x40DF_0B66,
    0x37D8_3BF0,
    0xA9BC_AE53,
    0xDEBB_9EC5,
    0x47B2_CF7F,
    0x30B5_FFE9,
    0xBDBD_F21C,
    0xCABA_C28A,
    0x53B3_9330,
    0x24B4_A3A6,
    0xBAD0_3605,
    0xCDD7_0693,
    0x54DE_5729,
    0x23D9_67BF,
    0xB366_7A2E,
    0xC461_4AB8,
    0x5D68_1B02,
    0x2A6F_2B94,
    0xB40B_BE37,
    0xC30C_8EA1,
    0x5A05_DF1B,
    0x2D02_EF8D,
];

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Magic and Format Tests
    // =========================================================================

    #[test]
    fn test_apr_magic() {
        assert_eq!(APR_MAGIC, [b'A', b'P', b'R', b'1']);
    }

    #[test]
    fn test_writer_creates_valid_magic() {
        let writer = AprWriter::new();
        let bytes = writer.to_bytes().unwrap();
        assert_eq!(&bytes[0..4], &APR_MAGIC);
    }

    // =========================================================================
    // Metadata Tests
    // =========================================================================

    #[test]
    fn test_empty_metadata() {
        let writer = AprWriter::new();
        let bytes = writer.to_bytes().unwrap();
        let reader = AprReader::from_bytes(bytes).unwrap();
        assert!(reader.metadata.is_empty());
    }

    #[test]
    fn test_string_metadata() {
        let mut writer = AprWriter::new();
        writer.set_metadata("model_name", JsonValue::String("whisper-tiny".into()));

        let bytes = writer.to_bytes().unwrap();
        let reader = AprReader::from_bytes(bytes).unwrap();

        assert_eq!(
            reader.get_metadata("model_name"),
            Some(&JsonValue::String("whisper-tiny".into()))
        );
    }

    #[test]
    fn test_numeric_metadata() {
        let mut writer = AprWriter::new();
        writer.set_metadata("n_vocab", JsonValue::Number(51865.into()));
        writer.set_metadata("n_layers", JsonValue::Number(4.into()));

        let bytes = writer.to_bytes().unwrap();
        let reader = AprReader::from_bytes(bytes).unwrap();

        assert_eq!(
            reader.get_metadata("n_vocab"),
            Some(&JsonValue::Number(51865.into()))
        );
    }

    #[test]
    fn test_array_metadata() {
        let mut writer = AprWriter::new();
        let vocab = vec![
            JsonValue::String("hello".into()),
            JsonValue::String("world".into()),
        ];
        writer.set_metadata("vocab", JsonValue::Array(vocab));

        let bytes = writer.to_bytes().unwrap();
        let reader = AprReader::from_bytes(bytes).unwrap();

        let vocab = reader.get_metadata("vocab").unwrap();
        assert!(vocab.is_array());
        assert_eq!(vocab.as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_object_metadata() {
        let mut writer = AprWriter::new();
        let mut config = serde_json::Map::new();
        config.insert("dim".into(), JsonValue::Number(384.into()));
        config.insert("heads".into(), JsonValue::Number(6.into()));
        writer.set_metadata("config", JsonValue::Object(config));

        let bytes = writer.to_bytes().unwrap();
        let reader = AprReader::from_bytes(bytes).unwrap();

        let config = reader.get_metadata("config").unwrap();
        assert!(config.is_object());
    }

    // =========================================================================
    // Tensor Tests
    // =========================================================================

    #[test]
    fn test_no_tensors() {
        let writer = AprWriter::new();
        let bytes = writer.to_bytes().unwrap();
        let reader = AprReader::from_bytes(bytes).unwrap();
        assert!(reader.tensors.is_empty());
    }

    #[test]
    fn test_single_tensor() {
        let mut writer = AprWriter::new();
        writer.add_tensor_f32("weights", vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let bytes = writer.to_bytes().unwrap();
        let reader = AprReader::from_bytes(bytes).unwrap();

        assert_eq!(reader.tensors.len(), 1);
        assert_eq!(reader.tensors[0].name, "weights");
        assert_eq!(reader.tensors[0].shape, vec![2, 3]);

        let data = reader.read_tensor_f32("weights").unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_multiple_tensors() {
        let mut writer = AprWriter::new();
        writer.add_tensor_f32("a", vec![2], &[1.0, 2.0]);
        writer.add_tensor_f32("b", vec![3], &[3.0, 4.0, 5.0]);

        let bytes = writer.to_bytes().unwrap();
        let reader = AprReader::from_bytes(bytes).unwrap();

        assert_eq!(reader.tensors.len(), 2);

        let a = reader.read_tensor_f32("a").unwrap();
        let b = reader.read_tensor_f32("b").unwrap();

        assert_eq!(a, vec![1.0, 2.0]);
        assert_eq!(b, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_tensor_not_found() {
        let writer = AprWriter::new();
        let bytes = writer.to_bytes().unwrap();
        let reader = AprReader::from_bytes(bytes).unwrap();

        let result = reader.read_tensor_f32("nonexistent");
        assert!(result.is_err());
    }

    // =========================================================================
    // Combined Metadata + Tensor Tests
    // =========================================================================

    #[test]
    fn test_metadata_and_tensors() {
        let mut writer = AprWriter::new();

        // Add metadata
        writer.set_metadata("model", JsonValue::String("test".into()));
        writer.set_metadata("version", JsonValue::Number(1.into()));

        // Add tensors
        writer.add_tensor_f32("layer.0.weight", vec![4, 4], &vec![0.5; 16]);
        writer.add_tensor_f32("layer.0.bias", vec![4], &[0.1, 0.2, 0.3, 0.4]);

        let bytes = writer.to_bytes().unwrap();
        let reader = AprReader::from_bytes(bytes).unwrap();

        // Verify metadata
        assert_eq!(
            reader.get_metadata("model"),
            Some(&JsonValue::String("test".into()))
        );

        // Verify tensors
        let weight = reader.read_tensor_f32("layer.0.weight").unwrap();
        assert_eq!(weight.len(), 16);

        let bias = reader.read_tensor_f32("layer.0.bias").unwrap();
        assert_eq!(bias, vec![0.1, 0.2, 0.3, 0.4]);
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_invalid_magic() {
        let data = vec![b'X', b'Y', b'Z', b'1', 0, 0, 0, 0];
        let result = AprReader::from_bytes(data);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid magic"));
    }

    #[test]
    fn test_file_too_short() {
        let data = vec![b'A', b'P', b'R'];
        let result = AprReader::from_bytes(data);
        assert!(result.is_err());
    }

    // =========================================================================
    // CRC32 Tests
    // =========================================================================

    #[test]
    fn test_crc32_empty() {
        assert_eq!(crc32(&[]), 0x0000_0000);
    }

    #[test]
    fn test_crc32_hello() {
        // Known CRC32 for "hello"
        let crc = crc32(b"hello");
        assert_eq!(crc, 0x3610_A686);
    }

    // =========================================================================
    // Roundtrip Tests
    // =========================================================================

    #[test]
    fn test_full_roundtrip() {
        let mut writer = AprWriter::new();

        // Complex metadata
        let mut bpe_merges = Vec::new();
        bpe_merges.push(JsonValue::Array(vec![
            JsonValue::String("h".into()),
            JsonValue::String("e".into()),
        ]));
        bpe_merges.push(JsonValue::Array(vec![
            JsonValue::String("he".into()),
            JsonValue::String("llo".into()),
        ]));
        writer.set_metadata("bpe_merges", JsonValue::Array(bpe_merges));

        // Tensors
        writer.add_tensor_f32("embed", vec![100, 64], &vec![0.1; 6400]);

        let bytes = writer.to_bytes().unwrap();
        let reader = AprReader::from_bytes(bytes).unwrap();

        // Verify
        let merges = reader.get_metadata("bpe_merges").unwrap();
        assert_eq!(merges.as_array().unwrap().len(), 2);

        let embed = reader.read_tensor_f32("embed").unwrap();
        assert_eq!(embed.len(), 6400);
    }
}
