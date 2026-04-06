
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
                    match Self::parse_tensor_proto(tensor_data) {
                        Ok(tensor) => tensors.push(tensor),
                        Err(e) => {
                            eprintln!("[ONNX] Warning: skipping malformed tensor initializer: {e}");
                        }
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
        let mut fields = TensorProtoFields::default();

        while reader.has_more() {
            let (field_num, wire_type) = reader.read_tag()?;
            Self::parse_tensor_field(&mut reader, &mut fields, field_num, wire_type)?;
        }

        Ok(fields.into_tensor())
    }

    /// Parse a single TensorProto field from the protobuf stream
    fn parse_tensor_field(
        reader: &mut ProtobufReader<'_>,
        fields: &mut TensorProtoFields,
        field_num: u32,
        wire_type: u32,
    ) -> Result<()> {
        match (field_num, wire_type) {
            // dims (field 1, repeated int64 - unpacked)
            (1, 0) => fields.dims.push(reader.read_varint()? as usize),
            // dims (field 1, packed repeated int64)
            (1, 2) => reader.read_packed_varints_into(&mut fields.dims)?,
            // data_type (field 2, varint)
            (2, 0) => fields.data_type = OnnxDataType::from_i32(reader.read_varint()? as i32),
            // float_data (field 4, packed repeated float)
            (4, 2) => reader.read_packed_f32_into(&mut fields.float_data)?,
            // float_data (field 4, unpacked float)
            (4, 5) => fields.float_data.push(reader.read_f32()?),
            // int32_data (field 5, packed repeated int32)
            (5, 2) => reader.read_packed_varints_i32_into(&mut fields.int32_data)?,
            // int32_data (field 5, unpacked varint)
            (5, 0) => fields.int32_data.push(reader.read_varint()? as i32),
            // int64_data (field 7, packed repeated int64)
            (7, 2) => reader.read_packed_varints_i64_into(&mut fields.int64_data)?,
            // int64_data (field 7, unpacked varint)
            (7, 0) => fields.int64_data.push(reader.read_varint()? as i64),
            // name (field 8, string)
            (8, 2) => fields.name = reader.read_string()?,
            // raw_data (field 9, bytes) -- PyTorch ONNX convention
            (9, 2) => fields.raw_data = reader.read_bytes()?.to_vec(),
            // double_data (field 10, packed repeated double)
            (10, 2) => reader.read_packed_f64_into(&mut fields.double_data)?,
            // double_data (field 10, unpacked double)
            (10, 1) => fields.double_data.push(reader.read_f64()?),
            // raw_data (field 13, bytes)
            (13, 2) => fields.raw_data = reader.read_bytes()?.to_vec(),
            // Skip unknown fields
            _ => reader.skip_field(wire_type)?,
        }
        Ok(())
    }
}

/// Accumulated fields during `TensorProto` parsing.
///
/// Collects typed data arrays and metadata, then converts to `OnnxTensor`
/// once all fields have been read.
#[derive(Default)]
struct TensorProtoFields {
    name: String,
    dims: Vec<usize>,
    data_type: OnnxDataType,
    raw_data: Vec<u8>,
    float_data: Vec<f32>,
    int32_data: Vec<i32>,
    int64_data: Vec<i64>,
    double_data: Vec<f64>,
}

impl Default for OnnxDataType {
    fn default() -> Self {
        Self::Float
    }
}

impl TensorProtoFields {
    /// Convert accumulated fields into an `OnnxTensor`, reconstructing
    /// `raw_data` from typed arrays if the protobuf did not include it.
    fn into_tensor(self) -> OnnxTensor {
        let raw_data = if !self.raw_data.is_empty() {
            self.raw_data
        } else if !self.float_data.is_empty() {
            self.float_data.iter().flat_map(|f| f.to_le_bytes()).collect()
        } else if !self.int32_data.is_empty() {
            self.int32_data.iter().flat_map(|i| i.to_le_bytes()).collect()
        } else if !self.int64_data.is_empty() {
            self.int64_data.iter().flat_map(|i| i.to_le_bytes()).collect()
        } else if !self.double_data.is_empty() {
            self.double_data.iter().flat_map(|d| d.to_le_bytes()).collect()
        } else {
            Vec::new()
        };

        OnnxTensor {
            name: self.name,
            shape: self.dims,
            data_type: self.data_type,
            raw_data,
        }
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

    /// Skip an unknown protobuf field based on its wire type.
    fn skip_field(&mut self, wire_type: u32) -> Result<()> {
        match wire_type {
            0 => { self.read_varint()?; }
            1 => self.skip(8)?,
            2 => {
                let len = self.read_varint()? as usize;
                self.skip(len)?;
            }
            5 => self.skip(4)?,
            _ => { self.read_varint()?; }
        }
        Ok(())
    }

    /// Read a packed repeated field of varints into a `Vec<usize>`.
    fn read_packed_varints_into(&mut self, out: &mut Vec<usize>) -> Result<()> {
        let packed = self.read_bytes()?;
        let mut pr = ProtobufReader::new(packed);
        while pr.has_more() {
            out.push(pr.read_varint()? as usize);
        }
        Ok(())
    }

    /// Read a packed repeated field of varints into a `Vec<i32>`.
    fn read_packed_varints_i32_into(&mut self, out: &mut Vec<i32>) -> Result<()> {
        let packed = self.read_bytes()?;
        let mut pr = ProtobufReader::new(packed);
        while pr.has_more() {
            out.push(pr.read_varint()? as i32);
        }
        Ok(())
    }

    /// Read a packed repeated field of varints into a `Vec<i64>`.
    fn read_packed_varints_i64_into(&mut self, out: &mut Vec<i64>) -> Result<()> {
        let packed = self.read_bytes()?;
        let mut pr = ProtobufReader::new(packed);
        while pr.has_more() {
            out.push(pr.read_varint()? as i64);
        }
        Ok(())
    }

    /// Read a packed repeated field of little-endian f32 values.
    fn read_packed_f32_into(&mut self, out: &mut Vec<f32>) -> Result<()> {
        let packed = self.read_bytes()?;
        for chunk in packed.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(())
    }

    /// Read a packed repeated field of little-endian f64 values.
    fn read_packed_f64_into(&mut self, out: &mut Vec<f64>) -> Result<()> {
        let packed = self.read_bytes()?;
        for chunk in packed.chunks_exact(8) {
            out.push(f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
                chunk[4], chunk[5], chunk[6], chunk[7],
            ]));
        }
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
