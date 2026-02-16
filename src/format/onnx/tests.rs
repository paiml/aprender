//! Tests for ONNX reader (GH-238)

pub(crate) use super::*;

// ============================================================================
// Protobuf Wire Format Tests
// ============================================================================

/// Build a minimal ONNX-like protobuf with a single tensor initializer
pub(super) fn build_test_onnx(tensor_name: &str, dims: &[i64], float_data: &[f32]) -> Vec<u8> {
    let mut buf = Vec::new();

    // ModelProto.ir_version = 7 (field 1, varint)
    buf.push(0x08); // field 1, wire type 0
    buf.push(7); // value 7

    // ModelProto.producer_name = "test" (field 2, length-delimited)
    buf.push(0x12); // field 2, wire type 2
    write_string(&mut buf, "test");

    // ModelProto.graph (field 7, length-delimited)
    let graph_bytes = build_graph_proto(tensor_name, dims, float_data);
    buf.push(0x3A); // field 7, wire type 2
    write_varint(&mut buf, graph_bytes.len() as u64);
    buf.extend_from_slice(&graph_bytes);

    buf
}

/// Build a GraphProto with a single initializer tensor
pub(super) fn build_graph_proto(name: &str, dims: &[i64], float_data: &[f32]) -> Vec<u8> {
    let mut buf = Vec::new();

    // GraphProto.initializer (field 5, length-delimited TensorProto)
    let tensor_bytes = build_tensor_proto(name, dims, float_data);
    buf.push(0x2A); // field 5, wire type 2
    write_varint(&mut buf, tensor_bytes.len() as u64);
    buf.extend_from_slice(&tensor_bytes);

    buf
}

/// Build a TensorProto with float data
pub(super) fn build_tensor_proto(name: &str, dims: &[i64], float_data: &[f32]) -> Vec<u8> {
    let mut buf = Vec::new();

    // dims (field 1, packed repeated int64)
    if !dims.is_empty() {
        buf.push(0x0A); // field 1, wire type 2 (packed)
        let mut dims_buf = Vec::new();
        for &d in dims {
            write_varint(&mut dims_buf, d as u64);
        }
        write_varint(&mut buf, dims_buf.len() as u64);
        buf.extend_from_slice(&dims_buf);
    }

    // data_type = FLOAT (1) (field 2, varint)
    buf.push(0x10); // field 2, wire type 0
    buf.push(1); // FLOAT = 1

    // float_data (field 4, packed repeated float)
    if !float_data.is_empty() {
        buf.push(0x22); // field 4, wire type 2 (packed)
        let float_bytes: Vec<u8> = float_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        write_varint(&mut buf, float_bytes.len() as u64);
        buf.extend_from_slice(&float_bytes);
    }

    // name (field 8, string)
    buf.push(0x42); // field 8, wire type 2
    write_string(&mut buf, name);

    buf
}

/// Build a TensorProto with raw_data instead of float_data
pub(super) fn build_tensor_proto_raw(name: &str, dims: &[i64], data_type: i32, raw: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();

    // dims
    if !dims.is_empty() {
        buf.push(0x0A);
        let mut dims_buf = Vec::new();
        for &d in dims {
            write_varint(&mut dims_buf, d as u64);
        }
        write_varint(&mut buf, dims_buf.len() as u64);
        buf.extend_from_slice(&dims_buf);
    }

    // data_type
    buf.push(0x10);
    write_varint(&mut buf, data_type as u64);

    // name
    buf.push(0x42);
    write_string(&mut buf, name);

    // raw_data (field 13, bytes)
    buf.push(0x6A); // field 13, wire type 2
    write_varint(&mut buf, raw.len() as u64);
    buf.extend_from_slice(raw);

    buf
}

/// Build a TensorProto with raw_data in field 9 (PyTorch ONNX format)
pub(super) fn build_tensor_proto_field9(name: &str, dims: &[i64], data_type: i32, raw: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();

    // dims
    if !dims.is_empty() {
        buf.push(0x0A);
        let mut dims_buf = Vec::new();
        for &d in dims {
            write_varint(&mut dims_buf, d as u64);
        }
        write_varint(&mut buf, dims_buf.len() as u64);
        buf.extend_from_slice(&dims_buf);
    }

    // data_type
    buf.push(0x10);
    write_varint(&mut buf, data_type as u64);

    // name
    buf.push(0x42);
    write_string(&mut buf, name);

    // raw_data in field 9 (PyTorch ONNX convention)
    buf.push(0x4A); // field 9, wire type 2
    write_varint(&mut buf, raw.len() as u64);
    buf.extend_from_slice(raw);

    buf
}

pub(super) fn write_varint(buf: &mut Vec<u8>, mut val: u64) {
    loop {
        let byte = (val & 0x7F) as u8;
        val >>= 7;
        if val == 0 {
            buf.push(byte);
            break;
        }
        buf.push(byte | 0x80);
    }
}

pub(super) fn write_string(buf: &mut Vec<u8>, s: &str) {
    write_varint(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

// ============================================================================
// Core Parser Tests
// ============================================================================

#[test]
fn test_parse_single_tensor() {
    let data = build_test_onnx("weight", &[3, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                                                       7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let reader = OnnxReader::from_bytes(&data).expect("parse ONNX");
    assert_eq!(reader.tensors().len(), 1);

    let t = &reader.tensors()[0];
    assert_eq!(t.name, "weight");
    assert_eq!(t.shape, vec![3, 4]);
    assert_eq!(t.data_type, OnnxDataType::Float);

    let f32_data = t.to_f32();
    assert_eq!(f32_data.len(), 12);
    assert!((f32_data[0] - 1.0).abs() < 1e-6);
    assert!((f32_data[11] - 12.0).abs() < 1e-6);
}

#[test]
fn test_parse_metadata() {
    let data = build_test_onnx("w", &[2], &[1.0, 2.0]);
    let reader = OnnxReader::from_bytes(&data).expect("parse ONNX");
    assert_eq!(reader.metadata().ir_version, 7);
    assert_eq!(reader.metadata().producer_name, "test");
}

#[test]
fn test_parse_raw_data() {
    let float_bytes: Vec<u8> = [1.0f32, 2.0, 3.0].iter().flat_map(|f| f.to_le_bytes()).collect();
    let tensor = build_tensor_proto_raw("bias", &[3], 1, &float_bytes);

    // Wrap in graph + model
    let mut graph = Vec::new();
    graph.push(0x2A);
    write_varint(&mut graph, tensor.len() as u64);
    graph.extend_from_slice(&tensor);

    let mut model = Vec::new();
    model.push(0x08);
    model.push(7);
    model.push(0x3A);
    write_varint(&mut model, graph.len() as u64);
    model.extend_from_slice(&graph);

    let reader = OnnxReader::from_bytes(&model).expect("parse");
    assert_eq!(reader.tensors().len(), 1);
    let t = &reader.tensors()[0];
    assert_eq!(t.name, "bias");
    let vals = t.to_f32();
    assert_eq!(vals.len(), 3);
    assert!((vals[2] - 3.0).abs() < 1e-6);
}

#[test]
fn test_to_f32_tensors() {
    let data = build_test_onnx("layer.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let reader = OnnxReader::from_bytes(&data).expect("parse");
    let tensors = reader.to_f32_tensors();
    assert_eq!(tensors.len(), 1);
    let (values, shape) = &tensors["layer.weight"];
    assert_eq!(shape, &[2, 3]);
    assert_eq!(values.len(), 6);
}

#[test]
fn test_empty_onnx() {
    // Minimal valid: just ir_version
    let data = vec![0x08, 7];
    let reader = OnnxReader::from_bytes(&data).expect("parse");
    assert_eq!(reader.tensors().len(), 0);
}

#[test]
fn test_invalid_data() {
    let result = OnnxReader::from_bytes(&[]);
    assert!(result.is_ok()); // Empty data = no tensors

    // Truncated varint
    let result = OnnxReader::from_bytes(&[0x80, 0x80]);
    assert!(result.is_err());
}

// ============================================================================
// OnnxDataType Tests
// ============================================================================

#[test]
fn test_data_type_from_i32() {
    assert_eq!(OnnxDataType::from_i32(1), OnnxDataType::Float);
    assert_eq!(OnnxDataType::from_i32(10), OnnxDataType::Float16);
    assert_eq!(OnnxDataType::from_i32(11), OnnxDataType::Double);
    assert_eq!(OnnxDataType::from_i32(7), OnnxDataType::Int64);
    assert!(matches!(OnnxDataType::from_i32(99), OnnxDataType::Unknown(99)));
}

#[test]
fn test_data_type_element_size() {
    assert_eq!(OnnxDataType::Float.element_size(), 4);
    assert_eq!(OnnxDataType::Double.element_size(), 8);
    assert_eq!(OnnxDataType::Float16.element_size(), 2);
    assert_eq!(OnnxDataType::Int8.element_size(), 1);
    assert_eq!(OnnxDataType::Unknown(99).element_size(), 0);
}

// ============================================================================
// F16 Conversion Tests
// ============================================================================

#[test]
fn test_f16_to_f32_zero() {
    assert_eq!(f16_to_f32(0x0000), 0.0f32);
    assert_eq!(f16_to_f32(0x8000), -0.0f32);
}

#[test]
fn test_f16_to_f32_one() {
    // f16 1.0 = 0x3C00
    let val = f16_to_f32(0x3C00);
    assert!((val - 1.0).abs() < 1e-6);
}

#[test]
fn test_f16_to_f32_negative() {
    // f16 -1.0 = 0xBC00
    let val = f16_to_f32(0xBC00);
    assert!((val - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_f16_to_f32_inf() {
    let val = f16_to_f32(0x7C00);
    assert!(val.is_infinite() && val > 0.0);
}

#[test]
fn test_f16_to_f32_nan() {
    let val = f16_to_f32(0x7C01);
    assert!(val.is_nan());
}

// ============================================================================
// File Detection Tests
// ============================================================================

#[test]
fn test_is_onnx_file_by_extension() {
    assert!(is_onnx_file(Path::new("model.onnx")));
    assert!(!is_onnx_file(Path::new("model.safetensors")));
}

#[test]
fn test_is_nemo_file() {
    assert!(is_nemo_file(Path::new("model.nemo")));
    assert!(!is_nemo_file(Path::new("model.onnx")));
}

// ============================================================================
// Multiple Tensor Tests
// ============================================================================

#[test]
fn test_multiple_tensors() {
    // Build a graph with 2 initializer tensors
    let t1 = build_tensor_proto("weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let t2 = build_tensor_proto("bias", &[3], &[0.1, 0.2, 0.3]);

    let mut graph = Vec::new();
    graph.push(0x2A);
    write_varint(&mut graph, t1.len() as u64);
    graph.extend_from_slice(&t1);
    graph.push(0x2A);
    write_varint(&mut graph, t2.len() as u64);
    graph.extend_from_slice(&t2);

    let mut model = Vec::new();
    model.push(0x08);
    model.push(7);
    model.push(0x3A);
    write_varint(&mut model, graph.len() as u64);
    model.extend_from_slice(&graph);

    let reader = OnnxReader::from_bytes(&model).expect("parse");
    assert_eq!(reader.tensors().len(), 2);
    assert_eq!(reader.tensors()[0].name, "weight");
    assert_eq!(reader.tensors()[1].name, "bias");
}

// ============================================================================
// Conversion Tests
// ============================================================================

#[test]
fn test_int8_to_f32() {
    let raw: Vec<u8> = vec![0xFF, 0x01, 0x7F]; // -1, 1, 127 as i8
    let tensor = OnnxTensor {
        name: "t".to_string(),
        shape: vec![3],
        data_type: OnnxDataType::Int8,
        raw_data: raw,
    };
    let vals = tensor.to_f32();
    assert_eq!(vals.len(), 3);
    assert!((vals[0] - (-1.0)).abs() < 1e-6);
    assert!((vals[1] - 1.0).abs() < 1e-6);
    assert!((vals[2] - 127.0).abs() < 1e-6);
}

#[test]
fn test_double_to_f32() {
    let raw: Vec<u8> = 3.14f64.to_le_bytes().to_vec();
    let tensor = OnnxTensor {
        name: "t".to_string(),
        shape: vec![1],
        data_type: OnnxDataType::Double,
        raw_data: raw,
    };
    let vals = tensor.to_f32();
    assert_eq!(vals.len(), 1);
    assert!((vals[0] - 3.14).abs() < 0.001);
}

// ============================================================================
// Field 9 raw_data Tests (PyTorch ONNX convention)
// ============================================================================

#[test]
fn test_field9_raw_data_pytorch_convention() {
    // PyTorch ONNX exporter stores raw_data in field 9 instead of field 13
    let float_bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let tensor = build_tensor_proto_field9("weight", &[2, 2], 1, &float_bytes);

    // Wrap in graph + model
    let mut graph = Vec::new();
    graph.push(0x2A);
    write_varint(&mut graph, tensor.len() as u64);
    graph.extend_from_slice(&tensor);

    let mut model = Vec::new();
    model.push(0x08);
    model.push(7);
    model.push(0x3A);
    write_varint(&mut model, graph.len() as u64);
    model.extend_from_slice(&graph);

    let reader = OnnxReader::from_bytes(&model).expect("parse field 9 ONNX");
    assert_eq!(reader.tensors().len(), 1);
    let t = &reader.tensors()[0];
    assert_eq!(t.name, "weight");
    assert_eq!(t.shape, vec![2, 2]);
    assert_eq!(t.data_type, OnnxDataType::Float);
    assert_eq!(t.raw_data.len(), 16);

    let vals = t.to_f32();
    assert_eq!(vals.len(), 4);
    assert!((vals[0] - 1.0).abs() < 1e-6);
    assert!((vals[3] - 4.0).abs() < 1e-6);
}

#[test]
fn test_field9_and_field13_both_handled() {
    // Verify field 13 still works (standard ONNX)
    let float_bytes: Vec<u8> = [5.0f32, 6.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let tensor13 = build_tensor_proto_raw("bias13", &[2], 1, &float_bytes);
    let tensor9 = build_tensor_proto_field9("bias9", &[2], 1, &float_bytes);

    let mut graph = Vec::new();
    // Add field 13 tensor
    graph.push(0x2A);
    write_varint(&mut graph, tensor13.len() as u64);
    graph.extend_from_slice(&tensor13);
    // Add field 9 tensor
    graph.push(0x2A);
    write_varint(&mut graph, tensor9.len() as u64);
    graph.extend_from_slice(&tensor9);

    let mut model = Vec::new();
    model.push(0x08);
    model.push(7);
    model.push(0x3A);
    write_varint(&mut model, graph.len() as u64);
    model.extend_from_slice(&graph);

    let reader = OnnxReader::from_bytes(&model).expect("parse mixed ONNX");
    assert_eq!(reader.tensors().len(), 2);
    assert_eq!(reader.tensors()[0].name, "bias13");
    assert_eq!(reader.tensors()[1].name, "bias9");
    // Both should have data
    assert_eq!(reader.tensors()[0].to_f32().len(), 2);
    assert_eq!(reader.tensors()[1].to_f32().len(), 2);
}

// ============================================================================
// Real ONNX File Tests (GH-238)
// ============================================================================

#[test]
fn test_real_onnx_file_debug() {
    // MiniLM-L6-v2 ONNX model from fastembed cache (optional)
    let path = std::path::Path::new(
        "/home/noah/src/trueno-rag/.fastembed_cache/models--Qdrant--all-MiniLM-L6-v2-onnx/snapshots/5f1b8cd78bc4fb444dd171e59b18f3a3af89a079/model.onnx"
    );
    if !path.exists() {
        return; // Skip if model not available
    }

    let reader = OnnxReader::from_file(path).expect("Failed to parse ONNX");
    assert_eq!(reader.tensors().len(), 101);
    assert_eq!(reader.metadata().ir_version, 6);
    assert_eq!(reader.metadata().producer_name, "pytorch");

    // Verify first tensor (word embeddings)
    let t0 = &reader.tensors()[0];
    assert_eq!(t0.name, "embeddings.word_embeddings.weight");
    assert_eq!(t0.shape, vec![30522, 384]);
    assert_eq!(t0.data_type, OnnxDataType::Float);
    assert_eq!(t0.raw_data.len(), 30522 * 384 * 4);

    // Verify to_f32_tensors produces all tensors
    let tensors = reader.to_f32_tensors();
    assert_eq!(tensors.len(), 101);
}
