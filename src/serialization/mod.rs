//! Model Serialization Module
//!
//! Provides two serialization formats:
//!
//! ## `SafeTensors` Format
//! Industry-standard format compatible with `HuggingFace` ecosystem.
//! ```text
//! [8-byte header: u64 metadata length (little-endian)]
//! [JSON metadata: tensor names, dtypes, shapes, data_offsets]
//! [Raw tensor data: F32 values in little-endian]
//! ```
//!
//! ## APR Format
//! Compact binary format optimized for WASM deployment with JSON metadata.
//! ```text
//! [4-byte magic: "APR1"]
//! [4-byte metadata_len][JSON metadata]
//! [4-byte n_tensors][tensor index][tensor data]
//! [4-byte CRC32]
//! ```
//!
//! Example (SafeTensors):
//! ```rust
//! use aprender::linear_model::LinearRegression;
//! use aprender::primitives::{Matrix, Vector};
//! use aprender::traits::Estimator;
//!
//! # let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
//! # let y = Vector::from_vec(vec![2.0, 3.0, 4.0]);
//! let mut model = LinearRegression::new();
//! model.fit(&x, &y).unwrap();
//!
//! // Save to SafeTensors
//! model.save_safetensors("model.safetensors").unwrap();
//!
//! // Load from SafeTensors
//! let loaded = LinearRegression::load_safetensors("model.safetensors").unwrap();
//! # std::fs::remove_file("model.safetensors").ok();
//! ```
//!
//! Example (APR with metadata):
//! ```rust
//! use aprender::serialization::apr::{AprWriter, AprReader};
//! use serde_json::json;
//!
//! let mut writer = AprWriter::new();
//! writer.set_metadata("model_type", json!("whisper-tiny"));
//! writer.set_metadata("vocab", json!(["hello", "world"]));
//! writer.add_tensor_f32("weights", vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
//!
//! let bytes = writer.to_bytes().unwrap();
//! let reader = AprReader::from_bytes(bytes).unwrap();
//! assert_eq!(reader.get_metadata("model_type").unwrap(), "whisper-tiny");
//! ```

pub mod apr;
pub mod safetensors;

pub use apr::{AprReader, AprWriter};
pub use safetensors::SafeTensorsMetadata;
