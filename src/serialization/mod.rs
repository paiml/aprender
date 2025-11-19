//! Model Serialization Module
//!
//! Implements SafeTensors format for model serialization, compatible with:
//! - realizar inference engine
//! - HuggingFace ecosystem
//! - Ollama (can convert to GGUF)
//! - PyTorch, TensorFlow
//!
//! SafeTensors Format:
//! ```text
//! [8-byte header: u64 metadata length (little-endian)]
//! [JSON metadata: tensor names, dtypes, shapes, data_offsets]
//! [Raw tensor data: F32 values in little-endian]
//! ```
//!
//! Example:
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

pub mod safetensors;

pub use safetensors::SafeTensorsMetadata;
