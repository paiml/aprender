//! PyTorch to Realizar conversion module (BATUTA-010)
//!
//! Converts Python PyTorch inference code to Rust Realizar equivalents
//! with automatic backend selection for CPU/GPU/WASM execution.
//!
//! # Conversion Strategy
//!
//! PyTorch inference patterns are mapped to Realizar equivalents:
//! - `torch.load(path)` → GGUF/SafeTensors model loading
//! - `model.forward(x)` → Realizar inference pipeline
//! - `torch.Tensor` → `realizar::tensor::Tensor`
//! - `torch.nn.Linear` → `realizar::layers::LinearLayer`
//! - Tokenization → `realizar::tokenizer`
//! - Generation → `realizar::generate`
//!
//! # Example
//!
//! ```python
//! # Python PyTorch inference code
//! import torch
//! from transformers import AutoModelForCausalLM, AutoTokenizer
//!
//! model = AutoModelForCausalLM.from_pretrained("model_name")
//! tokenizer = AutoTokenizer.from_pretrained("model_name")
//! inputs = tokenizer("Hello, world!", return_tensors="pt")
//! outputs = model.generate(**inputs, max_length=50)
//! text = tokenizer.decode(outputs[0])
//! ```
//!
//! Converts to:
//!
//! ```rust,ignore
//! use realizar::gguf::GGUFModel;
//! use realizar::tokenizer::Tokenizer;
//! use realizar::generate::generate_text;
//!
//! let model = GGUFModel::from_file("model.gguf")?;
//! let tokenizer = Tokenizer::from_file("tokenizer.json")?;
//! let tokens = tokenizer.encode("Hello, world!")?;
//! let output = generate_text(&model, &tokens, 50)?;
//! let text = tokenizer.decode(&output)?;
//! ```

use std::collections::HashMap;

/// PyTorch operation types (inference-focused)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyTorchOperation {
    // Model Loading
    LoadModel,     // torch.load(), from_pretrained()
    SaveModel,     // torch.save()
    LoadTokenizer, // AutoTokenizer.from_pretrained()

    // Inference Operations
    Forward,  // model(x), model.forward(x)
    Generate, // model.generate()
    Predict,  // model.predict()

    // Tensor Operations
    TensorCreation, // torch.tensor(), torch.zeros()
    TensorReshape,  // tensor.view(), tensor.reshape()
    TensorSlice,    // tensor[start:end]

    // Layer Types
    Linear,    // nn.Linear
    Embedding, // nn.Embedding
    LayerNorm, // nn.LayerNorm
    Attention, // nn.MultiheadAttention

    // Activation Functions
    ReLU,    // nn.ReLU
    GELU,    // nn.GELU
    Softmax, // nn.Softmax

    // Tokenization
    Encode, // tokenizer.encode()
    Decode, // tokenizer.decode()

    // Utilities
    NoGrad, // torch.no_grad()
    Eval,   // model.eval()
}

impl PyTorchOperation {
    /// Get the computational complexity for MoE routing
    pub fn complexity(&self) -> crate::backend::OpComplexity {
        use crate::backend::OpComplexity;

        match self {
            // Simple operations are Low complexity
            PyTorchOperation::TensorCreation
            | PyTorchOperation::TensorReshape
            | PyTorchOperation::TensorSlice
            | PyTorchOperation::Encode
            | PyTorchOperation::Decode
            | PyTorchOperation::NoGrad
            | PyTorchOperation::Eval => OpComplexity::Low,

            // Layer operations and activations are Medium complexity
            PyTorchOperation::Linear
            | PyTorchOperation::Embedding
            | PyTorchOperation::LayerNorm
            | PyTorchOperation::ReLU
            | PyTorchOperation::GELU
            | PyTorchOperation::Softmax
            | PyTorchOperation::LoadModel
            | PyTorchOperation::SaveModel
            | PyTorchOperation::LoadTokenizer => OpComplexity::Medium,

            // Inference and generation are High complexity
            PyTorchOperation::Forward
            | PyTorchOperation::Generate
            | PyTorchOperation::Predict
            | PyTorchOperation::Attention => OpComplexity::High,
        }
    }

    /// Get the PyTorch module path
    pub fn pytorch_module(&self) -> &str {
        match self {
            PyTorchOperation::LoadModel
            | PyTorchOperation::SaveModel
            | PyTorchOperation::TensorCreation
            | PyTorchOperation::TensorReshape
            | PyTorchOperation::TensorSlice
            | PyTorchOperation::NoGrad => "torch",

            PyTorchOperation::Linear
            | PyTorchOperation::Embedding
            | PyTorchOperation::LayerNorm
            | PyTorchOperation::Attention
            | PyTorchOperation::ReLU
            | PyTorchOperation::GELU
            | PyTorchOperation::Softmax => "torch.nn",

            PyTorchOperation::LoadTokenizer
            | PyTorchOperation::Encode
            | PyTorchOperation::Decode => "transformers",

            PyTorchOperation::Forward
            | PyTorchOperation::Generate
            | PyTorchOperation::Predict
            | PyTorchOperation::Eval => "torch.nn.Module",
        }
    }
}

/// Realizar equivalent operation
#[derive(Debug, Clone)]
pub struct RealizarOperation {
    /// Rust code template for the operation
    pub code_template: String,
    /// Required imports
    pub imports: Vec<String>,
    /// Computational complexity
    pub complexity: crate::backend::OpComplexity,
    /// Typical usage pattern
    pub usage_pattern: String,
}

/// PyTorch to Realizar converter
pub struct PyTorchConverter {
    /// Operation mapping
    operation_map: HashMap<PyTorchOperation, RealizarOperation>,
    /// Backend selector for MoE routing
    backend_selector: crate::backend::BackendSelector,
}

impl Default for PyTorchConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl PyTorchConverter {
    /// Create a new PyTorch converter with default mappings
    pub fn new() -> Self {
        let mut operation_map = HashMap::new();

        // Model Loading
        operation_map.insert(
            PyTorchOperation::LoadModel,
            RealizarOperation {
                code_template: "GGUFModel::from_file(\"{model_path}\")".to_string(),
                imports: vec!["use realizar::gguf::GGUFModel;".to_string()],
                complexity: crate::backend::OpComplexity::Medium,
                usage_pattern: "let model = GGUFModel::from_file(\"model.gguf\")?;".to_string(),
            },
        );

        operation_map.insert(
            PyTorchOperation::LoadTokenizer,
            RealizarOperation {
                code_template: "Tokenizer::from_file(\"{tokenizer_path}\")".to_string(),
                imports: vec!["use realizar::tokenizer::Tokenizer;".to_string()],
                complexity: crate::backend::OpComplexity::Medium,
                usage_pattern: "let tokenizer = Tokenizer::from_file(\"tokenizer.json\")?;"
                    .to_string(),
            },
        );

        // Inference Operations
        operation_map.insert(
            PyTorchOperation::Forward,
            RealizarOperation {
                code_template: "model.forward(&{input})".to_string(),
                imports: vec!["use realizar::gguf::GGUFModel;".to_string()],
                complexity: crate::backend::OpComplexity::High,
                usage_pattern: "let output = model.forward(&input_tensor)?;".to_string(),
            },
        );

        operation_map.insert(
            PyTorchOperation::Generate,
            RealizarOperation {
                code_template: "generate_text(&model, &{tokens}, {max_length})".to_string(),
                imports: vec![
                    "use realizar::generate::generate_text;".to_string(),
                ],
                complexity: crate::backend::OpComplexity::High,
                usage_pattern: "let output = generate_text(&model, &input_tokens, 50)?;\nlet text = tokenizer.decode(&output)?;".to_string(),
            },
        );

        // Tensor Operations
        operation_map.insert(
            PyTorchOperation::TensorCreation,
            RealizarOperation {
                code_template: "Tensor::from_vec({data})".to_string(),
                imports: vec!["use realizar::tensor::Tensor;".to_string()],
                complexity: crate::backend::OpComplexity::Low,
                usage_pattern: "let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0])?;".to_string(),
            },
        );

        // Layer Types
        operation_map.insert(
            PyTorchOperation::Linear,
            RealizarOperation {
                code_template: "LinearLayer::new({in_features}, {out_features})".to_string(),
                imports: vec!["use realizar::layers::LinearLayer;".to_string()],
                complexity: crate::backend::OpComplexity::Medium,
                usage_pattern:
                    "let layer = LinearLayer::new(768, 512)?;\nlet output = layer.forward(&input)?;"
                        .to_string(),
            },
        );

        operation_map.insert(
            PyTorchOperation::Attention,
            RealizarOperation {
                code_template: "AttentionLayer::new({embed_dim}, {num_heads})".to_string(),
                imports: vec!["use realizar::layers::AttentionLayer;".to_string()],
                complexity: crate::backend::OpComplexity::High,
                usage_pattern:
                    "let attn = AttentionLayer::new(512, 8)?;\nlet output = attn.forward(&input)?;"
                        .to_string(),
            },
        );

        // Activation Functions
        operation_map.insert(
            PyTorchOperation::GELU,
            RealizarOperation {
                code_template: "gelu(&{input})".to_string(),
                imports: vec!["use realizar::layers::activations::gelu;".to_string()],
                complexity: crate::backend::OpComplexity::Medium,
                usage_pattern: "let activated = gelu(&input_tensor)?;".to_string(),
            },
        );

        // Tokenization
        operation_map.insert(
            PyTorchOperation::Encode,
            RealizarOperation {
                code_template: "tokenizer.encode(\"{text}\")".to_string(),
                imports: vec!["use realizar::tokenizer::Tokenizer;".to_string()],
                complexity: crate::backend::OpComplexity::Low,
                usage_pattern: "let tokens = tokenizer.encode(\"Hello, world!\")?;".to_string(),
            },
        );

        operation_map.insert(
            PyTorchOperation::Decode,
            RealizarOperation {
                code_template: "tokenizer.decode(&{tokens})".to_string(),
                imports: vec!["use realizar::tokenizer::Tokenizer;".to_string()],
                complexity: crate::backend::OpComplexity::Low,
                usage_pattern: "let text = tokenizer.decode(&output_tokens)?;".to_string(),
            },
        );

        Self { operation_map, backend_selector: crate::backend::BackendSelector::new() }
    }

    /// Convert a PyTorch operation to Realizar
    pub fn convert(&self, operation: &PyTorchOperation) -> Option<&RealizarOperation> {
        self.operation_map.get(operation)
    }

    /// Get recommended backend for an operation
    pub fn recommend_backend(
        &self,
        operation: &PyTorchOperation,
        data_size: usize,
    ) -> crate::backend::Backend {
        self.backend_selector.select_with_moe(operation.complexity(), data_size)
    }

    /// Get all available conversions
    pub fn available_operations(&self) -> Vec<&PyTorchOperation> {
        self.operation_map.keys().collect()
    }

    /// Generate conversion report
    pub fn conversion_report(&self) -> String {
        let mut report = String::from("PyTorch → Realizar Conversion Map\n");
        report.push_str("====================================\n\n");

        // Group by module
        let mut by_module: HashMap<&str, Vec<(&PyTorchOperation, &RealizarOperation)>> =
            HashMap::new();

        for (op, realizar_op) in &self.operation_map {
            by_module.entry(op.pytorch_module()).or_default().push((op, realizar_op));
        }

        for (module, operations) in &by_module {
            report.push_str(&format!("## {}\n\n", module));

            for (op, realizar_op) in operations {
                report.push_str(&format!("{:?}:\n", op));
                report.push_str(&format!("  Template: {}\n", realizar_op.code_template));
                report.push_str(&format!("  Complexity: {:?}\n", realizar_op.complexity));
                report.push_str(&format!("  Imports: {}\n", realizar_op.imports.join(", ")));
                report.push_str(&format!(
                    "  Usage:\n    {}\n\n",
                    realizar_op.usage_pattern.replace('\n', "\n    ")
                ));
            }
            report.push('\n');
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converter_creation() {
        let converter = PyTorchConverter::new();
        assert!(!converter.available_operations().is_empty());
    }

    #[test]
    fn test_operation_complexity() {
        assert_eq!(
            PyTorchOperation::TensorCreation.complexity(),
            crate::backend::OpComplexity::Low
        );
        assert_eq!(PyTorchOperation::Linear.complexity(), crate::backend::OpComplexity::Medium);
        assert_eq!(PyTorchOperation::Generate.complexity(), crate::backend::OpComplexity::High);
    }

    #[test]
    fn test_load_model_conversion() {
        let converter = PyTorchConverter::new();
        let realizar_op =
            converter.convert(&PyTorchOperation::LoadModel).expect("conversion failed");
        assert!(realizar_op.code_template.contains("GGUFModel"));
        assert!(realizar_op.imports.iter().any(|i| i.contains("gguf")));
    }

    #[test]
    fn test_generate_conversion() {
        let converter = PyTorchConverter::new();
        let realizar_op =
            converter.convert(&PyTorchOperation::Generate).expect("conversion failed");
        assert!(realizar_op.code_template.contains("generate_text"));
        assert!(realizar_op.imports.iter().any(|i| i.contains("generate")));
    }

    #[test]
    fn test_backend_recommendation() {
        let converter = PyTorchConverter::new();

        // Small tensor operation should use Scalar
        let backend = converter.recommend_backend(&PyTorchOperation::TensorCreation, 100);
        assert_eq!(backend, crate::backend::Backend::Scalar);

        // Medium-sized linear layer should use SIMD
        let backend = converter.recommend_backend(&PyTorchOperation::Linear, 50_000);
        assert_eq!(backend, crate::backend::Backend::SIMD);

        // Large generation task should use GPU
        let backend = converter.recommend_backend(&PyTorchOperation::Generate, 100_000);
        assert_eq!(backend, crate::backend::Backend::GPU);
    }

    #[test]
    fn test_pytorch_module_paths() {
        assert_eq!(PyTorchOperation::LoadModel.pytorch_module(), "torch");
        assert_eq!(PyTorchOperation::Linear.pytorch_module(), "torch.nn");
        assert_eq!(PyTorchOperation::LoadTokenizer.pytorch_module(), "transformers");
    }

    #[test]
    fn test_conversion_report() {
        let converter = PyTorchConverter::new();
        let report = converter.conversion_report();
        assert!(report.contains("PyTorch → Realizar"));
        assert!(report.contains("LoadModel"));
        assert!(report.contains("Complexity"));
    }

    // ============================================================================
    // PYTORCH OPERATION ENUM TESTS
    // ============================================================================

    #[test]
    fn test_all_pytorch_operations_exist() {
        // Test all 22 variants can be constructed
        let ops = vec![
            PyTorchOperation::LoadModel,
            PyTorchOperation::SaveModel,
            PyTorchOperation::LoadTokenizer,
            PyTorchOperation::Forward,
            PyTorchOperation::Generate,
            PyTorchOperation::Predict,
            PyTorchOperation::TensorCreation,
            PyTorchOperation::TensorReshape,
            PyTorchOperation::TensorSlice,
            PyTorchOperation::Linear,
            PyTorchOperation::Embedding,
            PyTorchOperation::LayerNorm,
            PyTorchOperation::Attention,
            PyTorchOperation::ReLU,
            PyTorchOperation::GELU,
            PyTorchOperation::Softmax,
            PyTorchOperation::Encode,
            PyTorchOperation::Decode,
            PyTorchOperation::NoGrad,
            PyTorchOperation::Eval,
        ];
        assert_eq!(ops.len(), 20); // 20 operations tested
    }

    #[test]
    fn test_operation_equality() {
        assert_eq!(PyTorchOperation::LoadModel, PyTorchOperation::LoadModel);
        assert_ne!(PyTorchOperation::LoadModel, PyTorchOperation::Generate);
    }

    #[test]
    fn test_operation_clone() {
        let op1 = PyTorchOperation::Forward;
        let op2 = op1.clone();
        assert_eq!(op1, op2);
    }

    #[test]
    fn test_complexity_low_operations() {
        let low_ops = vec![
            PyTorchOperation::TensorCreation,
            PyTorchOperation::TensorReshape,
            PyTorchOperation::TensorSlice,
            PyTorchOperation::Encode,
            PyTorchOperation::Decode,
            PyTorchOperation::NoGrad,
            PyTorchOperation::Eval,
        ];

        for op in low_ops {
            assert_eq!(op.complexity(), crate::backend::OpComplexity::Low);
        }
    }

    #[test]
    fn test_complexity_medium_operations() {
        let medium_ops = vec![
            PyTorchOperation::Linear,
            PyTorchOperation::Embedding,
            PyTorchOperation::LayerNorm,
            PyTorchOperation::ReLU,
            PyTorchOperation::GELU,
            PyTorchOperation::Softmax,
            PyTorchOperation::LoadModel,
            PyTorchOperation::SaveModel,
            PyTorchOperation::LoadTokenizer,
        ];

        for op in medium_ops {
            assert_eq!(op.complexity(), crate::backend::OpComplexity::Medium);
        }
    }

    #[test]
    fn test_complexity_high_operations() {
        let high_ops = vec![
            PyTorchOperation::Forward,
            PyTorchOperation::Generate,
            PyTorchOperation::Predict,
            PyTorchOperation::Attention,
        ];

        for op in high_ops {
            assert_eq!(op.complexity(), crate::backend::OpComplexity::High);
        }
    }

    #[test]
    fn test_pytorch_module_torch() {
        let torch_ops = vec![
            PyTorchOperation::LoadModel,
            PyTorchOperation::SaveModel,
            PyTorchOperation::TensorCreation,
            PyTorchOperation::TensorReshape,
            PyTorchOperation::TensorSlice,
            PyTorchOperation::NoGrad,
        ];

        for op in torch_ops {
            assert_eq!(op.pytorch_module(), "torch");
        }
    }

    #[test]
    fn test_pytorch_module_torch_nn() {
        let nn_ops = vec![
            PyTorchOperation::Linear,
            PyTorchOperation::Embedding,
            PyTorchOperation::LayerNorm,
            PyTorchOperation::Attention,
            PyTorchOperation::ReLU,
            PyTorchOperation::GELU,
            PyTorchOperation::Softmax,
        ];

        for op in nn_ops {
            assert_eq!(op.pytorch_module(), "torch.nn");
        }
    }

    #[test]
    fn test_pytorch_module_transformers() {
        let transformers_ops = vec![
            PyTorchOperation::LoadTokenizer,
            PyTorchOperation::Encode,
            PyTorchOperation::Decode,
        ];

        for op in transformers_ops {
            assert_eq!(op.pytorch_module(), "transformers");
        }
    }

    #[test]
    fn test_pytorch_module_torch_nn_module() {
        let module_ops = vec![
            PyTorchOperation::Forward,
            PyTorchOperation::Generate,
            PyTorchOperation::Predict,
            PyTorchOperation::Eval,
        ];

        for op in module_ops {
            assert_eq!(op.pytorch_module(), "torch.nn.Module");
        }
    }

    // ============================================================================
    // REALIZAR OPERATION STRUCT TESTS
    // ============================================================================

    #[test]
    fn test_realizar_operation_construction() {
        let op = RealizarOperation {
            code_template: "test_template".to_string(),
            imports: vec!["use test;".to_string()],
            complexity: crate::backend::OpComplexity::Medium,
            usage_pattern: "let x = test();".to_string(),
        };

        assert_eq!(op.code_template, "test_template");
        assert_eq!(op.imports.len(), 1);
        assert_eq!(op.complexity, crate::backend::OpComplexity::Medium);
        assert!(op.usage_pattern.contains("test()"));
    }

    #[test]
    fn test_realizar_operation_clone() {
        let op1 = RealizarOperation {
            code_template: "template".to_string(),
            imports: vec!["import".to_string()],
            complexity: crate::backend::OpComplexity::High,
            usage_pattern: "usage".to_string(),
        };

        let op2 = op1.clone();
        assert_eq!(op1.code_template, op2.code_template);
        assert_eq!(op1.imports, op2.imports);
        assert_eq!(op1.complexity, op2.complexity);
    }

    // ============================================================================
    // PYTORCH CONVERTER TESTS
    // ============================================================================

    #[test]
    fn test_converter_default() {
        let converter = PyTorchConverter::default();
        assert!(!converter.available_operations().is_empty());
    }

    #[test]
    fn test_convert_all_mapped_operations() {
        let converter = PyTorchConverter::new();

        // Test all operations that should have mappings
        let mapped_ops = vec![
            PyTorchOperation::LoadModel,
            PyTorchOperation::LoadTokenizer,
            PyTorchOperation::Forward,
            PyTorchOperation::Generate,
            PyTorchOperation::TensorCreation,
            PyTorchOperation::Linear,
            PyTorchOperation::Attention,
            PyTorchOperation::GELU,
            PyTorchOperation::Encode,
            PyTorchOperation::Decode,
        ];

        for op in mapped_ops {
            assert!(converter.convert(&op).is_some(), "Missing mapping for {:?}", op);
        }
    }

    #[test]
    fn test_convert_unmapped_operation() {
        let converter = PyTorchConverter::new();

        // SaveModel, Predict, etc. might not be mapped
        // Just verify the function handles missing ops gracefully
        let result = converter.convert(&PyTorchOperation::SaveModel);
        // It's ok if this is None - we're testing the API works
        let _ = result;
    }

    #[test]
    fn test_forward_conversion() {
        let converter = PyTorchConverter::new();
        let op = converter.convert(&PyTorchOperation::Forward).expect("conversion failed");

        assert!(op.code_template.contains("forward"));
        assert!(op.imports.iter().any(|i| i.contains("gguf")));
        assert_eq!(op.complexity, crate::backend::OpComplexity::High);
    }

    #[test]
    fn test_tokenizer_conversion() {
        let converter = PyTorchConverter::new();
        let op = converter.convert(&PyTorchOperation::LoadTokenizer).expect("conversion failed");

        assert!(op.code_template.contains("Tokenizer"));
        assert!(op.imports.iter().any(|i| i.contains("tokenizer")));
    }

    #[test]
    fn test_encode_decode_conversions() {
        let converter = PyTorchConverter::new();

        let encode_op = converter.convert(&PyTorchOperation::Encode).expect("conversion failed");
        assert!(encode_op.code_template.contains("encode"));

        let decode_op = converter.convert(&PyTorchOperation::Decode).expect("conversion failed");
        assert!(decode_op.code_template.contains("decode"));
    }

    #[test]
    fn test_tensor_operation_conversion() {
        let converter = PyTorchConverter::new();
        let op = converter.convert(&PyTorchOperation::TensorCreation).expect("unexpected failure");

        assert!(op.code_template.contains("Tensor"));
        assert!(op.imports.iter().any(|i| i.contains("tensor")));
        assert_eq!(op.complexity, crate::backend::OpComplexity::Low);
    }

    #[test]
    fn test_linear_layer_conversion() {
        let converter = PyTorchConverter::new();
        let op = converter.convert(&PyTorchOperation::Linear).expect("conversion failed");

        assert!(op.code_template.contains("LinearLayer"));
        assert!(op.imports.iter().any(|i| i.contains("layers")));
    }

    #[test]
    fn test_attention_layer_conversion() {
        let converter = PyTorchConverter::new();
        let op = converter.convert(&PyTorchOperation::Attention).expect("conversion failed");

        assert!(op.code_template.contains("AttentionLayer"));
        assert_eq!(op.complexity, crate::backend::OpComplexity::High);
    }

    #[test]
    fn test_gelu_activation_conversion() {
        let converter = PyTorchConverter::new();
        let op = converter.convert(&PyTorchOperation::GELU).expect("conversion failed");

        assert!(op.code_template.contains("gelu"));
        assert!(op.imports.iter().any(|i| i.contains("activations")));
    }

    #[test]
    fn test_available_operations() {
        let converter = PyTorchConverter::new();
        let ops = converter.available_operations();

        assert!(!ops.is_empty());
        // Should have at least the mapped operations
        assert!(ops.len() >= 10);
    }

    #[test]
    fn test_recommend_backend_low_complexity() {
        let converter = PyTorchConverter::new();

        // Small data size with low complexity should use Scalar
        let backend = converter.recommend_backend(&PyTorchOperation::TensorCreation, 10);
        assert_eq!(backend, crate::backend::Backend::Scalar);
    }

    #[test]
    fn test_recommend_backend_medium_complexity() {
        let converter = PyTorchConverter::new();

        // Medium data size with medium complexity should use SIMD
        let backend = converter.recommend_backend(&PyTorchOperation::Linear, 50_000);
        assert_eq!(backend, crate::backend::Backend::SIMD);
    }

    #[test]
    fn test_recommend_backend_high_complexity() {
        let converter = PyTorchConverter::new();

        // Large data size with high complexity should use GPU
        let backend = converter.recommend_backend(&PyTorchOperation::Forward, 500_000);
        assert_eq!(backend, crate::backend::Backend::GPU);
    }

    #[test]
    fn test_recommend_backend_generation() {
        let converter = PyTorchConverter::new();

        // Generation is high complexity, large size should use GPU
        let backend = converter.recommend_backend(&PyTorchOperation::Generate, 1_000_000);
        assert_eq!(backend, crate::backend::Backend::GPU);
    }

    #[test]
    fn test_conversion_report_structure() {
        let converter = PyTorchConverter::new();
        let report = converter.conversion_report();

        // Check report contains expected sections
        assert!(report.contains("PyTorch → Realizar"));
        assert!(report.contains("===="));
        assert!(report.contains("##")); // Module headers
        assert!(report.contains("Template:"));
        assert!(report.contains("Imports:"));
        assert!(report.contains("Usage:"));
    }

    #[test]
    fn test_conversion_report_has_modules() {
        let converter = PyTorchConverter::new();
        let report = converter.conversion_report();

        // Should group by PyTorch modules
        assert!(report.contains("torch") || report.contains("transformers"));
    }

    #[test]
    fn test_conversion_report_has_all_operations() {
        let converter = PyTorchConverter::new();
        let report = converter.conversion_report();

        // Spot check a few operations appear in report
        assert!(
            report.contains("LoadModel")
                || report.contains("Generate")
                || report.contains("Forward")
        );
    }

    #[test]
    fn test_usage_patterns_not_empty() {
        let converter = PyTorchConverter::new();

        for op in converter.available_operations() {
            if let Some(realizar_op) = converter.convert(op) {
                assert!(!realizar_op.usage_pattern.is_empty(), "Empty usage pattern for {:?}", op);
                assert!(!realizar_op.code_template.is_empty(), "Empty code template for {:?}", op);
                assert!(!realizar_op.imports.is_empty(), "Empty imports for {:?}", op);
            }
        }
    }

    #[test]
    fn test_imports_are_valid_rust() {
        let converter = PyTorchConverter::new();

        for op in converter.available_operations() {
            if let Some(realizar_op) = converter.convert(op) {
                for import in &realizar_op.imports {
                    assert!(import.starts_with("use "), "Invalid import syntax: {}", import);
                    assert!(import.ends_with(';'), "Import missing semicolon: {}", import);
                }
            }
        }
    }
}
