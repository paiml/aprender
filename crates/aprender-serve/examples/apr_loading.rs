#![allow(unexpected_cfgs)]

//! APR Format Loading Example
//!
//! Demonstrates loading and using Aprender's native .apr format,
//! the PRIMARY inference format for the sovereign AI stack.
//!
//! NOTE: This example is disabled until full APR model type support is implemented.

// Disabled: APR model types not yet exported from apr module
fn main() {
    println!("APR Loading example is disabled - APR model types not yet implemented");
}

#[cfg(feature = "apr_model_types")] // Disable until AprModelType enum is implemented
mod disabled {
    //!
    //! ## Format Priority (Sovereign Stack)
    //!
    //! 1. `.apr` (PRIMARY) - Aprender native format with CRC32, Ed25519, AES-256-GCM
    //! 2. `.gguf` (FALLBACK) - GGUF quantized models (llama.cpp/Ollama)
    //! 3. `.safetensors` (FALLBACK) - HuggingFace format
    //!
    //! ## Usage
    //!
    //! ```bash
    //! cargo run --example apr_loading
    //! ```
    //!
    //! ## Integration
    //!
    //! For end-to-end evaluation with .apr models, see:
    //! - [`single-shot-eval`](https://github.com/paiml/single-shot-eval) - SLM Pareto Frontier Evaluation

    use realizar::apr::{
        detect_format, AprHeader, AprMetadata, AprModelType, ModelWeights, HEADER_SIZE, MAGIC,
    };

    fn main() {
        println!("=== APR Format Loading Demo ===\n");

        // 1. Show format constants
        println!("APR Format Specification:");
        println!("  Magic: {:?} (\"APRN\")", MAGIC);
        println!("  Header Size: {} bytes", HEADER_SIZE);
        println!();

        // 2. Demonstrate model types
        println!("Supported Model Types:");
        let model_types = [
            (AprModelType::LinearRegression, "Linear Regression"),
            (AprModelType::LogisticRegression, "Logistic Regression"),
            (AprModelType::DecisionTree, "Decision Tree"),
            (AprModelType::RandomForest, "Random Forest"),
            (AprModelType::GradientBoosting, "Gradient Boosting"),
            (AprModelType::KMeans, "K-Means Clustering"),
            (AprModelType::Pca, "Principal Component Analysis"),
            (AprModelType::NaiveBayes, "Naive Bayes"),
            (AprModelType::Knn, "K-Nearest Neighbors"),
            (AprModelType::Svm, "Support Vector Machine"),
            (AprModelType::NgramLm, "N-gram Language Model"),
            (AprModelType::Tfidf, "TF-IDF Vectorizer"),
            (
                AprModelType::NeuralSequential,
                "Neural Network (Sequential)",
            ),
            (AprModelType::MixtureOfExperts, "Mixture of Experts"),
        ];

        for (model_type, name) in &model_types {
            println!("  0x{:04X} - {}", model_type.as_u16(), name);
        }
        println!();

        // 3. Create a test model in-memory
        println!("Creating test 2-layer neural network...\n");

        let weights = ModelWeights {
            weights: vec![
                vec![0.5, -0.3, 0.8, -0.1, 0.2, -0.4], // 3x2 input layer
                vec![0.7, -0.5],                       // 2x1 output layer
            ],
            biases: vec![vec![0.1, 0.1], vec![0.0]],
            dimensions: vec![3, 2, 1], // 3 inputs -> 2 hidden -> 1 output
        };

        let metadata = AprMetadata {
            name: Some("demo-mlp".to_string()),
            description: Some("Demo MLP for APR format example".to_string()),
            trained_at: Some("2024-12-09T00:00:00Z".to_string()),
            framework_version: Some("aprender-0.16.0".to_string()),
            custom: std::collections::HashMap::new(),
        };

        // In a real scenario, you would use AprModel::load("model.apr")
        // For this demo, we construct directly
        let model = create_demo_model(weights, metadata);

        println!("Model Info:");
        println!("  Type: {:?}", model.model_type());
        println!("  Parameters: {}", model.num_parameters());
        if let Some(name) = &model.metadata().name {
            println!("  Name: {}", name);
        }
        if let Some(desc) = &model.metadata().description {
            println!("  Description: {}", desc);
        }
        println!();

        // 4. Run inference
        println!("Running Inference:");
        let inputs = [
            vec![1.0, 0.5, 0.2],
            vec![0.0, 1.0, 0.0],
            vec![0.5, 0.5, 0.5],
        ];

        for (i, input) in inputs.iter().enumerate() {
            match model.predict(input) {
                Ok(output) => {
                    println!("  Input {}: {:?} -> Output: {:.4}", i + 1, input, output[0]);
                },
                Err(e) => {
                    println!("  Input {}: Error - {}", i + 1, e);
                },
            }
        }
        println!();

        // 5. Demonstrate format detection
        println!("Format Detection:");
        let test_paths = [
            ("model.apr", "apr"),
            ("model.gguf", "gguf"),
            ("model.safetensors", "safetensors"),
            ("model.unknown", "unknown"),
        ];

        for (path, expected) in &test_paths {
            let detected = detect_format(path);
            let status = if detected == *expected {
                "OK"
            } else {
                "MISMATCH"
            };
            println!("  {} -> {} [{}]", path, detected, status);
        }
        println!();

        // 6. Header parsing demo
        println!("Header Parsing Demo:");
        let mut header_bytes = vec![0u8; HEADER_SIZE];
        header_bytes[0..4].copy_from_slice(&MAGIC);
        header_bytes[4] = 1; // version major
        header_bytes[5] = 0; // version minor
        header_bytes[6] = 0x05; // flags: compressed + signed
        header_bytes[8..10].copy_from_slice(&AprModelType::NeuralSequential.as_u16().to_le_bytes());
        header_bytes[10..14].copy_from_slice(&100u32.to_le_bytes()); // metadata_len
        header_bytes[14..18].copy_from_slice(&1000u32.to_le_bytes()); // payload_len

        match AprHeader::from_bytes(&header_bytes) {
            Ok(header) => {
                println!("  Version: {}.{}", header.version.0, header.version.1);
                println!("  Model Type: {:?}", header.model_type);
                println!("  Compressed: {}", header.is_compressed());
                println!("  Encrypted: {}", header.is_encrypted());
                println!("  Signed: {}", header.is_signed());
                println!("  Metadata Length: {} bytes", header.metadata_len);
                println!("  Payload Length: {} bytes", header.payload_len);
            },
            Err(e) => {
                println!("  Header parse error: {}", e);
            },
        }
        println!();

        // 7. Batch inference demo
        println!("Batch Inference:");
        let batch_inputs: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ];

        match model.predict_batch(&batch_inputs) {
            Ok(outputs) => {
                println!("  Batch size: {}", batch_inputs.len());
                for (i, output) in outputs.iter().enumerate() {
                    println!("  Sample {}: {:.4}", i + 1, output[0]);
                }
            },
            Err(e) => {
                println!("  Batch error: {}", e);
            },
        }
        println!();

        println!("=== APR Demo Complete! ===");
        println!();
        println!("For production use with .apr models:");
        println!("  let model = AprModel::load(\"path/to/model.apr\")?;");
        println!("  let output = model.predict(&input)?;");
        println!();
        println!("For SLM evaluation with Pareto frontier analysis:");
        println!("  See: https://github.com/paiml/single-shot-eval");
    }

    /// Create a demo model (helper for the example)
    /// In production, use AprModel::load() instead
    fn create_demo_model(weights: ModelWeights, metadata: AprMetadata) -> DemoAprModel {
        DemoAprModel {
            model_type: AprModelType::NeuralSequential,
            weights,
            metadata,
        }
    }

    /// Demo wrapper that mirrors AprModel's public interface
    struct DemoAprModel {
        model_type: AprModelType,
        weights: ModelWeights,
        metadata: AprMetadata,
    }

    impl DemoAprModel {
        fn model_type(&self) -> AprModelType {
            self.model_type
        }

        fn metadata(&self) -> &AprMetadata {
            &self.metadata
        }

        fn num_parameters(&self) -> usize {
            let weight_params: usize = self.weights.weights.iter().map(Vec::len).sum();
            let bias_params: usize = self.weights.biases.iter().map(Vec::len).sum();
            weight_params + bias_params
        }

        fn predict(&self, input: &[f32]) -> Result<Vec<f32>, String> {
            if self.weights.dimensions.is_empty() {
                return Err("Model has no layers".to_string());
            }

            let expected_input_dim = self.weights.dimensions[0];
            if input.len() != expected_input_dim {
                return Err(format!(
                    "Input dimension mismatch: expected {}, got {}",
                    expected_input_dim,
                    input.len()
                ));
            }

            let mut current = input.to_vec();

            for (i, (weights, biases)) in self
                .weights
                .weights
                .iter()
                .zip(self.weights.biases.iter())
                .enumerate()
            {
                let in_dim = self.weights.dimensions[i];
                let out_dim = self.weights.dimensions[i + 1];

                let mut output = vec![0.0; out_dim];

                for (j, out_val) in output.iter_mut().enumerate() {
                    let mut sum = biases.get(j).copied().unwrap_or(0.0);
                    for (k, &in_val) in current.iter().enumerate() {
                        let weight_idx = j * in_dim + k;
                        if let Some(&w) = weights.get(weight_idx) {
                            sum += in_val * w;
                        }
                    }
                    *out_val = sum;
                }

                // ReLU for hidden layers
                if i < self.weights.weights.len() - 1 {
                    for val in &mut output {
                        *val = val.max(0.0);
                    }
                }

                current = output;
            }

            Ok(current)
        }

        fn predict_batch(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
            inputs.iter().map(|input| self.predict(input)).collect()
        }
    }
} // end mod disabled
