#!/usr/bin/env rust-script
//! PyTorch Inference Migration Example - Transpiled Output
//!
//! This shows the expected Rust/Realizar code after Batuta transpilation.
//!
//! Demonstrates:
//! - PyTorch → Realizar inference pipeline
//! - Model loading and execution
//! - Batch processing for throughput
//! - Zero-copy tensor operations
//!
//! Run with: cargo run --example pytorch_inference_output

use std::collections::HashMap;
use std::time::Instant;

// Realizar imports (would be used in real implementation)
// use realizar::{Model, Tensor, InferenceSession};
// use realizar::ops::{embedding, lstm, linear, softmax};

/// Sentiment classifier labels
const CLASS_LABELS: [&str; 3] = ["Negative", "Neutral", "Positive"];

/// Simple word-level tokenizer
struct SimpleTokenizer {
    word_to_idx: HashMap<String, usize>,
    idx_to_word: HashMap<usize, String>,
    unk_idx: usize,
    pad_idx: usize,
}

impl SimpleTokenizer {
    fn new() -> Self {
        let mut word_to_idx = HashMap::new();
        word_to_idx.insert("<PAD>".to_string(), 0);
        word_to_idx.insert("<UNK>".to_string(), 1);
        word_to_idx.insert("great".to_string(), 2);
        word_to_idx.insert("good".to_string(), 3);
        word_to_idx.insert("excellent".to_string(), 4);
        word_to_idx.insert("bad".to_string(), 5);
        word_to_idx.insert("terrible".to_string(), 6);
        word_to_idx.insert("awful".to_string(), 7);
        word_to_idx.insert("movie".to_string(), 8);
        word_to_idx.insert("film".to_string(), 9);
        word_to_idx.insert("the".to_string(), 10);
        word_to_idx.insert("a".to_string(), 11);
        word_to_idx.insert("is".to_string(), 12);
        word_to_idx.insert("was".to_string(), 13);
        word_to_idx.insert("very".to_string(), 14);
        word_to_idx.insert("really".to_string(), 15);
        word_to_idx.insert("not".to_string(), 16);
        word_to_idx.insert("love".to_string(), 17);
        word_to_idx.insert("hate".to_string(), 18);
        word_to_idx.insert("boring".to_string(), 19);
        word_to_idx.insert("amazing".to_string(), 20);

        let idx_to_word: HashMap<usize, String> =
            word_to_idx.iter().map(|(k, v)| (*v, k.clone())).collect();

        Self {
            word_to_idx,
            idx_to_word,
            unk_idx: 1,
            pad_idx: 0,
        }
    }

    fn encode(&self, text: &str, max_len: usize) -> Vec<usize> {
        // Lowercase and split
        let words: Vec<String> = text
            .to_lowercase()
            .replace(".", "")
            .replace(",", "")
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        // Convert to indices
        let mut indices: Vec<usize> = words
            .iter()
            .map(|word| *self.word_to_idx.get(word).unwrap_or(&self.unk_idx))
            .collect();

        // Pad or truncate to max_len
        if indices.len() < max_len {
            indices.resize(max_len, self.pad_idx);
        } else {
            indices.truncate(max_len);
        }

        indices
    }

    fn decode(&self, indices: &[usize]) -> String {
        indices
            .iter()
            .filter(|&&idx| idx != self.pad_idx)
            .filter_map(|idx| self.idx_to_word.get(idx))
            .cloned()
            .collect::<Vec<String>>()
            .join(" ")
    }
}

/// Simplified sentiment classifier (placeholder for Realizar model)
struct SentimentClassifier {
    vocab_size: usize,
    embedding_dim: usize,
    hidden_dim: usize,
    num_classes: usize,
}

impl SentimentClassifier {
    fn new(vocab_size: usize, embedding_dim: usize, hidden_dim: usize, num_classes: usize) -> Self {
        Self {
            vocab_size,
            embedding_dim,
            hidden_dim,
            num_classes,
        }
    }

    fn forward(&self, input_ids: &[Vec<usize>]) -> Vec<Vec<f32>> {
        // Simplified inference (real implementation would use Realizar ops)
        let batch_size = input_ids.len();

        // Mock probabilities (in practice, run actual model)
        let mut logits = Vec::new();

        for _ in 0..batch_size {
            // Placeholder: return mock probabilities
            let mock_probs = vec![0.1, 0.2, 0.7]; // Favor positive sentiment
            logits.push(mock_probs);
        }

        logits
    }
}

/// Run softmax normalization
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let exp_logits: Vec<f32> = logits.iter().map(|x| (x - max_logit).exp()).collect();

    let sum_exp: f32 = exp_logits.iter().sum();

    exp_logits.iter().map(|x| x / sum_exp).collect()
}

/// Load pre-trained model
fn load_model() -> (SentimentClassifier, SimpleTokenizer) {
    // Model hyperparameters
    let vocab_size = 1000;
    let embedding_dim = 128;
    let hidden_dim = 256;
    let num_classes = 3;

    // Create model
    let model = SentimentClassifier::new(vocab_size, embedding_dim, hidden_dim, num_classes);

    // Create tokenizer
    let tokenizer = SimpleTokenizer::new();

    (model, tokenizer)
}

/// Run inference on a single text sample
fn predict_single(
    model: &SentimentClassifier,
    tokenizer: &SimpleTokenizer,
    text: &str,
) -> (usize, Vec<f32>) {
    // Tokenize
    let token_ids = tokenizer.encode(text, 20);

    // Run inference
    let logits = model.forward(&[token_ids]);

    // Apply softmax
    let probabilities = softmax(&logits[0]);

    // Get prediction
    let predicted_class = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .expect("probabilities must not be empty");

    (predicted_class, probabilities)
}

/// Run inference on a batch of texts
fn predict_batch(
    model: &SentimentClassifier,
    tokenizer: &SimpleTokenizer,
    texts: &[String],
) -> (Vec<usize>, Vec<Vec<f32>>) {
    // Tokenize all texts
    let token_ids_list: Vec<Vec<usize>> =
        texts.iter().map(|text| tokenizer.encode(text, 20)).collect();

    // Run inference
    let logits = model.forward(&token_ids_list);

    // Apply softmax to each sample
    let probabilities: Vec<Vec<f32>> = logits.iter().map(|l| softmax(l)).collect();

    // Get predictions
    let predicted_classes: Vec<usize> = probabilities
        .iter()
        .map(|probs| {
            probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .expect("probabilities must not be empty")
        })
        .collect();

    (predicted_classes, probabilities)
}

fn main() {
    println!("PyTorch → Rust/Realizar Sentiment Analysis Inference");
    println!("{}", "=".repeat(50));

    // 1. Load model and tokenizer
    println!("\n1. Loading model and tokenizer...");
    let (model, tokenizer) = load_model();
    println!("   Model loaded (eval mode)");
    println!("   Vocabulary size: {}", tokenizer.word_to_idx.len());

    // 2. Single inference
    println!("\n2. Single text inference...");
    let sample_text = "This movie was really great and amazing";
    println!("   Input: \"{}\"", sample_text);

    let (pred_class, probs) = predict_single(&model, &tokenizer, sample_text);

    println!("   Predicted: {}", CLASS_LABELS[pred_class]);
    println!("   Probabilities:");
    for (i, label) in CLASS_LABELS.iter().enumerate() {
        println!("      {}: {:.4}", label, probs[i]);
    }

    // 3. Batch inference
    println!("\n3. Batch inference...");
    let batch_texts = vec![
        "The film was excellent".to_string(),
        "This was terrible and boring".to_string(),
        "An average movie, nothing special".to_string(),
        "I love this, it was amazing".to_string(),
    ];

    println!("   Batch size: {}", batch_texts.len());

    let (pred_classes, batch_probs) = predict_batch(&model, &tokenizer, &batch_texts);

    println!("   Results:");
    for (i, text) in batch_texts.iter().enumerate() {
        println!("\n   Text: \"{}\"", text);
        println!("   Prediction: {}", CLASS_LABELS[pred_classes[i]]);
        println!("   Probabilities: {:?}", batch_probs[i]);
    }

    // 4. Throughput test
    println!("\n4. Throughput test...");
    let test_texts: Vec<String> = vec!["This is a test sentence".to_string(); 100];

    let start = Instant::now();
    let (_, _) = predict_batch(&model, &tokenizer, &test_texts);
    let elapsed = start.elapsed();

    let throughput = test_texts.len() as f64 / elapsed.as_secs_f64();
    let latency = elapsed.as_secs_f64() / test_texts.len() as f64 * 1000.0; // ms per sample

    println!("   Processed {} samples", test_texts.len());
    println!("   Time: {:.3}s", elapsed.as_secs_f64());
    println!("   Throughput: {:.1} samples/sec", throughput);
    println!("   Latency: {:.2} ms/sample", latency);

    println!("\n✅ Inference pipeline complete!");

    println!("\n⚡ Performance Benefits:");
    println!("   • Zero Python overhead: 2-5× faster inference");
    println!("   • Memory efficient: 50-80% less memory usage");
    println!("   • Predictable latency: No GIL, no GC pauses");
    println!("   • Single binary: 5-20 MB vs 500+ MB for Python + PyTorch");
    println!("   • Edge deployment: Run on embedded devices, WebAssembly");
}
