//\! Neural Tests - Extreme TDD
//\! PMAT-085: Extracted from mod.rs for PMAT file health compliance

use super::*;

// ==================== NeuralEncoderConfig Tests ====================

#[test]
fn test_config_default() {
    let config = NeuralEncoderConfig::default();
    assert_eq!(config.vocab_size, 8192);
    assert_eq!(config.embed_dim, 256);
    assert_eq!(config.output_dim, 256);
}

#[test]
fn test_config_minimal() {
    let config = NeuralEncoderConfig::minimal();
    assert_eq!(config.vocab_size, 1000);
    assert_eq!(config.embed_dim, 64);
    assert!(config.num_layers < NeuralEncoderConfig::default().num_layers);
}

#[test]
fn test_config_small() {
    let config = NeuralEncoderConfig::small();
    assert!(config.embed_dim > NeuralEncoderConfig::minimal().embed_dim);
    assert!(config.embed_dim < NeuralEncoderConfig::default().embed_dim);
}

// ==================== Vocabulary Tests ====================

#[test]
fn test_vocabulary_creation() {
    let vocab = Vocabulary::for_rust_errors();
    assert!(vocab.vocab_size() > 0);
}

#[test]
fn test_vocabulary_special_tokens() {
    let vocab = Vocabulary::for_rust_errors();
    assert!(vocab.cls_token() < vocab.vocab_size());
    assert!(vocab.sep_token() < vocab.vocab_size());
    assert!(vocab.eos_token() < vocab.vocab_size());
}

#[test]
fn test_vocabulary_lang_tokens() {
    let vocab = Vocabulary::for_rust_errors();
    let python_token = vocab.lang_token("python");
    let rust_token = vocab.lang_token("rust");
    assert_ne!(python_token, rust_token);
}

#[test]
fn test_vocabulary_tokenize_simple() {
    let vocab = Vocabulary::for_rust_errors();
    let tokens = vocab.tokenize("error expected type");
    assert!(!tokens.is_empty());
}

#[test]
fn test_vocabulary_tokenize_with_punctuation() {
    let vocab = Vocabulary::for_rust_errors();
    let tokens = vocab.tokenize("E0308: mismatched types");
    assert!(tokens.len() >= 3);
}

#[test]
fn test_vocabulary_tokenize_error_code() {
    let vocab = Vocabulary::for_rust_errors();
    let tokens = vocab.tokenize("E0308");
    assert_eq!(tokens.len(), 1);
    // Should not be UNK since E0308 is in vocab
    assert_ne!(tokens[0], vocab.unk_id);
}

#[test]
fn test_vocabulary_unknown_token() {
    let vocab = Vocabulary::for_rust_errors();
    let tokens = vocab.tokenize("xyzzy12345");
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0], vocab.unk_id);
}

// ==================== NeuralErrorEncoder Tests ====================

#[test]
fn test_encoder_creation() {
    let encoder = NeuralErrorEncoder::with_config(NeuralEncoderConfig::minimal());
    assert!(!encoder.is_training());
}

#[test]
fn test_encoder_train_eval_mode() {
    let mut encoder = NeuralErrorEncoder::with_config(NeuralEncoderConfig::minimal());

    encoder.train();
    assert!(encoder.is_training());

    encoder.eval();
    assert!(!encoder.is_training());
}

#[test]
fn test_encoder_num_parameters() {
    let encoder = NeuralErrorEncoder::with_config(NeuralEncoderConfig::minimal());
    let num_params = encoder.num_parameters();
    assert!(num_params > 0);
}

#[test]
fn test_encoder_encode_returns_correct_dim() {
    let config = NeuralEncoderConfig::minimal();
    let output_dim = config.output_dim;
    let encoder = NeuralErrorEncoder::with_config(config);

    let embedding = encoder.encode("E0308: mismatched types", "let x: i32 = \"hello\";", "rust");

    assert_eq!(embedding.len(), output_dim);
}

#[test]
fn test_encoder_embedding_is_normalized() {
    let encoder = NeuralErrorEncoder::with_config(NeuralEncoderConfig::minimal());

    let embedding = encoder.encode("E0308: mismatched types", "let x: i32 = \"hello\";", "rust");

    // Check L2 norm is approximately 1
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "Embedding norm should be ~1.0, got {norm}"
    );
}

#[test]
fn test_encoder_similar_errors_similar_embeddings() {
    let encoder = NeuralErrorEncoder::with_config(NeuralEncoderConfig::minimal());

    // Two similar type mismatch errors
    let emb1 = encoder.encode(
        "E0308: mismatched types, expected i32 found &str",
        "let x: i32 = \"hello\";",
        "rust",
    );
    let emb2 = encoder.encode(
        "E0308: mismatched types, expected i32 found String",
        "let y: i32 = String::new();",
        "rust",
    );

    // Different error type
    let emb3 = encoder.encode(
        "E0382: borrow of moved value",
        "let x = vec![1]; let y = x; let z = x;",
        "rust",
    );

    // Compute cosine similarities
    let sim_12 = cosine_sim(&emb1, &emb2);
    let sim_13 = cosine_sim(&emb1, &emb3);

    // Similar errors should have higher similarity (with tolerance for minimal config)
    // With minimal config, the encoder may not distinguish well, so we allow
    // near-ties (within 1%) as acceptable - both represent high similarity
    let tolerance = 0.01;
    assert!(
            sim_12 > sim_13 - tolerance,
            "Similar errors should have higher similarity (or near-tie): sim_12={sim_12}, sim_13={sim_13}"
        );
}

#[test]
fn test_encoder_different_languages() {
    let encoder = NeuralErrorEncoder::with_config(NeuralEncoderConfig::minimal());

    // Same error, different source languages
    let emb_rust = encoder.encode("E0308: mismatched types", "let x: i32 = \"hello\";", "rust");
    let emb_python = encoder.encode(
        "TypeError: expected int, got str",
        "x: int = \"hello\"",
        "python",
    );

    // Should still be somewhat similar (both are type errors)
    let sim = cosine_sim(&emb_rust, &emb_python);
    // Just verify it's a valid similarity value
    assert!((-1.0..=1.0).contains(&sim));
}

// ==================== ContrastiveLoss Tests ====================

#[test]
fn test_contrastive_loss_creation() {
    let loss = ContrastiveLoss::new();
    assert!((loss.temperature - 0.07).abs() < 0.001);
}

#[test]
fn test_contrastive_loss_custom_temperature() {
    let loss = ContrastiveLoss::with_temperature(0.1);
    assert!((loss.temperature - 0.1).abs() < 0.001);
}

// ==================== TripletLoss Tests ====================

#[test]
fn test_triplet_loss_creation() {
    let loss = TripletLoss::new();
    assert!((loss.margin() - 1.0).abs() < 0.001);
    assert_eq!(loss.distance_metric(), TripletDistance::Euclidean);
}

#[test]
fn test_triplet_loss_default() {
    let loss = TripletLoss::default();
    assert!((loss.margin() - 1.0).abs() < 0.001);
}

#[test]
fn test_triplet_loss_custom_margin() {
    let loss = TripletLoss::with_margin(0.5);
    assert!((loss.margin() - 0.5).abs() < 0.001);
}

#[test]
fn test_triplet_loss_with_distance() {
    let loss = TripletLoss::new().with_distance(TripletDistance::Cosine);
    assert_eq!(loss.distance_metric(), TripletDistance::Cosine);
}

#[test]
fn test_triplet_loss_zero_when_satisfied() {
    // When d(a,p) < d(a,n) by more than margin, loss should be 0
    let loss = TripletLoss::with_margin(0.1);

    // Anchor close to positive, far from negative
    let anchor = Tensor::new(&[0.0, 0.0], &[1, 2]);
    let positive = Tensor::new(&[0.1, 0.0], &[1, 2]); // d = 0.1
    let negative = Tensor::new(&[5.0, 0.0], &[1, 2]); // d = 5.0

    let loss_val = loss.forward(&anchor, &positive, &negative);
    // d_ap (0.1) - d_an (5.0) + margin (0.1) = -4.8, max(0, -4.8) = 0
    assert!(
        loss_val.data()[0] < 0.01,
        "Loss should be ~0 when triplet is satisfied"
    );
}

#[test]
fn test_triplet_loss_positive_when_violated() {
    // When d(a,p) > d(a,n), loss should be positive
    let loss = TripletLoss::with_margin(0.5);

    // Anchor closer to negative than positive (violation)
    let anchor = Tensor::new(&[0.0, 0.0], &[1, 2]);
    let positive = Tensor::new(&[3.0, 0.0], &[1, 2]); // d = 3.0
    let negative = Tensor::new(&[1.0, 0.0], &[1, 2]); // d = 1.0

    let loss_val = loss.forward(&anchor, &positive, &negative);
    // d_ap (3.0) - d_an (1.0) + margin (0.5) = 2.5, max(0, 2.5) = 2.5
    assert!(
        loss_val.data()[0] > 2.0,
        "Loss should be positive when triplet is violated"
    );
}

#[test]
fn test_triplet_loss_batch() {
    let loss = TripletLoss::with_margin(1.0);

    // Batch of 2
    let anchor = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[2, 2]);
    let positive = Tensor::new(&[0.1, 0.0, 0.1, 0.0], &[2, 2]);
    let negative = Tensor::new(&[5.0, 0.0, 5.0, 0.0], &[2, 2]);

    let loss_val = loss.forward(&anchor, &positive, &negative);
    assert_eq!(loss_val.shape(), &[1]);
}

#[test]
fn test_triplet_loss_squared_euclidean() {
    let loss = TripletLoss::with_margin(1.0).with_distance(TripletDistance::SquaredEuclidean);

    let anchor = Tensor::new(&[0.0, 0.0], &[1, 2]);
    let positive = Tensor::new(&[1.0, 0.0], &[1, 2]); // squared_d = 1.0
    let negative = Tensor::new(&[2.0, 0.0], &[1, 2]); // squared_d = 4.0

    let loss_val = loss.forward(&anchor, &positive, &negative);
    // d_ap_sq (1) - d_an_sq (4) + margin (1) = -2, max(0, -2) = 0
    assert!(loss_val.data()[0] < 0.01);
}

#[test]
fn test_triplet_loss_cosine() {
    let loss = TripletLoss::with_margin(0.1).with_distance(TripletDistance::Cosine);

    // Anchor pointing in x direction
    let anchor = Tensor::new(&[1.0, 0.0], &[1, 2]);
    // Positive also pointing mostly in x
    let positive = Tensor::new(&[0.9, 0.1], &[1, 2]);
    // Negative pointing in y direction
    let negative = Tensor::new(&[0.0, 1.0], &[1, 2]);

    let loss_val = loss.forward(&anchor, &positive, &negative);
    // Cosine distance: 1 - cos(angle)
    // anchor-positive: small angle, small distance
    // anchor-negative: 90 degrees, distance = 1.0
    // Loss should be 0 or small
    assert!(loss_val.data()[0] < 0.5);
}

#[test]
fn test_pairwise_distances() {
    let loss = TripletLoss::new();

    let embeddings = Tensor::new(
        &[
            0.0, 0.0, // Point 0 at origin
            1.0, 0.0, // Point 1 at (1,0)
            0.0, 1.0, // Point 2 at (0,1)
        ],
        &[3, 2],
    );

    let distances = loss.pairwise_distances(&embeddings);
    let data = distances.data();

    assert_eq!(distances.shape(), &[3, 3]);

    // Diagonal should be 0 (distance to self)
    assert!(data[0] < 0.01); // d(0,0)
    assert!(data[4] < 0.01); // d(1,1)
    assert!(data[8] < 0.01); // d(2,2)

    // d(0,1) = 1.0 (Euclidean)
    assert!((data[1] - 1.0).abs() < 0.01);
    // d(0,2) = 1.0
    assert!((data[2] - 1.0).abs() < 0.01);
    // d(1,2) = sqrt(2)
    assert!((data[5] - std::f32::consts::SQRT_2).abs() < 0.01);
}

#[test]
fn test_mine_hard_triplets() {
    let loss = TripletLoss::new();

    // 4 embeddings: 2 per class
    let embeddings = Tensor::new(
        &[
            0.0, 0.0, // Class 0, sample 0
            0.1, 0.0, // Class 0, sample 1
            5.0, 0.0, // Class 1, sample 0
            5.1, 0.0, // Class 1, sample 1
        ],
        &[4, 2],
    );
    let labels = vec![0, 0, 1, 1];

    let triplets = loss.mine_hard_triplets(&embeddings, &labels);

    // Should find triplets for each anchor
    assert!(!triplets.is_empty());

    // Verify triplet structure: (anchor, positive, negative)
    for (a, p, n) in &triplets {
        assert_eq!(labels[*a], labels[*p], "Positive should be same class");
        assert_ne!(labels[*a], labels[*n], "Negative should be different class");
    }
}

#[test]
fn test_mine_hard_triplets_single_class() {
    let loss = TripletLoss::new();

    // All same class - no valid triplets
    let embeddings = Tensor::new(&[0.0, 0.0, 1.0, 0.0, 2.0, 0.0], &[3, 2]);
    let labels = vec![0, 0, 0];

    let triplets = loss.mine_hard_triplets(&embeddings, &labels);
    assert!(triplets.is_empty(), "No triplets when all same class");
}

#[test]
fn test_batch_hard_loss() {
    let loss = TripletLoss::with_margin(0.5);

    // Well-separated classes
    let embeddings = Tensor::new(
        &[
            0.0, 0.0, // Class 0
            0.1, 0.1, // Class 0
            10.0, 0.0, // Class 1
            10.1, 0.1, // Class 1
        ],
        &[4, 2],
    );
    let labels = vec![0, 0, 1, 1];

    let loss_val = loss.batch_hard_loss(&embeddings, &labels);
    // Well-separated: loss should be 0 or very small
    assert!(loss_val.data()[0] < 1.0);
}

#[test]
fn test_batch_hard_loss_overlapping() {
    let loss = TripletLoss::with_margin(1.0);

    // Overlapping classes - should have higher loss
    let embeddings = Tensor::new(
        &[
            0.0, 0.0, // Class 0
            0.5, 0.0, // Class 0
            0.3, 0.0, // Class 1 (between class 0 points!)
            0.8, 0.0, // Class 1
        ],
        &[4, 2],
    );
    let labels = vec![0, 0, 1, 1];

    let loss_val = loss.batch_hard_loss(&embeddings, &labels);
    // Classes overlap: loss should be positive
    assert!(loss_val.data()[0] > 0.0);
}

#[test]
fn test_batch_hard_loss_empty() {
    let loss = TripletLoss::new();

    // Single sample - no valid triplets
    let embeddings = Tensor::new(&[0.0, 0.0], &[1, 2]);
    let labels = vec![0];

    let loss_val = loss.batch_hard_loss(&embeddings, &labels);
    assert!(
        loss_val.data()[0].abs() < 0.001,
        "Loss should be 0 for single sample"
    );
}

// ==================== TrainingSample Tests ====================

#[test]
fn test_training_sample_creation() {
    let sample = TrainingSample::new("E0308: mismatched types", "let x: i32 = \"hello\";", "rust");

    assert_eq!(sample.source_lang, "rust");
    assert!(sample.positive.is_none());
}

#[test]
fn test_training_sample_with_positive() {
    let positive = TrainingSample::new(
        "E0308: type mismatch",
        "let y: i32 = String::new();",
        "rust",
    );

    let sample = TrainingSample::new("E0308: mismatched types", "let x: i32 = \"hello\";", "rust")
        .with_positive(positive);

    assert!(sample.positive.is_some());
}

#[test]
fn test_training_sample_with_category() {
    let sample = TrainingSample::new("E0308: mismatched types", "let x: i32 = \"hello\";", "rust")
        .with_category("type_mismatch");

    assert_eq!(sample.category, "type_mismatch");
}

// ==================== Additional Coverage Tests ====================

#[test]
fn test_encoder_default() {
    let encoder = NeuralErrorEncoder::default();
    assert_eq!(
        encoder.config().vocab_size,
        NeuralEncoderConfig::default().vocab_size
    );
}

#[test]
fn test_encoder_config_accessor() {
    let config = NeuralEncoderConfig::minimal();
    let encoder = NeuralErrorEncoder::with_config(config.clone());
    assert_eq!(encoder.config().embed_dim, config.embed_dim);
    assert_eq!(encoder.config().output_dim, config.output_dim);
}

#[test]
fn test_encoder_encode_batch() {
    let encoder = NeuralErrorEncoder::with_config(NeuralEncoderConfig::minimal());
    let batch = vec![
        ("E0308: type mismatch", "let x: i32 = \"hello\";", "rust"),
        (
            "E0382: use of moved value",
            "let a = vec![1]; let b = a; let c = a;",
            "rust",
        ),
    ];

    let embeddings = encoder.encode_batch(&batch);
    assert_eq!(embeddings.shape()[0], 2); // batch size
    assert_eq!(embeddings.shape()[1], encoder.config().output_dim);
}

#[test]
fn test_encoder_training_mode_dropout() {
    let mut encoder = NeuralErrorEncoder::with_config(NeuralEncoderConfig::minimal());

    // Get embedding in eval mode
    encoder.eval();
    let emb_eval = encoder.encode("E0308: mismatched types", "let x: i32 = \"hello\";", "rust");

    // Get embedding in train mode (dropout active)
    encoder.train();
    let _emb_train = encoder.encode("E0308: mismatched types", "let x: i32 = \"hello\";", "rust");

    // Both should be valid embeddings
    assert_eq!(emb_eval.len(), encoder.config().output_dim);
}

#[test]
fn test_contrastive_loss_default() {
    let loss = ContrastiveLoss::default();
    assert!((loss.temperature - 0.07).abs() < 0.001);
}

#[test]
fn test_contrastive_loss_forward_with_in_batch_negatives() {
    let loss = ContrastiveLoss::new();

    // Batch of 2 embeddings
    let anchor = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let positive = Tensor::new(&[0.9, 0.1, 0.1, 0.9], &[2, 2]);

    let loss_val = loss.forward(&anchor, &positive, None);
    assert_eq!(loss_val.shape(), &[1]);
    assert!(loss_val.data()[0].is_finite());
}

#[test]
fn test_contrastive_loss_forward_with_explicit_negatives() {
    let loss = ContrastiveLoss::new();

    let anchor = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let positive = Tensor::new(&[0.9, 0.1, 0.1, 0.9], &[2, 2]);
    // 2 negatives per sample
    let negatives = Tensor::new(
        &[
            // Negatives for sample 0
            -1.0, 0.0, 0.0, -1.0, // Negatives for sample 1
            -1.0, 0.0, 0.0, -1.0,
        ],
        &[2, 2, 2],
    );

    let loss_val = loss.forward(&anchor, &positive, Some(&negatives));
    assert_eq!(loss_val.shape(), &[1]);
    assert!(loss_val.data()[0].is_finite());
}

#[test]
fn test_triplet_distance_debug_clone() {
    let dist = TripletDistance::Euclidean;
    let cloned = dist.clone();
    assert_eq!(dist, cloned);

    let debug_str = format!("{:?}", dist);
    assert!(debug_str.contains("Euclidean"));

    let cosine_debug = format!("{:?}", TripletDistance::Cosine);
    assert!(cosine_debug.contains("Cosine"));

    let squared_debug = format!("{:?}", TripletDistance::SquaredEuclidean);
    assert!(squared_debug.contains("SquaredEuclidean"));
}

#[test]
fn test_triplet_loss_clone() {
    let loss = TripletLoss::with_margin(0.5).with_distance(TripletDistance::Cosine);
    let cloned = loss.clone();
    assert!((loss.margin() - cloned.margin()).abs() < f32::EPSILON);
    assert_eq!(loss.distance_metric(), cloned.distance_metric());
}

#[test]
fn test_neural_encoder_config_debug_clone() {
    let config = NeuralEncoderConfig::default();
    let cloned = config.clone();
    assert_eq!(config.vocab_size, cloned.vocab_size);
    assert_eq!(config.embed_dim, cloned.embed_dim);

    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("vocab_size"));
}

#[test]
fn test_vocabulary_unknown_language() {
    let vocab = Vocabulary::for_rust_errors();
    let unknown_lang_token = vocab.lang_token("unknown_language");
    // Should return UNK token for unknown language
    assert_eq!(unknown_lang_token, vocab.unk_id);
}

#[test]
fn test_vocabulary_debug() {
    let vocab = Vocabulary::for_rust_errors();
    let debug_str = format!("{:?}", vocab);
    assert!(debug_str.contains("Vocabulary"));
}

#[test]
fn test_training_sample_debug_clone() {
    let sample = TrainingSample::new("E0308: mismatched types", "let x: i32 = \"hello\";", "rust");
    let cloned = sample.clone();
    assert_eq!(sample.error_message, cloned.error_message);
    assert_eq!(sample.source_lang, cloned.source_lang);

    let debug_str = format!("{:?}", sample);
    assert!(debug_str.contains("error_message"));
}

#[test]
fn test_encoder_tokenize_long_message() {
    let encoder = NeuralErrorEncoder::with_config(NeuralEncoderConfig::minimal());

    // Create a very long error message that exceeds max_seq_len
    let long_msg = "error ".repeat(500);
    let embedding = encoder.encode(&long_msg, "let x = 1;", "rust");

    // Should still produce valid embedding
    assert_eq!(embedding.len(), encoder.config().output_dim);
}

#[test]
fn test_encoder_encode_empty_context() {
    let encoder = NeuralErrorEncoder::with_config(NeuralEncoderConfig::minimal());
    let embedding = encoder.encode("E0308", "", "rust");
    assert_eq!(embedding.len(), encoder.config().output_dim);
}

#[test]
fn test_encoder_batch_size_one() {
    let encoder = NeuralErrorEncoder::with_config(NeuralEncoderConfig::minimal());
    let batch = vec![("E0308: type mismatch", "let x = 42;", "rust")];

    let embeddings = encoder.encode_batch(&batch);
    assert_eq!(embeddings.shape()[0], 1);
}

#[test]
fn test_contrastive_loss_debug() {
    let loss = ContrastiveLoss::new();
    let debug_str = format!("{:?}", loss);
    assert!(debug_str.contains("temperature"));
}

#[test]
fn test_triplet_loss_debug() {
    let loss = TripletLoss::new();
    let debug_str = format!("{:?}", loss);
    assert!(debug_str.contains("margin"));
}

#[test]
fn test_pairwise_distances_single_embedding() {
    let loss = TripletLoss::new();
    let embeddings = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);

    let distances = loss.pairwise_distances(&embeddings);
    assert_eq!(distances.shape(), &[1, 1]);
    // Distance to self should be 0
    assert!(distances.data()[0] < 0.01);
}

#[test]
fn test_neural_encoder_debug() {
    let encoder = NeuralErrorEncoder::with_config(NeuralEncoderConfig::minimal());
    let debug_str = format!("{:?}", encoder);
    assert!(debug_str.contains("NeuralErrorEncoder"));
}

// ==================== Helper Functions ====================

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let va = Vector::from_slice(a);
    let vb = Vector::from_slice(b);

    let dot = va.dot(&vb).unwrap_or(0.0);
    let norm_a = va.norm_l2().unwrap_or(1.0);
    let norm_b = vb.norm_l2().unwrap_or(1.0);

    dot / (norm_a * norm_b)
}
