use super::*;

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

pub(super) fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let va = Vector::from_slice(a);
    let vb = Vector::from_slice(b);

    let dot = va.dot(&vb).unwrap_or(0.0);
    let norm_a = va.norm_l2().unwrap_or(1.0);
    let norm_b = vb.norm_l2().unwrap_or(1.0);

    dot / (norm_a * norm_b)
}
