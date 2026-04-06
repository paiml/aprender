pub(crate) use super::*;

#[test]
fn test_embedding_config_default() {
    let config = EmbeddingConfig::default();
    assert_eq!(config.embedding_dim, 192);
    assert_eq!(config.sample_rate, 16000);
    assert!(config.validate().is_ok());
}

#[test]
fn test_embedding_config_ecapa() {
    let config = EmbeddingConfig::ecapa_tdnn();
    assert_eq!(config.embedding_dim, 192);
    assert_eq!(config.n_mels, 80);
}

#[test]
fn test_embedding_config_xvector() {
    let config = EmbeddingConfig::x_vector();
    assert_eq!(config.embedding_dim, 512);
    assert_eq!(config.n_mels, 30);
}

#[test]
fn test_embedding_config_validation() {
    let mut config = EmbeddingConfig::default();
    config.embedding_dim = 0;
    assert!(config.validate().is_err());

    config.embedding_dim = 192;
    config.sample_rate = 0;
    assert!(config.validate().is_err());
}

#[test]
fn test_speaker_embedding_from_vec() {
    let emb = SpeakerEmbedding::from_vec(vec![1.0, 2.0, 3.0]);
    assert_eq!(emb.dim(), 3);
    assert!(!emb.is_normalized());
}

#[test]
fn test_speaker_embedding_zeros() {
    let emb = SpeakerEmbedding::zeros(192);
    assert_eq!(emb.dim(), 192);
    assert_eq!(emb.l2_norm(), 0.0);
}

#[test]
fn test_speaker_embedding_normalize() {
    let mut emb = SpeakerEmbedding::from_vec(vec![3.0, 4.0]);
    emb.normalize();
    assert!(emb.is_normalized());
    assert!((emb.l2_norm() - 1.0).abs() < 1e-6);
    assert!((emb.as_slice()[0] - 0.6).abs() < 1e-6);
    assert!((emb.as_slice()[1] - 0.8).abs() < 1e-6);
}

#[test]
fn test_speaker_embedding_dot() {
    let emb1 = SpeakerEmbedding::from_vec(vec![1.0, 0.0]);
    let emb2 = SpeakerEmbedding::from_vec(vec![0.0, 1.0]);
    assert!((emb1.dot(&emb2).unwrap() - 0.0).abs() < 1e-6);

    let emb3 = SpeakerEmbedding::from_vec(vec![1.0, 1.0]);
    assert!((emb1.dot(&emb3).unwrap() - 1.0).abs() < 1e-6);
}

#[test]
fn test_speaker_embedding_dot_dimension_mismatch() {
    let emb1 = SpeakerEmbedding::from_vec(vec![1.0, 0.0]);
    let emb2 = SpeakerEmbedding::from_vec(vec![0.0, 1.0, 0.0]);
    assert!(emb1.dot(&emb2).is_err());
}

#[test]
fn test_speaker_embedding_euclidean_distance() {
    let emb1 = SpeakerEmbedding::from_vec(vec![0.0, 0.0]);
    let emb2 = SpeakerEmbedding::from_vec(vec![3.0, 4.0]);
    assert!((emb1.euclidean_distance(&emb2).unwrap() - 5.0).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_identical() {
    let emb = SpeakerEmbedding::from_vec(vec![1.0, 2.0, 3.0]);
    let sim = cosine_similarity(&emb, &emb);
    assert!((sim - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let emb1 = SpeakerEmbedding::from_vec(vec![1.0, 0.0, 0.0]);
    let emb2 = SpeakerEmbedding::from_vec(vec![0.0, 1.0, 0.0]);
    let sim = cosine_similarity(&emb1, &emb2);
    assert!((sim - 0.0).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_opposite() {
    let emb1 = SpeakerEmbedding::from_vec(vec![1.0, 0.0]);
    let emb2 = SpeakerEmbedding::from_vec(vec![-1.0, 0.0]);
    let sim = cosine_similarity(&emb1, &emb2);
    assert!((sim - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_dimension_mismatch() {
    let emb1 = SpeakerEmbedding::from_vec(vec![1.0, 0.0]);
    let emb2 = SpeakerEmbedding::from_vec(vec![0.0, 1.0, 0.0]);
    let sim = cosine_similarity(&emb1, &emb2);
    assert_eq!(sim, 0.0);
}

#[test]
fn test_normalize_embedding() {
    let emb = SpeakerEmbedding::from_vec(vec![3.0, 4.0]);
    let normalized = normalize_embedding(&emb);
    assert!(normalized.is_normalized());
    assert!((normalized.l2_norm() - 1.0).abs() < 1e-6);
}

#[test]
fn test_average_embeddings() {
    let embeddings = vec![
        SpeakerEmbedding::from_vec(vec![1.0, 0.0]),
        SpeakerEmbedding::from_vec(vec![0.0, 1.0]),
    ];
    let avg = average_embeddings(&embeddings).unwrap();
    assert!((avg.as_slice()[0] - 0.5).abs() < 1e-6);
    assert!((avg.as_slice()[1] - 0.5).abs() < 1e-6);
}

#[test]
fn test_average_embeddings_empty() {
    let embeddings: Vec<SpeakerEmbedding> = vec![];
    assert!(average_embeddings(&embeddings).is_err());
}

#[test]
fn test_average_embeddings_dimension_mismatch() {
    let embeddings = vec![
        SpeakerEmbedding::from_vec(vec![1.0, 0.0]),
        SpeakerEmbedding::from_vec(vec![0.0, 1.0, 0.0]),
    ];
    assert!(average_embeddings(&embeddings).is_err());
}

#[test]
fn test_similarity_matrix() {
    let embeddings = vec![
        SpeakerEmbedding::from_vec(vec![1.0, 0.0]),
        SpeakerEmbedding::from_vec(vec![0.0, 1.0]),
        SpeakerEmbedding::from_vec(vec![1.0, 1.0]),
    ];
    let matrix = similarity_matrix(&embeddings);

    assert_eq!(matrix.len(), 3);
    // Diagonal should be 1.0
    assert!((matrix[0][0] - 1.0).abs() < 1e-6);
    assert!((matrix[1][1] - 1.0).abs() < 1e-6);
    // [0][1] and [1][0] should be 0.0 (orthogonal)
    assert!((matrix[0][1] - 0.0).abs() < 1e-6);
    assert!((matrix[1][0] - 0.0).abs() < 1e-6);
}

#[test]
fn test_ecapa_tdnn_stub() {
    let extractor = EcapaTdnn::default_config();
    assert_eq!(extractor.embedding_dim(), 192);
    assert_eq!(extractor.sample_rate(), 16000);

    let audio = vec![0.0_f32; 16000];
    let result = extractor.extract(&audio);
    assert!(result.is_err());
}

#[test]
fn test_ecapa_tdnn_empty_audio() {
    let extractor = EcapaTdnn::default_config();
    let result = extractor.extract(&[]);
    assert!(matches!(result, Err(VoiceError::InvalidAudio(_))));
}

#[test]
fn test_xvector_stub() {
    let extractor = XVector::default_config();
    assert_eq!(extractor.embedding_dim(), 512);
    assert_eq!(extractor.sample_rate(), 16000);
}

// ===== Additional coverage tests =====

#[test]
fn test_embedding_config_resnet() {
    let config = EmbeddingConfig::resnet();
    assert_eq!(config.embedding_dim, 256);
    assert_eq!(config.n_mels, 64);
    assert!(config.validate().is_ok());
}

#[test]
fn test_embedding_config_validate_frame_length_zero() {
    let config = EmbeddingConfig {
        frame_length_ms: 0,
        ..EmbeddingConfig::default()
    };
    let err = config.validate().unwrap_err();
    match err {
        VoiceError::InvalidConfig(msg) => assert!(msg.contains("frame_length_ms")),
        other => panic!("Expected InvalidConfig, got {other:?}"),
    }
}

#[test]
fn test_embedding_config_validate_n_mels_zero() {
    let config = EmbeddingConfig {
        n_mels: 0,
        ..EmbeddingConfig::default()
    };
    let err = config.validate().unwrap_err();
    match err {
        VoiceError::InvalidConfig(msg) => assert!(msg.contains("n_mels")),
        other => panic!("Expected InvalidConfig, got {other:?}"),
    }
}

#[test]
fn test_embedding_config_debug_clone() {
    let config = EmbeddingConfig::default();
    let cloned = config.clone();
    let debug_str = format!("{config:?}");
    assert!(!debug_str.is_empty());
    assert_eq!(cloned.embedding_dim, config.embedding_dim);
}

#[test]
fn test_speaker_embedding_as_mut_slice() {
    let mut emb = SpeakerEmbedding::from_vec(vec![1.0, 2.0, 3.0]);
    let slice = emb.as_mut_slice();
    slice[0] = 10.0;
    assert!((emb.as_slice()[0] - 10.0).abs() < f32::EPSILON);
}

#[test]
fn test_speaker_embedding_into_vec() {
    let emb = SpeakerEmbedding::from_vec(vec![1.0, 2.0, 3.0]);
    let vec = emb.into_vec();
    assert_eq!(vec, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_speaker_embedding_normalize_zero_vector() {
    let mut emb = SpeakerEmbedding::zeros(10);
    emb.normalize();
    // norm <= EPSILON, so vector stays zero but normalized flag is set
    assert!(emb.is_normalized());
    assert_eq!(emb.l2_norm(), 0.0);
    // Values should remain zero
    for &v in emb.as_slice() {
        assert!((v - 0.0).abs() < f32::EPSILON);
    }
}

#[test]
fn test_speaker_embedding_euclidean_distance_mismatch() {
    let emb1 = SpeakerEmbedding::from_vec(vec![1.0, 0.0]);
    let emb2 = SpeakerEmbedding::from_vec(vec![1.0, 0.0, 0.0]);
    let result = emb1.euclidean_distance(&emb2);
    assert!(result.is_err());
    match result.unwrap_err() {
        VoiceError::DimensionMismatch { expected, got } => {
            assert_eq!(expected, 2);
            assert_eq!(got, 3);
        }
        other => panic!("Expected DimensionMismatch, got {other:?}"),
    }
}

#[test]
fn test_speaker_embedding_euclidean_distance_same() {
    let emb = SpeakerEmbedding::from_vec(vec![1.0, 2.0, 3.0]);
    let dist = emb.euclidean_distance(&emb).expect("same dim");
    assert!((dist - 0.0).abs() < 1e-6);
}

#[test]
fn test_speaker_embedding_debug_clone() {
    let emb = SpeakerEmbedding::from_vec(vec![1.0, 2.0]);
    let cloned = emb.clone();
    let debug_str = format!("{emb:?}");
    assert!(!debug_str.is_empty());
    assert_eq!(cloned.dim(), emb.dim());
}

#[test]
fn test_cosine_similarity_zero_dim() {
    let emb1 = SpeakerEmbedding::from_vec(vec![]);
    let emb2 = SpeakerEmbedding::from_vec(vec![]);
    let sim = cosine_similarity(&emb1, &emb2);
    assert_eq!(sim, 0.0);
}

#[test]
fn test_cosine_similarity_zero_norm() {
    let emb1 = SpeakerEmbedding::zeros(3);
    let emb2 = SpeakerEmbedding::from_vec(vec![1.0, 0.0, 0.0]);
    let sim = cosine_similarity(&emb1, &emb2);
    assert_eq!(sim, 0.0);

    // Both zero
    let sim2 = cosine_similarity(&emb1, &SpeakerEmbedding::zeros(3));
    assert_eq!(sim2, 0.0);
}

#[test]
fn test_normalize_embedding_zero_vector() {
    let emb = SpeakerEmbedding::zeros(5);
    let normalized = normalize_embedding(&emb);
    assert!(normalized.is_normalized());
    assert_eq!(normalized.l2_norm(), 0.0);
}

#[test]
fn test_average_embeddings_single() {
    let embeddings = vec![SpeakerEmbedding::from_vec(vec![3.0, 4.0])];
    let avg = average_embeddings(&embeddings).expect("single avg");
    assert!((avg.as_slice()[0] - 3.0).abs() < 1e-6);
    assert!((avg.as_slice()[1] - 4.0).abs() < 1e-6);
}

#[test]
fn test_similarity_matrix_empty() {
    let embeddings: Vec<SpeakerEmbedding> = vec![];
    let matrix = similarity_matrix(&embeddings);
    assert!(matrix.is_empty());
}

#[test]
fn test_similarity_matrix_single() {
    let embeddings = vec![SpeakerEmbedding::from_vec(vec![1.0, 0.0])];
    let matrix = similarity_matrix(&embeddings);
    assert_eq!(matrix.len(), 1);
    assert!((matrix[0][0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_ecapa_tdnn_new_direct() {
    let config = EmbeddingConfig {
        embedding_dim: 256,
        ..EmbeddingConfig::default()
    };
    let extractor = EcapaTdnn::new(config);
    assert_eq!(extractor.embedding_dim(), 256);
}

#[test]
fn test_ecapa_tdnn_debug() {
    let extractor = EcapaTdnn::default_config();
    let debug_str = format!("{extractor:?}");
    assert!(!debug_str.is_empty());
}

#[test]
fn test_xvector_new_direct() {
    let config = EmbeddingConfig {
        embedding_dim: 128,
        ..EmbeddingConfig::default()
    };
    let extractor = XVector::new(config);
    assert_eq!(extractor.embedding_dim(), 128);
}

#[test]
fn test_xvector_extract_empty_audio() {
    let extractor = XVector::default_config();
    let result = extractor.extract(&[]);
    assert!(matches!(result, Err(VoiceError::InvalidAudio(_))));
}

#[test]
fn test_xvector_extract_valid_audio() {
    let extractor = XVector::default_config();
    let audio = vec![0.5_f32; 16000];
    let result = extractor.extract(&audio);
    // Stub returns NotImplemented
    assert!(matches!(result, Err(VoiceError::NotImplemented(_))));
}

#[test]
fn test_xvector_debug() {
    let extractor = XVector::default_config();
    let debug_str = format!("{extractor:?}");
    assert!(!debug_str.is_empty());
}

#[test]
fn test_speaker_embedding_dot_self() {
    let emb = SpeakerEmbedding::from_vec(vec![3.0, 4.0]);
    let dot = emb.dot(&emb).expect("same dim");
    // 3*3 + 4*4 = 9 + 16 = 25
    assert!((dot - 25.0).abs() < 1e-6);
}

#[test]
fn test_average_embeddings_three() {
    let embeddings = vec![
        SpeakerEmbedding::from_vec(vec![1.0, 0.0, 0.0]),
        SpeakerEmbedding::from_vec(vec![0.0, 1.0, 0.0]),
        SpeakerEmbedding::from_vec(vec![0.0, 0.0, 1.0]),
    ];
    let avg = average_embeddings(&embeddings).expect("three avg");
    for &v in avg.as_slice() {
        assert!((v - 1.0 / 3.0).abs() < 1e-6);
    }
}

#[test]
fn test_cosine_similarity_negative() {
    let emb1 = SpeakerEmbedding::from_vec(vec![1.0, 1.0]);
    let emb2 = SpeakerEmbedding::from_vec(vec![-1.0, -1.0]);
    let sim = cosine_similarity(&emb1, &emb2);
    assert!((sim - (-1.0)).abs() < 1e-6);
}
