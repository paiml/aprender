pub(crate) use super::*;

#[test]
fn test_diarization_config_default() {
    let config = DiarizationConfig::default();
    assert!(config.validate().is_ok());
    assert_eq!(config.min_speakers, 1);
    assert!(config.max_speakers.is_none());
}

#[test]
fn test_diarization_config_with_speakers() {
    let config = DiarizationConfig::default().with_speakers(2, Some(4));
    assert_eq!(config.min_speakers, 2);
    assert_eq!(config.max_speakers, Some(4));
}

#[test]
fn test_diarization_config_validation() {
    let mut config = DiarizationConfig::default();
    config.min_speakers = 0;
    assert!(config.validate().is_err());

    config.min_speakers = 2;
    config.max_speakers = Some(1);
    assert!(config.validate().is_err());

    config.max_speakers = Some(3);
    config.clustering_threshold = 1.5;
    assert!(config.validate().is_err());
}

#[test]
fn test_speaker_new() {
    let speaker = Speaker::new(0, vec![1.0, 0.0, 0.0]);
    assert_eq!(speaker.id, 0);
    assert_eq!(speaker.label, "SPEAKER_00");
    assert_eq!(speaker.embedding.len(), 3);
}

#[test]
fn test_speaker_similarity() {
    let s1 = Speaker::new(0, vec![1.0, 0.0, 0.0]);
    let s2 = Speaker::new(1, vec![1.0, 0.0, 0.0]);
    let s3 = Speaker::new(2, vec![0.0, 1.0, 0.0]);

    assert!((s1.similarity(&s2) - 1.0).abs() < 0.001); // Identical
    assert!(s1.similarity(&s3).abs() < 0.001); // Orthogonal
}

#[test]
fn test_speaker_segment_new() {
    let seg = SpeakerSegment::new(0, 1000, 2000);
    assert_eq!(seg.speaker_id, 0);
    assert_eq!(seg.speaker_label, "SPEAKER_00");
    assert_eq!(seg.duration_ms(), 1000);
}

#[test]
fn test_diarization_result_empty() {
    let result = DiarizationResult::new();
    assert!(result.is_empty());
    assert_eq!(result.speaker_count(), 0);
}

#[test]
fn test_cosine_similarity() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let c = vec![0.0, 1.0, 0.0];
    let d = vec![-1.0, 0.0, 0.0];

    assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
    assert!(cosine_similarity(&a, &c).abs() < 0.001);
    assert!((cosine_similarity(&a, &d) + 1.0).abs() < 0.001);
}

#[test]
fn test_cosine_similarity_edge_cases() {
    assert_eq!(cosine_similarity(&[], &[]), 0.0);
    assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0); // Different lengths
    assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0); // Zero vector
}

#[test]
fn test_average_embeddings() {
    let e1 = vec![1.0, 0.0];
    let e2 = vec![0.0, 1.0];
    let embeddings: Vec<&Vec<f32>> = vec![&e1, &e2];

    let avg = average_embeddings(&embeddings, 2);
    assert!((avg[0] - 0.5).abs() < 0.001);
    assert!((avg[1] - 0.5).abs() < 0.001);
}

#[test]
fn test_diarize_empty() {
    let config = DiarizationConfig::default();
    let result = diarize(&[], &[], &config);
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_diarize_single_speaker() {
    let config = DiarizationConfig {
        embedding_dim: 3,
        clustering_threshold: 0.9,
        ..Default::default()
    };

    // Two very similar embeddings = same speaker
    let embeddings = vec![vec![1.0, 0.0, 0.0], vec![0.99, 0.1, 0.0]];
    let times = vec![(0, 1000), (1000, 2000)];

    let result = diarize(&embeddings, &times, &config).unwrap();
    assert_eq!(result.speaker_count(), 1);
    assert_eq!(result.segments.len(), 2);
}

#[test]
fn test_diarize_two_speakers() {
    let config = DiarizationConfig {
        embedding_dim: 3,
        clustering_threshold: 0.5,
        ..Default::default()
    };

    // Two very different embeddings = different speakers
    let embeddings = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
    let times = vec![(0, 1000), (1000, 2000)];

    let result = diarize(&embeddings, &times, &config).unwrap();
    assert_eq!(result.speaker_count(), 2);
}

#[test]
fn test_diarize_speaking_time() {
    let config = DiarizationConfig {
        embedding_dim: 3,
        clustering_threshold: 0.9,
        ..Default::default()
    };

    let embeddings = vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
    let times = vec![(0, 1000), (1000, 3000)];

    let result = diarize(&embeddings, &times, &config).unwrap();
    assert_eq!(result.speaking_time_ms(0), 3000);
}

#[test]
fn test_diarize_mismatched_lengths() {
    let config = DiarizationConfig::default();
    let embeddings = vec![vec![0.0; 192]];
    let times = vec![(0, 1000), (1000, 2000)]; // One extra

    let result = diarize(&embeddings, &times, &config);
    assert!(result.is_err());
}

#[test]
fn test_diarize_wrong_embedding_dim() {
    let config = DiarizationConfig {
        embedding_dim: 192,
        ..Default::default()
    };

    let embeddings = vec![vec![0.0; 100]]; // Wrong dim
    let times = vec![(0, 1000)];

    let result = diarize(&embeddings, &times, &config);
    assert!(result.is_err());
}
