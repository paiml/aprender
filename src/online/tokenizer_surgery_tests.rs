use super::*;

// ============================================================================
// VocabMapping tests
// ============================================================================

#[test]
fn test_compute_vocab_overlap_identical() {
    let tokens: Vec<String> = vec!["hello".into(), "world".into(), "foo".into()];
    let mapping = compute_vocab_overlap(&tokens, &tokens);
    assert_eq!(mapping.overlap_count, 3);
    assert!((mapping.overlap_ratio - 1.0).abs() < 1e-10);
}

#[test]
fn test_compute_vocab_overlap_none() {
    let source: Vec<String> = vec!["a".into(), "b".into()];
    let target: Vec<String> = vec!["x".into(), "y".into()];
    let mapping = compute_vocab_overlap(&source, &target);
    assert_eq!(mapping.overlap_count, 0);
    assert!((mapping.overlap_ratio - 0.0).abs() < 1e-10);
}

#[test]
fn test_compute_vocab_overlap_partial() {
    let source: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
    let target: Vec<String> = vec!["b".into(), "c".into(), "d".into()];
    let mapping = compute_vocab_overlap(&source, &target);
    assert_eq!(mapping.overlap_count, 2);
}

#[test]
fn test_compute_vocab_overlap_empty() {
    let source: Vec<String> = vec![];
    let target: Vec<String> = vec!["a".into()];
    let mapping = compute_vocab_overlap(&source, &target);
    assert_eq!(mapping.overlap_count, 0);
}

// ============================================================================
// Transplant tests
// ============================================================================

#[test]
fn test_transplant_embeddings_direct_copy() {
    let source: Vec<String> = vec!["a".into(), "b".into()];
    let target: Vec<String> = vec!["b".into(), "a".into()];
    let mapping = compute_vocab_overlap(&source, &target);

    let source_emb = vec![1.0, 2.0, 3.0, 4.0]; // 2 tokens, dim=2
    let mut target_emb = vec![0.0; 4];
    let config = TokenizerSurgeryConfig {
        source_vocab_size: 2,
        target_vocab_size: 2,
        overlap_threshold: 0.0,
        method: SurgeryMethod::DirectCopy,
    };

    transplant_embeddings(&source_emb, &mut target_emb, &mapping, &config, 2);

    // target[0] = "b" → source[1] = [3.0, 4.0]
    assert!((target_emb[0] - 3.0).abs() < 1e-6);
    assert!((target_emb[1] - 4.0).abs() < 1e-6);
    // target[1] = "a" → source[0] = [1.0, 2.0]
    assert!((target_emb[2] - 1.0).abs() < 1e-6);
    assert!((target_emb[3] - 2.0).abs() < 1e-6);
}

#[test]
fn test_transplant_embeddings_no_overlap() {
    let source: Vec<String> = vec!["a".into()];
    let target: Vec<String> = vec!["x".into()];
    let mapping = compute_vocab_overlap(&source, &target);

    let source_emb = vec![1.0, 2.0];
    let mut target_emb = vec![0.0; 2];
    let config = TokenizerSurgeryConfig {
        source_vocab_size: 1,
        target_vocab_size: 1,
        overlap_threshold: 0.0,
        method: SurgeryMethod::DirectCopy,
    };

    transplant_embeddings(&source_emb, &mut target_emb, &mapping, &config, 2);
    // No overlap: target should remain zeros
    assert!((target_emb[0] - 0.0).abs() < 1e-6);
}

// ============================================================================
// Validation tests
// ============================================================================

#[test]
fn test_validate_surgery_ok() {
    let source: Vec<String> = vec!["a".into(), "b".into()];
    let target: Vec<String> = vec!["a".into(), "c".into()];
    let mapping = compute_vocab_overlap(&source, &target);
    let config = TokenizerSurgeryConfig {
        source_vocab_size: 2,
        target_vocab_size: 2,
        overlap_threshold: 0.4,
        method: SurgeryMethod::DirectCopy,
    };
    assert!(validate_surgery(&mapping, &config).is_ok());
}

#[test]
fn test_validate_surgery_low_overlap() {
    let source: Vec<String> = vec!["a".into(), "b".into(), "c".into(), "d".into()];
    let target: Vec<String> = vec!["x".into(), "y".into(), "z".into(), "a".into()];
    let mapping = compute_vocab_overlap(&source, &target);
    let config = TokenizerSurgeryConfig {
        source_vocab_size: 4,
        target_vocab_size: 4,
        overlap_threshold: 0.5,
        method: SurgeryMethod::DirectCopy,
    };
    assert!(validate_surgery(&mapping, &config).is_err());
}

// ============================================================================
// SurgeryMethod tests
// ============================================================================

#[test]
fn test_surgery_method_average_pool() {
    let source: Vec<String> = vec!["a".into(), "b".into()];
    let target: Vec<String> = vec!["c".into()]; // no overlap
    let mapping = compute_vocab_overlap(&source, &target);

    let source_emb = vec![1.0, 2.0, 3.0, 4.0]; // 2 tokens, dim=2
    let mut target_emb = vec![0.0; 2];
    let config = TokenizerSurgeryConfig {
        source_vocab_size: 2,
        target_vocab_size: 1,
        overlap_threshold: 0.0,
        method: SurgeryMethod::AveragePool,
    };

    transplant_embeddings(&source_emb, &mut target_emb, &mapping, &config, 2);
    // AveragePool for unmapped: should average all source embeddings
    // avg = [(1+3)/2, (2+4)/2] = [2.0, 3.0]
    assert!((target_emb[0] - 2.0).abs() < 1e-6);
    assert!((target_emb[1] - 3.0).abs() < 1e-6);
}

// ============================================================================
// Falsification tests
// ============================================================================

/// FALSIFY-SURGERY-001: Overlap ratio is in [0, 1].
#[test]
fn falsify_surgery_001_overlap_bounded() {
    let combos: Vec<(Vec<String>, Vec<String>)> = vec![
        (vec![], vec![]),
        (vec!["a".into()], vec![]),
        (vec![], vec!["a".into()]),
        (vec!["a".into(), "b".into()], vec!["b".into(), "c".into()]),
        (vec!["x".into()], vec!["x".into()]),
    ];
    for (src, tgt) in &combos {
        let m = compute_vocab_overlap(src, tgt);
        assert!(
            m.overlap_ratio >= 0.0 && m.overlap_ratio <= 1.0,
            "Overlap ratio {} out of [0,1] for {:?} vs {:?}",
            m.overlap_ratio,
            src,
            tgt
        );
    }
}

/// FALSIFY-SURGERY-002: Transplant preserves embedding dimensions.
#[test]
fn falsify_surgery_002_dimension_preserved() {
    for dim in [1, 2, 4, 8, 16] {
        let source: Vec<String> = vec!["a".into(), "b".into()];
        let target: Vec<String> = vec!["a".into(), "c".into()];
        let mapping = compute_vocab_overlap(&source, &target);

        let source_emb = vec![1.0; 2 * dim];
        let mut target_emb = vec![0.0; 2 * dim];
        let config = TokenizerSurgeryConfig {
            source_vocab_size: 2,
            target_vocab_size: 2,
            overlap_threshold: 0.0,
            method: SurgeryMethod::DirectCopy,
        };

        transplant_embeddings(&source_emb, &mut target_emb, &mapping, &config, dim);
        assert_eq!(
            target_emb.len(),
            2 * dim,
            "Dimension preserved for dim={}",
            dim
        );
        assert!(
            target_emb.iter().all(|x| x.is_finite()),
            "All finite for dim={}",
            dim
        );
    }
}

/// FALSIFY-SURGERY-003: Transplant of identical vocabs is identity.
#[test]
fn falsify_surgery_003_identity() {
    let tokens: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
    let mapping = compute_vocab_overlap(&tokens, &tokens);
    let source_emb: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let mut target_emb = vec![0.0; 6];
    let config = TokenizerSurgeryConfig {
        source_vocab_size: 3,
        target_vocab_size: 3,
        overlap_threshold: 0.0,
        method: SurgeryMethod::DirectCopy,
    };

    transplant_embeddings(&source_emb, &mut target_emb, &mapping, &config, 2);
    for (i, (&s, &t)) in source_emb.iter().zip(target_emb.iter()).enumerate() {
        assert!(
            (s - t).abs() < 1e-10,
            "Identity transplant failed at idx {}",
            i
        );
    }
}
