use super::*;

#[test]
fn test_uncertainty_sampling() {
    let strategy = UncertaintySampling::new();
    let preds = vec![
        Vector::from_slice(&[0.9, 0.1]), // Confident
        Vector::from_slice(&[0.5, 0.5]), // Uncertain
        Vector::from_slice(&[0.7, 0.3]), // Medium
    ];

    let scores = strategy.score(&preds);
    assert!(scores[1] > scores[0]); // More uncertain = higher score
    assert!(scores[1] > scores[2]);
}

#[test]
fn test_uncertainty_select() {
    let strategy = UncertaintySampling::new();
    let preds = vec![
        Vector::from_slice(&[0.9, 0.1]),
        Vector::from_slice(&[0.5, 0.5]),
        Vector::from_slice(&[0.6, 0.4]),
    ];

    let selected = strategy.select(&preds, 2);
    assert_eq!(selected.len(), 2);
    assert!(selected.contains(&1)); // Most uncertain
}

#[test]
fn test_margin_sampling() {
    let strategy = MarginSampling::new();
    let preds = vec![
        Vector::from_slice(&[0.9, 0.1]),   // Large margin
        Vector::from_slice(&[0.51, 0.49]), // Small margin
    ];

    let scores = strategy.score(&preds);
    assert!(scores[1] > scores[0]); // Smaller margin = higher score
}

#[test]
fn test_entropy_sampling() {
    let strategy = EntropySampling::new();
    let preds = vec![
        Vector::from_slice(&[1.0, 0.0]), // Zero entropy
        Vector::from_slice(&[0.5, 0.5]), // Max entropy (2 classes)
    ];

    let scores = strategy.score(&preds);
    assert!(scores[1] > scores[0]);
}

#[test]
fn test_query_by_committee() {
    let qbc = QueryByCommittee::new(3);
    assert_eq!(qbc.n_members(), 3);

    // 3 members, 2 samples
    let committee = vec![
        vec![
            Vector::from_slice(&[0.9, 0.1]),
            Vector::from_slice(&[0.1, 0.9]),
        ],
        vec![
            Vector::from_slice(&[0.8, 0.2]),
            Vector::from_slice(&[0.9, 0.1]),
        ],
        vec![
            Vector::from_slice(&[0.7, 0.3]),
            Vector::from_slice(&[0.2, 0.8]),
        ],
    ];

    let scores = qbc.score_committee(&committee);
    assert_eq!(scores.len(), 2);
    // Sample 1 has disagreement (votes: [1, 2]), sample 0 agrees (votes: [3, 0])
    assert!(scores[1] > scores[0]);
}

#[test]
fn test_random_sampling() {
    let strategy = RandomSampling::new();
    let selected = strategy.select(10, 3);

    assert_eq!(selected.len(), 3);
    for &idx in &selected {
        assert!(idx < 10);
    }
}

#[test]
fn test_select_more_than_available() {
    let strategy = UncertaintySampling::new();
    let preds = vec![
        Vector::from_slice(&[0.5, 0.5]),
        Vector::from_slice(&[0.6, 0.4]),
    ];

    let selected = strategy.select(&preds, 5);
    assert_eq!(selected.len(), 2); // Only 2 available
}

// Core-Set Selection Tests
#[test]
fn test_coreset_new() {
    let cs = CoreSetSelection::new();
    assert!(cs.labeled_indices.is_empty());
}

#[test]
fn test_coreset_with_labeled() {
    let cs = CoreSetSelection::with_labeled(vec![0, 1]);
    assert_eq!(cs.labeled_indices, vec![0, 1]);
}

#[test]
fn test_coreset_select() {
    let cs = CoreSetSelection::new();

    // 4 points in 2D forming a square
    let embeddings = vec![
        vec![0.0, 0.0], // 0: bottom-left
        vec![1.0, 0.0], // 1: bottom-right
        vec![0.0, 1.0], // 2: top-left
        vec![1.0, 1.0], // 3: top-right
    ];

    // When no labeled set, first point is added, then k more are selected
    let selected = cs.select(&embeddings, 3);

    // Should select diverse points (first point + up to 3 more)
    assert!(selected.len() >= 3 && selected.len() <= 4);
    for &idx in &selected {
        assert!(idx < 4);
    }
}

#[test]
fn test_coreset_respects_labeled() {
    let cs = CoreSetSelection::with_labeled(vec![0]);

    let embeddings = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];

    let selected = cs.select(&embeddings, 2);

    // Should not include already labeled point
    assert!(!selected.contains(&0));
    assert_eq!(selected.len(), 2);
}

#[test]
fn test_coreset_diversity_score() {
    let cs = CoreSetSelection::new();

    let embeddings = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![10.0, 0.0]];

    // Closer points should have lower diversity
    let close = cs.diversity_score(&embeddings, &[0, 1]);
    let far = cs.diversity_score(&embeddings, &[0, 2]);

    assert!(far > close, "Farther points should have higher diversity");
}

#[test]
fn test_coreset_empty() {
    let cs = CoreSetSelection::new();
    let selected = cs.select(&[], 5);
    assert!(selected.is_empty());
}

// Expected Model Change Tests
#[test]
fn test_emc_new() {
    let emc = ExpectedModelChange::new();
    assert!((emc.min_grad_norm - 0.0).abs() < 1e-10);
}

#[test]
fn test_emc_score() {
    let emc = ExpectedModelChange::new();
    let preds = vec![
        Vector::from_slice(&[1.0, 0.0]), // Certain
        Vector::from_slice(&[0.5, 0.5]), // Uncertain
    ];

    let scores = emc.score(&preds, None);

    // Uncertain sample should have higher score (entropy)
    assert!(scores[1] > scores[0]);
}

#[test]
fn test_emc_score_with_grads() {
    let emc = ExpectedModelChange::new();
    let preds = vec![
        Vector::from_slice(&[1.0, 0.0]),
        Vector::from_slice(&[0.5, 0.5]),
    ];
    let grads = vec![0.5, 2.0];

    let scores = emc.score(&preds, Some(&grads));

    assert!((scores[0] - 0.5).abs() < 1e-6);
    assert!((scores[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_emc_select() {
    let emc = ExpectedModelChange::new();
    let preds = vec![
        Vector::from_slice(&[0.9, 0.1]), // Low entropy
        Vector::from_slice(&[0.5, 0.5]), // High entropy
        Vector::from_slice(&[0.7, 0.3]), // Medium entropy
    ];

    let selected = emc.select(&preds, 2);

    assert_eq!(selected.len(), 2);
    assert!(selected.contains(&1)); // Highest entropy
}

#[test]
fn test_emc_with_threshold() {
    let emc = ExpectedModelChange::with_min_grad(1.0);
    let preds = vec![Vector::from_slice(&[0.5, 0.5])];
    let grads = vec![0.5]; // Below threshold

    let scores = emc.score(&preds, Some(&grads));
    assert!((scores[0] - 0.0).abs() < 1e-6); // Should be filtered
}
