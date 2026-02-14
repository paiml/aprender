use super::*;

#[test]
fn test_lf_output() {
    let abstain = LFOutput::Abstain;
    let label = LFOutput::Label(1);

    assert_eq!(abstain.to_i32(), -1);
    assert_eq!(label.to_i32(), 1);

    assert_eq!(LFOutput::from_i32(-1), LFOutput::Abstain);
    assert_eq!(LFOutput::from_i32(2), LFOutput::Label(2));
}

#[test]
fn test_label_model_creation() {
    let model = LabelModel::new(3, 5);
    assert_eq!(model.n_classes, 3);
    assert_eq!(model.n_lfs, 5);
    assert_eq!(model.class_priors.len(), 3);
}

#[test]
fn test_label_model_fit() {
    let mut model = LabelModel::new(2, 3);

    // Simple test: LFs agree on labels
    let lf_matrix = vec![
        vec![LFOutput::Label(0), LFOutput::Label(0), LFOutput::Abstain],
        vec![LFOutput::Label(1), LFOutput::Label(1), LFOutput::Label(1)],
        vec![LFOutput::Label(0), LFOutput::Abstain, LFOutput::Label(0)],
        vec![LFOutput::Abstain, LFOutput::Label(1), LFOutput::Label(1)],
    ];

    model.fit(&lf_matrix, 10, 0.5);

    // Should have learned something
    assert!(model.accuracies[0][0] > 0.0);
}

#[test]
fn test_label_model_predict() {
    let mut model = LabelModel::new(2, 2);
    model.accuracies = vec![vec![0.9, 0.9], vec![0.8, 0.8]];

    let lf_matrix = vec![
        vec![LFOutput::Label(0), LFOutput::Label(0)],
        vec![LFOutput::Label(1), LFOutput::Label(1)],
    ];

    let predictions = model.predict(&lf_matrix);

    assert_eq!(predictions.len(), 2);
    assert_eq!(predictions[0], 0);
    assert_eq!(predictions[1], 1);
}

#[test]
fn test_label_model_proba() {
    let mut model = LabelModel::new(2, 2);
    model.accuracies = vec![vec![0.9, 0.9], vec![0.8, 0.8]];

    let lf_matrix = vec![vec![LFOutput::Label(0), LFOutput::Label(0)]];

    let probs = model.predict_proba(&lf_matrix);

    assert_eq!(probs.len(), 1);
    assert_eq!(probs[0].len(), 2);

    // Sum should be 1
    let sum: f32 = probs[0].as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // Class 0 should be more likely
    assert!(probs[0][0] > probs[0][1]);
}

#[test]
fn test_label_model_coverage() {
    let model = LabelModel::new(2, 3);

    let lf_matrix = vec![
        vec![LFOutput::Label(0), LFOutput::Label(0), LFOutput::Abstain],
        vec![LFOutput::Label(1), LFOutput::Abstain, LFOutput::Abstain],
        vec![LFOutput::Abstain, LFOutput::Label(1), LFOutput::Label(1)],
        vec![LFOutput::Label(0), LFOutput::Label(0), LFOutput::Label(0)],
    ];

    let coverage = model.get_lf_coverage(&lf_matrix);

    assert_eq!(coverage.len(), 3);
    assert!((coverage[0] - 0.75).abs() < 1e-5); // 3/4
    assert!((coverage[1] - 0.75).abs() < 1e-5); // 3/4
    assert!((coverage[2] - 0.5).abs() < 1e-5); // 2/4
}

#[test]
fn test_confident_learning_creation() {
    let cl = ConfidentLearning::new(3);
    assert_eq!(cl.n_classes, 3);
    assert!((cl.threshold - 0.5).abs() < 1e-10);

    let cl2 = ConfidentLearning::with_threshold(3, 0.7);
    assert!((cl2.threshold - 0.7).abs() < 1e-10);
}

#[test]
fn test_confident_learning_find_issues() {
    let cl = ConfidentLearning::with_threshold(2, 0.6);

    let labels = vec![0, 1, 0, 1];
    let pred_probs = vec![
        Vector::from_slice(&[0.9, 0.1]), // Correct, confident
        Vector::from_slice(&[0.8, 0.2]), // Mislabeled (pred=0, label=1)
        Vector::from_slice(&[0.4, 0.6]), // Mislabeled (pred=1, label=0)
        Vector::from_slice(&[0.3, 0.7]), // Correct, confident
    ];

    let issues = cl.find_label_issues(&labels, &pred_probs);

    assert!(issues.contains(&1)); // Mislabeled
    assert!(issues.contains(&2)); // Mislabeled
    assert!(!issues.contains(&0)); // Correct
    assert!(!issues.contains(&3)); // Correct
}

#[test]
fn test_confident_joint() {
    let cl = ConfidentLearning::new(2);

    let labels = vec![0, 0, 1, 1];
    let pred_probs = vec![
        Vector::from_slice(&[0.9, 0.1]),
        Vector::from_slice(&[0.8, 0.2]),
        Vector::from_slice(&[0.2, 0.8]),
        Vector::from_slice(&[0.3, 0.7]),
    ];

    let joint = cl.compute_confident_joint(&labels, &pred_probs);

    assert_eq!(joint.len(), 2);
    assert_eq!(joint[0].len(), 2);

    // Sum should be 1
    let total: f32 = joint.iter().flat_map(|r| r.iter()).sum();
    assert!((total - 1.0).abs() < 0.1 || total == 0.0);
}

#[test]
fn test_noise_matrix_estimation() {
    let cl = ConfidentLearning::new(2);

    let labels = vec![0, 0, 1, 1, 0, 1];
    let pred_probs = vec![
        Vector::from_slice(&[0.9, 0.1]),
        Vector::from_slice(&[0.8, 0.2]),
        Vector::from_slice(&[0.1, 0.9]),
        Vector::from_slice(&[0.2, 0.8]),
        Vector::from_slice(&[0.7, 0.3]),
        Vector::from_slice(&[0.3, 0.7]),
    ];

    let noise = cl.estimate_noise_matrix(&labels, &pred_probs);

    // Each column should sum to ~1
    for j in 0..2 {
        let col_sum: f32 = (0..2).map(|i| noise[i][j]).sum();
        assert!((col_sum - 1.0).abs() < 0.1 || col_sum == 0.0);
    }
}
