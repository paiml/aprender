//! Bug prediction examples from the book

use verificar::data::CodeFeatures;
use verificar::ml::{BugPredictor, FeatureExtractor};

#[test]
fn test_bug_predictor_example() {
    // Example: Using heuristic bug predictor
    let predictor = BugPredictor::new();
    let features = CodeFeatures {
        ast_depth: 10,
        num_operators: 50,
        num_control_flow: 10,
        cyclomatic_complexity: 15.0,
        uses_edge_values: true,
        ..Default::default()
    };

    let probability = predictor.predict(&features);

    assert!((0.0..=1.0).contains(&probability));
    assert!(probability > 0.5); // Complex code should have higher bug probability
}

#[test]
fn test_feature_extraction_example() {
    // Example: Extracting features from code
    let extractor = FeatureExtractor::new();
    let code = "x = 0\nif x < 1:\n    y = -1";

    let features = extractor.extract(code);

    assert!(features.num_operators > 0);
    assert!(features.num_control_flow > 0);
    assert!(features.uses_edge_values);
}

#[test]
fn test_simple_vs_complex_code_example() {
    // Example: Comparing bug probabilities for simple vs complex code
    let predictor = BugPredictor::new();

    let simple_features = CodeFeatures {
        ast_depth: 3,
        num_operators: 5,
        num_control_flow: 1,
        cyclomatic_complexity: 2.0,
        uses_edge_values: false,
        ..Default::default()
    };

    let complex_features = CodeFeatures {
        ast_depth: 15,
        num_operators: 60,
        num_control_flow: 12,
        cyclomatic_complexity: 20.0,
        uses_edge_values: true,
        ..Default::default()
    };

    let simple_prob = predictor.predict(&simple_features);
    let complex_prob = predictor.predict(&complex_features);

    assert!(complex_prob > simple_prob);
}
