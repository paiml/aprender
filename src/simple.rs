
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple linear model for testing
    struct LinearModel {
        weights: Vec<f32>,
        bias: f32,
    }

    impl LinearModel {
        fn new(weights: Vec<f32>, bias: f32) -> Self {
            Self { weights, bias }
        }
    }

    impl Explainable for LinearModel {
        fn predict(&self, instance: &[f32]) -> Result<f32, ExplainError> {
            if instance.len() != self.weights.len() {
                return Err(ExplainError::InvalidInput {
                    expected: self.weights.len(),
                    actual: instance.len(),
                });
            }
            let dot: f32 = instance.iter().zip(&self.weights).map(|(x, w)| x * w).sum();
            Ok(dot + self.bias)
        }

        fn n_features(&self) -> usize {
            self.weights.len()
        }
    }

    /// Simple tree model for testing
    struct SimpleTreeModel {
        structure: TreeStructure,
    }

    impl SimpleTreeModel {
        fn new(tree: DecisionTree) -> Self {
            let n_features = tree
                .feature
                .iter()
                .filter(|&&f| f >= 0)
                .map(|&f| f as usize + 1)
                .max()
                .unwrap_or(1);
            Self {
                structure: TreeStructure {
                    n_trees: 1,
                    n_features,
                    trees: vec![tree],
                },
            }
        }
    }

    impl Explainable for SimpleTreeModel {
        fn predict(&self, instance: &[f32]) -> Result<f32, ExplainError> {
            let sum: f32 = self
                .structure
                .trees
                .iter()
                .map(|t| t.predict(instance))
                .sum();
            Ok(sum / self.structure.n_trees as f32)
        }

        fn n_features(&self) -> usize {
            self.structure.n_features
        }

        fn is_tree_model(&self) -> bool {
            true
        }

        fn get_tree_structure(&self) -> Option<&TreeStructure> {
            Some(&self.structure)
        }
    }

    // === ShapExplanation Tests ===

    #[test]
    fn test_shap_explanation_new() {
        let exp = ShapExplanation::new(0.5, vec![0.1, -0.2, 0.3], 0.7);
        assert_eq!(exp.base_value, 0.5);
        assert_eq!(exp.shap_values.len(), 3);
        assert_eq!(exp.prediction, 0.7);
        assert_eq!(exp.feature_names.len(), 3);
    }

    #[test]
    fn test_shap_explanation_with_feature_names() {
        let exp = ShapExplanation::new(0.5, vec![0.1, -0.2], 0.4)
            .with_feature_names(vec!["age".to_string(), "income".to_string()]);
        assert_eq!(exp.feature_names, vec!["age", "income"]);
    }

    #[test]
    fn test_shap_explanation_top_features() {
        let exp = ShapExplanation::new(0.0, vec![0.1, -0.3, 0.2], 0.0).with_feature_names(vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
        ]);

        let top = exp.top_features(2);
        assert_eq!(top.len(), 2);
        // Should be sorted by absolute value
        assert_eq!(top[0].0, "b"); // |-0.3| = 0.3
        assert_eq!(top[1].0, "c"); // |0.2| = 0.2
    }

    #[test]
    fn test_shap_explanation_verify_consistency() {
        // Perfect consistency
        let exp = ShapExplanation::new(0.5, vec![0.2, 0.3], 1.0);
        assert!(exp.verify_consistency(0.01));

        // Not consistent
        let exp_bad = ShapExplanation::new(0.5, vec![0.2, 0.3], 2.0);
        assert!(!exp_bad.verify_consistency(0.01));
    }

    #[test]
    fn test_shap_explanation_display() {
        let exp = ShapExplanation::new(0.5, vec![0.1, -0.2], 0.4);
        let display = format!("{exp}");
        assert!(display.contains("SHAP Explanation"));
        assert!(display.contains("Base value"));
        assert!(display.contains("Prediction"));
    }

    // === Decision Tree Tests ===

    #[test]
    fn test_decision_tree_predict_simple() {
        // Simple tree: if feature_0 <= 0.5 then 1.0 else 2.0
        let tree = DecisionTree::new(
            vec![0, -1, -1],     // feature indices (-1 = leaf)
            vec![0.5, 0.0, 0.0], // thresholds
            vec![1, 0, 0],       // left children
            vec![2, 0, 0],       // right children
            vec![0.0, 1.0, 2.0], // values
        );

        assert_eq!(tree.predict(&[0.3]), 1.0); // <= 0.5, go left
        assert_eq!(tree.predict(&[0.7]), 2.0); // > 0.5, go right
    }

    #[test]
    fn test_decision_tree_n_nodes() {
        let tree = DecisionTree::new(
            vec![0, -1, -1],
            vec![0.5, 0.0, 0.0],
            vec![1, 0, 0],
            vec![2, 0, 0],
            vec![0.0, 1.0, 2.0],
        );
        assert_eq!(tree.n_nodes(), 3);
    }

    #[test]
    fn test_decision_tree_is_leaf() {
        let tree = DecisionTree::new(
            vec![0, -1, -1],
            vec![0.5, 0.0, 0.0],
            vec![1, 0, 0],
            vec![2, 0, 0],
            vec![0.0, 1.0, 2.0],
        );
        assert!(!tree.is_leaf(0)); // root is not leaf
        assert!(tree.is_leaf(1)); // leaf
        assert!(tree.is_leaf(2)); // leaf
    }

    // === ShapExplainer Tests ===

    #[test]
    fn test_shap_explainer_new() {
        let background = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let explainer = ShapExplainer::new(background);
        assert_eq!(explainer.nsamples, 100);
        assert_eq!(explainer.feature_names.len(), 2);
    }

    #[test]
    fn test_shap_explainer_with_nsamples() {
        let explainer = ShapExplainer::new(vec![vec![1.0]]).with_nsamples(50);
        assert_eq!(explainer.nsamples, 50);
    }

    #[test]
    fn test_shap_explainer_with_feature_names() {
        let explainer = ShapExplainer::new(vec![vec![1.0, 2.0]])
            .with_feature_names(vec!["age".to_string(), "income".to_string()]);
        assert_eq!(explainer.feature_names, vec!["age", "income"]);
    }

    #[test]
    fn test_shap_explainer_linear_model() {
        let model = LinearModel::new(vec![1.0, 2.0], 0.0);
        let background = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let explainer = ShapExplainer::new(background);

        let instance = vec![2.0, 3.0]; // prediction = 2*1 + 3*2 = 8
        let explanation = explainer.explain(&model, &instance).expect("test");

        assert_eq!(explanation.prediction, 8.0);
        assert_eq!(explanation.shap_values.len(), 2);
    }

    #[test]
    fn test_shap_explainer_tree_model() {
        // Simple tree: if feature_0 <= 0.5 then 1.0 else 2.0
        let tree = DecisionTree::new(
            vec![0, -1, -1],
            vec![0.5, 0.0, 0.0],
            vec![1, 0, 0],
            vec![2, 0, 0],
            vec![0.0, 1.0, 2.0],
        );
        let model = SimpleTreeModel::new(tree);

        let background = vec![vec![0.3], vec![0.7]];
        let explainer = ShapExplainer::new(background);

        let explanation = explainer.explain(&model, &[0.3]).expect("test");
        assert_eq!(explanation.prediction, 1.0);
    }

    #[test]
    fn test_shap_explainer_invalid_input() {
        let model = LinearModel::new(vec![1.0, 2.0], 0.0);
        let background = vec![vec![0.0, 0.0]];
        let explainer = ShapExplainer::new(background);

        let result = explainer.explain(&model, &[1.0, 2.0, 3.0]); // 3 features, model expects 2
        assert!(matches!(result, Err(ExplainError::InvalidInput { .. })));
    }

    #[test]
    fn test_shap_explainer_empty_background() {
        let model = LinearModel::new(vec![1.0], 0.0);
        let explainer = ShapExplainer::new(vec![]); // Empty background

        let result = explainer.explain(&model, &[1.0]);
        assert!(matches!(result, Err(ExplainError::NoBackground)));
    }

    // === ExplainError Tests ===

    #[test]
    fn test_explain_error_display() {
        let err = ExplainError::UnsupportedModel {
            reason: "not a tree".to_string(),
        };
        assert!(err.to_string().contains("not a tree"));

        let err = ExplainError::InvalidInput {
            expected: 3,
            actual: 2,
        };
        assert!(err.to_string().contains("expected 3"));
        assert!(err.to_string().contains("got 2"));

        let err = ExplainError::NoBackground;
        assert!(err.to_string().contains("Background"));

        let err = ExplainError::ComputationError("overflow".to_string());
        assert!(err.to_string().contains("overflow"));
    }

    // === Binomial Tests ===

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 1), 5);
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(5, 3), 10);
        assert_eq!(binomial(5, 4), 5);
        assert_eq!(binomial(5, 5), 1);
        assert_eq!(binomial(5, 6), 0); // k > n
    }

    #[test]
    fn test_binomial_edge_cases() {
        assert_eq!(binomial(0, 0), 1);
        assert_eq!(binomial(1, 0), 1);
        assert_eq!(binomial(1, 1), 1);
        assert_eq!(binomial(10, 5), 252);
    }

    // === TreeStructure Tests ===

    #[test]
    fn test_tree_structure_serialization() {
        let tree = DecisionTree::new(
            vec![0, -1, -1],
            vec![0.5, 0.0, 0.0],
            vec![1, 0, 0],
            vec![2, 0, 0],
            vec![0.0, 1.0, 2.0],
        );
        let structure = TreeStructure {
            n_trees: 1,
            n_features: 1,
            trees: vec![tree],
        };

        let json = serde_json::to_string(&structure).expect("test");
        assert!(json.contains("n_trees"));
        assert!(json.contains("n_features"));
    }

    #[test]
    fn test_shap_explanation_serialization() {
        let exp = ShapExplanation::new(0.5, vec![0.1, -0.2], 0.4);
        let json = serde_json::to_string(&exp).expect("test");
        let parsed: ShapExplanation = serde_json::from_str(&json).expect("test");

        assert_eq!(parsed.base_value, exp.base_value);
        assert_eq!(parsed.shap_values, exp.shap_values);
        assert_eq!(parsed.prediction, exp.prediction);
    }

    // === Additional Edge Case Tests ===

    #[test]
    fn test_shap_explanation_empty_values() {
        let exp = ShapExplanation::new(0.5, vec![], 0.5);
        assert!(exp.verify_consistency(0.01));
        assert!(exp.top_features(3).is_empty());
    }

    #[test]
    fn test_decision_tree_empty_instance() {
        let tree = DecisionTree::new(
            vec![-1], // Just a leaf
            vec![0.0],
            vec![0],
            vec![0],
            vec![5.0],
        );
        assert_eq!(tree.predict(&[]), 5.0);
    }

    #[test]
    fn test_linear_model_batch_predict() {
        let model = LinearModel::new(vec![1.0, 2.0], 0.0);
        let instances = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        let predictions = model.predict_batch(&instances).expect("test");
        assert_eq!(predictions, vec![3.0, 6.0]);
    }
}
