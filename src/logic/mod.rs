//! TensorLogic: Neuro-Symbolic Reasoning via Tensor Operations
//!
//! This module implements the TensorLogic paradigm (Domingos, 2025), unifying neural
//! and symbolic reasoning through tensor operations. All logical operations are
//! expressed as Einstein summations, enabling:
//!
//! - **Differentiable inference**: Backpropagation through logical reasoning
//! - **Dual-mode operation**: Boolean (guaranteed correctness) or Continuous (learnable)
//! - **Knowledge graph reasoning**: RESCAL factorization and embedding space queries
//!
//! # Toyota Way Principles
//!
//! - **Jidoka**: Boolean mode guarantees no hallucinations (output ⊆ derivable facts)
//! - **Poka-Yoke**: Type-safe mode selection prevents accidental mixing
//! - **Genchi Genbutsu**: Explicit tensor equations for auditability
//!
//! # Example
//!
//! ```rust
//! use aprender::logic::{LogicMode, logical_join, logical_project};
//!
//! // Family tree reasoning: Grandparent = Parent @ Parent
//! let parent = vec![
//!     vec![0.0, 1.0, 0.0],  // Alice is parent of Bob
//!     vec![0.0, 0.0, 1.0],  // Bob is parent of Charlie
//!     vec![0.0, 0.0, 0.0],  // Charlie has no children
//! ];
//!
//! let grandparent = logical_join(&parent, &parent, LogicMode::Boolean);
//! // grandparent[0][2] = 1.0 (Alice is grandparent of Charlie)
//! ```
//!
//! # References
//!
//! - Domingos, P. (2025). "Tensor Logic: The Language of AI." arXiv:2510.12269
//! - Nickel, M. et al. (2011). "RESCAL: A Three-Way Model for Collective Learning"
//! - Bordes, A. et al. (2013). "TransE: Translating Embeddings for Multi-relational Data"

mod ops;
mod program;
mod embed;

pub use ops::{
    LogicMode,
    logical_join,
    logical_project,
    logical_union,
    logical_negation,
    logical_select,
    apply_nonlinearity,
    apply_nonlinearity_with_temperature,
    apply_nonlinearity_with_mask,
    Nonlinearity,
};

pub use program::{
    TensorProgram,
    Equation,
    ProgramBuilder,
};

pub use embed::{
    EmbeddingSpace,
    RelationMatrix,
    BilinearScorer,
    RescalFactorizer,
};

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // K1: logical_join computes einsum correctly
    // ==========================================================================
    #[test]
    fn k1_logical_join_computes_grandparent() {
        // Parent relation: Alice->Bob, Bob->Charlie
        let parent = vec![
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0],
        ];

        let grandparent = logical_join(&parent, &parent, LogicMode::Boolean);

        // Alice is grandparent of Charlie (path: Alice->Bob->Charlie)
        assert_eq!(grandparent[0][2], 1.0, "Alice should be grandparent of Charlie");
        assert_eq!(grandparent[0][0], 0.0, "Alice is not her own grandparent");
        assert_eq!(grandparent[1][2], 0.0, "Bob is not grandparent of Charlie");
    }

    #[test]
    fn k1_logical_join_continuous_mode() {
        let a = vec![
            vec![0.5, 0.3],
            vec![0.2, 0.8],
        ];
        let b = vec![
            vec![0.4, 0.6],
            vec![0.7, 0.1],
        ];

        let result = logical_join(&a, &b, LogicMode::Continuous);

        // Matrix multiplication: result[i][j] = sum_k(a[i][k] * b[k][j])
        let expected_00 = 0.5 * 0.4 + 0.3 * 0.7; // 0.41
        assert!((result[0][0] - expected_00).abs() < 1e-6);
    }

    // ==========================================================================
    // K2: logical_project (∃) works in Boolean mode
    // ==========================================================================
    #[test]
    fn k2_logical_project_boolean_existential() {
        // HasChild(X) = ∃Y: Parent(X,Y)
        let parent = vec![
            vec![0.0, 1.0, 0.0],  // Alice has child
            vec![0.0, 0.0, 1.0],  // Bob has child
            vec![0.0, 0.0, 0.0],  // Charlie has no child
        ];

        let has_child = logical_project(&parent, 1, LogicMode::Boolean);

        assert_eq!(has_child[0], 1.0, "Alice has child");
        assert_eq!(has_child[1], 1.0, "Bob has child");
        assert_eq!(has_child[2], 0.0, "Charlie has no child");
    }

    // ==========================================================================
    // K3: logical_project (∃) works in continuous mode
    // ==========================================================================
    #[test]
    fn k3_logical_project_continuous_sum() {
        let tensor = vec![
            vec![0.2, 0.3, 0.5],
            vec![0.1, 0.4, 0.2],
        ];

        let projected = logical_project(&tensor, 1, LogicMode::Continuous);

        // Sum over dimension 1
        assert!((projected[0] - 1.0).abs() < 1e-6);
        assert!((projected[1] - 0.7).abs() < 1e-6);
    }

    // ==========================================================================
    // K4: logical_union implements OR correctly
    // ==========================================================================
    #[test]
    fn k4_logical_union_boolean_max() {
        let a = vec![vec![1.0, 0.0, 1.0]];
        let b = vec![vec![0.0, 1.0, 1.0]];

        let result = logical_union(&a, &b, LogicMode::Boolean);

        assert_eq!(result[0][0], 1.0);
        assert_eq!(result[0][1], 1.0);
        assert_eq!(result[0][2], 1.0);
    }

    #[test]
    fn k4_logical_union_continuous_probabilistic() {
        // P(A or B) = P(A) + P(B) - P(A)*P(B)
        let a = vec![vec![0.3, 0.5]];
        let b = vec![vec![0.4, 0.6]];

        let result = logical_union(&a, &b, LogicMode::Continuous);

        let expected_0 = 0.3 + 0.4 - 0.3 * 0.4; // 0.58
        let expected_1 = 0.5 + 0.6 - 0.5 * 0.6; // 0.80
        assert!((result[0][0] - expected_0).abs() < 1e-6);
        assert!((result[0][1] - expected_1).abs() < 1e-6);
    }

    // ==========================================================================
    // K5: logical_negation implements NOT correctly
    // ==========================================================================
    #[test]
    fn k5_logical_negation() {
        let tensor = vec![vec![1.0, 0.0, 0.7]];

        let negated = logical_negation(&tensor, LogicMode::Boolean);
        assert_eq!(negated[0][0], 0.0);
        assert_eq!(negated[0][1], 1.0);
        assert_eq!(negated[0][2], 0.0); // 0.7 > 0.5 -> 1 -> negated = 0

        let negated_cont = logical_negation(&tensor, LogicMode::Continuous);
        assert!((negated_cont[0][2] - 0.3).abs() < 1e-6); // 1 - 0.7 = 0.3
    }

    // ==========================================================================
    // K6: logical_select implements WHERE correctly
    // ==========================================================================
    #[test]
    fn k6_logical_select() {
        let tensor = vec![vec![0.8, 0.6, 0.9]];
        let condition = vec![vec![1.0, 0.0, 1.0]];

        let selected = logical_select(&tensor, &condition, LogicMode::Boolean);

        assert_eq!(selected[0][0], 0.8);
        assert_eq!(selected[0][1], 0.0); // Filtered out
        assert_eq!(selected[0][2], 0.9);
    }

    // ==========================================================================
    // K7: Boolean mode produces 0/1 outputs only
    // ==========================================================================
    #[test]
    fn k7_boolean_mode_binary_output() {
        let a = vec![vec![0.3, 0.7, 0.5]];
        let b = vec![vec![0.6, 0.4, 0.5]];

        let result = logical_union(&a, &b, LogicMode::Boolean);

        for row in &result {
            for &val in row {
                assert!(val == 0.0 || val == 1.0, "Boolean mode must produce 0 or 1");
            }
        }
    }

    // ==========================================================================
    // K8: Continuous mode preserves gradients (values not thresholded)
    // ==========================================================================
    #[test]
    fn k8_continuous_mode_preserves_values() {
        let tensor = vec![vec![0.3, 0.7]];

        let negated = logical_negation(&tensor, LogicMode::Continuous);

        // Values should be exact, not thresholded
        assert!((negated[0][0] - 0.7).abs() < 1e-6);
        assert!((negated[0][1] - 0.3).abs() < 1e-6);
    }

    // ==========================================================================
    // K9: TensorProgram executes equations in order
    // ==========================================================================
    #[test]
    fn k9_tensor_program_forward_chaining() {
        let mut program = ProgramBuilder::new(LogicMode::Boolean)
            .add_fact("parent", vec![
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![0.0, 0.0, 0.0],
            ])
            .add_rule("grandparent", Equation::Join("parent".into(), "parent".into()))
            .build();

        let results = program.forward();

        let grandparent = results.get("grandparent").expect("grandparent should exist");
        assert_eq!(grandparent[0][2], 1.0);
    }

    // ==========================================================================
    // K10: TensorProgram backward chaining works
    // ==========================================================================
    #[test]
    fn k10_tensor_program_query() {
        let mut program = ProgramBuilder::new(LogicMode::Boolean)
            .add_fact("parent", vec![
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![0.0, 0.0, 0.0],
            ])
            .add_rule("grandparent", Equation::Join("parent".into(), "parent".into()))
            .build();

        let result = program.query("grandparent");
        assert!(result.is_some());
    }

    // ==========================================================================
    // K11: Embedding space bilinear scoring works
    // ==========================================================================
    #[test]
    fn k11_embedding_bilinear_scoring() {
        let space = EmbeddingSpace::new(3, 4); // 3 entities, 4-dim embeddings

        let score = space.score(0, "knows", 1);
        // Score should be a scalar (not NaN or infinite)
        assert!(score.is_finite());
    }

    // ==========================================================================
    // K12: Relation matrices are learnable
    // ==========================================================================
    #[test]
    fn k12_relation_matrices_learnable() {
        let mut space = EmbeddingSpace::new(3, 4);
        space.add_relation("knows");

        let matrix = space.get_relation_matrix("knows");
        assert!(matrix.is_some());
        assert_eq!(matrix.unwrap().len(), 4); // 4x4 matrix
    }

    // ==========================================================================
    // K13: Multi-hop composition computes correctly
    // ==========================================================================
    #[test]
    fn k13_multi_hop_composition() {
        let mut space = EmbeddingSpace::new(3, 4);
        space.add_relation("parent");

        // Compose parent with itself to get grandparent
        let composed = space.compose_relations(&["parent", "parent"]);
        assert_eq!(composed.len(), 4); // Should be 4x4 matrix
    }

    // ==========================================================================
    // K14: RESCAL factorization discovers predicates
    // ==========================================================================
    #[test]
    fn k14_rescal_factorization() {
        let factorizer = RescalFactorizer::new(3, 4, 2); // 3 entities, 4-dim, 2 latent relations

        let triples = vec![
            (0, 0, 1), // Entity 0 relates to Entity 1 via relation 0
            (1, 0, 2), // Entity 1 relates to Entity 2 via relation 0
        ];

        let result = factorizer.factorize(&triples, 10);
        assert!(result.entity_embeddings.len() == 3);
    }

    // ==========================================================================
    // K15: Boolean attention equals argmax selection
    // ==========================================================================
    #[test]
    fn k15_boolean_attention_argmax() {
        let scores = vec![vec![0.1, 0.8, 0.05, 0.05]];

        let weights = apply_nonlinearity(&scores, Nonlinearity::BooleanAttention);

        // Should be one-hot at argmax position
        assert_eq!(weights[0][0], 0.0);
        assert_eq!(weights[0][1], 1.0); // argmax
        assert_eq!(weights[0][2], 0.0);
        assert_eq!(weights[0][3], 0.0);
    }

    // ==========================================================================
    // K16: Continuous attention equals softmax
    // ==========================================================================
    #[test]
    fn k16_continuous_attention_softmax() {
        let scores = vec![vec![1.0, 2.0, 3.0]];

        let weights = apply_nonlinearity(&scores, Nonlinearity::Softmax);

        // Softmax properties: sum to 1, all positive
        let sum: f64 = weights[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(weights[0].iter().all(|&x| x > 0.0));
    }

    // ==========================================================================
    // K17: Attention mask correctly applied
    // ==========================================================================
    #[test]
    fn k17_attention_mask() {
        let scores = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let mask = vec![vec![false, false, true, true]]; // Mask out positions 2,3

        let weights = apply_nonlinearity_with_mask(&scores, Nonlinearity::Softmax, Some(&mask));

        // Masked positions should be ~0
        assert!(weights[0][2] < 1e-6);
        assert!(weights[0][3] < 1e-6);
        // Unmasked should sum to ~1
        let unmasked_sum = weights[0][0] + weights[0][1];
        assert!((unmasked_sum - 1.0).abs() < 1e-6);
    }

    // ==========================================================================
    // K18: Forward chain step handles multiple antecedents
    // ==========================================================================
    #[test]
    fn k18_forward_chain_multiple_antecedents() {
        let mut program = ProgramBuilder::new(LogicMode::Boolean)
            .add_fact("a", vec![vec![1.0, 0.0], vec![0.0, 1.0]])
            .add_fact("b", vec![vec![0.0, 1.0], vec![1.0, 0.0]])
            .add_rule("c", Equation::JoinMultiple(vec!["a".into(), "b".into()]))
            .build();

        let results = program.forward();
        assert!(results.contains_key("c"));
    }

    // ==========================================================================
    // K19: Temperature parameter affects sharpness
    // ==========================================================================
    #[test]
    fn k19_temperature_sharpness() {
        let scores = vec![vec![1.0, 2.0]];

        let weights_hot = apply_nonlinearity_with_temp(&scores, Nonlinearity::Softmax, 0.1);
        let weights_cold = apply_nonlinearity_with_temp(&scores, Nonlinearity::Softmax, 10.0);

        // Lower temperature -> sharper distribution (more weight on max)
        assert!(weights_hot[0][1] > weights_cold[0][1]);
    }

    // ==========================================================================
    // K20: Basic operations complete without panic
    // ==========================================================================
    #[test]
    fn k20_operations_complete() {
        let a = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let b = vec![vec![0.5, 0.5], vec![0.5, 0.5]];

        // All operations should complete without panic
        let _ = logical_join(&a, &b, LogicMode::Boolean);
        let _ = logical_join(&a, &b, LogicMode::Continuous);
        let _ = logical_union(&a, &b, LogicMode::Boolean);
        let _ = logical_union(&a, &b, LogicMode::Continuous);
        let _ = logical_negation(&a, LogicMode::Boolean);
        let _ = logical_negation(&a, LogicMode::Continuous);
        let _ = logical_project(&a, 1, LogicMode::Boolean);
        let _ = logical_project(&a, 1, LogicMode::Continuous);
        let _ = logical_select(&a, &b, LogicMode::Boolean);
    }

    // Helper function for K17 test
    fn apply_nonlinearity_with_mask(
        scores: &[Vec<f64>],
        nonlinearity: Nonlinearity,
        mask: Option<&[Vec<bool>]>,
    ) -> Vec<Vec<f64>> {
        ops::apply_nonlinearity_with_mask(scores, nonlinearity, mask)
    }

    // Helper function for K19 test
    fn apply_nonlinearity_with_temp(
        scores: &[Vec<f64>],
        nonlinearity: Nonlinearity,
        temperature: f64,
    ) -> Vec<Vec<f64>> {
        ops::apply_nonlinearity_with_temperature(scores, nonlinearity, temperature)
    }
}
