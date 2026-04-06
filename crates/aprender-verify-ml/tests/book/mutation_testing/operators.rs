//! Mutation operator examples from the book

use verificar::mutator::{MutationOperator, Mutator};

#[test]
fn test_aor_mutation_example() {
    // Example: Arithmetic Operator Replacement (AOR)
    let mutator = Mutator::with_operators(vec![MutationOperator::Aor]);
    let code = "x = a + b";

    let mutations = mutator.mutate(code).expect("mutation should succeed");

    assert!(!mutations.is_empty());
    assert_eq!(mutations[0].operator, MutationOperator::Aor);
    assert!(mutations[0].mutated.contains('-'));
}

#[test]
fn test_ror_mutation_example() {
    // Example: Relational Operator Replacement (ROR)
    let mutator = Mutator::with_operators(vec![MutationOperator::Ror]);
    let code = "if x < 5:";

    let mutations = mutator.mutate(code).expect("mutation should succeed");

    assert!(!mutations.is_empty());
    assert_eq!(mutations[0].operator, MutationOperator::Ror);
    assert!(mutations[0].mutated.contains("<="));
}

#[test]
fn test_bsr_mutation_example() {
    // Example: Boundary Substitution Replacement (BSR)
    let mutator = Mutator::with_operators(vec![MutationOperator::Bsr]);
    let code = "x = 0";

    let mutations = mutator.mutate(code).expect("mutation should succeed");

    assert!(!mutations.is_empty());
    assert_eq!(mutations[0].operator, MutationOperator::Bsr);
    assert!(mutations[0].mutated.contains("-1"));
}

#[test]
fn test_all_operators_example() {
    // Example: Using all mutation operators
    let mutator = Mutator::new();
    let code = "result = x + 1 if y < 0";

    let mutations = mutator.mutate(code).expect("mutation should succeed");

    // Should have mutations from AOR, ROR, BSR
    assert!(!mutations.is_empty());
    assert!(mutations
        .iter()
        .any(|m| m.operator == MutationOperator::Aor));
    assert!(mutations
        .iter()
        .any(|m| m.operator == MutationOperator::Ror));
}
