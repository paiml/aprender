pub(crate) use super::*;

#[test]
fn test_apriori_new() {
    let apriori = Apriori::new();
    assert_eq!(apriori.min_support, 0.1);
    assert_eq!(apriori.min_confidence, 0.5);
    assert_eq!(apriori.frequent_itemsets.len(), 0);
    assert_eq!(apriori.rules.len(), 0);
}

#[test]
fn test_apriori_with_min_support() {
    let apriori = Apriori::new().with_min_support(0.3);
    assert_eq!(apriori.min_support, 0.3);
}

#[test]
fn test_apriori_with_min_confidence() {
    let apriori = Apriori::new().with_min_confidence(0.7);
    assert_eq!(apriori.min_confidence, 0.7);
}

#[test]
fn test_apriori_fit_basic() {
    // Simple market basket transactions
    let transactions = vec![
        vec![1, 2, 3],    // Transaction 1: milk, bread, butter
        vec![1, 2],       // Transaction 2: milk, bread
        vec![1, 3],       // Transaction 3: milk, butter
        vec![2, 3],       // Transaction 4: bread, butter
        vec![1, 2, 3, 4], // Transaction 5: milk, bread, butter, eggs
    ];

    let mut apriori = Apriori::new()
        .with_min_support(0.4) // 40% support
        .with_min_confidence(0.6); // 60% confidence

    apriori.fit(&transactions);

    // Should have found frequent itemsets
    assert!(!apriori.frequent_itemsets.is_empty());
}

#[test]
fn test_frequent_itemsets() {
    let transactions = vec![vec![1, 2, 3], vec![1, 2], vec![1, 3], vec![2, 3]];

    let mut apriori = Apriori::new().with_min_support(0.5); // 50% support
    apriori.fit(&transactions);

    let itemsets = apriori.get_frequent_itemsets();

    // With 4 transactions and min_support=0.5, need >= 2 occurrences
    // {1} appears in 3 transactions (75%) - frequent
    // {2} appears in 3 transactions (75%) - frequent
    // {3} appears in 3 transactions (75%) - frequent
    // {1,2} appears in 2 transactions (50%) - frequent
    // {1,3} appears in 2 transactions (50%) - frequent
    // {2,3} appears in 2 transactions (50%) - frequent
    // {1,2,3} appears in 1 transaction (25%) - not frequent

    assert!(itemsets.len() >= 6);

    // Verify itemsets are sorted by support descending
    for i in 1..itemsets.len() {
        assert!(itemsets[i - 1].1 >= itemsets[i].1);
    }
}

#[test]
fn test_association_rules() {
    let transactions = vec![vec![1, 2, 3], vec![1, 2], vec![1, 3], vec![2, 3]];

    let mut apriori = Apriori::new()
        .with_min_support(0.5)
        .with_min_confidence(0.6);

    apriori.fit(&transactions);
    let rules = apriori.get_rules();

    // Should have generated some rules
    assert!(!rules.is_empty());

    // All rules should meet minimum confidence
    for rule in &rules {
        assert!(rule.confidence >= 0.6);
    }

    // Rules should be sorted by confidence descending
    for i in 1..rules.len() {
        assert!(rules[i - 1].confidence >= rules[i].confidence);
    }
}

#[test]
fn test_support_calculation() {
    let transactions = vec![vec![1, 2, 3], vec![1, 2], vec![1, 3], vec![2, 3]];

    // Calculate support for {1, 2}
    let itemset: HashSet<usize> = vec![1, 2].into_iter().collect();
    let support = Apriori::calculate_support(&itemset, &transactions);

    // {1,2} appears in 2 out of 4 transactions = 0.5
    assert!((support - 0.5).abs() < 1e-10);

    // Calculate support for {1}
    let itemset: HashSet<usize> = vec![1].into_iter().collect();
    let support = Apriori::calculate_support(&itemset, &transactions);

    // {1} appears in 3 out of 4 transactions = 0.75
    assert!((support - 0.75).abs() < 1e-10);
}

#[test]
fn test_confidence_calculation() {
    let transactions = vec![vec![1, 2, 3], vec![1, 2], vec![1, 3], vec![2, 3]];

    let mut apriori = Apriori::new()
        .with_min_support(0.5)
        .with_min_confidence(0.0); // Accept all rules to verify confidence

    apriori.fit(&transactions);
    let rules = apriori.get_rules();

    // Find rule {1} => {2}
    let rule = rules
        .iter()
        .find(|r| r.antecedent == vec![1] && r.consequent == vec![2])
        .expect("Should have rule {1} => {2}");

    // Confidence({1} => {2}) = P({1,2}) / P({1}) = 0.5 / 0.75 = 0.667
    assert!((rule.confidence - 0.6666666).abs() < 1e-5);
}

#[test]
fn test_lift_calculation() {
    let transactions = vec![vec![1, 2, 3], vec![1, 2], vec![1, 3], vec![2, 3]];

    let mut apriori = Apriori::new()
        .with_min_support(0.5)
        .with_min_confidence(0.0);

    apriori.fit(&transactions);
    let rules = apriori.get_rules();

    // Find rule {1} => {2}
    let rule = rules
        .iter()
        .find(|r| r.antecedent == vec![1] && r.consequent == vec![2])
        .expect("Should have rule {1} => {2}");

    // Lift({1} => {2}) = confidence / P({2}) = 0.667 / 0.75 = 0.889
    assert!((rule.lift - 0.8888888).abs() < 1e-5);

    // Lift > 1.0 means positive correlation
    // Lift < 1.0 means negative correlation
    // Lift = 1.0 means independence
}

#[test]
fn test_min_support_filter() {
    let transactions = vec![
        vec![1, 2],
        vec![1, 2],
        vec![1, 2],
        vec![3, 4], // Infrequent items
    ];

    let mut apriori = Apriori::new().with_min_support(0.5);
    apriori.fit(&transactions);

    let itemsets = apriori.get_frequent_itemsets();

    // Only {1}, {2}, {1,2} should be frequent (75% support each)
    // {3}, {4}, {3,4} are infrequent (25% support)
    for (itemset, support) in itemsets {
        assert!(support >= 0.5, "All itemsets should meet min_support");
        assert!(
            !itemset.contains(&3) && !itemset.contains(&4),
            "Infrequent items should be pruned"
        );
    }
}

#[test]
fn test_min_confidence_filter() {
    let transactions = vec![vec![1, 2, 3], vec![1, 2], vec![1, 3], vec![1]];

    let mut apriori = Apriori::new()
        .with_min_support(0.25)
        .with_min_confidence(0.8); // High confidence threshold

    apriori.fit(&transactions);
    let rules = apriori.get_rules();

    // All rules should meet minimum confidence
    for rule in &rules {
        assert!(
            rule.confidence >= 0.8,
            "Rule {:?} => {:?} has confidence {:.2} < 0.8",
            rule.antecedent,
            rule.consequent,
            rule.confidence
        );
    }
}

#[test]
fn test_empty_transactions() {
    let transactions: Vec<Vec<usize>> = vec![];

    let mut apriori = Apriori::new();
    apriori.fit(&transactions);

    let itemsets = apriori.get_frequent_itemsets();
    assert_eq!(itemsets.len(), 0, "Should have no frequent itemsets");

    let rules = apriori.get_rules();
    assert_eq!(rules.len(), 0, "Should have no rules");
}

#[test]
fn test_single_item_transactions() {
    let transactions = vec![vec![1], vec![2], vec![3], vec![4]];

    let mut apriori = Apriori::new().with_min_support(0.25);
    apriori.fit(&transactions);

    let itemsets = apriori.get_frequent_itemsets();

    // Each item appears once (25% support)
    // Should have 4 frequent 1-itemsets
    assert_eq!(itemsets.len(), 4);

    // No multi-item itemsets possible
    for (itemset, _) in itemsets {
        assert_eq!(itemset.len(), 1);
    }

    let rules = apriori.get_rules();
    // No rules can be generated from single-item itemsets
    assert_eq!(rules.len(), 0);
}

#[test]
fn test_get_rules_before_fit() {
    let apriori = Apriori::new();
    let rules = apriori.get_rules();
    assert_eq!(rules.len(), 0, "Should have no rules before fit");
}

#[test]
fn test_get_itemsets_before_fit() {
    let apriori = Apriori::new();
    let itemsets = apriori.get_frequent_itemsets();
    assert_eq!(itemsets.len(), 0, "Should have no itemsets before fit");
}
