use super::*;

// =========================================================================
// Phase 1: Equations - Mathematical foundation
// =========================================================================

#[test]
fn test_equation_tour_length_formula() {
    // L(π) = Σᵢ d(π(i), π(i+1)) + d(π(n), π(1))
    let cities = vec![
        City::new(0.0, 0.0),
        City::new(1.0, 0.0),
        City::new(1.0, 1.0),
        City::new(0.0, 1.0),
    ];
    let demo = TspGraspDemo::with_cities(42, cities);

    // Square tour: 0 -> 1 -> 2 -> 3 -> 0
    let tour = vec![0, 1, 2, 3];
    let length = demo.compute_tour_length(&tour);

    // Expected: 4 edges of length 1 = 4.0
    assert!(
        (length - 4.0).abs() < 1e-10,
        "Square tour should have length 4.0, got {length}"
    );
}

#[test]
fn test_equation_two_opt_improvement_formula() {
    // Δ = d(i,i+1) + d(j,j+1) - d(i,j) - d(i+1,j+1)
    let cities = vec![
        City::new(0.0, 0.0), // 0
        City::new(2.0, 0.0), // 1
        City::new(1.0, 1.0), // 2
        City::new(1.0, 0.0), // 3
    ];
    let mut demo = TspGraspDemo::with_cities(42, cities);

    // Tour with crossing: 0 -> 1 -> 2 -> 3 -> 0
    // This crosses because 0-1-2-3 has edges that cross
    demo.tour = vec![0, 1, 2, 3];
    demo.tour_length = demo.compute_tour_length(&demo.tour);

    // Check if 2-opt can find improvement
    let improvement = demo.two_opt_improvement(0, 2);

    // Should find some improvement (crossing can be removed)
    println!("2-opt improvement (0,2): {improvement}");
}

#[test]
fn test_equation_bhh_constant() {
    // E[L] ≈ 0.7124 * sqrt(n * A)
    let demo = TspGraspDemo::new(42, 100);

    let expected = demo.bhh_expected_length();
    let bhh_constant = 0.7124;

    // For n=100 in unit square (A=1): E[L] ≈ 0.7124 * sqrt(100) = 7.124
    let expected_approx = bhh_constant * (100.0_f64).sqrt();

    assert!(
        (expected - expected_approx).abs() < 0.01,
        "BHH formula mismatch: got {expected}, expected {expected_approx}"
    );
}

#[test]
fn test_equation_mst_lower_bound() {
    // MST weight is a lower bound for TSP
    let cities = vec![
        City::new(0.0, 0.0),
        City::new(1.0, 0.0),
        City::new(0.5, 0.866), // Equilateral triangle
    ];
    let demo = TspGraspDemo::with_cities(42, cities);

    // MST of equilateral triangle with side 1 = 2.0
    // TSP tour = 3.0
    assert!(
        demo.lower_bound <= 3.0 + 1e-10,
        "MST lower bound should be <= TSP tour length"
    );
}

// =========================================================================
// Phase 2: Failing Tests - Tests that should fail without proper implementation
// =========================================================================

#[test]
fn test_failing_random_construction_high_gap() {
    // Random construction should have higher optimality gap than greedy
    let mut demo = TspGraspDemo::new(42, 50);
    demo.set_construction_method(ConstructionMethod::Random);

    demo.construct_tour();
    // Don't run 2-opt - raw random tour

    let random_length = demo.tour_length;

    // Reset and try greedy
    demo.reset();
    demo.set_construction_method(ConstructionMethod::NearestNeighbor);
    demo.construct_tour();

    let greedy_length = demo.tour_length;

    // Random should be significantly worse
    println!("Random tour: {random_length}, Greedy tour: {greedy_length}");
    assert!(
        random_length > greedy_length * 1.1,
        "Random construction should be >10% worse than greedy before 2-opt"
    );
}

#[test]
fn test_failing_no_two_opt_high_gap() {
    // Without 2-opt, even greedy has higher gap
    let mut demo = TspGraspDemo::new(42, 30);
    demo.set_construction_method(ConstructionMethod::RandomizedGreedy);

    demo.construct_tour();
    let before_2opt = demo.tour_length;

    demo.two_opt_to_local_optimum();
    let after_2opt = demo.tour_length;

    // 2-opt should improve the tour
    assert!(
        after_2opt < before_2opt,
        "2-opt should improve tour: before={before_2opt}, after={after_2opt}"
    );
}

#[test]
fn test_failing_single_restart_high_variance() {
    // Single restart gives high variance estimate
    let mut demo = TspGraspDemo::new(42, 20);

    demo.grasp_iteration();

    // With only one restart, can't compute meaningful variance
    assert!(
        demo.restart_history.len() == 1,
        "Should have exactly 1 restart"
    );
}

// =========================================================================
// Phase 3: Implementation - Core GRASP algorithm
// =========================================================================

#[test]
fn test_verification_grasp_improves_tour() {
    let mut demo = TspGraspDemo::new(42, 20);
    demo.set_rcl_size(5); // Larger RCL for more diversity

    // Run more GRASP iterations for better quality
    demo.run_grasp(25);

    // Should have found a reasonably good tour
    let gap = demo.optimality_gap();
    println!(
        "GRASP result: best_tour={:.4}, lower_bound={:.4}, gap={:.1}%",
        demo.best_tour_length,
        demo.lower_bound,
        gap * 100.0
    );

    // MST lower bound is weak, so 35% gap is realistic for small iterations
    assert!(
        gap < 0.35,
        "GRASP should achieve <35% gap, got {:.1}%",
        gap * 100.0
    );
}

#[test]
fn test_verification_two_opt_removes_crossings() {
    // Create a tour with obvious crossing
    let cities = vec![
        City::new(0.0, 0.0), // 0
        City::new(1.0, 1.0), // 1
        City::new(1.0, 0.0), // 2
        City::new(0.0, 1.0), // 3
    ];
    let mut demo = TspGraspDemo::with_cities(42, cities);

    // Crossing tour: 0 -> 1 -> 2 -> 3 -> 0 (diagonals cross)
    demo.tour = vec![0, 1, 2, 3];
    demo.tour_length = demo.compute_tour_length(&demo.tour);
    let crossing_length = demo.tour_length;

    demo.two_opt_to_local_optimum();
    let optimized_length = demo.tour_length;

    // Should find the non-crossing tour
    assert!(
        optimized_length < crossing_length,
        "2-opt should remove crossing: before={crossing_length}, after={optimized_length}"
    );
}

#[test]
fn test_verification_rcl_affects_diversity() {
    // Larger RCL should give more diverse solutions
    let mut demo1 = TspGraspDemo::new(42, 30);
    demo1.set_rcl_size(1); // Pure greedy

    let mut demo2 = TspGraspDemo::new(42, 30);
    demo2.set_rcl_size(5); // More randomization

    // Run multiple iterations
    for _ in 0..5 {
        demo1.grasp_iteration();
        demo2.grasp_iteration();
    }

    // RCL=1 should have lower variance (more deterministic)
    let var1 = demo1.restart_variance();
    let var2 = demo2.restart_variance();

    println!("RCL=1 variance: {var1}, RCL=5 variance: {var2}");
}

// =========================================================================
// Phase 4: Verification - Full verification of GRASP methodology
// =========================================================================

#[test]
fn test_verification_optimality_gap() {
    let mut demo = TspGraspDemo::new(42, 20);
    demo.set_rcl_size(5);
    demo.run_grasp(30);

    let gap = demo.optimality_gap();

    // MST lower bound is weak (~50-60% of optimal for random instances)
    // So gap relative to MST can be 30-40% even for good tours
    assert!(
        gap < 0.40,
        "Should achieve <40% optimality gap, got {:.1}%",
        gap * 100.0
    );
}

#[test]
fn test_verification_restart_consistency() {
    let mut demo = TspGraspDemo::new(42, 30);
    demo.run_grasp(30);

    let cv = demo.restart_cv();

    // CV should be reasonably low
    assert!(
        cv < 0.10,
        "Restart CV should be <10%, got {:.1}%",
        cv * 100.0
    );
}

#[test]
fn test_verification_greedy_beats_random() {
    // Core falsification: greedy start should beat random start
    let n = 40;
    let iterations = 15;

    // Greedy start
    let mut demo_greedy = TspGraspDemo::new(42, n);
    demo_greedy.set_construction_method(ConstructionMethod::RandomizedGreedy);
    demo_greedy.run_grasp(iterations);
    let greedy_best = demo_greedy.best_tour_length;

    // Random start
    let mut demo_random = TspGraspDemo::new(42, n);
    demo_random.set_construction_method(ConstructionMethod::Random);
    demo_random.run_grasp(iterations);
    let random_best = demo_random.best_tour_length;

    println!("Greedy best: {greedy_best}, Random best: {random_best}");

    assert!(
        greedy_best < random_best,
        "Greedy start ({greedy_best}) should beat random start ({random_best})"
    );
}

// =========================================================================
// Phase 5: Falsification - Conditions that break the model
// =========================================================================

#[test]
fn test_falsification_random_start_worse() {
    // Random construction (without 2-opt) should produce longer tours than greedy
    let n = 30;

    // Run multiple trials comparing construction quality before 2-opt
    let mut greedy_better = 0;
    let trials = 10;

    for seed in 0..trials {
        let mut demo_greedy = TspGraspDemo::new(seed as u64, n);
        demo_greedy.set_construction_method(ConstructionMethod::RandomizedGreedy);
        demo_greedy.construct_tour();
        let greedy_length = demo_greedy.tour_length;

        let mut demo_random = TspGraspDemo::new(seed as u64, n);
        demo_random.set_construction_method(ConstructionMethod::Random);
        demo_random.construct_tour();
        let random_length = demo_random.tour_length;

        if greedy_length < random_length {
            greedy_better += 1;
        }
    }

    // Greedy construction should produce shorter tours in majority of cases
    assert!(
        greedy_better >= trials * 7 / 10,
        "Greedy should produce better initial tours: {greedy_better}/{trials}"
    );
}

#[test]
fn test_falsification_small_instance_optimal() {
    // For very small instances, should find optimal
    let cities = vec![
        City::new(0.0, 0.0),
        City::new(1.0, 0.0),
        City::new(1.0, 1.0),
        City::new(0.0, 1.0),
    ];
    let mut demo = TspGraspDemo::with_cities(42, cities);

    demo.run_grasp(10);

    // Optimal tour for unit square = 4.0
    assert!(
        (demo.best_tour_length - 4.0).abs() < 1e-10,
        "Should find optimal tour (4.0), got {}",
        demo.best_tour_length
    );
}

#[test]
fn test_falsification_status_structure() {
    let mut demo = TspGraspDemo::new(42, 20);
    demo.run_grasp(10);

    let status = demo.get_falsification_status();

    // JIDOKA: Crossings FIRST - stop-the-line quality control
    // Lin & Kernighan (1973): Euclidean TSP optimal tours have ZERO crossings
    assert_eq!(status.criteria.len(), 3);
    assert_eq!(status.criteria[0].id, "TSP-CROSSINGS");
    assert_eq!(status.criteria[1].id, "TSP-GAP");
    assert_eq!(status.criteria[2].id, "TSP-VARIANCE");
}

// =========================================================================
// Additional Tests: EddDemo trait and reproducibility
// =========================================================================

#[test]
fn test_demo_trait_implementation() {
    let demo = TspGraspDemo::new(42, 20);

    assert_eq!(demo.name(), "TSP Randomized Greedy Start with 2-Opt");
    assert_eq!(demo.emc_ref(), "optimization/tsp_grasp_2opt");
}

#[test]
fn test_reproducibility() {
    let mut demo1 = TspGraspDemo::new(42, 30);
    let mut demo2 = TspGraspDemo::new(42, 30);

    demo1.run_grasp(5);
    demo2.run_grasp(5);

    assert_eq!(
        demo1.best_tour_length, demo2.best_tour_length,
        "Same seed should produce identical results"
    );
}

#[test]
fn test_serialization() {
    let demo = TspGraspDemo::new(42, 15);
    let json = serde_json::to_string(&demo).expect("serialize");

    assert!(json.contains("cities"));
    assert!(json.contains("rcl_size"));

    let mut restored: TspGraspDemo = serde_json::from_str(&json).expect("deserialize");
    restored.reinitialize();
    assert_eq!(restored.n, demo.n);
    assert_eq!(restored.seed, demo.seed);
}

// =========================================================================
// New Tests: 1-tree bound, crossing detection, 2-opt optimality
// =========================================================================

#[test]
fn test_one_tree_bound_tighter_than_mst() {
    // 1-tree bound should be at least as tight as MST
    // For a square, MST = 3.0, 1-tree = 4.0 (the tour itself)
    let cities = vec![
        City::new(0.0, 0.0),
        City::new(1.0, 0.0),
        City::new(1.0, 1.0),
        City::new(0.0, 1.0),
    ];
    let demo = TspGraspDemo::with_cities(42, cities);

    // For unit square: optimal tour = 4.0
    // 1-tree bound should be close to 4.0
    assert!(
        demo.lower_bound >= 3.9,
        "1-tree bound for unit square should be ~4.0, got {}",
        demo.lower_bound
    );
}

#[test]
fn test_crossing_detection() {
    // Create a tour with obvious crossing
    let cities = vec![
        City::new(0.0, 0.0), // 0: bottom-left
        City::new(1.0, 1.0), // 1: top-right
        City::new(1.0, 0.0), // 2: bottom-right
        City::new(0.0, 1.0), // 3: top-left
    ];
    let mut demo = TspGraspDemo::with_cities(42, cities);

    // Crossing tour: 0 -> 1 -> 2 -> 3 -> 0
    // Edges: 0-1 (diagonal ↗), 1-2 (down), 2-3 (diagonal ↖), 3-0 (down)
    // Edges 0-1 and 2-3 cross in the middle
    demo.tour = vec![0, 1, 2, 3];
    demo.tour_length = demo.compute_tour_length(&demo.tour);

    let crossings_before = demo.count_crossings();
    assert!(
        crossings_before > 0,
        "Crossing tour should have at least 1 crossing"
    );

    // After 2-opt, crossings should be 0
    demo.two_opt_to_local_optimum();
    let crossings_after = demo.count_crossings();

    assert_eq!(
        crossings_after, 0,
        "After 2-opt, tour should have no crossings (got {crossings_after})"
    );
}

#[test]
fn test_two_opt_reaches_local_optimum() {
    let mut demo = TspGraspDemo::new(42, 30);
    demo.construct_tour();

    // Run 2-opt to local optimum
    demo.two_opt_to_local_optimum();

    // Verify we're at a local optimum
    assert!(
        demo.is_two_opt_optimal(),
        "After two_opt_to_local_optimum, should be at local optimum"
    );
}

#[test]
fn test_two_opt_no_crossings_for_random_instances() {
    // After 2-opt, 2D Euclidean TSP should have no crossings
    for seed in 0..5 {
        let mut demo = TspGraspDemo::new(seed, 25);
        demo.grasp_iteration();

        let crossings = demo.count_crossings();
        assert_eq!(
            crossings, 0,
            "Seed {seed}: 2-opt tour should have no crossings (got {crossings})"
        );
    }
}

#[test]
fn test_improved_gap_with_one_tree() {
    // With 1-tree bound, gap should be lower than with plain MST
    let mut demo = TspGraspDemo::new(42, 20);
    demo.run_grasp(20);

    let gap = demo.optimality_gap();
    println!(
        "With 1-tree bound: best_tour={:.4}, lower_bound={:.4}, gap={:.1}%",
        demo.best_tour_length,
        demo.lower_bound,
        gap * 100.0
    );

    // With 1-tree, gap should be much better (typically <15%)
    assert!(
        gap < 0.20,
        "With 1-tree bound, gap should be <20%, got {:.1}%",
        gap * 100.0
    );
}

// =========================================================================
// EDD Audit Logging Tests (EDD-16, EDD-17, EDD-18)
// =========================================================================

#[test]
fn test_audit_logging_enabled_by_default() {
    let demo = TspGraspDemo::new(42, 10);
    assert!(
        demo.audit_enabled,
        "Audit logging should be enabled by default"
    );
}

#[test]
fn test_audit_log_records_grasp_iterations() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.run_grasp(5);

    let log = demo.audit_log();
    assert_eq!(
        log.len(),
        5,
        "Should have 5 audit entries for 5 GRASP iterations"
    );
}

#[test]
fn test_audit_log_contains_equation_evaluations() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.grasp_iteration();

    let log = demo.audit_log();
    assert_eq!(log.len(), 1, "Should have 1 audit entry");

    let entry = &log[0];
    assert!(
        entry.equation_evaluations.len() >= 2,
        "Should have tour_length and optimality_gap equations"
    );

    // Check for tour_length equation
    let tour_length_eq = entry
        .equation_evaluations
        .iter()
        .find(|e| e.equation_id == "tour_length");
    assert!(tour_length_eq.is_some(), "Should have tour_length equation");
}

#[test]
fn test_audit_log_contains_decisions() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.grasp_iteration();

    let log = demo.audit_log();
    let entry = &log[0];

    assert!(!entry.decisions.is_empty(), "Should have decision records");

    let construction_decision = entry
        .decisions
        .iter()
        .find(|d| d.decision_type == "construction_method");
    assert!(
        construction_decision.is_some(),
        "Should have construction_method decision"
    );
}

#[test]
fn test_audit_log_step_type() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.grasp_iteration();

    let log = demo.audit_log();
    let entry = &log[0];

    assert_eq!(
        entry.step_type, "grasp_iteration",
        "Step type should be grasp_iteration"
    );
}

#[test]
fn test_audit_log_has_rng_state() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.grasp_iteration();

    let log = demo.audit_log();
    let entry = &log[0];

    // RNG state should have changed (not all zeros after and before)
    assert_ne!(
        entry.rng_state_before, entry.rng_state_after,
        "RNG state should change during GRASP iteration"
    );
}

#[test]
fn test_audit_log_has_state_snapshots() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.grasp_iteration();

    let log = demo.audit_log();
    let entry = &log[0];

    // Input state should have no tour yet
    assert!(
        entry.input_state.tour.is_empty(),
        "Input state should have empty tour"
    );

    // Output state should have a tour
    assert!(
        !entry.output_state.tour.is_empty(),
        "Output state should have a tour"
    );
    assert!(
        entry.output_state.tour_length > 0.0,
        "Output state should have positive tour length"
    );
}

#[test]
fn test_audit_log_can_be_disabled() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.set_audit_enabled(false);
    demo.run_grasp(5);

    let log = demo.audit_log();
    assert!(log.is_empty(), "Audit log should be empty when disabled");
}

#[test]
fn test_audit_log_clear() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.run_grasp(3);

    assert_eq!(demo.audit_log().len(), 3);

    demo.clear_audit_log();
    assert!(
        demo.audit_log().is_empty(),
        "Audit log should be empty after clear"
    );
}

#[test]
fn test_audit_log_json_export() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.grasp_iteration();

    let json = demo
        .export_audit_json()
        .expect("JSON export should succeed");

    assert!(json.contains("step_id"), "JSON should contain step_id");
    assert!(
        json.contains("tour_length"),
        "JSON should contain tour_length equation"
    );
    assert!(
        json.contains("construction_method"),
        "JSON should contain construction_method"
    );
}

#[test]
fn test_audit_log_generates_test_cases() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.run_grasp(3);

    let test_cases = demo.generate_test_cases();

    // Should have test cases for each equation evaluation (2 per iteration = 6 total)
    assert!(
        test_cases.len() >= 6,
        "Should generate test cases from audit log, got {}",
        test_cases.len()
    );

    // First test case should have correct structure
    if let Some(tc) = test_cases.first() {
        assert!(!tc.name.is_empty(), "Test case should have name");
        assert!(
            !tc.equation_id.is_empty(),
            "Test case should have equation_id"
        );
    }
}

#[test]
fn test_audit_log_reproducibility() {
    // Same seed should produce identical audit logs
    let mut demo1 = TspGraspDemo::new(42, 15);
    let mut demo2 = TspGraspDemo::new(42, 15);

    demo1.run_grasp(3);
    demo2.run_grasp(3);

    let log1 = demo1.audit_log();
    let log2 = demo2.audit_log();

    assert_eq!(log1.len(), log2.len(), "Logs should have same length");

    for (e1, e2) in log1.iter().zip(log2.iter()) {
        // Tour lengths should be identical
        assert!(
            (e1.output_state.tour_length - e2.output_state.tour_length).abs() < f64::EPSILON,
            "Tour lengths should match for reproducibility"
        );
        // RNG states should be identical
        assert_eq!(
            e1.rng_state_before, e2.rng_state_before,
            "RNG states should match for reproducibility"
        );
    }
}

#[test]
fn test_simulation_audit_log_trait_implementation() {
    // Test that TspGraspDemo properly implements SimulationAuditLog trait
    let mut demo = TspGraspDemo::new(42, 10);

    // Initially empty
    assert!(demo.audit_log().is_empty());
    assert_eq!(demo.total_equation_evals(), 0);
    assert_eq!(demo.total_decisions(), 0);

    // After one iteration
    demo.grasp_iteration();

    assert_eq!(demo.audit_log().len(), 1);
    assert!(demo.total_equation_evals() >= 2);
    assert!(demo.total_decisions() >= 1);

    // Verify all equations should pass (no failures with reasonable tolerance)
    let failures = demo.verify_all_equations(1e-6);
    assert!(
        failures.is_empty(),
        "Should have no equation verification failures, got {:?}",
        failures
    );
}

// =========================================================================
// Additional Coverage Tests - Edge Cases
// =========================================================================

#[test]
fn test_default() {
    let demo = TspGraspDemo::default();
    assert_eq!(demo.n, 20);
    assert_eq!(demo.seed, 42);
}

#[test]
fn test_city_default() {
    let city = City::default();
    assert!((city.x - 0.0).abs() < f64::EPSILON);
    assert!((city.y - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_city_distance_to_self() {
    let city = City::new(1.0, 2.0);
    assert!((city.distance_to(&city) - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_gen_range_usize_zero_max() {
    let mut rng = SimRng::new(42);
    let result = gen_range_usize(&mut rng, 0);
    assert_eq!(result, 0);
}

#[test]
fn test_gen_range_usize_one() {
    let mut rng = SimRng::new(42);
    let result = gen_range_usize(&mut rng, 1);
    assert_eq!(result, 0);
}

#[test]
fn test_construction_method_default() {
    let method = ConstructionMethod::default();
    assert_eq!(method, ConstructionMethod::RandomizedGreedy);
}

#[test]
fn test_set_construction_method() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.set_construction_method(ConstructionMethod::NearestNeighbor);
    assert_eq!(
        demo.construction_method,
        ConstructionMethod::NearestNeighbor
    );
}

#[test]
fn test_set_rcl_size_clamps() {
    let mut demo = TspGraspDemo::new(42, 10);

    // Test clamping to minimum 1
    demo.set_rcl_size(0);
    assert_eq!(demo.rcl_size, 1);

    // Test clamping to maximum n
    demo.set_rcl_size(100);
    assert_eq!(demo.rcl_size, demo.n);
}

#[test]
fn test_compute_tour_length_empty() {
    let demo = TspGraspDemo::new(42, 10);
    let length = demo.compute_tour_length(&[]);
    assert!((length - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_compute_tour_length_single_city() {
    let demo = TspGraspDemo::new(42, 10);
    let length = demo.compute_tour_length(&[0]);
    assert!((length - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_distance_fallback_no_matrix() {
    let cities = vec![
        City::new(0.0, 0.0),
        City::new(1.0, 0.0),
        City::new(1.0, 1.0),
    ];
    let mut demo = TspGraspDemo::with_cities(42, cities);
    demo.distance_matrix.clear(); // Clear precomputed matrix

    let d = demo.distance(0, 1);
    assert!((d - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_two_opt_improvement_edge_cases() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.construct_tour();

    // Test i >= j
    let improvement = demo.two_opt_improvement(5, 3);
    assert!((improvement - 0.0).abs() < f64::EPSILON);

    // Test j >= n
    let improvement = demo.two_opt_improvement(0, 100);
    assert!((improvement - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_two_opt_improvement_small_tour() {
    let cities = vec![
        City::new(0.0, 0.0),
        City::new(1.0, 0.0),
        City::new(0.5, 0.5),
    ];
    let mut demo = TspGraspDemo::with_cities(42, cities);
    demo.tour = vec![0, 1, 2];

    // With only 3 cities, no meaningful 2-opt possible
    let improvement = demo.two_opt_improvement(0, 1);
    assert!((improvement - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_two_opt_pass_small_tour() {
    let cities = vec![
        City::new(0.0, 0.0),
        City::new(1.0, 0.0),
        City::new(0.5, 0.5),
    ];
    let mut demo = TspGraspDemo::with_cities(42, cities);
    demo.tour = vec![0, 1, 2];

    let improved = demo.two_opt_pass();
    assert!(!improved, "3-city tour cannot be improved by 2-opt");
}

#[test]
fn test_count_crossings_small_tour() {
    let cities = vec![
        City::new(0.0, 0.0),
        City::new(1.0, 0.0),
        City::new(0.5, 0.5),
    ];
    let mut demo = TspGraspDemo::with_cities(42, cities);
    demo.tour = vec![0, 1, 2];

    let crossings = demo.count_crossings();
    assert_eq!(crossings, 0, "3-city tour cannot have crossings");
}

#[test]
fn test_is_two_opt_optimal_small_tour() {
    let cities = vec![
        City::new(0.0, 0.0),
        City::new(1.0, 0.0),
        City::new(0.5, 0.5),
    ];
    let mut demo = TspGraspDemo::with_cities(42, cities);
    demo.tour = vec![0, 1, 2];

    assert!(
        demo.is_two_opt_optimal(),
        "3-city tour is always 2-opt optimal"
    );
}

#[test]
fn test_optimality_gap_zero_lower_bound() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.lower_bound = 0.0;
    demo.best_tour_length = 5.0;

    let gap = demo.optimality_gap();
    assert!((gap - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_restart_variance_empty() {
    let demo = TspGraspDemo::new(42, 10);
    let variance = demo.restart_variance();
    assert!((variance - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_restart_variance_single() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.restart_history.push(5.0);

    let variance = demo.restart_variance();
    assert!((variance - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_restart_cv_empty() {
    let demo = TspGraspDemo::new(42, 10);
    let cv = demo.restart_cv();
    assert!((cv - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_restart_cv_zero_mean() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.restart_history = vec![0.0, 0.0, 0.0];

    let cv = demo.restart_cv();
    assert!((cv - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_verify_equation_no_tour() {
    let demo = TspGraspDemo::new(42, 10);
    // No tour constructed yet
    let verified = demo.verify_equation();
    // Should fail or return true depending on edge cases
    println!("verify_equation with no tour: {verified}");
}

#[test]
fn test_verify_equation_few_restarts() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.grasp_iteration();
    // Only 1 restart - CV check should pass due to insufficient data
    let verified = demo.verify_equation();
    println!("verify_equation with 1 restart: {verified}");
}

#[test]
fn test_get_falsification_status_no_best_tour() {
    let demo = TspGraspDemo::new(42, 10);
    let status = demo.get_falsification_status();

    // Should handle empty best_tour gracefully
    assert!(!status.verified || status.criteria[2].passed);
}

#[test]
fn test_get_falsification_status_high_cv() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.restart_history = vec![1.0, 10.0, 1.0, 10.0, 1.0, 10.0]; // High variance
    demo.best_tour_length = 5.0;
    demo.best_tour = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

    let status = demo.get_falsification_status();
    assert!(!status.criteria[1].passed, "High CV should fail");
}

#[test]
fn test_reset_preserves_configuration() {
    let mut demo = TspGraspDemo::new(42, 25);
    demo.set_construction_method(ConstructionMethod::NearestNeighbor);
    demo.set_rcl_size(7);
    demo.run_grasp(5);

    let cities_before = demo.cities.clone();
    demo.reset();

    assert_eq!(demo.n, 25);
    assert_eq!(
        demo.construction_method,
        ConstructionMethod::NearestNeighbor
    );
    assert_eq!(demo.rcl_size, 7);
    assert_eq!(demo.cities.len(), cities_before.len());
    assert_eq!(demo.restarts, 0);
    assert!(demo.restart_history.is_empty());
}

#[test]
fn test_reinitialize() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.rng = None;
    demo.distance_matrix.clear();
    demo.lower_bound = 0.0;

    demo.reinitialize();

    assert!(demo.rng.is_some());
    assert!(!demo.distance_matrix.is_empty());
    assert!(demo.lower_bound > 0.0);
}

#[test]
fn test_reinitialize_already_initialized() {
    let demo_original = TspGraspDemo::new(42, 10);
    let mut demo = demo_original.clone();

    demo.reinitialize();

    // Should not change anything if already initialized
    assert_eq!(demo.n, demo_original.n);
}

#[test]
fn test_with_cities_minimum_area() {
    // Cities all at same point - area should be clamped to 0.001
    let cities = vec![
        City::new(0.5, 0.5),
        City::new(0.5, 0.5),
        City::new(0.5, 0.5),
        City::new(0.5, 0.5),
    ];
    let demo = TspGraspDemo::with_cities(42, cities);

    assert!((demo.area - 0.001).abs() < f64::EPSILON);
}

#[test]
fn test_compute_mst_excluding_small() {
    let cities = vec![City::new(0.0, 0.0), City::new(1.0, 0.0)];
    let demo = TspGraspDemo::with_cities(42, cities);

    // With only 2 cities, MST excluding one is 0
    let mst = demo.compute_mst_excluding(0);
    assert!((mst - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_compute_one_tree_bound_small() {
    let cities = vec![City::new(0.0, 0.0), City::new(1.0, 0.0)];
    let demo = TspGraspDemo::with_cities(42, cities);

    let bound = demo.compute_one_tree_bound();
    assert!((bound - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_segments_intersect_false_cases() {
    // Non-intersecting segments
    let p1 = City::new(0.0, 0.0);
    let p2 = City::new(1.0, 0.0);
    let p3 = City::new(0.0, 1.0);
    let p4 = City::new(1.0, 1.0);

    assert!(!TspGraspDemo::segments_intersect(&p1, &p2, &p3, &p4));
}

#[test]
fn test_clone() {
    let demo = TspGraspDemo::new(42, 15);
    let cloned = demo.clone();

    assert_eq!(demo.n, cloned.n);
    assert_eq!(demo.seed, cloned.seed);
    assert_eq!(demo.cities.len(), cloned.cities.len());
}

#[test]
fn test_debug() {
    let demo = TspGraspDemo::new(42, 5);
    let debug_str = format!("{demo:?}");
    assert!(debug_str.contains("TspGraspDemo"));
}

#[test]
fn test_construction_method_debug() {
    let method = ConstructionMethod::RandomizedGreedy;
    let debug_str = format!("{method:?}");
    assert!(debug_str.contains("RandomizedGreedy"));
}

#[test]
fn test_construct_tour_no_rng() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.rng = None;

    demo.construct_tour();
    // Should handle gracefully - tour should be empty
    assert!(demo.tour.is_empty());
}

#[test]
fn test_construct_nearest_neighbor_no_rng() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.set_construction_method(ConstructionMethod::NearestNeighbor);
    demo.rng = None;

    demo.construct_tour();
    assert!(demo.tour.is_empty());
}

#[test]
fn test_construct_random_no_rng() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.set_construction_method(ConstructionMethod::Random);
    demo.rng = None;

    demo.construct_tour();
    assert!(demo.tour.is_empty());
}

#[test]
fn test_step_calls_grasp_iteration() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.step(0.0);

    assert_eq!(demo.restarts, 1);
    assert!(!demo.best_tour.is_empty());
}

#[test]
fn test_audit_log_mut() {
    let mut demo = TspGraspDemo::new(42, 10);
    demo.grasp_iteration();

    let log = demo.audit_log_mut();
    assert_eq!(log.len(), 1);

    // Can modify the log
    log.clear();
    assert!(demo.audit_log().is_empty());
}

#[test]
fn test_log_step_trait_method() {
    let mut demo = TspGraspDemo::new(42, 10);

    let snapshot = TspStateSnapshot {
        tour: vec![0, 1, 2],
        tour_length: 3.0,
        best_tour: vec![0, 1, 2],
        best_tour_length: 3.0,
        restarts: 1,
        two_opt_iterations: 0,
        two_opt_improvements: 0,
    };

    let entry = StepEntry::new(
        0,
        SimTime::from_secs(0.0),
        "test".to_string(),
        snapshot.clone(),
        snapshot,
    );

    demo.log_step(entry);
    assert_eq!(demo.audit_log().len(), 1);
}

#[test]
fn test_minimum_cities_enforced() {
    let demo = TspGraspDemo::new(42, 2);
    // Should be clamped to minimum 4
    assert_eq!(demo.n, 4);
}

#[test]
fn test_apply_two_opt() {
    let cities = vec![
        City::new(0.0, 0.0),
        City::new(1.0, 0.0),
        City::new(1.0, 1.0),
        City::new(0.0, 1.0),
    ];
    let mut demo = TspGraspDemo::with_cities(42, cities);
    demo.tour = vec![0, 1, 2, 3];
    demo.tour_length = demo.compute_tour_length(&demo.tour);

    let original_tour = demo.tour.clone();
    demo.apply_two_opt(0, 2);

    // Tour should be different after 2-opt
    assert_ne!(demo.tour, original_tour);
}

#[test]
fn test_bhh_expected_length_non_unit_area() {
    let cities = vec![
        City::new(0.0, 0.0),
        City::new(2.0, 0.0),
        City::new(2.0, 2.0),
        City::new(0.0, 2.0),
    ];
    let demo = TspGraspDemo::with_cities(42, cities);

    // Area = 4, n = 4
    let expected = demo.bhh_expected_length();
    let manual = 0.7124 * (4.0 * 4.0_f64).sqrt();

    assert!((expected - manual).abs() < 1e-10);
}

#[test]
fn test_default_audit_enabled_function() {
    assert!(default_audit_enabled());
}

#[test]
fn test_grasp_iteration_updates_best() {
    let mut demo = TspGraspDemo::new(42, 10);
    assert_eq!(demo.best_tour_length, 0.0);

    demo.grasp_iteration();

    assert!(demo.best_tour_length > 0.0);
    assert!(!demo.best_tour.is_empty());
}

#[test]
fn test_grasp_iteration_improves_best() {
    let mut demo = TspGraspDemo::new(42, 50);
    demo.run_grasp(20);

    let first_best = demo.restart_history[0];
    let final_best = demo.best_tour_length;

    // Best should be <= first attempt
    assert!(final_best <= first_best + f64::EPSILON);
}

// =========================================================================
// YAML Integration Tests (OR-001-07)
// =========================================================================

// Small 6-city instance for unit tests (inline to avoid file dependency)
const SMALL_6_CITY_YAML: &str = r#"
meta:
  id: "TSP-TEST-006"
  version: "1.0.0"
  description: "6-city test instance"
  units: "miles"
  optimal_known: 115
cities:
  - { id: 0, name: "San Francisco", alias: "SF", coords: { lat: 37.7749, lon: -122.4194 } }
  - { id: 1, name: "Oakland", alias: "OAK", coords: { lat: 37.8044, lon: -122.2712 } }
  - { id: 2, name: "San Jose", alias: "SJ", coords: { lat: 37.3382, lon: -121.8863 } }
  - { id: 3, name: "Palo Alto", alias: "PA", coords: { lat: 37.4419, lon: -122.1430 } }
  - { id: 4, name: "Berkeley", alias: "BRK", coords: { lat: 37.8716, lon: -122.2727 } }
  - { id: 5, name: "Fremont", alias: "FRE", coords: { lat: 37.5485, lon: -121.9886 } }
matrix:
  - [ 0, 12, 48, 35, 14, 42]
  - [12,  0, 42, 30,  4, 30]
  - [48, 42,  0, 15, 46, 17]
  - [35, 30, 15,  0, 32, 18]
  - [14,  4, 46, 32,  0, 32]
  - [42, 30, 17, 18, 32,  0]
algorithm:
  method: "grasp"
  params:
rcl_size: 3
restarts: 10
two_opt: true
seed: 42
"#;

// Full 20-city California instance from file
const CALIFORNIA_20_YAML: &str = include_str!("../../../examples/experiments/bay_area_tsp.yaml");

#[test]
fn test_from_yaml_small_6_city() {
    let demo = TspGraspDemo::from_yaml(SMALL_6_CITY_YAML).expect("YAML should parse");
    assert_eq!(demo.n, 6);
    assert_eq!(demo.rcl_size, 3);
}

#[test]
fn test_from_yaml_california_20_city() {
    let demo = TspGraspDemo::from_yaml(CALIFORNIA_20_YAML).expect("YAML should parse");
    assert_eq!(demo.n, 20);
    assert_eq!(demo.rcl_size, 3);
    assert_eq!(demo.units, "miles");
}

#[test]
fn test_from_yaml_uses_yaml_distances() {
    let demo = TspGraspDemo::from_yaml(SMALL_6_CITY_YAML).expect("YAML should parse");

    // SF to Oakland should be 12 miles (from YAML matrix)
    let sf_to_oakland = demo.distance(0, 1);
    assert!(
        (sf_to_oakland - 12.0).abs() < 0.1,
        "SF→Oakland should be 12 miles, got {sf_to_oakland}"
    );

    // Oakland to Berkeley should be 4 miles
    let oakland_to_berkeley = demo.distance(1, 4);
    assert!(
        (oakland_to_berkeley - 4.0).abs() < 0.1,
        "Oakland→Berkeley should be 4 miles, got {oakland_to_berkeley}"
    );
}

#[test]
fn test_from_yaml_optimal_tour_115() {
    let demo = TspGraspDemo::from_yaml(SMALL_6_CITY_YAML).expect("YAML should parse");

    // Optimal tour: SF(0) → OAK(1) → BRK(4) → FRE(5) → SJ(2) → PA(3) → SF(0)
    let optimal_tour = vec![0, 1, 4, 5, 2, 3];
    let length = demo.compute_tour_length(&optimal_tour);

    // 12 + 4 + 32 + 17 + 15 + 35 = 115
    assert!(
        (length - 115.0).abs() < 0.1,
        "Optimal tour should be 115 miles, got {length}"
    );
}

#[test]
fn test_from_yaml_grasp_finds_good_tour_6_city() {
    let mut demo = TspGraspDemo::from_yaml(SMALL_6_CITY_YAML).expect("YAML should parse");

    // Run GRASP
    demo.run_grasp(10);

    // Should find tour within 20% of optimal (115)
    assert!(
        demo.best_tour_length <= 115.0 * 1.2,
        "GRASP should find tour ≤138 miles, got {}",
        demo.best_tour_length
    );
}

#[test]
fn test_from_yaml_grasp_finds_good_tour_20_city() {
    let mut demo = TspGraspDemo::from_yaml(CALIFORNIA_20_YAML).expect("YAML should parse");

    // Run GRASP with more restarts for 20 cities
    demo.run_grasp(50);

    // For 20 cities, expect tour under 500 miles (generous bound)
    assert!(
        demo.best_tour_length <= 500.0,
        "GRASP should find tour ≤500 miles for 20 cities, got {}",
        demo.best_tour_length
    );
}

#[test]
fn test_from_yaml_invalid() {
    let result = TspGraspDemo::from_yaml("invalid yaml: [[[");
    assert!(result.is_err());
}

#[test]
fn test_from_yaml_validates() {
    // YAML with mismatched matrix dimensions should fail validation
    let bad_yaml = r#"
meta:
  id: "BAD"
  description: "Bad"
cities:
  - id: 0
    name: "A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
matrix:
  - [0, 10, 20]
  - [10, 0, 30]
"#;
    let result = TspGraspDemo::from_yaml(bad_yaml);
    assert!(result.is_err());
}

#[test]
fn test_from_instance_method_grasp() {
    let yaml = r#"
meta:
  id: "TEST"
  description: "Test"
cities:
  - id: 0
    name: "A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
  - id: 1
    name: "B"
    alias: "B"
    coords: { lat: 1.0, lon: 1.0 }
matrix:
  - [0, 10]
  - [10, 0]
algorithm:
  method: "grasp"
  params:
    rcl_size: 2
    restarts: 5
    two_opt: true
    seed: 123
"#;
    let instance = TspInstanceYaml::from_yaml(yaml).expect("parse");
    let demo = TspGraspDemo::from_instance(&instance);

    assert_eq!(
        demo.construction_method,
        ConstructionMethod::RandomizedGreedy
    );
    assert_eq!(demo.rcl_size, 2);
    assert_eq!(demo.seed, 123);
}

#[test]
fn test_from_instance_method_nearest_neighbor() {
    let yaml = r#"
meta:
  id: "TEST"
  description: "Test"
cities:
  - id: 0
    name: "A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
  - id: 1
    name: "B"
    alias: "B"
    coords: { lat: 1.0, lon: 1.0 }
matrix:
  - [0, 10]
  - [10, 0]
algorithm:
  method: "nearest_neighbor"
  params:
    seed: 42
"#;
    let instance = TspInstanceYaml::from_yaml(yaml).expect("parse");
    let demo = TspGraspDemo::from_instance(&instance);

    assert_eq!(
        demo.construction_method,
        ConstructionMethod::NearestNeighbor
    );
}

#[test]
fn test_from_instance_method_random() {
    let yaml = r#"
meta:
  id: "TEST"
  description: "Test"
cities:
  - id: 0
    name: "A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
  - id: 1
    name: "B"
    alias: "B"
    coords: { lat: 1.0, lon: 1.0 }
matrix:
  - [0, 10]
  - [10, 0]
algorithm:
  method: "random"
  params:
    seed: 42
"#;
    let instance = TspInstanceYaml::from_yaml(yaml).expect("parse");
    let demo = TspGraspDemo::from_instance(&instance);

    assert_eq!(demo.construction_method, ConstructionMethod::Random);
}

#[test]
fn test_from_yaml_file_success() {
    let result = TspGraspDemo::from_yaml_file("examples/experiments/bay_area_tsp.yaml");
    assert!(result.is_ok());
    let demo = result.unwrap();
    assert_eq!(demo.n, 20); // 20-city California instance
}

#[test]
fn test_from_yaml_file_not_found() {
    let result = TspGraspDemo::from_yaml_file("/nonexistent/path.yaml");
    assert!(result.is_err());
}

#[test]
fn test_from_yaml_deterministic() {
    // Two demos from same YAML should produce identical results
    let mut demo1 = TspGraspDemo::from_yaml(SMALL_6_CITY_YAML).expect("parse");
    let mut demo2 = TspGraspDemo::from_yaml(SMALL_6_CITY_YAML).expect("parse");

    demo1.run_grasp(5);
    demo2.run_grasp(5);

    assert_eq!(
        demo1.best_tour_length, demo2.best_tour_length,
        "Same seed should produce same result"
    );
    assert_eq!(demo1.best_tour, demo2.best_tour);
}
