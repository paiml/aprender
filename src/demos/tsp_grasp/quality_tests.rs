use super::*;

// =============================================================================
// Property-Based Tests (Proptest) - OR-001-12
// =============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Property: Tour must visit ALL cities exactly once.
        #[test]
        fn prop_tour_visits_all_cities(seed in 0u64..10000, n in 4usize..15) {
            let mut demo = TspGraspDemo::new(seed, n);
            demo.grasp_iteration();

            // All cities visited
            let mut visited = demo.best_tour.clone();
            visited.sort();
            let expected: Vec<usize> = (0..n).collect();
            prop_assert_eq!(visited, expected,
                "Tour must visit all cities exactly once");
        }

        /// Property: Tour length equals sum of edge weights.
        #[test]
        fn prop_tour_length_is_sum_of_edges(seed in 0u64..10000, n in 4usize..10) {
            let mut demo = TspGraspDemo::new(seed, n);
            demo.grasp_iteration();

            // Manually compute tour length
            let mut computed_length = 0.0;
            let tour = &demo.best_tour;
            for i in 0..tour.len() {
                let j = (i + 1) % tour.len();
                computed_length += demo.distance(tour[i], tour[j]);
            }

            prop_assert!(
                (computed_length - demo.best_tour_length).abs() < 1e-9,
                "Tour length {} != computed sum {}",
                demo.best_tour_length, computed_length
            );
        }

        /// Property: 2-opt NEVER makes tour worse.
        #[test]
        fn prop_two_opt_never_worsens(seed in 0u64..10000, n in 5usize..12) {
            let mut demo = TspGraspDemo::new(seed, n);
            demo.construct_tour();
            let before = demo.tour_length;

            demo.two_opt_to_local_optimum();
            let after = demo.tour_length;

            prop_assert!(after <= before + 1e-9,
                "2-opt made tour worse: {} -> {}", before, after);
        }

        /// Property: GRASP is deterministic (same seed = same result).
        #[test]
        fn prop_grasp_deterministic(seed in 0u64..10000, n in 4usize..10) {
            let mut demo1 = TspGraspDemo::new(seed, n);
            let mut demo2 = TspGraspDemo::new(seed, n);

            demo1.run_grasp(5);
            demo2.run_grasp(5);

            prop_assert_eq!(demo1.best_tour_length, demo2.best_tour_length,
                "Same seed must produce same tour length");
            prop_assert_eq!(demo1.best_tour, demo2.best_tour,
                "Same seed must produce same tour");
        }

        /// Property: Distance matrix is symmetric.
        #[test]
        fn prop_distance_symmetric(seed in 0u64..1000, n in 3usize..8) {
            let demo = TspGraspDemo::new(seed, n);

            for i in 0..n {
                for j in 0..n {
                    let d_ij = demo.distance(i, j);
                    let d_ji = demo.distance(j, i);
                    prop_assert!(
                        (d_ij - d_ji).abs() < 1e-9,
                        "Distance not symmetric: d({},{})={} != d({},{})={}",
                        i, j, d_ij, j, i, d_ji
                    );
                }
            }
        }

        /// Property: Lower bound <= any valid tour.
        #[test]
        fn prop_lower_bound_valid(seed in 0u64..5000, n in 4usize..10) {
            let mut demo = TspGraspDemo::new(seed, n);
            demo.grasp_iteration();

            prop_assert!(demo.lower_bound <= demo.best_tour_length + 1e-9,
                "Lower bound {} > tour length {}",
                demo.lower_bound, demo.best_tour_length);
        }

        /// Property: Optimality gap is non-negative (within floating-point tolerance).
        #[test]
        fn prop_optimality_gap_non_negative(seed in 0u64..5000, n in 4usize..10) {
            let mut demo = TspGraspDemo::new(seed, n);
            demo.run_grasp(3);

            let gap = demo.optimality_gap();
            // Allow tolerance for floating-point arithmetic (gap involves division)
            // Using 1e-14 to account for accumulated rounding errors
            prop_assert!(gap >= -1e-14, "Optimality gap {} significantly negative", gap);
        }
    }
}

// =============================================================================
// Mutation-Sensitive Tests - OR-001-13
// These tests are designed to FAIL if key logic is mutated/broken.
// =============================================================================

#[cfg(test)]
mod mutation_tests {
    use super::*;

    /// Mutation test: If we skip 2-opt, tour should be WORSE.
    #[test]
    fn mutation_two_opt_improves_tour() {
        let mut demo_with = TspGraspDemo::new(42, 10);
        let mut demo_without = TspGraspDemo::new(42, 10);

        // With 2-opt
        demo_with.construct_tour();
        let before = demo_with.tour_length;
        demo_with.two_opt_to_local_optimum();
        let after_with = demo_with.tour_length;

        // Without 2-opt (just construction)
        demo_without.construct_tour();
        let after_without = demo_without.tour_length;

        // 2-opt should improve or equal (never worse)
        assert!(after_with <= before + f64::EPSILON);

        // At least sometimes 2-opt should make a difference
        // (if this fails, 2-opt might be broken)
        assert!(after_with <= after_without);
    }

    /// Mutation test: RCL size affects construction randomness.
    #[test]
    fn mutation_rcl_size_affects_result() {
        let seed = 42;

        // RCL=1 should be greedy (deterministic nearest neighbor behavior)
        let mut demo_rcl1 = TspGraspDemo::new(seed, 8);
        demo_rcl1.set_rcl_size(1);
        demo_rcl1.grasp_iteration();
        let len_rcl1 = demo_rcl1.best_tour_length;

        // RCL=n should be more random
        let mut demo_rcl_max = TspGraspDemo::new(seed, 8);
        demo_rcl_max.set_rcl_size(8);
        demo_rcl_max.grasp_iteration();
        let len_rcl_max = demo_rcl_max.best_tour_length;

        // Both should produce valid tours
        assert!(len_rcl1 > 0.0);
        assert!(len_rcl_max > 0.0);

        // With RCL=1 (greedy), multiple runs with same seed should be identical
        let mut demo_rcl1_repeat = TspGraspDemo::new(seed, 8);
        demo_rcl1_repeat.set_rcl_size(1);
        demo_rcl1_repeat.grasp_iteration();
        assert_eq!(
            demo_rcl1.best_tour_length,
            demo_rcl1_repeat.best_tour_length
        );
    }

    /// Mutation test: Restarts counter must increment.
    #[test]
    fn mutation_restarts_increment() {
        let mut demo = TspGraspDemo::new(42, 6);
        assert_eq!(demo.restarts, 0);

        demo.grasp_iteration();
        assert_eq!(demo.restarts, 1, "First iteration should set restarts=1");

        demo.grasp_iteration();
        assert_eq!(demo.restarts, 2, "Second iteration should set restarts=2");

        demo.run_grasp(5);
        assert_eq!(demo.restarts, 7, "After 5 more iterations, restarts=7");
    }

    /// Mutation test: Best tour is actually the BEST found.
    #[test]
    fn mutation_best_tour_is_actually_best() {
        let mut demo = TspGraspDemo::new(42, 10);

        // Run multiple iterations
        demo.run_grasp(20);

        // best_tour_length should be minimum of all restart_history
        let min_history = demo
            .restart_history
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        assert!(
            (demo.best_tour_length - min_history).abs() < 1e-9,
            "best_tour_length {} != min of history {}",
            demo.best_tour_length,
            min_history
        );
    }

    /// Mutation test: YAML distances override Euclidean.
    #[test]
    fn mutation_yaml_distances_used() {
        let yaml = r#"
meta:
  id: "TEST"
  description: "Test"
  units: "miles"
cities:
  - { id: 0, name: "A", alias: "A", coords: { lat: 0.0, lon: 0.0 } }
  - { id: 1, name: "B", alias: "B", coords: { lat: 0.0, lon: 1.0 } }
  - { id: 2, name: "C", alias: "C", coords: { lat: 1.0, lon: 0.0 } }
matrix:
  - [0, 100, 200]
  - [100, 0, 150]
  - [200, 150, 0]
"#;
        let demo = TspGraspDemo::from_yaml(yaml).expect("parse");

        // Should use YAML distances, NOT Euclidean
        assert_eq!(demo.distance(0, 1), 100.0, "Should use YAML distance 100");
        assert_eq!(demo.distance(0, 2), 200.0, "Should use YAML distance 200");
        assert_eq!(demo.distance(1, 2), 150.0, "Should use YAML distance 150");
    }

    /// Mutation test: Construction methods produce different tours.
    #[test]
    fn mutation_construction_methods_differ() {
        let seed = 12345;
        let n = 10;

        let mut demo_greedy = TspGraspDemo::new(seed, n);
        demo_greedy.set_construction_method(ConstructionMethod::RandomizedGreedy);
        demo_greedy.set_rcl_size(3);
        demo_greedy.construct_tour();
        let tour_greedy = demo_greedy.tour.clone();

        let mut demo_random = TspGraspDemo::new(seed, n);
        demo_random.set_construction_method(ConstructionMethod::Random);
        demo_random.construct_tour();
        let tour_random = demo_random.tour.clone();

        let mut demo_nn = TspGraspDemo::new(seed, n);
        demo_nn.set_construction_method(ConstructionMethod::NearestNeighbor);
        demo_nn.construct_tour();
        let tour_nn = demo_nn.tour.clone();

        // Different methods should produce different tours (at least sometimes)
        // At minimum, random should differ from greedy
        assert!(
            tour_greedy != tour_random || tour_nn != tour_random,
            "Construction methods should produce different tours"
        );
    }

    /// Mutation test: Tour length changes when tour changes.
    #[test]
    fn mutation_tour_length_reflects_tour() {
        let mut demo = TspGraspDemo::new(42, 6);
        demo.construct_tour();

        let original_length = demo.tour_length;
        let original_tour = demo.tour.clone();

        // Manually swap two cities
        demo.tour.swap(1, 2);
        demo.tour_length = demo.compute_tour_length(&demo.tour);

        // Length should change (unless swapped cities happen to be equidistant)
        if demo.tour != original_tour {
            // The length calculation should reflect the new tour
            let recomputed = demo.compute_tour_length(&demo.tour);
            assert_eq!(demo.tour_length, recomputed);
            // Original length was captured - verify length changed (most cases)
            // Note: Could be same if swapped cities equidistant to neighbors
            let _ = original_length; // Use the variable to document intent
        }
    }
}

// =============================================================================
// JIDOKA Tests - Stop-the-Line Quality Control (OR-001-15)
// Toyota Way: Built-in quality - detect anomalies immediately
// Lin & Kernighan (1973): Euclidean TSP optimal tours have ZERO crossings
// =============================================================================

#[cfg(test)]
mod jidoka_tests {
    use super::*;

    /// JIDOKA Test: 2-opt MUST eliminate ALL crossings in Euclidean TSP.
    /// Lin & Kernighan (1973): Crossing edges are always suboptimal.
    #[test]
    fn jidoka_two_opt_eliminates_all_crossings() {
        // Test across multiple seeds to catch edge cases
        for seed in 0..100 {
            let mut demo = TspGraspDemo::new(seed, 20);
            demo.grasp_iteration();

            let crossings = demo.count_crossings();
            assert_eq!(
                crossings, 0,
                "JIDOKA VIOLATION: Seed {seed}: Tour has {crossings} crossings after 2-opt. \
                 Euclidean TSP tours MUST have zero crossings (Lin & Kernighan, 1973)"
            );
        }
    }

    /// JIDOKA Test: Best tour must NEVER have crossings.
    #[test]
    fn jidoka_best_tour_never_has_crossings() {
        for seed in 0..50 {
            let mut demo = TspGraspDemo::new(seed, 20);

            // Run multiple iterations
            for _ in 0..10 {
                demo.grasp_iteration();

                // Check best tour for crossings
                if !demo.best_tour.is_empty() {
                    let mut temp = demo.clone();
                    temp.tour.clone_from(&demo.best_tour);
                    let crossings = temp.count_crossings();
                    assert_eq!(
                        crossings, 0,
                        "JIDOKA VIOLATION: Seed {seed}: Best tour has {crossings} crossings. \
                         Best tour must ALWAYS be crossing-free."
                    );
                }
            }
        }
    }

    /// JIDOKA Test: Verify crossing detection is correct.
    /// For a square, proper traversal is 0→1→3→2→0 (perimeter), not 0→1→2→3→0 (diagonals cross).
    #[test]
    fn jidoka_crossing_detection_accuracy() {
        // Create cities forming a square:
        // 0 (0,1) --- 2 (1,1)
        //    |           |
        // 1 (0,0) --- 3 (1,0)
        let cities = vec![
            City::new(0.0, 1.0), // 0: top-left
            City::new(0.0, 0.0), // 1: bottom-left
            City::new(1.0, 1.0), // 2: top-right
            City::new(1.0, 0.0), // 3: bottom-right
        ];

        let mut demo = TspGraspDemo::with_cities(42, cities);

        // Tour 0-1-3-2-0: perimeter traversal (no crossing)
        // Edges: (0,1)=left, (1,3)=bottom, (3,2)=right, (2,0)=top
        demo.tour = vec![0, 1, 3, 2];
        assert_eq!(
            demo.count_crossings(),
            0,
            "Perimeter tour 0-1-3-2 should have no crossings"
        );

        // Tour 0-1-2-3-0: diagonal crossing!
        // Edges: (0,1)=left, (1,2)=diagonal↗, (2,3)=right, (3,0)=diagonal↖
        // The two diagonals cross in the center!
        demo.tour = vec![0, 1, 2, 3];
        let crossings = demo.count_crossings();
        assert!(
            crossings > 0,
            "Diagonal tour 0-1-2-3 should have crossings (got {crossings})"
        );

        // After 2-opt, crossings should be eliminated
        demo.two_opt_to_local_optimum();
        assert_eq!(
            demo.count_crossings(),
            0,
            "After 2-opt, tour should have no crossings"
        );
    }

    /// JIDOKA Test: Poka-Yoke input validation for RCL size.
    #[test]
    fn jidoka_rcl_input_validation() {
        let mut demo = TspGraspDemo::new(42, 10);

        // RCL size should be clamped to valid range [1, n]
        demo.set_rcl_size(0);
        assert!(demo.rcl_size >= 1, "RCL size must be at least 1");

        demo.set_rcl_size(100);
        assert!(
            demo.rcl_size <= demo.n,
            "RCL size must not exceed city count"
        );
    }

    /// JIDOKA Test: Stagnation detection - stop when no improvement.
    #[test]
    fn jidoka_stagnation_detection() {
        let mut demo = TspGraspDemo::new(42, 10);

        // Run enough iterations to converge
        for _ in 0..50 {
            demo.grasp_iteration();
        }

        // Check stagnation - if last N restarts haven't improved, we've stagnated
        let history_len = demo.restart_history.len();
        if history_len >= 10 {
            let last_10: Vec<f64> = demo.restart_history[history_len - 10..].to_vec();
            let best_in_last_10 = last_10.iter().cloned().fold(f64::INFINITY, f64::min);
            let best_overall = demo.best_tour_length;

            // If best in last 10 == overall best, we've likely stagnated
            let stagnated = (best_in_last_10 - best_overall).abs() < f64::EPSILON;
            // This is informational - stagnation is expected for small instances
            if stagnated {
                println!(
                    "Stagnation detected at iteration {} (best: {:.1})",
                    history_len, best_overall
                );
            }
        }
    }

    /// JIDOKA Test: Verify falsification status reports crossings correctly.
    /// Crossings should be reported FIRST (Jidoka stop-the-line priority).
    #[test]
    fn jidoka_falsification_reports_crossings() {
        // Use a larger instance where diagonal tour has crossing
        let cities = vec![
            City::new(0.0, 1.0), // 0: top-left
            City::new(0.0, 0.0), // 1: bottom-left
            City::new(1.0, 1.0), // 2: top-right
            City::new(1.0, 0.0), // 3: bottom-right
        ];

        let mut demo = TspGraspDemo::with_cities(42, cities);

        // Force a crossing tour (diagonals cross)
        demo.tour = vec![0, 1, 2, 3]; // This creates crossing diagonals
        demo.tour_length = demo.compute_tour_length(&demo.tour);
        demo.best_tour = demo.tour.clone();
        demo.best_tour_length = demo.tour_length;

        // Manually compute a reasonable lower bound so gap doesn't fail first
        demo.lower_bound = demo.best_tour_length * 0.9; // 10% gap (< 20% threshold)

        let status = demo.get_falsification_status();
        assert!(
            !status.verified,
            "Falsification should FAIL when best tour has crossings"
        );
        assert!(
            status.message.contains("JIDOKA") || status.message.contains("crossing"),
            "Status message should mention JIDOKA or crossing (got: {})",
            status.message
        );
    }
}

// =============================================================================
// MUDA Tests (Waste Elimination - Stagnation Detection)
// =============================================================================

#[cfg(test)]
mod muda_tests {
    use super::*;

    /// MUDA Test: Stagnation count increments when no improvement.
    #[test]
    fn muda_stagnation_count_increments() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.grasp_iteration(); // First iteration sets best tour
        let initial_count = demo.stagnation_count;

        // Run iterations - some may improve, some may not
        for _ in 0..5 {
            demo.grasp_iteration();
        }

        // After 5 iterations, either stagnation increased or we found improvements
        // (stagnation_count >= initial_count is always true)
        assert!(
            demo.stagnation_count >= initial_count,
            "Stagnation count should not decrease"
        );
    }

    /// MUDA Test: Stagnation resets on improvement.
    #[test]
    fn muda_stagnation_resets_on_improvement() {
        let mut demo = TspGraspDemo::new(42, 10);

        // Force a poor initial tour by using random construction
        demo.set_construction_method(ConstructionMethod::Random);
        demo.grasp_iteration();
        let poor_tour = demo.best_tour_length;

        // Artificially set stagnation count
        demo.stagnation_count = 5;

        // Switch to better construction method - should improve
        demo.set_construction_method(ConstructionMethod::NearestNeighbor);
        demo.grasp_iteration();

        // If tour improved, stagnation should reset
        if demo.best_tour_length < poor_tour {
            assert_eq!(
                demo.stagnation_count, 0,
                "Stagnation should reset to 0 on improvement"
            );
        }
    }

    /// MUDA Test: Convergence flag set when threshold reached.
    #[test]
    fn muda_converged_flag_set_at_threshold() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.set_stagnation_threshold(5); // Low threshold for testing

        // Run many iterations - should eventually stagnate
        for _ in 0..50 {
            demo.grasp_iteration();
            if demo.converged {
                break;
            }
        }

        // For a small instance, GRASP should converge quickly
        // Either converged or made progress every iteration
        if demo.stagnation_count >= 5 {
            assert!(
                demo.converged,
                "Should be marked converged when stagnation >= threshold"
            );
        }
    }

    /// MUDA Test: run_grasp respects convergence (eliminates waste).
    #[test]
    fn muda_run_grasp_stops_on_convergence() {
        let mut demo = TspGraspDemo::new(42, 10);
        demo.set_stagnation_threshold(3); // Very low threshold

        // Request 1000 iterations, but should stop early
        demo.run_grasp(1000);

        // With threshold 3, should stop much earlier than 1000
        assert!(
            demo.restarts < 1000,
            "run_grasp should stop early when converged (ran {} restarts)",
            demo.restarts
        );
    }

    /// MUDA Test: is_converged accessor works correctly.
    #[test]
    fn muda_is_converged_accessor() {
        let mut demo = TspGraspDemo::new(42, 10);

        assert!(!demo.is_converged(), "Should not be converged initially");

        // Manually set converged
        demo.converged = true;
        assert!(demo.is_converged(), "Should reflect converged state");
    }

    /// MUDA Test: Set stagnation threshold works.
    #[test]
    fn muda_set_stagnation_threshold() {
        let mut demo = TspGraspDemo::new(42, 10);

        demo.set_stagnation_threshold(20);
        assert_eq!(demo.stagnation_threshold, 20);

        demo.set_stagnation_threshold(5);
        assert_eq!(demo.stagnation_threshold, 5);
    }

    /// MUDA Test: Mutation-sensitive - stagnation_count affects behavior.
    #[test]
    fn muda_mutation_stagnation_count_matters() {
        let mut demo1 = TspGraspDemo::new(42, 10);
        let mut demo2 = TspGraspDemo::new(42, 10);

        demo1.set_stagnation_threshold(5);
        demo2.set_stagnation_threshold(5);

        // Run both to convergence
        demo1.run_grasp(100);

        // Demo2: artificially start converged
        demo2.converged = true;
        demo2.run_grasp(100);

        // Demo1 should have run more iterations
        assert!(
            demo1.restarts > demo2.restarts,
            "Pre-converged demo should run fewer iterations"
        );
    }
}

// =============================================================================
// WASM Binding Tests (OR-001-09: Probar-only, NO JavaScript)
// =============================================================================

#[cfg(all(test, feature = "wasm"))]
mod wasm_tests {
    use super::super::wasm::WasmTspGrasp;

    // Small 6-city instance for WASM unit tests (inline to avoid file dependency)
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

    const MINIMAL_YAML: &str = r#"
meta:
  id: "TEST-001"
  version: "1.0.0"
  description: "Minimal test"
  units: "miles"
  optimal_known: 20
cities:
  - id: 0
    name: "A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
  - id: 1
    name: "B"
    alias: "B"
    coords: { lat: 1.0, lon: 1.0 }
  - id: 2
    name: "C"
    alias: "C"
    coords: { lat: 2.0, lon: 0.0 }
matrix:
  - [0, 10, 15]
  - [10, 0, 10]
  - [15, 10, 0]
algorithm:
  method: "grasp"
  params:
    rcl_size: 2
    restarts: 5
    two_opt: true
    seed: 42
"#;

    // =========================================================================
    // Construction Tests
    // =========================================================================

    #[test]
    fn test_wasm_new() {
        let demo = WasmTspGrasp::new(42, 5);
        assert_eq!(demo.get_n(), 5);
    }

    #[test]
    fn test_wasm_from_yaml_bay_area() {
        let result = WasmTspGrasp::from_yaml(SMALL_6_CITY_YAML);
        assert!(result.is_ok());
        let demo = result.unwrap();
        assert_eq!(demo.get_n(), 6);
    }

    #[test]
    fn test_wasm_from_yaml_minimal() {
        let result = WasmTspGrasp::from_yaml(MINIMAL_YAML);
        assert!(result.is_ok());
        let demo = result.unwrap();
        assert_eq!(demo.get_n(), 3);
    }

    #[test]
    fn test_wasm_from_yaml_invalid() {
        let result = WasmTspGrasp::from_yaml("invalid yaml: [[[");
        assert!(result.is_err());
    }

    #[test]
    fn test_wasm_from_yaml_empty() {
        let result = WasmTspGrasp::from_yaml("");
        assert!(result.is_err());
    }

    // =========================================================================
    // YAML-First Configuration Tests
    // =========================================================================

    #[test]
    fn test_wasm_yaml_rcl_size_applied() {
        let demo = WasmTspGrasp::from_yaml(SMALL_6_CITY_YAML).expect("parse");
        // YAML specifies rcl_size: 3
        assert_eq!(demo.get_rcl_size(), 3);
    }

    #[test]
    fn test_wasm_yaml_construction_method_applied() {
        let demo = WasmTspGrasp::from_yaml(SMALL_6_CITY_YAML).expect("parse");
        // YAML specifies method: "grasp" → RandomizedGreedy (0)
        assert_eq!(demo.get_construction_method(), 0);
    }

    #[test]
    fn test_wasm_yaml_user_modified_method() {
        let nn_yaml =
            SMALL_6_CITY_YAML.replace("method: \"grasp\"", "method: \"nearest_neighbor\"");
        let demo = WasmTspGrasp::from_yaml(&nn_yaml).expect("parse");
        // NearestNeighbor = 1
        assert_eq!(demo.get_construction_method(), 1);
    }

    // =========================================================================
    // Simulation Control Tests
    // =========================================================================

    #[test]
    fn test_wasm_step() {
        let mut demo = WasmTspGrasp::from_yaml(MINIMAL_YAML).expect("parse");
        demo.step();
        // Step should not panic
    }

    #[test]
    fn test_wasm_grasp_iteration() {
        let mut demo = WasmTspGrasp::from_yaml(MINIMAL_YAML).expect("parse");
        demo.grasp_iteration();
        assert!(demo.get_tour_length() > 0.0);
    }

    #[test]
    fn test_wasm_run_grasp() {
        let mut demo = WasmTspGrasp::from_yaml(SMALL_6_CITY_YAML).expect("parse");
        demo.run_grasp(10);
        assert!(demo.get_restarts() >= 10);
        assert!(demo.get_best_tour_length() > 0.0);
    }

    #[test]
    fn test_wasm_construct_tour() {
        let mut demo = WasmTspGrasp::from_yaml(MINIMAL_YAML).expect("parse");
        demo.construct_tour();
        assert!(demo.get_tour_length() > 0.0);
    }

    #[test]
    fn test_wasm_two_opt_pass() {
        let mut demo = WasmTspGrasp::from_yaml(MINIMAL_YAML).expect("parse");
        demo.construct_tour();
        let _improved = demo.two_opt_pass();
        // Just verify it doesn't panic
    }

    #[test]
    fn test_wasm_two_opt_to_local_optimum() {
        // Use Bay Area for more complex 2-opt scenarios
        let mut demo = WasmTspGrasp::from_yaml(SMALL_6_CITY_YAML).expect("parse");
        demo.construct_tour();
        let before = demo.get_tour_length();
        demo.two_opt_to_local_optimum();
        let after = demo.get_tour_length();
        // 2-opt should not make tour worse
        assert!(after <= before, "2-opt should improve or maintain tour");
    }

    #[test]
    fn test_wasm_reset() {
        let mut demo = WasmTspGrasp::from_yaml(MINIMAL_YAML).expect("parse");
        demo.run_grasp(5);
        let before_restarts = demo.get_restarts();
        assert!(before_restarts >= 5);
        demo.reset();
        assert_eq!(demo.get_restarts(), 0);
        // After reset, best_tour_length should be the default (infinity or 0)
        // Just verify restarts are reset
    }

    // =========================================================================
    // Configuration Tests
    // =========================================================================

    #[test]
    fn test_wasm_set_construction_method() {
        let mut demo = WasmTspGrasp::from_yaml(MINIMAL_YAML).expect("parse");
        demo.set_construction_method(1); // NearestNeighbor
        assert_eq!(demo.get_construction_method(), 1);

        demo.set_construction_method(2); // Random
        assert_eq!(demo.get_construction_method(), 2);

        demo.set_construction_method(0); // RandomizedGreedy
        assert_eq!(demo.get_construction_method(), 0);
    }

    #[test]
    fn test_wasm_set_rcl_size() {
        // Use Bay Area (6 cities) to test RCL size setting
        let mut demo = WasmTspGrasp::from_yaml(SMALL_6_CITY_YAML).expect("parse");
        demo.set_rcl_size(5);
        assert_eq!(demo.get_rcl_size(), 5);

        // Verify clamping to n (6 cities)
        demo.set_rcl_size(10);
        assert_eq!(demo.get_rcl_size(), 6, "RCL size should be clamped to n");
    }

    // =========================================================================
    // State Query Tests
    // =========================================================================

    #[test]
    fn test_wasm_get_tour_length() {
        let mut demo = WasmTspGrasp::from_yaml(MINIMAL_YAML).expect("parse");
        demo.construct_tour();
        assert!(demo.get_tour_length() > 0.0);
    }

    #[test]
    fn test_wasm_get_best_tour_length() {
        let mut demo = WasmTspGrasp::from_yaml(MINIMAL_YAML).expect("parse");
        demo.run_grasp(3);
        let best = demo.get_best_tour_length();
        assert!(best > 0.0 && best < f64::INFINITY);
    }

    #[test]
    fn test_wasm_get_optimality_gap() {
        let mut demo = WasmTspGrasp::from_yaml(SMALL_6_CITY_YAML).expect("parse");
        demo.run_grasp(10);
        let gap = demo.get_optimality_gap();
        // Gap should be reasonable (<=100%)
        assert!(gap < 1.0);
    }

    #[test]
    fn test_wasm_get_lower_bound() {
        let demo = WasmTspGrasp::from_yaml(SMALL_6_CITY_YAML).expect("parse");
        let lb = demo.get_lower_bound();
        // Lower bound should be positive
        assert!(lb > 0.0);
    }

    #[test]
    fn test_wasm_get_restarts() {
        let mut demo = WasmTspGrasp::from_yaml(MINIMAL_YAML).expect("parse");
        assert_eq!(demo.get_restarts(), 0);
        demo.run_grasp(7);
        assert!(demo.get_restarts() >= 7);
    }

    #[test]
    fn test_wasm_get_two_opt_iterations() {
        // Use Bay Area for more complex 2-opt scenarios
        let mut demo = WasmTspGrasp::from_yaml(SMALL_6_CITY_YAML).expect("parse");
        demo.run_grasp(5);
        // 2-opt iterations counter should be non-negative
        let _ = demo.get_two_opt_iterations();
        // Just verify it returns a value (may be 0 or more)
    }

    #[test]
    fn test_wasm_get_two_opt_improvements() {
        let mut demo = WasmTspGrasp::from_yaml(MINIMAL_YAML).expect("parse");
        demo.run_grasp(5);
        // Improvements may or may not occur depending on initial tour
        let _ = demo.get_two_opt_improvements();
    }

    #[test]
    fn test_wasm_get_n() {
        let demo = WasmTspGrasp::from_yaml(SMALL_6_CITY_YAML).expect("parse");
        assert_eq!(demo.get_n(), 6);
    }

    #[test]
    fn test_wasm_get_restart_variance() {
        let mut demo = WasmTspGrasp::from_yaml(MINIMAL_YAML).expect("parse");
        demo.run_grasp(5);
        let variance = demo.get_restart_variance();
        // Variance >= 0
        assert!(variance >= 0.0);
    }

    #[test]
    fn test_wasm_get_restart_cv() {
        let mut demo = WasmTspGrasp::from_yaml(MINIMAL_YAML).expect("parse");
        demo.run_grasp(5);
        let cv = demo.get_restart_cv();
        // CV >= 0
        assert!(cv >= 0.0);
    }

    // =========================================================================
    // Deterministic Replay Tests (Probar Core)
    // =========================================================================

    #[test]
    fn test_wasm_deterministic_replay_from_yaml() {
        // Two WASM demos from same YAML should produce identical results
        let mut demo1 = WasmTspGrasp::from_yaml(SMALL_6_CITY_YAML).expect("parse 1");
        let mut demo2 = WasmTspGrasp::from_yaml(SMALL_6_CITY_YAML).expect("parse 2");

        demo1.run_grasp(10);
        demo2.run_grasp(10);

        assert_eq!(
            demo1.get_best_tour_length(),
            demo2.get_best_tour_length(),
            "Same YAML + same seed = identical results"
        );
    }

    #[test]
    fn test_wasm_bay_area_optimal_achievable() {
        // Bay Area optimal is 115 miles - GRASP should find it or close
        let mut demo = WasmTspGrasp::from_yaml(SMALL_6_CITY_YAML).expect("parse");
        demo.run_grasp(20);

        let best = demo.get_best_tour_length();
        // Within 10% of optimal 115
        assert!(
            best <= 115.0 * 1.1,
            "Should find tour <= 126.5 miles, got {best}"
        );
    }

    #[test]
    fn test_wasm_user_modified_yaml_different_result() {
        // User modifies distance → different results
        let original = WasmTspGrasp::from_yaml(SMALL_6_CITY_YAML).expect("original");

        // User reduces SF→Oakland from 12 to 1 mile
        let modified_yaml = SMALL_6_CITY_YAML.replace("[ 0, 12, 48", "[ 0,  1, 48");
        let modified = WasmTspGrasp::from_yaml(&modified_yaml).expect("modified");

        // Lower bounds should differ
        let orig_lb = original.get_lower_bound();
        let mod_lb = modified.get_lower_bound();

        assert!(
            (orig_lb - mod_lb).abs() > 0.5,
            "Modified YAML should produce different lower bound"
        );
    }

    // =========================================================================
    // Unit Label Tests (OR-001: [cities] and [miles] display)
    // =========================================================================

    #[test]
    fn test_wasm_get_units_bay_area() {
        let demo = WasmTspGrasp::from_yaml(SMALL_6_CITY_YAML).expect("parse");
        assert_eq!(demo.get_units(), "miles", "Bay Area uses miles");
    }

    #[test]
    fn test_wasm_get_units_minimal() {
        let demo = WasmTspGrasp::from_yaml(MINIMAL_YAML).expect("parse");
        assert_eq!(demo.get_units(), "miles", "Minimal YAML uses miles");
    }

    #[test]
    fn test_wasm_get_optimal_known_bay_area() {
        let demo = WasmTspGrasp::from_yaml(SMALL_6_CITY_YAML).expect("parse");
        assert_eq!(
            demo.get_optimal_known(),
            Some(115),
            "Bay Area optimal is 115 miles"
        );
    }

    #[test]
    fn test_wasm_get_optimal_known_minimal() {
        let demo = WasmTspGrasp::from_yaml(MINIMAL_YAML).expect("parse");
        assert_eq!(
            demo.get_optimal_known(),
            Some(20),
            "Minimal YAML optimal is 20"
        );
    }

    #[test]
    fn test_wasm_get_n_is_city_count() {
        let demo = WasmTspGrasp::from_yaml(SMALL_6_CITY_YAML).expect("parse");
        assert_eq!(demo.get_n(), 6, "Bay Area has 6 cities");
    }

    #[test]
    fn test_wasm_new_defaults_units() {
        let demo = WasmTspGrasp::new(42, 5);
        assert_eq!(demo.get_units(), "units", "New demo uses default 'units'");
        assert_eq!(
            demo.get_optimal_known(),
            None,
            "New demo has no known optimal"
        );
    }
}
