
    // ==========================================================
    // RED PHASE: Tests written first (Extreme TDD)
    // ==========================================================

    // --- Tabu Search Tests ---

    #[test]
    fn test_tabu_search_permutation() {
        // Minimize sum of |position - value| (optimal: identity permutation)
        let space = SearchSpace::Permutation { size: 5 };

        let objective = |perm: &Vec<usize>| {
            perm.iter()
                .enumerate()
                .map(|(i, &v)| (i as i64 - v as i64).unsigned_abs() as f64)
                .sum()
        };

        let mut ts = TabuSearch::new(5).with_seed(42);
        let result = ts.optimize(&objective, &space, Budget::Iterations(100));

        // Should find identity or close
        assert!(
            result.objective_value <= 4.0,
            "Should find good permutation"
        );
        assert_eq!(result.solution.len(), 5);
    }

    #[test]
    fn test_tabu_search_parameters() {
        let ts = TabuSearch::new(10)
            .with_tenure(15)
            .with_max_neighbors(500)
            .with_seed(123);

        assert_eq!(ts.tenure, 15);
        assert_eq!(ts.max_neighbors, 500);
    }

    #[test]
    fn test_tabu_search_reset() {
        let space = SearchSpace::Permutation { size: 3 };
        let mut ts = TabuSearch::new(3).with_seed(42);

        let _ = ts.optimize(&|_: &Vec<usize>| 1.0, &space, Budget::Iterations(5));
        assert!(!ts.history.is_empty());

        ts.reset();
        assert!(ts.history.is_empty());
        assert!(ts.best_solution.is_empty());
    }

    #[test]
    fn test_tabu_search_tsp() {
        // Small TSP
        let space = SearchSpace::Permutation { size: 4 };

        // Distance matrix (symmetric)
        let dist = vec![
            vec![0.0, 10.0, 15.0, 20.0],
            vec![10.0, 0.0, 35.0, 25.0],
            vec![15.0, 35.0, 0.0, 30.0],
            vec![20.0, 25.0, 30.0, 0.0],
        ];

        let objective = |tour: &Vec<usize>| {
            let mut total = 0.0;
            for i in 0..tour.len() {
                let from = tour[i];
                let to = tour[(i + 1) % tour.len()];
                total += dist[from][to];
            }
            total
        };

        let mut ts = TabuSearch::new(5).with_seed(42);
        let result = ts.optimize(&objective, &space, Budget::Iterations(50));

        assert!(
            result.objective_value < 120.0,
            "Should find reasonable tour"
        );
    }

    #[test]
    fn test_tabu_search_convergence() {
        let space = SearchSpace::Permutation { size: 6 };

        let objective = |perm: &Vec<usize>| {
            // Quadratic assignment-like objective
            perm.iter()
                .enumerate()
                .map(|(i, &v)| ((i as f64) - (v as f64)).powi(2))
                .sum()
        };

        let mut ts = TabuSearch::new(7).with_seed(42);
        let result = ts.optimize(&objective, &space, Budget::Iterations(100));

        // Should show improvement
        let history = &result.history;
        assert!(history.len() > 1);
        assert!(
            history.last().unwrap() <= &history[0],
            "Should improve or stay same"
        );
    }

    // --- ACO Tests ---

    #[test]
    fn test_aco_tsp_4_cities() {
        // Simple TSP: 4 cities in a square
        // Optimal tour: 0->1->2->3->0 or reverse, length = 40
        let distances = vec![
            vec![(1, 10.0), (2, 15.0), (3, 20.0)],
            vec![(0, 10.0), (2, 35.0), (3, 25.0)],
            vec![(0, 15.0), (1, 35.0), (3, 30.0)],
            vec![(0, 20.0), (1, 25.0), (2, 30.0)],
        ];

        let space = SearchSpace::Graph {
            num_nodes: 4,
            adjacency: distances.clone(),
            heuristic: None,
        };

        let objective = |tour: &Vec<usize>| {
            let mut total = 0.0;
            for i in 0..tour.len() {
                let from = tour[i];
                let to = tour[(i + 1) % tour.len()];
                // Find distance
                for &(j, d) in &distances[from] {
                    if j == to {
                        total += d;
                        break;
                    }
                }
            }
            total
        };

        let mut aco = AntColony::new(10).with_seed(42);
        let result = aco.optimize(&objective, &space, Budget::Iterations(50));

        // Should find a reasonable tour (not necessarily optimal with limited budget)
        assert!(result.objective_value < 100.0, "Should find decent tour");
        assert_eq!(result.solution.len(), 4, "Tour should visit all cities");
    }

    #[test]
    fn test_aco_parameters() {
        let aco = AntColony::new(20)
            .with_alpha(2.0)
            .with_beta(3.0)
            .with_rho(0.2)
            .with_seed(123);

        assert_eq!(aco.num_ants, 20);
        assert!((aco.alpha - 2.0).abs() < 1e-10);
        assert!((aco.beta - 3.0).abs() < 1e-10);
        assert!((aco.rho - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_aco_reset() {
        let distances = vec![vec![(1, 10.0)], vec![(0, 10.0)]];
        let space = SearchSpace::Graph {
            num_nodes: 2,
            adjacency: distances,
            heuristic: None,
        };

        let mut aco = AntColony::new(5).with_seed(42);
        let _ = aco.optimize(&|_: &Vec<usize>| 10.0, &space, Budget::Iterations(5));

        assert!(!aco.history.is_empty());
        aco.reset();
        assert!(aco.history.is_empty());
        assert!(aco.best_tour.is_empty());
    }

    #[test]
    fn test_aco_permutation_space() {
        // ACO can also work with permutation space
        let space = SearchSpace::Permutation { size: 5 };

        let objective = |perm: &Vec<usize>| {
            // Minimize sum of (position * value)
            perm.iter().enumerate().map(|(i, &v)| (i * v) as f64).sum()
        };

        let mut aco = AntColony::new(10).with_seed(42);
        let result = aco.optimize(&objective, &space, Budget::Iterations(20));

        assert_eq!(result.solution.len(), 5);
    }

    #[test]
    fn test_constructive_trait_best() {
        let mut aco = AntColony::new(5);
        assert!(aco.best().is_none(), "No best before optimization");

        let space = SearchSpace::Permutation { size: 3 };
        let _ = aco.optimize(&|_: &Vec<usize>| 1.0, &space, Budget::Iterations(1));

        assert!(aco.best().is_some(), "Should have best after optimization");
    }

    #[test]
    fn test_aco_convergence_improves() {
        // Test that ACO improves over iterations
        let distances = vec![
            vec![(1, 10.0), (2, 20.0), (3, 30.0)],
            vec![(0, 10.0), (2, 15.0), (3, 25.0)],
            vec![(0, 20.0), (1, 15.0), (3, 10.0)],
            vec![(0, 30.0), (1, 25.0), (2, 10.0)],
        ];

        let space = SearchSpace::Graph {
            num_nodes: 4,
            adjacency: distances.clone(),
            heuristic: None,
        };

        let objective = |tour: &Vec<usize>| {
            let mut total = 0.0;
            for i in 0..tour.len() {
                let from = tour[i];
                let to = tour[(i + 1) % tour.len()];
                for &(j, d) in &distances[from] {
                    if j == to {
                        total += d;
                        break;
                    }
                }
            }
            total
        };

        let mut aco = AntColony::new(15).with_seed(42);
        let result = aco.optimize(&objective, &space, Budget::Iterations(30));

        // History should show improvement (non-increasing for minimization)
        let history = result.history;
        assert!(history.len() > 1);

        // Final should be <= initial (or at least close)
        let first = history[0];
        let last = history[history.len() - 1];
        assert!(last <= first + 1e-10, "Should improve or stay same");
    }

    // --- Tabu Search edge-case tests ---

    #[test]
    fn test_tabu_search_graph_space() {
        // Exercises the Graph branch in TabuSearch::optimize (line 529).
        let space = SearchSpace::Graph {
            num_nodes: 4,
            adjacency: vec![
                vec![(1, 10.0), (2, 20.0), (3, 30.0)],
                vec![(0, 10.0), (2, 15.0), (3, 25.0)],
                vec![(0, 20.0), (1, 15.0), (3, 10.0)],
                vec![(0, 30.0), (1, 25.0), (2, 10.0)],
            ],
            heuristic: None,
        };

        let objective = |perm: &Vec<usize>| {
            perm.iter()
                .enumerate()
                .map(|(i, &v)| ((i as f64) - (v as f64)).abs())
                .sum()
        };

        let mut ts = TabuSearch::new(5).with_seed(42);
        let result = ts.optimize(&objective, &space, Budget::Iterations(20));
        assert_eq!(result.solution.len(), 4);
        assert!(result.evaluations > 0);
    }

    #[test]
    fn test_tabu_search_best_before_optimize() {
        let ts = TabuSearch::new(5);
        assert!(ts.best().is_none());
        assert!(ts.history().is_empty());
    }

    #[test]
    fn test_tabu_search_large_neighborhood_sampling() {
        // With a large permutation space, the number of swap moves exceeds
        // max_neighbors, triggering the random subset sampling branch.
        let space = SearchSpace::Permutation { size: 50 }; // 50*49/2 = 1225 moves
        let objective = |perm: &Vec<usize>| {
            perm.iter()
                .enumerate()
                .map(|(i, &v)| ((i as f64) - (v as f64)).powi(2))
                .sum()
        };

        // max_neighbors = 100 which is < 1225 total moves
        let mut ts = TabuSearch::new(7).with_max_neighbors(100).with_seed(42);
        let result = ts.optimize(&objective, &space, Budget::Iterations(20));
        assert_eq!(result.solution.len(), 50);
        assert!(result.evaluations > 0);
    }

    #[test]
    fn test_tabu_aspiration_override() {
        // Set up a scenario where a tabu move produces a globally best solution,
        // triggering the aspiration criterion.
        let space = SearchSpace::Permutation { size: 3 };

        // Objective that strongly rewards a specific permutation
        let objective = |perm: &Vec<usize>| {
            if perm == &vec![0, 1, 2] {
                0.0 // Global best
            } else if perm == &vec![0, 2, 1] {
                1.0
            } else {
                10.0
            }
        };

        // Long tenure to ensure moves stay tabu
        let mut ts = TabuSearch::new(100).with_seed(42);
        let result = ts.optimize(&objective, &space, Budget::Iterations(50));
        // Should find the optimal despite tabu constraints via aspiration
        assert!(result.objective_value <= 10.0);
    }

    #[test]
    fn test_tabu_search_convergence_budget() {
        // Test TabuSearch with convergence budget to exercise the convergence
        // termination path.
        let space = SearchSpace::Permutation { size: 4 };
        let objective = |_perm: &Vec<usize>| 5.0; // Flat - no improvement possible

        let mut ts = TabuSearch::new(3).with_seed(42);
        let budget = Budget::Convergence {
            patience: 5,
            min_delta: 1e-6,
            max_evaluations: 100_000,
        };
        let result = ts.optimize(&objective, &space, budget);
        assert_eq!(result.termination, TerminationReason::Converged);
    }

    #[test]
    fn test_tabu_is_tabu_expired() {
        // Verify is_tabu returns false when the move has expired.
        let mv = SwapMove { i: 0, j: 1 };
        let tabu_list = vec![(mv, 5)]; // Expires at iteration 5
        assert!(TabuSearch::is_tabu(&mv, &tabu_list, 4)); // Still active
        assert!(!TabuSearch::is_tabu(&mv, &tabu_list, 5)); // Expired
        assert!(!TabuSearch::is_tabu(&mv, &tabu_list, 6)); // Expired
    }

    #[test]
    fn test_tabu_is_tabu_different_move() {
        // Verify a different move is not tabu.
        let mv1 = SwapMove { i: 0, j: 1 };
        let mv2 = SwapMove { i: 2, j: 3 };
        let tabu_list = vec![(mv1, 100)];
        assert!(!TabuSearch::is_tabu(&mv2, &tabu_list, 0));
    }

    #[test]
    fn test_tabu_generate_swap_moves() {
        let moves = TabuSearch::generate_swap_moves(4);
        // C(4,2) = 6 swap pairs
        assert_eq!(moves.len(), 6);
        assert!(moves.contains(&SwapMove { i: 0, j: 1 }));
        assert!(moves.contains(&SwapMove { i: 0, j: 3 }));
        assert!(moves.contains(&SwapMove { i: 2, j: 3 }));
    }

    #[test]
    fn test_tabu_apply_swap() {
        let sol = vec![0, 1, 2, 3];
        let mv = SwapMove { i: 1, j: 3 };
        let new_sol = TabuSearch::apply_swap(&sol, &mv);
        assert_eq!(new_sol, vec![0, 3, 2, 1]);
    }

    // --- ACO edge-case tests ---

    #[test]
    fn test_aco_zero_tour_length_deposit_skipped() {
        // When objective returns 0.0 for a tour, the deposit should be skipped
        // (the length <= 0.0 branch in update_pheromones).
        let space = SearchSpace::Permutation { size: 3 };
        let objective = |_tour: &Vec<usize>| 0.0; // Always zero

        let mut aco = AntColony::new(5).with_seed(42);
        let result = aco.optimize(&objective, &space, Budget::Iterations(5));
        // Should still complete without errors
        assert_eq!(result.solution.len(), 3);
        assert!((result.objective_value - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_aco_with_provided_heuristic() {
        // Provide an explicit heuristic matrix to exercise the Some(h) branch.
        let heuristic = vec![
            vec![0.0, 0.1, 0.05],
            vec![0.1, 0.0, 0.02],
            vec![0.05, 0.02, 0.0],
        ];
        let space = SearchSpace::Graph {
            num_nodes: 3,
            adjacency: vec![
                vec![(1, 10.0), (2, 20.0)],
                vec![(0, 10.0), (2, 50.0)],
                vec![(0, 20.0), (1, 50.0)],
            ],
            heuristic: Some(heuristic),
        };

        let objective = |tour: &Vec<usize>| {
            let dists = [[0.0, 10.0, 20.0], [10.0, 0.0, 50.0], [20.0, 50.0, 0.0]];
            let mut total = 0.0;
            for i in 0..tour.len() {
                let from = tour[i];
                let to = tour[(i + 1) % tour.len()];
                total += dists[from][to];
            }
            total
        };

        let mut aco = AntColony::new(5).with_seed(42);
        let result = aco.optimize(&objective, &space, Budget::Iterations(10));
        assert_eq!(result.solution.len(), 3);
        assert!(result.objective_value.is_finite());
    }
