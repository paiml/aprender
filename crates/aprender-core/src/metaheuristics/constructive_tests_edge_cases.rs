
    #[test]
    fn test_aco_budget_exhausted_termination() {
        // Use a tiny evaluation budget to hit BudgetExhausted termination.
        let space = SearchSpace::Permutation { size: 3 };
        let objective = |tour: &Vec<usize>| {
            tour.iter()
                .enumerate()
                .map(|(i, &v)| ((i as f64) - (v as f64)).abs())
                .sum()
        };

        let mut aco = AntColony::new(10).with_seed(42);
        let result = aco.optimize(&objective, &space, Budget::Evaluations(10));
        assert_eq!(result.termination, TerminationReason::BudgetExhausted);
    }

    #[test]
    fn test_aco_convergence_termination() {
        // Use convergence budget with flat objective to trigger convergence.
        let space = SearchSpace::Permutation { size: 3 };
        let objective = |_tour: &Vec<usize>| 42.0; // Flat

        let mut aco = AntColony::new(5).with_seed(42);
        let budget = Budget::Convergence {
            patience: 3,
            min_delta: 1e-6,
            max_evaluations: 100_000,
        };
        let result = aco.optimize(&objective, &space, budget);
        assert_eq!(result.termination, TerminationReason::Converged);
    }

    #[test]
    fn test_aco_num_ants_clamped_to_one() {
        // Verify that AntColony::new(0) clamps num_ants to 1.
        let aco = AntColony::new(0);
        assert_eq!(aco.num_ants, 1);
    }

    #[test]
    fn test_aco_rho_clamped() {
        // Verify rho is clamped to [0, 1].
        let aco = AntColony::new(5).with_rho(2.0);
        assert!((aco.rho - 1.0).abs() < 1e-10);

        let aco2 = AntColony::new(5).with_rho(-0.5);
        assert!((aco2.rho - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_aco_alpha_clamped() {
        // Verify alpha is clamped to non-negative.
        let aco = AntColony::new(5).with_alpha(-1.0);
        assert!((aco.alpha - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_aco_beta_clamped() {
        // Verify beta is clamped to non-negative.
        let aco = AntColony::new(5).with_beta(-1.0);
        assert!((aco.beta - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_tabu_tenure_clamped() {
        // Verify tenure is clamped to at least 1.
        let ts = TabuSearch::new(0);
        assert_eq!(ts.tenure, 1);

        let ts2 = TabuSearch::new(5).with_tenure(0);
        assert_eq!(ts2.tenure, 1);
    }

    #[test]
    fn test_tabu_max_neighbors_clamped() {
        // Verify max_neighbors is clamped to at least 1.
        let ts = TabuSearch::new(5).with_max_neighbors(0);
        assert_eq!(ts.max_neighbors, 1);
    }
