
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_creation() {
        let example = Example::new("ex1", "print('hello')", "println!(\"hello\");")
            .with_difficulty(Difficulty::Trivial)
            .with_tags(vec!["hello".to_string()]);

        assert_eq!(example.id, "ex1");
        assert_eq!(example.difficulty, Difficulty::Trivial);
        assert_eq!(example.tags.len(), 1);
    }

    #[test]
    fn test_difficulty_levels() {
        assert_eq!(Difficulty::Trivial.level(), 1);
        assert_eq!(Difficulty::Expert.level(), 5);
        assert_eq!(Difficulty::all().len(), 5);
    }

    #[test]
    fn test_eval_result_creation() {
        let mut result = EvalResult::new("model1", "py2rs", 1_000_000);
        assert_eq!(result.model_id, "model1");
        assert_eq!(result.model_size_bytes, 1_000_000);
        assert!(result.example_results.is_empty());

        // Add some example results
        result.add_example(ExampleResult::solved(
            "ex1",
            Difficulty::Easy,
            1,
            vec![100],
            vec![Duration::from_millis(50)],
        ));
        result.add_example(ExampleResult::solved(
            "ex2",
            Difficulty::Medium,
            2,
            vec![100, 150],
            vec![Duration::from_millis(50), Duration::from_millis(75)],
        ));
        result.add_example(ExampleResult::failed(
            "ex3",
            Difficulty::Hard,
            3,
            "Compile error",
            vec![100, 150, 200],
            vec![Duration::from_millis(50); 3],
        ));

        result.finalize(3);

        assert_eq!(result.example_results.len(), 3);
        assert!((result.overall_success_rate - 2.0 / 3.0).abs() < 0.01);
        assert!(result.avg_turns_to_success > 0.0);
    }

    #[test]
    fn test_eval_result_success_by_turn() {
        let mut result = EvalResult::new("model1", "task1", 1000);

        // 3 examples: turn 1, turn 2, failed
        result.add_example(ExampleResult::solved(
            "ex1",
            Difficulty::Easy,
            1,
            vec![100],
            vec![Duration::from_millis(10)],
        ));
        result.add_example(ExampleResult::solved(
            "ex2",
            Difficulty::Medium,
            2,
            vec![100, 100],
            vec![Duration::from_millis(10); 2],
        ));
        result.add_example(ExampleResult::failed(
            "ex3",
            Difficulty::Hard,
            3,
            "Failed",
            vec![100; 3],
            vec![Duration::from_millis(10); 3],
        ));

        result.finalize(3);

        // Turn 1: 1/3 solved
        assert!((result.success_at_turn(1) - 1.0 / 3.0).abs() < 0.01);
        // Turn 2: 2/3 solved
        assert!((result.success_at_turn(2) - 2.0 / 3.0).abs() < 0.01);
        // Turn 3: still 2/3 (failed didn't solve)
        assert!((result.success_at_turn(3) - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_model_comparison() {
        let mut comparison = ModelComparison::new("py2rs");

        // Add two models
        let mut model1 = EvalResult::new("small", "py2rs", 1_000_000);
        model1.add_example(ExampleResult::solved(
            "ex1",
            Difficulty::Easy,
            1,
            vec![100],
            vec![Duration::from_millis(10)],
        ));
        model1.finalize(3);

        let mut model2 = EvalResult::new("large", "py2rs", 10_000_000);
        model2.add_example(ExampleResult::solved(
            "ex1",
            Difficulty::Easy,
            1,
            vec![100],
            vec![Duration::from_millis(10)],
        ));
        model2.finalize(3);

        comparison.add_result(model1);
        comparison.add_result(model2);

        assert_eq!(comparison.results.len(), 2);

        // Both have 100% success
        let smallest = comparison.smallest_meeting_threshold(0.9);
        assert!(smallest.is_some());
        assert_eq!(smallest.unwrap().model_id, "small");
    }

    #[test]
    fn test_example_status() {
        let solved = ExampleStatus::Solved { turn: 2 };
        let failed = ExampleStatus::Failed {
            attempts: 3,
            last_error: "Error".to_string(),
        };
        let timeout = ExampleStatus::Timeout { turn: 1 };
        let skipped = ExampleStatus::Skipped {
            reason: "No deps".to_string(),
        };

        // Just verify they serialize
        assert!(serde_json::to_string(&solved).is_ok());
        assert!(serde_json::to_string(&failed).is_ok());
        assert!(serde_json::to_string(&timeout).is_ok());
        assert!(serde_json::to_string(&skipped).is_ok());
    }

    #[test]
    fn test_eval_suite_config_default() {
        let config = EvalSuiteConfig::default();
        assert_eq!(config.max_turns, 5);
        assert_eq!(config.turn_timeout_secs, 60);
        assert!(config.success_thresholds.contains(&0.9));
    }

    #[test]
    fn test_stratified_by_difficulty() {
        let mut comparison = ModelComparison::new("test");

        let mut result = EvalResult::new("model1", "test", 1000);
        result.add_example(ExampleResult::solved(
            "e1",
            Difficulty::Easy,
            1,
            vec![10],
            vec![Duration::ZERO],
        ));
        result.add_example(ExampleResult::solved(
            "e2",
            Difficulty::Easy,
            1,
            vec![10],
            vec![Duration::ZERO],
        ));
        result.add_example(ExampleResult::failed(
            "e3",
            Difficulty::Hard,
            3,
            "err",
            vec![10; 3],
            vec![Duration::ZERO; 3],
        ));
        result.finalize(3);
        comparison.add_result(result);

        let stratified = comparison.stratified_by_difficulty();
        let model1 = stratified.get("model1").unwrap();

        // Easy: 2/2 = 100%
        assert!((model1.get(&Difficulty::Easy).unwrap() - 1.0).abs() < 0.01);
        // Hard: 0/1 = 0%
        assert!((model1.get(&Difficulty::Hard).unwrap() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_generate_recommendations() {
        let mut comparison = ModelComparison::new("test");

        let mut small = EvalResult::new("small", "test", 1000);
        small.overall_success_rate = 0.8;
        comparison.add_result(small);

        let mut large = EvalResult::new("large", "test", 10000);
        large.overall_success_rate = 0.95;
        comparison.add_result(large);

        comparison.generate_recommendations();

        assert!(comparison.recommendations.len() >= 2);
        assert!(comparison
            .recommendations
            .iter()
            .any(|r| r.scenario.contains("footprint")));
        assert!(comparison
            .recommendations
            .iter()
            .any(|r| r.scenario.contains("accuracy")));
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_difficulty_name() {
        assert_eq!(Difficulty::Trivial.name(), "Trivial");
        assert_eq!(Difficulty::Easy.name(), "Easy");
        assert_eq!(Difficulty::Medium.name(), "Medium");
        assert_eq!(Difficulty::Hard.name(), "Hard");
        assert_eq!(Difficulty::Expert.name(), "Expert");
    }

    #[test]
    fn test_eval_result_finalize_empty() {
        let mut result = EvalResult::new("model1", "task1", 1000);
        result.finalize(3); // Should early return
        assert!(result.success_by_turn.is_empty());
        assert!((result.avg_turns_to_success - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_success_at_turn_out_of_bounds() {
        let result = EvalResult::new("model1", "task1", 1000);
        // No finalize, so success_by_turn is empty
        assert!((result.success_at_turn(1) - 0.0).abs() < 0.001);
        assert!((result.success_at_turn(10) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_fastest_meeting_threshold() {
        let mut comparison = ModelComparison::new("test");

        let mut fast = EvalResult::new("fast", "test", 5000);
        fast.overall_success_rate = 0.9;
        fast.avg_turns_to_success = 1.2;
        comparison.add_result(fast);

        let mut slow = EvalResult::new("slow", "test", 2000);
        slow.overall_success_rate = 0.9;
        slow.avg_turns_to_success = 2.8;
        comparison.add_result(slow);

        let fastest = comparison.fastest_meeting_threshold(0.85);
        assert!(fastest.is_some());
        assert_eq!(fastest.unwrap().model_id, "fast");
    }

    #[test]
    fn test_fastest_meeting_threshold_none() {
        let mut comparison = ModelComparison::new("test");

        let mut result = EvalResult::new("model", "test", 1000);
        result.overall_success_rate = 0.5;
        comparison.add_result(result);

        let fastest = comparison.fastest_meeting_threshold(0.9);
        assert!(fastest.is_none());
    }

    #[test]
    fn test_smallest_meeting_threshold_none() {
        let mut comparison = ModelComparison::new("test");

        let mut result = EvalResult::new("model", "test", 1000);
        result.overall_success_rate = 0.5;
        comparison.add_result(result);

        let smallest = comparison.smallest_meeting_threshold(0.9);
        assert!(smallest.is_none());
    }

    #[test]
    fn test_compute_pareto_frontier() {
        let mut comparison = ModelComparison::new("test");

        let mut model1 = EvalResult::new("m1", "test", 1000);
        model1.overall_success_rate = 0.9;
        model1.avg_turns_to_success = 1.5;
        comparison.add_result(model1);

        let mut model2 = EvalResult::new("m2", "test", 5000);
        model2.overall_success_rate = 0.95;
        model2.avg_turns_to_success = 1.2;
        comparison.add_result(model2);

        comparison.compute_pareto_frontier();
        // Both should be on frontier (different trade-offs)
        assert!(!comparison.pareto_frontier.is_empty());
    }

    #[test]
    fn test_generate_recommendations_with_pareto() {
        let mut comparison = ModelComparison::new("test");

        let mut model1 = EvalResult::new("m1", "test", 1000);
        model1.overall_success_rate = 0.9;
        model1.avg_turns_to_success = 1.5;
        comparison.add_result(model1);

        let mut model2 = EvalResult::new("m2", "test", 5000);
        model2.overall_success_rate = 0.95;
        model2.avg_turns_to_success = 1.2;
        comparison.add_result(model2);

        // Compute pareto first so the "best balance" recommendation uses it
        comparison.compute_pareto_frontier();
        comparison.generate_recommendations();

        assert!(comparison.recommendations.len() >= 2);
        assert!(comparison
            .recommendations
            .iter()
            .any(|r| r.scenario.contains("balance")));
    }

    #[test]
    fn test_pareto_point_struct() {
        let point = ParetoPoint {
            model_id: "test".to_string(),
            size_bytes: 1000,
            success_rate: 0.9,
            avg_turns: 1.5,
            is_pareto_optimal: true,
        };
        assert_eq!(point.model_id, "test");
        assert!(point.is_pareto_optimal);
    }

    #[test]
    fn test_recommendation_struct() {
        let rec = Recommendation {
            scenario: "test scenario".to_string(),
            model_id: "model1".to_string(),
            rationale: "because".to_string(),
        };
        assert_eq!(rec.scenario, "test scenario");
        assert_eq!(rec.model_id, "model1");
    }

    #[test]
    fn test_eval_suite_config_custom() {
        let config = EvalSuiteConfig {
            id: "custom".to_string(),
            description: "Custom suite".to_string(),
            max_turns: 10,
            turn_timeout_secs: 120,
            examples_path: PathBuf::from("/custom/path"),
            success_thresholds: vec![0.7, 0.8],
        };
        assert_eq!(config.id, "custom");
        assert_eq!(config.max_turns, 10);
    }

    #[test]
    fn test_example_default_values() {
        let example = Example::new("id", "input", "expected");
        assert_eq!(example.difficulty, Difficulty::Medium);
        assert!(example.tags.is_empty());
    }

    #[test]
    fn test_model_comparison_empty() {
        let comparison = ModelComparison::new("test");
        assert!(comparison.results.is_empty());
        assert!(comparison.pareto_frontier.is_empty());
        assert!(comparison.recommendations.is_empty());
    }

    #[test]
    fn test_eval_result_with_params() {
        let mut result = EvalResult::new("model", "task", 1000);
        result.model_params = Some(1_000_000);
        assert_eq!(result.model_params, Some(1_000_000));
    }

    #[test]
    fn test_finalize_with_only_failed() {
        let mut result = EvalResult::new("model", "task", 1000);
        result.add_example(ExampleResult::failed(
            "ex1",
            Difficulty::Hard,
            3,
            "error",
            vec![100; 3],
            vec![Duration::ZERO; 3],
        ));
        result.finalize(3);

        assert!((result.overall_success_rate - 0.0).abs() < 0.001);
        assert!((result.avg_turns_to_success - 0.0).abs() < 0.001);
    }

    /// Test implementation of EvalTask trait
    struct TestTask {
        id: String,
        description: String,
        examples: Vec<Example>,
    }

    impl EvalTask for TestTask {
        fn id(&self) -> &str {
            &self.id
        }

        fn description(&self) -> &str {
            &self.description
        }

        fn examples(&self) -> &[Example] {
            &self.examples
        }
    }

    #[test]
    fn test_eval_task_defaults() {
        let task = TestTask {
            id: "test".to_string(),
            description: "Test task".to_string(),
            examples: vec![],
        };

        assert_eq!(task.max_turns(), 5);
        assert_eq!(task.turn_timeout(), Duration::from_secs(60));
        assert_eq!(task.id(), "test");
        assert_eq!(task.description(), "Test task");
    }

    #[test]
    fn test_difficulty_clone_copy() {
        let d1 = Difficulty::Expert;
        let d2 = d1; // Copy
        let d3 = d1.clone();
        assert_eq!(d1, d2);
        assert_eq!(d1, d3);
    }

    #[test]
    fn test_difficulty_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Difficulty::Easy);
        set.insert(Difficulty::Hard);
        assert!(set.contains(&Difficulty::Easy));
        assert!(!set.contains(&Difficulty::Expert));
    }
}
