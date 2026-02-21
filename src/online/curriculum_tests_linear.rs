
    #[test]
    fn test_linear_curriculum_basic() {
        let mut curriculum = LinearCurriculum::new(10);

        assert_eq!(curriculum.stage(), 0.0);
        assert!(!curriculum.is_complete());

        curriculum.advance();
        assert!((curriculum.stage() - 0.1).abs() < 0.01);

        // Advance to completion
        for _ in 0..15 {
            curriculum.advance();
        }

        assert!(curriculum.is_complete());
        assert!((curriculum.stage() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_curriculum_threshold() {
        let curriculum = LinearCurriculum::new(10).with_difficulty_range(0.2, 0.8);

        // At stage 0, threshold should be 0.2
        assert!((curriculum.current_threshold() - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_linear_curriculum_reset() {
        let mut curriculum = LinearCurriculum::new(10);

        curriculum.advance();
        curriculum.advance();
        curriculum.reset();

        assert_eq!(curriculum.stage(), 0.0);
    }

    #[test]
    fn test_exponential_curriculum() {
        let mut curriculum = ExponentialCurriculum::new(0.3);

        assert_eq!(curriculum.stage(), 0.0);

        curriculum.advance();
        let stage1 = curriculum.stage();

        curriculum.advance();
        let stage2 = curriculum.stage();

        // Should grow exponentially
        assert!(stage1 > 0.0);
        assert!(stage2 > stage1);
        assert!(stage2 < 1.0);
    }

    #[test]
    fn test_exponential_curriculum_completion() {
        let mut curriculum = ExponentialCurriculum::new(0.5);

        for _ in 0..50 {
            curriculum.advance();
        }

        assert!(curriculum.is_complete());
    }

    #[test]
    fn test_self_paced_curriculum_basic() {
        let mut curriculum = SelfPacedCurriculum::new(0.25, 1.5);

        let samples = vec![
            ScoredSample::new(vec![1.0], 1.0, 0.1),
            ScoredSample::new(vec![2.0], 2.0, 0.2),
            ScoredSample::new(vec![3.0], 3.0, 0.5),
            ScoredSample::new(vec![4.0], 4.0, 0.8),
        ];

        curriculum.add_samples(samples);

        // With threshold 0.25, only first two samples (0.1, 0.2) should be eligible
        let initial = curriculum.n_eligible();
        assert!(
            initial >= 1 && initial <= 2,
            "Expected 1-2 eligible, got {}",
            initial
        );

        // Advance threshold (0.25 * 1.5 = 0.375)
        curriculum.advance();
        // Now samples with difficulty <= 0.375 should be eligible
        let after_advance = curriculum.n_eligible();
        assert!(
            after_advance >= initial,
            "Should have more eligible after advance"
        );
    }

    #[test]
    fn test_self_paced_curriculum_next_batch() {
        let mut curriculum = SelfPacedCurriculum::new(1.0, 1.5);

        let samples = vec![
            ScoredSample::new(vec![1.0], 1.0, 0.1),
            ScoredSample::new(vec![2.0], 2.0, 0.2),
            ScoredSample::new(vec![3.0], 3.0, 0.3),
        ];

        curriculum.add_samples(samples);

        let batch1 = curriculum.next_batch(2);
        assert_eq!(batch1.len(), 2);

        let batch2 = curriculum.next_batch(2);
        assert_eq!(batch2.len(), 1); // Only one left
    }

    #[test]
    fn test_self_paced_curriculum_update_difficulties() {
        let mut curriculum = SelfPacedCurriculum::new(0.5, 1.5);

        let samples = vec![
            ScoredSample::new(vec![1.0], 1.0, 0.1),
            ScoredSample::new(vec![2.0], 2.0, 0.2),
        ];

        curriculum.add_samples(samples);

        // Update with new difficulty function
        curriculum.update_difficulties(|_features, target| target * 0.1);

        // Check difficulties were updated
        let eligible = curriculum.eligible_samples();
        assert!(eligible[0].difficulty <= eligible.get(1).map_or(f64::MAX, |s| s.difficulty));
    }

    #[test]
    fn test_loss_difficulty_scorer() {
        let mut scorer = LossDifficultyScorer::new();
        scorer.fit(&[1.0, 2.0, 3.0, 4.0, 5.0]); // mean = 3.0

        // Sample at mean should have low difficulty
        let diff_at_mean = scorer.score(&[0.0], 3.0);
        // Sample far from mean should have high difficulty
        let diff_far = scorer.score(&[0.0], 10.0);

        assert!(diff_at_mean < diff_far);
    }

    #[test]
    fn test_feature_norm_scorer() {
        let scorer = FeatureNormScorer::new();

        // Small feature vector
        let small = scorer.score(&[1.0, 0.0], 0.0);
        // Large feature vector
        let large = scorer.score(&[3.0, 4.0], 0.0);

        assert!(small < large);
        assert!((large - 5.0).abs() < 0.01); // sqrt(9 + 16) = 5
    }

    #[test]
    fn test_curriculum_trainer_basic() {
        let scheduler = LinearCurriculum::new(5);
        let mut trainer = CurriculumTrainer::new(scheduler);

        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let targets = vec![1.0, 2.0, 3.0];
        let scorer = FeatureNormScorer::new();

        trainer
            .add_samples(&features, &targets, 2, &scorer)
            .unwrap();

        assert_eq!(trainer.stage(), 0.0);
        assert!(!trainer.is_complete());
    }

    #[test]
    fn test_curriculum_trainer_eligible_samples() {
        let scheduler = LinearCurriculum::new(2);
        let mut trainer = CurriculumTrainer::new(scheduler);

        // Features with varying norms
        let features = vec![
            1.0, 0.0, // norm = 1
            3.0, 4.0, // norm = 5
            0.5, 0.0, // norm = 0.5
        ];
        let targets = vec![1.0, 2.0, 3.0];
        let scorer = FeatureNormScorer::new();

        trainer
            .add_samples(&features, &targets, 2, &scorer)
            .unwrap();

        // At stage 0, only easiest samples
        let eligible = trainer.n_eligible();
        assert!(eligible > 0);
        assert!(eligible <= 3);

        // Advance and check more are eligible
        trainer.advance();
        let eligible_after = trainer.n_eligible();
        assert!(eligible_after >= eligible);
    }

    #[test]
    fn test_curriculum_trainer_dimension_mismatch() {
        let scheduler = LinearCurriculum::new(5);
        let mut trainer = CurriculumTrainer::new(scheduler);

        let features = vec![1.0, 2.0, 3.0]; // 3 features, doesn't divide by 2
        let targets = vec![1.0];
        let scorer = FeatureNormScorer::new();

        let result = trainer.add_samples(&features, &targets, 2, &scorer);
        assert!(result.is_err());
    }

    #[test]
    fn test_curriculum_trainer_target_mismatch() {
        let scheduler = LinearCurriculum::new(5);
        let mut trainer = CurriculumTrainer::new(scheduler);

        let features = vec![1.0, 2.0, 3.0, 4.0]; // 2 samples
        let targets = vec![1.0, 2.0, 3.0]; // 3 targets (mismatch!)
        let scorer = FeatureNormScorer::new();

        let result = trainer.add_samples(&features, &targets, 2, &scorer);
        assert!(result.is_err());
    }

    #[test]
    fn test_scored_sample_creation() {
        let sample = ScoredSample::new(vec![1.0, 2.0], 3.0, 0.5);

        assert_eq!(sample.features, vec![1.0, 2.0]);
        assert_eq!(sample.target, 3.0);
        assert_eq!(sample.difficulty, 0.5);
    }

    #[test]
    fn test_linear_curriculum_default() {
        let curriculum = LinearCurriculum::default();
        assert_eq!(curriculum.stage(), 0.0);
    }

    #[test]
    fn test_exponential_curriculum_default() {
        let curriculum = ExponentialCurriculum::default();
        assert_eq!(curriculum.stage(), 0.0);
    }

    #[test]
    fn test_self_paced_max_threshold() {
        let mut curriculum = SelfPacedCurriculum::new(0.1, 2.0).with_max_threshold(0.5);

        // Advance many times
        for _ in 0..10 {
            curriculum.advance();
        }

        assert!(curriculum.threshold() <= 0.5);
    }

    #[test]
    fn test_self_paced_empty() {
        let curriculum = SelfPacedCurriculum::new(0.5, 1.5);

        // Empty curriculum should be "complete"
        assert!(curriculum.is_complete());
        assert_eq!(curriculum.stage(), 1.0);
    }

    #[test]
    fn test_linear_curriculum_zero_stages() {
        let mut curriculum = LinearCurriculum::new(0);
        // step_size should be 1.0 when n_stages is 0
        assert_eq!(curriculum.stage(), 0.0);

        curriculum.advance();
        // After one advance with step_size 1.0, should be complete
        assert!(curriculum.is_complete());
        assert!((curriculum.stage() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_curriculum_reset() {
        let mut curriculum = ExponentialCurriculum::new(0.3);

        // Advance several times
        for _ in 0..5 {
            curriculum.advance();
        }
        assert!(curriculum.stage() > 0.0);

        // Reset should bring back to 0
        curriculum.reset();
        assert_eq!(curriculum.stage(), 0.0);
        assert!(!curriculum.is_complete());
    }

    #[test]
    fn test_exponential_curriculum_threshold() {
        let mut curriculum = ExponentialCurriculum::new(0.3);

        // Threshold equals stage for exponential curriculum
        curriculum.advance();
        assert!((curriculum.current_threshold() - curriculum.stage()).abs() < 1e-10);
    }

    #[test]
    fn test_self_paced_curriculum_reset() {
        let mut curriculum = SelfPacedCurriculum::new(0.1, 2.0);

        let samples = vec![
            ScoredSample::new(vec![1.0], 1.0, 0.05),
            ScoredSample::new(vec![2.0], 2.0, 0.5),
        ];
        curriculum.add_samples(samples);

        // Advance and change state
        curriculum.advance();
        assert!(curriculum.threshold() > 0.1);

        // Reset should restore initial threshold
        curriculum.reset();
        assert!((curriculum.threshold() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_self_paced_curriculum_n_total() {
        let mut curriculum = SelfPacedCurriculum::new(0.5, 1.5);

        assert_eq!(curriculum.n_total(), 0);

        let samples = vec![
            ScoredSample::new(vec![1.0], 1.0, 0.1),
            ScoredSample::new(vec![2.0], 2.0, 0.2),
            ScoredSample::new(vec![3.0], 3.0, 0.3),
        ];
        curriculum.add_samples(samples);

        assert_eq!(curriculum.n_total(), 3);
    }

    #[test]
    fn test_self_paced_next_batch_exhaustion() {
        let mut curriculum = SelfPacedCurriculum::new(1.0, 1.5);

        let samples = vec![
            ScoredSample::new(vec![1.0], 1.0, 0.1),
            ScoredSample::new(vec![2.0], 2.0, 0.2),
        ];
        curriculum.add_samples(samples);

        // Get all samples
        let batch1 = curriculum.next_batch(10);
        assert_eq!(batch1.len(), 2);

        // Next batch should be empty (exhausted) and reset batch_idx
        let batch2 = curriculum.next_batch(10);
        assert!(batch2.is_empty());
    }

    #[test]
    fn test_self_paced_stage_with_samples() {
        let mut curriculum = SelfPacedCurriculum::new(0.15, 1.5);

        let samples = vec![
            ScoredSample::new(vec![1.0], 1.0, 0.1),
            ScoredSample::new(vec![2.0], 2.0, 0.2),
            ScoredSample::new(vec![3.0], 3.0, 0.5),
            ScoredSample::new(vec![4.0], 4.0, 0.8),
        ];
        curriculum.add_samples(samples);

        // With threshold 0.15, only sample with difficulty 0.1 is eligible
        let stage = curriculum.stage();
        // stage = n_eligible / n_total = 1/4 = 0.25
        assert!((stage - 0.25).abs() < 0.01);
        assert!(!curriculum.is_complete());
    }

    #[test]
    fn test_self_paced_eligible_samples() {
        let mut curriculum = SelfPacedCurriculum::new(0.3, 1.5);

        let samples = vec![
            ScoredSample::new(vec![1.0], 1.0, 0.1),
            ScoredSample::new(vec![2.0], 2.0, 0.2),
            ScoredSample::new(vec![3.0], 3.0, 0.5),
        ];
        curriculum.add_samples(samples);

        let eligible = curriculum.eligible_samples();
        // Threshold is 0.3, so samples with difficulty 0.1 and 0.2 are eligible
        assert_eq!(eligible.len(), 2);
    }

    #[test]
    fn test_loss_difficulty_scorer_with_mean() {
        let scorer = LossDifficultyScorer::with_mean(5.0);

        // Distance from mean 5.0
        let d1 = scorer.score(&[0.0], 5.0); // 0.0 distance
        let d2 = scorer.score(&[0.0], 10.0); // 5.0 distance

        assert!((d1 - 0.0).abs() < 1e-10);
        assert!((d2 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_loss_difficulty_scorer_fit_empty() {
        let mut scorer = LossDifficultyScorer::new();
        // fit with empty slice should not change target_mean
        scorer.fit(&[]);
        assert!((scorer.score(&[0.0], 0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_loss_difficulty_scorer_default() {
        let scorer = LossDifficultyScorer::default();
        // Default target_mean is 0.0
        let d = scorer.score(&[1.0, 2.0], 3.0);
        assert!((d - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_feature_norm_scorer_default() {
        let scorer = FeatureNormScorer::default();
        let d = scorer.score(&[3.0, 4.0], 0.0);
        assert!((d - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_feature_norm_scorer_empty_features() {
        let scorer = FeatureNormScorer::new();
        let d = scorer.score(&[], 0.0);
        assert!((d - 0.0).abs() < 1e-10);
    }
