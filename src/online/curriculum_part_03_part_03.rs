
    #[test]
    fn test_curriculum_trainer_reset() {
        let scheduler = LinearCurriculum::new(5);
        let mut trainer = CurriculumTrainer::new(scheduler);

        let features = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![1.0, 2.0];
        let scorer = FeatureNormScorer::new();

        trainer
            .add_samples(&features, &targets, 2, &scorer)
            .unwrap();

        trainer.advance();
        assert!(trainer.stage() > 0.0);

        trainer.reset();
        assert_eq!(trainer.stage(), 0.0);
    }

    #[test]
    fn test_curriculum_trainer_scheduler_access() {
        let scheduler = LinearCurriculum::new(5);
        let trainer = CurriculumTrainer::new(scheduler);

        let sched = trainer.scheduler();
        assert_eq!(sched.stage(), 0.0);
    }

    #[test]
    fn test_curriculum_trainer_is_complete() {
        let scheduler = LinearCurriculum::new(2);
        let mut trainer = CurriculumTrainer::new(scheduler);

        assert!(!trainer.is_complete());

        // Advance past completion
        for _ in 0..5 {
            trainer.advance();
        }
        assert!(trainer.is_complete());
    }

    #[test]
    fn test_scored_sample_debug_clone() {
        let sample = ScoredSample::new(vec![1.0, 2.0], 3.0, 0.5);
        let debug_str = format!("{:?}", sample);
        assert!(debug_str.contains("ScoredSample"));

        let cloned = sample.clone();
        assert_eq!(cloned.features, vec![1.0, 2.0]);
        assert_eq!(cloned.target, 3.0);
        assert_eq!(cloned.difficulty, 0.5);
    }

    #[test]
    fn test_linear_curriculum_debug_clone() {
        let curriculum = LinearCurriculum::new(5);
        let debug_str = format!("{:?}", curriculum);
        assert!(debug_str.contains("LinearCurriculum"));

        let cloned = curriculum.clone();
        assert_eq!(cloned.stage(), 0.0);
    }

    #[test]
    fn test_exponential_curriculum_debug_clone() {
        let curriculum = ExponentialCurriculum::new(0.3);
        let debug_str = format!("{:?}", curriculum);
        assert!(debug_str.contains("ExponentialCurriculum"));

        let cloned = curriculum.clone();
        assert_eq!(cloned.stage(), 0.0);
    }

    #[test]
    fn test_self_paced_curriculum_debug_clone() {
        let curriculum = SelfPacedCurriculum::new(0.5, 1.5);
        let debug_str = format!("{:?}", curriculum);
        assert!(debug_str.contains("SelfPacedCurriculum"));

        let cloned = curriculum.clone();
        assert_eq!(cloned.n_total(), 0);
    }

    #[test]
    fn test_loss_difficulty_scorer_debug_clone() {
        let scorer = LossDifficultyScorer::with_mean(3.0);
        let debug_str = format!("{:?}", scorer);
        assert!(debug_str.contains("LossDifficultyScorer"));

        let cloned = scorer.clone();
        // Cloned should produce same result
        let d1 = scorer.score(&[0.0], 5.0);
        let d2 = cloned.score(&[0.0], 5.0);
        assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn test_self_paced_advance_resets_batch_idx() {
        let mut curriculum = SelfPacedCurriculum::new(1.0, 1.5);

        let samples = vec![
            ScoredSample::new(vec![1.0], 1.0, 0.1),
            ScoredSample::new(vec![2.0], 2.0, 0.2),
        ];
        curriculum.add_samples(samples);

        // Consume some batch
        let _ = curriculum.next_batch(1);

        // Advance resets batch_idx
        curriculum.advance();

        // Should start from beginning again
        let batch = curriculum.next_batch(10);
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn test_linear_curriculum_full_cycle() {
        let mut curriculum = LinearCurriculum::new(4).with_difficulty_range(0.0, 1.0);

        // Stage 0: threshold = 0.0
        assert!((curriculum.current_threshold() - 0.0).abs() < 0.01);

        // Advance through all stages
        curriculum.advance(); // 0.25
        assert!((curriculum.current_threshold() - 0.25).abs() < 0.01);

        curriculum.advance(); // 0.5
        assert!((curriculum.current_threshold() - 0.5).abs() < 0.01);

        curriculum.advance(); // 0.75
        assert!((curriculum.current_threshold() - 0.75).abs() < 0.01);

        curriculum.advance(); // 1.0
        assert!((curriculum.current_threshold() - 1.0).abs() < 0.01);
        assert!(curriculum.is_complete());
    }

    #[test]
    fn test_curriculum_trainer_eligible_empty_samples() {
        let scheduler = LinearCurriculum::new(5);
        let trainer = CurriculumTrainer::<LinearCurriculum>::new(scheduler);

        // No samples added - eligible should be empty
        let eligible = trainer.eligible_samples();
        assert!(eligible.is_empty());
        assert_eq!(trainer.n_eligible(), 0);
    }
