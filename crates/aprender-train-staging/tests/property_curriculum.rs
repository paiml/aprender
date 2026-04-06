use entrenar::train::{
    efficiency_score, AdaptiveCurriculum, CurriculumScheduler as _, LinearCurriculum,
    TieredCurriculum,
};
use proptest::collection::vec;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    // =============================================================================
    // Shared Trait Property Tests
    // =============================================================================

    #[test]
    fn prop_sample_weights_valid_range(
      start in any::<f32>(),
      end in any::<f32>(),
      epochs in any::<usize>().prop_filter("epochs must be > 0", |&e| e > 0),
      steps in 10usize..100,
      sample_difficulty in any::<f32>()
    ) {
        // All weights in [0.0, 1.0]
        let mut curriculum = LinearCurriculum::new(start, end, epochs);

        for _ in 0..steps {
          curriculum.step(epochs, 1.0);
        }

        let weight = curriculum.sample_weight(sample_difficulty);
        prop_assert!(
          (0.0..=1.0).contains(&weight),
          "Weight {} out of range for sample_difficulty={}, curriculum_difficulty={}",
          weight, sample_difficulty, curriculum.difficulty()
        );

    }


    // ======================================
    // Linear Curriculum Tests
    // ======================================

    #[test]
    fn prop_difficulty_clamped_within_bounds_on_random_input(
      start in 0.0f32..=1.0f32,
      end in 0.0f32..=1.0f32,
      epochs in 1usize..1000,
      steps in 0usize..1000) {
        // Difficulty should be within [0.0, 1.0]
        let mut curriculum = LinearCurriculum::new(start, end, epochs);

        for _ in 0..steps {
          let d = curriculum.difficulty();
          prop_assert!((0.0..=1.0).contains(&d), "Difficulty {} out of bounds [0.0, 1.0]", d);
          curriculum.step(epochs, 1.0);
        }
        let d = curriculum.difficulty();
        prop_assert!((0.0..=1.0).contains(&d), "Difficulty {} out of bounds [0.0, 1.0]", d);
    }

    #[test]
    fn prop_difficulty_clamped_on_invalid_input(
      start in -1000.0f32..1000.0f32,
      end in -1000.0f32..1000.0f32,
      epochs in 0usize..1000) {
        // Difficulty should be clamped within [0.0, 1.0] even with invalid inputs
        let mut curriculum = LinearCurriculum::new(start, end, epochs);

        curriculum.step(0, 1.0); // zero epochs
        let d = curriculum.difficulty();
        prop_assert!((0.0..=1.0).contains(&d), "Difficulty {} out of bounds for input start={}, end={}, epochs={}", d, start, end, epochs);

    }

    #[test]
    fn prop_tier_clamped_within_bounds_on_random_input(
      start in any::<f32>(),
      end in any::<f32>(),
      epochs in any::<usize>().prop_filter("epochs must be > 0", |&e| e > 0),
      steps in 0usize..1000) {
        // Tier should be within [1, 4]
        let mut curriculum = LinearCurriculum::new(start, end, epochs);

        for _ in 0..steps {
          let t = curriculum.tier();
          prop_assert!((1..=4).contains(&t), "Tier {} out of bounds [1, 4]", t);
          curriculum.step(epochs, 1.0);
        }
        let t = curriculum.tier();
        prop_assert!((1..=4).contains(&t), "Tier {} out of bounds [1, 4]", t);
    }

    #[test]
    fn prop_curriculum_schedule_bounded(
        start in 0.0f32..=1.0f32,
        end in 0.0f32..=1.0f32,
        epochs in 1usize..1000,
        steps in 0usize..1000
    ) {
        // Schedule should stay within [start, end] (or [end, start] if end < start)
        let mut curriculum = LinearCurriculum::new(start, end, epochs);
        let (min_bound, max_bound) = if start <= end { (start, end) } else { (end, start) };

        for _ in 0..steps {
            let d = curriculum.difficulty();
            prop_assert!(
                d >= min_bound && d <= max_bound,
                "Difficulty {} out of schedule bounds [{}, {}]", d, min_bound, max_bound
            );
            curriculum.step(epochs, 1.0);
        }
        let d = curriculum.difficulty();
        prop_assert!(
            d >= min_bound && d <= max_bound,
            "Difficulty {} out of schedule bounds [{}, {}]", d, min_bound, max_bound
        );
    }

    #[test]
    fn prop_difficulty_score_monotonic(
      start in 0.0f32..=0.5,
      end in 0.5f32..=1.0,
      epochs in 1usize..1000,
      steps in 0usize..1000) {
        // Difficulty should not decrease during training
        let mut curriculum = LinearCurriculum::new(start, end, epochs);

        let mut last_difficulty = curriculum.difficulty();
        for _ in 0..steps {
          let current_difficulty = curriculum.difficulty();
          prop_assert!(current_difficulty >= last_difficulty, "Difficulty {} decreased from last difficulty {}", current_difficulty, last_difficulty);
          last_difficulty = current_difficulty;
          curriculum.step(epochs, 1.0);
        }
        let final_difficulty = curriculum.difficulty();
        prop_assert!(final_difficulty >= last_difficulty, "Difficulty {} decreased from last difficulty {}", final_difficulty, last_difficulty);
    }


    // ======================================
    // Tier Curriculum Tests
    // ======================================

    #[test]
    fn prop_tiered_tier_and_difficulty_within_bounds(
      tier_thresholds in vec(0.0f32..=1.0f32, 1..=3),
      patience in 1usize..10,
      steps in 0usize..100,
      accuracy in 0.0f32..1.0f32
    ) {
        // Difficulty should be within [0.0, 1.0] and tier within [1, 4]
        let mut curriculum = TieredCurriculum::new(tier_thresholds, patience);

        // test tier and difficulty over multiple steps
        for _ in 0..steps {
          curriculum.step(0, accuracy);

          let d = curriculum.difficulty();
          let t = curriculum.tier();

          prop_assert!((1..=4).contains(&t), "Tier {} out of bounds [1, 4]", t);
          prop_assert!((0.0..=1.0).contains(&d), "Difficulty {} out of bounds [0.0, 1.0]", d);


        }
    }


    #[test]
    fn prop_tiered_tier_monotonic(
      tier_thresholds in vec(0.0f32..=1.0f32, 1..=3),
      patience in 1usize..10,
      steps in 0usize..100,
      accuracy in 0.0f32..1.0f32
    ) {
        // Tier should not decrease during training
        let mut curriculum = TieredCurriculum::new(tier_thresholds, patience);

        let mut last_tier = curriculum.tier();

        for _ in 0..steps {
          curriculum.step(0, accuracy);

          let t = curriculum.tier();

          prop_assert!(t >= last_tier, "Tier {} decreased from last tier {}", t, last_tier);
          last_tier = t;

        }
    }


    //=====================================
    // Efficiency Score Tests
    //=====================================

    #[test]
    fn prop_efficiency_score_non_negative(
      accuracy in 0.0f32..=1.0f32,
      corpus_size in 2usize..1_000_000
    ) {
        let score = efficiency_score(accuracy, corpus_size);
        prop_assert!(score >= 0.0, "Efficiency score {} is negative for accuracy={} and corpus_size={}", score, accuracy, corpus_size);
    }

    #[test]
    fn prop_weight_for_class_bounded(
      correct_count in 0usize..1000,
      incorrect_count in 0usize..1000,
    ) {
      // Weight for class should be within [1.0, 3.0]

      let mut curriculum = AdaptiveCurriculum::new();

      // correct attempts decrease weight
      for _ in 0..correct_count {
        curriculum.update_class("E0001", true);
      }

      // incorrect attempts increase weight
      for _ in 0..incorrect_count {
        curriculum.update_class("E0001", false);
      }

      let weight = curriculum.weight_for_class("E0001");
      prop_assert!((1.0..=3.0).contains(&weight), "Weight {} out of bounds [1.0, 3.0]", weight);
    }

}
