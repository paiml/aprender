
#[cfg(test)]
mod proptests {
    use super::*;
    use crate::automl::params::RandomForestParam as RF;
    use crate::automl::SearchSpace;
    use proptest::prelude::*;

    proptest! {
        /// TPE should always respect budget constraint.
        #[test]
        fn prop_tpe_respects_budget(
            n_trials in 1_usize..50,
            seed in any::<u64>(),
            request in 1_usize..100
        ) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add(RF::NEstimators, 10..500);

            let mut tpe = TPE::new(n_trials).with_seed(seed);
            let trials = tpe.suggest(&space, request);

            prop_assert!(trials.len() <= n_trials);
            prop_assert!(trials.len() <= request);
        }

        /// Same seed should produce same initial trials.
        #[test]
        fn prop_tpe_deterministic(seed in any::<u64>()) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add(RF::NEstimators, 10..500);

            let mut tpe1 = TPE::new(10).with_seed(seed);
            let mut tpe2 = TPE::new(10).with_seed(seed);

            let t1 = tpe1.suggest(&space, 5);
            let t2 = tpe2.suggest(&space, 5);

            for (a, b) in t1.iter().zip(t2.iter()) {
                prop_assert_eq!(a.get(&RF::NEstimators), b.get(&RF::NEstimators));
            }
        }

        /// Gamma should always be in valid range.
        #[test]
        fn prop_gamma_clamped(gamma in -1.0_f32..2.0) {
            let tpe = TPE::new(10).with_gamma(gamma);
            prop_assert!(tpe.config.gamma >= 0.01);
            prop_assert!(tpe.config.gamma <= 0.5);
        }
    }
}
