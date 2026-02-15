
#[cfg(test)]
mod tests {
    use super::*;
include!("types_part_02_part_02.rs");
include!("types_part_02_part_03.rs");

    // Property-based tests
    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_percentile_monotonic(values in prop::collection::vec(0.0..100.0f64, 10..100)) {
                let p25 = percentile(&values, 0.25);
                let p50 = percentile(&values, 0.50);
                let p75 = percentile(&values, 0.75);

                prop_assert!(p25 <= p50);
                prop_assert!(p50 <= p75);
            }

            #[test]
            fn prop_percentile_bounded(values in prop::collection::vec(0.0..100.0f64, 1..100)) {
                let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                for p in [0.0, 0.25, 0.5, 0.75, 1.0] {
                    let pct = percentile(&values, p);
                    prop_assert!(pct >= min - 0.001);
                    prop_assert!(pct <= max + 0.001);
                }
            }

            #[test]
            fn prop_statistics_std_non_negative(values in prop::collection::vec(-100.0..100.0f64, 2..100)) {
                let stats = Statistics::from_values(&values);
                prop_assert!(stats.std >= 0.0);
            }

            #[test]
            fn prop_statistics_min_leq_max(values in prop::collection::vec(-100.0..100.0f64, 1..100)) {
                let stats = Statistics::from_values(&values);
                prop_assert!(stats.min <= stats.max);
            }
        }
    }
}
