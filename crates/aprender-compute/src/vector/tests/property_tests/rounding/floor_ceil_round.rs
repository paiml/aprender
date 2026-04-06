use super::super::super::super::*;
use proptest::prelude::*;

// Property test: floor correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_floor_correctness(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.floor().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.floor();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "floor failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: floor idempotence - floor(floor(x)) = floor(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_floor_idempotence(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let floor_once = va.floor().unwrap();
        let floor_twice = floor_once.floor().unwrap();

        for (i, (&once, &twice)) in floor_once.as_slice().iter()
            .zip(floor_twice.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (once - twice).abs() < 1e-5,
                "floor idempotence failed at {}: floor(floor({})) = {} != {}",
                i, a[i], twice, once
            );
        }
    }
}

// Property test: floor always <= original value
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_floor_less_than_or_equal(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.floor().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                output <= input,
                "floor should be <= input at {}: floor({}) = {} > {}",
                i, input, output, input
            );
        }
    }
}

// Property test: ceil correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_ceil_correctness(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.ceil().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.ceil();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "ceil failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: ceil idempotence - ceil(ceil(x)) = ceil(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_ceil_idempotence(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let ceil_once = va.ceil().unwrap();
        let ceil_twice = ceil_once.ceil().unwrap();

        for (i, (&once, &twice)) in ceil_once.as_slice().iter()
            .zip(ceil_twice.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (once - twice).abs() < 1e-5,
                "ceil idempotence failed at {}: ceil(ceil({})) = {} != {}",
                i, a[i], twice, once
            );
        }
    }
}

// Property test: ceil always >= original value
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_ceil_greater_than_or_equal(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.ceil().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                output >= input,
                "ceil should be >= input at {}: ceil({}) = {} < {}",
                i, input, output, input
            );
        }
    }
}

// Property test: round correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_round_correctness(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.round().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.round();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "round failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: round idempotence - round(round(x)) = round(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_round_idempotence(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let round_once = va.round().unwrap();
        let round_twice = round_once.round().unwrap();

        for (i, (&once, &twice)) in round_once.as_slice().iter()
            .zip(round_twice.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (once - twice).abs() < 1e-5,
                "round idempotence failed at {}: round(round({})) = {} != {}",
                i, a[i], twice, once
            );
        }
    }
}

// Property test: round distance - |round(x) - x| <= 0.5
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_round_distance(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.round().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let distance = (output - input).abs();
            prop_assert!(
                distance <= 0.5 + 1e-5,  // Small epsilon for floating point precision
                "round distance should be <= 0.5 at {}: |round({}) - {}| = {} > 0.5",
                i, input, input, distance
            );
        }
    }
}
