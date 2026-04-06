use super::super::super::super::*;
use proptest::prelude::*;

// Property test: trunc correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_trunc_correctness(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.trunc().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.trunc();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "trunc failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: trunc idempotence - trunc(trunc(x)) = trunc(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_trunc_idempotence(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let trunc_once = va.trunc().unwrap();
        let trunc_twice = trunc_once.trunc().unwrap();

        for (i, (&once, &twice)) in trunc_once.as_slice().iter()
            .zip(trunc_twice.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (once - twice).abs() < 1e-5,
                "trunc idempotence failed at {}: trunc(trunc({})) = {} != {}",
                i, a[i], twice, once
            );
        }
    }
}

// Property test: trunc moves toward zero - |trunc(x)| <= |x|
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_trunc_toward_zero(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.trunc().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                output.abs() <= input.abs() + 1e-5,  // Small epsilon for floating point
                "trunc should move toward zero at {}: |trunc({})| = {} > |{}| = {}",
                i, input, output.abs(), input, input.abs()
            );
        }
    }
}

// Property test: fract correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_fract_correctness(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.fract().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.fract();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "fract failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: fract decomposition - x = trunc(x) + fract(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_fract_decomposition(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let trunc_result = va.trunc().unwrap();
        let fract_result = va.fract().unwrap();

        for (i, (&input, (&t, &f))) in a.iter()
            .zip(trunc_result.as_slice().iter().zip(fract_result.as_slice().iter()))
            .enumerate() {
            let reconstructed = t + f;
            prop_assert!(
                (reconstructed - input).abs() < 1e-5,
                "decomposition failed at {}: {} != trunc({}) + fract({}) = {} + {} = {}",
                i, input, input, input, t, f, reconstructed
            );
        }
    }
}

// Property test: fract magnitude - |fract(x)| < 1
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_fract_magnitude(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.fract().unwrap();

        for (i, &output) in result.as_slice().iter().enumerate() {
            prop_assert!(
                output.abs() < 1.0,
                "fract magnitude should be < 1 at {}: |fract({})| = {} >= 1",
                i, a[i], output.abs()
            );
        }
    }
}
