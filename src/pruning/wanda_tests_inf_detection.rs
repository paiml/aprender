use super::*;

// ==========================================================================
// FALSIFICATION: Inf detection for norms
// ==========================================================================
#[test]
fn test_wanda_detects_inf_norms() {
    let wanda = WandaImportance::new("layer0");

    let weights = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let norms = Tensor::new(&[1.0, f32::INFINITY], &[2]);

    let result = wanda.compute_from_tensors(&weights, &norms);

    assert!(result.is_err(), "WND-18 FALSIFIED: Should detect Inf norms");
    match result.unwrap_err() {
        PruningError::NumericalInstability { method, details } => {
            assert_eq!(method, "wanda");
            assert!(details.contains("Inf"));
        }
        _ => panic!("WND-18 FALSIFIED: Expected NumericalInstability error"),
    }
}

// ==========================================================================
// FALSIFICATION: Empty norm tensor
// ==========================================================================
#[test]
fn test_wanda_empty_norms() {
    let wanda = WandaImportance::new("layer0");

    let weights = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let norms = Tensor::new(&[], &[0]);

    let result = wanda.compute_from_tensors(&weights, &norms);

    assert!(
        result.is_err(),
        "WND-19 FALSIFIED: Should error on empty norms"
    );
}

// ==========================================================================
// FALSIFICATION: Negative activation norms
// ==========================================================================
#[test]
fn test_wanda_negative_norms() {
    let wanda = WandaImportance::new("layer0");

    // Negative norms should be handled gracefully
    let weights = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let norms = Tensor::new(&[-1.0, 2.0], &[2]);

    let result = wanda.compute_from_tensors(&weights, &norms);

    assert!(
        result.is_ok(),
        "WND-20 FALSIFIED: Should handle negative norms"
    );

    let importance = result.unwrap();
    for &v in importance.data() {
        assert!(
            v.is_finite(),
            "WND-20 FALSIFIED: Negative norms should not produce non-finite"
        );
        assert!(
            v >= 0.0,
            "WND-20 FALSIFIED: Importance should be non-negative"
        );
    }
}
