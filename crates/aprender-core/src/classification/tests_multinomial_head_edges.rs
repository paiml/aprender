//! Public-API coverage for the multinomial head: stored-coefficient loading,
//! error rendering, regularisation and prediction guards (coverage ticket, Rule 7).
//! Contract: FALSIFY-MH-013 in contracts/multinomial-head-v1.yaml.

use crate::classification::multinomial::{
    HeadFitError, HeadInputError, MultinomialLogisticRegression, Regularization,
};

fn labels(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| (*s).to_string()).collect()
}

/// A 2-class, 2-feature head: class "pos" fires on feature 0, "neg" on feature 1.
fn head() -> MultinomialLogisticRegression {
    MultinomialLogisticRegression::from_stored_coefficients(
        labels(&["neg", "pos"]),
        2,
        vec![0.0, 4.0, 4.0, 0.0],
        vec![0.0, 0.0],
    )
    .expect("valid coefficients")
}

fn input_err(r: Result<MultinomialLogisticRegression, HeadFitError>) -> HeadInputError {
    match r {
        Err(HeadFitError::InvalidInput(e)) => e,
        other => panic!("expected InvalidInput, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn stored_head_exposes_its_shape_and_predicts() {
    let h = head();
    assert_eq!(h.n_classes(), 2);
    assert_eq!(h.n_features(), Some(2));
    assert_eq!(h.labels(), &["neg".to_string(), "pos".to_string()]);
    assert_eq!(h.weights().len(), 4);
    assert_eq!(h.intercepts(), &[0.0, 0.0]);
    assert!(h.report().is_none(), "a loaded head has no fit report");

    let rows = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    assert_eq!(h.predict(&rows).expect("predict"), vec!["pos", "neg"]);
    assert_eq!(h.predict_indices(&rows).expect("idx"), vec![1, 0]);
    let p = h.predict_proba(&rows).expect("proba");
    for row in &p {
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
    assert!(p[0][1] > 0.97 && p[1][0] > 0.97);
    // logits are exactly w.x + b
    let z = h.predict_logits(&rows).expect("logits");
    assert!((z[0][1] - 4.0).abs() < 1e-12 && z[0][0].abs() < 1e-12);
}

#[test]
fn stored_coefficients_reject_every_malformed_shape() {
    let ok = |l: Vec<String>, d, w: Vec<f32>, b: Vec<f32>| {
        MultinomialLogisticRegression::from_stored_coefficients(l, d, w, b)
    };
    assert!(matches!(
        input_err(ok(labels(&["a"]), 1, vec![0.0], vec![0.0])),
        HeadInputError::TooFewClasses { k: 1 }
    ));
    assert!(matches!(
        input_err(ok(labels(&["a", ""]), 1, vec![0.0; 2], vec![0.0; 2])),
        HeadInputError::EmptyLabel { index: 1 }
    ));
    assert!(matches!(
        input_err(ok(labels(&["a", "a"]), 1, vec![0.0; 2], vec![0.0; 2])),
        HeadInputError::DuplicateLabel {
            first: 0,
            second: 1,
            ..
        }
    ));
    assert!(matches!(
        input_err(ok(labels(&["a", "b"]), 0, vec![], vec![0.0; 2])),
        HeadInputError::ZeroFeatureDimension
    ));
    assert!(matches!(
        input_err(ok(labels(&["a", "b"]), 2, vec![0.0; 3], vec![0.0; 2])),
        HeadInputError::CoefficientCountMismatch {
            array: "weights",
            expected: 4,
            found: 3
        }
    ));
    assert!(matches!(
        input_err(ok(labels(&["a", "b"]), 2, vec![0.0; 4], vec![0.0; 3])),
        HeadInputError::CoefficientCountMismatch {
            array: "intercepts",
            expected: 2,
            found: 3
        }
    ));
    assert!(matches!(
        input_err(ok(
            labels(&["a", "b"]),
            2,
            vec![0.0, f32::NAN, 0.0, 0.0],
            vec![0.0; 2]
        )),
        HeadInputError::NonFiniteCoefficient {
            array: "weights",
            index: 1,
            ..
        }
    ));
    assert!(matches!(
        input_err(ok(
            labels(&["a", "b"]),
            2,
            vec![0.0; 4],
            vec![0.0, f32::INFINITY]
        )),
        HeadInputError::NonFiniteCoefficient {
            array: "intercepts",
            index: 1,
            ..
        }
    ));
}

#[test]
fn predict_guards_dimension_unfitted_and_overflow() {
    let h = head();
    let e = h.predict_logits(&[vec![1.0, 2.0], vec![1.0]]).unwrap_err();
    assert_eq!(
        e,
        HeadFitError::InvalidInput(HeadInputError::FeatureDimMismatch {
            row: 1,
            expected: 2,
            found: 1
        })
    );
    // An empty batch is valid and yields no rows.
    assert!(h.predict(&[]).expect("empty batch").is_empty());

    let fresh = MultinomialLogisticRegression::new(3);
    assert_eq!(
        fresh.predict_logits(&[vec![1.0]]).unwrap_err(),
        HeadFitError::NotFitted
    );
    assert_eq!(fresh.n_features(), None);

    // Weights of f32::MAX against f32::MAX features overflow even in f64 accumulation
    // only after many terms; a single product stays finite, so build 2 classes whose
    // logit is the sum of enough max-products to overflow f64.
    let big = MultinomialLogisticRegression::from_stored_coefficients(
        labels(&["a", "b"]),
        1,
        vec![f32::MAX, 0.0],
        vec![0.0, 0.0],
    )
    .expect("finite coefficients");
    // f32::MAX^2 ~ 1.16e77, finite in f64: the logit survives where f32 would overflow.
    let z = big
        .predict_logits(&[vec![f32::MAX]])
        .expect("finite in f64");
    assert!(z[0][0] > 1e76);
}

#[test]
fn regularization_resolves_lambda() {
    assert!((Regularization::Lambda(0.25).resolve_lambda(100) - 0.25).abs() < 1e-15);
    // lambda = 1 / (2 C n)
    let l = Regularization::SklearnEquivalentC { c: 2.0 }.resolve_lambda(10);
    assert!((l - 1.0 / 40.0).abs() < 1e-15);
}

#[test]
fn builders_do_not_disturb_loaded_state() {
    let h = head().with_max_iter(7).with_tol(1e-3).with_history_size(3);
    assert_eq!(h.n_classes(), 2);
    assert_eq!(h.n_features(), Some(2));
    assert_eq!(h.predict(&[vec![1.0, 0.0]]).expect("predict"), vec!["pos"]);
}

#[test]
fn error_messages_name_the_offending_values() {
    let cases: Vec<(HeadInputError, &str)> = vec![
        (HeadInputError::TooFewClasses { k: 1 }, "K >= 2"),
        (
            HeadInputError::LabelCountMismatch { labels: 3, k: 2 },
            "3 entries",
        ),
        (HeadInputError::EmptyLabel { index: 4 }, "ordered_labels[4]"),
        (
            HeadInputError::DuplicateLabel {
                first: 0,
                second: 2,
                label: "x".into(),
            },
            "\"x\"",
        ),
        (HeadInputError::EmptyDataset, "no feature rows"),
        (
            HeadInputError::RowCountMismatch {
                rows: 5,
                class_indices: 4,
            },
            "5 feature rows",
        ),
        (HeadInputError::ZeroFeatureDimension, "dimension is 0"),
        (
            HeadInputError::RaggedRow {
                row: 2,
                expected: 3,
                found: 1,
            },
            "ragged",
        ),
        (
            HeadInputError::NanFeature { row: 1, col: 2 },
            "[1][2] is NaN",
        ),
        (
            HeadInputError::InfiniteFeature {
                row: 0,
                col: 1,
                value: f32::INFINITY,
            },
            "not finite",
        ),
        (
            HeadInputError::LabelIndexOutOfRange {
                row: 3,
                index: 9,
                k: 4,
            },
            "0..4",
        ),
        (HeadInputError::UnrepresentedClass { class: 2 }, "class 2"),
        (HeadInputError::NegativeLambda { lambda: -1.0 }, "negative"),
        (
            HeadInputError::NonFiniteLambda { lambda: f64::NAN },
            "not finite",
        ),
        (HeadInputError::NonPositiveC { c: 0.0 }, "strictly positive"),
        (
            HeadInputError::NonFiniteC { c: f64::INFINITY },
            "not finite",
        ),
        (
            HeadInputError::FeatureDimMismatch {
                row: 1,
                expected: 8,
                found: 7,
            },
            "fitted on 8",
        ),
        (
            HeadInputError::CoefficientCountMismatch {
                array: "weights",
                expected: 6,
                found: 5,
            },
            "stored weights has 5",
        ),
        (
            HeadInputError::NonFiniteCoefficient {
                array: "intercepts",
                index: 2,
                value: f32::NAN,
            },
            "intercepts[2]",
        ),
    ];
    for (e, needle) in cases {
        let s = e.to_string();
        assert!(
            s.contains(needle),
            "{e:?} rendered {s:?}, missing {needle:?}"
        );
    }
}

#[test]
fn fit_error_messages_and_conversion() {
    let wrapped: HeadFitError = HeadInputError::EmptyDataset.into();
    assert_eq!(
        wrapped,
        HeadFitError::InvalidInput(HeadInputError::EmptyDataset)
    );
    assert!(wrapped.to_string().starts_with("invalid input: "));
    let msgs = [
        (
            HeadFitError::NotConverged {
                iterations: 9,
                gradient_norm: 1.0,
                tol: 1e-6,
            },
            "did not converge: 9",
        ),
        (
            HeadFitError::Stalled {
                iterations: 4,
                gradient_norm: 0.5,
            },
            "stalled after 4",
        ),
        (
            HeadFitError::NumericalError { iterations: 2 },
            "non-finite value after 2",
        ),
        (
            HeadFitError::NonFiniteLogit { row: 3, class: 1 },
            "row 3, class 1",
        ),
        (HeadFitError::NotFitted, "not been fitted"),
    ];
    for (e, needle) in msgs {
        assert!(e.to_string().contains(needle), "{e:?}");
    }
    let boxed: Box<dyn std::error::Error> = Box::new(HeadFitError::NotFitted);
    assert!(boxed.to_string().contains("fitted"));
}
