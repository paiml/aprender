//! ONT-10 L4 witness: the Rust merge shape pipeline agrees with its Lean simulation
//! `ProvableContracts/Theorems/LoRA/Shape_Preservation.lean` (`mergeWithBase`, `deltaOne`,
//! `validateModels`). The tables are that module's `#eval mergeRows / deltaRows / validateRows`
//! output, copied verbatim: exhaustive over one parameter `"w"` with lengths 0..=3 and absent.
//! If a row here and the Lean `#eval` disagree, one of the two changed.

use super::{compute_deltas, merge_with_base, validate_models, MergeError, Model};
use crate::autograd::Tensor;
use ndarray::Array1;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn model(len: Option<usize>) -> Model {
    let mut m = HashMap::new();
    if let Some(l) = len {
        m.insert("w".to_string(), Tensor::new(Array1::zeros(l), false));
    }
    m
}

/// `#eval mergeRows`: (base len, delta len or absent, merged len or `None` = panic).
const MERGE_ROWS: [(usize, Option<usize>, Option<usize>); 20] = [
    (0, None, Some(0)),
    (0, Some(0), Some(0)),
    (0, Some(1), Some(0)),
    (0, Some(2), None),
    (0, Some(3), None),
    (1, None, Some(1)),
    (1, Some(0), Some(0)),
    (1, Some(1), Some(1)),
    (1, Some(2), Some(2)),
    (1, Some(3), Some(3)),
    (2, None, Some(2)),
    (2, Some(0), None),
    (2, Some(1), Some(2)),
    (2, Some(2), Some(2)),
    (2, Some(3), None),
    (3, None, Some(3)),
    (3, Some(0), None),
    (3, Some(1), Some(3)),
    (3, Some(2), None),
    (3, Some(3), Some(3)),
];

/// `#eval deltaRows`: (base len or absent, model len, outcome, delta len).
/// Outcome 0 = Ok, 1 = IncompatibleArchitectures, 2 = ShapeMismatch.
const DELTA_ROWS: [(Option<usize>, usize, u8, usize); 20] = [
    (None, 0, 1, 0),
    (None, 1, 1, 0),
    (None, 2, 1, 0),
    (None, 3, 1, 0),
    (Some(0), 0, 0, 0),
    (Some(0), 1, 2, 0),
    (Some(0), 2, 2, 0),
    (Some(0), 3, 2, 0),
    (Some(1), 0, 2, 0),
    (Some(1), 1, 0, 1),
    (Some(1), 2, 2, 0),
    (Some(1), 3, 2, 0),
    (Some(2), 0, 2, 0),
    (Some(2), 1, 2, 0),
    (Some(2), 2, 0, 2),
    (Some(2), 3, 2, 0),
    (Some(3), 0, 2, 0),
    (Some(3), 1, 2, 0),
    (Some(3), 2, 2, 0),
    (Some(3), 3, 0, 3),
];

/// `#eval validateRows`: (second model's len or absent, outcome), reference len 2.
/// Outcome 0 = Ok, 1 = IncompatibleArchitectures, 2 = ShapeMismatch.
const VALIDATE_ROWS: [(Option<usize>, u8); 5] =
    [(None, 1), (Some(0), 2), (Some(1), 2), (Some(2), 0), (Some(3), 2)];

fn code(e: &MergeError) -> u8 {
    match e {
        MergeError::IncompatibleArchitectures(_) => 1,
        MergeError::ShapeMismatch(_) => 2,
        MergeError::InsufficientModels { .. } => 3,
        MergeError::InvalidConfig(_) => 4,
    }
}

#[test]
fn merge_with_base_matches_lean_merge_rows() {
    for (bl, dl, want) in MERGE_ROWS {
        let (base, delta) = (model(Some(bl)), model(dl));
        let got = catch_unwind(AssertUnwindSafe(|| merge_with_base(&base, delta)))
            .ok()
            .map(|m| {
                assert_eq!(m.len(), 1, "merge keeps exactly the base's names");
                m["w"].len()
            });
        assert_eq!(got, want, "base {bl}, delta {dl:?}");
    }
}

#[test]
fn compute_deltas_matches_lean_delta_rows() {
    for (bl, l, want, want_len) in DELTA_ROWS {
        let got = match compute_deltas(&[model(Some(l))], &model(bl)) {
            Ok(ds) => {
                assert_eq!(ds.len(), 1);
                (0, ds[0]["w"].len())
            }
            Err(e) => (code(&e), 0),
        };
        assert_eq!(got, (want, want_len), "base {bl:?}, model {l}");
    }
}

#[test]
fn validate_models_matches_lean_validate_rows() {
    for (l, want) in VALIDATE_ROWS {
        let got = validate_models(&[model(Some(2)), model(l)]).map_or_else(|e| code(&e), |()| 0);
        assert_eq!(got, want, "second model {l:?}");
    }
    // `#eval validateModels []` = insufficient (3).
    assert_eq!(validate_models(&[]).map_or_else(|e| code(&e), |()| 0), 3);
}
