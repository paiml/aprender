// TRN-07 / D-14 as amended by the phase-3 review — the caller cannot ASSERT the number.
//
// This is the compile-time half of the hole the review found. The pre-review shape was
// `ValidationMetric::new(&Split<Validation>, name, value)`: possessing a validation split does
// not prove a number came out of it, and does not prove the number was computed with the
// artifact whose canonical-test access it goes on to unlock. `evaluate_validation(run,
// dataset, metric)` now COMPUTES the value from the verified run and the typed split, and
// `ValidationEvaluation` has private fields with no public constructor — so a
// `SelectionCandidate`, and therefore a `SelectionLock`, can only ever commit a number that
// trusted code produced.
//
// # Why the `::new(kind, value)` half is NOT in this file — measured, not assumed
//
// The first draft attempted the struct literal AND a `ValidationEvaluation::new(kind, 0.99)`
// call, expecting E0451 plus E0599. Only E0599 appeared: rustc's privacy pass runs AFTER
// type checking, so the missing-associated-item typeck error aborted the compile and the
// blessed snapshot contained only the weaker half — "today there is no function called
// `new`" — while the structural claim, that the FIELDS are private so no literal works
// either, was silently missing from its own evidence. The literal is the load-bearing half
// and it is the one that is here.
//
// The absent-float-door half is covered where it can be checked exhaustively rather than one
// name at a time: 03-09's `evaluate_*` source guard scans every public function's PARAMETER
// LIST in `evaluate.rs` and requires ZERO of them to take an `f64` value, naming the single
// `#[cfg(test)]` exception explicitly. A compile-fail case can only ever refute the one
// signature it spells out.
//
// Expected diagnostic: `fields ... of struct 'ValidationEvaluation' are private` (E0451).

use entrenar::train::setfit::evaluate::{ValidationEvaluation, ValidationMetricKind};

fn assert_my_own_accuracy() -> ValidationEvaluation {
    ValidationEvaluation {
        metric_kind: ValidationMetricKind::Accuracy,
        value: 0.99,
        artifact_hash: String::new(),
        validation_split_fingerprint: String::new(),
        dataset_fingerprint: String::new(),
        n_rows: 1,
    }
}

fn main() {}
