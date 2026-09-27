//! L4 simulation witness for `ProvableContracts/Theorems/GgufExportSymmetry/Roundtrip.lean`
//! (formalization.yaml `models:`): the Lean `exportDtype` is `apr_dtype_to_ggml` arm for arm.
//!
//! The domain is the 12 `TensorDType` values, so the relation is checked on every input, not
//! sampled. `lean_export_dtype` is the Lean table transcribed row for row with NO wildcard arm:
//! a new `TensorDType` variant stops this file compiling until the Lean model gets an arm too.
use super::apr_dtype_to_ggml;
use crate::format::gguf::GgmlType;
use crate::format::v2::TensorDType;

/// `exportDtype` (Roundtrip.lean), one arm per `AprDType` constructor.
fn lean_export_dtype(d: TensorDType) -> Option<GgmlType> {
    match d {
        TensorDType::F32 => Some(GgmlType::F32),
        TensorDType::F16 => Some(GgmlType::F16),
        TensorDType::Q4K => Some(GgmlType::Q4K),
        TensorDType::Q6K => Some(GgmlType::Q6K),
        TensorDType::BF16
        | TensorDType::F64
        | TensorDType::I32
        | TensorDType::I64
        | TensorDType::I8
        | TensorDType::U8
        | TensorDType::AprQ4
        | TensorDType::AprQ8 => None,
    }
}

const ALL: [TensorDType; 12] = [
    TensorDType::F32,
    TensorDType::F16,
    TensorDType::BF16,
    TensorDType::F64,
    TensorDType::I32,
    TensorDType::I64,
    TensorDType::I8,
    TensorDType::U8,
    TensorDType::AprQ4,
    TensorDType::AprQ8,
    TensorDType::Q4K,
    TensorDType::Q6K,
];

#[test]
fn apr_dtype_to_ggml_is_the_lean_export_dtype_on_every_input() {
    for d in ALL {
        assert_eq!(apr_dtype_to_ggml(d), lean_export_dtype(d), "{d:?}");
    }
    // GES-REJECT-SYM-001: AprQ8 is rejected, never relabeled Q8_0.
    assert_eq!(apr_dtype_to_ggml(TensorDType::AprQ8), None);
    assert_eq!(ALL.iter().filter(|d| apr_dtype_to_ggml(**d).is_some()).count(), 4);
}
