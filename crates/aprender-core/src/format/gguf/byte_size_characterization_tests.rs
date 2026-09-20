//! CHARACTERIZATION — `GgufTensor::byte_size` for all 12 of core's variants,
//! snapshotted before PMAT-3430 Phase 3 replaces its hand-written match with
//! delegation to `trueno_quant`'s upstream-extracted table.
//!
//! core has NO id or name boundary — nothing here turns an integer or a string
//! into a `GgmlType`; the enum is only ever built from literals. So core's
//! entire exposure to M1 is this one exhaustive match, which stops compiling
//! (E0004) the moment the enum gains variants, and must be replaced by
//! delegation rather than patched with a bare `_ =>` (#3430 Q1-c).
//!
//! EXACTLY ONE ROW IS EXPECTED TO CHANGE, and it is named here in advance:
//! `Q4_1` is sized at **18** bytes per 32-element block by the hand-written arm
//! (`Q4_0 | Q4_1 => …`), where upstream ggml says **20** (2 x f16 scale+min,
//! plus 16 nibble bytes) and core's OWN other size table, `format/gguf/shape.rs`,
//! already says 20. Two tables in one crate disagreeing. When delegation lands,
//! the `Q4_1` expectation below moves 54 -> 60 and that one-line diff IS the
//! record of the behaviour change. Every other row must be untouched.

use super::{GgmlType, GgufTensor};

fn tensor(dtype: GgmlType, elements: u64) -> GgufTensor {
    GgufTensor {
        name: "t".to_string(),
        shape: vec![elements],
        dtype,
        data: vec![],
    }
}

/// (dtype, elements, bytes) for every variant core carries, at a block-aligned
/// count and a deliberately non-aligned one.
#[test]
fn byte_size_is_unchanged_for_all_twelve_variants() {
    let cases: [(GgmlType, u64, usize); 24] = [
        (GgmlType::F32, 64, 256),
        (GgmlType::F32, 65, 260),
        (GgmlType::F16, 64, 128),
        (GgmlType::F16, 65, 130),
        // Q4_0: 2 (scale) + 16 (nibbles) = 18 bytes per 32 elements. Correct.
        (GgmlType::Q4_0, 64, 36),
        (GgmlType::Q4_0, 65, 54),
        // Q4_1: THE ROW THAT MOVES. 18 today, 20 upstream — see the module doc.
        (GgmlType::Q4_1, 64, 36),
        (GgmlType::Q4_1, 96, 54),
        (GgmlType::Q8_0, 64, 68),
        (GgmlType::Q8_0, 65, 102),
        (GgmlType::Q4K, 512, 288),
        (GgmlType::Q4K, 513, 432),
        (GgmlType::Q6K, 256, 210),
        (GgmlType::Q6K, 257, 420),
        (GgmlType::I8, 64, 64),
        (GgmlType::I8, 65, 65),
        (GgmlType::I16, 64, 128),
        (GgmlType::I16, 65, 130),
        (GgmlType::I32, 64, 256),
        (GgmlType::I32, 65, 260),
        (GgmlType::I64, 64, 512),
        (GgmlType::I64, 65, 520),
        (GgmlType::F64, 64, 512),
        (GgmlType::F64, 65, 520),
    ];
    for (dtype, elements, expected) in cases {
        assert_eq!(
            tensor(dtype, elements).byte_size(),
            expected,
            "{dtype:?} at {elements} elements"
        );
    }
}

/// The disagreement, asserted directly so it cannot be fixed by accident and
/// go unnoticed. When Phase 3 lands, this test is DELETED in the same commit
/// that moves the row above — its whole purpose is to name a defect that
/// exists today.
#[test]
fn q4_1_currently_disagrees_with_cores_own_other_size_table() {
    // 96 elements = 3 blocks. The hand-written arm: 3 * 18 = 54.
    // shape.rs and upstream ggml:            3 * 20 = 60.
    assert_eq!(
        tensor(GgmlType::Q4_1, 96).byte_size(),
        54,
        "Q4_1 is no longer sized at 18 bytes/block here — if this is the Phase 3 \
         delegation, delete this test and move the row in the case table above"
    );
    assert_ne!(
        tensor(GgmlType::Q4_1, 96).byte_size(),
        60,
        "Q4_1 now agrees with upstream; this defect is fixed and the test is obsolete"
    );
}
