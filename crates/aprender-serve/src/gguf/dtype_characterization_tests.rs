//! CHARACTERIZATION — what these two boundaries accept TODAY, before PMAT-3430
//! Phases 3-4 move them onto `trueno_quant::GgmlType`.
//!
//! These tests are written against UNCHANGED code and must pass UNCHANGED after
//! the re-export and the admission functions land. That is the whole mechanism
//! behind #3430 Q1-c's claim that "no crate accepts an id or a name it refused
//! before": not a promise in a PR body, a snapshot that fails if it stops being
//! true.
//!
//! They live in a `#[cfg(test)]` child module of the boundary's own module
//! because both functions are PRIVATE, and a child module sees its ancestors'
//! private items. No visibility is widened to test them.
//!
//! The id sweep is 0..=255 and not "the interesting ids", because the property
//! is about what is REFUSED as much as what is accepted, and a sweep that only
//! visits the accepted set cannot see a widening.

use super::{apr_dtype_to_byte, apr_qtype_to_dtype};

/// The 16 ids serve admitted before M1, with the string each produced.
const ADMITTED_TODAY: [(u32, &str); 16] = [
    (0, "F32"),
    (1, "F16"),
    (2, "Q4_0"),
    (3, "Q4_1"),
    (6, "Q5_0"),
    (7, "Q5_1"),
    (8, "Q8_0"),
    (9, "Q8_1"),
    (10, "Q2_K"),
    (11, "Q3_K"),
    (12, "Q4_K"),
    (13, "Q5_K"),
    (14, "Q6_K"),
    (16, "IQ2_XXS"),
    (17, "IQ2_XS"),
    (30, "BF16"),
];

fn admitted(id: u32) -> Option<&'static str> {
    ADMITTED_TODAY
        .iter()
        .find(|(candidate, _)| *candidate == id)
        .map(|(_, name)| *name)
}

#[test]
fn apr_qtype_to_dtype_accepts_exactly_sixteen_ids_over_the_whole_byte_range() {
    for id in 0u32..=255 {
        let got = apr_qtype_to_dtype(id).ok();
        assert_eq!(
            got,
            admitted(id),
            "qtype {id}: admission changed (this boundary writes APR dtype strings; \
             accepting an id it used to refuse means emitting quantized bytes a reader \
             will decode as something else)"
        );
    }
}

#[test]
fn apr_qtype_to_dtype_refuses_with_a_message_that_names_the_id() {
    // The refusal is the point of this boundary (see its doc comment), so the
    // shape of the error is characterized too, not just that it is an error.
    for id in [4u32, 5, 15, 18, 29, 31, 42, 43, 200, 255] {
        let err = apr_qtype_to_dtype(id).expect_err("must refuse an unadmitted id");
        let text = format!("{err}");
        assert!(
            text.contains(&id.to_string()),
            "the refusal for qtype {id} does not name it: {text}"
        );
    }
}

#[test]
fn apr_dtype_to_byte_maps_every_admitted_name_and_falls_back_to_zero() {
    for (id, name) in ADMITTED_TODAY {
        assert_eq!(
            u32::from(apr_dtype_to_byte(name)),
            id,
            "dtype {name} no longer maps to byte {id}"
        );
        assert_eq!(
            u32::from(apr_dtype_to_byte(&name.to_ascii_lowercase())),
            id,
            "dtype {name} no longer maps from its lower-case spelling"
        );
    }
    // Unknown names warn and write F32 (0). That is a lossy fallback this
    // change must not silently alter.
    for name in ["", "nonsense", "Q9_9", "q4_K", "Q4_k", "IQ3_S", "TQ1_0", "MXFP4"] {
        assert_eq!(
            apr_dtype_to_byte(name),
            0,
            "dtype {name:?} is newly recognised by apr_dtype_to_byte"
        );
    }
}
