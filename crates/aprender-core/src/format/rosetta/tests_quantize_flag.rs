// ============================================================================
// Falsifiers for `apr rosetta convert --quantize` (issue #2382, finding 4).
//
// In 0.63.0 the flag was a no-op for every value and accepted garbage silently:
// int8, int4, fp16, bogus and BOGUS_XYZ all produced byte-identical output to a
// convert with no flag at all, each exiting 0 with no diagnostic.
// ============================================================================

use super::*;
use crate::format::converter::QuantizationType;

#[test]
fn falsifier_unknown_quantization_is_rejected() {
    // `--quantize BOGUS_XYZ` used to be indistinguishable from no flag at all.
    for bogus in ["bogus", "BOGUS_XYZ", "q3_k", "", "int2"] {
        let err = parse_quantization(Some(bogus))
            .err()
            .unwrap_or_else(|| panic!("--quantize {bogus:?} must be rejected, not ignored"));
        let msg = err.to_string();
        assert!(
            msg.contains("Unknown quantization"),
            "the error must name the bad value, got: {msg}"
        );
        assert!(
            msg.contains("int8"),
            "the error must list what IS supported, got: {msg}"
        );
    }
}

#[test]
fn falsifier_documented_quantization_values_all_map() {
    // Every value in the `--quantize` help text must map to a real type.
    for value in SUPPORTED_QUANTIZATIONS {
        let parsed = parse_quantization(Some(value))
            .unwrap_or_else(|e| panic!("documented value {value:?} must parse: {e}"));
        assert!(
            parsed.is_some(),
            "documented value {value:?} must map to a quantization type, not None"
        );
    }
}

#[test]
fn falsifier_quantization_values_map_to_distinct_types() {
    assert_eq!(
        parse_quantization(Some("int8")).expect("int8"),
        Some(QuantizationType::Int8)
    );
    assert_eq!(
        parse_quantization(Some("int4")).expect("int4"),
        Some(QuantizationType::Q4K)
    );
    assert_eq!(
        parse_quantization(Some("fp16")).expect("fp16"),
        Some(QuantizationType::Fp16)
    );
    // Case-insensitive, as the CLI lowercases nothing itself.
    assert_eq!(
        parse_quantization(Some("INT8")).expect("INT8"),
        Some(QuantizationType::Int8)
    );
}

#[test]
fn falsifier_no_quantize_flag_stays_none() {
    assert_eq!(parse_quantization(None).expect("no flag"), None);
}

#[test]
fn falsifier_intermediate_hop_drops_quantization() {
    // A multi-step conversion (SafeTensors → APR → GGUF) must quantize once, at
    // the final export. Quantizing the temp APR too would quantize twice.
    let opts = ConversionOptions {
        quantization: Some("int8".to_string()),
        verify: false,
        tolerance: 0.25,
        ..Default::default()
    };
    let hop = opts.without_quantization();
    assert_eq!(hop.quantization, None, "the intermediate hop must not quantize");
    assert!(!hop.verify, "other options must survive unchanged");
    assert_eq!(hop.tolerance, 0.25, "other options must survive unchanged");
}
