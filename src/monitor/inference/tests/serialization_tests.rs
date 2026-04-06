//! Serialization Property Tests

use super::helpers::arb_decision_trace;
use crate::monitor::inference::{
    DecisionTrace, LinearPath, PathType, TraceFormat, TraceSerializer,
};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn prop_binary_serialization_roundtrip(trace in arb_decision_trace()) {
        let serializer = TraceSerializer::new(TraceFormat::Binary);

        let bytes = serializer.serialize(&trace, PathType::Linear)
            .expect("Serialization failed");

        let restored: DecisionTrace<LinearPath> = serializer.deserialize(&bytes)
            .expect("Deserialization failed");

        prop_assert_eq!(trace.sequence, restored.sequence);
        prop_assert_eq!(trace.input_hash, restored.input_hash);
    }

    #[test]
    fn prop_json_serialization_roundtrip(trace in arb_decision_trace()) {
        let serializer = TraceSerializer::new(TraceFormat::Json);

        let bytes = serializer.serialize(&trace, PathType::Linear)
            .expect("Serialization failed");

        let restored: DecisionTrace<LinearPath> = serializer.deserialize(&bytes)
            .expect("Deserialization failed");

        prop_assert_eq!(trace.sequence, restored.sequence);
        prop_assert_eq!(trace.input_hash, restored.input_hash);
    }
}
