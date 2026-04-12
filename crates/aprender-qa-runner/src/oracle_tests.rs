use super::*;

use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};

fn make_test_scenario() -> QaScenario {
    QaScenario {
        id: "test_scenario".to_string(),
        model: ModelId {
            org: "test".to_string(),
            name: "model".to_string(),
            variant: None,
        },
        modality: Modality::Run,
        backend: Backend::Cpu,
        format: Format::Apr,
        prompt: "test".to_string(),
        temperature: 0.0,
        max_tokens: 32,
        seed: 0,
        trace_level: apr_qa_gen::TraceLevel::None,
        oracle_type: "garbage".to_string(),
    }
}


include!("oracle_tests_part_a.rs");
include!("oracle_tests_part_b.rs");
