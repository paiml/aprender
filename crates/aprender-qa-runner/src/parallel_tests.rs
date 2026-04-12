use super::*;

use apr_qa_gen::{Backend, Format, Modality, ModelId};

/// Create a default test scenario for parallel execution tests
fn test_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    )
}

/// Create a vector of test scenarios with sequential prompts
fn test_scenarios(count: usize) -> Vec<QaScenario> {
    (0..count)
        .map(|i| {
            QaScenario::new(
                ModelId::new("test", "model"),
                Modality::Run,
                Backend::Cpu,
                Format::Gguf,
                format!("What is {}+{}?", i, i + 1),
                i as u64,
            )
        })
        .collect()
}

include!("parallel_tests_part_a.rs");
include!("parallel_tests_part_b.rs");
