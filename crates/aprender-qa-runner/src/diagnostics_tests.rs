use super::*;

use crate::evidence::{HostInfo, Outcome, PerformanceMetrics};
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use chrono::Utc;
use std::collections::HashMap;

fn test_evidence() -> Evidence {
    Evidence {
        id: "test-evidence-001".to_string(),
        gate_id: "G3-STABLE".to_string(),
        scenario: QaScenario::new(
            ModelId::new("Qwen", "Qwen2.5-Coder-0.5B-Instruct"),
            Modality::Run,
            Backend::Cpu,
            Format::Apr,
            "What is 2+2?".to_string(),
            0,
        ),
        outcome: Outcome::Crashed,
        reason: "Process crashed with exit code -1".to_string(),
        output: String::new(),
        stderr: Some("SIGSEGV at 0x12345".to_string()),
        exit_code: Some(-1),
        metrics: PerformanceMetrics {
            duration_ms: 52740,
            ..Default::default()
        },
        timestamp: Utc::now(),
        host: HostInfo::default(),
        metadata: HashMap::new(),
    }
}


include!("diagnostics_tests_part_a.rs");
include!("diagnostics_tests_part_b.rs");
