
use super::*;
use crate::command::MockCommandRunner;
use apr_qa_gen::{Backend, Format, Modality, ModelId};

fn test_playbook() -> Playbook {
    let yaml = r#"
name: test-playbook
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 5
"#;
    Playbook::from_yaml(yaml).expect("Failed to parse")
}

include!("executor_tests_a_part_a.rs");

include!("executor_tests_a_part_b.rs");
