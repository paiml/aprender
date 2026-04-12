
use super::*;
use crate::command::MockCommandRunner;
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};

/// Helper: create a temp model directory with a safetensors file
fn make_temp_model_dir() -> tempfile::TempDir {
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let st_dir = dir.path().join("safetensors");
    std::fs::create_dir_all(&st_dir).expect("mkdir safetensors");
    std::fs::write(st_dir.join("model.safetensors"), b"fake").expect("write");
    dir
}


include!("executor_tests_e_part_a.rs");

include!("executor_tests_e_part_b.rs");

include!("executor_tests_e_part_c.rs");
