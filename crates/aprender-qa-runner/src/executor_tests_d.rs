
use super::*;
use crate::command::MockCommandRunner;
use apr_qa_gen::{Backend, Format, Modality, ModelId};


include!("executor_tests_d_part_a.rs");

include!("executor_jidoka_tests.rs");

include!("executor_tests_d_g0_integrity.rs");

include!("executor_tests_d_part_d.rs");

include!("executor_tests_lifecycle_g0.rs");

include!("executor_tests_d_part_e.rs");
