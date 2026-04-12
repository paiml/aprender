use super::*;

use tempfile::NamedTempFile;

const TEST_CSV: &str = r#"model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4,tps_gguf_cpu,tps_gguf_gpu,tps_apr_cpu,tps_apr_gpu,tps_st_cpu,tps_st_gpu,provenance_verified
Qwen/Qwen2.5-Coder-0.5B-Instruct,qwen-coder,0.5B,tiny,BLOCKED,246,F,quick,2026-02-04T13:28:18.663298968+00:00,true,true,true,true,,,,,,,false
Qwen/Qwen2.5-Coder-1.5B-Instruct,qwen-coder,1.5B,small,BLOCKED,415,-,none,2026-02-03T15:50:04.803811188+00:00,true,true,true,true,17.9,129.8,16.2,0.6,2.9,23.8,false
meta-llama/Llama-3.2-1B-Instruct,llama,1B,small,PENDING,0,-,none,2026-01-31T00:00:00+00:00,false,false,false,false,,,,,,,false
"#;

// FALSIFY-CERT-001: Round-trip integrity
//
// Falsification hypothesis: "CSV round-trip corrupts data"
// If read(write(rows)) != rows, implementation is broken.

include!("certification_data_tests_part_a.rs");
include!("certification_data_tests_part_b.rs");
include!("certification_data_tests_part_c.rs");
