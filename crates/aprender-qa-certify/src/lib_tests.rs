use super::*;

const SAMPLE_CSV: &str = r"model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4,tps_gguf_cpu,tps_gguf_gpu,tps_apr_cpu,tps_apr_gpu,tps_st_cpu,tps_st_gpu,provenance_verified,kernel_proof_ref
Qwen/Qwen2.5-Coder-0.5B-Instruct,qwen-coder,0.5B,tiny,PENDING,0,-,none,2026-01-31T00:00:00Z,false,false,false,false,,,,,,false,false,
Qwen/Qwen2.5-Coder-1.5B-Instruct,qwen-coder,1.5B,small,CERTIFIED,920,A,deep,2026-01-31T12:00:00Z,true,true,true,true,25.5,85.2,22.3,78.1,18.1,62.5,true,
meta-llama/Llama-3.2-1B-Instruct,llama,1B,small,BLOCKED,450,F,smoke,2026-01-31T00:00:00Z,true,false,false,false,12.0,45.0,,,,,false,";

include!("lib_tests_part_a.rs");

include!("lib_tests_part_b.rs");
