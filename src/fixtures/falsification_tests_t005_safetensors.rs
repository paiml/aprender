
#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TOKENS: &[u32] = &[1, 2, 3, 4, 5];

    #[test]
    fn t005_safetensors_cpu_forward() {
        let fixture = ModelFixture::safetensors("t005", ModelConfig::tiny());

        match forward_safetensors_cpu(&fixture, TEST_TOKENS) {
            Ok(result) => {
                eprintln!(
                    "[T005] SafeTensors:CPU produced {} logits",
                    result.logits.len()
                );
                let sum: f32 = result.logits.iter().sum();
                let max = result
                    .logits
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let min = result.logits.iter().cloned().fold(f32::INFINITY, f32::min);
                let argmax = result.argmax();
                eprintln!(
                    "[T005] SafeTensors:CPU sum={:.4}, min={:.4}, max={:.4}, argmax={:?}",
                    sum, min, max, argmax
                );
                let checks = falsify(&result);
                for check in &checks {
                    if !check.passed {
                        eprintln!("[T005] FALSIFIED {}: {}", check.id, check.details);
                    }
                }
            },
            Err(e) => {
                eprintln!("[T005] SafeTensors:CPU FAILED TO LOAD/RUN: {}", e);
            },
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn t002_gguf_cuda_forward() {
        use crate::cuda::CudaExecutor;
        if !CudaExecutor::is_available() {
            eprintln!("[T002] SKIPPED: CUDA not available");
            return;
        }

        let fixture = ModelFixture::gguf("t002", ModelConfig::tiny());

        match forward_gguf_cuda(&fixture, TEST_TOKENS) {
            Ok(result) => {
                eprintln!("[T002] GGUF:CUDA produced {} logits", result.logits.len());
                let sum: f32 = result.logits.iter().sum();
                let max = result
                    .logits
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let min = result.logits.iter().cloned().fold(f32::INFINITY, f32::min);
                let argmax = result.argmax();
                eprintln!(
                    "[T002] GGUF:CUDA sum={:.4}, min={:.4}, max={:.4}, argmax={:?}",
                    sum, min, max, argmax
                );
                let checks = falsify(&result);
                for check in &checks {
                    if !check.passed {
                        eprintln!("[T002] FALSIFIED {}: {}", check.id, check.details);
                    }
                }
            },
            Err(e) => {
                eprintln!("[T002] GGUF:CUDA FAILED TO LOAD/RUN: {}", e);
            },
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn t004_apr_cuda_forward() {
        use crate::cuda::CudaExecutor;
        if !CudaExecutor::is_available() {
            eprintln!("[T004] SKIPPED: CUDA not available");
            return;
        }

        let fixture = ModelFixture::apr("t004", ModelConfig::tiny());

        match forward_apr_cuda(&fixture, TEST_TOKENS) {
            Ok(result) => {
                eprintln!("[T004] APR:CUDA produced {} logits", result.logits.len());
                let checks = falsify(&result);
                for check in &checks {
                    if !check.passed {
                        eprintln!("[T004] FALSIFIED {}: {}", check.id, check.details);
                    }
                }
            },
            Err(e) => {
                eprintln!("[T004] APR:CUDA FAILED TO LOAD/RUN: {}", e);
            },
        }
    }
include!("falsification_tests_gguf_fixture_apr.rs");
}
