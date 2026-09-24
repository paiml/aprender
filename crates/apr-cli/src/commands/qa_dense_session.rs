// #4270 V2b: the qa gates drive a dense GGUF through the one engine
// (`realizar::gguf::dense_session`, #4268) instead of calling
// `generate_with_cache` / `generate_gpu_resident` themselves.

/// A dense GGUF session on the CPU.
#[cfg(feature = "inference")]
fn qa_dense_cpu(
    model: realizar::gguf::OwnedQuantizedModel,
) -> realizar::gguf::dense_session::DenseSession {
    use realizar::gguf::dense_session::{DenseForward, DenseSession};
    DenseSession::new(DenseForward::cpu(std::sync::Arc::new(model)))
}

/// A dense GGUF session on a CUDA model the caller has built.
#[cfg(feature = "cuda")]
fn qa_dense_cuda(
    model: realizar::gguf::OwnedQuantizedModelCuda,
) -> realizar::gguf::dense_session::DenseSession {
    use realizar::gguf::dense_session::{DenseForward, DenseSession};
    DenseSession::new(DenseForward::cuda(model))
}

/// One turn: prompt plus generated tokens, the shape `generate_with_cache`
/// returned. With `require_gpu`, a turn the engine moved to the CPU is an
/// error, so a GPU gate can never pass on a CPU fallback's output.
#[cfg(feature = "inference")]
fn qa_dense_generate(
    session: &mut realizar::gguf::dense_session::DenseSession,
    prompt: &[u32],
    config: &realizar::gguf::QuantizedGenerateConfig,
    require_gpu: bool,
) -> std::result::Result<Vec<u32>, String> {
    let (tokens, used_gpu) = realizar::gguf::dense_session::dense_turn(session, prompt, config)
        .map_err(|e| e.to_string())?;
    if require_gpu && !used_gpu {
        return Err("the engine served this turn on the CPU (GPU fallback)".to_string());
    }
    Ok(tokens)
}

#[cfg(all(test, feature = "inference"))]
mod qa_dense_session_tests {
    use super::*;
    use realizar::gguf::MappedGGUFModel;
    use realizar::session::{entries_for, EntryKind};

    const MODEL: &str = "/home/noah/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";

    fn mapped() -> Option<MappedGGUFModel> {
        std::path::Path::new(MODEL)
            .exists()
            .then(|| MappedGGUFModel::from_path(MODEL).expect("map"))
    }

    #[test]
    fn golden_cpu_gate_generates_through_the_dense_session() {
        let Some(mapped) = mapped() else { return };
        let prompt = "#4270 V2b: the golden gate names three primes:";
        let (tokens, _) =
            golden_output_gguf_cpu(&mapped, &mapped.model, prompt, 4).expect("golden cpu");
        let prompt_tokens = golden_prompt_tokens(&mapped.model, prompt);
        assert!(tokens.starts_with(&prompt_tokens) && tokens.len() > prompt_tokens.len());
        let entries = entries_for(&prompt_tokens);
        assert!(
            entries
                .iter()
                .any(|e| e.arch == "qwen2" && e.kind == EntryKind::Generate && !e.on_gpu),
            "no dense CPU Generate entry for the golden prompt: {entries:?}"
        );
    }

    #[test]
    fn a_gpu_gate_refuses_a_turn_served_on_the_cpu() {
        let Some(mapped) = mapped() else { return };
        let model = realizar::gguf::OwnedQuantizedModel::from_mapped(&mapped).expect("model");
        let prompt = golden_prompt_tokens(&mapped.model, "#4270 V2b: require_gpu");
        let config = realizar::gguf::QuantizedGenerateConfig {
            max_tokens: 2,
            temperature: 0.0,
            top_k: 1,
            ..Default::default()
        };
        let mut session = qa_dense_cpu(model);
        assert!(qa_dense_generate(&mut session, &prompt, &config, false).is_ok());
        let err = qa_dense_generate(&mut session, &prompt, &config, true)
            .expect_err("a CPU turn must not pass a GPU gate");
        assert!(err.contains("GPU fallback"), "{err}");
    }
}
