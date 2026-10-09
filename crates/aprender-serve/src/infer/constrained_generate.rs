// #3793 (#3568 PR 2): constrained generation for `apr run --json-schema` / `--grammar`.
//
// A constrained run takes the one engine (#4263) on the CPU, which applies the constraint:
// `Session::generate_constrained` on the dense forward or on the Qwen3.5 hybrid's. A run whose
// unconstrained dispatch would take any other loop REFUSES by name, before a token is
// generated: "a constraint that is silently ignored is decoration", and a quiet switch to
// another backend would be the fallback #3711 forbids.

/// Refuse a constrained run on a path whose loop applies no constraint (#3793).
fn refuse_constraint_on(config: &InferenceConfig, path: &str, removed_by: &str) -> Result<()> {
    if config.constraint.is_some() {
        return Err(RealizarError::Constraint(
            crate::constrain::ConstraintError::UnsupportedPath {
                path: path.to_string(),
                removed_by: removed_by.to_string(),
            },
        ));
    }
    Ok(())
}

/// What removes a refusal on a GPU loop: PR 3 wires CUDA; `--no-gpu` runs the CPU now.
const CUDA_REMOVED_BY: &str = "#3568 PR 3 wires the CUDA loops; `--no-gpu` runs the CPU, which \
     applies the constraint";

/// Whether the unconstrained dense dispatch would enter wgpu (`run_gguf_generate`): only when
/// an accelerator was asked for (#3757) and the request is greedy (#3760).
fn wgpu_attempted(
    config: &InferenceConfig,
    gen_config: &crate::gguf::QuantizedGenerateConfig,
    has_legacy_quant: bool,
) -> bool {
    #[cfg(feature = "gpu")]
    {
        wgpu_fallback_allowed(config.no_gpu, config.accel_forced, has_legacy_quant)
            && crate::sampling::is_greedy(gen_config.temperature, gen_config.top_k)
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (config, gen_config, has_legacy_quant);
        false
    }
}

/// The loop an unconstrained run of this GGUF would enter, when that loop applies no
/// constraint: `(path, removed_by)`. `None` is the engine on the CPU, which applies it.
///
/// The predicates are the dispatch's own, in its order, and depend on the build and the
/// request only, never on a device probe: a refusal is the same on every host, and a path
/// is refused exactly when the run would report it as `gpu_attempted` (#3826).
fn unconstrained_gguf_path(
    config: &InferenceConfig,
    gen_config: &crate::gguf::QuantizedGenerateConfig,
    model: &crate::gguf::OwnedQuantizedModel,
    is_qwen35: bool,
) -> Option<(&'static str, &'static str)> {
    if crate::gguf::moe_forward_handles(&model.config.architecture) {
        return Some((
            "qwen3-moe",
            "not scheduled: the MoE loop is outside #3568's four PRs",
        ));
    }
    let cuda_attempted = cfg!(feature = "cuda") && !config.no_gpu;
    if is_qwen35 {
        // The hybrid's engine takes its CUDA forward on a cuda build unless `--no-gpu`, and
        // it has no wgpu forward and no instrumented loop
        return cuda_attempted.then_some(("qwen35-cuda", CUDA_REMOVED_BY));
    }
    let has_legacy_quant = model_has_legacy_quant(model);
    if cuda_attempted && !has_legacy_quant {
        return Some(("gguf-cuda", CUDA_REMOVED_BY));
    }
    if wgpu_attempted(config, gen_config, has_legacy_quant) {
        return Some((
            "gguf-wgpu",
            "not scheduled: the wgpu loop is outside #3568's four PRs; `--no-gpu` runs the \
             CPU, which applies the constraint",
        ));
    }
    // #4268: `--trace` keeps the instrumented dense loop, which is not the engine
    gen_config.trace.then_some((
        "gguf-trace",
        "not scheduled: `--trace` runs the instrumented loop, which applies no constraint; \
         without `--trace` the engine runs and applies it",
    ))
}

/// Generate under `request` through the one engine on the CPU (#3793). The schema or grammar
/// is compiled against the model's vocabulary here, after load and before the first token, so
/// a schema the engine cannot enforce is refused (`SchemaUnsupported`) without generating.
fn generate_gguf_constrained(
    request: &crate::constrain::ConstraintRequest,
    config: &InferenceConfig,
    mapped: &crate::gguf::MappedGGUFModel,
    owned: Option<crate::gguf::OwnedQuantizedModel>,
    qwen35_host: Option<&'static crate::gguf::forward_qwen35::Qwen35Model<'static>>,
    input_tokens: &[u32],
    gen_config: &crate::gguf::QuantizedGenerateConfig,
) -> Result<(Vec<u32>, crate::session::ConstrainedStop)> {
    use crate::constrain::{ConstraintEnv, ConstraintError};
    let no_model = || RealizarError::InvalidShape {
        reason: "no model was loaded".to_string(),
    };
    let model = owned
        .as_ref()
        .or(qwen35_host.map(|q| q.base))
        .ok_or_else(no_model)?;
    if let Some((path, removed_by)) =
        unconstrained_gguf_path(config, gen_config, model, qwen35_host.is_some())
    {
        return Err(RealizarError::Constraint(ConstraintError::UnsupportedPath {
            path: path.to_string(),
            removed_by: removed_by.to_string(),
        }));
    }
    let vocab = mapped.model.constraint_vocab().ok_or_else(|| {
        RealizarError::Constraint(ConstraintError::Vocab(
            "this GGUF has no tokenizer vocabulary or no end-of-sequence id".to_string(),
        ))
    })?;
    let env = ConstraintEnv::new(&vocab).map_err(RealizarError::Constraint)?;
    let mut constraint = request.compile(&env).map_err(RealizarError::Constraint)?;
    let context_length = model.config.context_length;
    let (turn, stop) = if let Some(qwen) = qwen35_host {
        crate::gguf::forward_qwen35::qwen35_check_context(input_tokens.len(), context_length)?;
        let positions = (input_tokens.len() + gen_config.max_tokens).min(context_length);
        // `no_gpu`: a run that would take the CUDA forward was refused above
        let mut session = crate::gguf::qwen35_session::Qwen35Session::load_for_run(
            qwen, mapped, true, positions,
        )?;
        mark_generation_start(); // #3981
        session.generate_constrained(input_tokens, gen_config, constraint.as_mut())?
    } else if let Some(model) = owned {
        // The dense CPU turn's own admission (`dense_stream`)
        if input_tokens.len() > context_length {
            return Err(RealizarError::ContextLimitExceeded {
                provided: input_tokens.len(),
                maximum: context_length,
            });
        }
        log_cpu_backend(config.verbose, model_has_legacy_quant(&model));
        let mut session = crate::gguf::dense_session::DenseSession::new(
            crate::gguf::dense_session::DenseForward::cpu(std::sync::Arc::new(model)),
        );
        mark_generation_start(); // #3981
        session.generate_constrained(input_tokens, gen_config, constraint.as_mut())?
    } else {
        return Err(no_model());
    };
    Ok((turn.tokens, stop))
}

/// Why a GGUF run ended: a constrained run's own stop, else what the decoded tokens say.
fn gguf_finish_reason(
    constrained: Option<crate::session::ConstrainedStop>,
    generated: &[u32],
    stop_tokens: &[u32],
    budget: usize,
) -> run_report::FinishReason {
    use crate::session::ConstrainedStop;
    match constrained {
        Some(ConstrainedStop::Complete) => run_report::FinishReason::ConstraintComplete,
        Some(ConstrainedStop::Length) => run_report::FinishReason::Length,
        None => run_report::FinishReason::from_decode(generated, stop_tokens, budget),
    }
}
