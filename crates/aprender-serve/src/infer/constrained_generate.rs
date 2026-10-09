// #3793 (#3568 PRs 2 and 3): constrained generation for `apr run --json-schema` / `--grammar`.
//
// A constrained run takes the one engine (#4263), which applies the constraint:
// `Session::generate_constrained` on the dense forward or on the Qwen3.5 hybrid's, on the CPU
// or (PR 3) on the CUDA device, wherever the unconstrained run would take that engine. A run
// whose unconstrained dispatch would take any other loop REFUSES by name, before a token is
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

/// What removes a `--trace` refusal: nothing scheduled; the run without it takes the engine.
const TRACE_REMOVED_BY: &str = "not scheduled: `--trace` runs the instrumented loop, which \
     applies no constraint; without `--trace` the engine runs and applies it";

/// Whether the dense dispatch enters CUDA (`run_gguf_generate`): a cuda build, no `--no-gpu`,
/// and no legacy quant. A build-and-request fact, never a device probe, and the constrained
/// run reports it as `gpu_attempted` exactly as the unconstrained one does (#3826).
fn dense_cuda_attempted(config: &InferenceConfig, has_legacy_quant: bool) -> bool {
    session_cuda_attempted(config) && !has_legacy_quant
}

/// Whether a session dispatch (the MoE loop, the Qwen3.5 hybrid) tries CUDA: a cuda build and
/// no `--no-gpu` (#3826). The constrained and the unconstrained hybrid report it alike.
fn session_cuda_attempted(config: &InferenceConfig) -> bool {
    cfg!(feature = "cuda") && !config.no_gpu
}

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
/// constraint: `(path, removed_by)`. `None` is the engine, on the CPU or the CUDA device,
/// which applies it.
///
/// The predicates are the dispatch's own, in its order, and depend on the build and the
/// request only, never on a device probe: a refusal is the same on every host.
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
    if is_qwen35 {
        // The hybrid's engine, on its CUDA forward on a cuda build unless `--no-gpu`, else on
        // the CPU (#3568 PR 3). It has no wgpu forward and no instrumented loop
        return None;
    }
    let has_legacy_quant = model_has_legacy_quant(model);
    if dense_cuda_attempted(config, has_legacy_quant) {
        // The engine on the device (#3568 PR 3), unless `--trace` keeps the instrumented
        // device loop (`generate_gpu_resident`). A device that refuses the model hands the
        // turn to the CPU engine, never to the wgpu loop the unconstrained run tries next
        return gen_config
            .trace
            .then_some(("gguf-cuda-trace", TRACE_REMOVED_BY));
    }
    if wgpu_attempted(config, gen_config, has_legacy_quant) {
        return Some((
            "gguf-wgpu",
            "not scheduled: the wgpu loop is outside #3568's four PRs; `--no-gpu` runs the \
             CPU, which applies the constraint",
        ));
    }
    // #4268: `--trace` keeps the instrumented dense loop, which is not the engine
    gen_config.trace.then_some(("gguf-trace", TRACE_REMOVED_BY))
}

/// Generate under `request` through the one engine (#3793): `(tokens, used_gpu,
/// gpu_attempted, stop)`, the GPU pair as the unconstrained run reports it (#3826). The schema
/// or grammar is compiled against the model's vocabulary here, after load and before the first
/// token, so a schema the engine cannot enforce is refused (`SchemaUnsupported`) without
/// generating.
fn generate_gguf_constrained(
    request: &crate::constrain::ConstraintRequest,
    config: &InferenceConfig,
    mapped: &crate::gguf::MappedGGUFModel,
    owned: Option<crate::gguf::OwnedQuantizedModel>,
    qwen35_host: Option<&'static crate::gguf::forward_qwen35::Qwen35Model<'static>>,
    input_tokens: &[u32],
    gen_config: &crate::gguf::QuantizedGenerateConfig,
) -> Result<(Vec<u32>, bool, bool, crate::session::ConstrainedStop)> {
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
    let (turn, stop, gpu_attempted) = if let Some(qwen) = qwen35_host {
        crate::gguf::forward_qwen35::qwen35_check_context(input_tokens.len(), context_length)?;
        let positions = (input_tokens.len() + gen_config.max_tokens).min(context_length);
        // The CUDA forward on a cuda build unless `--no-gpu`, as the unconstrained run loads it;
        // a forward failure moves the session to the CPU, printed, and `used_gpu` says so
        let mut session = crate::gguf::qwen35_session::Qwen35Session::load_for_run(
            qwen,
            mapped,
            config.no_gpu,
            positions,
        )?;
        mark_generation_start(); // #3981
        let (turn, stop) =
            session.generate_constrained(input_tokens, gen_config, constraint.as_mut())?;
        (turn, stop, session_cuda_attempted(config))
    } else if let Some(model) = owned {
        // The dense turn's own admission (`dense_stream`)
        if input_tokens.len() > context_length {
            return Err(RealizarError::ContextLimitExceeded {
                provided: input_tokens.len(),
                maximum: context_length,
            });
        }
        // #3826: attempted is the dispatch's own predicate, fixed before the turn as the
        // unconstrained run fixes it, so a device that refuses the model still reads as tried
        let gpu_attempted = dense_cuda_attempted(config, model_has_legacy_quant(&model));
        let (turn, stop) = dense_constrained_turn(
            model,
            config,
            gpu_attempted,
            input_tokens,
            gen_config,
            constraint.as_mut(),
        )?;
        (turn, stop, gpu_attempted)
    } else {
        return Err(no_model());
    };
    Ok((turn.tokens, turn.used_gpu, gpu_attempted, stop))
}

/// The dense turn under `constraint`. On the device when `on_device` (the dense dispatch enters
/// CUDA, #3568 PR 3), through the same upload and F2 check as the unconstrained run; else, and
/// when the device refuses the model or F2 measures a mismatch, on the CPU. Both are the one
/// engine, which applies the constraint, and the turn's `used_gpu` says which ran. The wgpu
/// loop the unconstrained run would try after a refused device applies none, so it is never
/// entered.
fn dense_constrained_turn(
    model: crate::gguf::OwnedQuantizedModel,
    config: &InferenceConfig,
    on_device: bool,
    input_tokens: &[u32],
    gen_config: &crate::gguf::QuantizedGenerateConfig,
    constraint: &mut dyn crate::constrain::TokenConstraint,
) -> Result<(crate::session::Turn, crate::session::ConstrainedStop)> {
    use crate::gguf::dense_session::{DenseForward, DenseSession};
    #[cfg(feature = "cuda")]
    let model = if on_device {
        match cuda_dense_model(model, input_tokens, gen_config, config.verbose) {
            Ok(cuda_model) => {
                mark_generation_start(); // #3981: setup (upload + F2) ends here
                let mut session = DenseSession::new(DenseForward::cuda(cuda_model));
                return session.generate_constrained(input_tokens, gen_config, constraint);
            },
            Err(model) => *model,
        }
    } else {
        model
    };
    #[cfg(not(feature = "cuda"))]
    let _ = on_device; // always false here: `dense_cuda_attempted` needs the feature
    log_cpu_backend(config.verbose, model_has_legacy_quant(&model));
    let mut session = DenseSession::new(DenseForward::cpu(std::sync::Arc::new(model)));
    mark_generation_start(); // #3981
    session.generate_constrained(input_tokens, gen_config, constraint)
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
