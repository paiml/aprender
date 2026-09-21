// #3793 (#3568 PR 2): constrained generation for `apr run --json-schema` / `--grammar`.
//
// A constrained run takes the CPU loop that applies the constraint: the dense GGUF loop
// (`generate_with_cache_constrained`) or the Qwen3.5 hybrid's (`run_qwen35_generate_constrained`).
// A run that would take any other loop REFUSES by name, before a token is generated: "a
// constraint that is silently ignored is decoration", and a quiet switch to another backend
// would be the fallback #3711 forbids.

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

/// What removes a refusal on a GPU loop: PR 3 wires CUDA; `--no-gpu` runs the CPU loop now.
const CUDA_REMOVED_BY: &str = "#3568 PR 3 wires the CUDA loops; `--no-gpu` runs the CPU loop, \
     which applies the constraint";

/// A CUDA device would serve this run: the build has the `cuda` feature and a device answers.
fn cuda_would_serve() -> bool {
    #[cfg(feature = "cuda")]
    {
        crate::cuda::CudaExecutor::is_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// A wgpu adapter would serve this run: the build has the `gpu` feature and an adapter answers.
fn wgpu_would_serve() -> bool {
    #[cfg(feature = "gpu")]
    {
        trueno::backends::gpu::GpuDevice::is_available()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// The loop an unconstrained run of this GGUF would take, when that loop applies no
/// constraint: `(path, removed_by)`. `None` is the CPU loop, which applies it. The order is
/// the dispatch's: MoE first, then `--no-gpu`, then CUDA, then wgpu (dense only).
fn unconstrained_gguf_path(
    config: &InferenceConfig,
    model: &crate::gguf::OwnedQuantizedModel,
    is_qwen35: bool,
    canonical_arch: &str,
) -> Option<(&'static str, &'static str)> {
    if canonical_arch == "qwen3_moe" {
        return Some((
            "qwen3-moe",
            "not scheduled: the MoE loop is outside #3568's four PRs",
        ));
    }
    if config.no_gpu {
        return None;
    }
    if is_qwen35 {
        // The hybrid has a CUDA forward and no wgpu one
        return cuda_would_serve().then_some(("qwen35-cuda", CUDA_REMOVED_BY));
    }
    if model_has_legacy_quant(model) {
        return None;
    }
    if cuda_would_serve() {
        return Some(("gguf-cuda", CUDA_REMOVED_BY));
    }
    wgpu_would_serve().then_some((
        "gguf-wgpu",
        "not scheduled: the wgpu loop is outside #3568's four PRs; `--no-gpu` runs the CPU \
         loop, which applies the constraint",
    ))
}

/// Generate under `request` on the CPU loop that applies it (#3793). The schema or grammar is
/// compiled against the model's vocabulary here, after load and before the first token, so a
/// schema the engine cannot enforce is refused (`SchemaUnsupported`) without generating.
#[allow(clippy::too_many_arguments)]
fn generate_gguf_constrained(
    request: &crate::constrain::ConstraintRequest,
    config: &InferenceConfig,
    mapped: &crate::gguf::MappedGGUFModel,
    model: &crate::gguf::OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &crate::gguf::QuantizedGenerateConfig,
    is_qwen35: bool,
    canonical_arch: &str,
) -> Result<(Vec<u32>, crate::gguf::ConstrainedStop)> {
    use crate::constrain::{ConstraintEnv, ConstraintError};
    if let Some((path, removed_by)) =
        unconstrained_gguf_path(config, model, is_qwen35, canonical_arch)
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
    if is_qwen35 {
        crate::gguf::forward_qwen35::run_qwen35_generate_constrained(
            mapped,
            model,
            input_tokens,
            gen_config,
            constraint.as_mut(),
        )
    } else {
        model.generate_with_cache_constrained(input_tokens, gen_config, constraint.as_mut())
    }
}

/// Why a GGUF run ended: a constrained run's own stop, else what the decoded tokens say.
fn gguf_finish_reason(
    constrained: Option<crate::gguf::ConstrainedStop>,
    generated: &[u32],
    stop_tokens: &[u32],
    budget: usize,
) -> run_report::FinishReason {
    match constrained {
        Some(crate::gguf::ConstrainedStop::Complete) => {
            run_report::FinishReason::ConstraintComplete
        },
        Some(crate::gguf::ConstrainedStop::Length) => run_report::FinishReason::Length,
        None => run_report::FinishReason::from_decode(generated, stop_tokens, budget),
    }
}
