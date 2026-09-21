//! GH-280: Capability Match Gate for `apr qa`.
//!
//! Validates that the model's required operations are supported by the GPU
//! backend BEFORE running inference. This is Gate 0 — the earliest gate —
//! because if the GPU can't run the model, all GPU-dependent gates are moot.
//!
//! For GGUF files: reads architecture from metadata, derives constraints,
//! checks GPU capability.
//!
//! For non-GGUF files (APR, SafeTensors): skips gracefully (these formats
//! don't yet carry architecture constraints in a standardized way).

use super::qa::{GateResult, QaConfig};
use crate::error::Result;
use std::path::Path;
#[cfg(feature = "inference")]
use std::time::Duration;
use std::time::Instant;

/// Run the capability match gate.
///
/// Reads the model's architecture string from GGUF metadata, derives the
/// required operations from `ArchConstraints`, and checks whether the GPU
/// backend supports all of them.
///
/// # Returns
///
/// - `GateResult::passed` if all required ops are GPU-supported (or non-GGUF)
/// - `GateResult::failed` if the GPU lacks required kernels (lists missing ops)
/// - `GateResult::skipped` if the file can't be read or isn't GGUF
#[provable_contracts_macros::contract(
    "apr-cli-command-safety-v1",
    equation = "read_only_no_side_effects"
)]
pub fn run_capability_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!(
            "{}",
            colored::Colorize::yellow("Running capability match gate (GH-280)...")
        );
    }

    // Read file header to detect format
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            let duration = start.elapsed();
            return Ok(GateResult::failed(
                "capability_match",
                &format!("Failed to read model file: {e}"),
                None,
                None,
                duration,
            ));
        }
    };

    if data.len() < 4 {
        let duration = start.elapsed();
        return Ok(GateResult::failed(
            "capability_match",
            "Model file too small to detect format",
            None,
            None,
            duration,
        ));
    }

    // Only check GGUF files — APR/SafeTensors don't carry arch constraints yet
    let magic = &data[0..4];
    if magic != b"GGUF" {
        let duration = start.elapsed();
        return Ok(GateResult::passed(
            "capability_match",
            "Non-GGUF format — capability check not applicable",
            None,
            None,
            duration,
        ));
    }

    // Parse GGUF to get architecture string and tensor (name, GGML type) pairs
    let Some((arch, tensors)) = extract_gguf_arch_and_tensors(&data) else {
        let duration = start.elapsed();
        return Ok(GateResult::passed(
            "capability_match",
            "GGUF missing architecture metadata — skipping capability check",
            None,
            None,
            duration,
        ));
    };

    // PMAT-1098: Check for globally unsupported architectures (like SSM/Gated Delta Net).
    // #3091: judged per BACKEND — this binary's capability is the GPU's only when
    // it was built with `cuda`; otherwise the gate speaks for the CPU forward,
    // which runs Qwen3.5 Gated DeltaNet today.
    // #3477: a declined GPU is not automatically a failed gate — see
    // `ssm_gate_result`.
    #[cfg(feature = "inference")]
    {
        let tensor_refs: Vec<&str> = tensors.iter().map(|(n, _)| n.as_str()).collect();
        let backend = if cfg!(feature = "cuda") {
            Backend::Gpu
        } else {
            Backend::Cpu
        };
        // #3477 / PMAT-781/783/785: the hybrid's Gated DeltaNet projections are
        // invisible to the dense construction-time quant gate (the qwen35 base
        // model carries an empty layer list), so the GPU rung judges them here,
        // from the header, before anything is uploaded.
        if let Some(result) = hybrid_quant_gate_result(&arch, &tensors, backend, start.elapsed()) {
            return Ok(result);
        }
        if let Some(result) = ssm_gate_result(&arch, tensor_refs, backend, start.elapsed()) {
            return Ok(result);
        }
    }

    // A build without the `cuda` feature has no GPU backend compiled in, so
    // "all N required ops supported by GPU" would be a claim about kernels
    // this binary cannot reach. SKIP, matching the gpu_speedup and
    // gpu_state_isolation gates in the same report.
    #[cfg(all(feature = "inference", not(feature = "cuda")))]
    {
        let _ = arch;
        return Ok(GateResult::skipped(
            "capability_match",
            "Requires 'inference' and 'cuda' features",
        ));
    }

    // Derive constraints and check capability
    #[cfg(all(feature = "inference", feature = "cuda"))]
    {
        use realizar::capability::{check_capability, gpu_supported_ops, required_ops};
        use realizar::gguf::ArchConstraints;

        let constraints = ArchConstraints::from_architecture(&arch);
        let required = required_ops(&constraints);
        let supported = gpu_supported_ops();

        let duration = start.elapsed();
        match check_capability(&required, &supported) {
            Ok(()) => Ok(GateResult::passed(
                "capability_match",
                &format!(
                    "Architecture '{}': all {} required ops supported by GPU",
                    arch,
                    required.len()
                ),
                Some(required.len() as f64),
                Some(0.0),
                duration,
            )),
            Err(missing) => {
                let missing_names: Vec<String> = missing.iter().map(ToString::to_string).collect();
                Ok(GateResult::failed(
                    "capability_match",
                    &format!(
                        "Architecture '{}': GPU missing kernel support for [{}]. \
                         GPU inference will produce garbage — use CPU.",
                        arch,
                        missing_names.join(", ")
                    ),
                    Some(missing.len() as f64),
                    Some(0.0),
                    duration,
                ))
            }
        }
    }

    #[cfg(not(feature = "inference"))]
    {
        let _ = arch;
        let duration = start.elapsed();
        Ok(GateResult::passed(
            "capability_match",
            "Capability check requires inference feature",
            None,
            None,
            duration,
        ))
    }
}

/// Which backend's capability the gate is judging.
#[cfg(feature = "inference")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Backend {
    /// The CPU forward (`apr run --no-gpu`, and the default on a non-cuda build).
    Cpu,
    /// The CUDA backend.
    Gpu,
}

/// What the capability gate concluded for one backend.
#[cfg(feature = "inference")]
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum CapabilityVerdict {
    /// This backend can run the model — the gate must not fail on it.
    Admitted,
    /// This backend cannot; the string is the user-facing reason.
    Declined(String),
}

/// Is this the architecture string the runtime dispatches to the Qwen3.5
/// (Gated DeltaNet) hybrid forward?
///
/// #3091: `infer/inference_result.rs` compares the GGUF architecture to the
/// literal `"qwen35"` and, on a match, builds the model with
/// `realizar::gguf::forward_qwen35::Qwen35Model::create_base_model`. This gate
/// mirrors that comparison exactly: admitting a spelling the runtime does not
/// dispatch (`qwen3_5`, `qwen3.5`) would have `apr qa` promise a load that
/// `apr run` refuses — the very drift #3091 is about, with the sign flipped.
///
/// #3477: the comparison is no longer COPIED here. `realizar` exposes it as
/// `gguf::hybrid_forward_handles`, and `apr parity` / `apr ptx-map` ask the same
/// function, so the three tools cannot drift from the runtime or from each
/// other.
#[cfg(feature = "inference")]
pub(crate) fn cpu_forward_handles(arch: &str) -> bool {
    realizar::gguf::hybrid_forward_handles(arch)
}

/// Verdict for a GGUF carrying SSM / Gated DeltaNet tensors.
///
/// `realizar::gguf::unsupported_architecture_reason` stays the single predicate
/// for "is there any forward for this SSM file", and since #3477 it answers NO
/// REFUSAL for the hybrid the runtime dispatches — both backends implement it
/// (CPU #3091, GPU #3090). The per-backend judgement left here is about THIS
/// BINARY: a build without the `cuda` feature has no GPU backend linked in at
/// all, so it cannot certify one.
#[cfg(feature = "inference")]
pub(crate) fn hybrid_ssm_verdict<'n>(
    arch: &str,
    tensor_names: impl IntoIterator<Item = &'n str>,
    backend: Backend,
) -> CapabilityVerdict {
    // #3477: the hybrid is judged FIRST, because the shared predicate no longer
    // refuses it — falling through would admit it on a non-cuda build's GPU
    // backend, which does not exist.
    if cpu_forward_handles(arch) {
        return match backend {
            Backend::Cpu => CapabilityVerdict::Admitted,
            Backend::Gpu if cfg!(feature = "cuda") => CapabilityVerdict::Admitted,
            Backend::Gpu => CapabilityVerdict::Declined(format!(
                "Architecture '{arch}': the Gated DeltaNet GPU forward exists (#3090), but this \
                 binary has no CUDA backend — it was built without `--features cuda`. The CPU \
                 forward runs it here (#3091)."
            )),
        };
    }
    let Some(reason) = realizar::gguf::unsupported_architecture_reason(arch, tensor_names) else {
        return CapabilityVerdict::Admitted;
    };
    CapabilityVerdict::Declined(reason)
}

/// The Capability Match message for an architecture only the CPU rung certifies.
///
/// #3477: reachable only on a build WITHOUT `cuda` — the GPU forward for the
/// hybrid exists (#3090); what a non-cuda binary lacks is the backend itself.
#[cfg(feature = "inference")]
pub(crate) fn cpu_only_pass_message(arch: &str) -> String {
    format!(
        "Architecture '{arch}': CPU-only in this binary — Gated DeltaNet runs on the CPU (#3091) \
         and on the GPU (#3090), but this build has no CUDA backend, GPU gates skipped"
    )
}

/// The Capability Match result for a GGUF carrying SSM / Gated DeltaNet tensors,
/// or `None` when the file is admitted and the per-op GPU check should run.
///
/// #3477: the gate used to turn every `Declined` into a FAIL. On a cuda build
/// that made `apr qa <qwen35 gguf>` fail Gate 0 and skip 8 of 12 later gates
/// "due to capability match failure" — exit 5 for a model `apr run` serves
/// correctly on the CPU (#3091). The GPU refusal (#3090) is real, so the GPU
/// gates still skip; what changed is that the CPU rung now gets certified
/// instead of the whole run being abandoned. An architecture NO backend runs is
/// still a FAIL: `cpu_forward_handles` is the discriminator, and it mirrors the
/// runtime's own dispatch.
#[cfg(feature = "inference")]
pub(crate) fn ssm_gate_result<'n>(
    arch: &str,
    tensor_names: impl IntoIterator<Item = &'n str>,
    backend: Backend,
    duration: Duration,
) -> Option<GateResult> {
    let CapabilityVerdict::Declined(reason) = hybrid_ssm_verdict(arch, tensor_names, backend)
    else {
        return None;
    };
    if backend == Backend::Gpu && cpu_forward_handles(arch) {
        return Some(GateResult::passed(
            "capability_match",
            &cpu_only_pass_message(arch),
            None,
            None,
            duration,
        ));
    }
    Some(GateResult::failed(
        "capability_match",
        &reason,
        Some(1.0),
        Some(0.0),
        duration,
    ))
}

/// The Capability Match FAIL for a hybrid GGUF whose Gated DeltaNet weights
/// carry a quant type with no GPU GEMV kernel, or `None`.
///
/// PMAT-781/783/785: the GPU weight upload decodes an unrecognized GGML type as
/// Q4_K, so such a file would produce garbage logits rather than an error. The
/// dense construction-time gate (`OwnedQuantizedModel::has_gpu_unsupported_quant`)
/// cannot catch it for the hybrid — the `OwnedQuantizedModel` built for `qwen35`
/// has an EMPTY layer list, the DeltaNet weights living in `Qwen35Model` — so
/// the verdict is taken from the header here. Only the GPU rung asks: the CPU
/// forward decodes every quant type correctly.
#[cfg(feature = "inference")]
pub(crate) fn hybrid_quant_gate_result(
    arch: &str,
    tensors: &[(String, u32)],
    backend: Backend,
    duration: Duration,
) -> Option<GateResult> {
    if backend != Backend::Gpu {
        return None;
    }
    let reason = realizar::gguf::hybrid_gpu_quant_refusal(
        arch,
        tensors.iter().map(|(n, q)| (n.as_str(), *q)),
    )?;
    Some(GateResult::failed(
        "capability_match",
        &reason,
        Some(1.0),
        Some(0.0),
        duration,
    ))
}

/// Does the GPU decline this architecture while the CPU forward runs it?
///
/// This is the predicate the `apr qa` driver asks to decide which gates it can
/// still execute. It must stay exactly the (GPU declines, CPU runs) case:
/// answering yes for a dense transformer would silently strip that model's GPU
/// gates, and answering yes for an architecture no backend runs would certify a
/// model `apr run` refuses to load.
#[cfg(feature = "inference")]
pub(crate) fn is_cpu_only_architecture<'n>(
    arch: &str,
    tensor_names: impl IntoIterator<Item = &'n str>,
) -> bool {
    cpu_forward_handles(arch)
        && matches!(
            hybrid_ssm_verdict(arch, tensor_names, Backend::Gpu),
            CapabilityVerdict::Declined(_)
        )
}

/// Is the model at `path` an architecture only the CPU rung can certify?
///
/// Always `false` on a build without `cuda`: there is no GPU backend compiled
/// in, so there are no GPU gates to skip and the existing "Requires 'inference'
/// and 'cuda' features" skips already speak for them.
#[cfg(feature = "inference")]
pub(crate) fn cpu_only_architecture(path: &Path) -> bool {
    if !cfg!(feature = "cuda") {
        return false;
    }
    let Ok(data) = std::fs::read(path) else {
        return false;
    };
    if data.len() < 4 || &data[0..4] != b"GGUF" {
        return false;
    }
    let Some((arch, tensors)) = extract_gguf_arch_and_tensors(&data) else {
        return false;
    };
    is_cpu_only_architecture(&arch, tensors.iter().map(|(n, _)| n.as_str()))
}

/// Without the `inference` feature there is no runtime to speak for.
#[cfg(not(feature = "inference"))]
pub(crate) fn cpu_only_architecture(_path: &Path) -> bool {
    false
}

/// Is the model at `path` a hybrid the DENSE GGUF loader refuses?
///
/// #3477: `cpu_only_architecture` answers "which backend", which for `qwen35` on
/// a cuda build is now BOTH (#3090 GPU, #3091 CPU). A separate question survives
/// it: a gate built on `OwnedQuantizedModel::from_mapped` — Ollama parity,
/// cross-format parity, GPU speedup, GPU state isolation — still cannot run,
/// because that loader has no hybrid path on either rung. Those gates SKIP with
/// that reason; they must not FAIL, and they must not be answered with a number
/// measured on some other model shape.
#[cfg(feature = "inference")]
pub(crate) fn hybrid_loader_architecture(path: &Path) -> bool {
    let Ok(data) = std::fs::read(path) else {
        return false;
    };
    if data.len() < 4 || &data[0..4] != b"GGUF" {
        return false;
    }
    let Some((arch, _)) = extract_gguf_arch_and_tensors(&data) else {
        return false;
    };
    cpu_forward_handles(&arch)
}

/// Without `inference` there is no loader to speak for.
#[cfg(not(feature = "inference"))]
pub(crate) fn hybrid_loader_architecture(_path: &Path) -> bool {
    false
}

/// #3714: is this a Qwen3-MoE file? The dense loader BUILDS it (the MoE
/// placeholder fills its dense FFN) but cannot FORWARD it, so its golden and
/// throughput gates go through the runtime entry point — the routed-expert
/// forward, CUDA (#3714) or CPU (#3367) — and every gate measured through the
/// dense `OwnedQuantizedModel`/`OwnedQuantizedModelCuda` skips, saying so.
///
/// Reads the header through the mmap, never the whole file: the siblings
/// above `std::fs::read` all 18.5 GB of a 30B-A3B file to ask the same kind
/// of question.
#[cfg(feature = "inference")]
pub(crate) fn moe_loader_architecture(path: &Path) -> bool {
    realizar::gguf::MappedGGUFModel::from_path(path).is_ok_and(|mapped| {
        mapped
            .model
            .architecture()
            .is_some_and(realizar::gguf::moe_forward_handles)
    })
}

/// Without `inference` there is no runtime to route to.
#[cfg(not(feature = "inference"))]
pub(crate) fn moe_loader_architecture(_path: &Path) -> bool {
    false
}

/// Extract the architecture string and the tensor table from GGUF metadata.
///
/// Uses aprender's GGUF reader to parse metadata without loading tensors.
/// #3477: the GGML type travels with the name, because the hybrid GPU quant
/// gate judges `(name, type)` pairs — reading the names alone was what let an
/// unsupported-quant DeltaNet tensor through.
fn extract_gguf_arch_and_tensors(data: &[u8]) -> Option<(String, Vec<(String, u32)>)> {
    let reader = aprender::format::gguf::reader::GgufReader::from_bytes(data.to_vec()).ok()?;
    let arch = reader.architecture()?;
    let tensors = reader
        .tensors
        .into_iter()
        .map(|t| (t.name, t.dtype))
        .collect();
    Some((arch, tensors))
}

#[cfg(all(test, feature = "inference"))]
mod qa_capability_ssm_tests {
    use super::{hybrid_ssm_verdict, Backend, CapabilityVerdict};

    // #3091: the CPU forward RUNS Qwen3.5 (Gated DeltaNet) — `apr run` and
    // `apr chat` dispatch architecture "qwen35" to
    // `realizar::gguf::forward_qwen35` (infer/inference_result.rs). `apr qa`
    // kept refusing the same file with "NEITHER the CPU nor the GPU backend
    // implements Gated DeltaNet", so the tool contradicted the runtime.
    #[test]
    fn qwen35_ssm_model_is_admitted_on_cpu() {
        let verdict =
            hybrid_ssm_verdict("qwen35", ["token_embd.weight", "blk.0.ssm_a"], Backend::Cpu);
        assert_eq!(
            verdict,
            CapabilityVerdict::Admitted,
            "the CPU forward runs qwen35 (#3091), so the capability gate must admit it"
        );
    }

    // #3477 (operator ruling 2026-09-19): the GPU forward for the hybrid exists
    // (`Qwen35CudaModel`, #3090), so a binary built WITH `cuda` must admit it
    // on the GPU backend. A binary built WITHOUT `cuda` has no GPU backend
    // linked in at all — that is the only thing left to decline, and the reason
    // must say so rather than claiming the kernels do not exist.
    #[test]
    fn qwen35_ssm_model_is_admitted_on_gpu_under_cuda() {
        let verdict = hybrid_ssm_verdict("qwen35", ["blk.0.ssm_a"], Backend::Gpu);
        if cfg!(feature = "cuda") {
            assert_eq!(
                verdict,
                CapabilityVerdict::Admitted,
                "the GPU runs qwen35 since #3090 — the capability gate must admit it"
            );
            return;
        }
        let CapabilityVerdict::Declined(reason) = verdict else {
            panic!("a binary without the cuda feature has no GPU backend to certify");
        };
        assert!(
            reason.contains("no CUDA backend"),
            "the reason must be this BINARY's missing backend, not missing kernels: {reason}"
        );
        assert!(
            reason.contains("#3090"),
            "must cite the GPU issue: {reason}"
        );
        assert!(
            !reason.contains("NEITHER the CPU nor the GPU"),
            "the CPU claim is false since #3091: {reason}"
        );
        assert!(
            !reason.contains("the GPU backend has no"),
            "the GPU backend HAS Gated DeltaNet kernels since #3090: {reason}"
        );
    }

    // #3477 / PMAT-781/783/785: a Gated DeltaNet projection carrying a quant
    // type with no GPU GEMV kernel must FAIL the GPU rung loudly. The dense
    // construction-time gate cannot see these tensors at all (the qwen35 base
    // model has an empty layer list), so before this the file went to the GPU
    // and was decoded as Q4_K.
    #[test]
    fn qwen35_with_a_gpu_unsupported_quant_fails_the_gpu_rung() {
        let tensors = vec![
            ("token_embd.weight".to_string(), 12u32),
            ("blk.0.ssm_beta.weight".to_string(), 11), // Q3_K: no GPU kernel
        ];
        let result = super::hybrid_quant_gate_result(
            "qwen35",
            &tensors,
            Backend::Gpu,
            std::time::Duration::from_millis(1),
        )
        .expect("a Q3_K DeltaNet projection has no GPU GEMV kernel");
        assert!(
            !result.passed,
            "the GPU rung must FAIL, not warn: {result:?}"
        );
        assert!(
            result.message.contains("blk.0.ssm_beta.weight"),
            "the message must name the tensor: {}",
            result.message
        );
        // The CPU decodes every quant type correctly — it must not be failed.
        assert!(
            super::hybrid_quant_gate_result(
                "qwen35",
                &tensors,
                Backend::Cpu,
                std::time::Duration::from_millis(1)
            )
            .is_none(),
            "the CPU forward runs any quant type — only the GPU rung asks"
        );
        // A hybrid whose projections are all GPU-eligible passes through.
        let ok = vec![
            ("blk.0.attn_qkv.weight".to_string(), 8u32),
            ("blk.0.ssm_out.weight".to_string(), 12),
        ];
        assert!(
            super::hybrid_quant_gate_result(
                "qwen35",
                &ok,
                Backend::Gpu,
                std::time::Duration::from_millis(1)
            )
            .is_none(),
            "Q8_0/Q4_K have verified GPU kernels — this model is GPU-eligible"
        );
    }

    // The admission mirrors the runtime dispatch, which compares the GGUF
    // architecture to the literal "qwen35" (infer/inference_result.rs). A
    // spelling the runtime does not dispatch must keep the full refusal, or
    // `apr qa` would promise a load that `apr run` refuses.
    #[test]
    fn a_spelling_the_runtime_does_not_dispatch_is_still_refused() {
        for arch in ["qwen3_5", "qwen3.5", "mamba"] {
            let CapabilityVerdict::Declined(reason) =
                hybrid_ssm_verdict(arch, ["blk.0.ssm_a"], Backend::Cpu)
            else {
                panic!("'{arch}' is not dispatched to forward_qwen35 — it must be declined");
            };
            assert!(
                reason.contains("no CPU or GPU forward for architecture"),
                "an undispatched hybrid keeps the refusal: {reason}"
            );
        }
    }

    // #3477: on a cuda build the gate no longer stops here at all — the GPU
    // forward exists (#3090), so the hybrid falls through to the ordinary
    // per-op GPU check like any other architecture. On a build without `cuda`
    // there is no GPU backend to certify, and the CPU rung still carries the
    // report (the earlier FAIL turned 8 of 12 gates into "Skipped due to
    // capability match failure" and exited 5 on a model that generates correct
    // text).
    #[test]
    fn qwen35_on_the_gpu_backend_is_a_cpu_only_pass() {
        let result = super::ssm_gate_result(
            "qwen35",
            ["token_embd.weight", "blk.0.ssm_a"],
            Backend::Gpu,
            std::time::Duration::from_millis(1),
        );
        if cfg!(feature = "cuda") {
            assert!(
                result.is_none(),
                "the GPU runs qwen35 (#3090) — the per-op GPU check must run, not a CPU-only PASS: {result:?}"
            );
            return;
        }
        let result = result.expect("without cuda there is no GPU backend to certify");
        assert!(
            result.passed && !result.skipped,
            "the CPU rung certifies this model, so the gate PASSES: {result:?}"
        );
        assert!(
            result.message.contains("CPU-only"),
            "the message must name the verdict: {}",
            result.message
        );
        assert!(
            result.message.contains("#3091") && result.message.contains("#3090"),
            "the message must cite both forwards: {}",
            result.message
        );
        assert!(
            !result.message.contains("NEITHER the CPU nor the GPU"),
            "the CPU claim is false since #3091: {}",
            result.message
        );
    }

    // The CPU-only PASS is exactly the (GPU declines, CPU runs) case. An
    // architecture NO backend runs still FAILS — otherwise the gate would
    // certify a model `apr run` refuses to load.
    #[test]
    fn a_hybrid_no_backend_runs_is_not_a_cpu_only_pass() {
        for backend in [Backend::Cpu, Backend::Gpu] {
            let result = super::ssm_gate_result(
                "mamba",
                ["blk.0.ssm_a"],
                backend,
                std::time::Duration::from_millis(1),
            )
            .expect("an undispatched hybrid is declined on every backend");
            assert!(
                !result.passed,
                "no backend runs 'mamba' — the gate must FAIL ({backend:?}): {result:?}"
            );
        }
        assert!(
            !super::is_cpu_only_architecture("mamba", ["blk.0.ssm_a"]),
            "'mamba' has no CPU forward — it is not CPU-only, it is unsupported"
        );
    }

    // The driver asks this predicate which gates to skip. A dense transformer
    // must answer no, or a GPU-capable model would lose its GPU gates.
    #[test]
    fn only_a_gpu_declined_cpu_running_architecture_is_cpu_only() {
        assert_eq!(
            super::is_cpu_only_architecture("qwen35", ["blk.0.ssm_a"]),
            !cfg!(feature = "cuda"),
            "#3477: with cuda the GPU runs qwen35 (#3090) so its GPU gates must NOT be skipped; \
             without cuda this binary has no GPU backend and the CPU rung carries the report"
        );
        assert!(
            !super::is_cpu_only_architecture("qwen2", ["blk.0.attn_q.weight"]),
            "a dense transformer runs on the GPU — its GPU gates must still run"
        );
    }

    // A dense transformer must be admitted on both backends — a predicate that
    // declined everything would satisfy the GPU test above.
    #[test]
    fn a_dense_transformer_is_admitted_on_both_backends() {
        for backend in [Backend::Cpu, Backend::Gpu] {
            assert_eq!(
                hybrid_ssm_verdict("qwen2", ["blk.0.attn_q.weight"], backend),
                CapabilityVerdict::Admitted,
                "a dense transformer has no SSM tensors ({backend:?})"
            );
        }
    }
}
