// PMAT-1098 — `apr parity` refuses, BEFORE any forward pass, the architectures
// its dense CPU-vs-GPU loop cannot route.
//
// Included from parity.rs.
//
// The 0.68.0 T-2 pre-publish dogfood (release commit 27f070324) read three C14
// rows as MODEL defects:
//
//   FAIL qwen3-coder-30b: … matmul weight has EMPTY data buffer (qtype=0)
//   FAIL qwen3-30b:       … same
//   FAIL qwen3.5-0.8b:    … Architecture 'qwen35' uses SSM/Gated Delta Net layers
//
// None of the three is a model defect. `apr parity` builds an
// `OwnedQuantizedModel::from_mapped` and drives `forward_single_with_cache`
// (CPU) against `forward_gpu_resident` (GPU) — the DENSE loop:
//
//   * For a MoE GGUF, `from_gguf_for_moe` deliberately leaves the dense FFN
//     tensor refs as 0-byte F32 placeholders, because `run`/`serve` route MoE
//     through `run_qwen3_moe_generate`. The dense loop reads the placeholder,
//     so "EMPTY data buffer" is the TOOL describing its own wrong turn. Real
//     MoE parity support is #3367 (`apr chat` had the identical defect,
//     #3368).
//   * For Qwen3.5 (Gated DeltaNet + gated full attention), no GPU forward
//     exists at all (#3090), so GPU==CPU is unmeasurable BY CONSTRUCTION — no
//     run of this command could ever produce a number.
//
// A tool that cannot measure something must say so. Falling through to the
// dense loop produced a red row that named the MODEL, which is the exact class
// `scripts/dogfood.sh`'s C14 comment warns about; the refusal below names the
// TOOL and carries the issue that will lift it.

// Every item here is `#[allow(dead_code)]`: the only NON-test caller is the
// `#[cfg(feature = "cuda")]` arm of `parity::run`, so a build without `cuda`
// (the default `cargo clippy -p apr-cli --lib`) sees the predicate used by the
// case table alone. The refusal is deliberately NOT cuda-gated itself — it must
// stay testable in the ordinary `cargo nextest run -p apr-cli --lib` a
// non-GPU host runs.

/// Exit code for an architecture refusal: `CliError::NotImplemented` → **12**.
///
/// Distinct from every other code `apr parity` can emit — 3 (`FileNotFound`),
/// 5 (`ValidationFailed`, a disproven parity), 8 (`InferenceFailed`, a forward
/// that actually ran and died), 9 (`FeatureDisabled`, a non-cuda build). That
/// distinctness is what lets `scripts/check_model_parity.sh` classify a refusal
/// as `UNMEASURED-TOOL` without laundering a genuine crash as one, and it is
/// asserted in the case table rather than assumed.
///
/// `NotImplemented` is the honest variant: parity for this architecture is an
/// advertised capability whose implementation does not exist yet (#2407's rule
/// — a stub must fail, not print something and exit 0).
#[cfg(feature = "inference")]
#[allow(dead_code)]
pub(crate) const PARITY_REFUSED_EXIT: u8 = 12;

/// One architecture `apr parity` will not measure, and why.
#[cfg(feature = "inference")]
#[allow(dead_code)]
pub(crate) struct ParityRefusal {
    /// The architecture as this refusal names it: the canonical key from
    /// `realizar`'s normalizer where that key is itself the refused one
    /// (`qwen3moe` → `qwen3_moe`), and otherwise the RAW GGUF tag.
    ///
    /// `tensor_names::normalize_architecture` folds `qwen35` → `qwen3` and
    /// `qwen3_5moe` → `llama`, both DENSE architectures that parity runs
    /// happily. A refusal that printed `architecture=qwen3` would send the
    /// reader to the wrong code, so the normalizer is never allowed to rename a
    /// refusal into an architecture that is not refused.
    pub(crate) architecture: String,
    /// Why the dense parity loop cannot route it.
    pub(crate) reason: &'static str,
    /// The issue that will lift the refusal.
    pub(crate) issue: &'static str,
}

#[cfg(feature = "inference")]
#[allow(dead_code)]
impl ParityRefusal {
    /// The single stderr line. `scripts/check_model_parity.sh` greps the
    /// `parity: REFUSED architecture=` prefix; both case tables pin the shape.
    pub(crate) fn line(&self) -> String {
        format!(
            "parity: REFUSED architecture={} — {} ({})",
            self.architecture, self.reason, self.issue
        )
    }

    /// `--json` form. Deliberately carries NO `metrics`/`parity` key: a refusal
    /// is not a parity result, and the C14 judge must not read it as one.
    pub(crate) fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "refused": {
                "architecture": self.architecture,
                "reason": self.reason,
                "issue": self.issue,
            }
        })
    }

    /// The error carrying [`PARITY_REFUSED_EXIT`].
    pub(crate) fn into_error(self) -> CliError {
        CliError::NotImplemented(self.line())
    }
}

/// The refusal predicate: `Some(_)` iff the dense parity loop would compute
/// something other than this model's forward pass.
///
/// Takes the raw `general.architecture` string and the tensor NAMES — both
/// readable from the GGUF header (`MappedGGUFModel`), so the refusal happens
/// before any weight is materialized, which is what keeps an 18 GB MoE from
/// being loaded only to be refused.
///
/// Both arms delegate to `realizar`'s existing single sources of truth rather
/// than re-deriving them here:
///   * `ArchConstraints::from_architecture(..).is_moe` — the same flag that
///     gates `CudaExecutor::build_indexed_weights` off the dense FFN tensor
///     names, and which matches the raw (`qwen3moe`) as well as the canonical
///     (`qwen3_moe`) spelling.
///   * `unsupported_architecture_reason(..)` — the predicate `apr check` and
///     `apr ptx-map` already share so they cannot drift (GH-704 / #2399).
///
/// The Qwen3.5 arm additionally matches the architecture TAG, because a
/// header-only read must refuse even if the tensor scan is inconclusive; the
/// spelling set is the one `ArchConstraints` uses
/// (`crates/aprender-serve/src/gguf/arch_constraints_fallback.rs`).
#[cfg(feature = "inference")]
#[allow(dead_code)]
pub(crate) fn parity_refusal_for<'n>(
    architecture: &str,
    tensor_names: impl IntoIterator<Item = &'n str>,
) -> Option<ParityRefusal> {
    use realizar::gguf::{unsupported_architecture_reason, ArchConstraints};

    if ArchConstraints::from_architecture(architecture).is_moe {
        // `normalize_architecture` is not total over the MoE spellings
        // `ArchConstraints` accepts (`qwen3_5moe` hits its `_ => "llama"`
        // arm), so the canonical key is used only when it IS the MoE key;
        // otherwise the refusal reports the raw tag. A refusal must never
        // name a dense architecture.
        let canonical = realizar::tensor_names::normalize_architecture(architecture);
        return Some(ParityRefusal {
            architecture: if canonical == "qwen3_moe" {
                canonical.to_string()
            } else {
                architecture.to_string()
            },
            reason: "the dense parity loop would read the MoE placeholder tensors; \
                     run/serve route this architecture through the MoE forward and parity does not yet",
            issue: "#3367",
        });
    }

    let hybrid_tag = matches!(
        architecture.to_ascii_lowercase().as_str(),
        "qwen3_5" | "qwen3.5" | "qwen35"
    );
    if hybrid_tag || unsupported_architecture_reason(architecture, tensor_names).is_some() {
        return Some(ParityRefusal {
            architecture: architecture.to_string(),
            reason: "the GPU forward for this architecture is not implemented; \
                     GPU=CPU cannot be measured",
            issue: "#3090",
        });
    }

    None
}
