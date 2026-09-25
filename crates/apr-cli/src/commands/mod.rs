//! Command implementations
//!
//! Each command follows Toyota Way principles:
//! - Genchi Genbutsu: Go and see the actual data
//! - Jidoka: Stop on quality issues
//! - Visualization: Make problems visible

pub(crate) mod aliases;
pub(crate) mod threshold_arg;
// GH-2391 falsifiers for the whole threshold/tolerance flag family, not just
// the *-lint commands the module originally covered.
#[cfg(test)]
mod threshold_arg_gh2391;

pub(crate) mod attn_parity_classifier;
pub(crate) mod attn_parity_lint;
pub(crate) mod attn_viz_classifier;
pub(crate) mod attn_viz_lint;
pub(crate) mod audio_inspect;
pub(crate) mod audio_inspect_classifier;
pub(crate) mod audio_inspect_lint;
pub(crate) mod auto_quant;
pub(crate) mod awq_classifier;
pub(crate) mod awq_lint;
pub mod beat_run;
pub mod bench;
pub(crate) mod blob_gc;
pub mod canary;
pub mod capability;
pub mod cbtop;
pub mod chat;
pub mod check;
pub(crate) mod check_finite_classifier;
pub(crate) mod check_finite_lint;
pub mod compare_hf;
pub(crate) mod compile;
pub(crate) mod convert;
pub(crate) mod copy_tag;
pub(crate) mod ddp_metrics_classifier;
pub(crate) mod ddp_metrics_lint;
pub(crate) mod debug;
pub(crate) mod diff;
pub(crate) mod diff_quant_roundtrip;
pub(crate) mod distill;
#[cfg(all(feature = "cuda", feature = "training", feature = "inference"))]
pub(crate) mod distill_q4k_teacher;
pub mod modelfile;
pub(crate) mod rm_gc_lint;

pub(crate) mod data;
pub(crate) mod diagnose;

#[cfg(feature = "inference")]
pub(crate) mod devices;
pub(crate) mod dry_sampling_classifier;
pub(crate) mod dry_sampling_lint;
pub(crate) mod embed;
pub(crate) mod embed_viz;
pub(crate) mod embed_viz_classifier;
pub(crate) mod embed_viz_lint;
pub(crate) mod embeddings_classifier;
pub(crate) mod embeddings_lint;
pub(crate) mod eval;
pub(crate) mod eval_attach;
#[cfg(feature = "training")]
pub(crate) mod experiment;
pub(crate) mod explain;
pub(crate) mod explain_token_classifier;
pub(crate) mod explain_token_lint;
pub(crate) mod export;
#[cfg(feature = "training")]
pub(crate) mod finetune;
pub(crate) mod flow;
pub(crate) mod fp8_classifier;
pub(crate) mod fp8_lint;
pub(crate) mod gbnf_classifier;
pub(crate) mod gbnf_lint;
pub(crate) mod glob_filter;
pub(crate) mod gptq_classifier;
pub(crate) mod gptq_lint;
#[cfg(feature = "training")]
pub(crate) mod gpu;
pub(crate) mod gpu_memtrace_classifier;
pub(crate) mod gpu_memtrace_lint;
pub(crate) mod grad_norm;
pub(crate) mod hang_trace_classifier;
pub(crate) mod hang_trace_lint;
pub(crate) mod hex;
pub(crate) mod hf_endpoint;
pub(crate) mod imatrix_classifier;
pub(crate) mod imatrix_lint;
pub(crate) mod import;
pub(crate) mod inspect;
pub(crate) mod kernel_explain;
pub(crate) mod kernel_parity;
pub(crate) mod kv_timeline_classifier;
pub(crate) mod kv_timeline_lint;
pub(crate) mod lint;
pub(crate) mod lint_error;
// Poka-yoke for the *-lint family error surface (#2377-8/-9): scans the family's
// own source so the class cannot be reintroduced by the next copy-paste.
#[cfg(test)]
mod lint_exit_convention_tests;
pub(crate) mod lint_family_guard;
pub(crate) mod lint_vacuity;
pub(crate) mod manifest;
pub(crate) mod mcp;
pub(crate) mod merge;
#[cfg(feature = "training")]
pub(crate) mod model_config;
// #3661: a model parse failure names the format its magic bytes identify.
#[cfg(test)]
mod model_file_error_tests_3661;
#[cfg(feature = "training")]
pub(crate) mod monitor;
#[cfg(feature = "dev")]
pub mod mono;
pub(crate) mod multi_lora_classifier;
pub(crate) mod nccl_diag_classifier;
pub(crate) mod nccl_diag_lint;
pub(crate) mod nf4_classifier;
pub(crate) mod nf4_lint;
pub(crate) mod offline;
pub(crate) mod ollama_chat;
pub(crate) mod ollama_chat_classifier;
pub(crate) mod ollama_tool_call_classifier;
pub(crate) mod ollama_tools_lint;
pub(crate) mod oom_classifier;
pub(crate) mod oom_lint;
pub(crate) mod oracle;
pub(crate) mod otlp_classifier;
pub(crate) mod otlp_lint;
pub(crate) mod parity;
/// REG-15 model admission (#2971, PMAT-1065): a forced backend never
/// downgrades on a load-time parity-gate failure. Re-exported publicly at
/// `apr_cli::parity_admission` (see `lib.rs`) so `tests/reg15_admission.rs`
/// can exercise it without reaching into the private `commands` tree.
pub(crate) mod parity_admission;
pub(crate) mod parity_per_op;
pub(crate) mod parity_per_op_table;
pub(crate) mod pipeline;
pub(crate) mod png_encode;
pub(crate) mod ppl;
#[cfg(feature = "training")]
pub(crate) mod pretrain;
pub(crate) mod probar;
// GH-876 Milestone 2: `apr test llm`, a surface over the in-tree llm module.
pub(crate) mod profile;
pub(crate) mod progress;
pub(crate) mod prometheus_classifier;
pub(crate) mod prometheus_lint;
pub(crate) mod prune;
pub(crate) mod ps_schema;
pub(crate) mod test_llm;
// PERF-025: the `--band` mode of `apr test llm bench`. Its own file so that
// `test_llm.rs`'s complexity is untouched by it -- the PMAT pre-commit gate
// blocks a FILE with any pre-existing violation, not a function.
pub(crate) mod test_llm_band;
// #2399: gated on the crate it actually needs (aprender-explain, aliased
// `trueno-explain`) rather than on `full`, so `--features ptx` is enough and a
// user does not have to pull CUDA + training to analyze a .ptx file.
pub(crate) mod model_header;
#[cfg(feature = "trueno-explain")]
pub(crate) mod ptx_explain;
pub(crate) mod ptx_map;
pub(crate) mod publish;
pub(crate) mod pull;
pub(crate) mod pull_scheme;
pub(crate) mod pull_verify;
pub(crate) mod qa;
pub(crate) mod qa_capability;
pub(crate) mod qualify;
pub(crate) mod quant_preservation;
pub(crate) mod quantize;
pub(crate) mod quantize_flag_parity;
pub(crate) mod react_trace_classifier;
pub(crate) mod react_trace_lint;
pub(crate) mod recipe;
pub(crate) mod registry;
pub(crate) mod registry_quota;
pub(crate) mod registry_quota_lint;
pub(crate) mod registry_schema;
pub(crate) mod rerank;
pub(crate) mod resume_paths;
pub(crate) mod revision;
pub(crate) mod rosetta;
pub(crate) mod run;
#[cfg(feature = "training")]
pub(crate) mod runs;
pub(crate) mod search_merge;
pub(crate) mod serve;
pub(crate) mod serve_plan;
pub(crate) mod serve_plan_output;
pub(crate) mod shard;
pub(crate) mod shared_cache;
pub(crate) mod shared_cache_lint;
pub(crate) mod showcase;
pub(crate) mod sign_artifacts;
pub(crate) mod spdx;
pub(crate) mod stamp;
pub(crate) mod stop_op;
pub(crate) mod tensors;
pub(crate) mod token_redactor;
pub(crate) mod tokenize;
/// #3726: `apr tokenize encode` — a GGUF's own token ids for a text, and the path that made them.
#[cfg(feature = "inference")]
pub(crate) mod tokenize_encode;
pub(crate) mod tokenize_parquet;
pub(crate) mod tool_use_classifier;
pub(crate) mod tool_use_lint;
pub(crate) mod tp_pp_classifier;
pub(crate) mod trace;
#[cfg(feature = "inference")]
pub(crate) mod trace_save_tensor;
#[cfg(feature = "training")]
pub(crate) mod track;
#[cfg(feature = "training")]
pub(crate) mod train;
pub(crate) mod tree;
pub(crate) mod tui;
#[cfg(feature = "training")]
pub(crate) mod tune;
pub(crate) mod typical_p_classifier;
pub(crate) mod typical_p_lint;
pub(crate) mod unified_search_lint;
pub(crate) mod validate;
pub(crate) mod validate_manifest;
pub(crate) mod xet_mode;

/// #4018: at most the first `max` bytes of `s`, cut at a CHAR BOUNDARY, for a log or error line.
///
/// `&s[..s.len().min(max)]` panicked ("byte index N is not a char boundary") whenever byte `max`
/// fell inside a multi-byte UTF-8 char, so `apr run -v` crashed on a non-ASCII prompt instead of
/// answering. The cut floors to the previous boundary: at most 3 bytes short, since a char is at
/// most 4 (the CRUX judge reads a logged prompt of >= max-3 bytes as possibly cut, #3962 B2).
/// `str::floor_char_boundary` would do this, but is not stable at this crate's rust-version.
pub(crate) fn log_head(s: &str, max: usize) -> &str {
    let mut end = s.len().min(max);
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod log_head_4018 {
    use super::log_head;

    /// MUST-RED (#4018): decoded model output, a formatted prompt and a subprocess's stdout/stderr
    /// were each sliced at a fixed BYTE length, which panics when that byte falls inside a
    /// multi-byte char.
    #[test]
    fn non_ascii_text_cut_mid_char_does_not_panic() {
        let t = format!("x{}", "\u{6c34}".repeat(200));
        for max in [200usize, 500] {
            assert!(
                !t.is_char_boundary(max),
                "fixture: byte {max} must be mid-char"
            );
            let head = log_head(&t, max);
            assert!(
                head.len() <= max && head.len() + 3 >= max,
                "{max}: {}",
                head.len()
            );
            assert!(t.starts_with(head));
        }
    }

    #[test]
    fn ascii_and_short_inputs_are_unchanged() {
        assert_eq!(log_head("ok", 200), "ok");
        assert_eq!(log_head(&"a".repeat(600), 500).len(), 500);
    }

    /// #4018: the SITES use it -- the helper alone proves nothing if a caller still byte-slices.
    #[test]
    fn no_log_site_byte_slices_text_any_more() {
        for (f, old) in [
            (
                "chat_generate_session_02.rs",
                "&decoded[..decoded.len().min(200)]",
            ),
            (
                "chat_generate_session_02.rs",
                "&formatted_prompt[..formatted_prompt.len().min(500)]",
            ),
            (
                "inference_result.rs",
                "&stdout_text[..stdout_text.len().min(200)]",
            ),
            (
                "inference_result.rs",
                "&stderr_text[..stderr_text.len().min(200)]",
            ),
        ] {
            let src =
                std::fs::read_to_string(format!("{}/src/commands/{f}", env!("CARGO_MANIFEST_DIR")))
                    .expect("source");
            assert!(
                !src.contains(old),
                "{f} still slices text at a fixed byte length: {old}"
            );
        }
    }
}
