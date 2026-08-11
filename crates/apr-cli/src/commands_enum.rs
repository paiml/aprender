
/// Compute backends `--backend` accepts on `apr run` / `apr chat`.
///
/// The flag used to be a free-form `String`: `--backend banana` printed
/// `Backend override: banana` and then quietly ran the default backend. That is
/// the exact failure the `--backend cuda` guard in `dispatch.rs` exists to
/// prevent — a run whose throughput number was taken through a backend the
/// caller did not ask for — so a typo must be rejected by the parser, not
/// echoed back.
pub const BACKEND_VALUES: [&str; 3] = ["cuda", "cpu", "wgpu"];

/// Trace detail levels `--trace-level` accepts.
///
/// Each value is dispatched on by string equality in `run_entry.rs`; an
/// unrecognised value silently selected "no extra trace output at all" while
/// printing `Trace level: <typo>` as though it had taken effect.
pub const TRACE_LEVEL_VALUES: [&str; 5] = ["none", "basic", "layer", "payload", "chrome"];

/// Output formats `apr run -f/--format` accepts.
pub const RUN_FORMAT_VALUES: [&str; 4] = ["text", "json", "srt", "vtt"];

/// Output format for `apr code` non-interactive mode (PMAT-CODE-OUTPUT-FORMAT-001).
/// Mirrors Claude Code's `claude -p --output-format <fmt>` parity row.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum CodeOutputFormat {
    /// Plain assistant text on stdout (default; existing behavior).
    #[default]
    Text,
    /// Structured JSON envelope: `{type:"result", subtype:"success", result, session_id, duration_ms}`.
    Json,
}

/// Input format for `apr code` non-interactive mode (PMAT-CODE-INPUT-FORMAT-001).
/// `--input-format json` reads `{"role":"user","content":"..."}` from stdin instead
/// of treating stdin as raw prompt text.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum CodeInputFormat {
    /// Raw prompt text from positional args or stdin (default; existing behavior).
    #[default]
    Text,
    /// JSON message envelope on stdin: `{"role":"user","content":"..."}`.
    Json,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run model directly (auto-download, cache, execute)
    Run {
        /// Model source: local path, hf://org/repo, or URL
        #[arg(value_name = "SOURCE")]
        source: String,
        /// Text prompt (positional): `apr run model.gguf "What is 2+2?"`
        #[arg(value_name = "PROMPT")]
        positional_prompt: Option<String>,
        /// Input file (audio, text, etc.)
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Text prompt for generation (for LLM models)
        #[arg(short, long)]
        prompt: Option<String>,
        /// Maximum tokens to generate (default: 32)
        #[arg(short = 'n', long, default_value = "32")]
        max_tokens: usize,
        /// Enable streaming output
        #[arg(long)]
        stream: bool,
        /// Language code (for ASR models)
        #[arg(short, long)]
        language: Option<String>,
        /// Task (transcribe, translate)
        #[arg(short, long)]
        task: Option<String>,
        /// Output format (text, json, srt, vtt)
        #[arg(short = 'f', long, default_value = "text", value_parser = RUN_FORMAT_VALUES)]
        format: String,
        /// Disable GPU acceleration (force CPU-only inference)
        #[arg(long, alias = "cpu", conflicts_with = "gpu")]
        no_gpu: bool,
        /// Force GPU acceleration
        #[arg(long, conflicts_with = "no_gpu")]
        gpu: bool,
        /// Offline mode: block all network access (Sovereign AI compliance)
        #[arg(long)]
        offline: bool,
        /// Benchmark mode: output performance metrics (tok/s, latency)
        #[arg(long)]
        benchmark: bool,
        /// Enable inference tracing (APR-TRACE-001)
        #[arg(long)]
        trace: bool,
        /// Trace specific steps only (comma-separated)
        #[arg(long, value_delimiter = ',')]
        trace_steps: Option<Vec<String>>,
        /// Verbose tracing (show tensor values)
        #[arg(long)]
        trace_verbose: bool,
        /// Save trace output to JSON file
        #[arg(long, value_name = "FILE")]
        trace_output: Option<PathBuf>,
        /// Trace detail level (none, basic, layer, payload, chrome)
        /// "chrome" outputs chrome://tracing JSON integrating layer trace + brick profile.
        /// F-CLIPARITY-01 / PMAT-386 / paiml/aprender#574
        #[arg(long, value_name = "LEVEL", default_value = "basic", value_parser = TRACE_LEVEL_VALUES)]
        trace_level: String,
        /// Shorthand for --trace --trace-level payload (tensor value inspection)
        #[arg(long)]
        trace_payload: bool,
        /// Enable inline Roofline profiling (PMAT-SHOWCASE-METHODOLOGY-001)
        #[arg(long)]
        profile: bool,
        /// Apply chat template for Instruct models (GAP-UX-001)
        ///
        /// Wraps prompt in ChatML format for Qwen2, LLaMA, Mistral Instruct models.
        /// Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        #[arg(long)]
        chat: bool,
        /// Sampling temperature (0.0 = greedy, default: 0.0)
        #[arg(long, default_value = "0.0")]
        temperature: f32,
        /// Top-k sampling (default: 1 = greedy)
        #[arg(long, default_value = "1")]
        top_k: usize,
        /// Top-p nucleus sampling (0.0 = disabled). When set with --top-k, applies top-k first then top-p.
        /// F-CLIPARITY-01 / PMAT-381 / paiml/aprender#569
        #[arg(long)]
        top_p: Option<f32>,
        /// RNG seed for deterministic sampling (default: 299792458, matching Candle)
        /// F-CLIPARITY-01 / PMAT-382 / paiml/aprender#570
        #[arg(long, default_value = "299792458")]
        seed: u64,
        /// Repetition penalty (1.0 = no penalty, >1.0 penalizes repeats)
        /// F-CLIPARITY-01 / PMAT-383 / paiml/aprender#571
        #[arg(long, default_value = "1.0")]
        repeat_penalty: f32,
        /// Context window for repetition penalty (number of recent tokens to check)
        /// F-CLIPARITY-01 / PMAT-384 / paiml/aprender#571
        #[arg(long, default_value = "64")]
        repeat_last_n: usize,
        /// Process prompt tokens one-by-one instead of batched prefill.
        /// Useful for debugging prefill correctness (comparing per-token attention).
        /// F-CLIPARITY-01 / PMAT-385 / paiml/aprender#572
        #[arg(long)]
        split_prompt: bool,
        /// Batch mode: read prompts from JSONL, output results as JSONL.
        /// Model loads once, processes all prompts sequentially.
        /// Each input line: {"prompt": "...", "task_id": "..."}
        /// Chat template is applied automatically.
        #[arg(long, value_name = "FILE")]
        batch_jsonl: Option<PathBuf>,
        /// Show verbose output (model loading, backend info)
        #[arg(short, long)]
        verbose: bool,
        /// PMAT-488: Compute backend override (cuda, cpu, wgpu)
        #[arg(long, value_name = "BACKEND", value_parser = BACKEND_VALUES)]
        backend: Option<String>,
    },
    /// Inference server (plan/run)
    Serve {
        #[command(subcommand)]
        command: ServeCommands,
    },
    /// Inspect model metadata, vocab, and structure
    Inspect {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Show vocabulary details
        #[arg(long)]
        vocab: bool,
        /// Show filter/security details
        #[arg(long)]
        filters: bool,
        /// Show weight statistics
        #[arg(long)]
        weights: bool,
        /// Output as JSON
        #[arg(long)]
        json: bool,
        /// Emit a 0-100 model quality score block.
        ///
        /// Per SPEC-SHIP-TWO-001 §84 P3-A (AC-SHIP2-007 quality
        /// threshold ≥ 90). The score aggregates: physics checks
        /// (no NaN/Inf, no all-zero tensors), structural
        /// completeness (architecture / hidden_size / num_layers
        /// metadata present), provenance (license + data_source +
        /// data_license non-empty), HF identity (hf_architecture
        /// stamped per PMAT-690 P0-K), and tokenizer presence
        /// (has_vocab + embedded merges). A ship-blocking model
        /// MUST score ≥ 90 by this rubric.
        #[arg(long)]
        quality: bool,
    },
    /// Simple debugging output ("drama" mode available)
    Debug {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Theatrical "drama" mode output
        #[arg(long)]
        drama: bool,
        /// Show hex dump
        #[arg(long)]
        hex: bool,
        /// Extract ASCII strings
        #[arg(long)]
        strings: bool,
        /// Limit output lines
        #[arg(long, default_value = "256")]
        limit: usize,
    },
    /// Validate model integrity and quality
    Validate {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Show 100-point quality assessment
        #[arg(long)]
        quality: bool,
        /// Strict validation (fail on warnings)
        #[arg(long)]
        strict: bool,
        /// Minimum score to pass (0-100)
        #[arg(long)]
        min_score: Option<u8>,
    },
    /// Validate a publish manifest (FALSIFY-PM-001..006).
    ///
    /// Contract: `contracts/publish-manifest-v1.yaml`
    /// Spec:     SPEC-SHIP-TWO-001 §12.3 AC-EX-004
    ValidateManifest {
        /// Path to manifest YAML
        #[arg(value_name = "MANIFEST")]
        file: PathBuf,
        /// Optional local .apr artifact to discharge FALSIFY-PM-002 (sha256 match)
        #[arg(long, value_name = "APR_FILE")]
        artifact: Option<PathBuf>,
        /// Discharge FALSIFY-PM-003 via network: HTTP HEAD + streaming sha256.
        /// Default is DEFERRED (offline-safe). Ignored when --offline is set.
        /// Closes F-PUBLISH-EXTRA-001::dogfood_ex05 (no Python in ex-05).
        #[arg(long)]
        live: bool,
    },
    /// Compare two models
    Diff {
        /// First model file
        #[arg(value_name = "FILE1")]
        file1: PathBuf,
        /// Second model file
        #[arg(value_name = "FILE2")]
        file2: PathBuf,
        /// Show weight-level differences
        #[arg(long)]
        weights: bool,
        /// Compare actual tensor values with statistical analysis
        #[arg(long)]
        values: bool,
        /// Filter tensors by name pattern (for --values)
        #[arg(long)]
        filter: Option<String>,
        /// Maximum number of tensors to compare (for --values)
        #[arg(long, default_value = "10")]
        limit: usize,
        /// Account for transpose when comparing (GGUF col-major vs APR row-major)
        #[arg(long)]
        transpose_aware: bool,
        /// Output as JSON
        #[arg(long)]
        json: bool,
        /// CRUX-B-20: per-tensor quant roundtrip error report (RMSE / cosine / max_abs).
        /// FILE1 is the reference (fp16/fp32/bf16); FILE2 is the quantized variant.
        #[arg(long)]
        quant_roundtrip: bool,
        /// CRUX-B-20: cosine threshold for the quant-roundtrip exit-code gate.
        /// Any tensor with cosine < threshold makes the command exit non-zero.
        #[arg(long, default_value = "0.95")]
        threshold: f32,
        /// CRUX-B-20: suppress the threshold exit-code gate (still emits the report).
        #[arg(long)]
        no_threshold: bool,
    },
    /// List tensor names and shapes
    Tensors {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Show tensor statistics (mean, std, min, max)
        #[arg(long)]
        stats: bool,
        /// Filter tensors by name pattern
        #[arg(long)]
        filter: Option<String>,
        /// Limit number of tensors shown (0 = unlimited)
        #[arg(long, default_value = "0")]
        limit: usize,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Layer-by-layer trace analysis
    Trace {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Filter layers by name pattern
        #[arg(long)]
        layer: Option<String>,
        /// Compare with reference model
        #[arg(long)]
        reference: Option<PathBuf>,
        /// Output as JSON
        #[arg(long)]
        json: bool,
        /// Verbose output with per-layer stats
        #[arg(short, long)]
        verbose: bool,
        /// Trace payload through model
        #[arg(long)]
        payload: bool,
        /// Diff mode
        #[arg(long)]
        diff: bool,
        /// Interactive mode
        #[arg(long)]
        interactive: bool,
        /// Save per-stage F32 tensors during trace for SHIP-007 layer-0
        /// element-wise diff. Comma-separated stage names from
        /// `apr-cli-trace-save-tensor-v1.yaml` (e.g.
        /// `embedding,qkv_matmul,attention`). Pass `all` to save every
        /// stage. Output goes to `--save-tensor-dir` if provided,
        /// else `<file_dir>/trace-tensors/<run_id>/`.
        #[arg(long, value_name = "STAGES")]
        save_tensor: Option<String>,
        /// Output directory for `--save-tensor` (default: sibling
        /// `trace-tensors/<run_id>/`).
        #[arg(long, value_name = "DIR")]
        save_tensor_dir: Option<PathBuf>,
        /// Layer-id range for `--save-tensor` (default: 0..1, i.e.
        /// layer 0 only). Format: `START..END` (Rust range syntax,
        /// END exclusive).
        #[arg(long, value_name = "RANGE", default_value = "0..1")]
        save_tensor_layers: String,
    },
    /// Check for best practices and conventions
    Lint {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Fail on warnings as well as errors.
        ///
        /// By default only ERROR-level findings fail the run. Every real model
        /// carries advisory metadata warnings (missing license, model_card,
        /// provenance), so gating on warnings meant `apr lint` could not exit 0
        /// on anything and its exit code told you nothing.
        #[arg(long)]
        strict: bool,
    },
    /// Evaluate a BeatBenchmark contract against a measured value (PMAT-741)
    #[command(name = "beat-run")]
    BeatRun {
        /// Path to a beat-benchmark contract YAML (e.g. contracts/beat-sklearn-iris-v1.yaml)
        #[arg(value_name = "CONTRACT")]
        contract: PathBuf,
        /// Measured metric value; when given, emit a WON/REGRESSED verdict and
        /// exit non-zero on regression. Omit to just report the pinned baseline.
        #[arg(long, value_name = "VALUE")]
        measured: Option<f64>,
    },
    /// Emit a SHA-256 manifest of input files (CRUX-G-05)
    Manifest {
        /// Files to include in the manifest (one entry per file)
        #[arg(value_name = "FILES", num_args = 1..)]
        files: Vec<PathBuf>,
        /// Output JSON manifest path
        #[arg(short, long, value_name = "MAN_JSON")]
        output: PathBuf,
    },
    /// Explain errors, architecture, tensors, and kernel dispatch
    Explain {
        /// Error code, model file path, or family name (auto-detected)
        #[arg(value_name = "CODE_OR_FILE")]
        code_or_file: Option<String>,
        /// Path to .apr model file (optional context for --tensor)
        #[arg(short, long)]
        file: Option<PathBuf>,
        /// Explain a specific tensor
        #[arg(long)]
        tensor: Option<String>,
        /// Explain kernel dispatch pipeline for architecture
        #[arg(long)]
        kernel: bool,
        /// Output as JSON
        #[arg(long)]
        json: bool,
        /// Show kernel contract details and proof obligations
        #[arg(short, long)]
        verbose: bool,
        /// Show per-kernel proof status from contract tests
        #[arg(long)]
        proof_status: bool,
    },
    /// Manage canary tests for regression
    Canary {
        #[command(subcommand)]
        command: CanaryCommands,
    },
    /// Export model to other formats
    Export {
        /// Path to .apr model file
        #[arg(value_name = "FILE", required_unless_present = "list_formats")]
        file: Option<PathBuf>,
        /// Output format (safetensors, gguf, mlx, onnx, openvino, coreml)
        #[arg(long, default_value = "safetensors")]
        format: String,
        /// Output file/directory path
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Apply quantization during export (int8, int4, fp16)
        #[arg(long)]
        quantize: Option<String>,
        /// List all supported export formats
        #[arg(long)]
        list_formats: bool,
        /// Batch export to multiple formats (comma-separated: gguf,mlx,safetensors)
        #[arg(long)]
        batch: Option<String>,
        /// Output in JSON format
        #[arg(long)]
        json: bool,
        /// Plan mode (validate inputs, show export plan, no execution)
        #[arg(long)]
        plan: bool,
        /// #2392: Overwrite an existing output file (refused without it)
        #[arg(short, long)]
        force: bool,
    },
    /// Import from external formats (hf://org/repo, local files, URLs)
    Import {
        /// Source: hf://org/repo, local file, or URL
        #[arg(value_name = "SOURCE")]
        source: String,
        /// Output .apr file path (default: derived from source name)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Model architecture (whisper, llama, bert, qwen2, qwen3, gpt2, starcoder, gpt-neox, opt, phi, gemma, falcon, mamba, t5, auto)
        #[arg(long, default_value = "auto")]
        arch: String,
        /// Quantization (int8, int4, fp16)
        #[arg(long)]
        quantize: Option<String>,
        /// Strict mode: reject unverified architectures and fail on validation errors
        #[arg(long)]
        strict: bool,
        /// Preserve Q4K quantization for fused kernel inference (GGUF only)
        /// Uses realizar's Q4K converter instead of dequantizing to F32
        #[arg(long)]
        preserve_q4k: bool,
        /// PMAT-232: External tokenizer.json for weights-only GGUF files.
        /// Required if the GGUF has no embedded tokenizer vocabulary.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// F-GT-001: Enforce provenance chain. Rejects pre-baked GGUF imports
        /// (only SafeTensors sources allowed). Ensures single-provenance testing.
        #[arg(long)]
        enforce_provenance: bool,
        /// GH-223: Allow import without config.json (default: error).
        /// Without config.json, hyperparameters like rope_theta are inferred from
        /// tensor shapes and may be wrong, producing garbage output.
        #[arg(long)]
        allow_no_config: bool,
    },
    /// Download and cache model OR HuggingFace dataset (Ollama-like UX)
    Pull {
        /// Model reference (alias, hf:// URI, or org/repo) OR "dataset"
        /// asset-type discriminator. When this value is the literal
        /// string "dataset", the next positional `repo` is the
        /// HuggingFace dataset repo and dataset-pull semantics apply.
        #[arg(value_name = "MODEL_OR_ASSET_TYPE")]
        model_ref: String,
        /// Dataset repository (used only when model_ref == "dataset").
        /// Per `apr-cli-pull-dataset-v1.yaml`.
        #[arg(value_name = "REPO")]
        repo: Option<String>,
        /// Force re-download even if cached
        #[arg(long)]
        force: bool,
        /// Verify an already-cached model by re-hashing every file against the
        /// BLAKE3 checksums recorded in `.apr-manifest.json` at download time.
        ///
        /// `apr pull` has always RECORDED those hashes and never checked them:
        /// the only integrity check in the tree compares file SIZE. Size cannot
        /// see a same-length corruption (a 7.1 GB SafeTensors blob was found
        /// with 27 of 339 tensors zeroed, byte-length exactly correct). This
        /// costs O(bytes) by design, which is why it is opt-in. Performs no
        /// network I/O.
        #[arg(long)]
        verify: bool,
        /// CRUX-A-01: resolve short name to canonical URL and exit without
        /// performing any network I/O.
        #[arg(long)]
        dry_run: bool,
        /// CRUX-A-03: pin to a specific branch, tag, or git SHA on the remote
        /// (HuggingFace Hub). Defaults to "main" when omitted.
        #[arg(long, value_name = "REV")]
        revision: Option<String>,
        /// CRUX-A-20: offline mode — forbid any outbound network I/O.
        /// Equivalent to APR_OFFLINE=1 or HF_HUB_OFFLINE=1 in the environment.
        #[arg(long)]
        offline: bool,
        /// (dataset mode) Glob pattern for shard selection. May be passed
        /// multiple times; matches are unioned. fnmatch-compatible
        /// (`*`, `?`, `[a-z]`). No-match is fail-fast.
        #[arg(long, value_name = "GLOB")]
        include: Vec<String>,
        /// (dataset mode) Output directory. Default:
        /// `~/.cache/aprender/datasets/<repo>/`.
        #[arg(short = 'o', long)]
        output: Option<PathBuf>,
    },
    /// Registry operations (CRUX-A-01): inspect alias map, etc.
    Registry {
        #[command(subcommand)]
        command: crate::commands::registry::RegistryCommands,
    },
    /// List cached models
    #[command(name = "list", alias = "ls")]
    List,
    /// Remove model from cache
    #[command(name = "rm", alias = "remove")]
    Rm {
        /// Model reference to remove
        #[arg(value_name = "MODEL")]
        model_ref: String,
    },
    /// Convert/optimize model
    Convert {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Quantize to format (int8, int4, fp16, q4k)
        #[arg(long)]
        quantize: Option<String>,
        /// Compress output (none, zstd, zstd-max, lz4)
        #[arg(long)]
        compress: Option<String>,
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
        /// Force overwrite existing files
        #[arg(short, long)]
        force: bool,
    },
    /// Stamp provenance fields (license, data_source, data_license) onto an existing .apr file
    ///
    /// SHIP-009 full-discharge enabler — patches the three provenance fields on
    /// a pre-built APR v2 artifact (e.g., the shipped MODEL-1 teacher whose
    /// fields are all (missing) because it was built before GATE-APR-PROV-001..003
    /// shipped). Tensor bytes and header flags are preserved verbatim.
    Stamp {
        /// Path to input .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// SPDX license identifier (e.g., Apache-2.0)
        #[arg(long)]
        license: Option<String>,
        /// Training-data source (e.g., huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
        #[arg(long = "data-source")]
        data_source: Option<String>,
        /// SPDX license for data_source (e.g., Apache-2.0)
        #[arg(long = "data-license")]
        data_license: Option<String>,
        /// HuggingFace class name (e.g., Qwen2ForCausalLM, LlamaForCausalLM).
        ///
        /// PMAT-690 P0-K extension (SPEC §86): patch the upstream
        /// `architectures[0]` stamp on a pre-P0-K APR so downstream
        /// consumers (apr inspect --quality, apr pretrain --init,
        /// apr export → llama-cli) see the correct HF identity.
        #[arg(long = "hf-architecture")]
        hf_architecture: Option<String>,
        /// HuggingFace model_type slug (e.g., qwen2, llama).
        ///
        /// PMAT-690 P0-K extension (SPEC §86).
        #[arg(long = "hf-model-type")]
        hf_model_type: Option<String>,
        /// Lowercase architecture family slug (e.g., qwen2, llama).
        ///
        /// PMAT-690 P0-K extension (SPEC §86). This is the field
        /// `apr pretrain --init` reads for arch dispatch — without
        /// patching it, pre-P0-K checkpoints with the P0-H "LlamaForCausalLM"
        /// fallback in this field cannot be loaded as Qwen2 inits.
        #[arg(long)]
        architecture: Option<String>,
        /// Directory containing tokenizer files (vocab.json + merges.txt
        /// OR tokenizer.json). When provided, embeds the vocabulary +
        /// BPE merges into the APR's `custom.tokenizer.vocabulary` /
        /// `custom.tokenizer.merges` JSON metadata AND sets the
        /// HAS_VOCAB header flag.
        ///
        /// PMAT-690 P3-C-prep defect 1 fix (2026-05-17): pre-P0-K APRs
        /// trained from inits without embedded tokenizers fail `apr run`
        /// with PMAT-172. This flag lets the §86 salvage recipe embed
        /// the tokenizer post-hoc so the artifact is self-contained
        /// for inference (the apr binary's headline use case).
        #[arg(long = "tokenizer", value_name = "DIR")]
        tokenizer_dir: Option<PathBuf>,
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
        /// Force overwrite existing files
        #[arg(short, long)]
        force: bool,
    },
    /// Compile model into standalone executable (APR-SPEC §4.16)
    Compile {
        /// Input .apr model file
        #[arg(value_name = "FILE", required_unless_present = "list_targets")]
        file: Option<PathBuf>,
        /// Output binary path (default: derived from model name)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Target triple (e.g., x86_64-unknown-linux-musl)
        #[arg(long)]
        target: Option<String>,
        /// Quantize weights before embedding (int8, int4, fp16)
        #[arg(long)]
        quantize: Option<String>,
        /// Release mode (optimized)
        #[arg(long)]
        release: bool,
        /// Strip debug symbols
        #[arg(long)]
        strip: bool,
        /// Enable LTO (Link-Time Optimization)
        #[arg(long)]
        lto: bool,
        /// List available compilation targets
        #[arg(long)]
        list_targets: bool,
    },
    /// Merge multiple models
    Merge {
        /// Model files to merge
        #[arg(value_name = "FILES", num_args = 2..)]
        files: Vec<PathBuf>,
        /// Merge strategy (average, weighted, slerp, ties, dare)
        #[arg(long, default_value = "average")]
        strategy: String,
        /// Output file path (optional in --plan mode)
        #[arg(short, long, required_unless_present = "plan")]
        output: Option<PathBuf>,
        /// Weights for weighted merge (comma-separated, e.g., "0.7,0.3")
        #[arg(long, value_delimiter = ',')]
        weights: Option<Vec<f32>>,
        /// Base model for TIES/DARE (task vectors computed as delta from base)
        #[arg(long)]
        base_model: Option<PathBuf>,
        /// DARE drop probability (default: 0.9)
        #[arg(long, default_value = "0.9")]
        drop_rate: f32,
        /// TIES trim density threshold (default: 0.2)
        #[arg(long, default_value = "0.2")]
        density: f32,
        /// RNG seed for DARE (default: 42)
        #[arg(long, default_value = "42")]
        seed: u64,
        /// Plan mode (validate inputs, show merge plan, no execution)
        #[arg(long)]
        plan: bool,
        /// #2392: Overwrite an existing output file (refused without it)
        #[arg(short, long)]
        force: bool,
    },
    /// Quantize model weights (GH-243)
    Quantize {
        /// Input model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Quantization scheme: int8, int4, fp16, q4k
        #[arg(long, short = 's', default_value = "int4")]
        scheme: String,
        /// Output file path (required unless --plan)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Output format override (apr, gguf, safetensors)
        #[arg(long)]
        format: Option<String>,
        /// Batch quantization (comma-separated schemes)
        #[arg(long)]
        batch: Option<String>,
        /// Plan mode (estimate only, no execution)
        #[arg(long)]
        plan: bool,
        /// Force overwrite existing files
        #[arg(short, long)]
        force: bool,
    },
    /// Model optimization commands (fine-tune, prune, distill)
    #[command(flatten)]
    ModelOps(ModelOpsCommands),
    /// Start the MCP (Model Context Protocol) server over stdio
    ///
    /// Exposes `apr` as MCP tools for Claude Code, Cursor, Cline, and other
    /// MCP clients. Configure via `.mcp.json` with `{"command":"apr","args":["mcp"]}`.
    Mcp {},
    /// Interactive terminal UI
    Tui {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
    },
    /// Model self-test: 10-stage pipeline integrity check (APR-TRACE-001)
    Check {
        /// Path to model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Disable GPU acceleration
        #[arg(long)]
        no_gpu: bool,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// GPU status and VRAM reservation management (GPU-SHARE-001)
    #[cfg(feature = "training")]
    Gpu {
        /// Show reservations as JSON
        #[arg(long)]
        json: bool,
    },
    /// Sovereign AI coding assistant — all inference local via realizar (PMAT-182)
    
    Code {
        /// Path to local GGUF/APR model file (prefers .apr format)
        #[arg(long)]
        model: Option<PathBuf>,

        /// Project directory (loads APR.md/CLAUDE.md from this path)
        #[arg(long, default_value = ".")]
        project: PathBuf,

        /// Resume previous session (optionally by ID)
        #[arg(long)]
        resume: Option<Option<String>>,

        /// Agent manifest (advanced — overrides defaults)
        #[arg(long)]
        manifest: Option<PathBuf>,

        /// Initial prompt (non-interactive: print response and exit)
        #[arg(short, long)]
        print: bool,

        /// Prompt text (positional, for -p mode)
        #[arg(trailing_var_arg = true)]
        prompt: Vec<String>,

        /// Max turns before stopping
        #[arg(long, default_value = "50")]
        max_turns: u32,

        /// Emit a `ccpa-trace.jsonl` describing the run to this path.
        /// Format mirrors the schema at
        /// <https://github.com/paiml/claude-code-parity-apr/blob/main/contracts/claude-code-parity-apr-v1.yaml>
        /// (`§ trace_schema`). Used by `ccpa measure` to score apr-code
        /// against canonical Claude Code reference fixtures.
        #[arg(long)]
        emit_trace: Option<PathBuf>,

        /// Output format for non-interactive (`-p`) mode (PMAT-CODE-OUTPUT-FORMAT-001).
        /// `text` (default): plain assistant text.
        /// `json`: structured `{type:"result", subtype:"success", result, session_id, duration_ms}`
        /// envelope matching Claude Code's `claude -p --output-format json` shape.
        #[arg(long, value_enum, default_value_t = CodeOutputFormat::Text)]
        output_format: CodeOutputFormat,

        /// Input format for non-interactive stdin (PMAT-CODE-INPUT-FORMAT-001).
        /// `text` (default): treat stdin as raw prompt text.
        /// `json`: parse `{"role":"user","content":"..."}` from stdin and use `content`
        /// as the prompt. Matches Claude Code's `claude -p --input-format json` shape.
        #[arg(long, value_enum, default_value_t = CodeInputFormat::Text)]
        input_format: CodeInputFormat,
    },
    /// Extended analysis, profiling, QA, and visualization commands
    #[command(flatten)]
    Extended(ExtendedCommands),

    /// Monorepo management (publish, shims, audit, archive) [dev-only]
    #[cfg(feature = "dev")]
    #[command(subcommand)]
    Mono(crate::commands::mono::MonoCommands),
}
