// ============================================================================
// Server Commands (extracted from main.rs for testability)
// WAPR-PERF-004: Gated behind "server" feature since depends on crate::api
// ============================================================================
#[cfg(feature = "server")]
mod server_commands {
    use super::Result;

    /// Result of preparing server state (returned by `prepare_serve_state`)
    pub struct PreparedServer {
        /// The prepared AppState for the server
        pub state: crate::api::AppState,
        /// Whether batch mode is enabled
        pub batch_mode_enabled: bool,
        /// Model type that was loaded
        pub model_type: ModelType,
    }

    impl std::fmt::Debug for PreparedServer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("PreparedServer")
                .field("batch_mode_enabled", &self.batch_mode_enabled)
                .field("model_type", &self.model_type)
                .finish_non_exhaustive()
        }
    }

    /// Type of model being served
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ModelType {
        /// GGUF quantized model
        Gguf,
        /// SafeTensors model
        SafeTensors,
        /// APR format model
        Apr,
    }

    impl std::fmt::Display for ModelType {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ModelType::Gguf => write!(f, "GGUF"),
                ModelType::SafeTensors => write!(f, "SafeTensors"),
                ModelType::Apr => write!(f, "APR"),
            }
        }
    }

    /// Prepare server state by loading a model (GGUF/SafeTensors/APR)
    ///
    /// This function is extracted from `serve_model` for testability.
    /// It handles model loading and AppState creation without starting the server.
    ///
    /// # Arguments
    /// * `model_path` - Path to model file (.gguf, .safetensors, or .apr)
    /// * `batch_mode` - Enable batch processing (requires 'gpu' feature)
    /// * `force_gpu` - Force CUDA backend (requires 'cuda' feature)
    ///
    /// # Returns
    /// A `PreparedServer` containing the AppState and configuration
    /// Load and prepare a GGUF model for serving (CUDA/batch/CPU paths)
    fn prepare_gguf_serve_state(
        model_path: &str,
        batch_mode: bool,
        force_gpu: bool,
    ) -> Result<PreparedServer> {
        use crate::gguf::MappedGGUFModel;

        println!("Parsing GGUF file...");
        let mapped = MappedGGUFModel::from_path(model_path).map_err(|e| {
            crate::error::RealizarError::UnsupportedOperation {
                operation: "load_gguf".to_string(),
                reason: format!("Failed to load GGUF: {e}"),
            }
        })?;

        println!("Successfully loaded GGUF model");
        println!("  Tensors: {}", mapped.model.tensors.len());
        println!("  Metadata: {} entries", mapped.model.metadata.len());
        println!();

        // IMP-100: Use OwnedQuantizedModel with fused Q4_K ops (1.37x faster for single-token)
        println!("Creating quantized model (fused Q4_K ops)...");
        let quantized_model =
            crate::gguf::OwnedQuantizedModel::from_mapped(&mapped).map_err(|e| {
                crate::error::RealizarError::UnsupportedOperation {
                    operation: "create_quantized".to_string(),
                    reason: format!("Failed to create quantized model: {e}"),
                }
            })?;

        println!("Quantized model created successfully!");
        println!("  Vocab size: {}", quantized_model.config.vocab_size);
        println!("  Hidden dim: {}", quantized_model.config.hidden_dim);
        println!("  Layers: {}", quantized_model.layers.len());

        // Extract vocabulary from GGUF for proper token decoding
        let vocab = mapped.model.vocabulary().unwrap_or_else(|| {
            eprintln!("  Warning: No vocabulary in GGUF, using placeholder tokens");
            (0..quantized_model.config.vocab_size)
                .map(|i| format!("token{i}"))
                .collect()
        });
        println!("  Vocab loaded: {} tokens", vocab.len());
        println!();

        // PARITY-113: Enable CUDA backend via --gpu flag or REALIZAR_BACKEND environment variable
        #[cfg(feature = "cuda")]
        let use_cuda = force_gpu
            || std::env::var("REALIZAR_BACKEND")
                .map(|v| v.eq_ignore_ascii_case("cuda"))
                .unwrap_or(false);

        #[cfg(not(feature = "cuda"))]
        let use_cuda = false;

        #[cfg(not(feature = "cuda"))]
        if force_gpu {
            eprintln!("Warning: --gpu flag requires 'cuda' feature. Falling back to CPU.");
            eprintln!("Build with: cargo build --features cuda");
            eprintln!();
        }

        let state = if use_cuda && !batch_mode {
            prepare_gguf_cuda_state(quantized_model, vocab, force_gpu)?
        } else if batch_mode {
            prepare_gguf_batch_state(quantized_model, vocab)?
        } else {
            // CPU mode: Use quantized model for serving (fused CPU ops are faster for m=1)
            crate::api::AppState::with_quantized_model_and_vocab(quantized_model, vocab)?
        };

        Ok(PreparedServer {
            state,
            batch_mode_enabled: batch_mode,
            model_type: ModelType::Gguf,
        })
    }

    /// Create CUDA-backed AppState for GGUF serving (PAR-112-FIX)
    fn prepare_gguf_cuda_state(
        quantized_model: crate::gguf::OwnedQuantizedModel,
        vocab: Vec<String>,
        force_gpu: bool,
    ) -> Result<crate::api::AppState> {
        #[cfg(feature = "cuda")]
        {
            use crate::gguf::OwnedQuantizedModelCuda;

            let source = if force_gpu {
                "--gpu flag"
            } else {
                "REALIZAR_BACKEND=cuda"
            };
            println!("Creating CUDA model ({source})...");

            // GH-286: Configurable context length (was hardcoded 4096)
            let max_seq_len = std::env::var("REALIZR_CONTEXT_LENGTH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4096);
            let cuda_model =
                OwnedQuantizedModelCuda::with_max_seq_len(quantized_model, 0, max_seq_len)
                    .map_err(|e| e.error)?;

            println!("  CUDA model created on GPU: {}", cuda_model.device_name());
            println!("  Max sequence length: {} (--context-length)", max_seq_len);
            println!("  TRUE STREAMING: enabled (PAR-112)");

            let state = crate::api::AppState::with_cuda_model_and_vocab(cuda_model, vocab)?;

            // PMAT-088: Choose scheduler based on env var
            let cuda_model_arc = state.cuda_model().expect("just created").clone();
            let use_iteration_scheduler =
                std::env::var("ITERATION_SCHEDULER").as_deref() == Ok("1");

            let batch_tx = if use_iteration_scheduler {
                let iter_config =
                    crate::api::iteration_scheduler::IterationSchedulerConfig::default();
                println!(
                    "  ITERATION SCHEDULER: max_slots={}, prefill_chunk={} (PMAT-088)",
                    iter_config.max_slots, iter_config.prefill_chunk_size,
                );
                crate::api::iteration_scheduler::spawn_iteration_scheduler(
                    cuda_model_arc,
                    iter_config,
                )
            } else {
                let batch_config = crate::api::cuda_batch_scheduler::CudaBatchConfig::default();
                println!(
                    "  CONTINUOUS BATCHING: max_batch={}, window={}ms (PMAT-044)",
                    batch_config.max_batch, batch_config.window_ms
                );
                crate::api::cuda_batch_scheduler::spawn_cuda_batch_scheduler(
                    cuda_model_arc,
                    batch_config,
                )
            };
            println!();

            Ok(state.with_cuda_batch_tx(batch_tx))
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = force_gpu;
            crate::api::AppState::with_quantized_model_and_vocab(quantized_model, vocab)
        }
    }

    /// Create batch-processing AppState for GGUF serving (PARITY-093/094)
    fn prepare_gguf_batch_state(
        quantized_model: crate::gguf::OwnedQuantizedModel,
        vocab: Vec<String>,
    ) -> Result<crate::api::AppState> {
        #[cfg(feature = "gpu")]
        {
            use crate::gguf::OwnedQuantizedModelCachedSync;

            println!("Initializing batch inference mode (PARITY-093/094)...");

            let cached_model = OwnedQuantizedModelCachedSync::new(quantized_model);

            println!("  Warming up GPU cache (dequantizing FFN weights)...");
            match cached_model.warmup_gpu_cache() {
                Ok((memory_bytes, num_layers)) => {
                    println!(
                        "  GPU cache ready: {:.2} GB ({} layers)",
                        memory_bytes as f64 / 1e9,
                        num_layers
                    );
                },
                Err(e) => {
                    eprintln!(
                        "  Warning: GPU cache warmup failed: {}. Falling back to CPU batch.",
                        e
                    );
                },
            }

            let state = crate::api::AppState::with_cached_model_and_vocab(cached_model, vocab)?;

            let cached_model_arc = state
                .cached_model()
                .expect("cached_model should exist")
                .clone();

            let batch_config = crate::api::BatchConfig::default();
            println!("  Batch window: {}ms", batch_config.window_ms);
            println!("  Min batch size: {}", batch_config.min_batch);
            println!("  Optimal batch: {}", batch_config.optimal_batch);
            println!("  Max batch size: {}", batch_config.max_batch);
            println!(
                "  GPU threshold: {} (GPU GEMM for batch >= this)",
                batch_config.gpu_threshold
            );

            let batch_tx =
                crate::api::spawn_batch_processor(cached_model_arc, batch_config.clone());

            println!("  Batch processor: RUNNING");
            println!();

            Ok(state.with_batch_config(batch_tx, batch_config))
        }

        #[cfg(not(feature = "gpu"))]
        {
            eprintln!(
                "Warning: --batch requires 'gpu' feature. Falling back to single-request mode."
            );
            crate::api::AppState::with_quantized_model_and_vocab(quantized_model, vocab)
        }
    }

    /// Load and prepare a SafeTensors model for serving
    fn prepare_safetensors_serve_state(model_path: &str, force_gpu: bool) -> Result<PreparedServer> {
        use std::path::Path;

        println!("Loading SafeTensors model for serving...");
        let model_path_obj = Path::new(model_path);

        // #169: GPU path via SafeTensorsCudaModel (format parity with GGUF)
        #[cfg(feature = "cuda")]
        if force_gpu {
            use crate::safetensors_cuda::SafeTensorsCudaModel;

            println!("  Mode: GPU (SafeTensors CUDA — #169 format parity)");
            let mut cuda_model = SafeTensorsCudaModel::load(model_path_obj, 0).map_err(|e| {
                crate::error::RealizarError::UnsupportedOperation {
                    operation: "safetensors_cuda_load".to_string(),
                    reason: format!("Failed to load SafeTensors for GPU: {e}"),
                }
            })?;

            let vocab_size = cuda_model.config().vocab_size;
            #[allow(clippy::map_unwrap_or)]
            let vocab = crate::apr::AprV2Model::load_tokenizer_from_sibling(model_path_obj)
                .map(|(v, _, _)| v)
                .unwrap_or_else(|| {
                    println!("  Warning: No tokenizer.json found, using simple vocabulary");
                    (0..vocab_size).map(|i| format!("token{i}")).collect()
                });

            println!("  Vocab size: {}", vocab.len());

            let state = crate::api::AppState::with_safetensors_cuda_model_and_vocab(
                cuda_model, vocab,
            )?;

            return Ok(PreparedServer {
                state,
                batch_mode_enabled: false,
                model_type: ModelType::SafeTensors,
            });
        }

        #[cfg(not(feature = "cuda"))]
        if force_gpu {
            eprintln!("Warning: --gpu flag requires 'cuda' feature for SafeTensors. Falling back to CPU.");
        }

        // CPU fallback path
        use crate::safetensors_infer::SafetensorsToAprConverter;

        let transformer = SafetensorsToAprConverter::convert(model_path_obj).map_err(|e| {
            crate::error::RealizarError::UnsupportedOperation {
                operation: "convert_safetensors".to_string(),
                reason: format!("Failed to convert SafeTensors: {e}"),
            }
        })?;

        println!("  Architecture: {}", transformer.config.architecture);
        println!("  Layers: {}", transformer.config.num_layers);
        println!("  Hidden: {}", transformer.config.hidden_dim);

        #[allow(clippy::map_unwrap_or)]
        let vocab = crate::apr::AprV2Model::load_tokenizer_from_sibling(model_path_obj)
            .map(|(v, _, _)| v)
            .unwrap_or_else(|| {
                println!("  Warning: No tokenizer.json found, using simple vocabulary");
                (0..transformer.config.vocab_size)
                    .map(|i| format!("token{i}"))
                    .collect()
            });

        println!("  Vocab size: {}", vocab.len());
        println!("  Mode: CPU (F32 inference)");

        let state =
            crate::api::AppState::with_apr_transformer_and_vocab(transformer.into_inner(), vocab)?;

        Ok(PreparedServer {
            state,
            batch_mode_enabled: false,
            model_type: ModelType::SafeTensors,
        })
    }

    /// Load and prepare an APR model for serving
    fn prepare_apr_serve_state(model_path: &str, force_gpu: bool) -> Result<PreparedServer> {
        use std::path::Path;

        println!("Loading APR model for serving...");

        // #170: Route Q4K APR through GGUF CUDA serve path (OwnedQuantizedModelCuda).
        // The Q4K-specific apr_q4k_scheduler produces garbage — disabled.
        #[cfg(feature = "cuda")]
        if force_gpu
            || std::env::var("REALIZAR_BACKEND")
                .map(|v| v.eq_ignore_ascii_case("cuda"))
                .unwrap_or(false)
        {
            return prepare_gguf_cuda_state_from_apr(model_path);
        }

        // F32 CPU path (original)
        use crate::apr_transformer::AprTransformer;

        let file_data = std::fs::read(model_path).map_err(|e| {
            crate::error::RealizarError::UnsupportedOperation {
                operation: "read_model_file".to_string(),
                reason: format!("Failed to read {model_path}: {e}"),
            }
        })?;

        let transformer = AprTransformer::from_apr_bytes(&file_data).map_err(|e| {
            crate::error::RealizarError::UnsupportedOperation {
                operation: "load_apr".to_string(),
                reason: format!("Failed to load APR: {e}"),
            }
        })?;

        println!("  Architecture: {}", transformer.config.architecture);
        println!("  Layers: {}", transformer.config.num_layers);
        println!("  Hidden: {}", transformer.config.hidden_dim);

        let model_path_obj = Path::new(model_path);
        let vocab = crate::apr::AprV2Model::load_tokenizer_from_sibling(model_path_obj)
            .map(|(v, _, _)| v)
            .or_else(|| {
                crate::apr::AprV2Model::load(model_path_obj)
                    .ok()
                    .and_then(|m| m.load_embedded_tokenizer())
                    .map(|t| t.id_to_token.clone())
            })
            .unwrap_or_else(|| {
                println!("  Warning: No vocabulary found, using simple vocabulary");
                (0..transformer.config.vocab_size)
                    .map(|i| format!("token{i}"))
                    .collect()
            });

        println!("  Vocab size: {}", vocab.len());
        println!("  Mode: CPU (F32 inference)");

        let state = crate::api::AppState::with_apr_transformer_and_vocab(transformer, vocab)?;

        Ok(PreparedServer {
            state,
            batch_mode_enabled: false,
            model_type: ModelType::Apr,
        })
    }

    /// ALB-095: Load APR Q4K model and spawn GPU inference thread for serving
    /// #170: Load APR model through GGUF CUDA path (OwnedQuantizedModel::from_apr)
    #[cfg(feature = "cuda")]
    fn prepare_gguf_cuda_state_from_apr(model_path: &str) -> Result<PreparedServer> {
        use crate::apr::MappedAprModel;
        use crate::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};
        use std::path::Path;

        let path = Path::new(model_path);
        println!("  Mode: GPU (APR → GGUF CUDA — #170 fix)");

        let mapped = MappedAprModel::from_path(path).map_err(|e| {
            crate::error::RealizarError::FormatError { reason: format!("APR load: {e}") }
        })?;
        let model = OwnedQuantizedModel::from_apr(&mapped).map_err(|e| {
            crate::error::RealizarError::FormatError { reason: format!("APR→GGUF: {e}") }
        })?;

        println!("  Layers: {}, Hidden: {}, Vocab: {}",
            model.config.num_layers, model.config.hidden_dim, model.config.vocab_size);

        // GH-286: Configurable context length (was hardcoded 4096)
        let max_seq_len = std::env::var("REALIZR_CONTEXT_LENGTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        let cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(model, 0, max_seq_len)
            .map_err(|e| e.error)?;

        println!("  CUDA model on GPU: {}", cuda_model.device_name());

        // Load vocab from embedded APR metadata
        let apr_v2 = crate::apr::AprV2Model::load(path).ok();
        let vocab = apr_v2.as_ref()
            .and_then(|m| m.metadata().get_embedded_vocabulary())
            .or_else(|| crate::apr::AprV2Model::load_tokenizer_from_sibling(path).map(|(v, _, _)| v))
            .unwrap_or_else(|| {
                println!("  Warning: No tokenizer found");
                (0..cuda_model.model().config.vocab_size).map(|i| format!("token{i}")).collect()
            });

        println!("  Vocab size: {}", vocab.len());

        let state = crate::api::AppState::with_cuda_model_and_vocab(cuda_model, vocab)?;

        Ok(PreparedServer {
            state,
            batch_mode_enabled: false,
            model_type: ModelType::Apr,
        })
    }

    #[cfg(feature = "cuda")]
    fn prepare_apr_q4k_serve_state(model_path: &str) -> Result<PreparedServer> {
        use crate::api::apr_q4k_scheduler;
        use crate::apr::AprV2Model;
        use std::path::Path;

        println!("  Mode: GPU (Q4K CUDA — ALB-095)");

        // Load tokenizer for the HTTP serving path
        // ALB-109: Capture EOS token ID — Qwen3 uses 151643, not 0/2
        let model_path_obj = Path::new(model_path);
        let (vocab, eos_id) = AprV2Model::load_tokenizer_from_sibling(model_path_obj)
            .map(|(v, _, eos)| (v, eos))
            .or_else(|| {
                AprV2Model::load(model_path_obj)
                    .ok()
                    .and_then(|m| m.load_embedded_tokenizer())
                    .map(|t| (t.id_to_token.clone(), None))
            })
            .unwrap_or_else(|| {
                println!("  Warning: No vocabulary found, using simple vocabulary");
                ((0..151936).map(|i| format!("token{i}")).collect(), None)
            });

        println!("  Vocab: {} tokens", vocab.len());
        if let Some(eos) = eos_id {
            println!("  EOS token ID: {eos}");
        }

        // Spawn the Q4K inference thread (loads model, uploads weights to GPU)
        let q4k_tx =
            apr_q4k_scheduler::spawn_apr_q4k_inference_thread(model_path).map_err(|e| {
                crate::error::RealizarError::GpuError {
                    reason: format!("Q4K inference thread failed: {e}"),
                }
            })?;

        let state = crate::api::AppState::with_apr_q4k_and_vocab_eos(q4k_tx, vocab, eos_id)?;

        Ok(PreparedServer {
            state,
            batch_mode_enabled: false,
            model_type: ModelType::Apr,
        })
    }

    /// Prepare server state by loading a model (GGUF/SafeTensors/APR)
    ///
    /// Dispatches to format-specific loaders based on file extension.
    pub fn prepare_serve_state(
        model_path: &str,
        batch_mode: bool,
        force_gpu: bool,
    ) -> Result<PreparedServer> {
        println!("Loading model from: {model_path}");
        if batch_mode {
            println!("Mode: BATCH (PARITY-093 M4 parity)");
        } else {
            println!("Mode: SINGLE-REQUEST");
        }
        if force_gpu {
            println!("GPU: FORCED (--gpu flag)");
        }
        println!();

        if model_path.ends_with(".gguf") {
            prepare_gguf_serve_state(model_path, batch_mode, force_gpu)
        } else if model_path.ends_with(".safetensors") {
            prepare_safetensors_serve_state(model_path, force_gpu)
        } else if model_path.ends_with(".apr") {
            prepare_apr_serve_state(model_path, force_gpu)
        } else {
            Err(crate::error::RealizarError::UnsupportedOperation {
                operation: "detect_model_type".to_string(),
                reason: "Unsupported file extension. Expected .gguf, .safetensors, or .apr"
                    .to_string(),
            })
        }
    }

    /// Serve a GGUF/SafeTensors/APR model via HTTP API
    ///
    /// This function was extracted from main.rs (PAR-112-FIX) to enable:
    /// 1. Unit testing of server initialization logic
    /// 2. Coverage measurement (main.rs was at 3.66%)
    /// 3. Reuse from other entry points
    ///
    /// # Arguments
    /// * `host` - Host to bind to (e.g., "0.0.0.0")
    /// * `port` - Port to listen on
    /// * `model_path` - Path to model file (.gguf, .safetensors, or .apr)
    /// * `batch_mode` - Enable batch processing (requires 'gpu' feature)
    /// * `force_gpu` - Force CUDA backend (requires 'cuda' feature)
    /// * `openai_api` - Enable OpenAI-compatible API at /v1/* (GH-148)
    /// * `trace` - GH-103: Enable inference tracing (propagates into QuantizedGenerateConfig.trace)
    /// * `context_length` - GH-286: Max sequence length for KV cache (default 4096)
    /// * `no_fp8_cache` - GH-286: Skip FP8 weight cache warmup (saves ~1.5 GB RSS)
    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    pub async fn serve_model(
        host: &str,
        port: u16,
        model_path: &str,
        batch_mode: bool,
        force_gpu: bool,
        openai_api: bool,
        trace: bool,
        context_length: usize,
        no_fp8_cache: bool,
    ) -> Result<()> {
        // GH-286: Set context length and FP8 cache flags for downstream use
        if no_fp8_cache {
            std::env::set_var("REALIZR_NO_FP8_CACHE", "1");
            eprintln!("[GH-286] FP8 weight cache DISABLED (--no-fp8-cache). Saves ~1.5 GB RSS.");
        }
        // Store context_length for prepare_serve_state to pick up
        std::env::set_var("REALIZR_CONTEXT_LENGTH", context_length.to_string());

        // Prepare server state (testable)
        let prepared = prepare_serve_state(model_path, batch_mode, force_gpu)?;

        // GH-103: Wire trace flag into AppState
        let state = if trace {
            eprintln!("Tracing: ENABLED (--trace flag)");
            prepared.state.with_trace(true)
        } else {
            prepared.state
        };

        // GH-148: Create router with OpenAI API flag
        let router_config = crate::api::RouterConfig { openai_api };
        let app = crate::api::create_router_with_config(state, router_config);

        // Parse and validate address
        let addr: std::net::SocketAddr = format!("{host}:{port}").parse().map_err(|e| {
            crate::error::RealizarError::InvalidShape {
                reason: format!("Invalid address: {e}"),
            }
        })?;

        // GH-148: Server info to stderr to avoid stdout contamination
        eprintln!("Server listening on http://{addr}");
        eprintln!();
        eprintln!("Endpoints:");
        eprintln!("  GET  /health         - Health check");
        if openai_api {
            eprintln!("  POST /v1/completions - OpenAI-compatible completions");
        }
        if prepared.batch_mode_enabled && openai_api {
            eprintln!("  POST /v1/batch/completions - GPU batch completions (PARITY-022)");
            eprintln!("  POST /v1/gpu/warmup  - Warmup GPU cache");
            eprintln!("  GET  /v1/gpu/status  - GPU status");
        }
        eprintln!("  POST /generate       - Generate text (Q4_K fused)");
        if !openai_api {
            eprintln!("  (OpenAI-compatible /v1/* routes disabled)");
        }
        eprintln!();

        if prepared.batch_mode_enabled {
            eprintln!("M4 Parity Target: 192 tok/s at concurrency >= 4");
            eprintln!("Benchmark with: wrk -t4 -c4 -d30s http://{addr}/v1/completions");
            eprintln!();
        }

        // Bind and serve
        let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
            crate::error::RealizarError::UnsupportedOperation {
                operation: "bind".to_string(),
                reason: format!("Failed to bind: {e}"),
            }
        })?;

        axum::serve(listener, app).await.map_err(|e| {
            crate::error::RealizarError::UnsupportedOperation {
                operation: "serve".to_string(),
                reason: format!("Server error: {e}"),
            }
        })?;

        Ok(())
    }

    /// Start a demo inference server (no model required)
    ///
    /// This is useful for testing the API without loading a real model.
    /// * `openai_api` - Enable OpenAI-compatible API at /v1/* (GH-148)
    pub async fn serve_demo(host: &str, port: u16, openai_api: bool) -> Result<()> {
        use std::net::SocketAddr;

        // GH-148: Server info to stderr to avoid stdout contamination
        eprintln!("Starting Realizar inference server (demo mode)...");

        let state = crate::api::AppState::demo()?;
        let router_config = crate::api::RouterConfig { openai_api };
        let app = crate::api::create_router_with_config(state, router_config);

        let addr: SocketAddr = format!("{host}:{port}").parse().map_err(|e| {
            crate::error::RealizarError::InvalidShape {
                reason: format!("Invalid address: {e}"),
            }
        })?;

        eprintln!("Server listening on http://{addr}");
        eprintln!();
        eprintln!("Endpoints:");
        eprintln!("  GET  /health   - Health check");
        eprintln!("  POST /tokenize - Tokenize text");
        eprintln!("  POST /generate - Generate text");
        if !openai_api {
            eprintln!("  (OpenAI-compatible /v1/* routes disabled)");
        }
        eprintln!();
        eprintln!("Example:");
        eprintln!("  curl http://{addr}/health");
        eprintln!();

        let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
            crate::error::RealizarError::InvalidShape {
                reason: format!("Failed to bind: {e}"),
            }
        })?;

        axum::serve(listener, app).await.map_err(|e| {
            crate::error::RealizarError::InvalidShape {
                reason: format!("Server error: {e}"),
            }
        })?;

        Ok(())
    }
} // mod server_commands

#[cfg(feature = "server")]
pub use server_commands::*;
