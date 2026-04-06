//! Component registration for the Sovereign AI Stack.

use super::super::types::*;
use super::types::KnowledgeGraph;

impl KnowledgeGraph {
    /// Register all Sovereign AI Stack components
    pub(crate) fn register_sovereign_stack(&mut self) {
        // Layer 0: Compute Primitives
        self.register_trueno();
        self.register_trueno_db();
        self.register_trueno_graph();
        self.register_trueno_viz();
        self.register_trueno_rag();
        self.register_trueno_zram();
        self.register_trueno_ublk();
        self.register_pepita();

        // Layer 1: ML Algorithms
        self.register_aprender();

        // Layer 2: Training & Inference
        self.register_entrenar();
        self.register_realizar();
        self.register_whisper_apr();

        // Layer 3: Transpilers
        self.register_depyler();
        self.register_decy();
        self.register_bashrs();
        self.register_ruchy();

        // Layer 4: Orchestration
        self.register_batuta();
        self.register_repartir();
        self.register_pforge();
        self.register_forjar();

        // Layer 5: Quality
        self.register_certeza();
        self.register_pmat();
        self.register_renacer();
        self.register_apr_qa();
        self.register_provable_contracts();
        self.register_tiny_model_ground_truth();

        // Layer 6: Data & MLOps
        self.register_alimentar();
        self.register_pacha();

        // Layer 7: Media Production
        self.register_rmedia();

        // Simulation & Testing
        self.register_simular();
        self.register_probar();
    }

    fn register_trueno(&mut self) {
        let component = StackComponent::new(
            "trueno",
            "0.11.0",
            StackLayer::Primitives,
            "SIMD-accelerated tensor operations with GPU support and LZ4 compression",
        )
        .with_capabilities(vec![
            Capability::new("vector_ops", CapabilityCategory::Compute)
                .with_description("SIMD-accelerated vector operations"),
            Capability::new("matrix_ops", CapabilityCategory::Compute)
                .with_description("High-performance matrix multiplication"),
            Capability::new("simd", CapabilityCategory::Compute)
                .with_description("SIMD auto-vectorization (AVX2/AVX-512/NEON)"),
            Capability::new("gpu", CapabilityCategory::Compute)
                .with_description("GPU acceleration via wgpu (fixed PTX codegen)"),
            Capability::new("lz4_compression", CapabilityCategory::Compute)
                .with_description("LZ4 tensor compression for memory efficiency"),
        ]);
        self.register_component(component);
    }

    fn register_trueno_db(&mut self) {
        let component = StackComponent::new(
            "trueno-db",
            "0.3.3",
            StackLayer::Primitives,
            "GPU-first embedded analytics database with SIMD fallback",
        )
        .with_capabilities(vec![
            Capability::new("vector_store", CapabilityCategory::Storage)
                .with_description("Efficient vector storage"),
            Capability::new("similarity_search", CapabilityCategory::Storage)
                .with_description("Fast similarity search with HNSW"),
            Capability::new("persistence", CapabilityCategory::Storage)
                .with_description("Durable data persistence"),
            Capability::new("knn_search", CapabilityCategory::Storage)
                .with_description("K-nearest neighbor queries"),
        ]);
        self.register_component(component);
    }

    fn register_trueno_graph(&mut self) {
        let component = StackComponent::new(
            "trueno-graph",
            "0.1.1",
            StackLayer::Primitives,
            "GPU-first embedded graph database for code analysis",
        )
        .with_capabilities(vec![
            Capability::new("pathfinding", CapabilityCategory::Compute)
                .with_description("Dijkstra and A* pathfinding"),
            Capability::new("centrality", CapabilityCategory::Compute)
                .with_description("PageRank and betweenness centrality"),
            Capability::new("community_detection", CapabilityCategory::Compute)
                .with_description("Label propagation clustering"),
        ]);
        self.register_component(component);
    }

    fn register_trueno_viz(&mut self) {
        let component = StackComponent::new(
            "trueno-viz",
            "0.1.1",
            StackLayer::Primitives,
            "SIMD/GPU/WASM-accelerated visualization",
        )
        .with_capabilities(vec![Capability::new(
            "visualization",
            CapabilityCategory::Compute,
        )
        .with_description("High-performance data visualization")]);
        self.register_component(component);
    }

    fn register_trueno_rag(&mut self) {
        let component = StackComponent::new(
            "trueno-rag",
            "0.1.0",
            StackLayer::Primitives,
            "RAG pipeline with chunking, retrieval, and reranking",
        )
        .with_capabilities(vec![
            Capability::new("chunking", CapabilityCategory::Storage)
                .with_description("Text chunking strategies (fixed, semantic, recursive)"),
            Capability::new("dense_retrieval", CapabilityCategory::Storage)
                .with_description("Vector similarity search via trueno-db"),
            Capability::new("sparse_retrieval", CapabilityCategory::Storage)
                .with_description("BM25 keyword search"),
            Capability::new("hybrid_retrieval", CapabilityCategory::Storage)
                .with_description("Combined dense + sparse retrieval"),
            Capability::new("reranking", CapabilityCategory::MachineLearning)
                .with_description("Cross-encoder reranking"),
            Capability::new("context_assembly", CapabilityCategory::MachineLearning)
                .with_description("Context window packing and citation tracking"),
        ]);
        self.register_component(component);
    }

    fn register_trueno_zram(&mut self) {
        let component = StackComponent::new(
            "trueno-zram",
            "0.1.0",
            StackLayer::Primitives,
            "SIMD-accelerated memory compression for Linux zram",
        )
        .with_capabilities(vec![
            Capability::new("lz4_compression", CapabilityCategory::Compute)
                .with_description("SIMD-accelerated LZ4 compression (≥3 GB/s)"),
            Capability::new("zstd_compression", CapabilityCategory::Compute)
                .with_description("Zstandard compression with configurable levels"),
            Capability::new("adaptive_compression", CapabilityCategory::Compute)
                .with_description("Entropy-based algorithm selection"),
            Capability::new("page_compression", CapabilityCategory::Compute)
                .with_description("4KB page-aligned memory compression"),
            Capability::new("simd_dispatch", CapabilityCategory::Compute)
                .with_description("Runtime SIMD backend selection (AVX2/AVX-512/NEON)"),
        ]);
        self.register_component(component);
    }

    fn register_trueno_ublk(&mut self) {
        let component = StackComponent::new(
            "trueno-ublk",
            "0.1.0",
            StackLayer::Primitives,
            "GPU-accelerated ZRAM block device replacement via userspace block driver (ublk)",
        )
        .with_capabilities(vec![
            Capability::new("ublk_driver", CapabilityCategory::Compute)
                .with_description("Userspace block device driver via libublk"),
            Capability::new("gpu_compression", CapabilityCategory::Compute)
                .with_description("GPU-accelerated page compression (CUDA/wgpu)"),
            Capability::new("zram_replacement", CapabilityCategory::Storage)
                .with_description("Drop-in Linux ZRAM replacement with GPU offload"),
            Capability::new("adaptive_backend", CapabilityCategory::Compute)
                .with_description("Automatic GPU/SIMD/CPU backend selection"),
            Capability::new("io_uring", CapabilityCategory::Compute)
                .with_description("High-performance I/O via io_uring"),
        ]);
        self.register_component(component);
    }

    fn register_pepita(&mut self) {
        let component = StackComponent::new(
            "pepita",
            "0.1.0",
            StackLayer::Primitives,
            "Pure Rust kernel interfaces and distributed computing primitives for Sovereign AI",
        )
        .with_capabilities(vec![
            Capability::new("io_uring", CapabilityCategory::Compute)
                .with_description("Linux async I/O interface"),
            Capability::new("ublk", CapabilityCategory::Compute)
                .with_description("Userspace block device driver"),
            Capability::new("blk_mq", CapabilityCategory::Compute)
                .with_description("Multi-queue block layer"),
            Capability::new("zram", CapabilityCategory::Storage)
                .with_description("Compressed RAM storage with LZ4"),
            Capability::new("vmm", CapabilityCategory::Distribution)
                .with_description("KVM-based MicroVM runtime"),
            Capability::new("virtio", CapabilityCategory::Distribution)
                .with_description("Virtio vsock and block devices"),
            Capability::new("simd", CapabilityCategory::Compute)
                .with_description("SIMD operations (AVX-512/AVX2/SSE/NEON)"),
            Capability::new("gpu", CapabilityCategory::Compute)
                .with_description("GPU compute via wgpu"),
            Capability::new("work_stealing", CapabilityCategory::Distribution)
                .with_description("Blumofe-Leiserson work-stealing scheduler"),
        ]);
        self.register_component(component);
    }

    fn register_aprender(&mut self) {
        let component = StackComponent::new(
            "aprender",
            "0.21.0",
            StackLayer::MlAlgorithms,
            "Next-generation machine learning library with APR v2 format (LZ4/ZSTD compression)",
        )
        .with_capabilities(vec![
            Capability::new("apr_v2", CapabilityCategory::Storage)
                .with_description("APR v2 format with binary index, LZ4/ZSTD compression"),
            Capability::new("int4_quantization", CapabilityCategory::MachineLearning)
                .with_description("Int4 model quantization"),
            Capability::new("int8_quantization", CapabilityCategory::MachineLearning)
                .with_description("Int8 model quantization"),
            Capability::new("zero_copy_loading", CapabilityCategory::Storage)
                .with_description("Zero-copy model loading via mmap"),
            Capability::new("linear_regression", CapabilityCategory::MachineLearning),
            Capability::new("logistic_regression", CapabilityCategory::MachineLearning),
            Capability::new("decision_tree", CapabilityCategory::MachineLearning),
            Capability::new("random_forest", CapabilityCategory::MachineLearning),
            Capability::new("gbm", CapabilityCategory::MachineLearning)
                .with_description("Gradient boosting machines"),
            Capability::new("naive_bayes", CapabilityCategory::MachineLearning),
            Capability::new("knn", CapabilityCategory::MachineLearning)
                .with_description("K-nearest neighbors"),
            Capability::new("svm", CapabilityCategory::MachineLearning)
                .with_description("Support vector machines"),
            Capability::new("kmeans", CapabilityCategory::MachineLearning),
            Capability::new("pca", CapabilityCategory::MachineLearning)
                .with_description("Principal component analysis"),
            Capability::new("dbscan", CapabilityCategory::MachineLearning),
            Capability::new("hierarchical", CapabilityCategory::MachineLearning),
            Capability::new("standard_scaler", CapabilityCategory::MachineLearning),
            Capability::new("minmax_scaler", CapabilityCategory::MachineLearning),
            Capability::new("label_encoder", CapabilityCategory::MachineLearning),
            Capability::new("train_test_split", CapabilityCategory::MachineLearning),
            Capability::new("cross_validate", CapabilityCategory::MachineLearning),
            Capability::new("grid_search", CapabilityCategory::MachineLearning),
        ]);
        self.register_component(component);
    }

    fn register_entrenar(&mut self) {
        let component = StackComponent::new(
            "entrenar",
            "0.2.1",
            StackLayer::MlPipeline,
            "Training library with autograd, LoRA, and quantization",
        )
        .with_capabilities(vec![
            Capability::new("autograd", CapabilityCategory::MachineLearning)
                .with_description("Automatic differentiation"),
            Capability::new("lora", CapabilityCategory::MachineLearning)
                .with_description("Low-rank adaptation"),
            Capability::new("qlora", CapabilityCategory::MachineLearning)
                .with_description("Quantized LoRA"),
            Capability::new("quantization", CapabilityCategory::MachineLearning)
                .with_description("Model quantization (int8, int4)"),
            Capability::new("model_merge", CapabilityCategory::MachineLearning),
            Capability::new("distillation", CapabilityCategory::MachineLearning),
        ]);
        self.register_component(component);
    }

    fn register_realizar(&mut self) {
        let component = StackComponent::new(
            "realizar",
            "0.4.0",
            StackLayer::MlPipeline,
            "Pure Rust ML inference engine with APR v2, GPU kernels (FlashAttention, Q4K/Q5K/Q6K)",
        )
        .with_capabilities(vec![
            Capability::new("gguf", CapabilityCategory::MachineLearning)
                .with_description("GGUF model format support"),
            Capability::new("safetensors", CapabilityCategory::MachineLearning)
                .with_description("Safetensors model loading"),
            Capability::new("apr_v2_format", CapabilityCategory::MachineLearning)
                .with_description("APR v2 model format with compression"),
            Capability::new("gemm_kernel", CapabilityCategory::Compute)
                .with_description("GPU matrix multiplication (naive, tiled, tensor core)"),
            Capability::new("attention_kernel", CapabilityCategory::Compute)
                .with_description("FlashAttention-style tiled attention"),
            Capability::new("softmax_kernel", CapabilityCategory::Compute)
                .with_description("Numerically stable softmax with warp shuffle"),
            Capability::new("layernorm_kernel", CapabilityCategory::Compute)
                .with_description("Fused layer normalization"),
            Capability::new("quantize_kernel", CapabilityCategory::Compute)
                .with_description("Q4_K/Q5_K/Q6_K dequantization fused with matmul"),
            Capability::new("transformer_serving", CapabilityCategory::MachineLearning)
                .with_description("LLM/transformer inference runtime"),
            Capability::new("kv_cache", CapabilityCategory::MachineLearning)
                .with_description("KV-cache for efficient generation"),
            Capability::new("continuous_batching", CapabilityCategory::MachineLearning)
                .with_description("Dynamic request batching"),
            Capability::new("model_serving", CapabilityCategory::MachineLearning)
                .with_description("Production model serving"),
            Capability::new("moe_routing", CapabilityCategory::MachineLearning)
                .with_description("Mixture-of-experts routing"),
            Capability::new("circuit_breaker", CapabilityCategory::MachineLearning)
                .with_description("Fault tolerance"),
            Capability::new("lambda", CapabilityCategory::Distribution)
                .with_description("AWS Lambda (53,000x faster cold start)"),
            Capability::new("container", CapabilityCategory::Distribution)
                .with_description("Container deployment"),
            Capability::new("edge", CapabilityCategory::Distribution)
                .with_description("Edge deployment"),
        ]);
        self.register_component(component);
    }

    fn register_whisper_apr(&mut self) {
        let component = StackComponent::new(
            "whisper-apr",
            "0.1.0",
            StackLayer::MlPipeline,
            "Pure Rust OpenAI Whisper ASR - WASM-first, Int4/Int8 quantization, streaming",
        )
        .with_capabilities(vec![
            Capability::new("speech_recognition", CapabilityCategory::MachineLearning)
                .with_description("Automatic speech recognition (Whisper architecture)"),
            Capability::new("streaming_transcription", CapabilityCategory::MachineLearning)
                .with_description("Real-time streaming transcription"),
            Capability::new("multilingual", CapabilityCategory::MachineLearning)
                .with_description("99+ language support"),
            Capability::new("apr_v2_whisper", CapabilityCategory::MachineLearning)
                .with_description("APR v2 format with LZ4/ZSTD compression"),
            Capability::new("whisper_quantization", CapabilityCategory::MachineLearning)
                .with_description("Int4/Int8 quantized Whisper models"),
            Capability::new("wasm_first", CapabilityCategory::Distribution)
                .with_description("WASM-first design for browser deployment"),
            Capability::new("no_std", CapabilityCategory::Distribution)
                .with_description("no_std compatible for embedded"),
        ]);
        self.register_component(component);
    }

    fn register_depyler(&mut self) {
        let component = StackComponent::new(
            "depyler",
            "0.1.0",
            StackLayer::Transpilers,
            "Python to Rust transpiler with ML oracle",
        )
        .with_capabilities(vec![
            Capability::new("type_inference", CapabilityCategory::Transpilation)
                .with_description("Python type inference"),
            Capability::new("sklearn_to_aprender", CapabilityCategory::Transpilation)
                .with_description("sklearn to aprender conversion"),
            Capability::new("numpy_to_trueno", CapabilityCategory::Transpilation)
                .with_description("NumPy to trueno conversion"),
        ]);
        self.register_component(component);
    }

    fn register_decy(&mut self) {
        let component = StackComponent::new(
            "decy",
            "0.1.0",
            StackLayer::Transpilers,
            "C/C++ to Rust transpiler",
        )
        .with_capabilities(vec![
            Capability::new("ownership_inference", CapabilityCategory::Transpilation)
                .with_description("Infer Rust ownership from C code"),
            Capability::new("unsafe_elimination", CapabilityCategory::Transpilation)
                .with_description("Eliminate unsafe blocks where possible"),
        ]);
        self.register_component(component);
    }

    fn register_bashrs(&mut self) {
        let component = StackComponent::new(
            "bashrs",
            "6.41.0",
            StackLayer::Transpilers,
            "Rust-to-Shell transpiler for deterministic bootstrap scripts",
        )
        .with_capabilities(vec![
            Capability::new("rust_to_shell", CapabilityCategory::Transpilation)
                .with_description("Transpile Rust to portable shell scripts"),
            Capability::new("bootstrap_scripts", CapabilityCategory::Transpilation)
                .with_description("Generate deterministic bootstrap scripts"),
            Capability::new("cross_platform_shell", CapabilityCategory::Transpilation)
                .with_description("Generate POSIX-compliant shell code"),
        ]);
        self.register_component(component);
    }

    fn register_ruchy(&mut self) {
        let component = StackComponent::new(
            "ruchy",
            "3.213.0",
            StackLayer::Transpilers,
            "Systems scripting language that transpiles to idiomatic Rust",
        )
        .with_capabilities(vec![
            Capability::new("script_to_rust", CapabilityCategory::Transpilation)
                .with_description("Transpile scripts to idiomatic Rust"),
            Capability::new("shell_semantics", CapabilityCategory::Transpilation)
                .with_description("Shell-like semantics with Rust safety"),
            Capability::new("wasm_target", CapabilityCategory::Transpilation)
                .with_description("Compile to WebAssembly"),
            Capability::new("extreme_tdd", CapabilityCategory::Validation)
                .with_description("Built-in extreme TDD methodology"),
        ]);
        self.register_component(component);
    }

    fn register_batuta(&mut self) {
        let component = StackComponent::new(
            "batuta",
            "0.1.0",
            StackLayer::Orchestration,
            "Workflow orchestrator for transpilation pipelines",
        )
        .with_capabilities(vec![
            Capability::new("analysis", CapabilityCategory::Validation),
            Capability::new("transpilation", CapabilityCategory::Transpilation),
            Capability::new("optimization", CapabilityCategory::Compute),
            Capability::new("validation", CapabilityCategory::Validation),
            Capability::new("deployment", CapabilityCategory::Distribution),
        ]);
        self.register_component(component);
    }

    fn register_repartir(&mut self) {
        let component = StackComponent::new(
            "repartir",
            "2.0.0",
            StackLayer::Orchestration,
            "Sovereign AI-grade distributed computing primitives (CPU, GPU, HPC, pepita integration)",
        )
        .with_capabilities(vec![
            Capability::new("cpu_executor", CapabilityCategory::Distribution)
                .with_description("Multi-core CPU execution with work-stealing"),
            Capability::new("gpu_executor", CapabilityCategory::Distribution)
                .with_description("wgpu GPU compute (Vulkan/Metal/DX12/WebGPU)"),
            Capability::new("remote_executor", CapabilityCategory::Distribution)
                .with_description("TCP-based distributed execution across machines"),
            Capability::new("remote_tls", CapabilityCategory::Distribution)
                .with_description("TLS-secured remote execution"),
            Capability::new("work_stealing", CapabilityCategory::Distribution)
                .with_description("Blumofe & Leiserson work-stealing scheduler"),
            Capability::new("task_pool", CapabilityCategory::Distribution)
                .with_description("High-level Pool API for task submission"),
            Capability::new("tensor_ops", CapabilityCategory::Compute)
                .with_description("trueno SIMD tensor integration"),
            Capability::new("checkpoint", CapabilityCategory::Storage)
                .with_description("trueno-db + Parquet state persistence"),
            Capability::new("job_flow_tui", CapabilityCategory::Profiling)
                .with_description("TUI dashboard for job monitoring"),
        ]);
        self.register_component(component);
    }

    fn register_certeza(&mut self) {
        let component = StackComponent::new(
            "certeza",
            "0.1.0",
            StackLayer::Quality,
            "Quality validation framework",
        )
        .with_capabilities(vec![
            Capability::new("coverage_check", CapabilityCategory::Validation),
            Capability::new("mutation_testing", CapabilityCategory::Validation),
            Capability::new("tdg_scoring", CapabilityCategory::Validation)
                .with_description("Technical debt gauge scoring"),
            Capability::new("privacy_audit", CapabilityCategory::Validation),
        ]);
        self.register_component(component);
    }

    fn register_pmat(&mut self) {
        let component = StackComponent::new(
            "pmat",
            "2.205.0",
            StackLayer::Quality,
            "Zero-config AI context generation and code quality toolkit",
        )
        .with_capabilities(vec![
            Capability::new("complexity_analysis", CapabilityCategory::Validation)
                .with_description("Cyclomatic and cognitive complexity analysis"),
            Capability::new("satd_detection", CapabilityCategory::Validation)
                .with_description("Self-admitted technical debt detection"),
            Capability::new("quality_gates", CapabilityCategory::Validation)
                .with_description("Tiered quality enforcement"),
            Capability::new("repo_health", CapabilityCategory::Validation)
                .with_description("Repository health scoring (0-125)"),
            Capability::new("mcp_server", CapabilityCategory::Distribution)
                .with_description("MCP protocol server for AI agents"),
            Capability::new("multi_language", CapabilityCategory::Validation)
                .with_description("17+ programming language support"),
        ]);
        self.register_component(component);
    }

    fn register_pforge(&mut self) {
        let component = StackComponent::new(
            "pforge",
            "0.1.4",
            StackLayer::Orchestration,
            "Zero-boilerplate MCP server framework with rust-mcp-sdk",
        )
        .with_capabilities(vec![
            Capability::new("mcp_codegen", CapabilityCategory::Transpilation)
                .with_description("Declarative MCP server code generation"),
            Capability::new("mcp_runtime", CapabilityCategory::Distribution)
                .with_description("MCP protocol runtime via rust-mcp-sdk"),
            Capability::new("zero_boilerplate", CapabilityCategory::Transpilation)
                .with_description("Macro-based boilerplate elimination"),
            Capability::new("tool_orchestration", CapabilityCategory::Distribution)
                .with_description("AI agent tool-use orchestration"),
            Capability::new("resource_provider", CapabilityCategory::Distribution)
                .with_description("MCP resource exposure for agents"),
            Capability::new("prompt_templates", CapabilityCategory::Distribution)
                .with_description("Reusable prompt template serving"),
            Capability::new("extreme_tdd", CapabilityCategory::Validation)
                .with_description("Built-in extreme TDD methodology"),
        ]);
        self.register_component(component);
    }

    fn register_renacer(&mut self) {
        let component = StackComponent::new(
            "renacer",
            "0.7.0",
            StackLayer::Quality,
            "Pure Rust system call tracer with source-aware correlation",
        )
        .with_capabilities(vec![
            Capability::new("syscall_trace", CapabilityCategory::Profiling)
                .with_description("System call tracing"),
            Capability::new("flamegraph", CapabilityCategory::Profiling)
                .with_description("Flamegraph generation"),
            Capability::new("golden_trace_comparison", CapabilityCategory::Profiling)
                .with_description("Compare traces for semantic equivalence"),
        ]);
        self.register_component(component);
    }

    fn register_alimentar(&mut self) {
        let component = StackComponent::new(
            "alimentar",
            "0.1.0",
            StackLayer::Data,
            "Data loading and preprocessing in pure Rust",
        )
        .with_capabilities(vec![
            Capability::new("csv", CapabilityCategory::Storage),
            Capability::new("parquet", CapabilityCategory::Storage),
            Capability::new("json", CapabilityCategory::Storage),
            Capability::new("streaming", CapabilityCategory::Storage)
                .with_description("Streaming data loading"),
        ]);
        self.register_component(component);
    }

    fn register_pacha(&mut self) {
        let component = StackComponent::new(
            "pacha",
            "0.1.0",
            StackLayer::Data,
            "Model, Data and Recipe Registry for MLOps",
        )
        .with_capabilities(vec![
            Capability::new("model_versioning", CapabilityCategory::Storage)
                .with_description("Semantic versioned model artifacts"),
            Capability::new("model_lineage", CapabilityCategory::Storage)
                .with_description("Model lineage tracking"),
            Capability::new("artifact_storage", CapabilityCategory::Storage)
                .with_description("Artifact storage and retrieval"),
            Capability::new("dataset_versioning", CapabilityCategory::Storage)
                .with_description("Dataset version control"),
            Capability::new("data_lineage", CapabilityCategory::Storage)
                .with_description("Data lineage and provenance"),
            Capability::new("recipe_management", CapabilityCategory::Storage)
                .with_description("Training recipe versioning"),
            Capability::new("experiment_tracking", CapabilityCategory::Validation)
                .with_description("Experiment metrics and comparison"),
        ]);
        self.register_component(component);
    }

    fn register_simular(&mut self) {
        let component = StackComponent::new(
            "simular",
            "0.2.0",
            StackLayer::MlPipeline,
            "Unified simulation engine with Zero-JS WASM architecture and TUI",
        )
        .with_capabilities(vec![
            Capability::new("physics_simulation", CapabilityCategory::Compute)
                .with_description("N-body orbital mechanics with symplectic integrators"),
            Capability::new("monte_carlo", CapabilityCategory::Compute)
                .with_description("Monte Carlo Pi, Kingman's formula simulations"),
            Capability::new("optimization", CapabilityCategory::Compute)
                .with_description("TSP GRASP with 2-opt local search"),
            Capability::new("discrete_event", CapabilityCategory::Compute)
                .with_description("Queue theory, Little's Law simulations"),
            Capability::new("zero_js_wasm", CapabilityCategory::Compute)
                .with_description("Pure Rust/WASM with single-line JS initialization"),
            Capability::new("wasm_canvas", CapabilityCategory::Compute)
                .with_description("Canvas 2D rendering via web-sys"),
            Capability::new("wasm_dom", CapabilityCategory::Compute)
                .with_description("DOM manipulation via web-sys"),
            Capability::new("edd_demos", CapabilityCategory::Validation)
                .with_description("Equation-Driven Development demonstration framework"),
            Capability::new("jidoka_guards", CapabilityCategory::Validation)
                .with_description("Toyota Way stop-on-error invariant checking"),
            Capability::new("poka_yoke", CapabilityCategory::Validation)
                .with_description("Type-safe units via uom crate"),
            Capability::new("tui_ratatui", CapabilityCategory::Compute)
                .with_description("Terminal UI with ratatui"),
            Capability::new("deterministic_replay", CapabilityCategory::Validation)
                .with_description("Reproducible simulations with PCG seeds"),
        ]);
        self.register_component(component);
    }

    fn register_probar(&mut self) {
        let component = StackComponent::new(
            "probar",
            "0.1.0",
            StackLayer::Quality,
            "Property-based testing and GUI/pixel coverage for WASM demos",
        )
        .with_capabilities(vec![
            Capability::new("pixel_coverage", CapabilityCategory::Validation)
                .with_description("Canvas pixel coverage analysis"),
            Capability::new("gui_coverage", CapabilityCategory::Validation)
                .with_description("Interactive element coverage tracking"),
            Capability::new("property_testing", CapabilityCategory::Validation)
                .with_description("Property-based test generation"),
            Capability::new("metamorphic_testing", CapabilityCategory::Validation)
                .with_description("Metamorphic relation validation"),
            Capability::new("invariant_checking", CapabilityCategory::Validation)
                .with_description("Runtime invariant assertion"),
        ]);
        self.register_component(component);
    }

    fn register_forjar(&mut self) {
        let component = StackComponent::new(
            "forjar",
            "0.1.0",
            StackLayer::Orchestration,
            "Rust-native IaC for bare-metal provisioning with BLAKE3 content-addressed state",
        )
        .with_capabilities(vec![
            Capability::new("iac_provisioning", CapabilityCategory::Distribution)
                .with_description("Bare-metal infrastructure provisioning via SSH"),
            Capability::new("blake3_state", CapabilityCategory::Validation)
                .with_description("BLAKE3 content-addressed drift detection"),
            Capability::new("idempotent_apply", CapabilityCategory::Validation)
                .with_description("Idempotent convergence with hash-based skip"),
            Capability::new("template_resolution", CapabilityCategory::Transpilation)
                .with_description("{{params.X}} and {{machine.Y.Z}} template engine"),
            Capability::new("recipe_system", CapabilityCategory::Distribution)
                .with_description("Nix-inspired composable configuration recipes"),
            Capability::new("topo_sort", CapabilityCategory::Compute)
                .with_description("DAG-based resource dependency ordering"),
        ]);
        self.register_component(component);
    }

    fn register_apr_qa(&mut self) {
        let component = StackComponent::new(
            "apr-qa",
            "0.1.0",
            StackLayer::Quality,
            "APR model QA playbook with test generation, execution, and reporting",
        )
        .with_capabilities(vec![
            Capability::new("qa_test_generation", CapabilityCategory::Validation)
                .with_description("Generate QA tests for APR models"),
            Capability::new("model_validation", CapabilityCategory::Validation)
                .with_description("Validate APR model correctness and integrity"),
            Capability::new("qa_runner", CapabilityCategory::Validation)
                .with_description("Execute QA test suites against APR models"),
            Capability::new("benchmark_runner", CapabilityCategory::Profiling)
                .with_description("Run performance benchmarks on APR models"),
            Capability::new("qa_report", CapabilityCategory::Validation)
                .with_description("Generate QA reports with metrics and recommendations"),
            Capability::new("coverage_report", CapabilityCategory::Validation)
                .with_description("Model coverage analysis and reporting"),
        ]);
        self.register_component(component);
    }

    fn register_provable_contracts(&mut self) {
        let component = StackComponent::new(
            "provable-contracts",
            "0.1.0",
            StackLayer::Quality,
            "YAML contract to Kani verification pipeline for ML kernel correctness",
        )
        .with_capabilities(vec![
            Capability::new("contract_parsing", CapabilityCategory::Validation)
                .with_description("Parse YAML contract specifications for ML kernels"),
            Capability::new("scaffold_generation", CapabilityCategory::Transpilation)
                .with_description("Generate Kani harness scaffolds from contracts"),
            Capability::new("kani_verification", CapabilityCategory::Validation)
                .with_description("Formal verification via Kani model checker"),
            Capability::new("probar_generation", CapabilityCategory::Validation)
                .with_description("Generate probar property tests from contracts"),
            Capability::new("traceability_audit", CapabilityCategory::Validation)
                .with_description("Contract-to-proof traceability audit trail"),
            Capability::new("binding_registry", CapabilityCategory::Validation)
                .with_description("Registry of contract bindings to kernel implementations"),
        ]);
        self.register_component(component);
    }

    fn register_rmedia(&mut self) {
        let component = StackComponent::new(
            "rmedia",
            "0.1.0",
            StackLayer::Media,
            "Pure Rust headless video editor — 7-crate workspace, MLT XML compatibility, 1.73x faster than melt",
        )
        .with_capabilities(vec![
            Capability::new("video_rendering", CapabilityCategory::Media)
                .with_description("Headless video rendering pipeline"),
            Capability::new("mlt_xml", CapabilityCategory::Media)
                .with_description("MLT XML timeline parsing and execution"),
            Capability::new("ffmpeg_encode", CapabilityCategory::Media)
                .with_description("FFmpeg-based encoding/decoding"),
            Capability::new("transition_blend", CapabilityCategory::Media)
                .with_description("Cross-dissolve, fade, and wipe transitions"),
            Capability::new("course_production", CapabilityCategory::Media)
                .with_description("End-to-end course video production pipeline"),
            Capability::new("audio_processing", CapabilityCategory::Media)
                .with_description("Audio mixing, normalization, and ducking"),
            Capability::new("subtitle_burn_in", CapabilityCategory::Media)
                .with_description("ASS/SRT subtitle burn-in rendering"),
            Capability::new("media_concat", CapabilityCategory::Media)
                .with_description("Frame-accurate media concatenation"),
            Capability::new("transcription_qa", CapabilityCategory::Validation)
                .with_description("Transcript quality scoring and validation"),
            Capability::new("vocabulary_enrichment", CapabilityCategory::ContentGeneration)
                .with_description("Key term extraction from transcripts"),
            Capability::new("content_generation", CapabilityCategory::ContentGeneration)
                .with_description("Outline, reflection prompt, and landing page generation"),
            Capability::new("svg_asset_pipeline", CapabilityCategory::Media)
                .with_description("SVG banner and thumbnail generation"),
            Capability::new("course_quality_scoring", CapabilityCategory::Validation)
                .with_description("Multi-dimensional course quality scoring"),
            Capability::new("course_publishing", CapabilityCategory::Distribution)
                .with_description("Coursera/platform publishing pipeline"),
        ]);
        self.register_component(component);
    }

    fn register_tiny_model_ground_truth(&mut self) {
        let component = StackComponent::new(
            "tiny-model-ground-truth",
            "0.1.0",
            StackLayer::Quality,
            "Popperian falsification test suite for model conversion parity",
        )
        .with_capabilities(vec![
            Capability::new("parity_checking", CapabilityCategory::Validation)
                .with_description("Validate output parity between HuggingFace and realizar"),
            Capability::new("oracle_generation", CapabilityCategory::Validation)
                .with_description("Generate oracle outputs from HuggingFace reference models"),
            Capability::new("falsification_testing", CapabilityCategory::Validation)
                .with_description("Popperian falsification of conversion correctness"),
            Capability::new("quantization_drift", CapabilityCategory::Validation)
                .with_description("Measure quantization drift across format conversions"),
            Capability::new("roundtrip_validation", CapabilityCategory::Validation)
                .with_description("Validate roundtrip conversion fidelity"),
        ]);
        self.register_component(component);
    }
}
