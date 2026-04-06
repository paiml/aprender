use anyhow::Result;
use cgp::{analysis, doctor, profilers};
use clap::{Parser, Subcommand};

/// CGP: Compute-GPU-Profile — Unified Performance Analysis CLI
///
/// Own the Stack: One Binary, All Backends, Zero Blind Spots.
/// Profiles scalar, SIMD (SSE2/AVX2/AVX-512/NEON/WASM SIMD128),
/// wgpu (Vulkan/Metal/DX12/WebGPU), and CUDA workloads.
#[derive(Parser)]
#[command(name = "cgp", version, about, long_about = None)]
struct Cli {
    /// Output JSON instead of human-readable text
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Profile a kernel or function (runtime execution)
    Profile {
        #[command(subcommand)]
        target: ProfileTarget,
    },
    /// Enhanced criterion benchmarking with hardware counters
    Bench {
        /// Benchmark name
        #[arg(long)]
        bench: String,
        /// Hardware counters to collect (comma-separated)
        #[arg(long)]
        counters: Option<String>,
        /// Check regression against saved baseline
        #[arg(long)]
        check_regression: bool,
        /// Regression threshold percentage
        #[arg(long, default_value = "5")]
        threshold: f64,
        /// Overlay roofline model
        #[arg(long)]
        roofline: bool,
    },
    /// Generate roofline model for target hardware
    Roofline {
        /// Target backend (cuda, avx2, avx512, neon, wgpu)
        #[arg(long)]
        target: String,
        /// Kernels to plot on roofline
        #[arg(long)]
        kernels: Option<String>,
        /// Export to file
        #[arg(long)]
        export: Option<String>,
        /// Use empirical measurement instead of spec values
        #[arg(long)]
        empirical: bool,
    },
    /// Compare two profiles (git integration)
    Diff {
        /// Baseline commit or profile path
        #[arg(long)]
        baseline: Option<String>,
        /// Current commit or profile path
        #[arg(long)]
        current: Option<String>,
        /// Before commit
        #[arg(long)]
        before: Option<String>,
        /// After commit
        #[arg(long)]
        after: Option<String>,
    },
    /// Verify performance contracts (CI/CD gate)
    Contract {
        #[command(subcommand)]
        action: ContractAction,
    },
    /// System-wide timeline (wraps nsys)
    Trace {
        /// Binary to trace
        binary: String,
        /// Trace duration
        #[arg(long)]
        duration: Option<String>,
    },
    /// Static code analysis (wraps trueno-explain)
    Explain {
        /// Analysis target (ptx, simd, wgsl)
        target: String,
        /// Kernel name
        #[arg(long)]
        kernel: Option<String>,
    },
    /// Interactive TUI exploration mode
    Tui,
    /// Save/load performance baselines
    Baseline {
        /// Save current profile as baseline
        #[arg(long)]
        save: Option<String>,
        /// Load baseline from file
        #[arg(long)]
        load: Option<String>,
    },
    /// Check tool availability and hardware capabilities
    Doctor,
    /// Head-to-head competitor comparison
    Compete {
        /// Workload name (e.g., gemm)
        workload: String,
        /// Our command
        #[arg(long)]
        ours: String,
        /// Competitor commands (can be repeated)
        #[arg(long)]
        theirs: Vec<String>,
        /// Labels for each entry (comma-separated)
        #[arg(long)]
        label: Option<String>,
    },
}

#[derive(Subcommand)]
enum ProfileTarget {
    /// Profile a CUDA PTX kernel via ncu + CUPTI
    Kernel {
        /// Kernel name
        #[arg(long)]
        name: String,
        /// Problem size (e.g., 512 for square matrix)
        #[arg(long)]
        size: u32,
        /// Generate roofline overlay
        #[arg(long)]
        roofline: bool,
        /// Specific ncu metrics to collect
        #[arg(long)]
        metrics: Option<String>,
    },
    /// Profile cuBLAS/cuBLASLt operations
    Cublas {
        /// Operation (gemm_f16, gemm_f32, etc.)
        #[arg(long)]
        op: String,
        /// Problem size
        #[arg(long)]
        size: u32,
    },
    /// Profile wgpu compute shaders
    Wgpu {
        /// WGSL shader path
        #[arg(long)]
        shader: String,
        /// Dispatch dimensions (e.g., 256,256,1)
        #[arg(long)]
        dispatch: Option<String>,
        /// Target (native or web)
        #[arg(long)]
        target: Option<String>,
    },
    /// Profile Apple Metal compute kernels
    Metal {
        /// Metal shader name
        #[arg(long)]
        shader: String,
        /// Dispatch size
        #[arg(long)]
        dispatch: Option<u32>,
    },
    /// Profile CPU SIMD functions
    Simd {
        /// Function name
        #[arg(long)]
        function: String,
        /// Problem size
        #[arg(long)]
        size: u32,
        /// Target architecture (avx2, avx512, neon, sse2)
        #[arg(long)]
        arch: String,
    },
    /// Profile WASM SIMD128 via wasmtime
    Wasm {
        /// Function name
        #[arg(long)]
        function: String,
        /// Problem size
        #[arg(long)]
        size: u32,
    },
    /// Profile quantized CPU kernels (Q4K/Q6K)
    Quant {
        /// Kernel name (q4k_gemv, q6k_gemv, q5k_gemv, q8_gemv, nf4_gemv)
        #[arg(long, required_unless_present = "all")]
        kernel: Option<String>,
        /// Dimensions (MxNxK format)
        #[arg(long, required_unless_present = "all")]
        size: Option<String>,
        /// Profile all standard LLM layer sizes (ffn_up, ffn_down, attn_qkv, generic_4K)
        #[arg(long)]
        all: bool,
    },
    /// Profile scalar baseline
    Scalar {
        /// Function name
        #[arg(long)]
        function: String,
        /// Problem size
        #[arg(long)]
        size: u32,
    },
    /// Profile Rayon parallel workloads
    Parallel {
        /// Function name
        #[arg(long)]
        function: String,
        /// Problem size
        #[arg(long)]
        size: u32,
        /// Thread count (or "auto")
        #[arg(long)]
        threads: Option<String>,
    },
    /// Cross-backend comparison
    Compare {
        /// Kernel name
        #[arg(long)]
        kernel: String,
        /// Problem size
        #[arg(long)]
        size: u32,
        /// Backends to compare (comma-separated)
        #[arg(long)]
        backends: String,
    },
    /// Parallel scaling sweep (thread count vs throughput)
    Scaling {
        /// Problem size
        #[arg(long)]
        size: u32,
        /// Max threads to test (default: num_cpus)
        #[arg(long)]
        max_threads: Option<usize>,
        /// Runs per thread count for min-of-N timing
        #[arg(long, default_value = "3")]
        runs: usize,
    },
    /// Profile an arbitrary binary
    Binary {
        /// Binary path
        path: String,
        /// Kernel name filter
        #[arg(long)]
        kernel_filter: Option<String>,
        /// Enable system trace
        #[arg(long)]
        trace: bool,
        /// Trace duration
        #[arg(long)]
        duration: Option<String>,
    },
    /// Profile a Python script
    Python {
        /// Arguments after --
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Profile a shared library function
    Library {
        /// Path to .so file
        #[arg(long)]
        so: String,
        /// Symbol name
        #[arg(long)]
        symbol: String,
        /// Arguments (key=value pairs)
        #[arg(long)]
        args: Option<String>,
    },
}

#[derive(Subcommand)]
enum ContractAction {
    /// Verify performance contracts
    Verify {
        /// Directory containing contract YAML files
        #[arg(long)]
        contracts_dir: Option<String>,
        /// Specific contract file
        #[arg(long)]
        contract: Option<String>,
        /// Fail on any regression
        #[arg(long)]
        fail_on_regression: bool,
        /// Verify cgp's own contracts
        #[arg(long, name = "self")]
        self_verify: bool,
    },
    /// Generate contract from current measurement
    Generate {
        /// Kernel name
        #[arg(long)]
        kernel: String,
        /// Problem size
        #[arg(long)]
        size: u32,
        /// Regression tolerance percentage
        #[arg(long, default_value = "10")]
        tolerance: f64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let json = cli.json;

    match cli.command {
        Commands::Doctor => doctor::run_doctor(json),
        Commands::Profile { target } => dispatch_profile(target, json),
        Commands::Roofline { target, kernels, export, empirical } => {
            analysis::roofline::run_roofline(
                &target,
                kernels.as_deref(),
                export.as_deref(),
                empirical,
                json,
            )
        }
        Commands::Bench { bench, counters, check_regression, threshold, roofline } => {
            analysis::bench::run_bench(
                &bench,
                counters.as_deref(),
                check_regression,
                threshold,
                roofline,
            )
        }
        Commands::Diff { baseline, current, before, after } => analysis::diff::run_diff(
            baseline.as_deref(),
            current.as_deref(),
            before.as_deref(),
            after.as_deref(),
            json,
        ),
        Commands::Contract { action } => dispatch_contract(action),
        Commands::Trace { binary, duration } => {
            profilers::cuda::run_trace(&binary, duration.as_deref())
        }
        Commands::Explain { target, kernel } => {
            analysis::explain::run_explain(&target, kernel.as_deref())
        }
        Commands::Tui => {
            println!("cgp tui: interactive mode (requires presentar)");
            println!("  (Not yet implemented — use stdout commands for now)");
            Ok(())
        }
        Commands::Baseline { save, load } => {
            analysis::baseline::run_baseline(save.as_deref(), load.as_deref())
        }
        Commands::Compete { workload, ours, theirs, label } => {
            analysis::compete::run_compete(&workload, &ours, &theirs, label.as_deref(), json)
        }
    }
}

fn dispatch_profile(target: ProfileTarget, json: bool) -> Result<()> {
    match target {
        ProfileTarget::Kernel { name, size, roofline, metrics } => {
            profilers::cuda::profile_kernel(&name, size, roofline, metrics.as_deref())
        }
        ProfileTarget::Cublas { op, size } => profilers::cuda::profile_cublas(&op, size),
        ProfileTarget::Wgpu { shader, dispatch, target } => {
            profilers::wgpu_profiler::profile_wgpu(&shader, dispatch.as_deref(), target.as_deref())
        }
        ProfileTarget::Metal { shader, dispatch } => {
            #[cfg(target_os = "macos")]
            {
                println!("cgp profile metal: shader={shader} dispatch={dispatch:?}");
                Ok(())
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (&shader, dispatch);
                anyhow::bail!("Metal backend requires macOS -- use --backend wgpu for Vulkan")
            }
        }
        ProfileTarget::Simd { function, size, arch } => {
            profilers::simd::profile_simd(&function, size, &arch)
        }
        ProfileTarget::Wasm { function, size } => profilers::wasm::profile_wasm(&function, size),
        ProfileTarget::Quant { kernel, size, all } => {
            if all {
                profilers::quant::profile_quant_all()
            } else {
                profilers::quant::profile_quant(
                    kernel.as_deref().unwrap_or("q4k_gemv"),
                    size.as_deref().unwrap_or("4096x1x4096"),
                )
            }
        }
        ProfileTarget::Scalar { function, size } => {
            profilers::scalar::profile_scalar(&function, size)
        }
        ProfileTarget::Parallel { function, size, threads } => {
            profilers::rayon_parallel::profile_parallel(&function, size, threads.as_deref())
        }
        ProfileTarget::Compare { kernel, size, backends } => {
            analysis::compare::run_compare(&kernel, size, &backends, json)
        }
        ProfileTarget::Scaling { size, max_threads, runs } => {
            profilers::rayon_parallel::profile_scaling(size, max_threads, runs, json)
        }
        ProfileTarget::Binary { path, kernel_filter, trace, duration } => {
            profilers::cuda::profile_binary(
                &path,
                kernel_filter.as_deref(),
                trace,
                duration.as_deref(),
            )
        }
        ProfileTarget::Python { args } => profilers::cuda::profile_python(&args),
        ProfileTarget::Library { so, symbol, args } => {
            println!("cgp profile library: {so}::{symbol} args={args:?}");
            Ok(())
        }
    }
}

fn dispatch_contract(action: ContractAction) -> Result<()> {
    match action {
        ContractAction::Verify { contracts_dir, contract, fail_on_regression, self_verify } => {
            analysis::contracts::run_verify(
                contracts_dir.as_deref(),
                contract.as_deref(),
                self_verify,
                fail_on_regression,
            )
        }
        ContractAction::Generate { kernel, size, tolerance } => {
            analysis::contracts::run_generate(&kernel, size, tolerance)
        }
    }
}
