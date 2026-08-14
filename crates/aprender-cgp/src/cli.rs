//! Command surface for `apr perf` — the unified performance-analysis CLI.
//!
//! This module used to be `src/main.rs` behind the standalone `aprender-cgp`
//! (`cgp`) binary. That binary is gone (APR-MONO: one installed binary, `apr`);
//! every command below is reached as `apr perf <COMMAND>`.
//!
//! [`Commands`] is embedded directly by `apr-cli` with `#[command(subcommand)]`,
//! so there is exactly one definition of the argument surface and no second
//! copy that can drift (defect class #2418 — an argument silently dropped when
//! a CLI is re-homed).
//!
//! `cgp`'s own `--json` was a top-level `global = true` flag; `apr`'s root
//! `--json` is likewise global, so it propagates into these subcommands and
//! reaches the same code by the same name.

use anyhow::Result;
use clap::Subcommand;

use crate::{analysis, doctor, profilers};

/// Top-level `apr perf` commands.
///
/// `Clone` so apr's dispatch tree, which hands out shared references, can
/// pass an owned command to [`run_command`].
#[derive(Debug, Clone, Subcommand)]
pub enum Commands {
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

/// Targets for `apr perf profile <TARGET>`.
#[derive(Debug, Clone, Subcommand)]
pub enum ProfileTarget {
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

/// Actions for `apr perf contract <ACTION>`.
#[derive(Debug, Clone, Subcommand)]
pub enum ContractAction {
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

/// Run one `apr perf` command.
///
/// `json` is the effective value of the global `--json` flag.
///
/// # Errors
///
/// Propagates whatever the selected profiler, analysis, or contract action
/// returns.
pub fn run_command(command: Commands, json: bool) -> Result<()> {
    match command {
        Commands::Doctor => doctor::run_doctor(json),
        Commands::Profile { target } => dispatch_profile(target, json),
        Commands::Roofline {
            target,
            kernels,
            export,
            empirical,
        } => analysis::roofline::run_roofline(
            &target,
            kernels.as_deref(),
            export.as_deref(),
            empirical,
            json,
        ),
        Commands::Bench {
            bench,
            counters,
            check_regression,
            threshold,
            roofline,
        } => analysis::bench::run_bench(
            &bench,
            counters.as_deref(),
            check_regression,
            threshold,
            roofline,
        ),
        Commands::Diff {
            baseline,
            current,
            before,
            after,
        } => analysis::diff::run_diff(
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
        // #2407 class: this arm used to print "(Not yet implemented)" and
        // return Ok(()), so every caller — including a CI gate — read "did
        // nothing" as "succeeded". An advertised command with no implementation
        // must fail.
        Commands::Tui => anyhow::bail!(
            "apr perf tui is not implemented — use the non-interactive \
             `apr perf profile` / `apr perf roofline` commands"
        ),
        Commands::Baseline { save, load } => {
            analysis::baseline::run_baseline(save.as_deref(), load.as_deref())
        }
        Commands::Compete {
            workload,
            ours,
            theirs,
            label,
        } => analysis::compete::run_compete(&workload, &ours, &theirs, label.as_deref(), json),
    }
}

/// Dispatch `apr perf profile <TARGET>`.
///
/// # Errors
///
/// Propagates whatever the selected profiler returns.
pub fn dispatch_profile(target: ProfileTarget, json: bool) -> Result<()> {
    match target {
        ProfileTarget::Kernel {
            name,
            size,
            roofline,
            metrics,
        } => profilers::cuda::profile_kernel(&name, size, roofline, metrics.as_deref()),
        ProfileTarget::Cublas { op, size } => profilers::cuda::profile_cublas(&op, size),
        ProfileTarget::Wgpu {
            shader,
            dispatch,
            target,
        } => {
            profilers::wgpu_profiler::profile_wgpu(&shader, dispatch.as_deref(), target.as_deref())
        }
        ProfileTarget::Metal { shader, dispatch } => {
            #[cfg(target_os = "macos")]
            {
                // #2407 class: printing the arguments back is not profiling.
                let _ = (&shader, dispatch);
                anyhow::bail!(
                    "apr perf profile metal is not implemented — use \
                     `apr perf profile wgpu` for Metal via wgpu"
                )
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (&shader, dispatch);
                anyhow::bail!("Metal backend requires macOS -- use --backend wgpu for Vulkan")
            }
        }
        ProfileTarget::Simd {
            function,
            size,
            arch,
        } => profilers::simd::profile_simd(&function, size, &arch),
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
        ProfileTarget::Parallel {
            function,
            size,
            threads,
        } => profilers::rayon_parallel::profile_parallel(&function, size, threads.as_deref()),
        ProfileTarget::Compare {
            kernel,
            size,
            backends,
        } => analysis::compare::run_compare(&kernel, size, &backends, json),
        ProfileTarget::Scaling {
            size,
            max_threads,
            runs,
        } => profilers::rayon_parallel::profile_scaling(size, max_threads, runs, json),
        ProfileTarget::Binary {
            path,
            kernel_filter,
            trace,
            duration,
        } => profilers::cuda::profile_binary(
            &path,
            kernel_filter.as_deref(),
            trace,
            duration.as_deref(),
        ),
        ProfileTarget::Python { args } => profilers::cuda::profile_python(&args),
        // #2407 class: this echoed its arguments and returned Ok(()).
        ProfileTarget::Library { so, symbol, args } => {
            let _ = (&so, &symbol, &args);
            anyhow::bail!(
                "apr perf profile library is not implemented — profile the \
                 calling binary with `apr perf profile binary <PATH>`"
            )
        }
    }
}

/// Dispatch `apr perf contract <ACTION>`.
///
/// # Errors
///
/// Propagates whatever the contract action returns.
pub fn dispatch_contract(action: ContractAction) -> Result<()> {
    match action {
        ContractAction::Verify {
            contracts_dir,
            contract,
            fail_on_regression,
            self_verify,
        } => analysis::contracts::run_verify(
            contracts_dir.as_deref(),
            contract.as_deref(),
            self_verify,
            fail_on_regression,
        ),
        ContractAction::Generate {
            kernel,
            size,
            tolerance,
        } => analysis::contracts::run_generate(&kernel, size, tolerance),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser as _;

    /// Harness so the re-homed `Subcommand` enum can be parsed standalone,
    /// with the same global `--json` that `apr` provides at its root.
    #[derive(Debug, clap::Parser)]
    #[command(name = "perf")]
    struct Harness {
        #[arg(long, global = true)]
        json: bool,
        #[command(subcommand)]
        command: Commands,
    }

    fn parse(argv: &[&str]) -> Commands {
        Harness::try_parse_from(argv)
            .unwrap_or_else(|e| panic!("argv {argv:?} must parse: {e}"))
            .command
    }

    /// Every top-level command the standalone `cgp` binary exposed must still
    /// be reachable. A command that disappears in a re-home is the #2418
    /// defect; this list is the ledger.
    #[test]
    fn every_top_level_command_is_still_reachable() {
        assert!(matches!(parse(&["perf", "doctor"]), Commands::Doctor));
        assert!(matches!(parse(&["perf", "tui"]), Commands::Tui));
        assert!(matches!(
            parse(&["perf", "bench", "--bench", "gemm"]),
            Commands::Bench { .. }
        ));
        assert!(matches!(
            parse(&["perf", "roofline", "--target", "cuda"]),
            Commands::Roofline { .. }
        ));
        assert!(matches!(parse(&["perf", "diff"]), Commands::Diff { .. }));
        assert!(matches!(
            parse(&["perf", "trace", "./a.out"]),
            Commands::Trace { .. }
        ));
        assert!(matches!(
            parse(&["perf", "explain", "ptx"]),
            Commands::Explain { .. }
        ));
        assert!(matches!(
            parse(&["perf", "baseline", "--save", "b.json"]),
            Commands::Baseline { .. }
        ));
        assert!(matches!(
            parse(&["perf", "contract", "generate", "--kernel", "k", "--size", "4"]),
            Commands::Contract { .. }
        ));
        assert!(matches!(
            parse(&["perf", "compete", "gemm", "--ours", "x"]),
            Commands::Compete { .. }
        ));
        assert!(matches!(
            parse(&[
                "perf",
                "profile",
                "scalar",
                "--function",
                "f",
                "--size",
                "8"
            ]),
            Commands::Profile { .. }
        ));
    }

    /// Every `profile` target the standalone binary exposed.
    #[test]
    fn every_profile_target_is_still_reachable() {
        let cases: &[(&[&str], &str)] = &[
            (
                &["perf", "profile", "kernel", "--name", "k", "--size", "512"],
                "kernel",
            ),
            (
                &[
                    "perf", "profile", "cublas", "--op", "gemm_f32", "--size", "512",
                ],
                "cublas",
            ),
            (&["perf", "profile", "wgpu", "--shader", "s.wgsl"], "wgpu"),
            (&["perf", "profile", "metal", "--shader", "s"], "metal"),
            (
                &[
                    "perf",
                    "profile",
                    "simd",
                    "--function",
                    "f",
                    "--size",
                    "8",
                    "--arch",
                    "avx2",
                ],
                "simd",
            ),
            (
                &["perf", "profile", "wasm", "--function", "f", "--size", "8"],
                "wasm",
            ),
            (&["perf", "profile", "quant", "--all"], "quant"),
            (
                &[
                    "perf",
                    "profile",
                    "scalar",
                    "--function",
                    "f",
                    "--size",
                    "8",
                ],
                "scalar",
            ),
            (
                &[
                    "perf",
                    "profile",
                    "parallel",
                    "--function",
                    "f",
                    "--size",
                    "8",
                ],
                "parallel",
            ),
            (
                &[
                    "perf",
                    "profile",
                    "compare",
                    "--kernel",
                    "k",
                    "--size",
                    "8",
                    "--backends",
                    "simd",
                ],
                "compare",
            ),
            (&["perf", "profile", "scaling", "--size", "8"], "scaling"),
            (&["perf", "profile", "binary", "./a.out"], "binary"),
            (&["perf", "profile", "python", "--", "x.py"], "python"),
            (
                &[
                    "perf", "profile", "library", "--so", "l.so", "--symbol", "s",
                ],
                "library",
            ),
        ];
        for (argv, label) in cases {
            let Commands::Profile { .. } = parse(argv) else {
                panic!("`apr perf profile {label}` must parse as a profile target");
            };
        }
    }

    /// `profile kernel` carries four arguments; all four must land.
    #[test]
    fn profile_kernel_carries_every_argument() {
        let Commands::Profile {
            target:
                ProfileTarget::Kernel {
                    name,
                    size,
                    roofline,
                    metrics,
                },
        } = parse(&[
            "perf",
            "profile",
            "kernel",
            "--name",
            "gemm_q4k",
            "--size",
            "1024",
            "--roofline",
            "--metrics",
            "sm__cycles_elapsed",
        ])
        else {
            panic!("expected profile kernel");
        };
        assert_eq!(name, "gemm_q4k");
        assert_eq!(size, 1024);
        assert!(roofline);
        assert_eq!(metrics.as_deref(), Some("sm__cycles_elapsed"));
    }

    /// `bench --threshold` defaults to 5 and is overridable.
    #[test]
    fn bench_threshold_default_and_override() {
        let Commands::Bench { threshold, .. } = parse(&["perf", "bench", "--bench", "b"]) else {
            panic!("expected bench");
        };
        assert_eq!(threshold, 5.0);

        let Commands::Bench {
            threshold,
            counters,
            check_regression,
            roofline,
            bench,
        } = parse(&[
            "perf",
            "bench",
            "--bench",
            "b",
            "--threshold",
            "12.5",
            "--counters",
            "c1,c2",
            "--check-regression",
            "--roofline",
        ])
        else {
            panic!("expected bench");
        };
        assert_eq!(threshold, 12.5);
        assert_eq!(bench, "b");
        assert_eq!(counters.as_deref(), Some("c1,c2"));
        assert!(check_regression);
        assert!(roofline);
    }

    /// `contract verify` keeps every flag, spelled as the standalone binary
    /// actually spelled them.
    ///
    /// Note the flag is `--self-verify`, not `--self`: the field carries
    /// `#[arg(long, name = "self")]`, and in clap 4 `name` sets the argument
    /// *id*, not the long spelling, which is still derived from the field
    /// name. So `--self` never existed. That is pre-existing behaviour and is
    /// deliberately unchanged here — this test pins the spelling that works so
    /// a future edit cannot quietly move it.
    #[test]
    fn contract_verify_keeps_the_self_verify_flag() {
        let Commands::Contract {
            action:
                ContractAction::Verify {
                    self_verify,
                    fail_on_regression,
                    contracts_dir,
                    contract,
                },
        } = parse(&[
            "perf",
            "contract",
            "verify",
            "--self-verify",
            "--fail-on-regression",
            "--contracts-dir",
            "contracts/",
            "--contract",
            "c.yaml",
        ])
        else {
            panic!("expected contract verify");
        };
        assert!(self_verify);
        assert!(fail_on_regression);
        assert_eq!(contracts_dir.as_deref(), Some("contracts/"));
        assert_eq!(contract.as_deref(), Some("c.yaml"));
    }

    /// `compete --theirs` is repeatable; a single-value parse would silently
    /// drop every competitor after the first.
    #[test]
    fn compete_collects_every_theirs_occurrence() {
        let Commands::Compete {
            theirs,
            label,
            ours,
            workload,
        } = parse(&[
            "perf", "compete", "gemm", "--ours", "apr", "--theirs", "a", "--theirs", "b",
            "--label", "apr,a,b",
        ])
        else {
            panic!("expected compete");
        };
        assert_eq!(workload, "gemm");
        assert_eq!(ours, "apr");
        assert_eq!(theirs, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(label.as_deref(), Some("apr,a,b"));
    }

    /// `profile quant` requires either `--kernel`+`--size` or `--all`; a bare
    /// invocation must be refused rather than silently profiling a default.
    #[test]
    fn profile_quant_without_kernel_or_all_is_refused() {
        let err = Harness::try_parse_from(["perf", "profile", "quant"])
            .expect_err("`profile quant` with neither --kernel nor --all must be refused");
        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    }

    /// The global `--json` reaches a nested subcommand, which is how `cgp`'s
    /// own top-level `--json` behaved.
    #[test]
    fn global_json_propagates_into_nested_subcommands() {
        let parsed = Harness::try_parse_from([
            "perf",
            "profile",
            "scalar",
            "--function",
            "f",
            "--size",
            "8",
            "--json",
        ])
        .expect("nested --json parses");
        assert!(parsed.json);
    }

    /// `apr perf tui` advertises an interactive mode that does not exist.
    /// It must FAIL, not print a note and exit 0 (#2407).
    #[test]
    fn tui_reports_failure_instead_of_succeeding_silently() {
        let err = run_command(Commands::Tui, false)
            .expect_err("an unimplemented command must not return Ok");
        assert!(
            err.to_string().contains("not implemented"),
            "error must say it is unimplemented, got: {err}"
        );
    }

    /// Same for `profile library`, which echoed its arguments and exited 0.
    #[test]
    fn profile_library_reports_failure_instead_of_echoing_arguments() {
        let err = dispatch_profile(
            ProfileTarget::Library {
                so: "l.so".into(),
                symbol: "s".into(),
                args: None,
            },
            false,
        )
        .expect_err("an unimplemented profile target must not return Ok");
        assert!(
            err.to_string().contains("not implemented"),
            "error must say it is unimplemented, got: {err}"
        );
    }
}
