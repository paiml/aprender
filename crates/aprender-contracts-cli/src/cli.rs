use std::path::PathBuf;

use clap::Subcommand;

/// Available subcommands for the `pv` CLI
#[derive(Subcommand, Clone, Debug)]
pub enum Commands {
    /// Explain a contract in detail
    Explain {
        contract: PathBuf,
        #[arg(long, default_value = "text")]
        format: String,
        #[arg(long)]
        binding: Option<PathBuf>,
    },
    /// Validate a YAML kernel contract
    Validate { contract: PathBuf },
    /// Execute cross_check_command per row of a parity-matrix contract (SEMANTIC gate)
    #[command(name = "check-parity")]
    CheckParity { contract: PathBuf },
    /// Generate Rust trait + test scaffolding from a contract
    Scaffold {
        contract: PathBuf,
        #[arg(long)]
        r#trait: bool,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Extract kernel equations from `PyTorch` source into YAML
    #[command(name = "extract-pytorch")]
    ExtractPytorch {
        target: String,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Generate Rust `debug_assert!()` from YAML contracts
    Codegen {
        #[arg(default_value = "contracts")]
        contract_dir: PathBuf,
        /// Output Rust file path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Generate Kani proof harnesses from a contract
    Kani { contract: PathBuf },
    /// Generate probar property tests from a contract
    Probar {
        contract: PathBuf,
        /// Path to binding registry YAML (generates wired tests)
        #[arg(long)]
        binding: Option<PathBuf>,
    },
    /// Show contract status (equations, obligations, coverage)
    Status {
        /// Path to the contract YAML file
        contract: PathBuf,
    },
    /// Run traceability audit on a contract
    Audit {
        /// Path to the contract YAML file
        contract: PathBuf,
        /// Path to binding registry YAML (adds binding audit)
        #[arg(long)]
        binding: Option<PathBuf>,
        /// Show Coq proof tier per obligation
        #[arg(long)]
        coq: bool,
        /// Show Flux shape coverage per obligation
        #[arg(long)]
        flux: bool,
    },
    /// Diff two contract versions and suggest semver bump
    Diff {
        /// Path to the old contract YAML file
        old: PathBuf,
        /// Path to the new contract YAML file
        new: PathBuf,
    },
    /// Classify contracts by whether their claims can be refuted, and ratchet
    /// the count that cannot
    ///
    /// `pv lint` reports 0 errors on a tree where 759 of 1726 contracts carry
    /// zero falsification tests, because PROVABILITY-001 applies only to
    /// non-registry `kind: kernel` contracts. Most of those 759 are legitimate
    /// catalogs. This command separates the catalogs from the contracts that
    /// assert something and ship no way to refute it, and pins the latter.
    Inert {
        /// Directory containing contract YAML files
        #[arg(default_value = "contracts")]
        contract_dir: PathBuf,
        /// Output format: text (default) or json
        #[arg(long, default_value = "text")]
        format: String,
        /// Ratchet: exit non-zero when the inert count exceeds this ceiling
        #[arg(long)]
        max: Option<usize>,
        /// List every inert contract with its trigger fields
        #[arg(long)]
        list: bool,
        /// Run the classifier's own must-match/must-not-match case table
        #[arg(long = "self-test")]
        self_test: bool,
    },
    /// Show cross-contract obligation coverage report
    Coverage {
        /// Directory containing contract YAML files
        #[arg(default_value = "contracts")]
        contract_dir: PathBuf,
        /// Path to binding registry YAML (adds binding coverage)
        #[arg(long)]
        binding: Option<PathBuf>,
        /// Include fuzz coverage data
        #[arg(long)]
        fuzz: bool,
        /// Reverse coverage: scan crate dir for unbound pub fns
        #[arg(long)]
        reverse: Option<PathBuf>,
        /// Enforcement quality: scan crate source for contract call sites and classify E0/E1/E2
        #[arg(long)]
        enforcement: Option<PathBuf>,
    },
    /// Generate all artifacts (scaffold, kani, probar) to disk
    Generate {
        /// Path to the contract YAML file
        contract: PathBuf,
        /// Output directory for generated files
        #[arg(short, long, default_value = "generated")]
        output: PathBuf,
        /// Path to binding registry YAML (generates wired tests)
        #[arg(long)]
        binding: Option<PathBuf>,
        /// Generate CONTRACT-README.md (requires --binding)
        #[arg(long)]
        readme: bool,
        /// Generate .github/workflows/contracts.yml
        #[arg(long)]
        ci: bool,
    },
    /// Show contract dependency graph
    Graph {
        /// Directory containing contract YAML files
        #[arg(default_value = "contracts")]
        contract_dir: PathBuf,
        /// Output format: text (default), dot, json, or mermaid
        #[arg(long, default_value = "text")]
        format: String,
    },
    /// Display equations from a contract
    Equations {
        contract: PathBuf,
        #[arg(long, default_value = "text")]
        format: String,
    },
    /// Generate Lean 4 definitions and theorem stubs
    Lean {
        contract: PathBuf,
        #[arg(long)]
        output_dir: Option<PathBuf>,
    },
    /// Report Lean 4 proof status across contracts
    LeanStatus {
        /// Path to a contract YAML file or directory of contracts
        #[arg(default_value = "contracts")]
        path: PathBuf,
    },
    /// Report hierarchical proof levels (L1–L5) across contracts
    ProofStatus {
        /// Path to a contract YAML file or directory of contracts
        #[arg(default_value = "contracts")]
        path: PathBuf,
        /// Path to binding registry YAML (adds binding coverage)
        #[arg(long)]
        binding: Option<PathBuf>,
        /// L5 gate: before counting a binding as implemented, verify its
        /// `function` actually exists in source (scanned from the given root,
        /// default `.`). Phantom "implemented" bindings are downgraded, so L5
        /// means "verified as implemented", not self-declared.
        #[arg(long, num_args = 0..=1, default_missing_value = ".")]
        verify_bindings: Option<PathBuf>,
        /// Output format: text (default) or json
        #[arg(long, default_value = "text")]
        format: String,
        /// Show per-obligation verification table
        #[arg(long)]
        table: bool,
        /// Filter: kernel|registry|model-family|pattern|schema
        #[arg(long)]
        kind: Option<String>,
    },
    /// Run all contract quality gates (validate + audit + score)
    Lint {
        /// Directory containing contract YAML files
        #[arg(default_value = "contracts")]
        contract_dir: PathBuf,
        /// Minimum composite score threshold (default: 0.0 = no score gate)
        #[arg(long, default_value = "0.0")]
        min_score: f64,
        /// Path to binding registry YAML
        #[arg(long)]
        binding: Option<PathBuf>,
        /// Output format: text (default), json, sarif, github
        #[arg(short, long)]
        format: Option<String>,
        /// Minimum severity to report: error, warning, info
        #[arg(long)]
        severity: Option<String>,
        /// Promote warnings to errors
        #[arg(long)]
        strict: bool,
        /// Suppress specific finding IDs (comma-separated)
        #[arg(long)]
        suppress: Option<String>,
        /// Suppress all findings for a rule (comma-separated)
        #[arg(long)]
        suppress_rule: Option<String>,
        /// Suppress all findings matching a file path (comma-separated)
        #[arg(long)]
        suppress_file: Option<String>,
        /// Override rule severity (e.g. PV-AUD-001=info)
        #[arg(long)]
        rule: Vec<String>,
        /// Path to .pv.toml config file
        #[arg(long)]
        config: Option<PathBuf>,
        /// Only lint contracts changed since base ref (e.g. main, HEAD~5)
        #[arg(long = "diff")]
        diff_ref: Option<String>,
        /// Record quality trend snapshot
        #[arg(long)]
        trend: bool,
        /// Show quality trend history
        #[arg(long)]
        show_trend: bool,
        /// Bypass lint cache
        #[arg(long)]
        no_cache: bool,
        /// Show cache hit/miss statistics
        #[arg(long)]
        cache_stats: bool,
        /// Show auto-fix suggestions (dry run)
        #[arg(long)]
        suggest: bool,
        /// Suppress findings in baseline SARIF file
        #[arg(long)]
        baseline: Option<PathBuf>,
        /// Apply deterministic auto-fixes
        #[arg(long)]
        fix: bool,
        /// Re-lint on file change (polling)
        #[arg(long)]
        watch: bool,
        /// Show aggregate contract coverage metric
        #[arg(long)]
        coverage: bool,
        /// Minimum coverage percentage (exit 1 if below)
        #[arg(long)]
        min_coverage: Option<f64>,
        /// Path to crate directory for reverse coverage gate
        #[arg(long)]
        crate_dir: Option<PathBuf>,
        /// Minimum enforcement level: basic, standard, strict, proven
        #[arg(long)]
        min_level: Option<String>,
        /// Explain a lint rule in detail (e.g. PV-ENF-001)
        #[arg(long)]
        explain: Option<String>,
        /// Enable strict test-binding gate (PV-VER-002): cross-checks every
        /// `falsification_tests[].test` cargo invocation against `#[test]` fns
        /// in the source tree. Catches drift classes that PV-VER-001 misses
        /// (suffix drift, module-path drift, convention drift, "or equivalent"
        /// placeholders). Default emits Warning; combine with `--strict` to
        /// promote to Error and fail CI. Issue #1510.
        #[arg(long)]
        strict_test_binding: bool,
    },
    /// Score contracts or a codebase directory
    Score {
        /// Path to a contract YAML file or directory of contracts
        #[arg(default_value = "contracts")]
        path: PathBuf,
        /// Path to binding registry YAML
        #[arg(long)]
        binding: Option<PathBuf>,
        /// Output format: text (default) or json
        #[arg(short, long, default_value = "text")]
        format: String,
        /// Minimum score threshold (exit 1 if below)
        #[arg(long)]
        min_score: Option<f64>,
        /// Show aggregate summary only (no per-contract detail)
        #[arg(long)]
        summary: bool,
        /// Show top N gaps by impact (default: 5)
        #[arg(long, default_value = "5")]
        top_gaps: usize,
        /// Custom weights as JSON
        #[arg(long)]
        weights: Option<String>,
        /// Exit with status 1 if any contract below --min-score
        #[arg(long)]
        exit_code: bool,
        /// Show 10-dimension `PVScore` (geometric mean)
        #[arg(long)]
        pvscore: bool,
    },
    /// Search contracts by intent, regex, or literal match
    Query(crate::query_args::QueryArgs),
    /// Generate type invariant trait + Kani preservation harnesses
    Invariants { contract: PathBuf },
    /// Generate Coq theorem stubs from a contract
    Coq { contract: PathBuf },
    /// Generate libfuzzer fuzz targets from a contract
    Fuzz { contract: PathBuf },
    /// Generate MIRAI annotations from a contract
    Mirai { contract: PathBuf },
    /// Generate Flux refinement types from a contract
    Flux { contract: PathBuf },
    /// Generate TLA+ specification from contract dependency DAG
    Tla {
        /// Directory containing contract YAML files
        #[arg(default_value = "contracts")]
        contract_dir: PathBuf,
    },
    /// Generate mdBook pages for contracts
    Book {
        /// Directory containing contract YAML files
        #[arg(default_value = "contracts")]
        contract_dir: PathBuf,
        /// Output directory for generated pages
        #[arg(short, long, default_value = "book/src/contracts")]
        output: PathBuf,
        /// Also update book/src/SUMMARY.md with contract links
        #[arg(long)]
        update_summary: bool,
        /// Path to SUMMARY.md (default: book/src/SUMMARY.md)
        #[arg(long)]
        summary_path: Option<PathBuf>,
    },
    /// Infer contracts and bindings for unbound functions in a crate
    Infer {
        /// Path to the crate directory to scan
        crate_dir: PathBuf,
        /// Path to binding registry YAML
        #[arg(long)]
        binding: PathBuf,
        /// Directory containing contract YAML files
        #[arg(long, default_value = "contracts")]
        contract_dir: PathBuf,
        /// Maximum number of suggestions to show
        #[arg(long, default_value = "20")]
        top: usize,
    },
    /// Remove enforcement level lock from a contract (requires --reason)
    Unlock {
        /// Path to the contract YAML file
        contract: PathBuf,
        /// Mandatory reason for unlocking (audit trail)
        #[arg(long)]
        reason: String,
    },
    /// Compute roofline ceilings from contract equations
    Roofline {
        #[arg(long, default_value = "contracts")]
        contract_dir: PathBuf,
        /// Total model parameters (e.g. 7000000000 for 7B)
        #[arg(long)]
        params: u64,
        /// Bits per weight (2, 4, 8, 16, 32)
        #[arg(long, default_value = "4")]
        bits: u32,
        /// Hardware profile: apple-m, a100
        #[arg(long, default_value = "apple-m")]
        hardware: String,
        /// Output format: text (default) or json
        #[arg(short, long, default_value = "text")]
        format: String,
    },
    /// Validate a pipeline contract (cross-repo verification)
    Pipeline {
        /// Path to the pipeline YAML file
        pipeline: PathBuf,
        /// Output format: text (default) or json
        #[arg(short, long, default_value = "text")]
        format: String,
    },
    /// Fleet-wide contract enforcement (kaizen loop)
    Kaizen {
        #[arg(long, default_value = "contracts")]
        contract_dir: PathBuf,
        #[arg(long)]
        src_root: Option<PathBuf>,
        #[arg(long)]
        repo: Option<String>,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        codegen: bool,
        #[arg(long)]
        fix: bool,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        min_score: Option<f64>,
    },
    /// Produce whole-model proof certificate (runs verify-pipeline + verify-structure)
    Certify {
        #[arg(default_value = "contracts")]
        contract_dir: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Verify model architecture structure matches contracts
    #[command(name = "verify-structure")]
    VerifyStructure {
        #[arg(default_value = "contracts")]
        contract_dir: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        model: Option<PathBuf>,
    },
    /// Verify compositional shape flow across contract dependency graph
    #[command(name = "verify-pipeline")]
    VerifyPipeline {
        #[arg(default_value = "contracts")]
        contract_dir: PathBuf,
        #[arg(long, default_value = "text")]
        format: String,
    },
    /// Generate a Rust test that verifies all bound functions exist
    VerifyBindings {
        /// Path to binding.yaml
        binding: PathBuf,
        /// Output file path (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Crate name for test label
        #[arg(long)]
        crate_name: Option<String>,
    },
    /// Migrate old-format contract YAMLs to current schema (GH-67)
    Migrate {
        #[arg(default_value = "contracts")]
        contract_dir: PathBuf,
        #[arg(long)]
        dry_run: bool,
    },
}
