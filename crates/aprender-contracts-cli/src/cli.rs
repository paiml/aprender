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
    Validate {
        contract: PathBuf,
        /// Report obligation-id denominators instead of validating:
        /// `N obligations, M with id, K referenced`. Counts come through the
        /// same deserialization every other pv command uses, which is what
        /// makes them evidence that a consumer reads the key (#3314).
        #[arg(long)]
        check_ids: bool,
    },
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
    /// What the Lean proofs rest on: axiom subset pins and the compiler-escape allowlist (PVL-001 EV-6a, #4139)
    Discharge {
        #[command(subcommand)]
        action: DischargeAction,
    },
    /// Pin each contract-bound theorem's STATEMENT apart from its proof: `<lean>/Challenge/<contract>.lean`
    /// (PVL-001 EV-7a, #4200)
    Challenge {
        #[command(subcommand)]
        action: ChallengeAction,
    },
    /// Census the contract corpus: one cardinality, by_anchoring, by_entity_type (ONT-001 ONT-1)
    Census {
        /// Directory containing contract YAML files
        #[arg(default_value = "contracts")]
        contract_dir: PathBuf,
        /// Output format. ONT-001 §5 ONT-1's probe runs `--format json`.
        #[arg(long, value_enum, default_value_t = CensusFormat::Table)]
        format: CensusFormat,
        /// Deprecated alias for `--format json`, kept because
        /// scripts/check_ont_ratchet.sh derives its consumer probe from this surface.
        #[arg(long)]
        json: bool,
    },
    /// Extract the corpus as RDF: contracts.nt (sorted N-Triples, no blank nodes) and shapes.ttl (ONT-001 ONT-4b, R-15, R-18)
    Extract {
        /// Directory containing contract YAML files
        #[arg(default_value = "contracts")]
        contract_dir: PathBuf,
        /// Write nothing; exit 1 if the tracked files differ from a fresh extraction (what CI runs)
        #[arg(long)]
        check: bool,
        /// With `--release-*`: write the corpus graph PLUS the release evidence to this N-Triples file, and leave
        /// the tracked contracts.nt / shapes.ttl untouched (aprender#3715)
        #[arg(long)]
        out: Option<PathBuf>,
        #[command(flatten)]
        release: Box<ReleaseArgs>,
    },
    /// Σ as OWL and its advisory TBox (ONT-001 §3.8, ONT-2c)
    Ontology {
        #[command(subcommand)]
        command: OntologyCommand,
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
        /// No-op alias (PVL-001 EV-2): `--binding` now ALWAYS resolves every
        /// `implemented` binding against source with the `pv verify-bindings`
        /// resolver, lists ghosts under `GHOST BINDINGS (n)` and exits 1. Kept so
        /// existing invocations still parse; its root argument is ignored.
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
        /// Git ref whose `lint-baseline.json` is the `armed_gates` comparand (ONT-001 section 3.9). Default:
        /// merge-base(HEAD, origin/main), else the origin/main tip; with neither, NOT CHECKED is printed.
        #[arg(long)]
        armed_baseline_ref: Option<String>,
        /// Run ONE named gate and report only it (ONT-001 section 5 ONT-2b): `--gate sigma`. Repeatable
        /// (PVL-001 EV-11): every named gate runs and reports, and the exit is their meet — a refusal over a
        /// reject over a decline over a pass.
        #[arg(long)]
        gate: Vec<String>,
        /// With `--gate shapes`: grade only this shape family (the shape and every `<id>.*` shape), armed
        /// whatever `armed_shapes` says (aprender#3715: `--shape release-readiness-v1`).
        #[arg(long)]
        shape: Option<String>,
        #[command(flatten)]
        release: Box<ReleaseArgs>,
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
    /// Obligation gate (PVL-001 EV-10): every contract under ROOT/contracts validates, hides no
    /// test under `falsification:`, and binds each `applies_to` to a `fn` under ROOT/src that
    /// mentions the contract's `proved_type`. Replaces pmat's `scripts/pv-obligation-gate.py`.
    Obligations {
        /// Repository root holding `contracts/` and `src/`
        #[arg(default_value = ".")]
        root: PathBuf,
        /// Exit 1 (`reject:`) when any problem is found; without it the report exits 0
        #[arg(long)]
        gate: bool,
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

/// `pv discharge` actions (PVL-001 EV-6a, #4139).
#[derive(Subcommand, Clone, Debug)]
pub enum DischargeAction {
    /// Generate `<lean-dir>/Axioms.lean`: a subset axiom pin per contract-bound theorem in the root's import cone
    GenAxioms {
        /// The Lean dir (holds ProvableContracts.lean)
        lean_dir: PathBuf,
        /// Directory of the contracts whose `lean_theorem:` references bind the roots
        #[arg(long, default_value = "contracts")]
        contracts: PathBuf,
        /// Do not write: rc 1 when the tracked Axioms.lean differs from its regeneration
        #[arg(long)]
        check: bool,
    },
    /// Judge the tree: escapes vs escape-allowlist.yaml, exact-name roots, the label ratchet, Axioms.lean
    /// freshness, then `lake env lean Axioms.lean` (after `build.sh`) unless `--no-lake`
    Check {
        lean_dir: PathBuf,
        #[arg(long, default_value = "contracts")]
        contracts: PathBuf,
        /// Skip the Lean elaboration of Axioms.lean
        #[arg(long)]
        no_lake: bool,
        /// Allowlist entries still `confirmed_by: pending` are RED
        #[arg(long)]
        strict: bool,
        /// Also judge `<lean-dir>/formalization.yaml`: `main_results` listed by discharge-summary.json,
        /// `status.axioms` the pinned kernel set, `sorry_count` the measured count; a missing file is RED
        /// (PVL-001 EV-8b, #4082)
        #[arg(long)]
        validate_formalization: bool,
        /// Also re-check the BUILT tree's .olean files: `timeout <T> lake env leanchecker ProvableContracts`
        /// (non-fresh; `--fresh`, which replays Mathlib, is the nightly's, PVL-F7). rc != 0 rejects; no
        /// `leanchecker` in the toolchain declines (PVL-001 EV-6b, #4199)
        #[arg(long, conflicts_with = "no_lake")]
        leanchecker: bool,
        /// `--leanchecker`'s wall-clock limit, seconds
        #[arg(long, default_value_t = 3600, requires = "leanchecker")]
        leanchecker_timeout: u64,
        /// `--leanchecker` under `ulimit -v <KIB>` (virtual memory, KiB); unset = no limit
        #[arg(long, requires = "leanchecker")]
        leanchecker_ulimit_v: Option<u64>,
        /// Also run the comparator: `lake env lean --run scripts/Comparator.lean Challenge/*.lean` on the BUILT
        /// tree. Each EV-7a challenge must be closed by a sorry-free solution of the SAME statement (sha256 of
        /// the canonical type). No Challenge file, or zero rows, declines (PVL-001 EV-7b, #4201)
        #[arg(long, conflicts_with = "no_lake")]
        comparator: bool,
        /// Wall-clock limit on each `lake env lean` call (Axioms.lean, the comparator), seconds. A call that
        /// exceeds it is killed with its process group and rejects: a hang is RED, not a wait (#4239)
        #[arg(long, default_value_t = crate::commands::discharge::LAKE_TIMEOUT_S)]
        lake_timeout: u64,
    },
    /// `build.sh`, then `check` with every arm (`--strict`, the comparator, `--leanchecker`), then write the
    /// untracked full log `<lean-dir>/discharge.json` and the TRACKED `<lean-dir>/../discharge-summary.json` --
    /// on failure too. The Lean steps run only after `build.sh` exits 0 (PVL-001 EV-8a, #4202)
    Run {
        lean_dir: PathBuf,
        #[arg(long, default_value = "contracts")]
        contracts: PathBuf,
        /// The leanchecker arm's wall-clock limit, seconds
        #[arg(long, default_value_t = 3600)]
        leanchecker_timeout: u64,
        /// The leanchecker arm under `ulimit -v <KIB>` (virtual memory, KiB); unset = no limit
        #[arg(long)]
        leanchecker_ulimit_v: Option<u64>,
        /// Wall-clock limit on each `lake env` call outside the leanchecker arm, seconds; a timeout rejects (#4239)
        #[arg(long, default_value_t = crate::commands::discharge::LAKE_TIMEOUT_S)]
        lake_timeout: u64,
    },
    /// `make label-ratchet`: rewrite <lean-dir>/unresolved-labels.json DOWNWARD (it never gains a label; a missing
    /// file is seeded). `check` never writes it.
    LabelRatchet {
        lean_dir: PathBuf,
        #[arg(long, default_value = "contracts")]
        contracts: PathBuf,
    },
}

/// `pv challenge` actions (PVL-001 EV-7a, #4200).
#[derive(Subcommand, Clone, Debug)]
pub enum ChallengeAction {
    /// Write `<lean-dir>/Challenge/<contract>.lean`: every bound theorem restated as `PvlChallenge.<fqn>` with
    /// its proof replaced by `sorry`. Stale files are removed.
    Gen {
        /// Directory of the contracts whose `lean_theorem:` references bind the roots
        contracts: PathBuf,
        /// The Lean dir (holds ProvableContracts.lean)
        lean_dir: PathBuf,
    },
    /// Regenerate in memory and compare with `<lean-dir>/Challenge/`: rc 1 on any difference, rc 2 on zero
    /// challenges
    Check {
        contracts: PathBuf,
        lean_dir: PathBuf,
    },
}

/// `pv census` output format (ONT-001 ONT-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum CensusFormat {
    /// Human-readable table.
    Table,
    /// The bytes `contracts/census.json` carries.
    Json,
}

/// aprender#3715 — the release subject `extract:release-evidence` reads, for `pv lint --gate shapes` and
/// `pv extract`. All absent → no release graph. `--release-version` and `--release-commit` come together; the
/// rest need them.
#[derive(clap::Args, Debug, Default, Clone)]
pub struct ReleaseArgs {
    /// The release version whose evidence `release-readiness-v1` grades
    #[arg(long)]
    pub release_version: Option<String>,
    /// The release commit (MC), full 40-hex: the dogfood receipt's commit, and the receipts' unless --receipts-commit
    #[arg(long)]
    pub release_commit: Option<String>,
    /// T-4 only: the sha the committed receipts were measured at, after R7 proved the tree equal modulo evidence/
    #[arg(long)]
    pub receipts_commit: Option<String>,
    /// The per-host model receipts (default: evidence/dogfood/models/<version>/)
    #[arg(long)]
    pub receipts: Option<PathBuf>,
    /// The per-host kernel-diff receipts (default: evidence/dogfood/kernels/<version>/)
    #[arg(long)]
    pub kernel_receipts: Option<PathBuf>,
    /// The dogfood receipt R5 judged (without it the release has no dogfood receipt, which is a violation)
    #[arg(long)]
    pub dogfood_receipt: Option<PathBuf>,
    /// The tokenizer-parity receipts, apr vs the pinned llama.cpp (default: evidence/dogfood/tokenizer/<version>/)
    #[arg(long)]
    pub tokenizer_receipts: Option<PathBuf>,
}

impl ReleaseArgs {
    /// Did the caller pass any release flag at all?
    #[must_use]
    pub fn any(&self) -> bool {
        self.release_version.is_some()
            || self.release_commit.is_some()
            || self.receipts_commit.is_some()
            || self.receipts.is_some()
            || self.kernel_receipts.is_some()
            || self.dogfood_receipt.is_some()
            || self.tokenizer_receipts.is_some()
    }

    /// The subject, or `None` when no flag was passed. A partial set is refused, never completed by a default.
    pub fn subject(
        &self,
    ) -> Result<Option<provable_contracts::ontology::extract::release_inputs::Subject>, String>
    {
        use provable_contracts::ontology::extract::release_inputs::Subject;
        if !self.any() {
            return Ok(None);
        }
        let (Some(v), Some(c)) = (&self.release_version, &self.release_commit) else {
            return Err(
                "--release-version and --release-commit are both required with any --release-*, \
                 --receipts*, --kernel-receipts or --dogfood-receipt flag"
                    .into(),
            );
        };
        let mut s = Subject::new(v, c).map_err(|e| e.to_string())?;
        if let Some(rc) = &self.receipts_commit {
            s = s.with_receipts_commit(rc).map_err(|e| e.to_string())?;
        }
        s.receipts_dir.clone_from(&self.receipts);
        s.kernel_receipts_dir.clone_from(&self.kernel_receipts);
        s.dogfood_receipt.clone_from(&self.dogfood_receipt);
        s.tokenizer_receipts_dir
            .clone_from(&self.tokenizer_receipts);
        Ok(Some(s))
    }
}

/// `pv ontology …` (ONT-001 §3.8, row ONT-2c).
#[derive(Subcommand, Clone, Debug)]
pub enum OntologyCommand {
    /// Write Σ as OWL 2 EL functional syntax (the in-house writer; byte-deterministic)
    Export {
        /// Σ, the ontology declaration
        #[arg(default_value = "contracts/ontology.yaml")]
        sigma: PathBuf,
        /// OWL 2 functional syntax. The only format this command writes; required so the output is named
        #[arg(long)]
        owl: bool,
        /// Write `ontology.ofn` next to Σ instead of printing it
        #[arg(long)]
        write: bool,
    },
    /// The told-closure TBox report (advisory; `tbox-report.json`). Exit 3 if its precondition fails
    Tbox {
        /// Σ, the ontology declaration
        #[arg(default_value = "contracts/ontology.yaml")]
        sigma: PathBuf,
        /// Write `tbox-report.json` next to Σ instead of printing it
        #[arg(long)]
        write: bool,
    },
}
