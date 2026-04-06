# Introduction

**Papers to Math to Contracts in Code**

A Rust library and CLI for converting peer-reviewed research papers into
mathematically provable kernel implementations via YAML contract
intermediaries with Kani bounded model checking verification.

Available as:
- **Library** (`provable-contracts`): Contract parsing, validation, scaffold
  generation, Kani harness codegen, probar test generation
- **CLI** (`provable-contracts-cli`): 18 commands including `pv validate`,
  `pv lint`, `pv score`, `pv query`, `pv lean-status`, `pv proof-status`

Primary consumers: [aprender](https://github.com/paiml/aprender),
[entrenar](https://github.com/paiml/entrenar),
[realizar](https://github.com/paiml/realizar), and
[trueno](https://github.com/paiml/trueno) — all at Level 3 compile-time
enforcement via `build.rs` binding verification.

---

## Project Structure (Library + CLI)

### Crate Layout

```
provable-contracts/
├── Cargo.toml                  # workspace root
├── crates/
│   ├── provable-contracts/         # library crate
│   │   └── src/
│   │       ├── lib.rs              # public API
│   │       ├── schema/             # YAML contract parser + validator
│   │       ├── scaffold/           # Rust trait codegen
│   │       ├── kani_gen/           # Kani harness codegen
│   │       ├── probar/             # probar test codegen
│   │       ├── audit.rs            # traceability audit
│   │       ├── scoring.rs          # 5-dimension quality scoring
│   │       ├── lint.rs             # quality gate (validate+audit+score)
│   │       ├── query.rs            # BM25 semantic search
│   │       └── lean.rs             # Lean 4 codegen
│   ├── provable-contracts-cli/     # binary crate (`pv`)
│   │   └── src/
│   │       ├── main.rs
│   │       └── commands/           # 18 CLI commands
│   └── provable-contracts-macros/  # proc-macro crate
├── contracts/                      # 164 YAML kernel contracts
│   ├── softmax-kernel-v1.yaml      # 107 core kernel contracts
│   ├── aprender/                   # 8 aprender-specific + binding.yaml
│   ├── entrenar/                   # 37 training contracts + binding.yaml
│   ├── realizar/                   # binding.yaml (23 bindings)
│   ├── trueno/                     # 7 SIMD contracts + binding.yaml
│   └── forjar/                     # 5 forjar contracts + binding.yaml
├── lean/                           # Lean 4 proofs (14/14 proved)
├── docs/specifications/            # canonical spec + sub-specs
├── book/                           # mdBook documentation
└── .pmat/project.toml              # pmat compliance config
```

### Library API (provable-contracts crate)

```rust
// Schema — Parse and validate YAML contracts
pub fn parse_contract(path: &Path) -> Result<Contract, SchemaError>;
pub fn validate_contract(contract: &Contract) -> Vec<Violation>;

// Codegen — Generate Rust code from contracts
pub fn generate_trait(contract: &Contract) -> TokenStream;
pub fn generate_contract_tests(contract: &Contract) -> TokenStream;
pub fn generate_kani_harnesses(contract: &Contract) -> TokenStream;
pub fn generate_probar_tests(contract: &Contract) -> TokenStream;

// Codegen — Generate debug_assert!() from contracts
pub fn generate_all(contract_dir: &Path) -> Vec<GeneratedContract>;
pub fn write_rust_module(contracts: &[GeneratedContract], output: &Path) -> io::Result<()>;

// Audit — Trace paper→code chain
pub fn audit_contract(contract: &Contract) -> AuditReport;

// Scoring — Five-dimension quality metric
pub fn score_contract(contract: &Contract, binding: Option<&BindingRegistry>, stem: &str) -> ContractScore;

// Lint — Quality gate (validate + audit + score)
pub fn run_lint(config: &LintConfig) -> LintReport;
```

### CLI Commands (pv binary)

```
pv validate <contract.yaml>     Validate YAML against contract schema
pv lint <contracts-dir/>        Quality gate: validate + audit + score
pv audit <contract.yaml>        Traceability audit (paper → proof chain)
pv score <path>                 Score contracts or codebase (A-F grades)
pv query "softmax"              Semantic search across contracts
pv scaffold <contract.yaml>     Generate Rust trait + test stubs
pv kani <contract.yaml>         Generate #[kani::proof] harnesses
pv probar <contract.yaml>       Generate probar property tests
pv lean <contract.yaml>         Generate Lean 4 theorem stubs
pv lean-status <contracts-dir/> Report Lean 4 proof status (14/14 proved)
pv proof-status <contracts-dir/> Hierarchical proof level (L1-L5) report
pv status <contract.yaml>       Show contract summary
pv diff <old.yaml> <new.yaml>   Diff versions, suggest semver bump
pv coverage <contracts-dir/>    Cross-contract obligation coverage
pv generate <contract.yaml>     Write all codegen artifacts to disk
pv graph <contracts-dir/>       Dependency DAG (text/dot/json/mermaid)
pv equations <contract.yaml>    Render equations (text/latex/ptx/asm)
pv book <contracts-dir/>        Generate mdBook pages
```

---

## Work Tracking

All work is tracked via `pmat work` tickets. Use `pmat work list` to see
current open items.
