# Phase 5: Benchmark and Claims Gate - Pattern Map

> **RECONCILED 2026-09-08 against `05-CONTEXT.md` D-19 — the 9B LoRA arm is DESCOPED.**
> Every claim below about the 40 LoRA cells, the lambda-vector GPU host, or a
> SetFit-versus-LoRA comparison is **superseded**: the host is unreachable and aprender
> cannot run the Qwen3.5-9B hybrid architecture. EVAL-02/EVAL-04 and the Phase 5 goal were
> amended accordingly; the arm is deferred as `D-ITEM-05-15`. Findings about the SetFit half,
> the numerics substrate, the row schema and the claims gate are UNAFFECTED and still hold.
> This file was reconciled in place rather than regenerated — the descope narrows scope, it
> does not invalidate the surviving research.


**Mapped:** 2026-08-16
**Files analyzed:** 16 new/modified surfaces
**Analogs found:** 13 exact / 16 (3 with no in-repo analog — see "No Analog Found")

All excerpts read from the tree this session. Every path is repo-root-relative; every line
number verified against HEAD (`gsd/phase-2-contract-gate`).

## File Classification

| New/Modified Surface | Role | Data Flow | Closest Analog | Match |
|---|---|---|---|---|
| `crates/aprender-train/src/train/setfit/thresholds.rs` (EDIT: per-regime ε tables, 2nd entry, test update) | gate/config constant module | contract-parse + validate | itself — the file IS the house pattern | exact (self) |
| `crates/aprender-train/src/train/setfit/evidence.rs` (EDIT: production calibration harness) | `#[ignore]`d test harness | batch measurement | `calibration_matrix_epsilon_basis` (`evidence.rs:1591`) | exact (self) |
| `crates/aprender-train/src/train/setfit/bench_row.rs` (NEW: D-12 row type + run manifest) | model/schema | file I/O (hashed serde JSON) | `SelectionManifest` (`aprender-contrastive-data/src/manifest.rs:169`) + `EvalRow` (`apr-cli/src/commands/eval/setfit.rs:111`) | exact |
| `crates/aprender-core/src/calibration.rs` (EDIT: top-label ECE + multiclass Brier) | utility/metrics | transform | binary `expected_calibration_error` (`calibration.rs:138`), `brier_score` (`:378`) | exact |
| `crates/aprender-core/src/stats/hypothesis.rs` (EDIT: f64 paired-CI helper) | utility/stats | transform | `ttest_rel` (`hypothesis.rs:191`), `ttest_1samp` (`:81`) | exact |
| `crates/apr-cli/src/setfit_commands.rs` (EDIT: `Bench { Run, Report }` variants) | CLI enum (clap) | request-response dispatch | `SetfitCommands::Train` (`setfit_commands.rs:16-101`) | exact |
| `crates/apr-cli/src/commands/setfit_bench.rs` (NEW: filesystem adapter) | CLI command adapter | file I/O | `commands/setfit_train.rs` + `commands/data_contrastive.rs` | exact |
| `crates/apr-cli/src/dispatch_analysis.rs` (EDIT: Bench dispatch arms) | dispatch | request-response | `dispatch_setfit_command` (`dispatch_analysis.rs:832-860`) | exact |
| `crates/apr-cli/src/model_ops_commands.rs` + `dispatch.rs` + `commands/finetune.rs` (EDIT: `--selection-manifest` + explicit `TrainingConfig`) | CLI command | training pipeline | `run_classify` (`finetune.rs:1563`) + eval's manifest→Selection door (`eval/setfit.rs:227-237`) | exact |
| `contracts/setfit-benchmark-claims-v1.yaml` (NEW) | contract | declarative | `tweet-eval-stance-benchmark-v1.yaml` (header + equations style) | exact |
| `contracts/setfit-train-lifecycle-v1.yaml` (EDIT: additive regime entry + per-regime thresholds block) | contract | declarative | its own `frozen_thresholds`/`calibration_regime` machine-readable blocks | exact (self) |
| `scripts/setfit_fixtures/` stats-fixture generator (NEW/EDIT) | script | batch file I/O | `generate_fixtures.py::write_manifest` (`:922-949`) | exact |
| `Makefile` (EDIT: `$(CONTRACTS)` append, `contract-audit-phase5`, bench test targets) | build config | — | `CONTRACTS` list (`:1739`), `setfit-tests` recipe (`:2122-2168`), `PHASE2_CONTRACTS` narrowing (`:1792`) | exact |
| `crates/aprender-train/src/train/setfit/apr_evaluate.rs` (EDIT: per-row-predictions door) | service/evaluator | request-response | `evaluate_validation_from_artifact` (`apr_evaluate.rs:185`) | exact |
| spawned bench-cell evidence test (NEW) | integration test (spawned-tier) | process orchestration | `crates/apr-cli/tests/setfit_cli_lifecycle.rs` (`:1-90`) | exact |
| stats/calibration reference tests (NEW) | fixture-parity test | — | `crates/aprender-core/src/glm/glm_tests.rs:274-298` (FALSIFY + recorded RED value) | exact |

## Pattern Assignments

### 1. `thresholds.rs` restructuring (gate/config, contract-parse)

**Analog:** the file itself — `crates/aprender-train/src/train/setfit/thresholds.rs`.
This is the highest-risk edit (D-02/D-03/D-04); the pattern the edit must PRESERVE is
the three-place synchronization.

**The constant the edit extends** (`thresholds.rs:61-62`):
```rust
pub(crate) const CALIBRATED_REGIMES: &[&str] =
    &["minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds=1,42,7|cells=s16e2b8,s8e1b4"];
```

**The global table D-03 must make per-regime** (`thresholds.rs:273-300`, `Thresholds::frozen()`):
```rust
let mut classes = BTreeMap::new();
classes.insert(
    ParameterClass::Embedding.tag(),
    ClassThreshold { eps: Some(1.1e-5), scale_floor: 1.0, sparse: true, gated: true },
);
// ... layer_norm_weight 2.9e-6, layer_norm_bias 1.1e-5, projection_weight 1.8e-5,
//     projection_bias 8.3e-6, attention_key_bias { eps: None, gated: false } ...
Self { classes, embedding_delta_floor: 2.7e-5, calibrated_regimes: CALIBRATED_REGIMES }
```
Today `Thresholds` holds ONE class table + ONE floor for the whole calibrated set. The
restructuring maps regime → table (fixture table byte-identical) and threads the regime
into `validate_evidence`'s threshold lookup (`tune.rs:1139`, order-of-checks item 1).

**The pinned test whose assertions change deliberately** (`thresholds.rs:394-491`,
`thresholds_match_the_contract`):
```rust
// :400-405 — non-vacuity FIRST (keep this shape for the per-regime blocks)
assert_eq!(
    parsed.equations.evidence_gate.frozen_thresholds.len(),
    ParameterClass::ALL.len(),
    "the contract must carry one entry per class; ...",
);
// :451-457 — the ONLY assertion whose literal changes (1 → 2):
assert_eq!(contracted_regimes.len(), 1,
    "exactly ONE calibrated fingerprint; a second would mean numbers measured on one \
     architecture are being applied to another");
// :474-481 — the CR-04 protection that must SURVIVE as full-list equality:
let rust_regimes: Vec<&str> = frozen.calibrated_regimes().to_vec();
let contract_regimes: Vec<&str> = contracted_regimes.iter().map(String::as_str).collect();
assert_eq!(rust_regimes, contract_regimes,
    "the Rust calibrated regime set and the contract's must be EQUAL, not merely \
     overlapping. ... (REVIEW CR-04)");
```
The `#[cfg(test)]` deserialization targets to extend for per-regime thresholds:
`ContractFile`/`ContractEquations`/`EvidenceGateEquation`/`CalibrationRegimeEquation`
(`thresholds.rs:349-382`). The contract is pinned via `include_str!`
(`thresholds.rs:47-49`) — keep that, never a runtime read.

**Membership stays component-wise** — `RegimeCoordinates::parse`/`covers`
(`thresholds.rs:107-159`); the negative case table pattern to extend is
`regime_membership_is_component_wise` (`thresholds.rs:617-656`). NOTE the invented
`minilm-full-...@production` string at `:643-650` is NOT a rendering — the production
entry must be copied byte-for-byte from a measured run's own `calibration_regime_id`
(the fingerprint prefix is hardcoded `minilm-slice-` even for the full model,
`aprender-core/src/setfit/encoder.rs:303-311`).

---

### 2. Production calibration harness (test harness, batch measurement)

**Analog:** `calibration_matrix_epsilon_basis`,
`crates/aprender-train/src/train/setfit/evidence.rs:1580-1719` (same file gains the
production sibling).

**Gating/invocation pattern** (`evidence.rs:1580-1592`):
```rust
/// THE calibration matrix — `#[ignore]`d, and deliberately so.
/// Invoke with:
/// `cargo test -p aprender-train --lib --features setfit calibration_matrix -- --ignored --nocapture`
#[test]
#[ignore = "12 full tuning passes; run explicitly with --ignored (plan 03-05 epsilon basis)"]
fn calibration_matrix_epsilon_basis() {
```

**Condition constants** (`evidence.rs:1563,1578`):
```rust
const CONTROL_LR: f64 = 1e-30;   // numerical null — bit-identical zero deltas
const NEAR_NULL_LR: f64 = 1e-8;  // the REAL lower bound: AdamW still writes a different f32
```

**Per-cell loop + per-class aggregation + separation assertion** (`evidence.rs:1622-1691`):
```rust
for variant in &variants {
    let real = evidence_for(*variant, None);
    let control = evidence_for(*variant, Some(CONTROL_LR));
    let near_null = evidence_for(*variant, Some(NEAR_NULL_LR));
    // ... per ParameterClass: real_min/median/max, ctrl_max, near_null_max + moved,
    //     rounding_noise_floor, support fraction, BINDING PARAMETER NAME ...
    assert!(control_max < real_min,
        "seed {} cell {} class {}: the 1e-30 control's max relative delta \
         ({control_max:e}) is not below the real run's min ({real_min:e})", ...);
}
```
The harness prints the regime id (`evidence.rs:1598` — `report.push_str(&format!("\nCALIBRATION REGIME: {}\n", regime_id()))`);
that printed id is what the contract entry copies. Measurement needs NO gate widening:
the `UncalibratedRegime` refusal lives in `validate_evidence` (judgement time), not in
`run_tuning`/`from_tune_output` (`evidence.rs:416`).

**Env-gating for the production checkout:** parameterize on `APRENDER_MINILM_DIR` with the
`~/.cache/aprender/minilm-l6-v2-1110a243` default, per
`crates/aprender-train/tests/full_weight_parity.rs:54`.

---

### 3. Row type + run manifest (`bench_row.rs`, model/schema, hashed file I/O)

**Analog A — the digest-committing envelope:**
`crates/aprender-contrastive-data/src/manifest.rs`.

Envelope shape (`manifest.rs:169-178`):
```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionManifest {
    /// Lowercase hex of `SHA-256(payload.to_canonical_bytes())`.
    pub semantic_hash: String,
    /// Volatile metadata. NEVER part of the digest.
    pub volatile: VolatileMetadata,
    /// The hashed payload.
    pub payload: SelectionPayload,
}
```

Verify-before-return (`manifest.rs:248-272`) — the property D-14's row/manifest readers
must copy: a forged artifact is unrepresentable downstream:
```rust
pub fn from_bytes(bytes: &[u8]) -> Result<Self, ContrastiveDataError> {
    let manifest: Self = serde_json::from_slice(bytes).map_err(|error| ...)?;
    manifest.verify_digest()?;   // SHA-256 over payload.to_canonical_bytes(), compared
    Ok(manifest)
}
```
Also copy: schema_version field + refusal on mismatch (`manifest.rs:313-316, 401-404`),
pretty-JSON-on-disk / compact-canonical-bytes-for-digest split (`to_file_bytes`,
`manifest.rs:228-237`), and the on-disk write via ONE serializer.

**Analog B — the row field vocabulary:** `EvalRow` in
`crates/apr-cli/src/commands/eval/setfit.rs:111-143`, written explicitly as "The
machine-readable row Phase 5 consumes (EVAL-03 shape)":
```rust
#[derive(Debug, Serialize)]
struct EvalRow {
    split: &'static str,
    metric: String,
    value: f64,
    value_bits: u64,        // IEEE-754 bits — the lock hashes bits, a decimal is a rendering
    n_rows: usize,
    artifact_sha256: String,
    dataset_fingerprint: String,
    validation_split_fingerprint: String,
    ordered_labels: Vec<String>,
    evidence_table_hash: Option<String>,
    selection_root_seed: Option<u64>,
    config_hash_derivation: &'static str,   // stated so Phase 5 can reproduce it
    candidates: Vec<CandidateRow>,
    lock: Option<LockRow>,
    notes: Vec<String>,
}
```
`LockRow` (`:158-171`) carries `lock_hash`, `chosen_artifact_sha256`, `rule`,
`role: "written"|"consumed"` — the lock-reference field D-12's shared core needs.
The D-12 row differs in two contracted ways: it is `Deserialize` + `deny_unknown_fields`
(EvalRow is serialize-only), and it adds the method-tagged evidence block. Backend
identity comes from execution: `ExecutionBackend::identity()` renders
`<device>:setfit-core:<kernel>` (`crates/aprender-core/src/setfit/encoder.rs:227-238`);
LoRA rows use the pipeline's executed GPU identity
(`pipeline.gpu_name().zip(pipeline.gpu_total_memory())`, `finetune.rs:1629`).

---

### 4. Multiclass ECE + Brier (`aprender-core/src/calibration.rs`, utility, transform)

**Analog:** the binary pair in the same file — site the multiclass functions beside them.

Contract binding + binning style to mirror (`calibration.rs:138-164`):
```rust
#[provable_contracts_macros::contract("calibration-v1", equation = "expected_calibration_error")]
#[must_use]
pub fn expected_calibration_error(predictions: &[f32], labels: &[bool], n_bins: usize) -> f32 {
    contract_pre_expected_calibration_error!(predictions);
    // equal-width bins over [0,1]:
    let bin = ((pred * n_bins as f32) as usize).min(n_bins - 1);
    // ...
    ece += (bin_counts[i] as f32 / n) * (avg_conf - avg_acc).abs();
```
Top-label ECE: `conf_i = max_k p_ik`, `pred_i = argmax_k p_ik`, same bin indexing.

Binary Brier to generalize (`calibration.rs:378-390`):
```rust
pub fn brier_score(predictions: &[f32], labels: &[bool]) -> f32 {
    // (1/n) * Σ(p_i - y_i)²
```
Multiclass form `BS = (1/N) Σ_i Σ_k (p_ik − y_ik)²` — the in-tree formula reference is
`compute_brier_score` at
`crates/aprender-train/src/finetune/classify_eval_report.rs:316`, which is UNfixtured
and sits beside bootstrap CIs: use it as a formula cross-check only, NEVER consume
`ClassifyEvalReport` in the claims path (D-06, RNG). Tests land in the sibling files
per house convention (`calibration.rs:392-398` — `calibration_tests.rs`,
`calibration_tests_contract.rs`).

---

### 5. Paired-t CI helper (`aprender-core/src/stats/hypothesis.rs`, utility, transform)

**Analog:** `ttest_rel` → `ttest_1samp` in the same file.

The shape to extend (`hypothesis.rs:191-208`):
```rust
pub fn ttest_rel(sample1: &[f32], sample2: &[f32]) -> Result<TTestResult> {
    if sample1.len() != sample2.len() {
        return Err(AprenderError::DimensionMismatch { ... });
    }
    let diffs: Vec<f32> = sample1.iter().zip(sample2.iter()).map(|(&x1, &x2)| x1 - x2).collect();
    ttest_1samp(&diffs, 0.0)
}
```
`ttest_1samp` (`:81-115`) is the closed-form core: (n−1) sample std, `se = std/√n`,
`t = (x̄ − μ₀)/se`, df = n−1. The exact p-value machinery already exists —
`t_distribution_pvalue` (`:360-365`), `p = I_x(df/2, 1/2)` with `x = df/(df+t²)` via
log-space `incomplete_beta` (`:428-454`) — scipy-oracle tested in
`tests_hypothesis_contract.rs` (included at `:470-472`). D-06's additions: f64 mirrors
+ the CI half-width `d̄ ± t_{0.975,9} · s_d/√10` with the t-critical value as a
contract-resident frozen constant (Ph1 D-14 pattern), verified against a
`scipy.stats.t.ppf(0.975, 9)` fixture — no inverse CDF implementation.

---

### 6. `SetfitCommands::Bench` (CLI enum, dispatch)

**Analog:** `SetfitCommands::Train`, `crates/apr-cli/src/setfit_commands.rs:16-101`.

Namespace-justifying doc + variant shape (`setfit_commands.rs:16-37`):
```rust
#[derive(Subcommand, Debug)]
pub enum SetfitCommands {
    /// Train a SetFit classifier from Phase 2 artifacts and write a verified APR
    ///
    /// Configuration is FILE-FIRST: the twelve training knobs come from --config, ...
    Train {
        #[arg(long, value_name = "FILE")]
        config: PathBuf,
        #[arg(long, value_name = "DIR")]
        data: PathBuf,
        #[arg(long, value_name = "FILE")]
        selection: PathBuf,
        // ... #[arg(long)] force: bool,  #[arg(long = "dry-run")] dry_run: bool,
```
Copy: doc comments that state the refusal semantics per flag, `value_name` on every
path arg, `--force` no-clobber, `--dry-run` where a pre-flight is cheaper than the run.
D-13 adds a nested `Bench { #[command(subcommand)] command: BenchCommands }` with
`Run {...}` / `Report {...}` — the nested-subcommand shape is
`ExtendedCommands::Setfit` itself (`extended_commands.rs:799-802`).

**Dispatch arm analog** (`dispatch_analysis.rs:578` and `:832-860`):
```rust
ExtendedCommands::Setfit { command } => dispatch_setfit_command(command, cli),
// ...
#[cfg(feature = "setfit")]
fn dispatch_setfit_command(command: &SetfitCommands, cli: &Cli)
    -> std::result::Result<(), CliError> {
    match command {
        SetfitCommands::Train { config, data, selection, model_dir, output, seed,
                                device, force, dry_run }
            => commands::setfit_train::run(config, data, selection, model_dir, output,
                                           *seed, device.as_deref(), *force, *dry_run,
                                           cli.json),
    }
}
```
Note the recorded reason `cli.offline` is NOT threaded (`:824-830`): bench opens no
socket either — keep that discipline and its comment style.

---

### 7. `commands/setfit_bench.rs` adapter (CLI command, file I/O)

**Analog A — module discipline + entry ordering:**
`crates/apr-cli/src/commands/setfit_train.rs`.

Check ordering (`setfit_train.rs:601-621`):
```rust
pub(crate) fn run(config_path: &Path, data: &Path, selection_path: &Path, ...) -> Result<()> {
    // (1) The REQUEST, in full, before anything is read from --data.
    let file_config = parse_config(config_path)?;
    let merged = merge_overrides(&file_config, seed, device)?;
    let resolved_device = resolve_requested_device(&merged)?;
    // (2) And the one filesystem fact that is also a property of the request.
    refuse_existing_output(output_path, force)?;
    // (3) Phase 2's artifacts, replayed strictly against each other.
    let inputs = read_phase2_inputs(data, selection_path)?;
```
The same ordering is restated at `eval/setfit.rs:197-247` (WR-10 lesson: refuse the
output path BEFORE the long work).

Atomic write (`setfit_train.rs:176-188`; streaming form `data_contrastive.rs:159-181`):
```rust
pub(crate) fn atomic_write(target: &Path, bytes: &[u8], force: bool) -> Result<()> {
    refuse_existing_output(target, force)?;
    let dir = target.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(dir)?;
    let temp = temp_path(target);   // temp IN the destination dir — rename atomic on one fs
    let result =
        fill_and_sync(&temp, bytes).and_then(|()| fs::rename(&temp, target).map_err(CliError::Io));
    if result.is_err() { let _ = fs::remove_file(&temp); }
    result
}
```
(WR-01's rename race is OPEN by human ruling — do not fix here, do not widen.)

**Analog B — the input doors bench report/run reuse verbatim**
(`eval/setfit.rs:227-237`):
```rust
let mut ledger = AccessLedger::new();
let dataset = data_contrastive::read_attested_canonical(data, &mut ledger)?;
let manifest = data_contrastive::read_selection_manifest(selection)?;
let replayed = Selection::replay(&manifest, &dataset, &mut ledger).map_err(|error| {
    CliError::ValidationFailed(format!(
        "--selection {} does not replay against --data {}: {error}", ...))
})?;
```
`read_selection_manifest` (`data_contrastive.rs:614-626`) shows the refusal-with-remedy
message style ("not found. Write one with `apr data select ...`") and that digest
verification happens inside `from_bytes`, in ONE reader.

The adapter adds no gating of its own and bypasses none — the header contract at
`eval/setfit.rs:23-30` is the statement to replicate for bench: completeness/pairing/
lock rules live in the library + claims contract; the CLI is a filesystem shim.

---

### 8. `--selection-manifest` on `apr finetune --task classify`

**Analogs:** arg declaration in `crates/apr-cli/src/model_ops_commands.rs`
(`Finetune` variant at `:6`, `--task` at `:46-48`); dispatch pass-through in
`crates/apr-cli/src/dispatch.rs:686-753` (`Commands::ModelOps(ModelOpsCommands::Finetune {...}) => finetune::run(...)`);
the function being modified is `run_classify` (`commands/finetune.rs:1563-1689`).

**The hardcodes D-10 must replace with explicit per-cell inputs**
(`finetune.rs:1675-1685`):
```rust
let training_config = TrainingConfig {
    epochs: epochs as usize,
    val_split: 0.2,               // uncontracted internal random split — must be controlled
    save_every: 5,
    early_stopping_patience: 10,  // model selection with no lock — disable under frozen defaults
    checkpoint_dir: output_dir.clone(),
    seed: 42,                     // "42 is NOT a contracted seed" (tweet-eval contract)
    log_interval: 1,
    distributed: distributed_config,
    ..TrainingConfig::default()
};
```
All fields are `pub` on `entrenar::finetune::classify_trainer::TrainingConfig` (`:26`).

**The manifest→Selection wiring to copy** is Analog B in §7 (the exact same three
calls, refusing to start if replay fails) — that is what makes EVAL-02's
identical-sampled-ID guarantee structural: both methods read one artifact through one
code path. Each row records the manifest's `semantic_hash`.

---

### 9. `contracts/setfit-benchmark-claims-v1.yaml` (contract, declarative)

**Analog A — metadata header house style (the D-04/pv-diff worked example):**
`contracts/tweet-eval-stance-benchmark-v1.yaml:1-36`:
```yaml
metadata:
  # 1.1.0, and the bump is `pv diff`'s own suggestion, not a judgement call:
  #   pv diff /tmp/tweet-eval-old.yaml contracts/tweet-eval-stance-benchmark-v1.yaml
  #   -> "Contract diff: v1.0.0 -> v1.0.0 / Suggested bump: minor"
  # (`pv diff` takes TWO FILESYSTEM PATHS — see crates/aprender-contracts-cli/src/cli.rs:73-78
  # — so the old revision is materialized with `git show <rev>:<path> > /tmp/...` first.)
  ...
  version: 2.0.0
```
In-file: the exact `pv diff` command, its suggested bump, and the reasoning. The same
header pattern governs the lifecycle-contract edit (D-04's checkpoint presents this).

**Analog B — machine-readable equation blocks that Rust constants are PARSED from:**
`contracts/setfit-train-lifecycle-v1.yaml:98-133` (`frozen_thresholds:` — "this is what
the Rust constants are PARSED from by thresholds_match_the_contract") and `:276-282`
(`calibration_regime:` with `calibrated_regimes:` list). The claims contract should
carry its stats equations (mean/std/min-max, paired deltas, CI with the frozen
`t_{0.975,9}` literal), the row schema, completeness rule (manifest-defined 80-cell
expectation set), and pairing rule in the same machine-readable-beside-prose style, and
quote PF-007/PF-008.

**Analog C — contract-resident constants:** the tweet-eval equations block
(`:60-68`, `official_f_avg`) shows formula + domain/codomain + invariants shape.

**Wiring:** new contract must be appended to `$(CONTRACTS)` (`Makefile:1739-1786` —
`contracts/setfit-apr-v1.yaml` is currently the last entry at `:1786`) AND to a scoped
`PHASE5_CONTRACTS`/`contract-audit-phase5` per the `PHASE2_CONTRACTS` narrowing
precedent (`Makefile:1788-1793`); prove the gate can fail once (Pitfall 5).

---

### 10. `setfit-train-lifecycle-v1.yaml` edit (the D-01..D-04 keystone)

**Analog:** its own current text — the edit is additive against these exact anchors.

Current entry + the prose that must be amended (`setfit-train-lifecycle-v1.yaml:281-285`):
```yaml
    calibrated_regimes:
      - 'minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds=1,42,7|cells=s16e2b8,s8e1b4'
    invariants:
      - 'THE CALIBRATED SET CONTAINS EXACTLY ONE FINGERPRINT: ...'
```
The Phase-5 consequence clause pre-authorizing the edit's SHAPE is at `:328-334`
("Per D-10(c) that is a deliberate contract edit which `pv diff` flags ... not
something a Phase 5 executor may do inline"). The ε-derivation table style the new
regime's thresholds block must reproduce is the DERIVATION invariant at `:148-154`
(window rule `10 × worst_control_delta ≤ eps ≤ best_real_delta / 10`, per-class table).
The fixture `frozen_thresholds` block (`:102-133`) stays byte-untouched; the new
per-regime block is additive beside it, mirrored by the `thresholds.rs` restructuring
(§1) in the SAME commit — Pitfall 1.

---

### 11. Stats/calibration fixture generator (script, batch)

**Analog:** `scripts/setfit_fixtures/generate_fixtures.py::write_manifest` (`:922-949`):
```python
def write_manifest(directory: Path = FIXTURE_DIR) -> None:
    """Write `manifest.sha256` over every file in `directory` except the manifest itself.
    Paths are recorded as BARE FILENAMES ... resolve each entry against the directory
    the manifest was read from ..."""
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.name != "manifest.sha256")
    lines = []
    for p in files:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{h}  {p.name}")
    (directory / "manifest.sha256").write_text("\n".join(lines) + "\n")
    proc = subprocess.run(["shasum", "-a", "256", "-c", "manifest.sha256"],
                          cwd=directory, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"FATAL: manifest self-verification failed\n{proc.stdout}\n{proc.stderr}")
```
Copy: SHA-256 manifest over every emitted fixture + self-verification before exit
(Ph1 D-13). Run inside the pinned uv env (`scripts/setfit_fixtures/uv.lock` already
resolves scipy 1.18.0 / scikit-learn 1.9.0 — no new pins). Upstream-pin refusal
pattern at `:870-880` (`FATAL: upstream_manifest.json pins ...`).

**Rust-side consumption pattern:** `glm_tests.rs:274-298` — record the reference
source AND the RED (wrong-implementation) value in the test comment:
```rust
/// ... the correct IRLS (matching a statsmodels/scipy reference) gives slope 1.1266 ...
#[test]
fn falsify_glm_irls_link_derivative() {
    // RED (link/inverse-link swapped): slope ~1.0333, P ~0.1124.
    // GREEN (correct IRLS): slope ~1.1266, P ~0.0951.
    assert!((slope - 1.1266).abs() < 0.01, ...);
```

---

### 12. Makefile bench gates (build config)

**Analog:** the `setfit-tests` recipe, `Makefile:2122-2168`. The load-bearing lines:
```make
	@set +e; CARGO_INCREMENTAL=0 cargo test -p aprender-core --features setfit --lib setfit:: \
		> target/setfit-tests-core.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-tests-core.log; \
	if [ $$rc -ne 0 ]; then echo "FAIL: aprender-core setfit tests are red (rc=$$rc)"; exit $$rc; fi
	@$(call assert_tests_ran,target/setfit-tests-core.log,230,setfit-tests/aprender-core)
```
Copy verbatim: rc captured on the line after the redirect, NEVER through a pipe
(CLAUDE.md rule 1; the block's comment at `:2101-2106` records the two shipped
defects); `assert_tests_ran` non-vacuity floor per suite (CR-02 — "libtest exits 0 on
a zero-match filter"); floors re-measured whenever the surface moves (`:2129-2139`);
`CARGO_INCREMENTAL=0`; `mkdir -p target` before the redirect. New bench suites get
their own measured floors in the `:2259-2264` comment-table style.

---

### 13. Per-row-predictions evaluator door (service, request-response)

**Analog:** `evaluate_validation_from_artifact`,
`crates/aprender-train/src/train/setfit/apr_evaluate.rs:185-195`:
```rust
pub fn evaluate_validation_from_artifact(
    credential: &ReloadedSetFitCredential,
    dataset: &PreparedDataset<Canonical>,
    metric: ValidationMetricKind,
) -> Result<ValidationEvaluation, SetFitTrainError> {
    let model = credential.model();
    // (1) THE CORPUS, from the ARTIFACT'S OWN RECORD. The reload door checked this against
    //     the dataset it was handed; checking it again here is not redundancy ...
```
The sibling returns per-row prediction vectors (labels + probabilities) instead of a
scalar, takes the SAME credential type (so it is unreachable without the reload door),
and re-checks the artifact-vs-dataset identity the same way. It is the ONE evaluation
door (OPS-03) that `bench run` calls for F_avg/MCC/confusion/calibration — never
classify-in-the-adapter. Metric assembly then uses the shipped, sklearn-parity-tested
surface: `MultiClassMetrics::from_predictions` (`eval/classification/metrics.rs:61`),
`f1_average_for_classes(f1, &[1, 2])` (`metrics.rs:10`, contract-bound
`official_f_avg`), `matthews_corrcoef` (`metrics/agreement.rs:63`).

---

### 14. Spawned bench-cell evidence test (integration, process orchestration)

**Analog:** `crates/apr-cli/tests/setfit_cli_lifecycle.rs:1-90`.

The rules stated in its header are the pattern:
```rust
#![cfg(feature = "setfit")]
//! ... every invocation in this file is a real `fork`/`exec` of
//! `env!("CARGO_BIN_EXE_apr")`, and every verdict is read from a **reaped `ExitStatus`**.
//! **Rule 1 — never read a status through a pipe.** ...
//! **Rule 3 — pin the binary.** `env!("CARGO_BIN_EXE_apr")` is an absolute path that cargo
//! computes for the binary it just built from THIS tree. ...
//! **Rule 2 — prove the mechanism ...** The first rung is a `--version` call whose output is
//! asserted non-empty ...
```
Rung table at `:64-71` (rung 4 = `setfit train` exit 6 is exactly what wave 1 flips to
0 — after the flip, blocked-rung tests panic with their embedded restore instructions,
per the convention at `:56-60`, and the file gains POSITIVE production-chain rungs
env-gated on the checkout). F-10 flip blast radius (files carrying refusal/prose
claims to re-audit) is enumerated in RESEARCH.md F-10 section.

## Shared Patterns

### Atomic write (apply to: every row, manifest, lock, and report file the phase writes)
**Source:** `crates/apr-cli/src/commands/setfit_train.rs:176-188` (byte form) /
`crates/apr-cli/src/commands/data_contrastive.rs:159-193` (streaming form).
Temp in destination dir → `write_all` → `sync_all` → single `rename` → temp cleanup on
every error path; no-clobber unless `--force`; the no-clobber gate ALSO callable
standalone BEFORE the work (`setfit_train.rs:190-209`, the WR-10 lesson).

### Digest-verify-before-return (apply to: row files, run manifest, any transported artifact)
**Source:** `SelectionManifest::from_bytes` + `verify_digest`
(`aprender-contrastive-data/src/manifest.rs:248-272`). Parse → recompute SHA-256 over
canonical bytes → compare → only then return. GPU-host transport needs nothing extra:
rows are self-verifying files, re-verified on ingest and again by `bench report`.

### `deny_unknown_fields` + `schema_version` envelope (apply to: every new serde type)
**Source:** `manifest.rs:169-178` and `:313-316`. This is also the D-06 guard: a
bootstrap field cannot ride along into a `deny_unknown_fields` row.

### Refusal messages that name the remedy (apply to: all adapters)
**Source:** `data_contrastive.rs:614-626`, `eval/setfit.rs:206-221, 254-262`.
Typed `CliError::ValidationFailed` naming the flag AND what to run instead. The D-11
`apr qa` error-message improvement follows this exactly: name
`apr validate --quality` / `apr eval` at the refusal site.

### Non-vacuous gates (apply to: every new Make/CI filter)
**Source:** `Makefile` `assert_tests_ran` calls (`:2147,2153,2164`) + measured-floor
comment blocks. Every new filter asserts non-zero matched tests; induce one RED before
trusting.

### Status never through a pipe (apply to: every recipe and spawned test)
**Source:** `Makefile:2101-2106` comment block; `setfit_cli_lifecycle.rs:20-25`.

### Contract-bound functions (apply to: multiclass ECE/Brier, paired stats)
**Source:** `calibration.rs:138` —
`#[provable_contracts_macros::contract("calibration-v1", equation = "...")]` +
`contract_pre_*!` macro preconditions. New claims equations bind the same way (whether
to `calibration-v1` or the new claims contract is planner's call; the binary pair
binds to `calibration-v1`).

### Backend identity read from execution (apply to: every row)
**Source:** `ExecutionBackend::identity` (`aprender-core/src/setfit/encoder.rs:227-238`,
`<device>:setfit-core:<kernel>`) for SetFit rows; `pipeline.gpu_name()/gpu_total_memory()`
(`finetune.rs:1629`) for LoRA rows. Binding row already resolved by 04-19 — use, don't
re-fix.

## No Analog Found

Planner should use RESEARCH.md patterns (cited) instead of a code analog:

| Surface | Role | Data Flow | Reason / Fallback |
|---|---|---|---|
| Peak-RSS measurement | resource metric | measurement | Nothing in-repo measures peak memory (`aprender-simulate`'s `peak_memory_bytes` is an always-`None` placeholder). Use RESEARCH Pattern 6: Linux `/proc/self/status` VmHWM text read (no `unsafe`), `sysinfo` 0.32 (already a workspace dep, root `Cargo.toml:266`) as sampled fallback — contract must say "sampled". Latency/warmup precedent DOES exist: `apr bench --warmup/--iterations` (`commands/bench.rs:102-117`, defaults warmup 3) and `ClassifyResponse.latency_ms` (bit-asserted, excluded from `PartialEq`). |
| LoRA-side selection-lock semantics | access-control rule | — | `create_selection_lock`/`mint_test_token`/`grant` are generic over a SEALED `SetFitCredential` (exactly two implementors, counted by `credential_seal_is_a_private_supertrait`). No LoRA analog exists by design. RESEARCH Open Q3 recommendation (b): claims-contract-level no-selection attestation + manifest hash in LoRA rows, with a doctored-row negative — do not widen the seal. |
| 40-cell GPU-host orchestration/transport **[DESCOPED 2026-09-08, D-19 — no longer to be built]** | orchestration script | batch/remote | Nearest precedent is `scripts/dispatch-*.sh` (e.g. `dispatch-distill-phase-3-gx10.sh` — ssh, build, run, artifacts return as files) — MEDIUM-confidence house pattern, not a strong analog. lambda-vector reachability is UNVERIFIED from this host (RESEARCH Open Q1, human checkpoint). Hash-committed rows make transport mechanism-agnostic (see Shared Patterns: digest-verify). |

## Metadata

**Analog search scope:** `crates/aprender-train/src/train/setfit/`,
`crates/aprender-core/src/{calibration.rs, stats/, metrics/, setfit/}`,
`crates/apr-cli/src/{commands/, setfit_commands.rs, dispatch*.rs, extended_commands.rs, model_ops_commands.rs}`,
`crates/aprender-contrastive-data/src/`, `contracts/`, `scripts/setfit_fixtures/`,
`Makefile`, `crates/apr-cli/tests/`.
**Files read this session:** 20 (targeted non-overlapping ranges for the 6 files > 1,000 lines).
**Pattern extraction date:** 2026-08-16
