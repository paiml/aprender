# Phase 4: APR Artifact and Production Parity - Pattern Map

**Mapped:** 2026-08-14
**Files analyzed:** 24 (11 new, 13 modified)
**Analogs found:** 22 / 24 (2 composite/partial — see "No Analog Found")

All paths relative to repo root. All line numbers read on branch `gsd/phase-2-contract-gate` @ d66678e7a.
The crate-level `crates/aprender-train/CLAUDE.md` and `crates/aprender-serve/CLAUDE.md` are stale
pre-monorepo docs — root `CLAUDE.md` governs (per 04-RESEARCH.md "Project Constraints").

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/aprender-train/src/train/setfit/apr_codec.rs` (NEW) | codec adapter | transform (bytes↔bundle) | `crates/aprender-train/src/train/setfit/verify.rs:149-192` (`SerdeJsonCodec`) | exact — shape prescribed by verify.rs:20-29 module docs |
| `crates/aprender-core/src/setfit/artifact.rs` (NEW, writer half) | format/serialization | file-I/O (write) | `crates/apr-format/src/v2/writer.rs:30-57` + `header_impl.rs:132-135,298` | exact |
| `crates/aprender-core/src/setfit/artifact.rs` (NEW, loader half) | format/validation | file-I/O (fail-closed read) | `crates/aprender-train/src/train/setfit/bundle.rs:456-524` + `crates/aprender-core/src/setfit/mod.rs:354-374` | exact composite |
| `crates/aprender-core/src/setfit/classify.rs` (NEW) | service + response envelope | request-response | `data_contrastive.rs:548-571` (`PairsReport`) + `verify.rs:336-341` (NaN idiom) | role-match |
| `crates/apr-cli/src/setfit_commands.rs` (NEW) | CLI command enum | request-response | `crates/apr-cli/src/data_commands.rs:5-57` (`DataCommands`) | exact |
| `crates/apr-cli/src/commands/setfit_train.rs` (NEW) | CLI filesystem adapter | file-I/O | `crates/apr-cli/src/commands/data_contrastive.rs:486-531` (`run_select`) | exact |
| `crates/apr-cli/src/commands/predict.rs` (NEW) | CLI command (generic) | request-response | dispatch pattern + `inspect.rs:504-532` (tag detection) | role-match |
| `crates/aprender-serve/src/api/setfit_handlers.rs` (NEW) | HTTP handler + AppState slot | request-response | `crates/aprender-serve/src/api/apr_handlers.rs:30-55` + `api/mod.rs:110-186` | exact |
| `contracts/setfit-apr-v1.yaml` (NEW) | contract | config | `contracts/setfit-train-lifecycle-v1.yaml` (Ph1 D-23 one-contract-per-phase) | exact |
| Parity harness tests (NEW, site per RESEARCH A4: `crates/apr-cli/tests/`) | test | request-response ×3 surfaces | `api/tests/app_state_default.rs:42-52` (oneshot leg only) | partial — see below |
| trybuild `VerifiedSetFitModel` non-constructibility (NEW) | test | compile-fail | `crates/aprender-train/tests/ui/setfit_external_codec_impl.rs` + `tests/ui.rs:60` | exact |
| `crates/apr-cli/src/extended_commands.rs` (MOD) | CLI enum wiring | — | itself, lines 730-734 (`Data` variant) | exact in-place |
| `crates/apr-cli/src/dispatch_analysis.rs` (MOD) | CLI dispatch | — | itself, lines 570 + 726-780 | exact in-place |
| `crates/apr-cli/src/commands/inspect.rs` (MOD) | CLI command | file-I/O | itself, lines 504-532 (custom-key read) | exact in-place |
| `crates/apr-cli/src/commands/eval*` (MOD, D-16 routing) | CLI adapter | request-response | `lock.rs:419,491,552,586` + `evaluate.rs` (Ph3 substrate) | exact substrate |
| `crates/aprender-serve/src/api/router.rs` (MOD) | router | request-response | itself, lines 57-110 (conditional group) + 265-276 (readiness) | exact in-place |
| `crates/aprender-serve/Cargo.toml` (MOD) | feature config | — | itself, line 225 (`aprender-serve = ["dep:aprender", "server"]`) | exact in-place |
| `crates/apr-cli/Cargo.toml` (MOD) | feature config | — | itself, lines 72-92 (`[features]`) | exact in-place |
| `crates/apr-cli/src/commands/serve/handlers.rs` (MOD) | serve startup detection | file-I/O | itself, lines 331-337 + 741-753 (magic-byte dispatch) | exact in-place |
| `crates/aprender-train/src/train/setfit/{mod.rs, verify.rs}` (MOD) | module wiring / seal visibility | — | `verify.rs:62-65` (private `mod sealed`) | exact — one-line visibility decision |
| `Makefile` (MOD) | build gates | batch | `assert_tests_ran` at 1381-1390; `$(CONTRACTS)` at 1138-1184; `setfit-feature-matrix` at 382+ | exact in-place |
| `.github/workflows/ci.yml` (MOD — **human-visible edit per CLAUDE.md autonomy rules**) | CI config | batch | itself, setfit step ~274-309 | exact in-place |
| `CLAUDE.md` (MOD — realizar-first table SetFit row, D-09) | docs | — | the realizar-first table itself | exact in-place |
| `crates/aprender-core/src/setfit/mod.rs` (MOD, exports) | module wiring | — | itself | exact in-place |

### Appended 2026-08-14 — files added by the revision that created plans 04-12..04-16 and the wave-2 library-API work

The rows above were written before plans 04-12..04-16 existed and before wave 2 was split into
04-02 / 04-13 / 04-14, so roughly a third of the files the plan set now touches had no row. The
discipline was never violated — each of those plans cites a concrete in-tree analog in its own
`<read_first>` — but the table was stale, so the analogs were only visible inside the plans. They
are recorded here as well:

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/aprender-train/src/train/setfit/apr_reload.rs` (NEW, 04-16) | trusted reload door | file-I/O (fail-closed read) | `verify.rs:577-660` (`run_verify_policy` — the identity-gate policy shape this door mirrors) | role-match — cited in 04-16 `read_first` |
| `crates/aprender-train/src/train/setfit/bundle.rs` (MOD, 04-13: `ProvenanceRecord` + field 20 + schema bump) | trusted value type | transform (run→bundle) | itself, lines 285-420 (`ResolvedConfigRecord`, `from_run_parts`, the `f32_to_hex` bit-pattern precedent) | exact in-place |
| `crates/aprender-train/src/train/setfit/bundle_tests.rs` (MOD, 04-13: allowlist completeness gate) | test | — | `evidence.rs:200-222` (`first_null_path`/`join_path` — the null-walk shape, widened from first-only to all) | exact — cited in 04-13 `read_first` |
| `crates/aprender-train/src/train/setfit/verify_tests.rs` (MOD, 04-13: two `from_run_parts` call sites) | test | — | itself, lines 205-225 and 330-350 (the two call sites; `#[path = "verify_tests.rs"] mod verify_tests;` at verify.rs:678-680 makes it part of the `setfit::verify` target) | exact in-place |
| `crates/aprender-train/src/train/setfit/{config.rs, lock.rs, lock_tests.rs, evaluate.rs}` (MOD, 04-14: library-API additions incl. `SelectionLock::from_canonical_bytes`) | trusted library API | transform + file-I/O | `bundle.rs:437-470` (the bounded/fail-closed `from_canonical_bytes` door discipline these follow) | exact — cited in 04-14 `read_first` |
| `crates/apr-cli/src/setfit_io.rs` (NEW, 04-06) | CLI bounded reader | file-I/O (fail-closed read) | `bundle.rs:437-470` (raw-length cap before parse) + apr-cli's existing file adapters | role-match |
| `crates/aprender-core/src/setfit/encoder.rs` (MOD, 04-04: `ExecutionBackend` channel) | execution identity channel | transform (encode→identity) | itself (the encode entry point) + `verify.rs:62-65` private-module sealing idiom for the no-public-constructor rule | role-match — cited in 04-04 `read_first` |
| `crates/aprender-train/tests/setfit_apr_lifecycle.rs` (NEW, 04-12) | integration test (in-process OPS-01 lifecycle) | request-response | `verify.rs` policy tests + the crate's existing `tests/ui.rs` harness siting | role-match — cited in 04-12 `read_first` |
| `crates/apr-cli/tests/setfit_cli_lifecycle.rs` (NEW, 04-15) | integration test (spawned-binary OPS-02 lifecycle + generic-tooling A3) | request-response ×5 processes | `env!(CARGO_BIN_EXE_apr)` spawn pattern in apr-cli's existing integration tests | role-match — cited in 04-15 `read_first` |

## Pattern Assignments

### `crates/aprender-train/src/train/setfit/apr_codec.rs` (codec adapter)

**Analog:** `crates/aprender-train/src/train/setfit/verify.rs` — the shape is literally prescribed by the module docs (lines 20-29): "phase 4's APR codec lands as a thin ADAPTER module inside this crate — an `impl Sealed for AprCodec` plus a `SetFitCodec` impl that calls `aprender-core`'s APR format code."

**Trait to implement** (verify.rs:83-100 — three methods, nothing else):
```rust
pub trait SetFitCodec: sealed::Sealed {
    fn format_id(&self) -> &'static str;
    fn serialize(&self, bundle: &SetFitBundle) -> Result<Vec<u8>, CodecError>;
    fn deserialize(&self, bytes: &[u8]) -> Result<SetFitBundle, CodecError>;
}
```

**Impl to copy** (verify.rs:149-192, `SerdeJsonCodec` — including the deliberate redundant format-id self-check):
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SerdeJsonCodec;

impl sealed::Sealed for SerdeJsonCodec {}

impl SetFitCodec for SerdeJsonCodec {
    fn format_id(&self) -> &'static str { SERDE_JSON_FORMAT_ID }

    fn serialize(&self, bundle: &SetFitBundle) -> Result<Vec<u8>, CodecError> {
        bundle.to_canonical_bytes().map_err(|source| CodecError::Bundle {
            format_id: SERDE_JSON_FORMAT_ID.to_string(),
            source,
        })
    }

    fn deserialize(&self, bytes: &[u8]) -> Result<SetFitBundle, CodecError> {
        let bundle = SetFitBundle::from_canonical_bytes(bytes).map_err(|source| {
            CodecError::Bundle { format_id: SERDE_JSON_FORMAT_ID.to_string(), source }
        })?;
        // This check is ALSO made by the trusted `decode`, and the redundancy is deliberate:
        // ... covers THIS codec's own `pub` surface ... Neither subsumes the other.
        if bundle.format_id() != SERDE_JSON_FORMAT_ID {
            return Err(CodecError::ForeignFormat {
                expected: SERDE_JSON_FORMAT_ID.to_string(),
                got: bundle.format_id().to_string(),
            });
        }
        Ok(bundle)
    }
}
```

**Seal visibility constraint** (verify.rs:61-65): `mod sealed` is private to `verify.rs`. The
`AprCodec` either lives in verify.rs's module tree or the marker gains `pub(crate)` — a one-line
decision, not a redesign (per RESEARCH Pattern 1).

**Inherited obligations** (verify.rs:496-529, `close_round_trip`; and 594-609 in `run_verify_policy`):
`serialize(deserialize(bytes)) == bytes` byte-for-byte or `SetFitTrainError::ReloadNotFromBytes`
fires. `verify_artifact` currently runs `Tolerance::EXACT` (verify.rs:250-253); any change must be
introduced in trusted code keyed on codec, never a trait parameter (verify.rs:236-241 says exactly this).
The format id is stamped by trusted code via `SetFitBundle::from_run_parts(codec.format_id(), ..)`
(verify.rs:290-297) and re-checked by trusted `decode` (verify.rs:225-234).

---

### `crates/aprender-core/src/setfit/artifact.rs` — writer half

**Analog:** `crates/apr-format/src/v2/writer.rs` (re-exported as `aprender::format::v2`) + `header_impl.rs`.

**Writer API** (writer.rs:30-57 — note LAYOUT_ROW_MAJOR is set automatically in `new`):
```rust
pub fn new(metadata: AprV2Metadata) -> Self {
    let mut header = AprV2Header::new();
    // LAYOUT-002: Mark all new APR files as row-major
    header.flags = header.flags.with(AprV2Flags::LAYOUT_ROW_MAJOR);
    Self { header, metadata, tensors: Vec::new() }
}

pub fn add_tensor(&mut self, name: impl Into<String>, dtype: TensorDType,
                  shape: Vec<usize>, data: Vec<u8>) { ... }        // U8 tokenizer blob goes here

pub fn add_f32_tensor(&mut self, name: impl Into<String>, shape: Vec<usize>, data: &[f32]) {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    self.add_tensor(name, TensorDType::F32, shape, bytes);
}
// writer.rs:317 `write() -> Result<Vec<u8>, V2FormatError>` — sorted index, deterministic
// offsets, CRC32 footer. `write_into(path)` at :396 exists but the codec needs BYTES
// (the seam is bytes↔bundle); file placement is the CLI adapter's job.
```

**Metadata struct** (header_impl.rs:132-135 typed tag; :184-185 created_at; :298 custom flatten):
```rust
pub struct AprV2Metadata {
    #[serde(default)]
    pub model_type: String,          // ← D-04 tag: "setfit". Declaration-order-serialized.
    ...
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,  // ← MUST stay None (determinism, Pitfall 2); absence is clean
    ...
    #[serde(default, flatten)]       // line 298 — HashMap<String, serde_json::Value>
    pub custom: ...                  // ← EXACTLY ONE key ("setfit"), value = serde_json::Map
}
```
Determinism rule (empirically verified in RESEARCH Pitfall 1): >1 custom key ⇒ nondeterministic
byte order ⇒ artifact hash varies and the closure check fails intermittently. One key holding a
`serde_json::Map` (BTreeMap-backed, sorted) sidesteps it with zero apr-format changes.
The full one-key construction excerpt is in 04-RESEARCH.md "Code Examples" — copy it verbatim.

---

### `crates/aprender-core/src/setfit/artifact.rs` — loader half (validation ladder → typestate)

**Analog 1 — the fail-closed ladder** (`crates/aprender-train/src/train/setfit/bundle.rs:456-524`,
`from_canonical_bytes_within`; order is load-bearing, doc comment at 437-449):
```rust
// 1. Input-length limit on the RAW slice, before serde is handed anything.
let observed = bytes.len() as u64;
if observed > limits.max_bundle_bytes {
    return Err(BundleError::BundleLimitExceeded {
        what: "input_bytes", limit: limits.max_bundle_bytes, observed,
    });
}
// 2. The parse (bounded by step 1). 3. The schema version:
if bundle.schema_version != BUNDLE_SCHEMA_VERSION {
    return Err(BundleError::UnsupportedSchemaVersion {
        got: bundle.schema_version, supported: BUNDLE_SCHEMA_VERSION,
    });
}
// 4. Counts/sizes computed from DECLARED lengths — no Vec<f32> allocated yet:
let elements = (tensor.data_hex.len() / 8) as u64;   // the APR loader mirrors this with
if elements > limits.max_elements_per_tensor { ... } // TensorIndexEntry.size == product(shape) × dtype_width
total = total.saturating_add(elements);
if total > limits.max_total_elements { ... }
```
Also copy the door discipline (bundle.rs:456-463): the `pub` entry names ONLY the contracted
bound (`BundleLimits::CONTRACTED`); the caller-chosen-bounds variant stays module-private.

**Analog 2 — the rebuild door** (`crates/aprender-core/src/setfit/mod.rs:354-374`,
`from_bundle_parts`; tokenizer hash checked FIRST, per doc at 340-344):
```rust
pub fn from_bundle_parts(
    tokenizer_bytes: &[u8],
    arch: &EncoderArchitecture,
    tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    root_seed: u64,
) -> Result<Self, SetFitError> {
    let observed = tokenizer::sha256_hex(tokenizer_bytes);
    if observed != arch.tokenizer_sha256 {
        return Err(SetFitError::TokenizerHashMismatch {
            expected: arch.tokenizer_sha256.clone(), got: observed,
        });
    }
    let tokenizer = MiniLmTokenizer::from_bytes(tokenizer_bytes)?;
    let encoder = BertSentenceEncoder::from_named_tensors(arch, tensors, root_seed)?;
    ...
}
```
`tokenizer_bytes()` accessor at mod.rs:389-391; head rebuild via
`MultinomialLogisticRegression::from_stored_coefficients` as used in verify.rs:552-555.

**Analog 3 — probe replay** (verify.rs:311-327 `probe_model` + verify.rs:483-494 `compare_probes`):
the D-11 consumer-side probe replay is the same five-rung comparison shape (row counts FIRST —
rung ordering rationale at verify.rs:357-362 — then ids, embeddings, probabilities, labels), at
contract-resident tolerances from the new `setfit-apr-v1.yaml` (one constants module, Pitfall 9).
Success constructs `VerifiedSetFitModel` with private constructors — the only door to `classify`.

---

### `crates/aprender-core/src/setfit/classify.rs` (D-08 envelope + classify path)

**Analog for the response struct family** (`data_contrastive.rs:548-571`, `PairsReport` — a
borrowed, fully-typed, `Serialize` report struct with a `command` discriminant; the CLI serializes
it for `--json` and renders a human view over the same values):
```rust
#[derive(Serialize)]
struct PairsReport<'a> {
    command: &'static str,
    selection: String,
    selection_hash: &'a str,
    pair_manifest_hash: &'a str,
    root_seed: u64,
    ...
    deviation: &'a [String; 3],   // contract text carried verbatim, not re-stated
}
```
D-08 difference: `ClassifyResponse` lives in **core** (both apr-cli and aprender-serve consume it;
dependency direction verified in RESEARCH), is versioned, and is owned — not borrowed — since the
HTTP handler returns it. Required fields per OPS-04: labels, full probabilities, optional logits,
margins, token/truncation facts (`SetFitMiniLm::tokenize` → `SentenceBatch`), latency, backend
identity, artifact hash.

**Backend identity (D-12):** read from execution — trueno runtime detection at
`crates/aprender-compute/src/lib.rs:281-312` (`detect_x86_backend`/`detect_arm_backend`); never
echo config. `resolve_device` (`crates/aprender-train/src/train/device.rs:110`) stays the
fail-closed request gate.

**Non-finite handling** (verify.rs:336-341 — the NaN-visible comparison idiom; CR-03 lesson):
```rust
fn within(delta: f64, bound: f64) -> bool {
    matches!(delta.partial_cmp(&bound),
        Some(core::cmp::Ordering::Less | core::cmp::Ordering::Equal))
}
```
Typed rejection of non-finite values before envelope serialization (serde_json renders NaN/inf as
`null` — Pitfall 5); stored floats in the metadata doc use the bundle.rs bit-pattern-hex precedent
(`f32_to_hex`, bundle.rs:390,399,408-409).

---

### `crates/apr-cli/src/setfit_commands.rs` + enum/dispatch wiring

**Analog:** the `apr data` namespace, three files, copied mechanically.

**1. Subcommand enum** (`data_commands.rs:5-24` — note doc comments become `--help` text and carry
protocol rationale, e.g. the no-default-seed comment at :40-45):
```rust
#[derive(Subcommand, Debug)]
pub enum DataCommands {
    /// Prepare the TweetEval abortion stance benchmark as aprender JSONL
    TweetEvalStance {
        #[arg(short, long, value_name = "DIR")]
        output: PathBuf,
        #[arg(long, value_enum, default_value_t = TweetEvalStanceProfile::Canonical)]
        profile: TweetEvalStanceProfile,
        ...
    },
    ...
}
```

**2. Parent variant** (`extended_commands.rs:730-734`):
```rust
    /// Data quality pipeline (audit, split, balance) — powered by alimentar
    Data {
        #[command(subcommand)]
        command: DataCommands,
    },
```

**3. Dispatch arm + fan-out** (`dispatch_analysis.rs:570` and `:726-764`):
```rust
ExtendedCommands::Data { command } => dispatch_data_command(command, cli),
...
fn dispatch_data_command(command: &DataCommands, cli: &Cli) -> std::result::Result<(), CliError> {
    let json = cli.json;
    match command {
        // `cli.offline` is deliberately NOT passed to these two arms. Neither command
        // opens a socket ... Threading `offline` through so it could be ignored would
        // advertise a network switch on a command that has no network to switch off.
        DataCommands::Select { data, shots, seed, any_seed, output, force } =>
            commands::data_contrastive::run_select(
                data, *shots, *seed, *any_seed, output.as_deref(), *force, json),
        ...
    }
}
```
`apr setfit train` clones this exactly: `SetfitCommands` enum → `Setfit { command }` variant →
`dispatch_setfit_command` → `commands::setfit_train::run`. `apr predict` is a NEW top-level generic
command (no `Predict` variant exists today — verified in RESEARCH) using the same
variant → dispatch arm → `commands::predict::run` wiring.

---

### `crates/apr-cli/src/commands/setfit_train.rs` (filesystem adapter)

**Analog:** `crates/apr-cli/src/commands/data_contrastive.rs` — the Ph2 D-04 adapter precedent:
CLI validates the REQUEST first, calls the library on typed values, atomically writes, then renders.

**Adapter shape** (`run_select`, lines 486-531):
```rust
pub(crate) fn run_select(data: &Path, shots: u32, seed: u64, any_seed: bool,
                         output: Option<&Path>, force: bool, json_output: bool) -> Result<()> {
    // Fail-closed on the REQUEST first. A bad shot count or an off-protocol seed is not a
    // problem with the data, and reporting it as one sends the user to the wrong place.
    validate_shots(shots)?;
    let seed_mode = resolve_seed_mode(seed, any_seed)?;
    ...
    let bytes = manifest.to_file_bytes().map_err(|e| dataset_error(&e))?;
    atomic_write(&manifest_path, &bytes, force)?;

    if json_output {
        println!("{}", select_report_json(...)?);
    } else {
        render_select_human(...);
    }
    Ok(())
}
```

**Atomic write** (data_contrastive.rs:159-193 — temp file + rename, `--force` gate, best-effort
cleanup that never masks the original error):
```rust
fn atomic_write_with<F>(target: &Path, force: bool, fill: F) -> Result<()>
where F: FnOnce(&mut fs::File) -> Result<()> {
    if !force && target.exists() {
        return Err(CliError::ValidationFailed(format!(
            "Refusing to replace existing file {} (pass --force to replace it)",
            target.display())));
    }
    let dir = target.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(dir)?;
    let temp = temp_path(target);
    let result = fill_and_sync(&temp, fill)
        .and_then(|()| fs::rename(&temp, target).map_err(CliError::Io));
    if result.is_err() {
        let _ = fs::remove_file(&temp);  // cleanup failure must not replace the reason it failed
    }
    result
}
```

**Error translation with remedy** (data_contrastive.rs:578-593 — the crate cannot name a CLI flag;
the adapter appends the knob):
```rust
fn pair_config_error(error: &ContrastiveDataError) -> CliError {
    let remedy = match error {
        ContrastiveDataError::BudgetExceedsHardCap { .. } =>
            " — raise --hard-cap or lower --budget. ...",
        ...
    };
    CliError::ValidationFailed(format!("contrastive data: {error}{remedy}"))
}
```

**D-07 config-file-first** — deserialization is already validating (RESEARCH-verified,
`crates/aprender-train/src/train/setfit/config.rs:16-27`: `#[serde(try_from = "SetFitTrainConfigWire")]`
routes every payload through `SetFitTrainConfig::new`):
```rust
let cfg: SetFitTrainConfig = toml::from_str(&fs::read_to_string(path)?)   // or serde_json
    .map_err(|e| CliError::ValidationFailed(format!("--config: {e}")))?;
```
`toml = { workspace = true }` added to apr-cli (workspace dep 0.8 already exists — no new package).

---

### `crates/apr-cli/src/commands/predict.rs` + inspect/eval auto-detect (D-04/D-06)

**Analog for tag detection:** `crates/apr-cli/src/commands/inspect.rs:504-532` — inspect already
seeks to the metadata offset, parses `AprV2Metadata`, and reads custom keys cheaply:
```rust
fn read_metadata(reader: &mut BufReader<File>, header: &HeaderData) -> MetadataInfo {
    ...
    let mut metadata_bytes = vec![0u8; header.metadata_size as usize];
    if reader.read_exact(&mut metadata_bytes).is_err() { return MetadataInfo::default(); }
    match AprV2Metadata::from_json(&metadata_bytes) {
        Ok(meta) => {
            let source_metadata = meta.custom.get("source_metadata").cloned();
            MetadataInfo {
                model_type: if meta.model_type.is_empty() { None } else { Some(meta.model_type) },
                ...
```
The SetFit branch reads `meta.model_type == "setfit"` + the one `"setfit"` custom key. Explicit tag
only (D-04) — no tensor-name sniffing. Untagged SetFit-shaped APRs are plain APRs (negative test).
Detection routes to the ONE core loader; the CLI defines no response struct — it serializes the
D-08 envelope directly (`cli.json` flag threading as in dispatch_data_command:727).

**Analog for `apr eval` D-16 routing:** the Ph3 substrate, consumed as shipped:
`crates/aprender-train/src/train/setfit/lock.rs:491` (`create_selection_lock`), `:419`
(`mint_test_token`), `:552`/`:586` (`CanonicalTestAccess`/`grant`), plus `evaluate.rs`'s trusted
evaluator. The CLI is a thin adapter over this workflow — no new eval gating.

**Exit codes** (`crates/apr-cli/src/error.rs:14-81` — extend `CliError`, don't invent):
```rust
pub fn exit_code(&self) -> ExitCode {
    contract_pre_exit_code_semantics!();
    ...
    match self {
        Self::FileNotFound(_) | Self::NotAFile(_) => ExitCode::from(3),
        Self::InvalidFormat(_) => ExitCode::from(4),
        Self::ValidationFailed(_) => ExitCode::from(5),
        Self::ModelLoadFailed(_) => ExitCode::from(6),
        Self::InferenceFailed(_) => ExitCode::from(8),
        Self::FeatureDisabled(_) => ExitCode::from(9),
        ...
    }
}
```
New variants (if any) get contract-macro'd codes here; the contract macros at error.rs:66-68 bind
the mapping. `ModelLoadFailed`/`InferenceFailed`/`FeatureDisabled` already cover most new paths.

---

### `crates/aprender-serve/src/api/setfit_handlers.rs` + router/AppState/readiness (D-09/D-10)

**AppState slot analog** (`crates/aprender-serve/src/api/mod.rs:110-186` — Clone struct of
`Option<Arc<...>>` model slots; the two closest precedents):
```rust
#[derive(Clone)]
pub struct AppState {
    ...
    /// APR model for /v1/predict endpoint (real inference, not mock)
    apr_model: Option<Arc<AprModel>>,                                  // line 129
    ...
    /// APR Transformer for SafeTensors/APR inference (PMAT-SERVE-FIX-001)
    apr_transformer: Option<Arc<crate::apr_transformer::AprTransformer>>,  // line 166
    ...
}
```
New slot: `setfit_model: Option<Arc<VerifiedSetFitModel>>` behind a new `setfit` feature.
Note api/mod.rs is assembled via `include!` (mod.rs:209-212 includes `router.rs` etc.) — the new
handlers file follows the existing `mod apr_handlers; pub(crate) use apr_handlers::...` pattern
(mod.rs:96-97), with `#[cfg(feature = "setfit")]`.

**Handler analog** (`api/apr_handlers.rs:30-55` — extract state, validate, 503 when slot empty):
```rust
pub(crate) async fn apr_predict_handler(
    State(state): State<AppState>,
    Json(request): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();
    if request.features.is_empty() {
        return Err((StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: "Input features cannot be empty".to_string() })));
    }
    let apr_model = state.apr_model.as_ref().ok_or_else(|| {
        (StatusCode::SERVICE_UNAVAILABLE,
         Json(ErrorResponse { error: "No APR model loaded. ...".to_string() }))
    })?;
    ...
}
```
Do NOT copy the body past line 55 — `/v1/predict` takes `features: Vec<f32>` (wrong shape for text,
RESEARCH-flagged deprecated pattern). New route `POST /v1/classify` takes texts, returns the D-08
`ClassifyResponse` from core. Add a contracted request-body/batch-size bound mirroring D-03
(Security V5).

**Conditional route install analog** (`api/router.rs:57-62` health group + `:80-110` — the GH-148
conditional group is the exact shape for feature/slot-gated installation):
```rust
pub fn create_router_with_config(state: AppState, config: RouterConfig) -> Router {
    let mut router = Router::new()
        .route("/health", get(health_handler))
        .route("/health/live", get(health_live_handler))
        .route("/health/ready", get(health_ready_handler))
        ...;
    if config.openai_api {                       // ← the conditional-group pattern to copy
        router = router
            .route("/v1/predict", post(apr_predict_handler))
            ...;
    }
```

**Readiness analog** (`api/router.rs:261-276` — extend the response with `artifact_sha256` +
`verified: true` when the SetFit slot is populated, per OPS-05):
```rust
/// 200 iff `status == "ok"` AND `model_loaded == true`; 503 otherwise.
async fn health_ready_handler(State(state): State<AppState>) -> (StatusCode, Json<HealthResponse>) {
    let body = build_health_response(&state);
    let code = if body.status == "ok" && body.model_loaded {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (code, Json(body))
}
```

**Feature declaration analog** (`crates/aprender-serve/Cargo.toml:225` — the core dep is already
optional, gated exactly like the new feature must be):
```toml
aprender-serve = ["dep:aprender", "server"]  # Aprender ML model serving (requires server)
# NEW: setfit = ["dep:aprender", "aprender/setfit", "server"]
```

**Serve startup detection** (`crates/apr-cli/src/commands/serve/handlers.rs:336-337` +
`:741-753` — branch inside the existing APR-format arm after reading the typed tag; one detection
point, no new serve command per D-10):
```rust
pub(crate) fn start_realizar_server(model_path: &Path, config: &ServerConfig) -> Result<()> {
    use realizar::format::{detect_format, ModelFormat};
    ...
    // Read only 8 bytes for format detection (avoid loading entire file)
    let mut magic = [0u8; 8];
    let bytes_read = file.read(&mut magic)?;
    ...
    let format = detect_format(&magic)
        .map_err(|e| CliError::InvalidFormat(format!("Format detection failed: {e}")))?;
```

**Auth caveat (deferred, note-only):** `APR_API_KEY*` protects only the CPU fallback router in
`crates/apr-cli/src/commands/serve/auth.rs`, not `api::router` paths — out of scope per CONTEXT.

---

### Parity harness (D-13/D-14) — composite

**In-process HTTP leg analog** (`crates/aprender-serve/src/api/tests/app_state_default.rs:42-52`,
tower `util` already a dep):
```rust
let response = app
    .oneshot(
        Request::builder()
            .method("POST")
            .uri("/generate")
            .header("content-type", "application/json")
            .body(Body::from(json))
            .expect("test"),
    )
    .await
    .expect("test");
```
(Do not copy the multi-status tolerance at lines 54-60 — the parity gate asserts exact outcomes.)

**CLI leg:** spawn via `env!("CARGO_BIN_EXE_apr")`-style resolution in tests, never PATH; Make
targets pin via `. scripts/apr_bin.sh` (CLAUDE.md Step 0). **Library leg:** direct core call.
**In-band negative:** a deliberately skewed surface variant must FAIL in every `cargo test`
(Ph1 D-24 / Ph2 D-25 / Ph3 D-08 house discipline). **One** tier3 spawned `apr serve` smoke test on
a loopback port. Site per RESEARCH A4: `crates/apr-cli/tests/` (verify no dep cycle with
`cargo tree` at plan time; fallback is aprender-serve tests or a dedicated test crate).

---

### trybuild non-constructibility for `VerifiedSetFitModel` (APR-04)

**Analog:** `crates/aprender-train/tests/ui/setfit_external_codec_impl.rs` (full file) + the runner
`crates/aprender-train/tests/ui.rs:60` (`t.compile_fail("tests/ui/*.rs")`). The case-file shape:
a long WHY-comment naming the expected diagnostic, then a minimal complete program:
```rust
// TRN-01 / D-07 as amended by the phase-3 review — an out-of-crate codec cannot enter the
// verification path.
// ...
// Expected diagnostic: `the trait bound 'MyCodec: verify::sealed::Sealed' is not satisfied`
// (E0277), naming both `SetFitCodec` and the private `Sealed`.

use entrenar::train::setfit::verify::{CodecError, SetFitCodec};

struct MyCodec;
impl SetFitCodec for MyCodec { ... }   // must fail: Sealed is unnameable

fn main() {}
```
Phase 4's cases: out-of-crate construction of `VerifiedSetFitModel` (aprender-core needs its own
`tests/ui.rs` runner if it lacks one — check at plan time; aprender-train's exists) and, if the
codec module adds public surface, an `AprCodec`-shaped external impl case. Each `.rs` pairs with a
committed `.stderr`.

---

### `contracts/setfit-apr-v1.yaml` + Makefile + ci.yml

**Contract:** one NEW file per Ph1 D-23, referencing (never editing)
`setfit-train-lifecycle-v1.yaml`, `setfit-encoder-conformance-v1.yaml`, `tensor-names-v1.yaml`,
`tensor-layout-v1.yaml`, `apr-model-lifecycle-v1.yaml`. Owns: artifact schema, load-validation
ladder, D-03 size cap, D-11 probe policy/count, parity + probe tolerances. Validate with `pv` only
(`$(PV_BIN)`, Makefile:1136); if `pv validate` rejects the shape, follow CLAUDE.md's three
sanctioned options (usually restructure to `KernelContract` shape).

**`$(CONTRACTS)` is an EXPLICIT list** (Makefile:1138-1184, currently ends at
`contracts/linear-probe-classifier-v1.yaml`) — append `contracts/setfit-apr-v1.yaml` or it is
validated by nothing (Ph2 lesson). A scoped `PHASE4_CONTRACTS` list following `PHASE2_CONTRACTS`
(Makefile:1190-1191) / `PHASE3_CONTRACTS` is the audit-gate pattern.

**Ran-something guard** (Makefile:1381-1390 — every new name-filtered target uses it, CR-02):
```make
define assert_tests_ran
ran=$$(awk '/^test result:/ { for (i = 1; i <= NF; i++) if ($$(i+1) ~ /^passed/) s += $$i } END { print s + 0 }' $(1)); \
if [ "$$ran" -lt "$(2)" ]; then \
	echo "FAIL: $(3) reported $$ran test(s) passed, expected at least $(2)."; \
	echo "A name filter that matches nothing exits 0 (REVIEW CR-02) — this gate was"; \
	echo "about to report success having run nothing. ..."; \
	exit 1; \
fi
endef
```

**Feature-matrix analog** (Makefile:382-394, `setfit-feature-matrix` — grows apr-cli and
aprender-serve legs per SAFE-02):
```make
setfit-feature-matrix: ## D-06/D-05: setfit feature isolation ...
	@cargo check -p aprender-core --features setfit
	@cargo check -p aprender-core --features setfit,conformance-fixtures,model-tests
	...
```

**CI analog** (`.github/workflows/ci.yml` setfit step, ~lines 274-309 — **editing ci.yml requires
human check-in per CLAUDE.md autonomy rules**; the step's own comment records the CR-01 rationale
and the no-pipes rule):
```yaml
# `set -e` and no pipes: a non-zero cargo exit fails the step directly. Deliberately NOT
# the `| tee ... ; grep -q` shape used below — that reads grep's status, not cargo's
# (CLAUDE.md verification rule 1).
run: |
  ... bash -c 'set -e; \
    cargo test -p aprender-core  --features setfit --lib setfit::; \
    cargo test -p aprender-train --features setfit --lib setfit::; \
    cargo test -p aprender-train --features setfit --test ui'
```
New legs: `cargo test -p apr-cli --features setfit ...` and
`cargo test -p aprender-serve --features setfit ...` — keep tier target and CI step command-identical
(the step comment says why: "Keeping them identical stops the tier and the CI job from drifting apart").

## Shared Patterns

### Fail-closed limits BEFORE allocation
**Source:** `crates/aprender-train/src/train/setfit/bundle.rs:456-524` (excerpted above)
**Apply to:** APR loader ladder (D-03), HTTP classify request bounds (V5), config parsing.
Order: raw length → parse → schema version → declared-length arithmetic → only then decode payloads.

### NaN-visible comparison
**Source:** `crates/aprender-train/src/train/setfit/verify.rs:336-341` (`within`, excerpted above)
**Apply to:** probe replay, parity comparisons, envelope validation. Never `delta <= bound` bare —
false for NaN means silent acceptance (CR-03 shape).

### Deterministic serialization (byte-canonical obligation)
**Source:** verify.rs:31-46 module docs + bundle.rs:418-425 (`to_canonical_bytes` doc) +
header_impl.rs:185 (`created_at` stays `None`), :298 (one custom key only)
**Apply to:** the APR writer, the metadata doc, probe records. Fixed field order, BTreeMap/`serde_json::Map`,
bit-pattern floats (bundle.rs `f32_to_hex` precedent), no timestamps/env values, `float_roundtrip`
stays enabled (already on in core+train Cargo.toml). Wave-0 test: write→parse→write byte equality
across TWO processes (cross-process half catches HashMap ordering).

### Typed errors + contract-bound exit codes
**Source:** `crates/apr-cli/src/error.rs:14-81` (excerpted above)
**Apply to:** all new CLI commands. Library errors stay typed (`CodecError` preserves `BundleError`
per verify.rs:102-106 — "a caller must be able to tell a contracted limit from a parse failure
without matching on message text"); the CLI adapter translates and appends the remedy flag
(data_contrastive.rs:578-593 pattern).

### Atomic writes with `--force` gate
**Source:** `crates/apr-cli/src/commands/data_contrastive.rs:159-193` (excerpted above)
**Apply to:** `apr setfit train` output APR, any dumped goldens/manifests.

### Ran-something guards on every name-filtered gate
**Source:** `Makefile:1381-1390` (`assert_tests_ran`, excerpted above)
**Apply to:** every new Make target and CI filter this phase adds (parity gate, setfit CLI/serve
suites). A gate that can go vacuous is not a gate.

### Trusted-policy hash discipline
**Source:** verify.rs:198-205 — `artifact_hash` is a free function; "a codec that hashed its own
output could report any digest it liked."
**Apply to:** artifact hash + tokenizer SHA-256 everywhere (loader, readiness, envelope). Use `sha2`
via the existing free-function pattern; never hash inside the codec.

## No Analog Found

| File | Role | Data Flow | Reason / fallback |
|------|------|-----------|-------------------|
| Three-surface parity harness (as a whole) | test | request-response ×3 | No existing test compares library + spawned CLI + HTTP pairwise. Each LEG has an analog (oneshot: app_state_default.rs:42-52; goldens + SHA-256 manifests: Ph1 D-13 fixture discipline; in-band negative: Ph1 D-24 house style) — the composition is new. Build from RESEARCH Pattern 6. |
| Embedded probe records (D-11, artifact-resident) | data schema | transform | Train-time `VerifyProbe`/`probe_model` (verify.rs:311-327) is the nearest shape, but probes must be dataset-independent synthetic strings (Pitfall 8 — no TweetEval text in the artifact). Contract-resident set, 3-8 strings covering short/long/truncation-boundary/unicode. |

## Metadata

**Analog search scope:** `crates/aprender-train/src/train/setfit/`, `crates/aprender-core/src/setfit/`,
`crates/apr-format/src/v2/`, `crates/apr-cli/src/{commands/,*.rs}`, `crates/aprender-serve/src/api/`,
`Makefile`, `.github/workflows/ci.yml`, `crates/aprender-train/tests/ui/`
**Files scanned:** 20 read directly (targeted, non-overlapping ranges), plus grep pinning across 8 more
**Pattern extraction date:** 2026-08-14

**Amended 2026-08-14 (checker warning W-D).** The original File Classification table was mapped
before plans 04-12..04-16 existed and before wave 2 was split into 04-02 / 04-13 / 04-14, so it
omitted `apr_reload.rs`, `bundle.rs`/`bundle_tests.rs`/`verify_tests.rs`,
`config.rs`/`lock.rs`/`lock_tests.rs`/`evaluate.rs`, `setfit_io.rs`, `encoder.rs`,
`tests/setfit_apr_lifecycle.rs` and `tests/setfit_cli_lifecycle.rs`. Rows for all of them are
appended under "Appended 2026-08-14" above. Note for later readers: plans 04-12..04-16 already carry
their analogs INLINE in each task's `<read_first>` (04-13 cites `evidence.rs:200-222`'s
`first_null_path` shape; 04-14 cites `bundle.rs:437-470`'s fail-closed door discipline; 04-16 cites
verify.rs's policy shape), so the omission was artifact staleness, not a pattern violation — the
plans were never analog-less.
