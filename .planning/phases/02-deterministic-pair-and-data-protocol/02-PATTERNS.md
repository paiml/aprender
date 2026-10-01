# Phase 2: Deterministic Pair and Data Protocol - Pattern Map

**Mapped:** 2026-08-08
**Files analyzed:** 27 (17 new-crate files, 10 modified/extended files)
**Analogs found:** 22 / 27 (5 files have no close in-repo analog — listed at the end)

All analog paths verified by direct read this session. Line numbers refer to current
working-tree state (branch `gsd/phase-1-differentiable-minilm-conformance`, which includes the
uncommitted D-06 baseline files).

## File Classification

### New crate: `crates/aprender-contrastive-data/`

| New File | Role | Data Flow | Closest Analog | Match Quality |
|----------|------|-----------|----------------|---------------|
| `Cargo.toml` | config | — | `crates/aprender-rand/Cargo.toml` | exact |
| `src/lib.rs` | library entry | — | `crates/aprender-rand/src/lib.rs` | exact |
| `src/error.rs` | error types | — | `crates/aprender-rand/src/error.rs` + `data_tweeteval.rs` message style | exact |
| `src/schema.rs` | model + parse | transform (bytes→typed) | `crates/apr-cli/src/commands/data_tweeteval.rs` (`load_split`, `encode_jsonl`) | exact |
| `src/split.rs` | model (typestate) | transform | `crates/aprender-core/src/format/validated_tensors.rs:402` + `validated_vector.rs` | role-match (strong) |
| `src/hash.rs` | utility | transform | `data_tweeteval.rs` (`sha256`, hash-then-parse) | exact |
| `src/dedup.rs` | service | batch/transform | partial — `data_tweeteval.rs` BTreeMap discipline only | partial |
| `src/buckets.rs` | model | batch | `crates/aprender-data/src/split.rs::stratified` (line 180) | role-match |
| `src/select.rs` | service | batch/transform | `aprender-data/src/split.rs::stratified` + `aprender-rand/src/philox.rs` | role-match |
| `src/pairs.rs` | service | streaming | none (RESEARCH.md Code Examples govern) | no analog |
| `src/prepared.rs` | model (profile typestate) | transform | `crates/aprender-core/src/format/validated_vector.rs` (PhantomData + SOLE validating constructor + "There is no way to create …" prose) — this is the analog for `PreparedDataset<Canonical>` / `PreparedDataset<Compatibility>` and for stating impossibility in doc words | role-match (strong) |
| `src/attestation.rs` | model + boundary validator | transform (bytes→verified identity) | `data_tweeteval.rs` manifest family (lines 64–113, nested Serialize structs + BTreeMap) for the record shape; `schema.rs`'s `deny_unknown_fields` boundary discipline for the untrusted-parse rule | role-match |
| `src/rng.rs` | utility | request-response (pure fn) | `crates/aprender-rand/src/philox.rs` (`generate_at`) | exact |
| `src/ledger.rs` | model (append-only record) | event-driven | none | no analog |
| `src/manifest.rs` | model + serialization | transform | `data_tweeteval.rs` manifest structs (lines 64–113) | exact |
| `tests/negative_leaky.rs` | test (in-band negative) | — | `crates/aprender-core/tests/setfit_conformance/detach_negative.rs` | exact discipline |
| `tests/negative_materializing.rs` | test (in-band negative) | — | same as above | exact discipline |
| `tests/goldens/` + `manifest.sha256` | fixtures | — | `crates/aprender-core/tests/fixtures/setfit/` + verifier at `tests/setfit_conformance.rs:1199` | exact |
| `tests/ui/` (trybuild) | test (compile-fail) | — | none — trybuild is a dev-dep of 2 crates but no `TestCases` harness exists | no analog |
| `tests/common/mod.rs` | shared test module (NOT a test target) | — | `crates/aprender-core/tests/setfit_conformance.rs`'s `#[path = "..."]` module include — same problem, different idiom; `tests/common/mod.rs` is preferred because cargo already treats that path as a module | role-match |
| `tests/reference_fixtures.rs` | test (integration, fixture integrity) | — | `crates/aprender-core/tests/setfit_conformance.rs:1199` manifest verification | exact |
| `tests/pair_counts.rs` | test (integration, fixture-driven counts) | — | same as `reference_fixtures.rs`; exists because `tests/common/` is invisible to lib unit tests and `src/` may not touch the filesystem | exact |
| `tests/setfit_reference/` + `manifest.sha256` | fixtures | — | `crates/aprender-core/tests/fixtures/setfit/` + its Phase 1 manifest writer | exact |
| `allowed-deps.txt` | build config (positive allowlist) | — | none in-repo — the closest discipline is `Makefile setfit-feature-matrix`'s capture-then-check status handling, which the consuming target copies | no analog |

### Modified / extended files

| File | Role | Data Flow | Analog / Baseline | Match Quality |
|------|------|-----------|-------------------|---------------|
| Root `Cargo.toml` (members + workspace deps) | config | — | its own existing entries (lines 27, 208, 239) | exact |
| `crates/apr-cli/Cargo.toml` (new dep) | config | — | `alimentar = { workspace = true }` at line 232 | exact |
| `crates/apr-cli/src/commands/data_tweeteval.rs` (thin-adapter refactor) | CLI command | request-response + file I/O | itself — the D-06 baseline IS the pattern to preserve | exact (self) |
| `crates/apr-cli/src/data_commands.rs` (new `Select`/`Pairs` variants) | CLI args | request-response | its own `Split` variant (lines 43–66) | exact |
| `crates/apr-cli/src/dispatch_analysis.rs` (dispatch arms) | dispatch | request-response | `dispatch_data_command` (lines 726–791) | exact |
| `contracts/contrastive-pair-protocol-v1.yaml` (new) | contract | config | `contracts/setfit-encoder-conformance-v1.yaml` | exact (only pv-passing precedent) |
| `contracts/tweet-eval-stance-benchmark-v1.yaml` (restructure) | contract | config | same as above | exact |
| `Makefile` (tier2 line, `$(CONTRACTS)`, bytes-boundary check) | build config | — | tier2 block (~185–213), `setfit-feature-matrix` (~266–291), `CONTRACTS` (~874+) | exact |
| `scripts/setfit_fixtures/generate_fixtures.py` (extend) | script | batch | itself (`main()` line 352, manifest writer lines 913–928) | exact (self) |
| `#[contract]` annotations (in new crate source) | annotation | — | `crates/aprender-core/src/autograd/ops/pooling.rs:50–53` | exact |
| `crates/apr-cli/src/commands/data_contrastive.rs` (new) | CLI command | request-response + file I/O | `crates/apr-cli/src/commands/data_tweeteval.rs` — read-once bytes, `create_new` no-clobber + `--force`, `written: Vec<PathBuf>` rollback, `--json` output shape | exact |
| `contracts/aprender/binding.yaml` (append) | contract binding registry | config | its own existing entries (contract / equation / module_path / function / signature / status / notes) | exact (self) |
| `docs/examples/tweet-eval-stance.md` (extend) | docs | — | itself — the D-06 baseline landed it in plan 02-01 | exact (self) |
| `CLAUDE.md` (`pv diff` example fix, plan 02-01) | docs | — | itself; the corrected form comes from `crates/aprender-contracts-cli/src/cli.rs:73-78` | exact (self) |

## Pattern Assignments

### `Cargo.toml` (new crate) — analog: `crates/aprender-rand/Cargo.toml`

The workspace's precedent for a small, publishable, dependency-light crate with a diverging
library name (lines 1–24):

```toml
[package]
name = "aprender-rand"
version.workspace = true
edition = "2021"
description = "Counter-based parallel RNG — Philox 4x32-10 with provable statistical properties"
license = "MIT OR Apache-2.0"
repository = "https://github.com/paiml/trueno"
keywords = ["rng", "random", "philox", "parallel", "counter-based"]
categories = ["science", "algorithms", "mathematics"]

[lib]
name = "trueno_rand"

[dependencies]
thiserror = "2"

[dev-dependencies]
proptest = "1"
```

Differences for the new crate: no `[lib] name` override needed (library name =
`aprender_contrastive_data` is fine); deps are `sha2`, `serde`, `serde_json`, `thiserror`,
`unicode-normalization`, `trueno_rand` (via `aprender-rand = { workspace = true }`),
`provable-contracts-macros`; dev-deps `proptest`, `trybuild`. NOTE: `aprender-rand` carries its
own `[lints]` tables (it predates workspace lints) — the new crate should instead use
`[lints] workspace = true` like the aprender-* crates (verify against a recent member, e.g.
`crates/aprender-train-common/Cargo.toml`). No `publish = false` — publishing is mandatory
(RESEARCH Finding F5, publish before `apr-cli` in the cascade).

**Workspace wiring** — copy the exact shape of root `Cargo.toml` line 239
(`aprender-data = { path = "crates/aprender-data", version = "0.63.0" }`):

```toml
# root Cargo.toml [workspace] members (alphabetical-ish block near line 27)
"crates/aprender-contrastive-data",
# root Cargo.toml [workspace.dependencies]
aprender-contrastive-data = { path = "crates/aprender-contrastive-data", version = "0.63.0" }
```

Consumption in `crates/apr-cli/Cargo.toml` — copy line 230–232's commented style:

```toml
# Data loading, splitting, imbalance detection (apr data subcommands)
# GH-344: path dep via .cargo/config.toml.dev-overrides; CI uses checkout+symlink
alimentar = { workspace = true }
```

After the workspace edit, re-run `scripts/check_include_files.sh` and
`scripts/check_package_includes.sh` (CB-510 discipline, CLAUDE.md).

---

### `src/lib.rs` — analog: `crates/aprender-rand/src/lib.rs` (lines 1–31)

Crate docs that name the governing contract, a runnable determinism doctest, flat `mod` +
`pub use` re-exports:

```rust
//! Counter-based parallel random number generation.
//!
//! # Contract: rand-philox-v1.yaml
//!
//! Implements Philox 4x32-10 (Salmon et al., SC 2011), ...
//!
//! # Example
//!
//! ```
//! use trueno_rand::Philox4x32;
//! let mut rng = Philox4x32::new(42);
//! let vals = rng.next_4u32();
//! // Deterministic: same seed → same output
//! let mut rng2 = Philox4x32::new(42);
//! assert_eq!(rng2.next_4u32(), vals);
//! ```

mod error;
mod philox;
...
pub use error::RngError;
pub use philox::Philox4x32;
```

For the new crate, the `# Contract:` header line must cite
`contrastive-pair-protocol-v1.yaml`. Caution: `rand-philox-v1.yaml` itself is a dangling
reference (not in this repo — RESEARCH Open Question 4); do not copy that citation, only the
convention.

---

### `src/error.rs` — analog: `crates/aprender-rand/src/error.rs` (lines 1–13, complete file)

The minimal thiserror template, one variant per failure class:

```rust
//! RNG error types.

/// Errors from RNG operations.
#[derive(Debug, thiserror::Error)]
pub enum RngError {
    /// Output buffer is empty.
    #[error("output buffer must not be empty")]
    EmptyBuffer,

    /// Invalid standard deviation for normal distribution.
    #[error("standard deviation must be positive, got {0}")]
    InvalidStdDev(f32),
}
```

`ContrastiveDataError` needs one variant per DATA-02 failure class (malformed row, non-UTF-8,
length mismatch, unknown label, invalid class counts, duplicate ID, conflicting source role,
cross-split duplicate underflow) plus `SelfPair { id }` (D-12), `BudgetExceedsCapacity` (D-11),
and boundary-validation variants (D-16). For error MESSAGE quality, copy
`data_tweeteval.rs`'s style — messages carry the split name, the index, and both expected and
got values (lines 331–335):

```rust
return Err(CliError::ValidationFailed(format!(
    "TweetEval {canonical_name} class-count contract failed: expected {expected_counts:?}, got {counts:?}"
)));
```

Error variant *documentation* style: copy the `# Errors` doc-comment ladder from
`crates/aprender-core/src/autograd/ops/pooling.rs` lines 41–49 (one bullet per typed variant,
stating the condition).

---

### `src/schema.rs` — analog: `crates/apr-cli/src/commands/data_tweeteval.rs`

**Imports pattern** (lines 7–17 — for the CLI-side; the crate itself imports no fs/path):

```rust
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
```

**Row schema** (lines 47–54) — this is the JSONL schema the crate's `LabeledExample`
generalizes; the relocated type must keep byte-compatible field order or bump
`schema_version`:

```rust
#[derive(Debug, Clone, Serialize)]
struct StanceSample {
    id: String,
    input: String,
    label: usize,
    label_text: &'static str,
    source_split: String,
}
```

**Typed validation ladder** (lines 304–329, `load_split`) — every branch is a typed error
naming split + index; this is the DATA-02 shape to preserve and extend:

```rust
for (index, (input, raw_label)) in texts.iter().zip(raw_labels.iter()).enumerate() {
    if input.trim().is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "TweetEval {canonical_name} sample {index} has empty text"
        )));
    }
    let label = raw_label.trim().parse::<usize>().map_err(|error| {
        CliError::ValidationFailed(format!(
            "Invalid TweetEval label '{}' in {canonical_name} sample {index}: {error}",
            raw_label.trim()
        ))
    })?;
    let label_text = LABEL_NAMES.get(label).copied().ok_or_else(|| { ... })?;
    counts[label] += 1;
    samples.push(StanceSample { id: format!("{canonical_name}:{index}"), ... });
}
```

**Deterministic JSONL encoding** (lines 436–445) — `serde_json::to_writer` per row + `\n`,
never line-joined strings:

```rust
fn encode_jsonl(samples: &[StanceSample]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for sample in samples {
        serde_json::to_writer(&mut bytes, sample).map_err(|error| { ... })?;
        bytes.push(b'\n');
    }
    Ok(bytes)
}
```

For deserialization the crate adds `#[derive(Deserialize)]` with
`#[serde(deny_unknown_fields)]` (RESEARCH Security V5) — no existing analog does this; it is a
deliberate strictening at the bytes→typed boundary.

---

### `src/split.rs` (typestate) — analog: `crates/aprender-core/src/format/validated_tensors.rs:402` + `validated_vector.rs`

The repo's one strong typestate precedent — `ValidatedWeight<L = RowMajor>` with
`PhantomData`, a SOLE validating constructor, and impossibility-by-nonexistence
(`validated_vector.rs` lines 2–34; struct at `validated_tensors.rs:402`; note
`validated_vector.rs` is an `include!` fragment of `validated_tensors.rs`, line 411):

```rust
// validated_tensors.rs:402
pub struct ValidatedWeight<L = RowMajor> { ... _layout: PhantomData<L> }

// validated_vector.rs:2-19
impl ValidatedWeight<RowMajor> {
    /// Construct a validated row-major weight matrix.
    ///
    /// This is the ONLY constructor. There is no way to create a
    /// `ValidatedWeight<ColumnMajor>` because `ColumnMajor` does not exist.
    ///
    /// # Errors
    /// Returns `ContractValidationError` if validation fails.
    pub fn new(
        data: Vec<f32>,
        out_dim: usize,
        in_dim: usize,
        name: &str,
    ) -> Result<Self, ContractValidationError> {
        // Gate 1: Shape validation ... Gate 5: L2 norm validation
        Ok(Self { data, out_dim, in_dim, name: name.to_string(), stats, _layout: PhantomData })
    }
```

Copy: private fields, `PhantomData<R>` role parameter, sole fallible constructor running the
full validation gate ladder, doc comment stating the impossibility in words. For
`Split<Train>/<Validation>/<Test>/<CompatibilityTest>` the roles are four ZSTs; the
constructor is the bytes→typed boundary (`from_jsonl_bytes(bytes, decl)`) per RESEARCH
Pattern 1. The numbered "Gate N:" comment convention (Gate 1..Gate 5) maps directly onto the
DATA-02 validation ladder.

---

### `src/hash.rs` — analog: `data_tweeteval.rs`

**Hash-from-parsed-bytes discipline** (lines 244–257 — the comment is load-bearing, keep it
with the code across the D-05 seam):

```rust
// Read every source file exactly once, then both hash and parse *those*
// bytes. Reading twice would let the recorded SHA-256 describe content
// that never passed the class-count contract.
let mut raw: BTreeMap<String, Vec<u8>> = BTreeMap::new();
for filename in SOURCE_FILES {
    raw.insert(filename.to_string(), read_required(&source_dir.join(filename))?);
}
let source_sha256: BTreeMap<String, String> = raw
    .iter()
    .map(|(name, bytes)| (name.clone(), sha256(bytes)))
    .collect();
```

**Hex digest helper** (lines 543–545):

```rust
fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
```

The normalized hash (`nfc-trim-ws-v1`) has no in-repo analog; use RESEARCH.md's Code Example
verbatim (NFC via `unicode_normalization::UnicodeNormalization::nfc()`, `split_whitespace`
collapse, NO casefold, versioned constant `CONTENT_NORMALIZATION_VERSION`).

---

### `src/buckets.rs` + `src/select.rs` — analog: `crates/aprender-data/src/split.rs::stratified` (lines 180–246)

The balanced-selection precedent: group indices by label, deterministic per-group shuffle with
a label-derived seed, take a prefix per group:

```rust
// Group indices by label value
let groups = group_by_label(label_array)?;
...
for (label_value, mut indices) in groups {
    // Shuffle within group
    if seed.is_some() {
        // Simple deterministic shuffle using label as additional seed component
        let group_seed = base_seed.wrapping_add(label_value as u64);
        shuffle_indices(&mut indices, group_seed);
    }
    let group_len = indices.len();
    let group_train = ((group_len as f64) * train_ratio).round() as usize;
    ...
    train_indices.extend_from_slice(&indices[..group_train]);
}
```

Copy the SHAPE (per-class buckets → per-class deterministic shuffle → prefix), NOT the
mechanics: `wrapping_add(label)` seed derivation and its shuffle are exactly what D-20/D-21
replace with domain-separated Philox (`key = trunc64(SHA-256(domain_tag ‖ root_seed ‖
"select/{class_label}"))`, partial Fisher–Yates driven by draw ordinal — RESEARCH Patterns
2–3). Also copy the fail-closed ratio/feasibility validation up front (lines 188–199: validate
config, reject empty input with a typed error, THEN compute). Sorted `Vec` buckets, never
HashMap iteration (RESEARCH Anti-Patterns).

---

### `src/rng.rs` — analog: `crates/aprender-rand/src/philox.rs`

**The stateless primitive** (lines 97–100) — the entire basis of D-20:

```rust
/// Generate values for a specific counter (stateless, for GPU-style parallel generation).
pub fn generate_at(key: [u32; 2], counter: [u32; 4]) -> [u32; 4] {
    philox4x32_10(counter, key)
}
```

Plus `with_key_counter` (lines 42–44) if a stream handle is ever needed. Import as
`use trueno_rand::Philox4x32;` (library name, not directory name). The domain-separation key
derivation and multiply-shift bounded draw have no in-repo analog — use RESEARCH Pattern 2
verbatim, and contract them in `contrastive-pair-protocol-v1.yaml` (Open Question 4: state RNG
obligations inline; do NOT cite `rand-philox-v1.yaml`, which does not exist in this repo).
Never `next_f32` for index draws (23-bit mantissa — see philox.rs lines 59–62 for why).

---

### `src/manifest.rs` — analog: `data_tweeteval.rs` manifest family (lines 64–113)

Nested `Serialize` structs, `BTreeMap` everywhere ordering matters, doc comments carrying
honesty semantics:

```rust
#[derive(Debug, Serialize)]
struct SourceManifest {
    repository: &'static str,
    data_path: &'static str,
    revision: String,
    /// True only when this run fetched the files from the pinned revision.
    /// With `--source` the revision is user-asserted and unverified, so
    /// consumers must not treat it as provenance.
    revision_verified: bool,
    files_sha256: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
struct BenchmarkManifest {
    schema_version: u32,
    dataset: &'static str,
    ...
    source: SourceManifest,
    splits: BTreeMap<String, SplitManifest>,
    ...
}
```

Serialization to bytes (lines 510–513): `serde_json::to_vec_pretty` + trailing `\n`. The
selection manifest adds a `semantic_hash` over canonical bytes with volatile metadata OUTSIDE
the hashed region (RESEARCH Pattern 7 — no analog; the `data_tweeteval` manifest is unhashed).
The `revision_verified` doc-comment pattern is the template for `singleton_policy` /
`singleton_policy_version` and the excluded-IDs record (D-13, D-18): state the honesty
semantics in the field's doc.

---

### `tests/negative_leaky.rs` / `tests/negative_materializing.rs` — analog: `crates/aprender-core/tests/setfit_conformance/detach_negative.rs` (full file, 153 lines)

The Phase 1 D-24 in-band negative — the exact discipline D-25 mandates for this phase's two
fakeable claims. Three structural elements to copy:

**1. Header doc explaining why the negative exists** (lines 1–18):

```rust
//! D-24 — the gradient gate is not theater.
//!
//! Obligation: `OBLIG-DETACH-REJECTION`.
//!
//! A gate that has only ever been observed passing is not evidence. This file
//! builds a deliberately DETACHED encoder variant — identical arithmetic, the
//! pooled output rebuilt with `Tensor::from_vec` so the graph is severed — and
//! requires the SAME helper that accepts the real encoder to REJECT it.
```

**2. The negative must fail for the RIGHT reason, and the error must NAME the offenders**
(lines 61–98):

```rust
// The leaf DID receive gradient: the loss is differentiable, the severance
// is upstream. Without this the assertion below would also pass against a
// loss that is not differentiable at all.
let leaf_grad = get_grad(za.id()).expect("the rebuilt leaf must receive a gradient");
...
let report = result.expect_err(
    "the ENC-04 gate ACCEPTED a detached encoder. The gate cannot distinguish a \
     connected graph from a severed one and every positive result in this harness is \
     worthless.",
);
assert!(report.contains("received NO gradient"), ...);
// The message must NAME the parameters, because that is what makes a real
// failure diagnosable rather than merely red.
for probe in ["embeddings.word_embeddings.weight", ...] { assert!(report.contains(probe), ...); }
```

**3. The mirror test — the same gate call must ACCEPT the honest implementation**
(lines 128–153):

```rust
#[test]
fn detach_negative_the_connected_encoder_passes_the_same_call() {
    // The mirror. Without it, "the gate rejects the detached variant" would also
    // be satisfied by a gate that rejects everything.
    ...
    super::assert_encoder_updates(&GateInput { ... })
        .expect("the identical call must ACCEPT the connected encoder");
}
```

There is also a same-helper meta-test (lines 101–126) asserting all gate files call ONE shared
helper — adopt this if the leaky/materializing gates share a membership/capacity checker. For
`negative_materializing.rs`: retained state must be measured structurally (`seen.len()`,
`#[cfg(test)]` introspection accessor), never self-reported (RESEARCH Pattern 6).

**Integration-test module wiring** — copy `crates/aprender-core/tests/setfit_conformance.rs`
lines 109–114:

```rust
#[path = "setfit_conformance/detach_negative.rs"]
mod detach_negative;
```

(For the new crate, plain top-level `tests/*.rs` files also work since there is no feature
gating; the `#[path]` split is only needed if the tests share a common harness module.)

---

### `tests/goldens/` + fixture manifest — analogs: `crates/aprender-core/tests/fixtures/setfit/` and `scripts/setfit_fixtures/generate_fixtures.py`

The Phase 1 D-13 pattern. On-disk layout: fixture JSONs + one `manifest.sha256`
(`crates/aprender-core/tests/fixtures/setfit/manifest.sha256` exists, 1.4K, covering 10
files). Generator side (`generate_fixtures.py` lines 913–928):

```python
files = sorted(p for p in FIXTURE_DIR.iterdir() if p.is_file() and p.name != "manifest.sha256")
...
(FIXTURE_DIR / "manifest.sha256").write_text("\n".join(lines) + "\n")
subprocess.run(["shasum", "-a", "256", "-c", "manifest.sha256"], ...)
print(f"manifest.sha256 covers {len(files)} files; shasum -c passed")
```

Verifier side: `crates/aprender-core/tests/setfit_conformance.rs:1199–1200` reads
`manifest.sha256` and checks each fixture's digest in-test. Copy both halves: Rust goldens for
pair identities get their own `manifest.sha256` in the new crate's test tree; SetFit
count fixtures are emitted by EXTENDING `generate_fixtures.py` (its `main()` at line 352 shows
the sectioned "corpus of record → per-family fixture" structure and the FATAL-on-inconsistency
style to follow). Per RESEARCH Pitfall 1: fixture families must record MEASURED reference
behavior (self-pairs included) separately from Aprender's contracted counts.

---

### CLI: `data_commands.rs` new variants — analog: its own `Split` variant (lines 43–66)

```rust
/// Stratified train/val/test split preserving class proportions
Split {
    /// Path to JSONL data file
    #[arg(value_name = "FILE")]
    file: PathBuf,
    /// Training set fraction
    #[arg(long, default_value = "0.8")]
    train: f64,
    ...
    /// Random seed for deterministic split
    #[arg(long, default_value = "42")]
    seed: u64,
    /// Output directory for split files
    #[arg(short, long)]
    output: PathBuf,
},
```

Copy: doc-comment per variant AND per field (clap renders them), `value_name` on positional
paths, `#[arg(long, default_value = ...)]` for seeds/knobs. The `TweetEvalStance` variant
(lines 7–24) shows the const-backed default pattern
(`default_value = crate::commands::data_tweeteval::CANONICAL_REVISION`) — reuse for the
contracted seed list / shots enum. Recommended shapes are in RESEARCH Open Question 5
(`apr data select`, `apr data pairs`, dump as a flag).

### CLI: `dispatch_analysis.rs` new arms — analog: `dispatch_data_command` (lines 726–743)

```rust
/// Dispatch `apr data` subcommands to alimentar-backed implementations.
fn dispatch_data_command(command: &DataCommands, cli: &Cli) -> std::result::Result<(), CliError> {
    let json = cli.json;
    match command {
        DataCommands::TweetEvalStance { output, profile, source, revision, force } =>
            commands::data_tweeteval::run(
                output, *profile, source.as_deref(), revision, *force, cli.offline, json,
            ),
        ...
    }
}
```

Copy: destructure the variant, deref copies, `as_deref()` options, pass `cli.offline` and
`cli.json` through explicitly. Note line 795: `dispatch_train_command` carries a
`#[provable_contracts_macros::contract(...)]` annotation on the dispatcher itself — precedent
if the phase contract wants a CLI-surface obligation.

### CLI: `data_tweeteval.rs` thin-adapter refactor — baseline is itself

Behaviors that MUST survive the D-05 seam unchanged (each is guarded by an existing test in
lines 562–813):

- `validate_revision` 40-hex check (lines 202–209)
- read-once/hash-then-parse (lines 244–257; Pitfall 7 — never two `fs::read` per file)
- `revision_verified = downloaded.is_some()` honesty (line 148; test at lines 731–754,
  FALSIFY-TWEET-EVAL-006)
- rollback-on-partial-write (lines 489–519: collect `written: Vec<PathBuf>`, `remove_all` on
  any failure) and `create_new` no-clobber without `--force` (lines 529–541)
- stale `validation.jsonl` removal on `--force` setfit re-prepare (lines 500–508)
- opt-in network test pattern:
  `#[ignore = "opt-in network test against the pinned canonical TweetEval revision"]`
  (lines 803–812) — reuse verbatim for the real-duplicate dedup golden (RESEARCH Pitfall 3)
- CLI test-fixture generator `write_fixture_split`/`write_canonical_fixture` (lines 566–585) —
  the synthetic-fixture pattern for the new error paths (label-conflict duplicates need a
  synthetic fixture; the real data has none)

---

### Contracts — analog: `contracts/setfit-encoder-conformance-v1.yaml` (the only pv-passing precedent; verified in RESEARCH)

Top-level shape `pv validate` accepts (section starts: `contract:` 1, `metadata:` 2,
`equations:` 97, `proof_obligations:` 360, `falsification_tests:` 613, `kani_harnesses:` 812,
`qa_gate:` 849).

**Metadata + rationale-in-description style** (lines 1–15):

```yaml
contract: setfit-encoder-conformance
metadata:
  version: 2.0.0
  created: '2026-08-08'
  author: PAIML Engineering
  description: >
    Phase 1 conformance gate for the differentiable MiniLM SetFit sentence encoder
    (ENC-01..ENC-06). Declares every equation the phase binds against ...
```

**Equation shape** (lines 102–125) — formula, domain/codomain, invariants as prose bullets,
pre/postconditions:

```yaml
equations:
  embedding_gather:
    formula: out[b][s][h] = W[ids[b*S + s]][h] ; dW[ids[i]] += grad_out[i] (scatter-ADD)
    domain: W in R^{V x H} row-major, ids in {0..V-1}^{B*S}, B >= 1, S >= 1, H >= 1
    codomain: R^{B x S x H} row-major
    invariants:
    - 'Row-major layout (LAYOUT-001): out is indexed b*S*H + s*H + h ...'
    - Fail-closed on out-of-vocabulary ids — an id at or above V is a typed error, never
      a zero-filled row ...
    preconditions:
    - 'ids.len() > 0'
    postconditions:
    - 'result.len() > 0'
```

**Proof obligation shape** (lines 362–372):

```yaml
proof_obligations:
- type: equivalence
  property: >
    OBLIG-ENC-02-TOKENIZER-PARITY: for every sentence in the frozen fixture corpus, the
    Rust batched tokenizer produces ... IDENTICAL to the pinned HuggingFace tokenizer ...
    Owned by plan 01-05.
  formal: rust_batch.input_ids == fixture.input_ids && ...
  applies_to: all
```

**Falsification test shape** (lines 615–628) — id, rule, prediction, runnable test command,
expected output, diagnosis:

```yaml
- id: FALSIFY-SETFIT-ENC-001
  rule: OBLIG-DETACH-REJECTION — out-of-vocabulary ids fail closed
  prediction: >
    embedding_gather with an id at or above vocab_size returns
    Err(OpError::OutOfVocabulary) naming the id and its position ...
  test: cargo test -p aprender-core --lib embedding_gather
  test_harness: cargo test -p aprender-core --lib embedding_gather
  expected_output: 'test result: ok'
  if_fails: >
    The gather copied the aprender-train zero-fill ... Validate ids
    against vocab_size BEFORE any allocation and return a typed error.
```

**Kani harness + qa_gate shape** (lines 813–864):

```yaml
kani_harnesses:
- id: KANI-SETFIT-ENC-001
  obligation: OBLIG-DETACH-REJECTION
  property: >
    embedding_gather never indexes outside the weight buffer ...
  bound: 4
  strategy: bounded_int
  solver: cadical
  harness: verify_embedding_gather_bounds

qa_gate:
  id: F-SEC-001
  name: setfit-encoder-conformance-v1 Contract
  checks:
  - validation
  - falsification
  pass_criteria: >
    pv validate exits 0; every falsification test above is green; the in-band detached
    encoder variant is RED under the same gradient gate
```

Note the qa_gate's `pass_criteria` explicitly requires the in-band negative to be RED — copy
this for the leaky/materializing negatives. Validate with `pv` only
(`PV_BIN := cargo run --release -p aprender-contracts-cli --bin pv --`, Makefile line 874;
prebuilt at `target/release/pv`). The existing `tweet-eval-stance-benchmark-v1.yaml` fails
PROVABILITY-001 ×2 today (no `proof_obligations`, no `kani_harnesses`) — restructure it to
this shape (RESEARCH Pitfall 4, Assumption A2).

---

### `Makefile` changes — analogs: three existing blocks

**1. tier2 wiring + measured-runtime comment discipline** (lines ~189–213). The comment block
is the pattern: placement decisions are recorded WITH measurements, and gates live inside a
tier:

```make
# Phase 1 SetFit conformance (D-26). The gates must live INSIDE a tier: a target
# outside the tiers is a target that stops being run.
#
# PLACEMENT WAS MEASURED, not assumed (2026-08-08, warm tree):
#   whole conformance suite, ONE invocation ......  7 s wall / 0.57 s test time
#   ...
	@echo "Phase 1 SetFit: encoder/tokenizer/import/loss/model unit gates..."
	@cargo test -p aprender-core --lib --features conformance-fixtures setfit::
```

Add one analogous line pair for `cargo test -p aprender-contrastive-data`, with a measured
runtime comment.

**2. Bytes-boundary / dependency-closure check** — copy `setfit-feature-matrix`
(lines ~266–291), including its capture-then-check status discipline (CLAUDE.md rule 1 —
never read `$?` through a pipe):

```make
setfit-feature-matrix: ## D-06: setfit feature isolation for aprender-core
	@cargo check -p aprender-core --no-default-features
	...
# The tree is captured to a file and `cargo tree`'s own status checked FIRST.
# Piping straight into `grep -q` would read grep's status, and a `cargo tree`
# that failed outright would feed grep nothing — the guard would then pass
# vacuously, which is exactly the CLAUDE.md rule 1 failure mode.
	@cargo tree -p aprender-core --no-default-features -e normal \
		> target/setfit-feature-matrix-tree.txt 2>&1 || \
		{ echo "FAIL: cargo tree failed; the D-06 negative check would pass vacuously"; \
		  cat target/setfit-feature-matrix-tree.txt; exit 1; }
	@if grep -q tokenizers target/setfit-feature-matrix-tree.txt; then \
		echo "FAIL: tokenizers leaked into a no-default-features build (D-06)"; ...; fi
```

The D-04 check is the same shape with `-p aprender-contrastive-data` and a deny-list
(`ureq|tokio|arrow|memmap2|aws-sdk`), plus the no-`std::fs`/`std::net`-outside-`#[cfg(test)]`
source assertion (RESEARCH Pattern 5).

**3. `$(CONTRACTS)` list** (lines 876+) — an EXPLICIT hardcoded list, not a glob. Append both
phase contracts as new continuation lines; adding the YAML files alone does nothing
(RESEARCH Pitfall 4):

```make
CONTRACTS := contracts/softmax-kernel-v1.yaml \
             contracts/rmsnorm-kernel-v1.yaml \
             ...
             contracts/contrastive-pair-protocol-v1.yaml \
             contracts/tweet-eval-stance-benchmark-v1.yaml
```

---

### `#[contract]` annotations — analog: `crates/aprender-core/src/autograd/ops/pooling.rs:50–53`

```rust
#[provable_contracts_macros::contract(
    "setfit-encoder-conformance-v1",
    equation = "masked_mean_pool"
)]
pub fn masked_mean_pool(hidden: &Tensor, mask: &[u8]) -> Result<Tensor, OpError> {
```

Annotate the crate's contract-bound functions (`from_jsonl_bytes`, selection, capacity math,
pair emission) with `("contrastive-pair-protocol-v1", equation = "...")` matching the YAML's
equation names. `dispatch_analysis.rs:795` shows the same macro on a CLI dispatcher.
`crates/aprender-contracts-macros/tests/contract_macros.rs` shows the `requires`/`ensures`/
`invariant` forms if fine-grained pre/postconditions are wanted.

## Shared Patterns

### Deterministic collections
**Source:** `data_tweeteval.rs` throughout (lines 13, 61, 73, 109, 247, 378)
**Apply to:** every crate module that feeds a hash or a manifest
`BTreeMap` for all keyed data whose order reaches bytes; sorted `Vec` for buckets. HashMap
iteration is banned in deterministic paths (RESEARCH Anti-Patterns / PF-006).

### Typed errors, no unwrap
**Source:** workspace `.clippy.toml` disallowed-methods; `data_tweeteval.rs` uses
`expect("label text comes from LABEL_NAMES")` (line 455) only where the invariant is stated
**Apply to:** all crate and CLI code. Every fallible path returns
`Result<_, ContrastiveDataError>` (crate) or `CliError` (CLI); `expect()` messages state the
invariant that makes the panic impossible.

### Hash-from-parsed-bytes
**Source:** `data_tweeteval.rs` lines 244–257
**Apply to:** `schema.rs`, `hash.rs`, the CLI adapter, and the crate's ingest boundary —
`ingest(bytes) -> (hash, rows)` from ONE buffer. Two reads of the same source is a defect
(RESEARCH Pitfall 7).

### Capture-then-check status in Make/scripts
**Source:** Makefile `setfit-feature-matrix` block (~lines 279–291)
**Apply to:** every new Makefile target and script. Never read `$?` through a pipe; check the
producing command's status before grepping its output (CLAUDE.md Verification Discipline 1).

### In-band negative + mirror
**Source:** `crates/aprender-core/tests/setfit_conformance/detach_negative.rs`
**Apply to:** both D-25 negatives, and to the contract qa_gate's `pass_criteria` text. Three
parts each time: (a) wrong-for-exactly-one-reason variant, (b) assert the gate rejects it with
a diagnosable named-offender error, (c) mirror test proving the identical gate call accepts
the honest implementation.

### Fixture integrity via manifest.sha256
**Source:** `crates/aprender-core/tests/fixtures/setfit/manifest.sha256`,
`generate_fixtures.py:913–928`, verifier `setfit_conformance.rs:1199`
**Apply to:** `tests/goldens/` and `tests/setfit_reference/` in the new crate. Re-baselining is
a reviewable manifest diff, never a silent overwrite.

### Code discovery during implementation
Per CLAUDE.md: `pmat query "<intent>"`, never grep/glob, when plans need to locate additional
code. All analogs in this document were pre-resolved from the CONTEXT/RESEARCH citations.

## No Analog Found

Files where the planner should build from RESEARCH.md's Code Examples / Patterns instead of a
codebase analog:

| File | Role | Data Flow | Reason / Governing Source |
|------|------|-----------|---------------------------|
| `src/pairs.rs` | service | streaming | No streaming pair sampler exists anywhere in the workspace (`aprender-data`'s `Balance` oversampling is row-resampling, not pair construction). Build from RESEARCH Pattern 4 + the `CanonicalPair` and closed-form capacity Code Examples; semantics pinned by measured setfit 1.1.3 behavior (Finding F2). |
| `src/dedup.rs` | service | batch | No cross-split duplicate detector exists (`apr data decontaminate` is n-gram overlap with thresholds — deliberately NOT the D-17 mechanism). Build from D-17/D-18 + Finding F1's real duplicate (train:70 ≡ validation:3) as the live golden. |
| `src/ledger.rs` | model | event-driven | No access-ledger precedent in-tree. Small append-only `Vec<AccessRecord>` with sorted serialization; consumed by Phase 5. |
| `src/rng.rs` key-derivation half | utility | — | `generate_at` primitive is exact-analog, but domain-separated key derivation + multiply-shift bounded draw exist nowhere; RESEARCH Pattern 2 is authoritative (and Assumption A3 applies). |
| `tests/ui/` (trybuild) | test | — | `trybuild` is a dev-dep of `aprender-contracts-macros` and `aprender-test-derive` (Cargo.toml:23 in each) but NO `trybuild::TestCases` harness file exists in the repo (grep-verified this session). Use the standard upstream harness (`let t = trybuild::TestCases::new(); t.compile_fail("tests/ui/*.rs");`) with committed `.stderr` snapshots pinning the compat-profile "cannot be constructed" errors. |

## Metadata

**Analog search scope:** `crates/apr-cli/src/` (commands, data_commands, dispatch),
`crates/aprender-rand/`, `crates/aprender-data/src/split.rs`,
`crates/aprender-core/src/format/` (typestate), `crates/aprender-core/src/setfit/` +
`tests/setfit_conformance/` (Phase 1 gates), `crates/aprender-core/tests/fixtures/setfit/`,
`contracts/`, `Makefile`, `scripts/setfit_fixtures/`, root and member `Cargo.toml`s,
`crates/aprender-contracts-macros/tests/`.
**Files scanned:** 21 read in full or in targeted excerpts; ~10 more located/verified by grep.
**Pattern extraction date:** 2026-08-08
**Caveat:** line numbers in `data_tweeteval.rs`, `data_commands.rs`, and the tweet-eval
contract refer to the UNCOMMITTED D-06 baseline currently in the working tree; they are stable
only after the D-06 PR lands as-is.
