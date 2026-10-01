# Phase 3: Faithful Two-Stage Trainer and Head - Pattern Map

**Mapped:** 2026-08-09
**Revised:** 2026-08-09 (revision 2 — re-synced to the 10-plan / 7-wave structure produced by the
cross-AI review replan; the original map was written against the 9-plan structure)
**Files analyzed:** 27 new/modified files (19 original + 8 added by the replan)
**Analogs found:** 24 / 27 (3 partial, 0 no-analog after revision 2)

All analog excerpts below were read directly from HEAD this session. Line numbers are pinned
against the working tree at branch `gsd/phase-2-contract-gate`.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/aprender-train/src/train/setfit/mod.rs` (SetFitRun typestate) | typestate lifecycle | transform (staged) | `crates/aprender-contrastive-data/src/split.rs` | exact (house style) |
| `crates/aprender-train/src/train/setfit/config.rs` | config (validated) | request-response (fail-closed) | `crates/aprender-train/src/train/device.rs` + `contrastive-data/src/pairs.rs` | exact |
| `crates/aprender-train/src/train/setfit/evidence.rs` | record/artifact | transform + hash-commit | `crates/aprender-contrastive-data/src/ledger.rs` | role-match |
| `crates/aprender-train/src/train/setfit/reduce.rs` | numeric utility | batch reduction | none (pattern supplied by RESEARCH; anti-analog: `par_iter().sum()`) | no analog |
| `crates/aprender-train/src/train/setfit/epoch.rs` | RNG derivation utility | deterministic sampling | `crates/aprender-contrastive-data/src/rng.rs` | exact |
| `crates/aprender-train/src/train/setfit/head_input.rs` | data adapter | batch encode (exactly-once) | `contrastive-data/src/select.rs` (`Selection`, `SelectedId`) + `aprender-core/src/setfit/mod.rs` | exact (consumes both) |
| `crates/aprender-train/src/train/setfit/lock.rs` | record + typestate token | hash-commit | `contrastive-data/src/ledger.rs` + `select.rs` (`SelectedId` opaque token) | role-match |
| `crates/aprender-train/src/train/setfit/verify.rs` | trait seam + serde impl | serialize/reload round-trip | `contrastive-data/src/ledger.rs` (`to_canonical_bytes`/`from_bytes`) | partial |
| `crates/aprender-train/src/train/setfit/baseline.rs` (FrozenProbeRun) | distinct marker type | transform | `contrastive-data/src/split.rs` (`CompatibilityTest` vs `Test`) | exact (pattern) |
| `crates/aprender-train/src/train/setfit/negative.rs` + its gate test | test-only in-band negative | adversarial fixture | `contrastive-data/tests/negative_leaky.rs` | exact |
| `crates/aprender-train/src/optim/scheduler/warmup_linear_decay.rs` | scheduler | step function | `aprender-train/src/optim/scheduler/warmup_cosine_decay.rs` | exact |
| `crates/aprender-train/tests/ui.rs` + `tests/ui/*.rs` | trybuild compile-fail test | non-constructibility proof | `contrastive-data/tests/ui.rs` + `tests/ui/split_constructed_directly.rs` | exact |
| `crates/aprender-train/Cargo.toml` (modified: `setfit` feature + dep edge) | build config | — | `crates/aprender-core/Cargo.toml:212-269` (`setfit` feature block) | exact |
| `crates/aprender-core/src/classification/multinomial.rs` | model (classifier head) | batch fit/predict | `classification/mod.rs` (binary LR, sibling) + `glm/glm_tests.rs` (reference falsification) | role-match |
| `crates/aprender-core/src/optim/lbfgs.rs` (modified: f64 widening) | solver | iterative optimization | itself + `tests_lbfgs_contract.rs` + `primitives/vector.rs:81` | exact (self) |
| `crates/aprender-core/src/setfit/` keyed dropout mask source | RNG utility | deterministic masking | `contrastive-data/src/rng.rs` (copy); anti-analog `nn/dropout/mod.rs:40-49` | exact (copy rng.rs) |
| `crates/aprender-core/Cargo.toml` (modified: `aprender-rand` dep under `setfit`) | build config | — | its own `setfit = ["dep:tokenizers", "dep:sha2"]` at line 265 | exact |
| `contracts/` new Phase 3 contract YAML(s) | contract | — | `contracts/contrastive-pair-protocol-v1.yaml` + `contracts/aprender/binding.yaml:879-916` | exact |
| `Makefile` (modified: `PHASE3_CONTRACTS` + scoped audit + tier3 repro gate) | build gate wiring | — | `Makefile:1086-1094` (`PHASE2_CONTRACTS`) + `:1156+` (`contract-audit-phase2`) | exact |

## Pattern Assignments

### `crates/aprender-train/src/train/setfit/mod.rs` — SetFitRun typestate (TRN-01, D-06/D-11)

**Analog:** `crates/aprender-contrastive-data/src/split.rs` (Phase 2's house-style typestate — CONTEXT names it the direct precedent)

**Marker types + role trait** (split.rs lines 36-70):
```rust
pub trait SplitRole {
    const ROLE: &'static str;
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Train;
// ...
impl SplitRole for Train { const ROLE: &'static str = "train"; }
```
Phase 3 form: `pub trait LifecycleState` with markers `Prepared`, `EncoderTuned`, `HeadFitted`,
`ArtifactReloadedAndVerified`. Per RESEARCH Pattern 1, use an **associated `Evidence` type** on the
state trait (absent field beats `Option<Evidence>` — Ph2 02-03 precedent; `Option` is an explicit
anti-pattern because it defeats the trybuild proof).

**Private fields + PhantomData + no public constructor** (split.rs lines 85-92):
```rust
pub struct Split<R: SplitRole> {
    rows: Vec<LabeledExample>,
    source_hash: [u8; 32],
    // ... all fields private ...
    role: PhantomData<R>,
}
```

**Gated transition — validation INSIDE the only door, `pub(crate)`/consuming** (split.rs lines 111-118):
```rust
pub(crate) fn from_jsonl_bytes(
    bytes: &[u8],
    decl: &SplitDeclaration,
) -> Result<Self, ContrastiveDataError> {
    let source_hash: [u8; 32] = Sha256::digest(bytes).into();
    let rows = parse_jsonl_bytes(bytes, R::ROLE)?;
    validate_ingest_ladder(&rows, R::ROLE, decl)?;   // gate BEFORE the state exists
    Ok(Self::assemble(rows, decl, source_hash))
}
```
Phase 3 form: `tune_encoder(self) -> Result<SetFitRun<EncoderTuned>, SetFitTrainError>` runs the
evidence gate before minting the next state; transition consumes `self`.

**Contract binding on the single gate function** (split.rs lines 240-244):
```rust
#[provable_contracts_macros::contract(
    "contrastive-pair-protocol-v1",
    equation = "split_ingest_boundary"
)]
fn validate_ingest_ladder(...) -> Result<(), ContrastiveDataError> {
```
Note from `contracts/aprender/binding.yaml:908-916`: the bound function may be module-private —
the registry records `module_path`, `function`, `signature`, `status: implemented`.

**Error-shape convention** (split.rs Gate 2, lines ~252-260): every typed error names both the
expected and the observed value (`SplitRoleMismatch { expected_role, embedded_role }`), and tests
assert on the rendered message containing both names.

---

### `crates/aprender-train/src/train/setfit/config.rs` — SetFitTrainConfig (TRN-02)

**Analogs:** `crates/aprender-train/src/train/device.rs` (fail-closed knob validation, reused in
place for the device knob) + `crates/aprender-contrastive-data/src/pairs.rs` (two-stage config +
cross-field resolution — RESEARCH Open Question 2's named precedent)

**Typed error with contract-citing Display** (device.rs lines 64-98):
```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceError {
    InvalidSpec(String),
    CudaNotAvailable { requested: String },
}
impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceError::InvalidSpec(s) => write!(f,
                "--device `{s}` does not match grammar \
                 ^(cpu|cuda(:[0-9]|:1[0-5])?|auto)$ \
                 (contract gpu-training-backend-v1 INV-GPUTRAIN-001)",),
            // ... error text NAMES the contract and the fix ...
        }
    }
}
```

**Pure parser separated from the environment probe** (device.rs lines 110-143): `resolve_device`
calls `parse_device_spec` (pure, grammar-testable without CUDA) then probes availability. Copy this
split for every knob: pure per-knob validation functions, then cross-field checks, each with a
FALSIFY case table in tests (device.rs tests are literally headed `FALSIFY-GPUTRAIN-001: grammar`).

**Cross-field resolution ladder with each rung naming what to change** (pairs.rs lines 400-441):
```rust
pub fn resolve_budget(cfg: &PairConfig, class_sizes: &[u64])
    -> Result<(u64, bool), ContrastiveDataError> {
    let hard_cap = cfg.resolved_hard_cap();
    if hard_cap == 0 { return Err(ContrastiveDataError::ZeroHardCap); }
    if let Some(budget) = cfg.budget {
        if budget == 0 { return Err(ContrastiveDataError::ZeroBudget); }
        if budget > hard_cap {
            return Err(ContrastiveDataError::BudgetExceedsHardCap { budget, hard_cap });
        }
    }
    // ... capacity checks, then resolution; NEVER silent clamping ...
}
```
The doc comment on `resolve_budget` ("The cap BINDS an explicit budget — it does not clamp it")
is the fail-closed philosophy TRN-02 requires: an invalid knob is a typed error naming the knob,
never a silent adjustment. RESEARCH recommends one `SetFitTrainConfig` validated at construction
(not a fallible builder chain) — cross-field rules (warmup fraction vs total steps, pair budget vs
selection capacity) live in the constructor exactly as `PairConfig::new` + `resolve_budget` split.

---

### `crates/aprender-train/src/train/setfit/evidence.rs` — evidence table + summary/hash (D-09..D-12)

**Analog:** `crates/aprender-contrastive-data/src/ledger.rs` (canonical bytes → SHA-256, schema
version, no timestamps)

**Canonical serialization — deterministic by construction** (ledger.rs lines 82-103):
```rust
/// Compact JSON over a struct with a fixed field order and a `Vec` whose order IS the
/// access order. There is no map to iterate and no timestamp to drift, so two runs
/// that touched the same splits in the same order produce byte-identical output.
#[provable_contracts_macros::contract(
    "contrastive-pair-protocol-v1",
    equation = "access_ledger_persistence"
)]
pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, ContrastiveDataError> {
    let wire = LedgerWire { schema_version: LEDGER_SCHEMA_VERSION, records: self.records.clone() };
    serde_json::to_vec(&wire).map_err(|error| ContrastiveDataError::Serialization { .. })
}
```

**Hash over canonical bytes, total when the form permits it** (ledger.rs lines 130-140):
```rust
pub fn ledger_hash(&self) -> [u8; 32] {
    let bytes = self.to_canonical_bytes()
        .expect("AccessLedger canonical form is strings and integers; serialization is total");
    Sha256::digest(bytes).into()
}
```

**Per-parameter source surface** (`crates/aprender-core/src/setfit/mod.rs` lines 440-457):
```rust
pub fn trainable_parameters_mut(&mut self) -> Vec<(String, &mut Tensor)> {
    let frozen = self.frozen_names();
    self.encoder.named_parameters_mut().into_iter()
        .filter(|(n, _)| !frozen.contains(n)).collect()
}
pub fn frozen_parameters(&self) -> Vec<(String, &Tensor)> { /* freeze-policy complement */ }
```
HF dotted names — exactly D-09's granularity, no translation. Use `BTreeMap<String, f64>` for
init/delta norms (fixed iteration order for hashing — Ph2 02-03 precedent, see split.rs's
`exact_hashes: BTreeMap` whose iteration order discharges the sorting obligation structurally).

**Pre-clip norm feed** (`crates/aprender-train/src/optim/clip.rs:65`):
```rust
pub fn clip_grad_norm_refs(params: &mut [&mut Tensor], max_norm: f32) -> f32 {
    // returns the global norm BEFORE clipping — feed straight into the evidence record
}
```

**Hash-poison rule:** never serialize `OptimizationResult` whole — it carries
`elapsed_time: Duration` (`optim/mod.rs:122` region; concrete construction sites at lbfgs.rs
lines 240, 259, 282, 325). Define `HeadFitReport`/evidence structs carrying only deterministic
fields, hash those. Add `#[serde(deny_unknown_fields)]` on wire structs (Ph2 precedent:
`contrastive-data/src/attestation.rs:73,87`).

---

### `crates/aprender-train/src/train/setfit/reduce.rs` — fixed-order reductions (D-13)

**Analog: NONE in-repo** — nothing currently guarantees fixed-order reductions (RESEARCH verified;
existing parallel reductions live in `aprender-compute` and `par_iter().sum()` is the named
anti-pattern). Use the RESEARCH-supplied pattern directly:

```rust
/// Order-fixed sum: identical bit pattern at any thread count, because there is
/// exactly one reduction tree. Sizes on this path are ≤ a few thousand elements.
pub fn sum_in_index_order(xs: &[f32]) -> f64 {
    xs.iter().fold(0.0_f64, |acc, &x| acc + f64::from(x)) // accumulate in f64, one order
}
```

Nearest stylistic kin for sequential index-order numeric loops: `LBFGS::norm` and the two-loop
recursion in `crates/aprender-core/src/optim/lbfgs.rs` (lines 111-200) — plain indexed loops, no
iterator parallelism. Do NOT build a general capability in `aprender-compute` (RESEARCH Open
Question 1: trainer-local module + empirical thread-count falsification gate; touch
`aprender-compute`'s `blis/parallel.rs:137` partitioner only if that gate goes red).

---

### `crates/aprender-train/src/train/setfit/epoch.rs` — cross-epoch pair order (D-15 kin, Ph2 D-14 hand-off)

**Analog:** `crates/aprender-contrastive-data/src/rng.rs` — copy the construction wholesale, new
domain tag.

**Frozen domain tag with NUL terminator** (rng.rs line 52):
```rust
const DOMAIN_TAG: &[u8] = b"apr-contrastive-v1\0";
// Phase 3 MUST mint its own, e.g. b"apr-setfit-train-v1\0" — reusing the Phase 2 tag is a
// silent cross-phase seed-reuse bug (RESEARCH anti-pattern list).
```

**Opaque key + stateless derivation** (rng.rs lines 59, 86-101):
```rust
pub struct DomainKey([u32; 2]);   // only door is derive_key — no call site can skip separation

#[provable_contracts_macros::contract("contrastive-pair-protocol-v1", equation = "rng_key_derivation")]
pub fn derive_key(root_seed: u64, domain: &str) -> DomainKey {
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN_TAG);
    hasher.update(root_seed.to_le_bytes());
    hasher.update(domain.as_bytes());
    let digest: [u8; 32] = hasher.finalize().into();
    let lane0 = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]);
    let lane1 = u32::from_le_bytes([digest[4], digest[5], digest[6], digest[7]]);
    DomainKey([lane0, lane1])
}
```

**Pure indexed draw + multiply-shift bound** (rng.rs lines 106-165):
```rust
pub fn draw(key: &DomainKey, stream_id: u32, ordinal: u64) -> [u32; 4] {
    let counter = [ordinal as u32, (ordinal >> 32) as u32, stream_id, 0];
    Philox4x32::generate_at(key.0, counter)
}
pub fn bounded(key: &DomainKey, stream_id: u32, ordinal: u64, n: NonZeroU64) -> u64 {
    let x = assemble64(draw(key, stream_id, ordinal));
    ((u128::from(x) * u128::from(n.get())) >> 64) as u64
}
```
`NonZeroU64` makes a zero bound a type error. Modulo and float scaling are FORBIDDEN (documented
in rng.rs with reasons — carry the doc discipline). Closed domain-string table lives in a
`pub mod domains` (rng.rs line 177) — Phase 3 adds e.g. `"epoch-shuffle"` (keyed by epoch) and
`"dropout"` under the NEW tag, as its own closed table.

**Golden-pinning discipline** (rng.rs tests, `rng_byte_encoding_golden_is_frozen`): constants
derived by an INDEPENDENT implementation from the contract text, never captured from a first run
of the code under test. Phase 3's epoch-shuffle and dropout streams need the same treatment.

---

### `crates/aprender-train/src/train/setfit/head_input.rs` — encode-once head input (TRN-05, D-08)

**Analogs:** `crates/aprender-contrastive-data/src/select.rs` (the ONLY input type the signature
accepts) + `crates/aprender-core/src/setfit/mod.rs` (eval-mode encode).

**The typed unique-ID surface** (select.rs lines 103-135, 306-336):
```rust
pub struct SelectedId(u32);        // constructor private to select.rs — proof of membership
pub struct Selection {
    ordered: Vec<SelectedExample>,
    by_class: BTreeMap<usize, Vec<SelectedId>>,
    semantic_hash: [u8; 32],
    ledger_hash: [u8; 32],
    // ...
}
impl Selection {
    pub fn examples(&self) -> &[SelectedExample] { &self.ordered }
    pub fn ordered_ids(&self) -> Vec<&str> { ... }
    pub fn semantic_hash(&self) -> [u8; 32] { self.semantic_hash }
}
```
Structural enforcement: `fit_head(&Selection)` — the signature has no pair-stream parameter, so
pair multiplicity is inexpressible. Iterate `examples()` (already deduplicated, deterministic
order), batch in PINNED consecutive windows.

**Eval/no-grad mode** (`setfit/mod.rs:293` `set_training(bool)`, `:287` `encode_texts`): call
`set_training(false)` before encoding; per-site training flags already exist (Ph1
`encoder_mode_dropout_*` tests pin them).

---

### `crates/aprender-train/src/train/setfit/lock.rs` — SelectionLock + token (TRN-07, D-14)

**Analogs:** `ledger.rs` (canonical-bytes+hash record — see evidence.rs excerpts above) +
`select.rs` `SelectedId` (opaque token whose constructor is private, lines 103-118):
```rust
pub struct SelectedId(u32);
impl SelectedId {
    /// This is an accessor, not a constructor: reading the number cannot mint a
    /// `SelectedId`, so the type remains proof of membership in the selection that
    /// produced it.
    pub fn ordinal(self) -> u32 { self.0 }
}
```
Phase 3 form: the canonical-test access token is mintable ONLY by a method on the lock that
compares the lock's committed artifact hash to the model about to be evaluated — private field,
no public constructor, accessors cannot mint. Append-only discipline from ledger.rs line 65:
"There is no removal API: an append-only log that can be rewritten is not evidence."
Hash rendering via `contrastive-data/src/hash.rs:73` `hex()`; content hashing via `exact_hash`
(hash.rs:43, takes `&str` — canonical-JSON-then-hash works) or a new workspace `sha2` dep in
aprender-train (`sha2 = "0.10"` is already in aprender-train's `[dependencies]` — verified in
its Cargo.toml — so no new dep line is needed; use it directly and keep one convention).

---

### `crates/aprender-train/src/train/setfit/verify.rs` — sealed `SetFitCodec` + trusted verify policy (D-07 as amended)

**Analog (partial):** `ledger.rs` round-trip pair `to_canonical_bytes` / `from_bytes`
(lines 94-127) — the serialize-with-schema-version, parse-with-version-check shape:
```rust
pub fn from_bytes(bytes: &[u8]) -> Result<Self, ContrastiveDataError> {
    let wire: LedgerWire = serde_json::from_slice(bytes).map_err(...)?;
    if wire.schema_version != LEDGER_SCHEMA_VERSION {
        return Err(ContrastiveDataError::UnsupportedSchemaVersion { field, got, supported });
    }
    Ok(Self { records: wire.records })
}
```
No in-repo trait seam analog exists for "close, reload from bytes, re-encode, re-predict, compare
within contracted tolerances". **Revision 2:** the seam is no longer a single `ReloadVerify` trait.
D-07 was amended so the implementable half is a SEALED, three-method pure codec (`SetFitCodec`:
`format_id` / `serialize` / `deserialize`) carrying no hashing, no comparison and no tolerance, while
artifact hashing, the drop-reload-rebuild-compare sequence, the round-trip closure check and the
minting of `ArtifactReloadedAndVerified` stay in trusted crate-internal policy. The sealing analog IS
in-repo: `SelectedId`'s private constructor (`select.rs:103-118`) — a type whose provenance is a
property of who may construct it. Tolerances still come from the contract per the Ph1 D-14
frozen-tolerance discipline in `contracts/setfit-encoder-conformance-v1.yaml`.

---

### `crates/aprender-train/src/train/setfit/baseline.rs` — FrozenProbeRun (D-11, SAFE-03)

**Analog:** split.rs's `CompatibilityTest` marker (lines 55-57, 69-70) — a role DISTINCT from
`Test` "so a compatibility corpus can never be mistaken for a canonical one by name alone":
```rust
/// The merged compatibility test split (D-19) — a role DISTINCT from [`Test`], so a
/// compatibility corpus can never be mistaken for a canonical one by name alone.
pub struct CompatibilityTest;
impl SplitRole for CompatibilityTest { const ROLE: &'static str = "compatibility_test"; }
```
Phase 3 form: `FrozenProbeRun` is a differently-NAMED type, not a flag on `SetFitRun` — a frozen
baseline structurally cannot claim SetFit because `EncoderTuned` cannot exist without passing
evidence (empty trainable set is unpassable). Binds `contracts/linear-probe-classifier-v1.yaml`
(frozen-encoder invariants already written there — verified present by RESEARCH).

---

### `crates/aprender-train/src/train/setfit/negative.rs` — in-band negative (D-08, Ph1 D-24 / Ph2 D-25)

**Analog:** `crates/aprender-contrastive-data/tests/negative_leaky.rs` — the three-element
discipline, verbatim from its module doc (lines 1-35):
```text
1. **The negative** — the poisoned list is rejected, and the message NAMES the offending
   identifier, because that is what makes a real failure diagnosable rather than red.
2. **The control** — the same list with the poisoned record REMOVED validates `Ok`.
   Without it, "the poisoned list is rejected" is equally satisfied by a gate that
   rejects everything, or by a dump that was malformed for some unrelated reason.
3. **The mirror** — the identical call over the untouched dump returns `Ok` and yields
   exactly the sampler's own pairs.
```
Also copy: the doc explains WHY the poison must be built from the untrusted surface ("A trusted
`LabeledPair` CANNOT REPRESENT this attack") — Phase 3's pair-weighted fitter must likewise attack
from whatever surface the typestate leaves expressible, and if it can be built from trusted types
"the typestate would be broken and THAT would be the finding." The fitter lives in-crate
(`cfg(test)` or `#[doc(hidden)]` test-support) so it runs and FAILS its gate in every `cargo test`.

---

### `crates/aprender-train/src/optim/scheduler/warmup_linear_decay.rs` — WarmupLinearDecayLR (Pitfall 6)

**Analog:** `crates/aprender-train/src/optim/scheduler/warmup_cosine_decay.rs` (exact structural
mirror — swap the cosine branch for linear decay to 0).

**Struct + constructor + apply** (warmup_cosine_decay.rs lines 12-36):
```rust
pub struct WarmupCosineDecayLR {
    lr_max: f32, lr_min: f32,
    warmup_steps: usize, total_steps: usize, current_step: usize,
}
impl WarmupCosineDecayLR {
    pub fn new(lr_max: f32, lr_min: f32, warmup_steps: usize, total_steps: usize) -> Self { ... }
    pub fn apply<O: Optimizer>(&self, optimizer: &mut O) { optimizer.set_lr(self.get_lr()); }
}
```

**Trait impl with zero-division guards** (lines 38-68):
```rust
impl LRScheduler for WarmupCosineDecayLR {
    fn get_lr(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            if self.warmup_steps == 0 { return self.lr_max; }
            let progress = self.current_step as f32 / self.warmup_steps as f32;
            return self.lr_max * progress;
        }
        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps);
        if decay_steps == 0 { return self.lr_min; }
        // ... decay branch — replace cosine with linear-to-zero here ...
    }
    fn step(&mut self) { self.current_step += 1; }
}
```
Register in `scheduler/mod.rs` (module list + `pub use` + doc bullet, mod.rs lines 1-21). The
`LRScheduler` trait (`get_lr` / `step`) is at mod.rs lines 23-30. Do NOT copy `LinearWarmupLR` —
it holds LR CONSTANT after warmup, which is exactly the reference-recipe deviation Pitfall 6 flags.

---

### `crates/aprender-train/tests/ui.rs` + `tests/ui/*.rs` — trybuild (TRN-01)

**Analog:** `crates/aprender-contrastive-data/tests/ui.rs` (harness) +
`tests/ui/split_constructed_directly.rs` (+ `.stderr`).

**Harness** (ui.rs lines 40-44):
```rust
#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
```

**Case-file discipline** (split_constructed_directly.rs): each case is a complete program using
ONLY the public API, headed by the obligation ID and the expected diagnostic:
```rust
// OBLIG-CPP-LEAKAGE-NOT-CONSTRUCTIBLE / DATA-06 / D-16.
// Expected diagnostic: `associated function 'from_jsonl_bytes' is private`.
use aprender_contrastive_data::split::{Split, SplitDeclaration, Train};
fn mint_a_split(bytes: &[u8], decl: &SplitDeclaration) {
    let _ = Split::<Train>::from_jsonl_bytes(bytes, decl);
}
fn main() {}
```
**Snapshot rules from the harness doc** (ui.rs lines 20-38): each `.stderr` must name the crate's
REAL types/methods/visibility (a syntax-error snapshot proves nothing); re-baseline command is
`TRYBUILD=overwrite cargo test -p <crate> --test ui` and the diff must be reviewed against the
named-types list. `trybuild = "1"` is already a workspace dep. Phase 3 cases: constructing
`SetFitRun<EncoderTuned>` directly; `fit_head` before `tune_encoder`; minting a canonical-test
token without a lock; passing the pair stream to `fit_head`. Wire as
`cargo test -p aprender-train --test ui --features setfit`.

---

### `crates/aprender-core/src/classification/multinomial.rs` — MultinomialLogisticRegression (TRN-04, D-01..D-04)

**Analogs:** `crates/aprender-core/src/classification/mod.rs` (the sibling it sits BESIDE and must
not touch) + `crates/aprender-core/src/glm/glm_tests.rs` (reference-falsification test) +
`crates/aprender-core/src/optim/lbfgs.rs` (solver surface).

**Sibling struct shape — private fitted state, builder setters** (classification/mod.rs lines 114-135, 143-160):
```rust
pub struct LogisticRegression {
    coefficients: Option<Vector<f32>>,
    intercept: f32,
    learning_rate: f32, max_iter: usize, tol: f32,
    // ...
}
#[must_use]
pub fn with_max_iter(mut self, max_iter: usize) -> Self { self.max_iter = max_iter; self }
```
Site the new type alongside at `classification/multinomial.rs`, registered from
`classification/mod.rs`. Depart deliberately where D-01 says to: typed error enum
(`HeadFitError` — RESEARCH supplies the full variant mapping from `ConvergenceStatus`), NOT
`Result<(), String>`; labels as ordered `Vec<String>` (index = W row, argmax lowest-index
tie-break); f64 fit / f32 store. Do NOT add `ClassWeight` (RESEARCH Open Question 8: reference head
is `LogisticRegression()` with defaults — no weighting knob in v1).

**Solver call shape the head programs against** (lbfgs.rs lines 94-104, 213):
```rust
let mut optimizer = LBFGS::new(100, 1e-5, 10);   // max_iter, tol (grad-norm), history m
let result: OptimizationResult = optimizer.minimize(objective, gradient, x0);
// result.status: Converged | MaxIterations | Stalled | NumericalError | Running | UserTerminated
```

**Reference-falsification test — the D-04 precedent** (glm_tests.rs lines 274-297):
```rust
/// FALSIFY-GLM-IRLS-LINK-DERIV (PMAT-838): IRLS must use the LINK derivative ...
/// With the two swapped, Binomial/logit on this data converged to slope 1.0333 (8.3% low) and
/// P(y=1 | x=-2) = 0.1124; the correct IRLS (matching a statsmodels/scipy reference) gives
/// slope 1.1266 and P = 0.0951.
#[test]
fn falsify_glm_irls_link_derivative() {
    // fixed small design matrix; fit; assert against the CORRECT reference value with the
    // RED (bug-present) value documented in the comment:
    assert!((slope - 1.1266).abs() < 0.01,
        "GLM IRLS slope {slope:.4} != correct 1.1266 (link-derivative swap?)");
}
```
Copy this shape for the contracted sklearn relation: document the RED value the factor-of-2 error
would produce, use n small enough that λ=1/(2Cn) vs 1/(Cn) separates beyond tolerance (RESEARCH:
n=24, C=1 → λ=1/48 vs 1/24). The contract equation must spell out BOTH objective conventions fully
expanded (the ½ inside sklearn's r(W) — Pitfall 1). Softmax-NLL must use the log-sum-exp shift
(the one algorithm Phase 3 hand-writes; "Don't Hand-Roll" table).

---

### `crates/aprender-core/src/optim/lbfgs.rs` — f64 widening (D-03, first-wave plan)

**Analog:** the file itself + its contract surface. Current f32 hardwiring to change
(lines 62-77, 213-216):
```rust
pub struct LBFGS {
    pub(crate) tol: f32,
    pub(crate) s_history: Vec<Vector<f32>>,
    pub(crate) y_history: Vec<Vector<f32>>,
    // ...
}
fn minimize<F, G>(&mut self, objective: F, gradient: G, x0: Vector<f32>) -> OptimizationResult
where F: Fn(&Vector<f32>) -> f32, G: Fn(&Vector<f32>) -> Vector<f32>,
```
RESEARCH recommendation (Open Question 4): genericize the implementation over the float
(`LbfgsImpl<T>`), keep `LBFGS` as the f32 alias — zero public API break, additive minor bump.
Constraint: `Vector<T>` is generic but arithmetic impls are f32-only (`primitives/vector.rs:81`
`impl Vector<f32>`) — the generic impl needs the handful of Vector ops it uses (zeros, len,
indexing) available for f64.

**Contract-test surface that must keep passing AND gain f64 twins**
(`optim/tests_lbfgs_contract.rs`, 133 lines): FALSIFY-LBFGS-001/002/003 (quadratic convergence,
objective decrease, finite result) + two proptest variants, all `Vector<f32>`-typed today. The
naming convention is `falsify_lbfgs_NNN_<property>` with `"FALSIFIED LBFGS-NNN: ..."` messages.
`pv diff` on `contracts/lbfgs-kernel-v1.yaml`: materialize old with `git show`, diff two real paths
(CLAUDE.md rule). Existing binding: `#[provable_contracts_macros::contract("lbfgs-kernel-v1",
equation = "two_loop_recursion")]` sits on `step` (lbfgs.rs line 205 region).

**Hash poison to design around:** every `OptimizationResult` constructed here carries
`elapsed_time: start_time.elapsed()` (lines 240, 259, 282, 325).

---

### Keyed dropout mask source in `aprender-core` (D-15, Pitfall 2)

**Analog to COPY:** `contrastive-data/src/rng.rs` (all excerpts under epoch.rs above — same
construction, new tag, `"dropout"` domain keyed by site/step/element).

**Anti-analog — the thing being replaced** (`crates/aprender-core/src/nn/dropout/mod.rs` lines 40-49):
```rust
pub struct Dropout {
    p: f32,
    training: bool,
    rng: Mutex<StdRng>,   // stateful: draw i depends on every prior draw; StdRng is
}                          // NOT portable across rand versions — both violate D-15
```

**Existing site-keying to reconcile** (`crates/aprender-core/src/setfit/encoder.rs` lines 103-124):
```rust
/// Derive a site's RNG seed from the root seed and its DOTTED NAME.
/// Non-positional on purpose: inserting a layer must not renumber the streams ...
fn site_seed(root_seed: u64, site: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;   // FNV-1a over the name
    for b in site.as_bytes() { h ^= u64::from(*b); h = h.wrapping_mul(0x0000_0100_0000_01b3); }
    crate::nn::transformer::mix_call_seed(h, root_seed)   // SplitMix64 finaliser
}
```
This FNV/SplitMix scheme is NOT the contracted SHA-256 derivation. RESEARCH leaves the choice
(migrate site keying to the rng.rs construction, or contract the existing one) to the plan — pick
one, do not leave two half-documented schemes. Scope the new mask source to the SetFit encoder
sites (reworking `nn::Dropout` globally touches every consumer); keep the `training()` flag
semantics so Ph1's `encoder_mode_dropout_*` tests survive. Site names are the four dotted HF names
built by `attention_output_site`/`ffn_output_site` etc. (encoder.rs lines 90-101).

New dependency edge: `aprender-core -> aprender-rand` (lib name `trueno_rand`, leaf crate with
`thiserror` only — verified acyclic), added under the `setfit` feature in the `dep:` form.

---

### `crates/aprender-train/Cargo.toml` + `crates/aprender-core/Cargo.toml` — feature wiring (Pitfall 11)

**Analog:** aprender-core's own `setfit` feature block (`crates/aprender-core/Cargo.toml`
lines 212-269):
```toml
[features]
# ... comment explains the dep:-form / implicit-feature interaction for sha2 ...
sha2 = ["dep:sha2"]
# The feature is declared dependency-CLOSED: enabling `setfit` alone must build.
setfit = ["dep:tokenizers", "dep:sha2"]
conformance-fixtures = ["setfit"]   # implying feature must IMPLY setfit
```
aprender-train form (RESEARCH Pitfall 11): `setfit = ["aprender/setfit", "dep:aprender-contrastive-data"]`,
modules `#[cfg(feature = "setfit")]`. aprender-train's current features (its Cargo.toml
`[features]`) have NO setfit entry and `default = ["tui"]` — the `--no-default-features
--features setfit` leg exercises a different module set than developers usually build. Note
aprender-train already has `sha2 = "0.10"`, `serde`, `serde_json` unconditionally. Test-module
gating precedent: `#[cfg(all(test, feature = "setfit"))]` (setfit/mod.rs lines 460-462).
Known-red consequence: `cargo package -p aprender-train` goes red when the path dep on unpublished
`aprender-contrastive-data` lands (Pitfall 9) — record in must_haves.caveats, extend the publish
cascade order.

---

### `contracts/` Phase 3 contract YAML(s) + binding entries

**Analog:** `contracts/contrastive-pair-protocol-v1.yaml` header (lines 1-36):
```yaml
contract: contrastive-pair-protocol
metadata:
  version: 1.0.0
  created: '2026-08-08'
  author: PAIML Engineering
  description: >
    ... names the requirements covered, the implementing crate, the "first consumer,
    not owner" boundary, and WHY THE CONTRACT EXISTS (which claims are fakeable by a
    passing suite and therefore need in-band negatives) ...
```
Also copy its inline-statement discipline: no dangling cross-references — derivations frozen
inline, byte for byte. Binding registry entry shape (`contracts/aprender/binding.yaml:880-887`):
```yaml
- contract: contrastive-pair-protocol-v1.yaml     # BARE filename (Phase 2 trap)
  equation: rng_key_derivation
  module_path: aprender_contrastive_data::rng
  function: derive_key
  signature: 'fn derive_key(root_seed: u64, domain: &str) -> DomainKey'
  status: implemented                              # only implemented|partial|not_implemented|pending
  notes: ...
```
Validate with `pv` only (`PV_BIN := cargo run --release -p aprender-contracts-cli --bin pv --`,
Makefile:1035). Check overlap with `contracts/classifier-pipeline-v1.yaml` and
`contracts/classification-finetune-v1.yaml` before authoring (CONTEXT canonical refs). Frozen
numbers that belong IN the contract before any run: D-10's ε, TRN-03's endpoint-comparison k and
margin, D-04's fully-expanded two-sided regularization equation.

---

### `Makefile` — PHASE3_CONTRACTS + scoped audit + tier wiring (D-16, Pitfall 10)

**Analog:** the Phase 2 block, copied with the number bumped.

**Scoped contract list** (Makefile lines 1082-1094):
```make
# The two Phase 2 contracts, audited as a BLOCKING tier3 gate by
# `contract-audit-phase2` below. Deliberately a separate, narrower list than
# $(CONTRACTS) — see that target's comment block for the measurement that
# forced the narrowing.
PHASE2_CONTRACTS := contracts/contrastive-pair-protocol-v1.yaml \
                    contracts/tweet-eval-stance-benchmark-v1.yaml

# NOTE (plan 02-01, D-24): $(CONTRACTS) is an EXPLICIT HARDCODED LIST, not a glob
# ... Every future phase contract needs its own line here or it is decoration.
```
New contract files need BOTH a `$(CONTRACTS)` line (reached by `contract-validate`, Makefile:1097)
and a `PHASE3_CONTRACTS` list with a blocking `contract-audit-phase3` target (mirror
`contract-audit-phase2`, Makefile:1156, invoked from tier3 at Makefile:314). The cross-process
reproducibility gate (D-16 tier3 half) is a new Make target spawning two processes and comparing
hashes — no direct analog; follow the tier3 wiring style and the Verification Discipline rule 1
(`rc=$?` never through a pipe; the repo has shipped that bug twice). Local verification caveat:
tier2 is red on arm64 (D-ITEM-02) and its headline test step runs zero tests (D-ITEM-03) — verify
via SCOPED commands (`cargo test -p aprender-train --lib --features setfit ...`), the Phase 2
precedent.

## Shared Patterns

### Contract annotation on the single decision function
**Source:** `split.rs:240`, `rng.rs:82-85`, `hash.rs:60-63`, `ledger.rs:90-93`
**Apply to:** every Phase 3 gate function (config validation ladder, evidence gate, lock
minting, epoch-shuffle derivation, sklearn-relation equation)
```rust
#[provable_contracts_macros::contract("<contract-name>-v1", equation = "<equation_name>")]
fn the_one_gate(...) -> Result<_, TypedError> { ... }
```
One validating function, both/all doors call it (split.rs: "Two ladders that agree today are two
ladders that will disagree eventually"). aprender-train side already uses
`provable_contracts_macros::requires` (adamw.rs:6,30,109) — both macro families are available.

### Canonical bytes → SHA-256, summary + hash-bound detail
**Source:** `ledger.rs:94-140`, `hash.rs:43-47,73`, `manifest.rs:533` (`pair_manifest_hash`)
**Apply to:** evidence record (D-12), selection lock (D-14), loss-trace hash (TRN-06)
Fixed-field-order struct + `schema_version` + `Vec` whose order IS the event order; no timestamps,
no Duration, no HashMap. Hash the canonical bytes. Emit the full table separately; persist the hash.

### Typed errors that name the offender and the fix
**Source:** `device.rs:64-98`, `split.rs` gate ladder, `pairs.rs:400+`
**Apply to:** all Phase 3 fallible paths (repo bans `unwrap()`; `Result<(), String>` is the
explicitly rejected shape)
Every variant carries named fields for expected AND observed; `Display` cites the contract ID.

### Typestate: private fields, `pub(crate)`/gated constructors, PhantomData, trybuild proof
**Source:** `split.rs:85-118`, `select.rs:103` (`SelectedId`), `tests/ui.rs`
**Apply to:** `SetFitRun<S>`, the canonical-test access token, `FrozenProbeRun` separation
"A runtime field would make the interesting mistake compile and fail at run time, and a
compile-fail proof of 'cannot be constructed' is unobtainable against an expression that
compiles" (split.rs module doc) — identity is a TYPE PARAMETER, never a runtime field.

### In-band negative: negative + control + mirror, red in every `cargo test`
**Source:** `tests/negative_leaky.rs` (template), `tests/negative_materializing.rs`
**Apply to:** pair-weighted head fitter (D-08), frozen/1e-30-LR evidence-gate negatives (D-10),
GEMM thread-count falsification harness (D-13)

### Golden constants derived independently, never blessed from a first run
**Source:** `rng.rs` `rng_byte_encoding_golden_is_frozen` (derivation documented in the test doc)
**Apply to:** new RNG domains (epoch-shuffle, dropout), loss-trace hash fixtures, sklearn fixture
(Python side must assert `n_iter_ < max_iter` and record versions — Pitfall 7)

### Reuse-in-place surfaces (consumed, not modified — pin conventions, don't fork them)
| Asset | Signature | Cited at |
|---|---|---|
| `AdamW::step_refs` | `fn step_refs(&mut self, params: &mut [&mut Tensor])` | `optim/adamw.rs:184` (trait default at `optim/optimizer.rs:13`) |
| `clip_grad_norm_refs` | `(&mut [&mut Tensor], max_norm: f32) -> f32` (pre-clip norm) | `optim/clip.rs:65` |
| `resolve_device` | `(&str) -> Result<Device, DeviceError>`, fail-closed | `train/device.rs:110` |
| `SetFitMiniLm::trainable_parameters_mut` | `-> Vec<(String, &mut Tensor)>` HF dotted names | `setfit/mod.rs:440` |
| `pair_cosine_mse` | `(za, zb, &[f32]) -> Result<Tensor, SetFitError>` | `setfit/loss.rs:71` |
| `Selection` / `Split<Role>` / `AccessLedger` / `hash::exact_hash` | see head_input/lock sections | contrastive-data |

Do NOT compose with `train/trainer/core.rs::Trainer` (owns `Vec<Tensor>` + `Box<dyn Optimizer>`,
core.rs:33,61 — wrong surface for borrowed NAMED params and mid-step evidence capture). Sit beside
it, reusing the pieces above (RESEARCH "Alternatives Considered", D-05 named the pieces, not the
`Trainer`).

## No Analog Found

| File | Role | Data Flow | Reason / What to use instead |
|------|------|-----------|------------------------------|
| `train/setfit/reduce.rs` | numeric utility | fixed-order reduction | Nothing in-repo guarantees reduction order (RESEARCH-verified). Use the RESEARCH-supplied sequential f64-accumulate pattern; style kin is lbfgs.rs's indexed loops. |
| `train/setfit/verify.rs` trait seam | trait definition | reload round-trip | No verify-by-reload trait exists anywhere. Serde mechanics from ledger.rs round-trip; the trait shape itself is new — design it as Phase 4's drop-in seam. |
| tier3 cross-process repro gate (Make target + harness) | build gate | subprocess compare | No two-process hash-compare gate exists. Follow tier3 wiring style + Verification Discipline rule 1 (`rc=$?` capture, never through a pipe). |

## Metadata

**Analog search scope:** `crates/aprender-contrastive-data/{src,tests}`,
`crates/aprender-core/src/{optim,classification,glm,setfit,nn/dropout,primitives}`,
`crates/aprender-train/src/{optim,train}`, `contracts/`, `Makefile`, both Cargo.tomls
**Files scanned:** 28 read (targeted ranges for files > 600 lines)
**Pattern extraction date:** 2026-08-09
**Upstream inputs:** `03-CONTEXT.md` (16 locked decisions), `03-RESEARCH.md` (verified assets,
8 resolved discretion questions, 12 pitfalls)

---

## Revision 2 Addendum — files created by the cross-AI review replan

The replan added eight files with no pattern assignment in the original map. Each is listed with its
owning plan/task and its closest in-repo analog; every analog named here also appears in that task's
`read_first`.

### `crates/aprender-train/src/train/setfit/tune.rs` — `run_tuning`, the stage-one loop (03-05 T2)

The phase's largest new function, and the one the original map omitted entirely.

**Analog (decision, not copy):** `crates/aprender-train/src/train/trainer/core.rs` and
`crates/aprender-train/src/train/train_loop/basic.rs` — the existing generic trainer. 03-CONTEXT.md's
canonical refs direct reading both "before deciding whether the SetFit trainer composes with it or
sits beside it", and RESEARCH's Alternatives table answers: **sit beside it.** `Trainer` owns
`Vec<Tensor>` parameters plus a `Box<dyn Optimizer>`; the SetFit loop needs BORROWED NAMED parameters
from `SetFitMiniLm::trainable_parameters_mut()` and evidence capture between the optimizer step and
the gradient clear, which the generic loop has no seam for. Reuse `AdamW`, `clip_grad_norm_refs` and
the scheduler directly — D-05 named those pieces, not the `Trainer`.

**Analog (structure):** `crates/aprender-core/src/optim/lbfgs.rs` — sequential indexed numeric loops
with no `par_iter`, the house style for anything whose reduction order is load-bearing.

**Complexity note:** the specified body (6 pre-loop steps + a 13-step per-batch body) breaches the
project ceiling of cyclomatic 10, so 03-05 T2 prescribes the decomposition
(`preflight` / `snapshot_initial` / `baseline_encode` / `run_batch` / `absorb_batch_digests`) rather
than leaving it to be discovered at 03-10's `pmat analyze complexity` gate.

### `crates/aprender-train/src/train/setfit/bundle.rs` — `SetFitBundle` (03-08 T1)

**Analog:** `ledger.rs` `to_canonical_bytes` / `from_bytes` (lines 94-127) for the
schema-versioned canonical wire form, and `attestation.rs` (:73, :87) for the
`deny_unknown_fields` wire-struct precedent. The bounded-deserialization limits have no in-repo
analog — they are new, and contracted in 03-08 T3 as `bundle_limits`.

### `crates/aprender-train/src/train/setfit/evaluate.rs` — trusted validation evaluator (03-09 T1)

**Analog:** `prepared.rs::validation_witness` (:327) + `ValidationWitness::fingerprint_hex` (:155) —
a borrowed witness whose existence proves which split a fact came from. The evaluator extends the
same idea from existence to computation: the metric is computed under the witness rather than
asserted beside it.

### `crates/aprender-train/src/train/setfit/thresholds.rs` — frozen threshold source (03-06 T2)

**Analog:** Ph1 D-14's frozen-tolerance discipline in
`contracts/setfit-encoder-conformance-v1.yaml`, plus `tolerances_measured.json` in the Phase 1
fixture set. The new element is the direction of truth: the YAML is authoritative and the Rust
constants are checked against it by a `serde_yaml` parse, rather than two hand-kept copies.

### `crates/aprender-train/src/train/setfit/test_fixtures.rs` — shared trainer fixture (03-05 T1)

**Analog:** `crates/aprender-contrastive-data/tests/common/` — Phase 2's synthetic-row dataset
builders; and `SetFitMiniLm::from_slice_fixture` (setfit/mod.rs:236) with `fixtures_dir()`
(model_tests.rs:403) for the network-free encoder. Both are already `pub`, so no test-support
backdoor is required.

### `calibration_matrix_epsilon_basis` — `#[ignore]`-marked in-crate lib test (03-05 T3)

**Analog:** none in-repo for the matrix itself; it exists because >= 12 full tuning runs must not sit
on the default `evidence_` unit filter. It is deliberately NOT an out-of-crate integration target
(revision 3 / N-01): `run_tuning` is `pub(crate)` and `calibration_variants()` is `#[cfg(test)]`, so
`tests/` cannot reach either, and every widening that would fix that is forbidden elsewhere in the
phase. `#[ignore]` achieves the same isolation with zero visibility change. Closest in-repo
discipline: the Makefile's heavy scoped targets kept out of the fast path — the same reasoning,
applied at the test-attribute level instead of the Make-target level.

### `crates/aprender-train/tests/setfit_repro.rs` and `crates/aprender-core/tests/gemm_thread_determinism.rs` — subprocess harnesses (03-10 T2, 03-02 T3)

**Analog:** the `std::env::current_exe()` child-spawn pattern; 03-02's harness is written first and
03-10's reuses its shape. Both follow CLAUDE.md Verification Discipline rule 1 (direct rc capture)
and rule 2 (prove the mechanism engaged — `THREADS` and `PARTITIONS` lines, fixed pool sizes).

### `scripts/gen_multinomial_sklearn_fixture.py` — sklearn reference generator (03-04 T2)

**Analog:** `scripts/setfit_fixtures/generate_fixtures.py` and its `uv.lock` / `.python-version` —
Phase 1's pinned Python fixture workflow. The revision adds PEP 723 self-pinning so the script
declares its own dependency versions instead of relying on a recipe recorded elsewhere.
