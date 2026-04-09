# Section 33: Forjar Heavy Types Contracts

Version: 1.0
Status: proposed
Date: 2026-04-09

> Parent: [pv-spec.md](../pv-spec.md) §33

## 33.1 Motivation

Forjar is a Rust-native IaC tool with 236+ types across 50+ modules. It currently has 5 contracts
(blake3-state, dag-ordering, execution-safety, recipe-determinism, codegen-dispatch) covering the
execution kernel. The **store module** (content-addressed derivation, purity classification, FAR
archive), **task/pipeline system**, **event rulebooks**, **Copia delta sync**, and **sandbox
isolation** are entirely uncontracted despite being correctness-critical.

Forjar's design philosophy is **verifiable determinism** — every operation is content-addressed,
every execution order is deterministic, every state change is auditable. Contracts must enforce
these guarantees formally.

## 33.2 Domain Inventory

| ID | Domain | Priority | Contracts | Surface |
|----|--------|----------|-----------|---------|
| FJ-HVY-1 | Store / Content-Addressed | P0 | store-cas-v1 | Core storage invariant |
| FJ-HVY-2 | OCI / Container Images | P0 | oci-manifest-v1 | Build reproducibility |
| FJ-HVY-3 | Task / Pipeline | P0 | task-pipeline-v1 | Execution correctness |
| FJ-HVY-4 | Event / Rulebook | P1 | event-rulebook-v1 | Automation safety |
| FJ-HVY-5 | Plugin Lifecycle | P1 | plugin-lifecycle-v1 | Extension safety |
| FJ-HVY-6 | Secret Provider | P1 | secret-provider-v1 | Credential safety |
| FJ-HVY-7 | Copia Delta Sync | P0 | copia-delta-v1 | Data integrity |
| FJ-HVY-8 | Sandbox Isolation | P1 | sandbox-isolation-v1 | Security boundary |

## 33.3 Store / Content-Addressed Storage (store-cas-v1)

### Type Surface

```rust
// Core CAS types
struct StorePath { hash: Blake3Hash, name: String }
struct Derivation { inputs: Vec<StorePath>, builder: String, env: BTreeMap<String, String> }
struct StoreEntry { path: StorePath, references: Vec<StorePath>, registration_time: u64 }
enum PurityLevel { Pure(0), NetworkAccess(1), Impure(2), Unrestricted(3) }
```

### Equations

**derivation_determinism**: Same inputs always produce same store path.

```
derive: Derivation -> StorePath
  derive(d₁) = derive(d₂) when d₁.inputs = d₂.inputs ∧ d₁.builder = d₂.builder ∧ d₁.env = d₂.env
  StorePath.hash = BLAKE3(canonical_serialize(derivation))
```

**closure_completeness**: A store path's closure contains all transitive references.

```
closure: StorePath -> Set<StorePath>
  ∀ ref in entry.references: ref ∈ closure(entry)
  ∀ ref in entry.references: closure(ref) ⊆ closure(entry)
  Closure is the least fixed point of the reference relation
```

**purity_monotonicity**: Higher purity level is more restrictive.

```
purity: Derivation -> PurityLevel
  Pure(0): no network, no impure inputs, hermetic sandbox
  NetworkAccess(1): network allowed, inputs still pure
  Impure(2): system state may affect output
  Unrestricted(3): no guarantees
  purity(d) = max(purity(input) for input in d.inputs) ∪ d.own_purity
```

**far_archive_roundtrip**: FAR (Forjar Archive) pack/unpack is identity.

```
far: Directory -> bool
  unpack(pack(dir)) = dir
  ∀ file in dir: hash(file_before) = hash(file_after)
  Preserves: permissions, ownership, symlinks, timestamps
```

**gc_safety**: Garbage collection never removes live store paths.

```
gc: (Store, RootSet) -> Store
  ∀ path in closure(root) for root in RootSet: path ∈ gc(store)
  Only removes: paths not reachable from any root
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| determinism | Derivation deterministic | d₁ = d₂ → derive(d₁) = derive(d₂) |
| completeness | Closure is transitive | ref ∈ closure(x) ∧ ref' ∈ closure(ref) → ref' ∈ closure(x) |
| monotonicity | Purity propagates upward | purity(d) ≥ max(purity(input_i)) |
| roundtrip | FAR identity | unpack(pack(dir)) = dir |
| conservation | GC preserves live paths | ∀ live path: path ∈ gc(store) |
| precondition | Store path hash valid | StorePath.hash = BLAKE3(canonical_serialize(derivation)) |

### Falsification Tests

- **FALSIFY-CAS-001**: Two derivations with same inputs produce identical store path
- **FALSIFY-CAS-002**: Two derivations with different env produce different store path
- **FALSIFY-CAS-003**: Closure of a path with no references is {self}
- **FALSIFY-CAS-004**: Closure of A→B→C includes all three paths
- **FALSIFY-CAS-005**: FAR roundtrip preserves file permissions (0o755 vs 0o644)
- **FALSIFY-CAS-006**: FAR roundtrip preserves symlink targets
- **FALSIFY-CAS-007**: GC with empty root set removes all paths
- **FALSIFY-CAS-008**: GC with full root set removes nothing
- **FALSIFY-CAS-009**: Pure derivation with impure input escalates to impure

## 33.4 OCI / Container Images (oci-manifest-v1)

### Type Surface

```rust
struct OciManifest { schema_version: u8, media_type: String, config: Descriptor, layers: Vec<Descriptor> }
struct Descriptor { media_type: String, digest: String, size: u64 }
struct ImageConfig { architecture: String, os: String, rootfs: RootFs, history: Vec<History> }
struct LayerCache { digest_to_path: HashMap<String, PathBuf> }
```

### Equations

**manifest_digest_consistency**: Manifest digest matches content hash.

```
digest: OciManifest -> String
  digest = "sha256:" ++ hex(SHA256(canonical_json(manifest)))
  len(digest) = 7 + 64  (prefix + hex SHA256)
```

**layer_ordering**: Layers are applied in order, later layers override earlier.

```
apply_layers: Vec<Layer> -> Filesystem
  apply(layers[0..n]) = apply(layers[0..n-1]) ∪ layers[n]
  Where ∪ means: files in layers[n] override files in layers[0..n-1]
```

**layer_cache_hit**: Unchanged layers reuse cached digests.

```
cache: (LayerCache, Layer) -> (Descriptor, bool)
  digest(layer) ∈ cache → (cached_descriptor, hit=true)
  digest(layer) ∉ cache → (build_descriptor(layer), hit=false)
  cache_hit → no rebuild, no re-upload
```

**reproducible_build**: Same Dockerfile + context → same image digest.

```
build: (Dockerfile, Context) -> OciManifest
  build(df, ctx₁) = build(df, ctx₂) when file_hashes(ctx₁) = file_hashes(ctx₂)
  Requires: deterministic timestamps, sorted directory entries
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| determinism | Manifest digest deterministic | same content → same digest |
| ordering | Layer application order | later layers override earlier |
| equivalence | Cache hit avoids rebuild | cached layer = built layer (by digest) |
| determinism | Reproducible build | same inputs → same manifest |
| bound | Digest format | len("sha256:") + 64 hex chars |

### Falsification Tests

- **FALSIFY-OCI-001**: Manifest with reordered JSON keys produces same digest (canonical JSON)
- **FALSIFY-OCI-002**: Image with 3 layers, file in layer 3 overrides layer 1
- **FALSIFY-OCI-003**: Cached layer digest matches freshly built layer digest
- **FALSIFY-OCI-004**: Two builds from identical context produce identical manifest digest
- **FALSIFY-OCI-005**: Empty layer produces valid descriptor with size=0

## 33.5 Task / Pipeline (task-pipeline-v1)

### Type Surface

```rust
enum TaskMode { Batch, Pipeline, Service, Dispatch }
struct PipelineStage { name: String, steps: Vec<Step>, depends_on: Vec<String>, quality_gate: Option<QualityGate> }
struct QualityGate { metric: String, threshold: f64, operator: CompareOp }
enum TaskStatus { Pending, Running, Succeeded, Failed, Skipped, Cancelled }
struct HealthCheck { command: String, interval: Duration, retries: u32, timeout: Duration }
```

### Equations

**pipeline_dag_execution**: Stages execute in dependency order.

```
execute: Vec<PipelineStage> -> Vec<(StageId, TaskStatus)>
  ∀ stage S with depends_on = [A, B]:
    start_time(S) > end_time(A) ∧ start_time(S) > end_time(B)
  Independent stages may execute in parallel
```

**quality_gate_enforcement**: Failed quality gate blocks downstream stages.

```
gate: (QualityGate, MetricValue) -> bool
  gate.operator.eval(value, gate.threshold) = true → stage passes
  gate.operator.eval(value, gate.threshold) = false → stage FAILS, all dependents SKIPPED
```

**task_status_terminal**: Terminal states are final.

```
terminal: TaskStatus -> bool
  Succeeded → terminal (no further transitions)
  Failed    → terminal
  Skipped   → terminal
  Cancelled → terminal
  Pending   → non-terminal
  Running   → non-terminal
```

**health_check_retry**: Health check retries with backoff before declaring failure.

```
health: (HealthCheck, Attempts) -> TaskStatus
  attempts ≤ retries → retry after interval
  attempts > retries → Failed
  Total wall clock ≤ retries * (timeout + interval)
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| ordering | DAG execution order | depends_on satisfied before start |
| postcondition | Quality gate blocks dependents | gate fails → all dependents skipped |
| state_machine | Terminal states are final | no transition from Succeeded/Failed/Skipped/Cancelled |
| termination | Health check terminates | attempts ≤ retries + 1 |
| bound | Wall clock bounded | total ≤ retries × (timeout + interval) |

### Falsification Tests

- **FALSIFY-PIPE-001**: Stage with failed dependency is skipped, not run
- **FALSIFY-PIPE-002**: Quality gate with threshold 0.8 blocks stage scoring 0.7
- **FALSIFY-PIPE-003**: Quality gate with threshold 0.8 passes stage scoring 0.9
- **FALSIFY-PIPE-004**: Task in Succeeded state cannot transition to Failed
- **FALSIFY-PIPE-005**: Health check with 0 retries fails immediately on first failure
- **FALSIFY-PIPE-006**: Pipeline with diamond dependency (A→B, A→C, B→D, C→D) executes D last
- **FALSIFY-PIPE-007**: Independent stages B and C run in parallel (wall clock < sequential)

## 33.6 Event / Rulebook (event-rulebook-v1)

### Type Surface

```rust
enum TriggerKind { File, Process, Cron, Webhook, Metric, Manual }
struct Rule { trigger: TriggerKind, condition: Condition, actions: Vec<Action>, cooldown: Duration }
struct Condition { expression: String }  // CEL or similar
struct Action { resource: String, operation: Operation }
enum Operation { Apply, Destroy, Restart, Notify }
```

### Equations

**trigger_dispatch_completeness**: Every trigger kind has a handler.

```
dispatch: TriggerKind -> Handler
  ∀ kind in TriggerKind: handler(kind) exists ∧ handler(kind) ≠ no-op
```

**cooldown_deduplication**: Events within cooldown window are deduplicated.

```
dedup: (Rule, Event, LastFired) -> bool
  now - last_fired < cooldown → event suppressed
  now - last_fired >= cooldown → event fires
```

**action_ordering**: Actions within a rule execute sequentially.

```
actions: Rule -> Vec<ActionResult>
  ∀ i < j: end_time(actions[i]) < start_time(actions[j])
  action[i] fails → actions[i+1..] skipped (fail-fast)
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| completeness | All triggers handled | ∀ kind: handler(kind) exists |
| idempotency | Cooldown dedup | events within cooldown → 1 execution |
| ordering | Sequential actions | action[i] completes before action[i+1] starts |
| soundness | Fail-fast | action failure → remaining skipped |

### Falsification Tests

- **FALSIFY-EVT-001**: File trigger on modified path fires exactly once
- **FALSIFY-EVT-002**: Two events within 10s cooldown fire only once
- **FALSIFY-EVT-003**: Event after cooldown expires fires normally
- **FALSIFY-EVT-004**: Second action not executed when first action fails
- **FALSIFY-EVT-005**: Manual trigger with no condition always fires

## 33.7 Plugin Lifecycle (plugin-lifecycle-v1)

### Type Surface

```rust
enum PluginState { Discovered, Loaded, Initialized, Running, Stopped, Error }
struct PluginManifest { name: String, version: Version, permissions: Vec<Permission> }
enum Permission { ReadFs, WriteFs, Network, Exec, Secrets }
struct PluginSchema { inputs: Vec<ParamDef>, outputs: Vec<ParamDef> }
```

### Equations

**lifecycle_state_machine**: Plugin follows valid state transitions.

```
transition: (PluginState, Event) -> Result<PluginState, InvalidTransition>
  Discovered → Loaded → Initialized → Running → Stopped
  Any → Error (on failure)
  Error → Discovered (on reload)
  No skip: Discovered → Running is INVALID
```

**permission_scoping**: Plugin cannot exceed declared permissions.

```
enforce: (Plugin, Operation) -> Result<(), PermissionDenied>
  operation.requires ⊆ plugin.manifest.permissions → Ok
  operation.requires ⊄ plugin.manifest.permissions → Err(PermissionDenied)
```

**schema_validation**: Plugin inputs validated against manifest schema.

```
validate: (PluginSchema, Inputs) -> Result<ValidInputs, SchemaError>
  ∀ required in schema.inputs: required.name ∈ inputs
  ∀ input in inputs: type(input.value) matches schema.inputs[input.name].type
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| state_machine | Valid transitions | No skip, Error recoverable via reload |
| precondition | Permission enforcement | operation.requires ⊆ declared |
| soundness | Schema validation | required inputs present, types match |

### Falsification Tests

- **FALSIFY-PLG-001**: Plugin without Network permission cannot make HTTP request
- **FALSIFY-PLG-002**: Plugin in Discovered state cannot execute (must initialize first)
- **FALSIFY-PLG-003**: Plugin with missing required input returns SchemaError
- **FALSIFY-PLG-004**: Plugin failure transitions to Error, not crash

> Continued in [forjar-heavy-types-contracts-2.md](forjar-heavy-types-contracts-2.md) (sections 33.8-33.12)
