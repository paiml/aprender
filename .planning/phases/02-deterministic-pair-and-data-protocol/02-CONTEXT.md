# Phase 2: Deterministic Pair and Data Protocol - Context

**Gathered:** 2026-08-08
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers **the deterministic, provenance-complete, leakage-safe, non-quadratic data
protocol** that every later phase replays from: canonical split acquisition with per-row content
identity, balanced few-shot selection (8/16/32/64 per class × the ten contracted seeds) emitting a
stable selected-ID manifest, and bounded contrastive pair generation with explicit singleton and
budget semantics.

**In scope:** DATA-01 through DATA-06 only.

**Out of scope for this phase** (each belongs elsewhere, do not pull forward):
- The `SetFitTrainer`, encoder tuning loop, batching schedule, multinomial head, lifecycle
  states — Phase 3
- APR persistence of the sampler policy/seeds, CLI predict/eval surfaces, serving parity —
  Phase 4
- Benchmark execution, the 40-cell matrix, metrics, calibration, claims — Phase 5
- MCP servers, object-storage adapters, forecasting or other model families — a future milestone
  (see `<deferred>`; this phase only refuses to make them harder)

Phase 2 **does** grow a user-facing surface — ROADMAP phrases every Phase 2 criterion as
"A user can…", unlike Phase 1's "A developer can…". Phase 1 was library-only; this one is not.

</domain>

<decisions>
## Implementation Decisions

### Crate Home and Boundaries

- **D-01:** Create a **new workspace crate `aprender-contrastive-data`** at
  `crates/aprender-contrastive-data/`. It owns contrastive/Siamese **data construction** as a
  general capability — class buckets, balanced few-shot selection, bounded positive/negative pair
  sampling, split roles, dataset fingerprints. SetFit is its first consumer, not its owner.
- **D-02:** This is a **deliberate, recorded deviation from `.planning/research/STACK.md`'s crate
  ownership table**, which assigns the sampler to `aprender-data`. Evidence gathered during
  discussion: `aprender-train` depends on **only** `aprender-core`; `apr-cli` depends on
  `aprender-core`, `aprender-contracts`, `aprender-mcp`, `aprender-train-*`; **neither depends on
  `aprender-data`**, and `aprender-data` does not depend on `aprender-core`. Following the table
  literally would add two new dependency edges into a crate that also carries `streaming.rs`,
  `s3.rs`, `federated.rs`, and a 238 KB `generated_contracts.rs`. The planner must verify these
  edges still hold before implementing.
- **D-03:** Ship **concrete, generically named types** — `ClassBuckets`, `FewShotSelector`,
  `PairSampler` — with SetFit's cosine-MSE pair semantics as the only implementation. **Do not
  introduce `PairStrategy` / `SamplingStrategy` trait abstractions in v1.** A trait carved from
  one implementation fits the second consumer badly, and PF-003 wants pair semantics pinned hard,
  which is easier against a concrete type. The extension point is that the crate has room, not
  that it ships empty traits.
- **D-04:** The crate's public API is **bytes-in/bytes-out and typed values — no `std::fs`, no
  network, no path-shaped APIs**. `apr-cli` owns every filesystem adapter. This is **enforced**,
  by a contract obligation plus a build check, not left as an aspiration. Rationale is the MCP
  direction in `<specifics>`: on Lambda the manifest is an S3 object, not a file, and a consumer
  there must be a wrapper rather than a rewrite.
- **D-05:** The **ETL seam**: the generic model moves into the crate — labeled-example schema,
  typed split roles, JSONL read/write, per-row content hashes, dataset fingerprint, cross-split
  duplicate detection. `apr-cli` keeps only TweetEval-specific parts: URLs, filenames, expected
  counts, label names, clap args, and the `ureq` network fetch. Phase 3 and Phase 5 then consume
  typed splits without depending on CLI modules (the OPS-01 principle applied early).

### Existing Work and Baseline

- **D-06:** The **813 uncommitted lines land as-is first, in their own PR**, before Phase 2
  planning proceeds: `crates/apr-cli/src/commands/data_tweeteval.rs`,
  `contracts/tweet-eval-stance-benchmark-v1.yaml`, `docs/examples/tweet-eval-stance.md`, and the
  accompanying `eval` `F_avg` changes. It already carries a contract and seven passing
  falsification tests. Phase 2 then relocates the core into `aprender-contrastive-data` as a
  reviewable diff against a tracked baseline, so a refactor regression is visible instead of
  tangled with new work and a bisect does not land on one giant commit.
- **D-07:** Treat DATA-01 as **largely built** and DATA-02 as **half-built**. Already covered:
  malformed rows, non-UTF-8, text/label length mismatch, unknown labels, exact per-split and
  per-class count contracts, SHA-256 hashed from the same bytes that were parsed, and the
  `revision_verified` honesty flag. **Not covered and therefore Phase 2 work:** duplicate IDs,
  cross-split duplicate *content*, and conflicting source roles. DATA-03/04/05/06 do not exist
  at all.

### Manifests and Replay

- **D-08:** The **selected-ID manifest is a materialized artifact** — ordered IDs plus semantic
  hash plus provenance — because EVAL-02 requires Phase 5 to hand *the identical* sampled-ID set
  to the 9B LoRA baseline in every one of the 40 cells. A shared file makes that a fact; two
  independent regenerations only make it a hash comparison.
- **D-09:** **Pair manifests are replayed, not stored.** Pairs regenerate from
  `(selection hash, seed, policy, budget)`; only the pair-manifest hash is persisted. An explicit
  dump path exists for audit and fixture generation. This keeps 40 cells × epochs of pair bytes
  off disk and lets a serverless trainer carry a ~200-byte config record instead of a pair file.

### Pair Semantics, Budget, and Degenerate Classes

- **D-10:** **Streaming draw + seen-set.** Per pair: draw a class then two distinct members
  (positive), or two distinct classes then one member each (negative). O(1) per draw, O(N) bucket
  state. **Canonicalize to `(min, max)`** so both orientations and conflicting labels for the same
  unordered pair are structurally impossible rather than merely tested against.
- **D-11:** The `unique` strategy keeps a **seen-set bounded by the budget** — still
  `O(examples + budget)` — and **fails closed with a typed error when the budget exceeds the
  available unique pair capacity**, rather than rejection-looping forever. Capacity is computable
  in O(K) from class sizes.
- **D-12:** **Self-pairs `(x, x)` are structurally impossible in the type**, matching SetFit's
  exclusion. No configuration value can resurrect them.
- **D-13:** **`SingletonPolicy` is an explicit, version-tagged enum recorded in the manifest**,
  default **`NegativesOnly`**: a class with exactly one selected example produces no positives,
  still appears in negatives, and still contributes its one row to the head. Never silent — the
  manifest states the policy and counts affected classes. This **closes the STATE blocker**
  *"Decide and version singleton-class and bounded-oversampling behavior during phase planning."*
  A class with **fewer examples than `shots_per_class`** is a separate and unambiguous case: a
  typed error at selection time (DATA-02 "invalid class counts").
- **D-14:** **Default `max_pairs_per_epoch` = SetFit's closed-form oversampling count**, clamped
  by a configurable hard cap. Positives `Σ C(n_k, 2)`, negatives `Σ_{j<k} n_j · n_k`, balanced to
  `2 · max(...)`, targeting strict 1:1. Computing that count is **O(K)**; only *enumerating* the
  pairs is quadratic — so faithful counts cost nothing. Worked values: 24 examples (8-shot × 3
  classes) → 384 pairs/epoch; 192 examples (64-shot) → 24,576.
- **D-15:** The **declared deviation from SetFit is therefore narrow and must be stated in both
  the manifest and the contract**: *identities are sampled rather than enumerate-then-shuffled,
  and the count is capped above N.* Per PF-008, it is labeled an explicit Aprender policy and
  **never attributed to SetFit**.

### Leakage Enforcement

- **D-16:** **Typed roles + runtime validation + access ledger — all three.** Phantom-typed
  `Split<Train>` / `Split<Validation>` / `Split<Test>`: the selector accepts only `Split<Train>`,
  and the pair sampler accepts only IDs carried by a `Selection` built from it, so a library
  caller cannot express leakage. Runtime hash and membership validation still runs at the
  **bytes → typed boundary**, because deserialized object-storage bytes are untrusted and a
  mislabeled `source_split` field would otherwise become a `Split<Train>` the compiler accepts.
  An **access ledger** records every split touched, which Phase 5's selection-lock gate reads.
- **D-17:** **Two hashes per row, for two different jobs.** Exact SHA-256 over the raw `input`
  bytes is **identity and provenance** (APR round-trip, manifest attestation). A **normalized**
  hash — NFC, trimmed, internal whitespace collapsed, **no casefolding** — is **leakage
  detection**, catching the retweet-with-a-trailing-space variant that exact matching misses.
  Casefolding is excluded deliberately: it would collide legitimately distinct short posts. The
  normalization is contracted and versioned so it cannot drift.
- **D-18:** **Cross-split duplicates are excluded from the training selection pool and recorded,
  not fatal.** At prepare time, detect duplicate groups and deterministically remove affected rows
  from the pool; record every excluded ID and the reduced per-class pool size in the manifest.
  Rationale: hard-failing at *selection* time makes failures seed-dependent, so some of the 40
  cells die and Phase 5's completeness gate rejects the run for a reason unrelated to the method;
  hard-failing at *prepare* time hands upstream data quality a veto over DATA-01 entirely. A typed
  error fires only when the reduced pool can no longer supply `shots_per_class` — a real failure.
  **Unverified:** whether canonical TweetEval abortion-stance actually contains any cross-split
  duplicates. Planning should determine this empirically; the design is correct either way.
- **D-19:** The **merged SetFit compatibility split deserializes as `Split<CompatibilityTest>`** —
  a role distinct from `Split<Test>` — and the compatibility profile emits **no `Split<Validation>`
  at all**. Any selection or tuning API demands `Split<Validation>`, so a compatibility-profile
  selection run **cannot be constructed**, not merely rejected. The access ledger records profile
  identity so Phase 5's selection-lock refuses a report built on it. Today's protection is a
  sentence in the manifest; this replaces documentation with a gate.

### Determinism and Evidence

- **D-20:** **`aprender-rand` (counter-based Philox/Threefry) drives selection and pair sampling**,
  keyed `key = hash(root_seed, domain)`, `counter = ordinal`. Draw *i* is a pure function of its
  index: thread-count independence becomes structural rather than asserted, domain separation
  falls out of the key, and the pair stream is shardable and resumable from an offset without
  replay. It is already in-workspace, so this adds no dependency.
- **D-21:** D-20 is the **second deliberate, recorded deviation from `.planning/research/STACK.md`**,
  which says use `rand_chacha` and *"do not add a second RNG crate."* Justification: PF-006's
  falsification test #2 requires identical results across 1 and N workers, and a stateful ChaCha
  stream makes draw *i* depend on every prior draw, so that property would be defended by
  discipline instead of by construction. Note `aprender-rand`'s library name is `trueno_rand`.
- **D-22:** **Pair identities cannot match SetFit's Python RNG — accepted deliberately.** This is
  a direct consequence of D-10 + D-20, and it partially limits PF-003's falsification test #2
  ("compare all pair identities and counts with frozen SetFit reference fixtures"). Counts are
  closed-form and fully comparable; identities are not.
- **D-23:** Evidence is therefore **invariant parity + Rust goldens**. The Phase 1 hash-locked
  `uv` environment (`setfit==1.1.3`) emits SetFit fixtures **only for RNG-independent facts**:
  pair counts on hand-enumerated tiny layouts, positive/negative balance, class correctness,
  self-pair and orientation exclusion, imbalanced and singleton cases. Pair **identities** get
  committed Rust golden fixtures under a **SHA-256 manifest (D-13 of Phase 1's pattern)** so
  re-baselining is a reviewable diff. Property tests cover the space between.
- **D-24:** **Contracts split to match the crate boundary.** New
  `contracts/contrastive-pair-protocol-v1.yaml` owns the dataset-agnostic surface — split roles,
  selection, pair semantics, budget, singleton policy, RNG derivation, bytes boundary — covering
  DATA-03/04/05/06 and the generic half of DATA-02. The existing
  `contracts/tweet-eval-stance-benchmark-v1.yaml` grows the dataset-specific half (DATA-01 and the
  dataset half of DATA-02); `pv diff` will suggest its semver bump. Rationale: a future
  forecasting or MCP consumer must inherit a contract that is not SetFit-shaped.
- **D-25:** **Capacity invariants + in-band negative variants.** Retained state is structurally
  bounded — sorted buckets O(N), seen-set with a hard capacity ceiling at the budget — and tests
  assert those capacity invariants plus the 10×-N-under-fixed-budget scaling property. Honesty
  comes from **two in-band negative variants that must fail their gates in every `cargo test`**:
  a **leaky sampler** that reaches for validation IDs, and a **materializing sampler** that builds
  the Cartesian product. This is Phase 1's D-24 discipline applied to this phase's two fakeable
  claims. A self-reported retained-state size is exactly as trustworthy as a self-reported
  decreasing loss.
- **D-26:** **`cargo-mutants` scoped to the new crate**, per Phase 1's D-25.
- **D-27:** **The DATA-02 / D-18 conflict resolves by splitting the semantics** (user decision,
  2026-08-08, prompted by cross-AI review). DATA-02 and ROADMAP criterion 1 previously demanded a
  typed failure for "cross-split duplicate content" while D-18 declared the same condition
  excluded-and-recorded, "not fatal" — a direct contradiction between two locked statements, made
  live by the real `train:70 ≡ val:3` duplicate the research found in canonical TweetEval.
  The resolution:
  - **Prepare-time duplicate *content* → exclude and record** (D-18 stands verbatim). Canonical
    prep on real TweetEval SUCCEEDS, with the excluded IDs and reduced per-class pool sizes in the
    manifest.
  - **Typed failure is reserved for actual span** — any pair or selection whose endpoints would
    cross split roles is fail-closed — **plus pool exhaustion**, when the reduced pool can no
    longer supply `shots_per_class`.
  - **ROADMAP criterion 5 is unchanged**: its "rejected fail-closed" refers to leakage into
    training, which is exactly what DATA-06's typestate protects. Criterion 1 and DATA-02 were
    amended to match this reading.
  Rationale: preserves D-18's Phase 5 argument (hard-failing at selection time makes failures
  seed-dependent, so cells die in the 40-cell completeness gate for reasons unrelated to the
  method) without weakening the leakage guarantee, which was always the point of "fail-closed".

### Carried Forward from Phase 1 (not re-litigated)

- Contract-first: `pv` is the only sanctioned contract tool; never a bash/yq/python workaround.
- `#[contract]` annotations land with the code, binding through `BindingRegistry` (Phase 1 D-27).
- Tier wiring unchanged (Phase 1 D-26): fast tests in `make tier2`, `pv validate` in tier3/tier4.
  A gate outside the tiers is a gate that stops being run.
- `unwrap()` banned via `.clippy.toml`; `unsafe_code = "forbid"`; all fallible paths return typed
  errors.
- The CPU feature matrix must stay green: `--no-default-features`, `--features setfit`,
  all-features (Phase 1 D-06).

### Claude's Discretion

The user explicitly delegated naming of the crate's internals only after choosing the crate name.
The following were surfaced and consciously left to research and planning as implementation
detail:

- Exact CLI command naming and shape for few-shot selection (`apr data few-shot` vs
  `apr data select`, flags, JSON output shape) and whether the pair dump is a subcommand or a flag
- The selected-ID manifest's field schema and file name
- Whether the `unique` and `undersampling` strategies ship in v1 at all, beyond the `oversampling`
  default that D-14 specifies
- How pair batches are ordered and reshuffled across epochs (Phase 3 consumes this; Phase 2 must
  not preclude it)
- Whether `aprender-contrastive-data` is published to crates.io in this milestone, and the
  MSRV / feature-matrix consequences of adding a workspace crate mid-milestone (see
  `.claude/skills/pre-release/SKILL.md`)
- The precise `SingletonPolicy` version-tag encoding in the manifest

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase requirements and scope
- `.planning/ROADMAP.md` — Phase 2 goal and the five success criteria that define "done"
- `.planning/REQUIREMENTS.md` — DATA-01 through DATA-06 verbatim; traceability table
- `.planning/PROJECT.md` — milestone core value, constraints (reproducibility, benchmark
  integrity, data licensing), and the out-of-scope list
- `.planning/STATE.md` — the Phase 2 blocker this context closes (D-13), and the cross-cutting
  constraint on CPU-only package/MSRV/feature combinations

### Prior phase context (locked, do not re-litigate)
- `.planning/phases/01-differentiable-minilm-conformance/01-CONTEXT.md` — D-05/D-06 feature gate
  and CPU matrix, D-12 hash-locked `uv` reference environment, **D-13 fixture SHA-256 manifest**,
  D-14 contract-resident constants committed before comparison, **D-23 one-contract-per-phase
  referencing pattern**, **D-24 in-band negative test**, D-25 scoped `cargo-mutants`, D-26 tier
  wiring, D-27 `#[contract]` co-evolution

### SetFit domain research (authoritative for this milestone)
- `.planning/research/PITFALLS.md` — **PF-002** (pair/split leakage) and **PF-003** (semantically
  wrong or quadratic pair generation) are this phase's reason to exist; **PF-006** (seeded runs
  still non-deterministic) begins here; **PF-007** (few-shot statistics hide seed sensitivity)
  creates the sampling manifest here. Also the Research Gaps section: *"Freeze the exact
  singleton-class sampler behavior and bounded-oversampling compatibility rule in Phase 2."*
- `.planning/research/STACK.md` — §2 "Tune the Encoder with SetFit's Pair Objective" (bucket and
  budget guidance); "Crate Ownership" table (**overridden by D-01/D-02 — read the override
  reasoning before following it**); "Supporting Libraries" `rand_chacha` row (**overridden by
  D-20/D-21**); "Numerical Reference and Verification Strategy" items 6 and 9
- `.planning/research/ARCHITECTURE.md`, `.planning/research/FEATURES.md`,
  `.planning/research/SUMMARY.md` — supporting milestone research

### Existing implementation this phase adopts and relocates
- `crates/apr-cli/src/commands/data_tweeteval.rs` — 813 lines; canonical + `setfit` profiles,
  count and class-count contracts, hash-from-parsed-bytes discipline, `revision_verified` flag,
  rollback-on-partial-write, JSONL row schema
- `contracts/tweet-eval-stance-benchmark-v1.yaml` — split sizes, label map, official `F_avg`
  invariant, few-shot protocol, provenance requirements, seven falsification tests, gate
  `F-TWEET-EVAL-001`
- `docs/examples/tweet-eval-stance.md` — user-facing workflow, offline/mirrored preparation,
  compatibility-profile warning
- `crates/apr-cli/src/data_commands.rs` — `DataCommands` / `TweetEvalStanceProfile` clap surface

### Codebase state
- `.planning/codebase/CONCERNS.md` — pair explosion, tokenizer/config drift, and the data-side
  gaps this phase closes
- `.planning/codebase/ARCHITECTURE.md` — layer boundaries and crate ownership rules
- `.planning/codebase/STACK.md` — workspace versions, feature flags, toolchain (Rust 1.93.0,
  MSRV 1.91), publishability constraints relevant to adding a crate
- `.planning/codebase/TESTING.md` — contract, property, mutation, and numerical-tolerance patterns
- `.planning/codebase/CONVENTIONS.md` — code conventions to match

### Existing contracts to reference
- `contracts/tensor-layout-v1.yaml` — row-major is mandatory repo-wide
- `contracts/rand-philox-v1.yaml` (referenced from `crates/aprender-rand/src/lib.rs`) — the RNG
  contract D-20 builds on

### Repository rules
- `CLAUDE.md` — Verification Discipline (all 8 rules; **rule 6 "one failing input is an anecdote"**
  applies directly to D-18's unverified duplicate question), "Contract Validation: DOGFOOD `pv`,
  NEVER bash", code search policy (`pmat query`, not grep/glob), tiered quality gates, git branch
  protection
- `.claude/skills/pre-release/SKILL.md` — publishability, MSRV, and feature-combination
  constraints that adding `aprender-contrastive-data` must not break
- `.claude/skills/apr-dogfood/SKILL.md` — dogfooding conventions

### Upstream sources (pinned)
- `cardiffnlp/tweeteval` @ revision `4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66`, path
  `datasets/stance/abortion` — never a mutable branch name
- SetFit sampling strategies (oversampling / undersampling / unique) — see
  `.planning/research/PITFALLS.md` Sources for the authoritative URL list

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `crates/apr-cli/src/commands/data_tweeteval.rs` — the canonical-dataset loader, class-count
  contract, hash-from-parsed-bytes discipline, and partial-write rollback are all directly
  reusable; D-05 splits them between the new crate and the CLI rather than rewriting them.
- `crates/aprender-rand/` (library name `trueno_rand`) — `Philox4x32`, `Threefry4x64`, and the
  `Rng` trait. Already contract-backed (`rand-philox-v1.yaml`) and already in-workspace; D-20
  builds selection and pair sampling on it.
- `crates/aprender-data/src/split.rs` — has a `stratified` split helper worth reading for
  balanced-selection precedent, even though D-01/D-02 place the new code elsewhere.
- Workspace `rand = "0.9"` (with `small_rng`) and `rand_chacha = "0.9"` already exist — they
  remain available for dropout and other stages; D-20 governs only this crate.
- `sha2`, `serde`, `serde_json` are existing workspace dependencies — the manifest, fingerprints,
  and both content hashes need nothing new.

### Established Patterns

- **Contract-first.** YAML in `contracts/` binds to Rust via `#[contract]` and
  `generated_contracts.rs`; `pv` is the only sanctioned tool.
- **Flat `crates/aprender-*` layout** — 78 workspace crates across 82 directories; adding one more
  follows the existing convention rather than inventing a nesting scheme.
- **Crate-name split.** `aprender-train`'s library name is `entrenar`, `aprender-compute`'s is
  `trueno`, `aprender-serve`'s is `realizar`, `aprender-rand`'s is `trueno_rand`. Use library names
  in Rust imports, directory names in path discussion.
- **Tiered gates.** `make tier1` (<1s) / `tier2` (<5s) / `tier3` (1–5min) / `tier4` (CI).
- **`unwrap()` banned**, `unsafe_code = "forbid"` workspace-wide.

### Integration Points

- New crate at `crates/aprender-contrastive-data/`, added to the root workspace members list and
  to the `[workspace.dependencies]` table.
- `apr-cli` gains a dependency on the new crate and keeps `data_tweeteval.rs` as a thin
  TweetEval-specific adapter over it (D-05).
- `aprender-train` gains the same dependency in Phase 3, when `SetFitTrainer` consumes typed
  splits, selections, and the pair stream. Phase 2 should not add that edge speculatively.
- New contract `contracts/contrastive-pair-protocol-v1.yaml`; existing
  `contracts/tweet-eval-stance-benchmark-v1.yaml` gains the dataset-specific obligations.
- **Nothing in `crates/aprender-core/` changes this phase** — the encoder from Phase 1 is not
  touched. **Nothing in `crates/aprender-serve/`.**

### Known Traps in This Territory

- **Dependency-edge assumption.** `.planning/research/STACK.md`'s crate ownership table does not
  match the actual `Cargo.toml` graph (D-02). Verify edges before trusting any ownership claim in
  the research documents.
- **`revision_verified` honesty.** The existing command already refuses to attest a local
  `--source` directory to an upstream revision. Any relocation must preserve that, and the
  falsification test `FALSIFY-TWEET-EVAL-006` guards it.
- **Hash-then-parse ordering.** The existing loader reads each source file exactly once and hashes
  *the same bytes it parsed*. Reading twice would let the recorded SHA-256 describe content that
  never passed the class-count contract. Preserve this when splitting the code.
- **Sourced-library option neutrality** and the `apr` binary-pinning rules in `CLAUDE.md` apply to
  any script this phase adds.

</code_context>

<specifics>
## Specific Ideas

- **The strategic frame the user supplied, which shaped D-04, D-09, and D-20.** The destination is
  aprender models exposed as **MCP tools**: an LLM agent drives training, then calls the deployed
  model as a tool for the exact symbolic/statistical work LLMs are weak at — classify a social
  post, forecast a series — instead of spending an LLM call per item. The delivery vehicle is
  **PMCP** (Rust MCP SDK) and **pmcp.run**, i.e. Rust on AWS Lambda with S3/DynamoDB behind it,
  reachable from agents and business-process automation. SetFit is the **first instance of a
  pattern**, not a one-off; other model families (time-series forecasting was the named example)
  follow the same shape.

  Concretely, that is why: the crate is generically named and dependency-light (a Lambda consumer
  must not transitively pull `ureq`/`s3`/`streaming`); its API is bytes-in/bytes-out (on Lambda a
  manifest is an S3 object, not a path); pairs replay from a tiny config record rather than
  shipping across a network boundary; `O(examples + budget)` is a cold-start and memory-ceiling
  constraint that is actually paid for rather than a purity argument; and per-row content hashes
  are the identity mechanism once bytes arrive from object storage instead of a trusted path.

- **The two overrides are the phase's most load-bearing decisions** and both were made from
  evidence found during discussion, not preference: the dependency graph contradicting the
  research's ownership table (D-02), and counter-based RNG making worker-count independence
  structural rather than asserted (D-21). Both must survive into planning **with their
  justifications attached**, or a later reader will "fix" them back toward the research doc.

- **The closed-form insight in D-14 is what makes fidelity and boundedness compatible.** PITFALLS
  states the two cannot both be claimed without defining the deviation. Because SetFit's
  oversampling *count* is O(K) to compute while only *enumeration* is quadratic, the deviation
  collapses to identities-and-cap alone — a far smaller thing to declare.

- **D-25's two negative variants are the phase's highest-value artifact**, exactly as Phase 1's
  detached-encoder test was. "No leakage" and "bounded memory" are both claims a passing test
  suite can assert without them being true.

</specifics>

<deferred>
## Deferred Ideas

- **MCP server exposure of trained aprender models** — PMCP + pmcp.run, Rust on Lambda with
  S3/DynamoDB, so LLM agents drive training and then call models as tools. A future milestone;
  `crates/aprender-mcp/` already exists in-tree as a home. Phase 2 does not build it, but D-04's
  bytes boundary and D-09's replayable pairs are chosen so it stays cheap to add.
- **The same MCP pattern for other model families** — time-series forecasting and other
  predictors. Future milestone; reinforces why D-01/D-03 keep the crate dataset- and
  algorithm-agnostic.
- **Triplet and N-way-K-shot episode samplers** in `aprender-contrastive-data` — v2 (EXT-02). The
  crate is *shaped* to accept them; v1 ships only what DATA-01→06 require, and the trait boundary
  gets extracted when a second consumer proves the shape (D-03).
- **Alternative contrastive objectives** (InfoNCE, SupCon, CoSENT, triplet) — v2 per
  REQUIREMENTS.md EXT-02; they change SetFit fidelity and numerical references.
- **Fuzzy near-duplicate detection** (MinHash/simhash over shingles) for leakage — considered and
  rejected for v1 in favor of D-17's conservative normalized hash. Revisit only if a real
  paraphrase leak is observed; it introduces a tunable threshold and false positives on short
  texts.
- **Persistent embedding or token caches** — explicitly out of scope for v1 (CACHE-01); PF-010
  documents the staleness risk.

</deferred>

---

*Phase: 2-Deterministic Pair and Data Protocol*
*Context gathered: 2026-08-08*
