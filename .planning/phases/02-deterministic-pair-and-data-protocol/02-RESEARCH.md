# Phase 2: Deterministic Pair and Data Protocol - Research

**Researched:** 2026-08-08
**Domain:** Deterministic few-shot data preparation, leakage-safe split handling, bounded contrastive pair sampling (SetFit fidelity) in pure Rust
**Confidence:** HIGH — every load-bearing claim below was verified this session against the pinned reference environment, the pinned upstream dataset, the live workspace dependency graph, crates.io, and the in-tree `pv` binary

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Crate Home and Boundaries

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
  *(Researcher note: verified 2026-08-08 — one edge claim has drifted; see Finding F3 below. The
  decision itself stands on its remaining rationale.)*
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

#### Existing Work and Baseline

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

#### Manifests and Replay

- **D-08:** The **selected-ID manifest is a materialized artifact** — ordered IDs plus semantic
  hash plus provenance — because EVAL-02 requires Phase 5 to hand *the identical* sampled-ID set
  to the 9B LoRA baseline in every one of the 40 cells. A shared file makes that a fact; two
  independent regenerations only make it a hash comparison.
- **D-09:** **Pair manifests are replayed, not stored.** Pairs regenerate from
  `(selection hash, seed, policy, budget)`; only the pair-manifest hash is persisted. An explicit
  dump path exists for audit and fixture generation. This keeps 40 cells × epochs of pair bytes
  off disk and lets a serverless trainer carry a ~200-byte config record instead of a pair file.

#### Pair Semantics, Budget, and Degenerate Classes

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
  *(Researcher note: this matches SetFit's DOCUMENTED semantics; the pinned 1.1.3 implementation
  actually emits self-pairs — see Finding F2. The deviation statement must widen accordingly.)*
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

#### Leakage Enforcement

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
  *(Researcher note: now verified empirically — exactly one duplicate group exists; Finding F1.)*
- **D-19:** The **merged SetFit compatibility split deserializes as `Split<CompatibilityTest>`** —
  a role distinct from `Split<Test>` — and the compatibility profile emits **no `Split<Validation>`
  at all**. Any selection or tuning API demands `Split<Validation>`, so a compatibility-profile
  selection run **cannot be constructed**, not merely rejected. The access ledger records profile
  identity so Phase 5's selection-lock refuses a report built on it. Today's protection is a
  sentence in the manifest; this replaces documentation with a gate.

#### Determinism and Evidence

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

#### Carried Forward from Phase 1 (not re-litigated)

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

### Deferred Ideas (OUT OF SCOPE)

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
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-01 | Acquire/mirror pinned TweetEval abortion-stance and produce canonical train/validation/test JSONL with exact labels, counts, hashes, and provenance without committing tweet text | Existing 813-line implementation verified working (Finding F6); pinned revision re-downloaded and re-validated this session — exact 587/66/280 counts and per-class counts match the contract (Finding F1); D-05 seam maps generic parts to the new crate |
| DATA-02 | Typed failure for malformed rows, duplicate IDs, unknown labels, invalid class counts, conflicting source roles, cross-split duplicate content | Existing typed errors cover the first half (D-07); the missing half (duplicate IDs, conflicting roles, cross-split content) has a REAL positive case in the pinned data — train:70 ≡ validation:3 (Finding F1) — so the detector gets a live golden, not only synthetic fixtures |
| DATA-03 | Select exactly 8/16/32/64 unique canonical-training examples per class per contracted seed with a stable selected-ID manifest | Counter-based selection algorithms (partial Fisher–Yates over sorted buckets, or keyed bottom-k) documented under Architecture Patterns; pool feasibility verified: after duplicate exclusion, per-class pools are 158/319/109, all ≥ 64 (Finding F1) |
| DATA-04 | Replay deterministic positive/negative pairs: labels agree with class identity, endpoints differ, unordered identities canonical, singleton behavior explicit | Pinned `setfit` 1.1.3 sampler source read and executed (Finding F2): pair label 1.0/0.0 semantics confirmed; orientation exclusion confirmed; self-pair behavior measured (docs and code DISAGREE — deviation statement must widen); singleton behavior in the reference measured (self-pairs, not skip) |
| DATA-05 | Per-epoch pair budget with `O(examples + budget)` state/storage, never a Cartesian product | Verified: the reference implementation itself materializes `np.triu_indices(n)` — O(N²) — regardless of `max_pairs` (Finding F2), so the bound is a genuine improvement; closed-form capacity formulas verified against both docs and executed reference; scaling-test and in-band-negative patterns documented |
| DATA-06 | Validation/test endpoints and merged compatibility split rejected fail-closed; manifest proves isolation | Typestate pattern + `trybuild` (already in-workspace, Finding F6) for compile-time "cannot be constructed" evidence; runtime boundary validation and access-ledger pattern documented; compatibility profile emits no validation split (verified in existing code) |
</phase_requirements>

## Summary

Phase 2 is unusually well-scoped: the user locked 26 decisions, the reference implementation is
pinned and installed in-tree, and a third of DATA-01/02 already exists as working, tested code.
Research therefore focused on **verifying the empirical unknowns the CONTEXT explicitly left
open** — and three of the four produced findings that materially change what the planner writes.

**Finding F1 (D-18's open question, resolved):** the pinned canonical TweetEval abortion-stance
data contains **exactly one cross-split duplicate**: train row 70 and validation row 3 are
byte-identical (SHA-256 `e3af840b5398272c89e5e1e3e1730c26b0f5bc856d3d46531a2a64dd8844a2c3`, 127
chars), both labeled `none`. No within-split duplicates exist; NFC/trim/whitespace normalization
and even casefolding find no additional groups. D-18's exclude-and-record design will therefore
be exercised by real data on every run: the `none` training pool shrinks 159 → 158, which still
comfortably supplies 64 shots. **Finding F2 (reference semantics, measured):** the pinned
`setfit==1.1.3` sampler **includes self-pairs `(x, x)`** — `shuffle_combinations` defaults to
`replacement=True`, i.e. `np.triu_indices(n, k=0)` — directly contradicting SetFit's own
documentation, which the CONTEXT's D-12 relied on. It also gives singleton classes a positive
self-pair (vs. D-13's `NegativesOnly`), shuffles with a **hardcoded internal seed 42** ignoring
the trainer seed, and materializes the full O(N²) index triangle regardless of `max_pairs`. For
this project's balanced layouts the D-14 worked values (384, 24,576) remain exactly correct, but
the declared deviation (D-15) must widen and the D-23 fixtures must record *measured* reference
behavior, not documented behavior. **Finding F3 (D-02's edge check, drifted):** `apr-cli` **does**
depend on `aprender-data` today — under the workspace alias `alimentar` (non-optional; it powers
`apr data audit/split/balance`). `aprender-train` does not, and `aprender-data` still has no
`aprender-core` edge. D-01 stands on its remaining rationale (arrow/parquet/zstd weight, Lambda
consumers, genericity), but planning documents must carry the corrected evidence. **Finding F4:**
`pv validate` **rejects** the in-flight `tweet-eval-stance-benchmark-v1.yaml` (PROVABILITY-001 ×2:
no `proof_obligations`, no `kani_harnesses`); it is currently invisible to tier3 only because
`$(CONTRACTS)` is an explicit list. Phase 2 must restructure it (CLAUDE.md fix option 1, shape
precedent in `setfit-encoder-conformance-v1.yaml`) and wire both phase contracts into
`$(CONTRACTS)`, or the D-24 gate is theater.

Everything else confirms the locked plan is buildable with zero new third-party dependencies:
`aprender-rand` (Philox, published 0.63.0), `sha2`, `serde`/`serde_json`, `thiserror`,
`unicode-normalization`, `proptest`, `trybuild`, and `cargo-mutants` are all already in the
workspace or installed. The crate name `aprender-contrastive-data` is unclaimed on crates.io, and
because `apr-cli` (published) will depend on it, **publishing it is mandatory, not optional** —
resolving one of the discretion items.

**Primary recommendation:** Land the D-06 baseline PR first, then build the crate in three
seams — (1) bytes→typed schema/split/hash/dedup layer, (2) Philox-keyed selection with a
materialized manifest, (3) budget-bounded pair streaming with closed-form capacity — each gated by
its own falsification tests, with the two in-band negative variants (leaky, materializing) written
*before* the real implementations they indict.

## Project Constraints (from CLAUDE.md)

Directives extracted from `./CLAUDE.md` that bind this phase's plans:

| Directive | Application to Phase 2 |
|-----------|------------------------|
| `main` protected; feature branch + PR + CI (`ci / gate`, `workspace-test`) | D-06 PR and every phase plan lands via PR |
| `pv` is the ONLY sanctioned contract tool — never bash/yq/python workarounds | Contract restructure (Finding F4) uses `pv validate`/`pv diff`; the fix menu in CLAUDE.md applies |
| Code search via `pmat query`, never grep/glob | Task actions referencing code discovery must use `pmat query` |
| `unwrap()` banned (`.clippy.toml` disallowed-methods); `unsafe_code = "forbid"`; SATD 0 | New crate inherits workspace lints; all fallible paths typed errors |
| Verification Discipline rule 1: never read `$?` through a pipe | Any scripts added by this phase |
| Verification Discipline rule 6: one failing input is an anecdote — vary before naming a cause | Applied in this research: duplicate detection run under three hash functions across all split pairs |
| Sourced shell libraries must be option-neutral; `bashrs lint` for scripts | Any new scripts |
| Root-anchored `.gitignore`/`Cargo.toml` exclude patterns; re-run `check_include_files.sh` + `check_package_includes.sh` after packaging changes | Adding a workspace crate touches packaging — run both scripts |
| Coverage + contracts co-evolve (Rule 7): coverage without contracts is REJECTED | `#[contract]` annotations land with the new crate's code |
| Tiered gates: tier1 (<1s) / tier2 (<5s) / tier3 (1–5min) / tier4 (CI) | New crate tests wire into tier2; `pv validate` into tier3 via `$(CONTRACTS)` |
| justfile preferred for project scripts (user global CLAUDE.md) | Repo uses Makefile tiers as established convention — follow the in-repo pattern, do not introduce a parallel justfile for this phase |
| Ask before crates.io publish cascade / CI workflow edits | Publishing the new crate happens at release time via existing `make publish` flow — flag, don't self-serve |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| JSONL schema, typed split roles, content hashes, dataset fingerprint | `aprender-contrastive-data` (new library crate) | — | D-05 ETL seam: generic model lives in the crate, bytes-in/bytes-out (D-04) |
| Cross-split duplicate detection + exclusion record | `aprender-contrastive-data` | — | Generic leakage capability; operates on typed rows, not files |
| Few-shot selection + selection manifest | `aprender-contrastive-data` | `apr-cli` (serialize to disk) | Crate computes; CLI owns the filesystem adapter |
| Pair sampling, budget, capacity, singleton policy | `aprender-contrastive-data` | — | The phase's core; Phase 3 consumes it as a library |
| RNG derivation (Philox key/counter discipline) | `aprender-contrastive-data` (thin `rng` module) | `crates/aprender-rand` (primitive) | D-20: primitive exists; only the domain-separation policy is new |
| TweetEval specifics: URLs, ureq fetch, expected counts, label names, clap args | `apr-cli` (`data_tweeteval.rs` as thin adapter) | — | D-05; network and paths are forbidden in the crate (D-04) |
| New CLI commands (selection, pair dump) | `apr-cli` (`DataCommands`) | — | Existing `apr data …` surface; dispatch via `dispatch_analysis.rs` |
| Access ledger | `aprender-contrastive-data` (type + recording) | Phase 5 (consumption) | D-16; Phase 2 only records |
| Contracts | `contracts/contrastive-pair-protocol-v1.yaml` (new) + `tweet-eval-stance-benchmark-v1.yaml` (grown, restructured) | `Makefile $(CONTRACTS)` | D-24 split + Finding F4 restructure |
| Reference fixtures (RNG-independent SetFit facts) | `scripts/setfit_fixtures/` (extend `generate_fixtures.py`) | committed fixtures + `manifest.sha256` in new crate's test tree | Phase 1 D-12/D-13 infrastructure reused (Finding F6) |
| Quality gates | `Makefile` tier2 (crate tests) / tier3 (`pv validate`, feature matrix) | CI | Phase 1 D-26 wiring precedent, measured-runtime comment style |

## Standard Stack

Everything is already in the workspace or in-tree. **This phase adds zero new third-party
dependencies.** All versions below read directly from the workspace manifests this session.

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `aprender-rand` (lib `trueno_rand`) | 0.63.0 (workspace member; **published on crates.io — verified via API this session**) | Philox4x32-10 counter-based RNG: `with_key_counter`, `generate_at` (stateless), `next_4u32` | D-20 locked; `generate_at(key, counter)` is exactly the pure-function-of-index primitive the determinism claims need `[VERIFIED: crates.io API + source read]` |
| `sha2` | 0.10 (workspace) | Exact + normalized content hashes, dataset fingerprint, manifest hashes | Already used by the existing loader with hash-from-parsed-bytes discipline `[VERIFIED: workspace Cargo.toml]` |
| `serde` / `serde_json` | 1.0 / 1.0 (workspace) | JSONL row schema, manifests; `BTreeMap` for deterministic key order | Existing pattern in `data_tweeteval.rs` `[VERIFIED: workspace Cargo.toml]` |
| `thiserror` | 2.0 (workspace) | Typed error enum (`ContrastiveDataError`) | Repo-wide convention; `aprender-rand/src/error.rs` is the minimal template `[VERIFIED: source read]` |
| `unicode-normalization` | 0.1 (already a dep of `aprender-train` and `apr-cli`; in Cargo.lock) | NFC step of the D-17 normalized hash | Already vetted in-workspace for the BPE tokenizer (C-TOK-BPE-001) `[VERIFIED: grep of workspace manifests + Cargo.lock]` |
| `provable-contracts-macros` | 0.3 (crates.io pin, as used by `aprender-core` and `apr-cli`) | `#[contract]` annotations binding to the phase contract | Phase 1 D-27 pattern; live example at `crates/aprender-core/src/autograd/ops/pooling.rs:50` `[VERIFIED: grep]` |

### Supporting (dev-dependencies)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `proptest` | 1.6 (workspace; note root `Cargo.toml` carries a debug-assertion workaround profile for it) | Pair-semantics property tests (D-23 "property tests cover the space between") | All invariants: positive same-class, negative diff-class, endpoints differ, canonical order, budget obeyed |
| `trybuild` | 1 (already a dev-dep of `aprender-contracts-macros` and `aprender-test-derive`) | Compile-fail proof that `Split<CompatibilityTest>` selection and validation-endpoint pairs **cannot be constructed** (D-16/D-19) | DATA-06 evidence; a compile-fail test is the only honest test of "not constructible" |
| `tempfile` | 3.14 (workspace) | CLI adapter tests only (the crate itself has no fs) | apr-cli side tests |
| `cargo-mutants` | 25.3.1 (**installed on this machine — verified**) | D-26 mutation scope on the new crate | `cargo mutants -p aprender-contrastive-data` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `aprender-rand` Philox | `rand_chacha` (STACK.md's original pick) | **Overridden by locked D-20/D-21** — stateful stream breaks structural worker-independence; do not re-litigate |
| Home in new crate | `aprender-data` (STACK.md ownership table) | **Overridden by locked D-01/D-02**; see Finding F3 for the corrected edge evidence — the weight argument (arrow 57 + parquet + zstd + lz4 + s3/streaming/federated) is the surviving and sufficient rationale |
| `unicode-normalization` | Hand-rolled NFC tables | Never — NFC is a large, versioned Unicode table; the crate is already in the workspace |
| `trybuild` compile-fail tests | Doc-comment `compile_fail` blocks | `compile_fail` doctests work but don't pin the error message; either is acceptable, `trybuild` gives reviewable `.stderr` snapshots |

**Installation:**
```bash
# No external installs. Workspace edits only:
# 1. Root Cargo.toml: add "crates/aprender-contrastive-data" to [workspace] members
# 2. Root Cargo.toml [workspace.dependencies]: add
#    aprender-contrastive-data = { path = "crates/aprender-contrastive-data", version = "0.63.0" }
# 3. crates/apr-cli/Cargo.toml: aprender-contrastive-data = { workspace = true }
```

**Version verification performed this session:**
```bash
# crates.io API queries (2026-08-08):
#   aprender-rand  -> max_version 0.63.0  (published; safe to depend on)
#   aprender-data  -> max_version 0.63.0
#   apr-cli        -> max_version 0.63.0  (published => new dep must also publish)
#   aprender-contrastive-data -> "does not exist"  (name available)
```

## Package Legitimacy Audit

This phase installs **no external packages**. Every dependency is either an in-tree workspace
crate or an existing entry in the workspace `Cargo.toml`/`Cargo.lock`. slopcheck was therefore
not required; the table below records registry status for the crates this phase newly *wires
together* (verified directly against crates.io this session, which is the correct ecosystem
registry for all of them).

| Package | Registry | Status | Source Repo | Disposition |
|---------|----------|--------|-------------|-------------|
| `aprender-rand` | crates.io | 0.63.0 published | in-tree `crates/aprender-rand` | Approved (in-tree source of truth) |
| `unicode-normalization` | crates.io | 0.1 (already in Cargo.lock, 2 in-tree consumers) | unicode-rs (established) | Approved (existing workspace dep) |
| `sha2`, `serde`, `serde_json`, `thiserror`, `proptest`, `tempfile`, `trybuild`, `provable-contracts-macros` | crates.io | all already pinned in workspace manifests + Cargo.lock | established | Approved (existing workspace deps) |
| `aprender-contrastive-data` | crates.io | **name unclaimed** (verified) | to be created in-tree | New crate — must be added to the publish cascade **before** `apr-cli` (Finding F5) |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                         apr-cli (filesystem + network tier)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  ureq fetch @ pinned revision ──┐                                       │
  │  --source local dir ────────────┼──> raw bytes (6 files, read ONCE)     │
  │  clap args / JSON output        │         │                             │
  └─────────────────────────────────┼─────────┼─────────────────────────────┘
                                    │         │ bytes           ^ manifests,
                                    │         v                 │ JSONL bytes
  ┌─────────────────────────────────┴───────────────────────────┴───────────┐
  │              aprender-contrastive-data (bytes-in/bytes-out)             │
  │                                                                         │
  │  [bytes -> typed boundary]                                              │
  │   parse + hash SAME bytes ──> typed rows ──┬─ malformed/UTF-8/label ──X │
  │                                            ├─ duplicate ID ───────────X │
  │                                            └─ conflicting role ───────X │
  │        │                                        (X = typed error)       │
  │        v                                                                │
  │  Split<Train> ── Split<Validation> ── Split<Test> ── Split<CompatTest>  │
  │        │              (typestate: selector accepts ONLY Split<Train>)   │
  │        v                                                                │
  │  cross-split dedup (exact + normalized hash) ──> exclusion record ──┐   │
  │        │  train pool minus excluded rows                            │   │
  │        v                                                            │   │
  │  ClassBuckets (sorted, stable) ──> FewShotSelector (Philox key/ctr) │   │
  │        │   pool < shots_per_class ──X typed error                   │   │
  │        v                                                            v   │
  │  Selection (ordered IDs + hashes) ════> selection manifest (D-08,       │
  │        │                                 materialized artifact)         │
  │        v                                                                │
  │  PairSampler (budget, SingletonPolicy, seen-set ≤ budget)               │
  │        │   budget > capacity ──X typed error (O(K) closed form)         │
  │        v                                                                │
  │  pair stream (CanonicalPair{min,max}, label from classes)               │
  │        ├──> pair-manifest hash (persisted; pairs replayed, D-09)        │
  │        └──> explicit dump (audit / fixture generation)                  │
  │                                                                         │
  │  AccessLedger: records every split touched + profile identity ──> P5    │
  └─────────────────────────────────────────────────────────────────────────┘
             │ consumed in Phase 3 (SetFitTrainer)          │ Phase 5 gate
             v                                              v
       typed Selection + pair stream                  selection-lock reads ledger
```

### Recommended Project Structure

```
crates/aprender-contrastive-data/
├── Cargo.toml               # version.workspace, rust-version via workspace (1.91), publishable
├── src/
│   ├── lib.rs               # crate docs, contract reference, re-exports; forbid(unsafe_code) via workspace lints
│   ├── error.rs             # ContrastiveDataError (thiserror; one variant per DATA-02 failure class)
│   ├── schema.rs            # LabeledExample, JSONL parse/encode over &[u8] (deny_unknown_fields)
│   ├── split.rs             # SplitRole typestate: Split<Train>/<Validation>/<Test>/<CompatibilityTest>
│   ├── hash.rs              # exact SHA-256 + normalized hash ("nfc-trim-ws-v1"), dataset fingerprint
│   ├── dedup.rs             # cross-split duplicate groups, deterministic exclusion record
│   ├── buckets.rs           # ClassBuckets — sorted per-class ID lists, O(N)
│   ├── select.rs            # FewShotSelector, Selection, selection-manifest model
│   ├── pairs.rs             # PairSampler, CanonicalPair, SingletonPolicy, capacity math
│   ├── rng.rs               # domain-separated Philox derivation (key = H(root_seed, domain), counter = ordinal)
│   ├── ledger.rs            # AccessLedger (split role + profile + purpose records)
│   └── manifest.rs          # canonical serialization + semantic hashes for all manifests
├── tests/
│   ├── goldens/             # Rust golden fixtures + manifest.sha256 (Phase 1 D-13 pattern)
│   ├── setfit_reference/    # committed RNG-independent SetFit count fixtures (from generate_fixtures.py)
│   ├── negative_leaky.rs    # in-band negative: sampler that reaches for validation IDs MUST fail its gate
│   ├── negative_materializing.rs  # in-band negative: Cartesian materializer MUST fail the memory gate
│   └── ui/                  # trybuild compile-fail: compat-profile selection not constructible
└── (no build.rs, no fs, no network — enforced; see Pattern 5)
```

### Pattern 1: Typestate split roles with untrusted-bytes validation (D-16, D-19)

**What:** Phantom-typed `Split<Role>` where the role is a zero-sized type; the *only*
constructor is the bytes→typed boundary function that validates the embedded `source_split`
field, per-row hashes, and (for compatibility profile) that no `Split<Validation>` value can be
produced at all.
**When to use:** All split handling in the crate.
**Example (sketch — planner refines):**
```rust
// Source: pattern; workspace precedent for typestate is Phase 3's planned lifecycle states
pub struct Train;
pub struct Validation;
pub struct Test;
pub struct CompatibilityTest;

pub struct Split<R> {
    rows: Vec<LabeledExample>,      // private — no public constructor
    fingerprint: DatasetFingerprint,
    _role: PhantomData<R>,
}

impl Split<Train> {
    /// The ONLY way to obtain Split<Train>: validates source_split field,
    /// row hashes, duplicate IDs, and class counts against the declaration.
    pub fn from_jsonl_bytes(bytes: &[u8], decl: &SplitDeclaration) -> Result<Self, ContrastiveDataError> { … }
}

// FewShotSelector::select(&Split<Train>, …) — no impl exists for any other role.
// The setfit compatibility profile deserializes ONLY as Split<CompatibilityTest>;
// there is no API from CompatibilityTest to Train/Validation. trybuild pins this.
```
Runtime validation still runs at the boundary (a mislabeled `source_split` in honest-looking
bytes must be a typed error, not a compiler-accepted `Split<Train>`).

### Pattern 2: Counter-based RNG derivation (D-20)

**What:** Every random decision is `Philox4x32::generate_at(key, counter)` — a pure function.
`key = trunc64(SHA-256(domain_tag ‖ root_seed_le ‖ domain_string))`, `counter = [ordinal_lo,
ordinal_hi, stream_id, 0]`. No mutable RNG state crosses a function boundary.
**When to use:** Selection draws and pair draws. Never `next_f32` for index draws (23-bit
mantissa); consume `u32`/`u64` lanes directly.
**Example:**
```rust
// Source: crates/aprender-rand/src/philox.rs (in-tree, read this session)
use trueno_rand::Philox4x32;

fn draw(key: [u32; 2], stream: u32, ordinal: u64) -> [u32; 4] {
    let counter = [ordinal as u32, (ordinal >> 32) as u32, stream, 0];
    Philox4x32::generate_at(key, counter) // stateless — pure function of (key, counter)
}
```
Bounded integers from a draw: use 64-bit multiply-shift (`((x as u128 * n as u128) >> 64) as u64`)
from a 64-bit lane. This is deterministic, branch-free, index-pure, and its non-uniformity is
< 2⁻⁴⁴ for the ranges in this phase (buckets ≤ 587, pair spaces ≤ ~10⁵); contract it as *the*
derivation rather than an approximation of some other one. `[ASSUMED — see Assumptions A3]`

### Pattern 3: Selection = partial Fisher–Yates over sorted buckets

**What:** For each class: sort candidate IDs (stable, canonical order), then run a k-step
Fisher–Yates where swap index `j_i` for step `i` comes from draw ordinal `i` in domain
`select/{class_label}`. First `shots_per_class` slots are the ordered selection. O(pool) state,
O(k) draws, unbiased, deterministic, and yields an *ordered* selection directly (DATA-03's
"ordered selected-ID manifest").
**Alternative:** keyed bottom-k (assign each candidate `generate_at(key, id_ordinal)`, take k
smallest) — order-independence is even more structural, but the output order needs a defined sort.
Either satisfies the contract; **recommend partial Fisher–Yates** because its output order is
self-evidently the draw order, which is easier to state as a contract equation.

### Pattern 4: Pair drawing with capacity-weighted class choice + unranking

**What (positive draw `i`):** choose class `k` with probability `C(n_k,2) / Σ C(n_j,2)` (prefix
sums, O(log K) binary search), then unrank a uniform index in `[0, C(n_k,2))` to an `(a, b)`
member pair via triangular unranking. **Negative draw:** choose unordered class pair `(j,k)`
weighted `n_j·n_k`, then one member each. Canonicalize IDs to `(min, max)` at construction.
**Why:** matches the uniform-over-possible-pairs marginal that SetFit's shuffled enumeration
implies, with zero rejection loops and O(1) state per draw; for the balanced few-shot layouts the
weights are equal, but the crate is generic (D-01) so contract the weighted form.
**Stream order:** alternate P,N,P,N — this matches pinned SetFit's `zip_longest` interleave
(verified in `sampler.py __iter__`) and gives strict 1:1 balance at every prefix.

### Pattern 5: Enforced bytes boundary (D-04)

**What:** Two mechanical checks, following the in-repo `setfit-feature-matrix` precedent
(`Makefile` lines ~265–290):
1. **Dependency closure check:** `cargo tree -p aprender-contrastive-data -e normal` must match an
   allowlist (no `ureq`, `tokio`, `arrow`, `memmap2`, …) — a Makefile target with the measured
   output committed to `target/`, exactly like the Phase 1 feature-matrix gate.
2. **Source surface check:** a unit test (or the contract obligation's falsification test) that
   the crate's public API contains no `std::path::Path`/`PathBuf` parameters and the crate does
   not link `std::fs`/`std::net` — simplest robust form: the crate has no `use std::fs`/`std::net`
   occurrences outside `#[cfg(test)]`, asserted by the build check target.
The contract obligation in `contrastive-pair-protocol-v1.yaml` names both checks so `pv` owns
the claim.

### Pattern 6: In-band negative variants (D-25, Phase 1 D-24 precedent)

**What:** Two deliberately wrong implementations compiled in every `cargo test`:
- `LeakySampler` (test-only): constructs pairs whose endpoint set includes a validation ID; the
  endpoint-membership gate must reject it with the exact typed error.
- `MaterializingSampler` (test-only): builds the full Cartesian product; the capacity-invariant
  gate (retained state ≤ f(examples, budget)) must fail on it at a size where the real sampler
  passes. Measure retained state structurally (e.g., `seen.len()`, bucket allocation counts
  exposed via a `#[cfg(test)]` introspection method), never by self-report.
**Precedent:** the Phase 1 detached-encoder negative in
`crates/aprender-core/src/setfit/encoder_tests.rs` — same discipline, new claims.

### Pattern 7: Manifest hashing discipline

**What:** Every manifest has (a) a canonical byte serialization — `serde_json` with `BTreeMap`
keys, no timestamps inside the hashed region — and (b) a `semantic_hash` = SHA-256 of those
bytes. The selection manifest is materialized (D-08); the pair manifest exists only as its hash
plus the replay tuple `(selection_hash, seed, policy_version, budget)` (D-09). Volatile metadata
(created-at) lives *outside* the hashed region, per PF-006 falsification test 1.

### Anti-Patterns to Avoid

- **Enumerate-then-cap** (the reference implementation's own trap): `np.triu_indices` materializes
  O(N²) before any cap applies — verified this session. The Rust sampler must never enumerate.
- **HashMap iteration anywhere in a deterministic path** — class buckets, duplicate groups, and
  manifests use sorted structures (`BTreeMap`, sorted `Vec`) exclusively (PF-006).
- **`next_f32` for index selection** — 23-bit precision and float rounding make derivations
  fragile; use integer lanes.
- **Modulo for bounded draws** — modulo bias is real at these ranges' edges and, worse,
  unauditable; use the contracted multiply-shift.
- **Asserting SetFit-doc behavior in fixtures** — the pinned implementation contradicts the docs
  on self-pairs (Finding F2); fixtures must record *measured* reference output.
- **Timestamps inside hashed manifest bytes** — breaks TRN-06/PF-006 replay comparison.
- **A gate outside the tiers** — new tests go in tier2, contract validation in tier3's
  `$(CONTRACTS)` list (which is explicit, not a glob — appending the YAML file alone does nothing).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Counter-based RNG | A new Philox/Threefry implementation | `trueno_rand::Philox4x32` (`generate_at`) | In-tree, tested, BigCrush-claimed, already published `[VERIFIED: source]` |
| Unicode NFC | Normalization tables | `unicode_normalization::UnicodeNormalization::nfc()` | Already in workspace; Unicode tables are versioned and large |
| Hashing | Custom digest/fingerprint math | `sha2::Sha256` | Existing provenance discipline hashes with it |
| Uniform k-subset selection | Ad-hoc shuffle of a cloned vec via `rand` | Partial Fisher–Yates driven by indexed Philox draws (Pattern 3) | Keeps draw-i purity; `rand::seq` shuffles are stateful-stream shaped |
| Unordered-pair canonical form | Runtime `assert!(a != b)` checks scattered at call sites | `CanonicalPair` newtype with the only constructor rejecting `a == b` and storing `(min, max)` | D-10/D-12: structural impossibility, not tested-against |
| "Cannot be constructed" claims | Doc prose | `trybuild` compile-fail tests | Already a workspace dev-dep; the only honest evidence for DATA-06's typestate claims |
| Contract validation / diffing | bash/yq/python scripts | `pv validate`, `pv diff`, `pv lint` | CLAUDE.md hard rule; a prebuilt `pv` 0.63.0 exists at `target/release/pv` |
| JSONL determinism | Custom serializer | `serde_json` + `BTreeMap` + explicit field order structs | Existing `data_tweeteval.rs` pattern round-trips today |

**Key insight:** every hard sub-problem in this phase (counter RNG, NFC, digesting, typestate,
compile-fail testing, contract tooling) already has a vetted in-workspace solution — the phase's
genuine new engineering is *composition and contracts*, not primitives.

## Runtime State Inventory

D-06 relocates existing (uncommitted) code, so the refactor categories were checked explicitly:

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | User-generated benchmark output dirs (`train.jsonl`, `validation.jsonl`, `test.jsonl`, `benchmark-manifest.json`, `schema_version: 1`) may exist on dev machines from running the uncommitted command | Keep output byte-compatible across the relocation, or bump `schema_version`; add a byte-parity golden against the D-06 baseline |
| Live service config | None — the command talks only to raw.githubusercontent.com at fetch time; no service stores its config | None (verified: no service integration in `data_tweeteval.rs`) |
| OS-registered state | None — no scheduled tasks, daemons, or shell registrations reference this code | None (verified by reading the module — pure CLI subcommand) |
| Secrets/env vars | None — unauthenticated HTTPS fetch; no env reads in the module | None (verified: no `std::env` use in `data_tweeteval.rs`) |
| Build artifacts | `target/release/pv` (0.63.0, current); published `apr-cli` 0.63.0 on crates.io does NOT contain the uncommitted files | None stale; the new crate enters the publish cascade at next release (Finding F5) |

## Common Pitfalls

### Pitfall 1: Fixtures encode SetFit's documentation instead of its behavior
**What goes wrong:** D-23 lists "self-pair … exclusion" among the fixture facts, and D-12 says
self-pair exclusion "match[es] SetFit's exclusion" — but the pinned `setfit==1.1.3`
implementation **emits self-pairs** (Finding F2). A fixture generated from the reference and
asserted as "no self-pairs" will be red on day one; a fixture hand-written from the docs will
falsely attest reference parity.
**Why it happens:** `shuffle_combinations(iterable)` defaults `replacement=True` →
`np.triu_indices(n, k=0)` → diagonal included. The docs describe intended semantics; the code was
never aligned.
**How to avoid:** Fixture files record **measured** pinned-reference counts (self-pairs included);
a separate fixture family records **Aprender's contracted counts** (self-pairs excluded). The
D-15 deviation statement in manifest + contract widens to three clauses: (1) identities sampled,
not enumerate-then-shuffle; (2) count capped above N; (3) self-pairs excluded, whereas pinned
setfit 1.1.3 includes the diagonal (contradicting its own documentation). Never attribute the
exclusion to SetFit's implementation (PF-008 discipline).
**Warning signs:** any fixture named `setfit_*` whose numbers match `Σ C(n_k,2)` exactly on a
layout where `Σ C(n_k,2)+N > Σ n_j n_k`.

### Pitfall 2: The closed-form default budget silently diverges from the reference on degenerate layouts
**What goes wrong:** D-14's formula (`2·max(Σ C(n_k,2), Σ n_j·n_k)`) equals the pinned reference's
epoch length on this project's balanced layouts (verified: 384 and 24,576 are exact — negatives
dominate for every `n ≥ 2` balanced 3-class layout) but **diverges when positives-with-self-pairs
exceed negatives** — measured: layout `[4,1]` → reference epoch length 22, doc-formula 12.
**How to avoid:** contract the formula as *Aprender's* default with the deviation stated; the
tiny-layout count fixtures must carry both numbers so the divergence is visible, versioned, and
deliberate.
**Warning signs:** a test comparing Aprender's default budget to a reference-measured epoch
length on an imbalanced or singleton layout.

### Pitfall 3: Assuming the pinned data is duplicate-free (or heavily duplicated)
**What goes wrong:** the design conversation treated cross-split duplicates as hypothetical.
**Measured:** exactly one group — train:70 ≡ validation:3, byte-identical, both `none`
(SHA-256 `e3af840b…44a2c3`). No within-split duplicates; normalization adds no groups.
**How to avoid:** the dedup golden test asserts *precisely this* exclusion on the real pinned
data path (opt-in network test, like the existing `pinned_upstream_satisfies_the_dataset_contract`),
and the reduced pools (none 158, against 319, favor 109) all still supply 64 shots. The
label-conflict variant of a duplicate (labels differ across splits) does NOT occur in real data —
that error path needs a synthetic fixture.
**Warning signs:** a manifest from the real dataset whose `excluded_ids` list is empty.

### Pitfall 4: The contract gate that pv rejects (or never reaches)
**What goes wrong:** `pv validate contracts/tweet-eval-stance-benchmark-v1.yaml` fails **today**
(PROVABILITY-001: no `proof_obligations`, no `kani_harnesses`). It escapes tier3 only because
`$(CONTRACTS)` (Makefile line 876) is an explicit hardcoded list. Appending the new
`contrastive-pair-protocol-v1.yaml` file to `contracts/` without touching the Makefile leaves the
D-24 gate unreachable — the exact failure mode Phase 1's D-26 comment block warns about.
**How to avoid:** (a) author `contrastive-pair-protocol-v1.yaml` in the pv-valid shape from the
start — the passing precedent is `setfit-encoder-conformance-v1.yaml` (`proof_obligations:` at
line 360, `kani_harnesses:` at line 812); (b) restructure the tweet-eval contract when it grows
its DATA-02 half (CLAUDE.md fix option 1); (c) append **both** to `$(CONTRACTS)`.
**Warning signs:** `make tier3` green while `./target/release/pv validate` on either phase
contract is red.

### Pitfall 5: Trusting the research docs' dependency table
**What goes wrong:** D-02's discussion-time evidence said "neither [aprender-train nor apr-cli]
depends on aprender-data." **Verified 2026-08-08: `apr-cli` DOES depend on it** — root
`Cargo.toml` line 208 aliases `alimentar = { path = "crates/aprender-data", package =
"aprender-data" }` and `crates/apr-cli/Cargo.toml` line 232 consumes it non-optionally (it powers
`apr data audit/split/balance`). `aprender-train` has no such edge; `aprender-data` has no
`aprender-core` edge.
**Consequence:** D-01 still stands — the decisive rationale is the *weight* of `aprender-data`
(arrow 57, parquet, zstd, lz4, s3/streaming/federated) flowing into `aprender-train` (Phase 3)
and future Lambda consumers, which the alias evidence does not change — but plans must not repeat
the stale two-edges claim, or a reviewer will "correct" the phase back toward STACK.md.
**Warning signs:** any plan text citing "apr-cli does not depend on aprender-data."

### Pitfall 6: Seen-set that quietly becomes the Cartesian product
**What goes wrong:** a `unique`-style dedup or the D-11 seen-set grows with pairs *drawn*, and a
budget near capacity plus rejection sampling turns into coupon-collector time or unbounded memory.
**How to avoid:** capacity is O(K) closed-form; **fail closed** when `budget > capacity` (typed
error, D-11). If the `unique` strategy ships at all (recommend it does not in v1 — see Open
Questions), draw it with Floyd's uniform k-subset algorithm over the global pair-index space +
unranking: exactly `budget` draws, O(budget) state, no rejection loop.
**Warning signs:** any `while !seen.insert(pair)` loop without a draw-count ceiling.

### Pitfall 7: Relocation breaks the hash-then-parse discipline or `revision_verified` honesty
**What goes wrong:** splitting `data_tweeteval.rs` across the crate boundary re-reads files or
re-serializes rows before hashing, so recorded SHA-256s describe bytes that never passed the
class-count contract; or the relocated API lets a local `--source` run claim a verified revision.
**How to avoid:** the crate's bytes-boundary function takes `&[u8]` and returns `(hash, rows)`
from the same buffer — the discipline moves *with* the seam. `FALSIFY-TWEET-EVAL-006` must keep
passing unchanged against the relocated code; D-06's tracked baseline makes the diff reviewable.
**Warning signs:** two `fs::read` calls for one source file anywhere in the CLI adapter; any
constructor of the source manifest taking `revision_verified` as a caller-supplied bool without
provenance.

### Pitfall 8: Publish-cascade breakage from the new crate
**What goes wrong:** `apr-cli` 0.63.0 is published; adding a path+version dependency on an
unpublished crate makes the next `make publish` fail (or worse, publishes apr-cli first and
yanks). Also, CB-510 class: a new crate directory with `include!()` or gitignore-shadowed files.
**How to avoid:** the new crate is publishable (no `publish = false`), enters the cascade before
`apr-cli`, and both `scripts/check_include_files.sh` and `scripts/check_package_includes.sh`
re-run after the workspace edit. Name availability on crates.io verified this session. Publishing
itself remains a human-approved release action (CLAUDE.md).
**Warning signs:** `cargo package -p apr-cli` failing on an unresolvable registry dep.

## Code Examples

### Measured reference behavior (pinned setfit 1.1.3 — run this session in the hash-locked venv)

```python
# Source: scripts/setfit_fixtures/.venv setfit==1.1.3, executed 2026-08-08
from setfit.sampler import ContrastiveDataset
# Docs' own worked layout: classes of 8, 4, 8  (docs claim 62 pos / 128 neg / 256 total)
ds = ContrastiveDataset([f"s{i}" for i in range(20)], [0]*8+[1]*4+[2]*8,
                        multilabel=False, sampling_strategy="oversampling")
# MEASURED: pos_pairs stored = 82  (62 + 20 SELF-PAIRS), neg = 128
# MEASURED: len_pos = len_neg = 128 -> total 256  (total matches docs; composition does not)
# MEASURED: both-orientation duplicates = 0  (triu enumeration -> one orientation only)
# Singleton layout [0,0,0,0,1]:
#   MEASURED: pos stored = 11 (incl. 5 self-pairs; the singleton class SELF-PAIRS), total len 22
#   doc-formula would give pos 6, neg 4, total 12  -> divergence class documented in Pitfall 2
# max_pairs=100: stored exactly 50 pos + 50 neg (max_pos_or_neg = max_pairs // 2)
# shuffle_combinations: np.random.RandomState(seed=42) — HARDCODED; trainer seed unused for identity
# np.triu_indices(n) materializes the FULL O(N^2) index array regardless of max_pairs
```

### Closed-form capacity and default budget (D-14, contracted as Aprender policy)

```rust
// Aprender semantics: self-pairs excluded (declared deviation clause 3).
// O(K); only enumeration is quadratic.
fn positive_capacity(class_sizes: &[u64]) -> u64 {
    class_sizes.iter().map(|&n| n * n.saturating_sub(1) / 2).sum()
}
fn negative_capacity(class_sizes: &[u64]) -> u64 {
    let total: u64 = class_sizes.iter().sum();
    let sq: u64 = class_sizes.iter().map(|&n| n * n).sum();
    (total * total - sq) / 2 // Σ_{j<k} n_j·n_k
}
fn default_epoch_budget(class_sizes: &[u64]) -> u64 {
    2 * positive_capacity(class_sizes).max(negative_capacity(class_sizes))
}
// Verified worked values (balanced 3-class): 8-shot -> 384; 64-shot -> 24,576.
// Balanced n>=2, 3 classes: negatives always dominate, so these equal the
// pinned reference's epoch length despite Finding F2.
```

### Canonical pair type (D-10/D-12 — structural impossibility)

```rust
/// Unordered, self-pair-free pair of selected-row ordinals. The ONLY constructor
/// canonicalizes; no public fields, no config can resurrect (x, x).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CanonicalPair { lo: SelectedId, hi: SelectedId }

impl CanonicalPair {
    pub fn new(a: SelectedId, b: SelectedId) -> Result<Self, ContrastiveDataError> {
        match a.cmp(&b) {
            Ordering::Less    => Ok(Self { lo: a, hi: b }),
            Ordering::Greater => Ok(Self { lo: b, hi: a }),
            Ordering::Equal   => Err(ContrastiveDataError::SelfPair { id: a }),
        }
    }
}
// Pair label derives from the endpoints' classes at emission (1.0 same / 0.0 diff —
// matches setfit sampler.py lines 96-100), never from caller input (PF-003).
```

### Normalized hash v1 (D-17 — contracted, versioned)

```rust
// Source: python replica of this exact pipeline reproduced the one real duplicate
// and found zero false groups on the pinned data (Finding F1).
pub const CONTENT_NORMALIZATION_VERSION: &str = "nfc-trim-ws-v1";

fn normalized_hash(input: &str) -> [u8; 32] {
    use unicode_normalization::UnicodeNormalization;
    let nfc: String = input.nfc().collect();
    let collapsed = nfc.split_whitespace().collect::<Vec<_>>().join(" "); // trim + collapse
    // NO casefold — deliberate (D-17): distinct short posts must not collide.
    sha2::Sha256::digest(collapsed.as_bytes()).into()
}
```

### Existing hash-from-parsed-bytes discipline to preserve across the seam

```rust
// Source: crates/apr-cli/src/commands/data_tweeteval.rs (load_canonical_dataset)
// Read each source file exactly ONCE; hash and parse THE SAME bytes. The crate's
// boundary keeps this shape: fn ingest(bytes: &[u8], decl: &SplitDeclaration)
//   -> Result<(SourceHash, Split<R>), ContrastiveDataError>
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| SetFit `num_iterations` pair sampling | `sampling_strategy` = oversampling (default) / undersampling / unique | setfit ≥ 1.0 (docs mark `num_iterations` deprecated) | The contracted target is the oversampling strategy; do not implement `num_iterations` `[CITED: huggingface.co/docs/setfit/en/conceptual_guides/sampling_strategies]` |
| Docs-described sampler (self-pairs excluded) | Pinned 1.1.3 implementation includes the diagonal, hardcoded internal seed 42, O(N²) enumeration under any cap | measured this session | The deviation statement and fixtures key off *measured* behavior (Findings F2, Pitfall 1) `[VERIFIED: executed pinned source]` |
| STACK.md: `rand_chacha`, sampler in `aprender-data` | Locked overrides D-20/D-21 and D-01/D-02 | CONTEXT (2026-08-08) | Research honored the overrides; corrected one piece of supporting evidence (Finding F3) |
| Phase 1: contracts could live outside tier reach | `$(CONTRACTS)` explicit list + tier3 `contract-validate` + measured-runtime comments | Phase 1 D-26 | Phase 2 must append its two contracts or the gates never run |

**Deprecated/outdated:**
- `num_iterations` sampling: deprecated upstream; out of scope.
- `contracts/rand-philox-v1.yaml`: referenced by `aprender-rand` doc comments but **not present
  in this repository** (searched the full tree) — see Open Question 4.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The `make publish` cascade can be ordered so `aprender-contrastive-data` publishes before `apr-cli`; the publish tooling handles a new workspace member without bespoke changes | Pitfall 8 / Finding F5 | Release-time failure; mitigated because publish is a human-approved action with the pre-release skill's gates `[ASSUMED]` |
| A2 | The restructured (pv-valid) shape for `tweet-eval-stance-benchmark-v1.yaml` — adding `proof_obligations` + `kani_harnesses` per the Phase 1 contract precedent — will be accepted by `pv validate` for this contract's content | Pitfall 4 | Falls back to CLAUDE.md fix options 2/3 (schema extension as its own ticket); validate early in Wave 0 `[ASSUMED — precedent verified, this specific contract not yet restructured]` |
| A3 | 64-bit multiply-shift bounded draws (non-uniformity < 2⁻⁴⁴) are acceptable as the *contracted* derivation; no contract requires exact-uniform bounded integers | Pattern 2 | If exact uniformity is demanded, switch to in-block rejection (4 lanes/block make failure probability negligible) — a localized change `[ASSUMED]` |
| A4 | Phase 3's trainer can consume a fixed per-epoch pair stream with offset-based resumption (epoch NOT in the RNG domain), matching pinned SetFit's fixed-per-epoch behavior | Open Question 3 | If Phase 3 wants epoch reshuffle, add `epoch` to the domain string as a versioned policy — counter-based design makes this a one-line, non-breaking extension `[ASSUMED]` |

All other factual claims in this document are `[VERIFIED]` (tool-checked this session) or
`[CITED]` (official docs), tagged inline.

## Open Questions (RESOLVED)

1. **Restructure timing for the pv-invalid contract (Finding F4)**
   - **RESOLVED** — adopted by plan **02-01**: the tweet-eval contract restructure to a
     pv-valid shape plus `$(CONTRACTS)` wiring is the first Phase 2 contract task,
     honoring D-06 verbatim (the 813 lines land as-is first).
   - What we know: D-06 says the 813 lines land "as-is first"; `pv validate` rejects the contract
     today; tier3 doesn't reach it because `$(CONTRACTS)` is explicit.
   - What's unclear: whether to fix the contract inside the D-06 PR (small, but violates "as-is")
     or as the first Phase 2 task (honors D-06 verbatim, leaves a known-red contract in-tree
     briefly).
   - Recommendation: honor D-06 verbatim; make "restructure tweet-eval contract to pv-valid shape
     + append both contracts to `$(CONTRACTS)`" the first contract task of Phase 2, before any
     `pv diff`-tracked growth.

2. **Do `unique` and `undersampling` ship in v1?** (explicit discretion item)
   - **RESOLVED** — adopted by plan **02-07**: v1 ships `oversampling` only; the capacity
     closed forms and `BudgetExceedsCapacity` typed error ship regardless (D-11), with
     strategy as a versioned one-variant enum so later additions are non-breaking.
   - What we know: D-14 contracts the `oversampling` default; D-11 pins `unique` semantics *if
     present*; no v1 consumer needs the other strategies; every shipped strategy needs fixtures,
     property tests, and contract clauses.
   - Recommendation: ship `oversampling` only. Ship the **capacity closed-form and the
     `BudgetExceedsCapacity` typed error regardless** (D-11's hard part, needed for the cap
     anyway). Represent strategy in the manifest as a versioned enum with one variant so adding
     `unique`/`undersampling` later is non-breaking.

3. **Pair order across epochs** (explicit discretion item; Phase 3 consumes)
   - **RESOLVED** — adopted by plan **02-07** under **Assumption A4**: one
     epoch-independent stream per (selection, seed, policy, budget), consumed via
     offsets. Freezing the 02-02 contract text does NOT wait on Phase 3: per A4, if
     Phase 3 wants epoch reshuffle, adding `epoch` to the RNG domain string is a
     versioned, non-breaking one-line policy extension, and the contract documents that
     extension path explicitly.
   - What we know: pinned SetFit reuses the identical pair sequence every epoch (dataset built
     once, cycled; internal seed hardcoded — verified). D-09's replay tuple has no epoch term.
   - Recommendation: v1 = one stream per `(selection, seed, policy, budget)`, epoch-independent,
     consumed via offsets (shardable/resumable per D-20). Document that epoch-varying order is a
     versioned policy extension. Phase 3 confirmation is NOT required before freezing (see RESOLVED line above — A4
     makes epoch reshuffle a non-breaking versioned extension).

4. **`contracts/rand-philox-v1.yaml` does not exist in this repo**
   - **RESOLVED** — adopted by plan **02-02**: the RNG derivation obligations are stated
     inline in `contrastive-pair-protocol-v1.yaml`; no dangling cross-reference is
     created (02-02 Task 1 forbids citing rand-philox-v1 in crate docs).
   - What we know: `aprender-rand` doc comments cite it; CONTEXT lists it as an existing contract
     to reference; a full-tree search finds nothing (it lives in the pre-monorepo trueno repo).
   - Recommendation: state the RNG derivation obligations directly inside
     `contrastive-pair-protocol-v1.yaml` (key derivation, counter mapping, purity property) and
     cite the Philox paper/implementation; do not create a dangling cross-reference. Vendoring
     the original contract is optional hygiene, not a Phase 2 requirement.

5. **CLI shape** (explicit discretion item)
   - **RESOLVED** — adopted by plan **02-09**: `apr data select` + `apr data pairs`
     under the existing namespace, dump as a flag per D-09, singleton-policy manifest
     encoding as recommended below.
   - Recommendation: two subcommands under the existing `apr data` namespace —
     `apr data select --data <dir> --shots <8|16|32|64> --seed <u64> [--output <dir>] [--json]`
     writing `selection-manifest.json`, and
     `apr data pairs --selection <manifest> --seed <u64> [--budget <n>] [--dump <path>] [--json]`
     printing the pair-manifest hash (dump as a flag per D-09's "explicit dump path", not a
     subcommand). Names follow the existing verb style (`Audit`, `Split`, `Balance`). Manifest
     encodes singleton policy as `{"singleton_policy": "negatives_only", "singleton_policy_version": 1}`.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | build/test | ✓ | rustc 1.93.0 (MSRV 1.91 per workspace) | — |
| `cargo-mutants` | D-26 mutation gate | ✓ | 25.3.1 | — |
| `uv` + hash-locked fixture env | D-23 fixture generation | ✓ | uv 0.9.5; `.venv` installed and functional (setfit 1.1.3 imported and executed this session) | — |
| `pv` (aprender-contracts-cli) | contract authoring/validation | ✓ | 0.63.0 prebuilt at `target/release/pv`; canonical invocation `cargo run --release -p aprender-contracts-cli --bin pv --` (`PV_BIN`, Makefile line 874) | rebuild via cargo |
| Network → raw.githubusercontent.com | DATA-01 opt-in network test, dev-time acquisition | ✓ (verified: pinned files downloaded) | — | `--source` local dir path (already implemented); CI stays offline (SAFE-02) |
| Network → crates.io API | publish-status checks | ✓ | — | — |
| `trybuild` | DATA-06 compile-fail tests | ✓ (workspace dev-dep of 2 crates) | 1.x | doc-comment `compile_fail` blocks |
| `proptest` | property tests | ✓ | 1.6 workspace (with root-profile debug-assertion workaround) | — |

**Missing dependencies with no fallback:** none.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`cargo test`), rustc 1.93.0, edition 2021 |
| Config file | Workspace `Cargo.toml` lints + `.clippy.toml`; new crate has none of its own beyond `Cargo.toml` |
| Quick run command | `cargo test -p aprender-contrastive-data` |
| Full suite command | `make tier2` (pre-commit) → `make tier3` (contracts + feature matrix + full workspace) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | Canonical acquisition, counts, hashes, provenance | unit + opt-in network | `cargo test -p apr-cli --lib data_tweeteval` | ⚠️ Wave 0 — code exists but **uncommitted** (D-06 PR is the gate) |
| DATA-02 | Typed failures incl. duplicate IDs, conflicting roles, cross-split content | unit + golden (real duplicate, Finding F1) | `cargo test -p aprender-contrastive-data dedup` | ❌ Wave 0/new |
| DATA-03 | Balanced selection ×4 shots ×10 seeds, stable ordered manifest | golden (manifest hashes) + property | `cargo test -p aprender-contrastive-data select` | ❌ new |
| DATA-04 | Pair semantics, canonical identity, singleton policy | property + Rust goldens + reference count fixtures | `cargo test -p aprender-contrastive-data pairs` | ❌ new |
| DATA-05 | `O(examples + budget)` under 10×-N; capacity fail-closed | capacity invariants + in-band materializing negative | `cargo test -p aprender-contrastive-data --test negative_materializing` | ❌ new |
| DATA-06 | Leakage not constructible; fail-closed at runtime boundary | trybuild compile-fail + boundary unit + in-band leaky negative | `cargo test -p aprender-contrastive-data --test negative_leaky` + trybuild target | ❌ new |

### Sampling Rate
- **Per task commit:** `cargo test -p aprender-contrastive-data` (new crate is small — seconds)
- **Per wave merge:** `make tier2` (must gain a line for the new crate, following the Phase 1
  measured-runtime comment pattern at Makefile lines 185–213)
- **Phase gate:** `make tier3` green, including `pv validate` over the **updated** `$(CONTRACTS)`
  list containing both phase contracts, before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] **Land the D-06 baseline PR** — `data_tweeteval.rs`, contract YAML, docs example, eval
  `F_avg` diff are all still uncommitted on this branch (verified via git status this session)
- [ ] `crates/aprender-contrastive-data/` — the entire crate (workspace member + workspace-dep
  entry + apr-cli dep)
- [ ] `contracts/contrastive-pair-protocol-v1.yaml` — authored pv-valid from the start
  (shape precedent: `setfit-encoder-conformance-v1.yaml`)
- [ ] Restructure `contracts/tweet-eval-stance-benchmark-v1.yaml` to pass `pv validate`
  (currently PROVABILITY-001 ×2) and append both contracts to Makefile `$(CONTRACTS)`
- [ ] `scripts/setfit_fixtures/generate_fixtures.py` — extend for pair-count reference fixtures
  (RNG-independent facts only, measured semantics per Pitfall 1) + `manifest.sha256` entries
- [ ] `make tier2` line + bytes-boundary build check target (Pattern 5)
- [ ] Re-run `scripts/check_include_files.sh` + `scripts/check_package_includes.sh` after the
  workspace edit (CB-510 discipline)

## Security Domain

`security_enforcement: true`, ASVS level 1 (from `.planning/config.json`).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface in this phase |
| V3 Session Management | no | — |
| V4 Access Control | partially | Typestate split roles + access ledger are the phase's access-control analogue for data (D-16/D-19) |
| V5 Input Validation | **yes** | `serde` strict deserialization (`deny_unknown_fields`) at the bytes→typed boundary; typed errors for every malformed-input class (DATA-02); UTF-8 validation before parse (existing pattern) |
| V6 Cryptography | yes | `sha2::Sha256` for all integrity hashes — never hand-rolled. **Note:** Philox is a *statistical* RNG, not a CSPRNG; it is used only for sampling determinism and must never be presented as cryptographic randomness |
| V10 Malicious Code / Supply Chain | yes | Zero new third-party deps (Package Legitimacy Audit); dependency-closure build check (Pattern 5) doubles as a supply-chain gate for the Lambda-bound crate |
| V12 File Handling | yes (CLI tier only) | Crate has no fs (enforced); CLI keeps `create_new` (no clobber without `--force`), rollback-on-partial-write, and root-anchored outputs — all existing verified behavior |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Dataset substitution / corrupted mirror | Tampering | Exact per-split class-count contracts + SHA-256 from the parsed bytes (existing, relocated intact — Pitfall 7) |
| Provenance spoofing (local dir attested as pinned revision) | Spoofing / Repudiation | `revision_verified` honesty flag + `FALSIFY-TWEET-EVAL-006` (must survive relocation) |
| Split leakage via mislabeled bytes from untrusted storage | Elevation (data) | Typestate + runtime boundary validation + access ledger — all three (D-16) |
| Resource exhaustion via pair explosion | DoS | Closed-form capacity + budget cap + `O(examples+budget)` invariant with in-band materializing negative (D-25) |
| Tweet text exfiltration into the repo / artifacts | Information Disclosure / licensing | No vendored text (existing license_notice discipline); duplicate rows referenced by hash only — this research doc itself records the real duplicate by SHA-256, not content |
| Network fetch downgrade / redirect | Tampering | HTTPS to pinned-commit raw URLs; 40-hex revision validation (existing `validate_revision`) |

## Sources

### Primary (HIGH confidence — verified by execution or direct read this session)
- `scripts/setfit_fixtures/.venv` — pinned `setfit==1.1.3` `sampler.py` read AND executed;
  measured pair counts, self-pair inclusion, singleton behavior, `max_pairs` semantics,
  hardcoded seed 42, O(N²) enumeration
- `https://raw.githubusercontent.com/cardiffnlp/tweeteval/4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66/datasets/stance/abortion/` —
  all six pinned source files downloaded; counts, class distributions, and duplicate analysis
  (exact + `nfc-trim-ws` + casefold) computed
- Workspace manifests: root `Cargo.toml` (members, workspace deps, `alimentar` alias line 208,
  MSRV 1.91, version 0.63.0), `crates/apr-cli/Cargo.toml` (alimentar line 232, ureq 2.10,
  unicode-normalization), `crates/aprender-train/Cargo.toml`, `crates/aprender-data/Cargo.toml`
  (lib name `alimentar`, arrow/parquet stack), `crates/aprender-rand/` (full source)
- `./target/release/pv` 0.63.0 — `validate` executed on both the in-flight tweet-eval contract
  (FAILS: PROVABILITY-001 ×2) and `setfit-encoder-conformance-v1.yaml` (passes)
- crates.io API — publish status for `aprender-rand`, `aprender-data`, `apr-cli` (all 0.63.0)
  and `aprender-contrastive-data` (unclaimed)
- `crates/apr-cli/src/commands/data_tweeteval.rs`, `contracts/tweet-eval-stance-benchmark-v1.yaml`,
  `crates/apr-cli/src/data_commands.rs`, `crates/apr-cli/src/dispatch_analysis.rs` — full reads
- `Makefile` — tier2/tier3 bodies, `$(CONTRACTS)` list (line 876), `PV_BIN` (line 874),
  `setfit-feature-matrix` precedent
- Phase 1 artifacts: `01-CONTEXT.md`, `scripts/setfit_fixtures/` (pyproject, README, uv.lock),
  `crates/aprender-core/src/setfit/` module tree, `#[contract]` usage in `autograd/ops/pooling.rs`
- Toolchain probes: rustc 1.93.0, cargo-mutants 25.3.1, uv 0.9.5, trybuild in-workspace

### Secondary (MEDIUM-HIGH — official docs)
- SetFit sampling strategies — `https://huggingface.co/docs/setfit/en/conceptual_guides/sampling_strategies`
  (fetched this session: pair definitions, strategy semantics, worked 62/128/256 example,
  documented self-pair/orientation exclusions — with the implementation divergence noted in F2)
- `.planning/research/PITFALLS.md` (PF-002/003/006/007/008 + Sources), `.planning/research/STACK.md`
  (§2, ownership table — read with the CONTEXT's recorded overrides), `.planning/codebase/TESTING.md`

### Tertiary (LOW)
- None — no unverified WebSearch-only claims were used.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — zero new dependencies; every crate version read from the workspace and cross-checked against crates.io
- Architecture: HIGH — patterns compose in-tree precedents (Phase 1 gates, typestate, fixture manifest, feature-matrix build check); the two locked overrides were honored and their evidence re-verified
- Reference semantics: HIGH — measured by executing the pinned reference, not read from docs (which the measurement contradicted)
- Pitfalls: HIGH — four of eight are new findings from this session's tool evidence, not speculation

**Research date:** 2026-08-08
**Valid until:** 2026-09-08 for workspace facts (30 days — stable monorepo conventions); the pinned dataset and pinned setfit measurements are permanent facts of those revisions
