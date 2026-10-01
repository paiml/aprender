# Phase 2: Deterministic Pair and Data Protocol - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-08
**Phase:** 2-Deterministic Pair and Data Protocol
**Areas discussed:** Manifests & surface, Pair policy & budget, Leakage enforcement, Determinism evidence

---

## Manifests & surface

### What gets written to disk — the selected-ID manifest, the pair manifest, both, or neither?

| Option | Description | Selected |
|--------|-------------|----------|
| Selection file, pairs replayed | Selected IDs become a real shared artifact (ordered IDs + semantic hash + provenance) because Phase 5 must hand the identical file to the 9B LoRA baseline; pairs regenerate from (selection hash, seed, policy, budget) with only the pair-manifest hash persisted, plus an explicit dump for audit and fixtures. | ✓ |
| Both materialized | Selection and pair manifests both on disk. Maximum auditability, pair storage still O(budget) so DATA-05 holds — at the cost of many pair files across 40 cells × epochs and a second artifact that can drift from the seed claiming to produce it. | |
| Both replayed from seed | Nothing materialized; persist (seed, config, split hashes) plus selection and pair hashes. Leanest storage, but Phase 5's cross-method "identical sampled IDs" becomes a hash comparison between two independent regenerations rather than one shared file. | |

**User's choice:** Selection file, pairs replayed
**Notes:** Driven by EVAL-02's requirement that both methods use an identical sampled-ID hash in every one of the 40 cells.

### Where does the selection + pair protocol live?

| Option | Description | Selected |
|--------|-------------|----------|
| aprender-core, setfit feature | Next to Phase 1's encoder behind the same gate; zero new dependency edges; deviates from the research ownership table. | |
| aprender-data, as researched | Follows `.planning/research/STACK.md`'s ownership table literally. Costs two new dependency edges into a large crate carrying s3/http/federated/streaming and a 238 KB generated_contracts.rs. | |
| New focused crate | Owns splits, selection, pairs, fingerprints and nothing else. Clean ownership, small dependency surface, easy to feature-gate. Costs one more workspace crate to publish, version, and keep inside the MSRV/feature matrix. | ✓ |

**User's choice:** Free text — *"Maybe we should build a crate for contrastive Siamese data, which can be useful in other similar algorithms of few shots."*
**Notes:** Generalized the option rather than just selecting it. The crate is framed as contrastive/Siamese **data construction** as a general capability, not a SetFit-private helper. Follow-up asked how much generality lands now and what to name it; user answered **(a) concrete, generically named types, name it `aprender-contrastive-data`** — trait extraction deferred until a second consumer proves the shape.

Context discovered during this question and reported to the user: `aprender-train` depends on only `aprender-core`; `apr-cli` depends on `aprender-core`, `aprender-contracts`, `aprender-mcp`, `aprender-train-*`; neither depends on `aprender-data`, and `aprender-data` does not depend on `aprender-core`.

### How do the dataset-ETL pieces split between apr-cli and the new crate?

| Option | Description | Selected |
|--------|-------------|----------|
| Core to crate, fetch stays CLI | Labeled-example schema, typed split roles, JSONL read/write, per-row content hashes, dataset fingerprint, cross-split duplicate detection move into the crate; TweetEval URLs/filenames/counts/labels, clap args and network fetch stay in apr-cli. Phase 3/5 consume typed splits without depending on CLI modules. | ✓ |
| Whole converter moves | TweetEval becomes a named dataset profile inside the crate, network fetch included. Most library-testable; costs the crate a `ureq` dependency and weakens its generic framing. | |
| Leave ETL where it is | data_tweeteval.rs untouched; only selection + pairs in the new crate. Smallest diff, but DATA-02's checks stay in a CLI crate aprender-train cannot reach. | |

**User's choice:** Core to crate, fetch stays CLI
**Notes:** Prompted by the observation that ROADMAP phrases every Phase 2 criterion as "A user can…", unlike Phase 1's "A developer can…" — so this phase does grow a CLI surface, and the question was where the seam falls.

### What happens to the 813 lines of uncommitted TweetEval work?

| Option | Description | Selected |
|--------|-------------|----------|
| Land as-is, then refactor | Commit data_tweeteval.rs, the contract, the eval F_avg changes and docs as their own PR first; Phase 2 then relocates the core as a reviewable diff against a tracked baseline. | ✓ |
| Fold into Phase 2 plan 1 | Leave uncommitted and make "adopt + relocate" the first plan. One coherent story, but the first Phase 2 commit mixes 813 lines of pre-existing work with new work. | |
| Treat as done, reference only | Commit it, declare DATA-01 satisfied, scope Phase 2 to DATA-02→06 and leave ETL in apr-cli. Contradicts the seam decision just made. | |

**User's choice:** Land as-is, then refactor
**Notes:** The existing work already carries `contracts/tweet-eval-stance-benchmark-v1.yaml` and seven passing falsification tests, which makes it a legitimate baseline.

---

## Pair policy & budget

### How does the sampler produce budget-many pairs without enumerating the pair space?

| Option | Description | Selected |
|--------|-------------|----------|
| Streaming draw + seen-set | Draw class then two distinct members (positive), or two distinct classes then one member each (negative). O(1) per draw, O(N) bucket state, canonicalized (min,max). Unique strategy keeps a budget-bounded seen-set and fails closed when the budget exceeds available unique pairs. | ✓ |
| Combinatorial unranking | Sample distinct ranks via a seeded pseudo-random permutation (Feistel) and unrank to (i,j). Uniqueness free and exact, replay a pure function of rank. Costs bit-fiddling math needing its own fixtures. | |
| Bucketed shuffle-and-walk | Deterministically shuffle within sorted buckets and walk index sequences. Simple, but ordering correlates pairs with bucket layout, creating coverage bias under the 10×-N scaling test. | |

**User's choice:** Streaming draw + seen-set
**Notes:** Answered alongside substantial strategic context (see below), which the user asked to have inform the remaining decisions. Summary of that context: the destination is aprender models exposed as **MCP tools** — an LLM agent drives training then calls the deployed model as a tool for exact symbolic/statistical work LLMs handle poorly, e.g. classifying social posts, rather than spending an LLM call per item. Delivery is via **PMCP** (Rust MCP SDK) and **pmcp.run**, i.e. Rust on AWS Lambda with S3/DynamoDB, callable by agents and business-process automation. SetFit is the first instance of a pattern; other families (time-series forecasting named explicitly) follow. Claude reflected back the specific Phase 2 implications (bytes not paths, replayable pairs, memory bound as a real cost, content hashes as identity, dependency-light crate) and explicitly held MCP servers, storage layers, and other model families out of scope as a future milestone.

### Should serverless-portability be a binding Phase 2 constraint on the crate's API?

| Option | Description | Selected |
|--------|-------------|----------|
| Bytes-in/bytes-out, enforced | No std::fs, no network, no path-shaped APIs in the public surface; apr-cli owns every filesystem adapter; enforced by contract obligation and build check. | ✓ |
| Bytes-first, paths as convenience | Byte-oriented core plus path helpers behind a `std-fs` feature. Pragmatic, but the convenience path tends to become the real path. | |
| Not a Phase 2 concern | Design for the CLI now, refactor when the MCP milestone starts. Costs a later breaking change to a crate Phases 3–5 already depend on. | |

**User's choice:** Bytes-in/bytes-out, enforced

### What happens when a class has exactly one selected example?

| Option | Description | Selected |
|--------|-------------|----------|
| Versioned policy, negatives-only default | `SingletonPolicy` enum recorded and version-tagged in the manifest; default NegativesOnly — no positives, still in negatives, still contributes its row to the head; self-pairs structurally impossible. | ✓ |
| Versioned policy, error default | Same enum, but default is a typed error requiring explicit opt-in. Maximally fail-closed; costs usability for genuine 1-shot work. | |
| Hard reject, no policy | Always a typed error, no configuration. Smallest surface; rules out 1-shot and would need a breaking change to support it later. | |

**User's choice:** Versioned policy, negatives-only default
**Notes:** Claude separated two degenerate cases first — a class with fewer examples than `shots_per_class` (unambiguous typed error at selection, DATA-02) versus a class with exactly one selected example (a legitimate 1-shot regime). Closes the STATE blocker for this phase.

### What is the default max_pairs_per_epoch?

| Option | Description | Selected |
|--------|-------------|----------|
| SetFit-equivalent count, capped | Default = the closed-form oversampling count SetFit would produce for these class sizes, strict 1:1, clamped by a configurable hard cap. 24 examples → 384 pairs; 192 → 24,576. | ✓ |
| Fixed multiple of N | e.g. 20 × example count — 480 at 8-shot, 3,840 at 64-shot. Predictable, but diverges sharply from SetFit at high shot counts. | |
| No default, required config | Mandatory explicit value, matching TRN-02's validate-before-training. Costs every caller a number and gives benchmark readers no canonical value. | |

**User's choice:** SetFit-equivalent count, capped
**Notes:** Enabled by the observation that SetFit's oversampling count is closed-form — positives `Σ C(n_k,2)`, negatives `Σ_{j<k} n_j·n_k`, balanced to `2·max(...)`, computable in O(K). Only *enumerating* is quadratic. This narrows the declared deviation to "sampled identities, capped above N".

---

## Leakage enforcement

### How is fail-closed leakage prevention actually enforced?

| Option | Description | Selected |
|--------|-------------|----------|
| Typed roles + runtime + ledger | Phantom-typed Split<Train>/Split<Validation>/Split<Test> so a library caller cannot express leakage; runtime hash/membership validation at the bytes→typed boundary because deserialized object-storage bytes are untrusted; access ledger records every split touched for Phase 5's selection-lock. | ✓ |
| Typed roles only | Compile-error guarantee, no runtime cost, no ledger. Guarantee evaporates at deserialization, where a mislabeled `source_split` becomes a Split<Train> the compiler accepts. | |
| Runtime validation only | Every ID and content hash checked at selection and pair time. Simpler generics; every check must be remembered at each call site — the failure mode PF-002 describes. | |

**User's choice:** Typed roles + runtime + ledger

### What content hashing does each row carry, and how aggressive is duplicate matching?

| Option | Description | Selected |
|--------|-------------|----------|
| Exact + conservative normalized | Exact SHA-256 over raw input bytes for identity/provenance, plus a normalized hash (NFC + trim + collapse internal whitespace, no casefolding) for cross-split duplicate detection; normalization contracted and versioned. | ✓ |
| Exact bytes only | One hash for both jobs. Matches PF-002's "identical text" literally, zero false positives; misses trailing-whitespace and Unicode-composition variants. | |
| Add fuzzy near-dup detection | Plus MinHash/simhash with a configurable threshold. Strongest leakage story; introduces a tunable threshold, false positives on short texts, and a similarity implementation to contract. | |

**User's choice:** Exact + conservative normalized
**Notes:** Claude separated the two jobs hiding in "content hash" — row identity/provenance (must be exact bytes) versus leakage detection (should tolerate variants) — since conflating them picks the wrong answer for one.

### What happens when a training row duplicates validation or test content?

| Option | Description | Selected |
|--------|-------------|----------|
| Exclude from pool, record it | Detect duplicate groups at prepare time and deterministically remove affected rows from the training selection pool, recording excluded IDs and reduced per-class pool size in the manifest; typed error only if the pool can no longer supply shots_per_class. | ✓ |
| Hard reject the dataset | Any cross-split duplicate is a typed error at prepare time. Maximally fail-closed; hands upstream data quality a veto over DATA-01 with no override. | |
| Record, fail at pair time | Prepare proceeds; error fires when a selected ID is implicated. Failure is seed-dependent, so cells die and Phase 5's completeness gate rejects the run for an unrelated reason. | |

**User's choice:** Exclude from pool, record it
**Notes:** Claude flagged that neither hard-fail option is safe — selection-time failure holes the 40-cell matrix, prepare-time failure blocks DATA-01 on upstream data quality — and stated explicitly that whether canonical TweetEval abortion-stance actually contains cross-split duplicates is **unverified**.

### How is the merged SetFit compatibility split blocked from model selection?

| Option | Description | Selected |
|--------|-------------|----------|
| Distinct type, no validation exists | Merged split deserializes as Split<CompatibilityTest>, distinct from Split<Test>; the compatibility profile emits no Split<Validation> at all, so a selection run cannot be constructed. Access ledger records profile identity for Phase 5's selection-lock. | ✓ |
| Runtime profile check | Manifest profile + source_splits trigger a typed error at every selection entry point. Simpler types; relies on remembering the check at each new entry point across Phases 3–5. | |
| Gate emission itself | Refuse to produce the profile without an explicit acknowledgement flag. Doesn't help once files exist — acknowledgement is at generation time, misuse at training time. | |

**User's choice:** Distinct type, no validation exists
**Notes:** Today's protection is a sentence in the manifest ("do not tune on this test split") — documentation, not a gate. Also noted that under `--profile setfit` there is no validation split at all, so selection has nothing to bind to.

---

## Determinism evidence

### Which RNG drives selection and pair sampling?

| Option | Description | Selected |
|--------|-------------|----------|
| aprender-rand, counter-based | key = hash(root_seed, domain), counter = ordinal; draw i is a pure function of its index, so thread-count independence is structural, domain separation falls out of the key, and the pair stream is shardable/resumable. Already in-workspace. | ✓ |
| rand_chacha, as researched | Follows STACK.md literally; same RNG family as the rest of the workspace. Sequential — resuming or sharding means replaying, and worker-count independence becomes an asserted property. | |
| Hybrid | ChaCha for selection, counter-based for pair streaming. Better-fitting tool per stage; costs two RNGs, two seed-derivation stories, two sets of fixtures in one small crate. | |

**User's choice:** aprender-rand, counter-based
**Notes:** Claude surfaced the tension first: STACK.md says "do not add a second RNG crate", but PF-006's falsification test #2 requires identical results across 1 and N workers, which a stateful stream defends by discipline rather than construction. `aprender-rand` is already in-workspace, so it is not a new dependency.

### What evidence proves the sampler correct?

| Option | Description | Selected |
|--------|-------------|----------|
| Invariant parity + Rust goldens | SetFit 1.1.3 fixtures for RNG-independent facts only (counts, balance, class correctness, self-pair/orientation exclusion, imbalanced and singleton cases); pair identities get Rust golden fixtures under a SHA-256 manifest; property tests between. | ✓ |
| Properties only | Proptest every invariant, no fixtures, no Python. Fast and offline, but a silent change to seed derivation still satisfies every property. | |
| Full SetFit identity parity | Reproduce SetFit's enumerate-then-shuffle order exactly. Requires materializing the pair space (DATA-05 forbids) or reimplementing Python's RNG. | |

**User's choice:** Invariant parity + Rust goldens
**Notes:** Claude flagged first that counter-based streaming makes pair identities structurally unable to match SetFit's Python RNG, partially limiting PF-003's falsification test #2 — better decided deliberately than discovered mid-implementation.

### How is the Phase 2 gate expressed as contracts?

| Option | Description | Selected |
|--------|-------------|----------|
| Split: generic + dataset | New `contrastive-pair-protocol-v1.yaml` owns the dataset-agnostic surface (DATA-03/04/05/06 + generic half of 02); the existing tweet-eval contract grows the dataset-specific half with a pv diff semver bump. | ✓ |
| One new contract | Single `setfit-data-protocol-v1.yaml` owning all six, referencing tweet-eval without editing it — Phase 1's D-23 pattern. One coherent gate, but generic obligations end up in a SetFit-named contract. | |
| Extend tweet-eval only | Grow the existing contract to cover DATA-01→06. One file, one gate; binds the generic sampler's obligations to one dataset forever. | |

**User's choice:** Split: generic + dataset

### What proves the memory bound and the leakage gate aren't theater?

| Option | Description | Selected |
|--------|-------------|----------|
| Capacity invariants + negative variants | Structurally bounded retained state, capacity-invariant assertions plus the 10×-N scaling property, and two in-band negative variants that must fail their gates in every cargo test: a leaky sampler and a materializing sampler. cargo-mutants scoped to the crate. | ✓ |
| Real allocation measurement | Counting GlobalAlloc measures actual bytes across the sweep. Strongest evidence; invasive, interacts badly with parallel test execution, measures the whole process. | |
| Criterion scaling benchmark | Criterion sweep asserting flat time and peak memory. Measures what users care about; flaky in CI, so it ends up advisory — and advisory gates stop being run. | |

**User's choice:** Capacity invariants + negative variants
**Notes:** Framed as Phase 1's D-24 discipline applied here: a sampler self-reporting its retained-state size is exactly as trustworthy as a loss self-reporting that it decreased.

---

## Claude's Discretion

The user delegated no decision with an explicit "you decide". The following were surfaced as
remaining gray areas and consciously left to research and planning as implementation detail when
the user selected "I'm ready for context":

- CLI command naming and shape for few-shot selection, and whether the pair dump is a subcommand or a flag
- The selected-ID manifest's field schema and file name
- Whether the `unique` and `undersampling` strategies ship in v1 beyond the `oversampling` default
- How pair batches are ordered and reshuffled across epochs (Phase 3 consumes this)
- Whether `aprender-contrastive-data` is published to crates.io this milestone, and the MSRV / feature-matrix consequences of adding a workspace crate mid-milestone
- The precise `SingletonPolicy` version-tag encoding in the manifest

## Deferred Ideas

- MCP server exposure of trained aprender models (PMCP + pmcp.run; Rust on Lambda with S3/DynamoDB) — future milestone; `crates/aprender-mcp/` already exists in-tree
- The same MCP pattern for other model families, e.g. time-series forecasting — future milestone
- Triplet and N-way-K-shot episode samplers in `aprender-contrastive-data` — v2 (EXT-02)
- Alternative contrastive objectives (InfoNCE, SupCon, CoSENT, triplet) — v2 (EXT-02)
- Fuzzy near-duplicate leakage detection (MinHash/simhash) — considered and rejected for v1
- Persistent embedding or token caches — out of scope for v1 (CACHE-01)
