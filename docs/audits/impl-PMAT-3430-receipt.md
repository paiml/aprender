# IMPL receipt — PMAT-3430 — PP-QUANT-001 M1 (plan phase only)

Verdict: **PARTIAL(escalate)** — plan quorum is not 3/3 after four rounds; the operator rules. No code was written. No v5.

## Identity
- ticket PMAT-3430 (#3430), kind=code, branch `PMAT-3430-one-ggml-type`, orchestrator model fable-5-1 admitted by `model-gate.sh` (`orch:fable`, `orch-basis:M>=3` — from main's fragment, landed by #3448; not written by this session).
- First launch of this session was refused at Phase 0 (`kind-gate` exit 2: ticket not filed on main; `model-gate` exit 1 against the then-unlabelled entry). Work started only after the entry carrying the basis existed on `origin/PMAT-3427-roadmap-entries`; the workaround commit that pulled it was dropped by rebase once #3448 merged. `git diff origin/main...HEAD -- docs/roadmaps` is empty.
- Harness findings: `pmat hooks install --strict --force` fails in a linked worktree (`Error: Not a directory (os error 20)`); the edit hook keys `active-ticket` on the session cwd's repo key, not the edited file's, so one was written by hand at the path the hook named; `estimate.sh aprender 5` exits 2 (47 rows, none pooled) so `K̂=24` is the issue's borrowed `[A]` figure entered as `first-run[U]`, the only basis `goal.sh` admits; `receipt-lint.sh` has no `--kind delegate`.

## Dispatch ledger
| round | description prefix | lane | width | route | agy conversations | delegate turns hit? |
|---|---|---|---|---|---|---|
| v1 | PMAT-3430/ph0.delegate | quorum/grillme | 3 | route=agy-plan w=1.00 basis=absent effort=1[U] | (receipt not emitted — delegate hit 30 turns; lanes + reduce-artifact read from disk) | yes |
| v2 | PMAT-3430/ph0.delegate2 | quorum/grillme | 3 | same | 5588db06…, 909d9ee2…, 99f05b71… | no |
| v3 | PMAT-3430/ph0.delegate3 | quorum/grillme | 3 | same | d0d7bffd…, 9ec33c0a…, 8ba37edd… | no |
| v4 | PMAT-3430/ph0.delegate4 | quorum/grillme | 3 | same | 2efab8bf…, 6efcbd2f…, f42ccf0d… | no |

Slots: 1 Claude subagent live at any instant (slots=3), 0 resumes, 0 hook denials, 0 stalls. Review lanes wrote rustc probes into their sandboxed clones in v1, v3, v4 (KEPT; nothing reached the checkout).

```
PASS transcript-gate: attempted=0 denied=0 stalled=0 running_peak=0 slots=3 — 0 subagents ran in /home/noah/.claude/projects/-home-noah-src-aprender-worktrees-PMAT-3430/a26ca3bb-cffd-4f6e-b6fd-358f489cf451 (rule=pid-file (/run/user/1000/paiml-implement/pid-2717702); vacuous but honest — say so in the receipt)
```

## PMAT-3430 plan quorum — history, v1–v4. **Not 3/3. No v5 written; no code written.**

Plan: `docs/audits/impl-PMAT-3430-plan.md` on branch `PMAT-3430-one-ggml-type` (draft PR linked below). Lanes: agy `grillme`, review-only, none in the author's model family (author `fable-5-1`). Every objection below was re-executed by the orchestrator before it was answered; two lane claims were refuted on re-run (lane 1 v1: "assoc-const alias works in patterns" — rustc E0308; lane 1 v1: "Q8_1 = 40, C struct has padding" — upstream `static_assert` says 36).

### 1. Grid

| lane (model) | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| 1 gemini-3.1-pro-high | FAIL | FAIL | FAIL | FAIL † |
| 2 gemini-3.8-flash-high | FAIL | FAIL † | FAIL | FAIL † |
| 3 gemini-3.7-flash-high | FAIL | PASS | PASS | PASS † |
| **PASS count** | **0/3** | **1/3** | **1/3** | **1/3** |
| reducer | partial | partial | clean | partial |

† lane exited 3, `LANE ISOLATION VIOLATED — a shared ref changed`: the plan worktree shares a git dir with the main checkout, and other sessions moved refs during the lane window (v2: `refs/remotes/origin/main`; v4: `refs/heads/PMAT-3445-t0-milestone-gate`, `refs/heads/PMAT-3451-ont-6-verdict-lattice`, a merge-queue ref). Each lane's own clone was unaffected, but under the rail those verdicts are unclean — **v3 is the only round with three clean lanes, and it was 1/3.** v1's `partial` was 2 stderr lines per lane beyond workspace narration.

### 2. Every REJECT / AMEND reason, verbatim from the lane's `findings[].claim`, one tag each

| round | lane | tag | at | reason (verbatim) |
|---|---|---|---|---|
| v1 | 1 | Q1 | `crates/aprender-compute/src/inference/model.rs:712` | Q1-c: Admission function fails to protect cross-crate passing. If serve admits IQ2XS and passes a TensorInfo to compute, it silently hits the wildcard and widens to returning zeros. |
| v1 | 1 | Q1 | `docs/audits/impl-PMAT-3430-plan.md:58` | Q1-d: gguf-py is authoritative for bytes-on-disk in a GGUF file. The TRAITS falsifier must compare against gguf-py (40), not the C struct sizeof (36). |
| v1 | 1 | Table E resolutions | `crates/aprender-serve/src/paged_kv/mod_compute_prefix.rs:254` | TABLE E: KvQuantType variants (FP32, Q8, Q4) are exactly ggml block formats. It meets the merge rule. |
| v1 | 1 | plan mechanics | `crates/aprender-core/src/format/tensors_tests_workflow.rs:82` | PHASES (Phase 3): Scope path misses core files using GgmlType. |
| v1 | 1 | plan mechanics | `crates/aprender-serve/src/apr/mod.rs:324` | PHASES (Phase 4): Scope path misses serve files using GgmlQuantType. |
| v1 | 2 | Q1 | `docs/audits/impl-PMAT-3430-plan.md:56` | compute's Bf16 becomes an associated const alias pub const Bf16: Self = Self::BF16 inside the quant crate — usable in expressions and patterns — so M1 touches no dispatch arm. |
| v1 | 2 | Q1 | `docs/audits/impl-PMAT-3430-plan.md:57` | M1 is behaviour-preserving: no crate accepts an id it refused before. Going from 12/15/16 to 35 variants would silently widen every match that ends in _ =>. So each crate keeps a thin, named admission function (supported(GgmlType) -> bool, one each in compute, core and serve) that reproduces its former id set exactly, and its parser calls it. |
| v1 | 2 | Q1 | `docs/audits/impl-PMAT-3430-plan.md:46` | pub const fn from_id(id: u32) -> Result<Self, GgmlTypeError>; ... serve's as_str/from_str_lossy/as_byte/Display move into the quant crate |
| v1 | 2 | Q1 | `docs/audits/impl-PMAT-3430-plan.md:49` | pub const fn tensor_bytes(self, n_elements: usize) -> Option<usize>; // checked: None on overflow or n % blck_size != 0 |
| v1 | 2 | plan mechanics | `docs/audits/impl-PMAT-3430-plan.md:101` | Phase 4 A_i: cargo test -p aprender-serve --lib gguf::types + bash scripts/check_one_ggml_type_enum.sh exits 0 |
| v1 | 2 | plan mechanics | `docs/audits/impl-PMAT-3430-plan.md:101` | Phase 4 scope_paths: crates/aprender-serve/src/gguf/, the #3405 file |
| v1 | 2 | Table E resolutions | `docs/audits/impl-PMAT-3430-plan.md:76` | Table E file paths: apr-cli quantize.rs:27, apr-cli quantize_flag_parity.rs:49, aprender-train quant_type.rs:5, aprender-core converter_types_expectations.rs:149, aprender-cgp profilers/quant.rs:9, aprender-serve paged_kv:254 |
| v1 | 2 | Q1 | `docs/audits/impl-PMAT-3430-plan.md:31` | The ticket's falsifier says 'equals gguf-py for every shared id', which would bake in 40. Phase 2 must read both upstream files at 3173a5647 and the quorum is asked to rule (§2, Q1-d). |
| v1 | 3 | Q1 | `docs/audits/impl-PMAT-3430-plan.md:56` | Associated const alias pub const Bf16 triggers non_upper_case_globals compiler warning/error and does not work with glob imports (use GgmlType::*). |
| v1 | 3 | Q1 | `crates/aprender-serve/src/gguf/tests/loader_tests_apr.rs:311` | GgmlQuantType::from_id changing from Option<Self> to Result<Self, GgmlTypeError> breaks existing call sites such as .is_some() in loader_tests_apr.rs:311. |
| v1 | 3 | Q1 | `crates/aprender-serve/src/apr/mod.rs:324` | Non-GGUF callers of from_id, specifically TensorEntry::from_binary in apr/mod.rs:324, will silently admit new variants when from_id expands to 35 variants without going through the GGUF parser admission function. |
| v1 | 3 | plan mechanics | `docs/audits/impl-PMAT-3430-plan.md:99` | Phase 2 acceptance command `cargo test -p aprender-quant --lib ggml_type` does not run integration tests in crates/aprender-quant/tests/ (declared in Phase 1 scope_paths). |
| v1 | 3 | plan mechanics | `docs/audits/impl-PMAT-3430-plan.md:101` | Phase 4 acceptance command `cargo test -p aprender-serve --lib gguf::types` runs only 10 unit tests in types.rs and does not execute qwen3_moe_load.rs (#3405 QTYPE_LABELS removal) or loader_tests_apr.rs. |
| v2 | 1 | Q1 | `docs/audits/impl-PMAT-3430-plan.md:59` | The plan claims `core: format/gguf/reader.rs` is an id-to-enum boundary and will get an admission function. |
| v2 | 1 | Q1 | `crates/aprender-core/src/format/gguf/types.rs:361` | The plan fails to account for an existing exhaustive match on `GgmlType` in core that will become non-exhaustive. |
| v2 | 2 | Q1 | `crates/aprender-core/src/format/gguf/types.rs:361` | The plan assumes all matches on GgmlType across the tree have wildcard arms (_ =>) and only accounts for wildcard disposition. In reality, crates/aprender-core/src/format/gguf/types.rs:361 (GgufTensor::byte_size) is an EXHAUSTIVE match over core's 12 variants with NO wildcard arm. Re-exporting trueno_quant::GgmlType (35 variants) causes rustc compilation error E0004 (non-exhaustive patterns). |
| v2 | 2 | Q1 | `docs/audits/impl-PMAT-3430-plan.md:45` | The Section 2 leaf API code definition (lines 45-52) omits existing inherent methods (as_byte, as_str, from_str_lossy, block_bytes, block_size) and trait impl Display, causing existing callers across aprender-serve and aprender-compute to fail compilation if implemented from the code block as written. |
| v2 | 2 | Q1 | `crates/aprender-core/src/format/gguf/reader.rs:325` | Section 2 Q1-c claims 'core: format/gguf/reader.rs' is an id-to-enum boundary where u32 becomes GgmlType and prescribes an admission function admitted_from_id there. In reality, format/gguf/reader.rs and reader_parsing.rs store raw dtype: u32 in GgufTensorMeta and never convert u32 to GgmlType. Core has no id-to-enum boundary. |
| v2 | 2 | Q1 | `crates/aprender-serve/src/gguf/qwen3_moe_load.rs:107` | Section 2 Q1-f states '#3405\'s private QTYPE_LABELS (13 rows, crates/aprender-serve/src/gguf/qwen3_moe_load.rs:90) is replaced by TRAITS[id].name in this ticket'. In qwen3_moe_load.rs:107, qtype_label(qtype: u32) accepts arbitrary u32. Direct array indexing TRAITS[id] will panic with out-of-bounds on any qtype >= 43. |
| v3 | 2 | Q1 | `docs/audits/impl-PMAT-3430-plan.md:45` | Section 2 code block claims to be the complete leaf API, but omits #[must_use] on methods returning values. Under aprender-quant's Cargo.toml lints (pedantic = warn), cargo clippy -p aprender-quant --lib -- -D warnings fails with clippy::must_use_candidate on every method. |
| v3 | 2 | Q1 | `docs/audits/impl-PMAT-3430-plan.md:40` | Auxiliary types GgmlFamily, QuantTraits, and GgmlTypeError in Section 2 lack derive attributes. Without #[derive(Debug, Clone, Copy, PartialEq, Eq)], Phase 2 test assertions like assert_eq!(GgmlType::try_from_id(4), Err(GgmlTypeError::Removed { .. })) fail compilation, and GgmlTypeError lacks Display/Error trait implementations. |
| v3 | 2 | Q1 | `docs/audits/impl-PMAT-3430-plan.md:59` | checked_tensor_bytes is declared as pub const fn, but block_size(&self) and block_bytes(&self) are declared as non-const pub fn. In Rust, a const fn cannot call non-const methods, causing E0015 if checked_tensor_bytes delegates to them. |
| v3 | 2 | plan mechanics | `docs/audits/impl-PMAT-3430-plan.md:69` | Phase 1 characterization tests cannot call compute's private from_u32 and pass unchanged in Phase 3, because in Phase 3 compute re-exports trueno_quant::GgmlType (which lacks from_u32) and replaces from_u32 with admitted_from_id, causing E0599 in Phase 3. |
| v3 | 2 | plan mechanics | `docs/audits/impl-PMAT-3430-plan.md:110` | Phase 1 scope_paths specifies crates/aprender-serve/src/gguf/tests/ and crates/aprender-compute/src/inference/, but compute's from_u32 in gguf.rs and serve's apr_qtype_to_dtype / apr_dtype_to_byte in dtype.rs are private (fn). External test files in those directories cannot call them without editing gguf.rs and dtype.rs to make them pub(crate), which is not explicit in Phase 1 scope. |
| v3 | 1 | Q1 | (summary; this lane's findings carried no fix) | First, fixing the Q4_1 byte_size calculation (18->20) within this refactoring violates the behavior-preserving invariant of M1 and must be landed as a separate PR first. |
| v3 | 1 | Q1 | (summary) | Second, moving the enum to `aprender-quant` without doc comments will fail the crate's existing `#![warn(missing_docs)]` lint, though it does pass `non_camel_case_types`. |
| v4 | 1 | plan mechanics | `crates/aprender-serve/src/apr/mod.rs:324` | serve's boundary functions (gguf/dtype.rs, infer/mod.rs:39, apr/mod.rs:324, apr/special_tokens.rs:210, apr/dequant.rs:235) keep their names and signatures |
| v4 | 1 | plan mechanics | `Makefile:1129` | Phase 1 scope_paths cover the creation and wiring of contracts/ggml-type-v1.yaml |
| v4 | 2 | plan mechanics | `crates/aprender-serve/src/apr/mod.rs:324` | apr/mod.rs:324 is a boundary function that keeps its name and signature and can be called directly by Phase 1 characterization tests |
| v4 | 2 | plan mechanics | `crates/aprender-serve/src/infer/mod.rs:38` | infer/mod.rs:39 is a boundary function that keeps its name and signature |
| v4 | 2 | plan mechanics | `contracts/census.json:5` | Adding contracts/ggml-type-v1.yaml requires regenerating contracts/census.json and synchronizing README.md, but both are omitted from scope_paths |
| v4 | 2 | plan mechanics | `README.md:44` | README.md contract count is checked for exact equality by scripts/readme_sync.sh --check during make contracts, which fails upon adding a new contract unless updated |
| v4 | 2 | plan mechanics | `ci/explicit-test-commands.d:1` | crates/aprender-quant/tests/ integration tests are not executed in CI unless an explicit command fragment is added under ci/explicit-test-commands.d/ |
| v4 | 2 | plan mechanics | `.github/workflows/ci.yml:938` | The plan asserts in Phase 5 and line 122 that the guard check_one_ggml_type_enum.sh must be wired into a CI selection line in .github, but no such selection line exists (ci.yml dispatches via scripts/guard_tree.sh --no-cargo) |

Counts by tag: Q1 (enum + TRAITS design) 21 · plan mechanics 16 · Table E resolutions 2 · **home crate 0 · X1 ordering 0**. `aprender-quant` as home was called sound by every lane in every round; no lane contradicted X1; Table E drew one flip (`KvQuantType`, lane 1, v1 only) and one path-format note.

### 3. Objections that repeated across rounds

No objection came back in the same words — each round's specific was re-run, answered, and not raised again. But three objection **classes** survived two or more rounds, each time with a new instance the previous revision had not found:

| class | tag | rounds | instances |
|---|---|---|---|
| **A. The §2 leaf API does not match the surface existing call sites use** | Q1 | v1, v2, v3 | v1: `from_id → Result` and `tensor_bytes → Option` break callers · v2: block omits `as_byte as_str from_str_lossy block_bytes block_size Display` · v3: aux types lack derives, `const fn` calls non-const, `must_use`/`missing_docs` under the crate's lint table |
| **B. The list of id→enum / name→enum admission boundaries is wrong** | Q1 | v1, v2, v4 | v1: gated the GGUF parser only, `apr/mod.rs:324` + 5 more bypass it · v2: listed core `reader.rs`, which keeps `u32` raw — core has no boundary · v4: `apr/mod.rs:324` is an inline match inside `TensorEntry::from_binary`, not a callable "boundary function that keeps its name" |
| **C. scope_paths / acceptance commands miss files or cannot fail** | plan mechanics | v1, v3, v4 | v1: Phase 3/4 scopes miss core `format/` siblings, serve `apr/` `infer/`; `gguf::types` filter exercises 0 sites · v3: Phase 1 scope omits serve `gguf/dtype.rs`; tests on private `from_u32` break in Phase 3 · v4: `contracts/census.json`, `README.md:44` (1791→1792), `ci/explicit-test-commands.d/`, Makefile `CONTRACTS` are outside every phase's scope; "CI selection line" does not exist (guards auto-dispatch via `guard_tree.sh --no-cargo`) |

Read against the operator's rule: **A and B are tagged Q1 and each survived three rounds** — by that rule they are design findings, not wording. What they have in common: the plan keeps trying to make M1 *source-compatible and behaviour-preserving at every existing call site* while replacing three enums with one 35-variant enum, and each round finds one more site where that promise needs another special case (an alias, an `Option`-returning twin, a surviving free function, a per-crate `ADMITTED` set, an inline match that is not a function). The lanes never disputed the target (one enum, `TRAITS[43]`, in `aprender-quant`); they dispute that the compatibility shim can be specified completely in advance. C is the same thing seen from the phase table.

Found on the orchestrator's re-runs, not by any lane — both stand regardless of the ruling:
- core `GgufTensor::byte_size` (`crates/aprender-core/src/format/gguf/types.rs:365`) sizes `Q4_1` at 18 bytes/block; upstream is 20. Latent export-size defect, to be filed and fixed on its own.
- Q8_1: ggml C `static_assert` = 36, gguf-py `GGML_QUANT_SIZES` = 40 at llama.cpp `3173a56471c1753650cd806694145ffd6dcace67`. The ticket's falsifier ("equals gguf-py for every shared id") cannot be met as worded.

Decision requested: rule on A/B — e.g. accept a compatibility shim specified by characterization tests rather than by enumeration, or drop source-compatibility from M1 and let the compiler enumerate the sites. Andon stands: 3/3 by 2026-09-18T12:00Z or the ticket is pulled. No code before 3/3.

