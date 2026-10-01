---
phase: 04-apr-artifact-and-production-parity
plan: 08
subsystem: serving
tags: [setfit, serve, http, classify, readiness, ops-05, ops-03, d-09, d-10, review-m4]

# Dependency graph
requires:
  - phase: 04-04
    provides: "VerifiedSetFitModel::classify + ClassifyRequestDocument + ClassifyResponse — the one classify path and the two types this route transports"
  - phase: 04-03
    provides: "load_setfit_apr + MODEL_TYPE_TAG + write_setfit_apr — the loader ladder and the tag startup routes on"
  - phase: 04-06
    provides: "setfit_io::read_setfit_apr_file_bounded — the bounded artifact door startup reads through, and the apr-cli setfit feature line this plan appends to"
provides:
  - "`POST /v1/classify` on aprender-serve — installed whenever the `setfit` feature is compiled in, 503 (never 404) with no model"
  - "aprender-serve `setfit` feature: dep:aprender + aprender/setfit + server"
  - "`AppState::default()` — the empty state, and `with_setfit_model` / `setfit_model` / `has_setfit_model`"
  - "HealthResponse.classifier_artifact_sha256 + classifier_verified — readiness reports the loaded artifact (OPS-05)"
  - "`apr serve <setfit.apr>` — one detection point inside the existing APR arm, bounded read, full ladder, shared router (D-10)"
affects: [04-09, 04-10, 04-11]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A route's EXISTENCE and a route's SERVICEABILITY are different questions; conflating them turns a missing model into a 404 that no client can tell apart from a missing build"
    - "A transport bound (body bytes) and a work bound (batch size) do not subsume each other — proven by deleting the first and watching a single 1 MiB text be served 200 OK"
    - "A cheap tag read must answer ONE question and return None on every failure, or it silently becomes a second validator with its own opinion about what is loadable"
    - "Absence beats null for a field that only exists on one class of build: it keeps every other build's wire body byte-identical"

key-files:
  created:
    - crates/aprender-serve/src/api/setfit_handlers.rs
  modified:
    - crates/aprender-serve/Cargo.toml
    - crates/aprender-serve/src/api/mod.rs
    - crates/aprender-serve/src/api/router.rs
    - crates/aprender-serve/src/api/types.rs
    - crates/aprender-serve/src/api/mod_app_state_new.rs
    - crates/aprender-serve/src/api/mod_app_state_gpu.rs
    - crates/apr-cli/Cargo.toml
    - crates/apr-cli/src/commands/serve/handlers.rs

key-decisions:
  - "`/v1/classify` is installed OUTSIDE the `config.openai_api` group as well as outside any slot condition: it is an aprender-native endpoint that merely shares the `/v1` prefix, and hanging it off the OpenAI toggle would reintroduce M4's 404 ambiguity through a second door"
  - "`AppState::model_loaded()` was extended rather than patching `build_health_response`: `model_loaded()` is the ONE place the question is answered, and a public accessor that reported `false` for a state holding a verified classifier would be a landmine for every other caller"
  - "The two readiness fields are `Option` with `skip_serializing_if`, so a build with no classifier emits a health body byte-identical to the pre-existing one — CRUX-C-34 is shared with vLLM/llama.cpp-parity consumers"
  - "`classifier_verified` is true-by-type, and the doc comment says so: it means 'the thing in the slot is of the verified kind', NOT 'a verification was re-run' — a field that can only be true is worth shipping only if it cannot be misread"
  - "The startup tag read returns `None` on EVERY failure rather than an error, so a broken APR container keeps the pre-existing path's diagnosis instead of getting one from a function not qualified to make it"

requirements-completed: []

# Metrics
duration: ~2h40m
completed: 2026-08-15
---

# Phase 04 Plan 08: `POST /v1/classify` and `apr serve <setfit.apr>` — Summary

**The SetFit HTTP surface ships, it is transport and nothing else, and its two
load-bearing behaviours were each shown able to fail before being claimed: making
the route slot-conditional produces the 404 review finding M4 named, and deleting
the body-limit layer serves a 1 MiB text with 200 OK.**

## Task Commits

| Task | Deliverable | Commit |
| ---- | ----------- | ------ |
| 1 | serve `setfit` feature, `AppState` slot + `Default`, `setfit_handlers.rs`, unconditional route, readiness fields | `82416fcc1` |
| 2 | `apr serve` tag detection + bounded read + ladder + shared router, `realizar?/setfit`, 6 tests | `05012923b` |
| 3 | In-process oneshot suite on the real router, 10 tests, both guards falsified | `64fe1435b` |

Every commit builds alone with the feature on and off.

## What shipped, exactly

### The route

```
POST /v1/classify
```

Installed by `create_router_with_config` under `#[cfg(feature = "setfit")]` with
**no slot condition and no `config.openai_api` condition**, carrying
`axum::extract::DefaultBodyLimit::max(classify_body_limit_bytes())` — 1 048 576,
read from core's `MAX_REQUEST_BODY_BYTES`, never spelled as a literal.

### The observed status codes

| Case | Status | Body |
| ---- | ------ | ---- |
| valid batch | **200** | core's `ClassifyResponse` |
| no model in the slot | **503** | transport `ErrorResponse`, "no SetFit model is loaded…" |
| `texts: []` | **400** | core's `ClassifyError::EmptyInput` rendering |
| 257 texts | **400** | core's `BatchTooLarge` rendering, naming 257 and 256 |
| body > 1 MiB | **413** | axum's payload-too-large |
| `ClassifyError::{EncodeFailed, HeadFailed, …}` | 500 | the error's own rendering |

**The missing-model case is 503 and is asserted NOT to be 404.** That assertion is
the plan's review-M4 pin and it is the one that was falsified below.

### The readiness fields, as shipped

```jsonc
// GET /health/ready, with a classifier resident -> 200
{
  "status": "ok",
  "model_loaded": true,
  "classifier_artifact_sha256": "<64 hex chars, read off the loaded model>",
  "classifier_verified": true,
  ...
}
// with nothing resident -> 503, and BOTH keys are ABSENT
```

`Option<String>` / `Option<bool>` with `#[serde(default, skip_serializing_if = "Option::is_none")]`.
The absence is deliberate: a build with no classifier emits the health body it
emitted before this plan, byte for byte.

### The apr-cli feature line, final

```toml
setfit = ["training", "aprender/setfit", "entrenar?/setfit", "realizar?/setfit"]
```

`crates/apr-cli/Cargo.toml:115`. The comment 04-06 left ("04-08 appends
`realizar?/setfit`") is replaced by one recording what the weak form buys: it turns
on aprender-serve's classify surface without naming the optional dependency twice
and without dragging axum into a `--features setfit` build that did not also ask for
`inference`. Serving needs `--features setfit,inference`; `inference` is default-on.

**Zero `Cargo.lock` change.** `tokenizers`, `sha2` and `aprender-rand` were already
locked as optional deps of `aprender-core`; enabling a feature adds no package.
T-04-SC measured, not assumed: `Cargo.lock` is not in `git diff --name-only`.

## The two guards, SHOWN ABLE TO FAIL

A test filter matching zero tests exits 0, and an assertion that has only ever been
observed passing proves nothing about the behaviour it names. Both mutations were
applied to `router.rs`, watched fail **by name**, and reverted with
`cp /tmp/router.pristine`; the green baseline was re-measured after each.

| | Mutation | Result | Killed by |
| - | -------- | ------ | --------- |
| — | baseline | 10 passed, rc=0 | — |
| **M4** | route installed only `if state.has_setfit_model()` | **9 passed, 1 FAILED, rc=101** | `setfit_classify_with_no_model_is_503_and_is_not_404` |
| — | reverted | 10 passed, rc=0 | — |
| **B** | `DefaultBodyLimit` layer deleted | **9 passed, 1 FAILED, rc=101** | `setfit_classify_refuses_a_body_over_the_contract_limit` |
| — | reverted | 10 passed, rc=0 | — |

**M4's failure message is the finding itself:**

```
panicked at crates/aprender-serve/src/api/setfit_handlers.rs:773:9:
assertion `left != right` failed: the route MUST exist whenever the feature is
compiled in (review M4)
  left: 404
 right: 404
```

That is not a hypothetical. The previous design's Task 1 installed the route
conditionally while its Task 3 expected a 503; the two could not both be satisfied,
and 404 is what the conditional version actually answers.

**B's failure message is the more interesting one, because it is not the obvious
number:**

```
an oversized body must be refused 4xx before any compute; got 200 OK
```

**200, not 413 and not 400.** Without the layer, axum's own 2 MB default does not
fire (a 1 MiB text plus JSON overhead is under it) and the 256-text batch bound does
not apply (it is ONE text) — so the request is fully parsed and classified. The two
bounds in T-04-23 genuinely do not subsume each other, and this is the measurement
that shows it rather than the argument that asserts it.

## Test counts — scoped filters, status captured directly, never through a pipe

Every command ran as `cmd > log 2>&1; rc=$?` (CLAUDE.md verification rule 1).

```
$ cargo test -p aprender-serve --features setfit --lib setfit
rc=0     10 passed, 15498 filtered out          (plan criterion: >= 8)

$ cargo test -p aprender-serve --features setfit --lib api::
rc=0   1702 passed, 3 ignored, 13793 filtered   (the api:: regression witness)

$ cargo test -p aprender-serve --features setfit --lib
rc=101 15399 passed; 51 failed; 58 ignored      (15389 passed / 51 failed BEFORE
                                                 task 3 = +10, failures UNCHANGED)

$ cargo test -p apr-cli --features setfit --lib setfit_serve
rc=0      6 passed, 6712 filtered out

$ cargo test -p apr-cli --features setfit --lib serve
rc=0    358 passed, 6360 filtered out

$ cargo test -p apr-cli --features setfit --lib
rc=0   6703 passed, 15 ignored, 0 FAILED        (04-17 recorded 6697/15: +6)

$ cargo check -p aprender-serve --features setfit --lib      rc=0
$ cargo check -p aprender-serve --lib        (feature OFF)   rc=0
$ cargo check -p apr-cli --features setfit --lib             rc=0
$ cargo check -p apr-cli                     (feature OFF)   rc=0
$ cargo fmt -p aprender-serve -- --check                     rc=0
$ cargo fmt -p apr-cli -- --check                            rc=0
$ cargo clippy -p aprender-serve --features setfit --lib --no-deps        rc=0
$ cargo clippy -p apr-cli --features setfit --lib --all-targets --no-deps rc=0
```

**The `--lib setfit_serve` filter matching 6 rather than 0 is what proves the
`cfg(all(feature = "setfit", feature = "inference"))` gate engaged** — CLAUDE.md
rule 2, never label a run by intent. A wrong cfg would have produced an identical
`cargo check` rc=0 with the code compiled out entirely.

`--no-deps` on both clippy legs is F-03's requirement. Zero clippy diagnostics —
of any severity — name `setfit_handlers.rs`, `router.rs`, `types.rs`,
`mod_app_state_*.rs` or `serve/handlers.rs`, measured on the `--all-targets` output
as well as `--lib`.

### The 51 red serve tests are ONE pre-existing defect, diagnosed not eyeballed

All 51 panic at the same line with the same message:

```
panicked at crates/aprender-serve/src/contract_gate.rs:428:21:
attempt to multiply with overflow
```

`grep -c "contract_gate.rs:428"` on the run log returns **51** — one defect with 51
witnesses, in `apr_transformer::*`, `contract_gate::*` and `convert::*`. This plan
touches none of those modules and none of them names `AppState`, `HealthResponse` or
anything under `setfit`. The count is **identical before and after task 3** (51 → 51)
while passes went 15389 → 15399, so the delta is exactly the 10 new tests. Logged as
**D-04-08-A** in `deferred-items.md`, together with the 19 pre-existing type-drift
errors in `tests/ffn_coverage.rs` and `tests/gguf_extended_coverage.rs` that make
`cargo check -p aprender-serve --tests` red at the base commit.

## The tests, and what each one is for

**aprender-serve — 10, all driving the REAL `create_router_with_config` through
`tower::util::ServiceExt::oneshot`:**

| Test | Claim |
| ---- | ----- |
| `setfit_classify_returns_core_envelope_for_a_mixed_batch` | 200; body deserializes INTO `ClassifyResponse` (type-level parity, not string matching); 3 ordered results; FULL probability vectors; hash == the model's == `artifact_sha256_hex(bytes)`; backend non-empty with no SIMD capability token |
| `setfit_classify_treats_an_embedded_newline_as_one_text` | the M2 witness at the HTTP boundary — one text in, one result out |
| `setfit_classify_refuses_an_empty_batch` | 400, core's typed message |
| `setfit_classify_refuses_a_batch_one_over_the_contract_bound` | 400, naming both 257 and 256 |
| `setfit_classify_refuses_a_body_over_the_contract_limit` | 413, and asserted `is_client_error()` first so the weaker claim is also pinned |
| `setfit_classify_with_no_model_is_503_and_is_not_404` | **review M4** — 503, and explicitly `assert_ne!(status, NOT_FOUND)` |
| `setfit_readiness_is_503_with_no_model_and_reports_no_classifier` | 503, `model_loaded: false`, BOTH classifier keys absent |
| `setfit_readiness_is_200_and_reports_the_exact_artifact_hash` | 200 and the EXACT hash value, twice (against the model and against the bytes) |
| `setfit_classify_and_readiness_agree_on_the_artifact` | the two WIRE values name the same artifact — the cross-surface claim OPS-05 makes |
| `setfit_classify_is_the_only_surface_added_and_predict_is_untouched` | `/v1/predict` keeps its own slot-empty 503 |

**apr-cli — 6:**

| Test | Claim |
| ---- | ----- |
| `setfit_serve_tag_read_identifies_a_setfit_container` | the tag read recognises a setfit-stamped container |
| `setfit_serve_tag_read_does_not_divert_a_plain_apr` | a `qwen2`-tagged APR is NOT diverted |
| `setfit_serve_tag_read_is_none_for_a_file_that_is_not_an_apr_container` | an unidentifiable file falls through rather than raising a diagnosis |
| `setfit_serve_refuses_a_corrupted_artifact_before_binding_a_socket` | a container tagged `setfit` with no setfit document is refused `ModelLoadFailed`, naming "did not pass verification" and "was NOT started" |
| `setfit_serve_refuses_an_over_cap_artifact_through_the_bounded_door` | a real setfit header + sparse `set_len(MAX_ARTIFACT_BYTES + 1)`: the tag read still succeeds (asserted, so the test cannot pass for the wrong reason) and the bounded door refuses naming `declared_length` |
| `setfit_serve_startup_reads_bounded_loads_through_the_one_door_and_builds_the_real_router` | source assertion, EXACTLY ONE occurrence each of the bounded read, the loader, the shared router and the slot builder — plus zero unbounded whole-file reads |

The startup tests use `ServerConfig::default().with_host("127.0.0.1").with_port(0)`.
Port 0 is a legal ephemeral bind **on purpose**: if any of them ever reached the bind
it would succeed and then block in `axum::serve` forever, so "the server was not
started" fails as a hung test rather than passing quietly for another reason.

The source-assertion needles are assembled from fragments at RUNTIME (`needle(&[..])`,
the `setfit_io.rs` discipline). A literal needle would appear in the scanned file —
the test scans the module it lives in — and every `contains` would be satisfied by its
own source. Because they are assembled, `matches().count() == 1` is a real measurement:
zero means the shared door was replaced, two means a second call site grew.

## The fixture, and why it is duplicated

`aprender-core`'s `setfit::artifact::fixture` is `#[cfg(test)]`, and its own header
records that it stays that way (03-10's acceptance criteria reject a
`#[doc(hidden)]` test-support door on the shipped surface). It is therefore
unreachable from `aprender-serve` by design. The plan anticipated this and sanctioned
duplicating the small view builder with a comment naming the source; that is what
`setfit_handlers.rs::fixture` is, and its header says so.

What is duplicated is the SHAPE. Everything is built through core's **public** API —
`SetFitArtifactView` and `EncoderArchitecture` have public fields,
`write_setfit_apr`/`load_setfit_apr` are the two production doors, and
`artifact_sha256_hex` is the crate's one hashing path. **No golden hash is pinned**:
every identity assertion compares two values measured in the same run, so a drift in
this copy cannot make an assertion vacuously true — it makes the fixture unloadable,
which is red.

The model in the slot comes out of `load_setfit_apr`, so it passed every rung
including probe replay. The fixture helper cannot mint a `VerifiedSetFitModel` that
skipped one, and the fact that it succeeds is itself evidence the duplicated shape is
still a valid artifact.

## Deviations from Plan

### Auto-fixed

**1. [Rule 3 — blocking] `AppState::default()` does not exist; `impl Default for AppState` was added**

- **Found during:** Task 1. The plan's Task 2 step 2 says
  `AppState::default().with_setfit_model(Arc::new(model))`.
- **Issue:** `aprender-serve` has `new`, `with_registry`, `with_cache`, `demo` and a
  dozen `with_*` constructors, and **no `Default`**. Every existing constructor either
  requires an LLM (`new`, `with_registry`) or mints a placeholder one (`with_cache`,
  `demo`), so none can express "no generator, one classifier". A `demo()`-based state
  would also have reported `model_loaded: true`, which would have made the
  missing-model readiness test unwritable.
- **Fix:** `impl Default for AppState` in `mod_app_state_new.rs` — the file that
  already holds the constructors — spelling the empty state explicitly. `model_loaded()`
  is `false` for it, which is the honest reading and is what the 503 readiness test pins.
- **Cost:** one more copy of the field list (a fourth). Refactoring the existing three
  to delegate was judged out of scope.
- **Commit:** `82416fcc1`

**2. [Rule 3 — blocking] `mod_app_state_gpu.rs` and `types.rs` are in the diff and are not in `files_modified`**

- **Issue:** the plan's `files_modified` lists four aprender-serve files. But `AppState`'s
  `model_loaded()` lives in `mod_app_state_gpu.rs` and `HealthResponse` lives in
  `types.rs`, and the plan's own Task 1 step 5 requires extending both the readiness
  response and its `model_loaded`-inclusive semantics. Neither instruction is
  satisfiable without those files.
- **Why `model_loaded()` rather than only `build_health_response`:** patching the
  response builder alone would have left the PUBLIC `AppState::model_loaded()`
  reporting `false` for a state holding a verified classifier. That is a landmine for
  every other caller and exactly the "two places answer one question" shape OPS-03
  exists to prevent.
- **Commit:** `82416fcc1`

**3. [Rule 3 — blocking] Adding two `HealthResponse` fields broke 8 struct literals in 3 files**

- **Issue:** Rust struct literals are exhaustive, so `classifier_artifact_sha256` and
  `classifier_verified` broke `src/api/tests/{imp_140c, chat_completion_03,
  deep_apicov_02, format_chat_02, completion_request_response}.rs` and
  `tests/{api_coverage, api_deep_coverage, property_api}.rs`.
- **Alternative rejected:** emitting the readiness body as a `serde_json::Map` in the
  ready handler only. That would have left the typed `HealthResponse` no longer
  describing what `/health/ready` emits, and `/health` and `/health/live` emitting a
  different shape than `/health/ready` — two definitions of one wire body, which is
  the defect this phase keeps arguing against.
- **Fix:** `classifier_artifact_sha256: None, classifier_verified: None` at each site,
  with a one-line comment. **No assertion in any of those tests changed**, and the two
  round-trip tests still pass because the fields carry `#[serde(default)]`.
- **Commit:** `82416fcc1`

**4. [Rule 2 — missing critical] The startup tag read bounds the metadata allocation; `apr inspect` does not**

- **Issue:** the plan says to read the tag "the same tag-read approach as inspect".
  `inspect.rs:517` does `vec![0u8; header.metadata_size as usize]` from a **`u32` a
  hostile header controls** — up to 4 GiB allocated before anything is parsed. Copying
  that verbatim into a path whose whole purpose is refusing hostile artifacts before
  allocation (T-04-50) would have been a new instance of the defect review B5 was about.
- **Fix:** `MAX_TAG_READ_METADATA_BYTES = 16 MiB`, two orders of magnitude above any
  real `setfit-apr-v1` document (tokenizer bytes and every tensor live in the DATA
  section). Over it, the tag read answers `None` and the pre-existing APR path handles
  the file exactly as before.
- **Not fixed:** `inspect.rs`'s own unbounded read. It is out of scope and belongs to
  04-07's files.
- **Commit:** `05012923b`

**5. [Rule 2 — missing critical] `/v1/classify` is installed outside the `config.openai_api` group**

- **Issue:** the plan says "install whenever `cfg(feature = "setfit")` is enabled" and
  says nothing about `RouterConfig::openai_api`. The `/v1/` prefix makes the OpenAI
  group the tempting home, and every other `/v1/*` route lives there.
- **Why that would be wrong:** `openai_api` is a runtime toggle. Hanging classify off
  it means a server started with the OpenAI group disabled answers 404 for
  `/v1/classify` — the *identical* ambiguity M4 is about, arriving through a second
  door and past a test that only exercises the default config.
- **Fix:** installed in its own `cfg` block after the group, with the reasoning in the
  comment.
- **Commit:** `82416fcc1`

### Out of scope, surfaced not fixed

**6. `aprender-serve`'s standing red (D-04-08-A).** 51 lib-test failures on one
overflow at `contract_gate.rs:428`, and 19 type-drift errors across two integration
test targets. Recorded in `deferred-items.md` with the measurements above; not fixed,
because none of the three files is this plan's and the overflow needs a decision about
the intended integer width.

---

**Total deviations:** 5 auto-fixed (3 blocking, 2 missing-critical) + 1 surfaced. No
scope creep: 18 files, all under `crates/aprender-serve/` plus `apr-cli`'s
`Cargo.toml` and `commands/serve/handlers.rs`. No new package, no `Cargo.lock` change,
no `contracts/` change.

## Wave ownership, verified

```
$ git diff --name-only 2374ed321 HEAD
```

18 files. **No apr-cli `predict`, `inspect` or `eval` path appears** — the wave
contract with 04-07 holds. `STATE.md`, `ROADMAP.md`, `REQUIREMENTS.md` and
`contracts/` are untouched; the orchestrator owns the first three and this plan had no
reason to touch the fourth.

## Threat Register, as shipped

| Threat ID | Mitigation as shipped |
| --------- | --------------------- |
| T-04-23 (unbounded HTTP batch/body) | 1 MiB `DefaultBodyLimit` on the classify route only, from core's `MAX_REQUEST_BODY_BYTES`; a `MAX_BATCH_TEXTS` check in the handler using core's constant; core re-checks both before tokenizing. Mutation B proves the body limit is not redundant with either — without it a 1 MiB single text is served **200 OK** |
| T-04-24 (serving an unverified model) | the slot's type is `Arc<VerifiedSetFitModel>`, non-constructible outside `aprender-core`; startup returns `Err` on any ladder rung strictly BEFORE `TcpListener::bind`, asserted by two tests |
| T-04-25 (readiness reporting a hash the model does not have) | the hash is `model.artifact_sha256()` read off the loaded object; the readiness test asserts the EXACT value twice — against the model and against `artifact_sha256_hex` of the bytes — and a third test requires readiness and the classify response to agree on the wire |
| T-04-50 (hostile artifact file at startup) | `setfit_io::read_setfit_apr_file_bounded`, refused from the declared length; plus a 16 MiB bound on the tag read's metadata allocation, which `apr inspect` does not have |
| T-04-26 (auth gap on api::router paths) | **accepted, unchanged.** `APR_API_KEY*` still covers only the CPU fallback router in `apr-cli serve/auth.rs`; `/v1/classify` inherits the same gap as every other `realizar::api` route. Deliberately not fixed — the CONTEXT deferred list records the user decision. Noted here so 04-11 does not read this plan as having closed it |
| T-04-SC (package installs) | zero: `Cargo.lock` is not in the diff |

## Known Stubs

**None.** Every path is wired end to end: the route, the handler, the slot, the
builder, readiness, startup detection, the bounded read, the ladder and the shared
router. The one thing a reader might mistake for a stub is
`classifier_verified: Some(true)` — a field that can only ever be `true`. It is not a
placeholder: its truth is guaranteed by the slot's TYPE, the doc comment says exactly
that and says it must not be read as evidence of a per-request re-verification, and
the alternative (omitting it) would leave OPS-05's "verified state" unreported.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: new-network-endpoint | `crates/aprender-serve/src/api/router.rs` | `POST /v1/classify` is a new unauthenticated HTTP surface on `realizar::api`. It inherits T-04-26 (the accepted, user-deferred auth gap on that router) rather than introducing it, and it is bounded on both body bytes and batch size before any compute. Flagged so 04-11's audit sees it explicitly rather than inferring it from the accept row above. |

## Notes for Later Plans

- **04-09 (spawned smoke test).** The deterministic in-process leg is done — 10 tests
  in `api::setfit_handlers::tests`, running in every `cargo test -p aprender-serve
  --features setfit`. Yours is the ONE spawned one. `apr serve <setfit.apr>` needs
  `--features setfit` (with default `inference`); it prints
  `SetFit classifier verified — artifact_sha256=<hex> labels=<n>` before binding, and
  then `SetFit classification server listening on http://<addr>`. Bind address comes
  from `ServerConfig::bind_addr()` (`handlers.rs`, host+port); port 0 works.
- **04-10 (gates).** Counted filters: `aprender-serve --features setfit --lib setfit`
  (**10**), `apr-cli --features setfit --lib setfit_serve` (**6**). Both clippy legs
  need `--no-deps` (F-03). **Do not add a whole-crate `cargo test -p aprender-serve
  --lib` leg** — it cannot go green (D-04-08-A) and so cannot distinguish a regression
  from the standing red; nor a `cargo check -p aprender-serve --tests` leg, red at the
  base for 19 type-drift errors. `cargo check -p aprender-serve` with the feature OFF
  is a legitimate leg and passes: the ungated build is what proves the gating. The
  SAFE-02 matrix now spans core/train/cli/serve.
- **04-11 (requirements audit).** **Nothing was flipped in REQUIREMENTS.md.** OPS-05's
  serving leg is delivered and tested at the fixture tier; whether that closes the
  requirement depends on 04-09's spawned proof and on OPS-02's train leg, neither of
  which this plan owns, and REQUIREMENTS.md is shared across concurrent worktree
  agents. T-04-26 is still open and still accepted — see the register above.
- **Anyone touching `HealthResponse`.** It now has two `Option` fields with
  `skip_serializing_if`. Adding a third non-`Option` field breaks 8 struct literals in
  3 files; adding an `Option` one breaks the same 8. The `#[serde(default)]` is what
  keeps the round-trip tests green.

## Self-Check: PASSED

Files claimed, checked on disk:

```
FOUND: crates/aprender-serve/src/api/setfit_handlers.rs   (35.6K)
FOUND: crates/aprender-serve/src/api/router.rs            (21.5K)
FOUND: crates/aprender-serve/src/api/types.rs             ( 6.5K)
FOUND: crates/apr-cli/src/commands/serve/handlers.rs      (67.5K)
```

Commits claimed, checked in the log: `82416fcc1`, `05012923b`, `64fe1435b`.

| Assertion | Criterion | Observed |
| --- | --- | --- |
| `git status --short --untracked-files=all` | empty | **empty** |
| `git diff --diff-filter=D --name-only` per commit | empty | **empty** on all three |
| files changed since `2374ed321` | serve + 2 apr-cli files | **18**, no others |
| apr-cli predict / inspect / eval in the diff | 0 | **0** (wave contract with 04-07) |
| `Cargo.lock` in the diff | no | **not present** — zero new packages |
| `STATE.md` / `ROADMAP.md` / `REQUIREMENTS.md` / `contracts/` | not modified | **not touched** |
| serve `--lib setfit` | >= 8 passed | **10 passed, 0 failed** |
| serve failures before vs after task 3 | unchanged | **51 → 51**, all at `contract_gate.rs:428` |
| apr-cli whole-crate delta vs 04-17 | +6 (the new tests) | **6697 → 6703 passed**, 15 ignored, 0 failed |
| M4 guard | shown able to fail | **404 observed**, rc=101, reverted |
| body-limit guard | shown able to fail | **200 OK observed**, rc=101, reverted |
| classify request/response structs defined in serve | 0 | **0** — the extractor and the success body are core's |

---
*Phase: 04-apr-artifact-and-production-parity*
*Plan: 08 — COMPLETE*
*Completed: 2026-08-15*
