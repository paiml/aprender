---
phase: 04-apr-artifact-and-production-parity
plan: 09
subsystem: apr-cli
tags: [setfit, parity, goldens, manifest, in-band-negative, spawned-smoke, d-13, d-14, safe-01, ops-04, ops-01, m2, t-04-27, t-04-28, t-04-29, t-04-53, t-04-61]

# Dependency graph
requires:
  - phase: 04-04
    provides: "ClassifyRequestDocument / ClassifyResponse — the ONE request document all three legs carry and the ONE envelope all three parse INTO; ClassifyResponse::new and ClassifyResult::new, the public validating constructors the in-band negative uses; PartialEq that already excludes latency_ms (F-14(3))"
  - phase: 04-03
    provides: "write_setfit_apr / load_setfit_apr / artifact_sha256_hex — the two production doors the fixture goes through and the one hashing path the manifest uses"
  - phase: 04-07
    provides: "`apr predict <apr> --input <doc> --json` — the CLI leg, whose --json is core's envelope verbatim"
  - phase: 04-08
    provides: "POST /v1/classify + AppState::default().with_setfit_model + the readiness classifier fields — the HTTP leg and what the spawned smoke asserts"
provides:
  - "crates/apr-cli/tests/setfit_parity.rs — the D-13/D-14 three-surface parity gate: 20 always-on tests + 1 tier3 spawned smoke"
  - "tests/fixtures/setfit_parity/goldens.json + goldens.sha256 — the frozen library answers and the Ph1 D-13 manifest, both proven able to fail"
  - "the [[test]] target `setfit_parity` with required-features = [setfit, inference]"
  - "D-04-09-A: `cargo check -p apr-cli --no-default-features` is red at base — 04-10 must not wire it as a gating leg"
affects: [04-10, 04-11, 04-15]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A parity gate over three surfaces must be handed ONE serialized document, not three constructions of 'the same' input — the CLI leg reads the exact bytes the HTTP leg posts."
    - "The comparator RETURNS a typed mismatch instead of panicking, so the in-band negative can assert rejection by matches! rather than by log inspection."
    - "A rejection test needs its mirror: a perturbation BELOW the bound must be accepted, or the rejection is consistent with a comparator that refuses any two distinct floats."
    - "A golden system needs two independent tamper classes proven: edit-the-data (caught by the manifest) and edit-the-data-and-re-sign (caught by the value comparison). Proving only one leaves the other untested."
    - "A labels-only golden comparison over a random-head fixture is near-vacuous: the argmax is constant, so a reversed row order passes. Found by mutation, not by review."
    - "Verify the leg a note names before writing the note. The plan named a gating command that does not compile."

key-files:
  created:
    - crates/apr-cli/tests/setfit_parity.rs
    - crates/apr-cli/tests/fixtures/setfit_parity/goldens.json
    - crates/apr-cli/tests/fixtures/setfit_parity/goldens.sha256
  modified:
    - crates/apr-cli/Cargo.toml
    - .planning/phases/04-apr-artifact-and-production-parity/deferred-items.md

key-decisions:
  - "NO realizar dev-dependency was added. The plan's Task 1 step 1 required one and pre-recorded its cost to SAFE-02; it turns out to be unnecessary, because `--features setfit,inference` (the test target's own required-features) already makes `realizar::api` reachable from an integration test WITH its setfit surface. T-04-61 is therefore not mitigated — it does not arise."
  - "The artifact under comparison is a SYNTHETIC fixture built through core's public writer, and the SUMMARY says so in the same breath as the parity claim. Parity is a statement about three READERS agreeing on one artifact; it is not an end-to-end claim, and F-10 still blocks the end-to-end one."
  - "The spawned smoke compares the FULL frozen rows, not the labels — because the labels-only version was measured passing under a reversed row order."
  - "The raw-TcpStream HTTP client was chosen over adding an HTTP client crate: T-04-SC (zero new packages this phase) binds this file too, and the round trip is 60 lines with an explicit de-chunker."
  - "`cargo check -p apr-cli --no-default-features` is NOT the SAFE-02 leg: it is red at the base commit for four pre-existing `inference`-gating defects. The green leg is `cargo check -p apr-cli --all-targets` with default features."

requirements-completed: []

# Metrics
duration: ~3h
completed: 2026-08-15
---

# Phase 04 Plan 09: The Three-Surface Parity Gate — Summary

**Library, spawned CLI and in-process HTTP provably agree on one committed input set;
the goldens that pin the answer were shown able to fail two independent ways; and the one
spawned-serve test starts a real `apr serve` on a port the parent actually knows. Two claims
the plan asked me to write down turned out to be wrong on measurement — the dev-dependency
the harness supposedly needed, and the build leg SAFE-02's evidence supposedly comes from —
and both were corrected against the tree rather than transcribed.**

## Task Commits

| Task | Deliverable | Commit |
| ---- | ----------- | ------ |
| 1 | harness: fixture, one request document, three legs, typed comparator, 5 source guards — 12 tests | `21fb17be1` |
| 2 | frozen goldens + SHA-256 manifest + 4 in-band negatives — 7 more | `671460737` |
| 3 | the ONE tier3 spawned-serve smoke + its structural guard — 2 more | `ac5e1f9ed` |
| — | correct the SAFE-02 leg the Task 1 note names (measured at base) | `1d3cfe39b` |

Working tree empty after every commit; zero deletions in any commit; `Cargo.lock` is not in
the diff (zero new packages, T-04-SC measured rather than asserted).

## THE HONEST BOUNDARY, stated first

This gate compares **three readers of one artifact**. It does **not** prove that training
produces an artifact the three agree on, and it does not claim to.

The artifact is a synthetic `setfit-apr-v1` fixture built through core's PUBLIC
`write_setfit_apr` and loaded through core's PUBLIC `load_setfit_apr` — the full-pin-shape
recipe (`vocab_remap: None`, `evidence.epsilon_used: null`), by way of `aprender-serve`'s own
duplicate of it, because core's `fixture_view_full_pin_shape()` is `#[cfg(test)]` and 03-10's
acceptance criteria keep it that way.

The end-to-end chain remains blocked by **F-10** and is a Phase 5 item: `CALIBRATED_REGIMES`
admits exactly one encoder, the phase-3 MiniLM slice, whose 97-row vocabulary closure cannot
compute `probe_unicode` — measured on three independent routes by 04-12. **This plan did not
route around it.** Synthesising an APR-capable encoder to manufacture "trained" bytes would
have compiled and would have satisfied every acceptance criterion to the letter, while the
model compared was not the model training produced; 04-12 identified and refused exactly that
shortcut and so does this plan. The module header of `setfit_parity.rs` says all of the above
in the file itself, so a reader cannot mistake the fixture for a training output.

**Why a synthetic fixture is nevertheless honest here:** the claim under test is
`library(x) == cli(x) == http(x)` for a fixed artifact. The artifact's provenance is not a
free variable in that claim. It WOULD be a free variable in an OPS-02 end-to-end claim, and
nothing here flips OPS-02.

## What shipped

`crates/apr-cli/tests/setfit_parity.rs` — 21 tests, 20 always-on and 1 `#[ignore]`d.

### The three legs, and the one document

```
              ┌── library:  VerifiedSetFitModel::classify(&document)
one           │
ClassifyRequest ── CLI:  spawn env!("CARGO_BIN_EXE_apr") predict <apr> --input <doc> --json
Document      │           (the SAME serialized bytes, written to a temp file)
serialized    │
ONCE          └── HTTP:  realizar::api::create_router_with_config + tower oneshot
                          POST /v1/classify (the SAME serialized bytes as the body)
```

The CLI and HTTP legs parse stdout / the response body **INTO
`aprender::setfit::ClassifyResponse`**, never into a `serde_json::Value`. A re-keyed field, a
dropped row or a row that violates the envelope's invariants is therefore a deserialization
failure at the boundary, and core's validated `Deserialize` doubles as the proof that each
surface emitted a well-formed envelope rather than merely valid JSON.

The HTTP leg's model is an **independent** `load_setfit_apr` of the same bytes, so the
artifact-hash equality the gate asserts is a claim about two loads rather than about one
shared object. The CLI leg's is a third, in a separate process.

### The committed input set — all six contract probes plus two extras

| # | text | why |
| - | ---- | --- |
| 0 | `ok` | `probe_minimal` |
| 1 | the pangram | `probe_ascii_pangram` |
| 2 | `El rapido zorro … — naive cafe, pi = 3.14159` | `probe_unicode` (3-byte UTF-8) |
| 3 | `PROBE_TRUNCATION_REPEAT_UNIT.repeat(PROBE_TRUNCATION_REPEAT_COUNT)` | `probe_truncation_boundary`, built from the contract's own exported constants |
| 4 | `Stance detection: … #debate @user123 https://example.com` | `probe_social` |
| 5 | `line one\nline two\ttabbed   spaced` | `probe_whitespace` — **the review-M2 witness** |
| 6 | `""` | an entry a `filter(|s| !s.is_empty())` would silently drop, shifting every later index |
| 7 | `🦊🌮 café naïve π ✅` | 4-byte UTF-8; the contract probes only reach 3 |

Plus a single-text document (`probe_whitespace` alone) and the batch with
`include_logits = true`.

### The comparator

`compare_parity(a, b) -> Result<(), ParityMismatch>` — a RETURNED typed mismatch, not a
panic, which is what lets the negative assert rejection by `matches!`.

| compared | how |
| -------- | --- |
| `label` | EXACT (a tolerance on a label is a category error) |
| `artifact_sha256`, `schema_version`, `backend` | EXACT |
| `token_count`, `truncated`, result arity, logits presence/arity | EXACT |
| probabilities, logits, margins | `within(delta, 1.0e-5)` — the contract's `parity_probabilities_abs` / `parity_logits_abs`, read from core's `PROBE_*_ABS_TOLERANCE` so the number lives in ONE place |
| `latency_ms` | **neither equality nor positivity** — only `is_finite() && >= 0.0` |

`within` is the mandated NaN-visible idiom,
`matches!(delta.partial_cmp(&bound), Some(Less | Equal))`. `backend` is additionally asserted
to be a three-segment identity carrying no CPU-capability token (D-12 / review B6): a
capability describes the host, never the dispatch that ran.

`assert_eq!(cli, http)` is also asserted directly, because F-14(3) already removed
`latency_ms` from core's `PartialEq` — the comparison 04-07's notes said would be the
intended one.

## Test counts — scoped filters, status captured DIRECTLY, never through a pipe

Every command ran as `cmd > log 2>&1; echo "rc=$?"`.

```
$ CARGO_INCREMENTAL=0 cargo test -p apr-cli --features setfit,inference --test setfit_parity
rc=0     20 passed, 1 ignored, 0 failed                (plan criterion Task 1: >= 6)

$ CARGO_INCREMENTAL=0 cargo test -p apr-cli --features setfit,inference \
      --test setfit_parity -- --ignored --nocapture
rc=0      1 passed, 20 filtered out

$ cargo test -p apr-cli --features setfit,inference --test setfit_parity golden
rc=0      7 passed, 14 filtered out                    (plan criterion Task 2: >= 4)

$ cargo test -p apr-cli --features setfit,inference --test setfit_parity parity_
rc=0     13 passed,  8 filtered out

$ cargo test -p apr-cli --features setfit --lib
rc=0   6751 passed, 15 ignored, 0 FAILED               (04-07 recorded 6751 — ZERO delta)

$ cargo check -p apr-cli --features setfit,inference --all-targets     rc=0
$ cargo check -p apr-cli --all-targets           (setfit OFF)          rc=0
$ cargo fmt -p apr-cli -- --check                                      rc=0
$ cargo clippy -p apr-cli --features setfit,inference --all-targets --no-deps  rc=0
        (zero diagnostics of any severity name setfit_parity.rs)
```

`--no-deps` is F-03's requirement. The lib count being **identical** to 04-07's is the
regression witness: this plan adds a test target and a dev-dep feature and touches no `src/`
file, so a moved lib count would have been a defect in this plan.

### The 21 tests

**Live pairwise parity (7)**

| test | claim |
| ---- | ----- |
| `parity_library_and_cli_agree_on_the_batch_document` | + both report the hash of the bytes on disk |
| `parity_library_and_http_agree_on_the_batch_document` | + the backend is a capability-free identity |
| `parity_cli_and_http_agree_on_the_batch_document` | + `assert_eq!(cli, http)` through core's own `PartialEq` |
| `parity_all_three_agree_on_a_single_text_document` | the arity a batch cannot witness |
| `parity_all_three_agree_with_logits_requested` | every leg carries logits, arity == probability arity |
| `parity_every_leg_returns_one_result_per_text` | **review M2 / T-04-53**: `results().len() == texts.len()` on all three legs, with four non-vacuity assertions that the input set really contains a newline, a tab, an empty string and non-ASCII |
| `parity_the_truncation_boundary_text_truncates_on_every_leg` | `truncated == true` on all three, so `truncated`-parity cannot hold vacuously by everyone reporting `false` |

**Source assertions over the harness itself (6)** — code lines only, needles assembled at
runtime from fragments (the `setfit_io.rs` discipline), each with a non-vacuity assertion
that the filter did not eat the file:

`parity_harness_resolves_the_binary_through_cargo_and_never_through_path` (T-04-29),
`parity_harness_never_splits_an_input_text_into_lines` (M2),
`parity_harness_makes_no_assertion_about_latency_being_positive`,
`parity_harness_writes_no_unwrap`,
`parity_harness_builds_its_fixture_from_the_full_pin_shape_recipe`,
`parity_harness_spawns_exactly_one_serve_process` (D-14).

The F-05 defect was designed out rather than discovered: the module header deliberately
spells `latency_ms` and describes the line-delimited format the gate forbids, so `code_lines`
strips comments — and it is implemented with `split('\n')` rather than the obvious iterator
precisely so the "never splits input into lines" scan can be honest.

**Goldens + negatives (7)** and **the spawned smoke (1)** — below.

## The goldens, and the proof they can fire

```
tests/fixtures/setfit_parity/goldens.json     6.7 KB, 256 lines
tests/fixtures/setfit_parity/goldens.sha256   the Ph1 D-13 manifest

goldens.json  sha256 = ee942f6c8dd6e8049d4c99dca5df4dade6363f3591b7d7f1f63a4ef7d6125331
fixture .apr  sha256 = a704f0e0836e5b993babbcc2f4b6d55254ad39514b1e980709892ed6e3bc80c7
backend (exact, frozen) = cpu:setfit-core:autograd-trueno-matmul
```

Three documents frozen (`batch` 8 rows, `single` 1 row, `batch_with_logits` 8 rows with
logits). **No latency value is recorded** — a measurement cannot be frozen. `backend` IS
recorded exactly, because an identity can be.

Blessing requires `APR_BLESS_SETFIT_PARITY_GOLDENS=1` and rewrites both files together, so it
is a deliberate act whose control is the git diff.

**Both halves shown able to FAIL on disk, then reverted and re-measured.** Two independent
tamper classes, because proving only one leaves the other untested:

| | mutation | result | killed by |
| - | -------- | ------ | --------- |
| — | baseline (`golden` filter) | 7 passed, rc=0 | — |
| **M1** | one byte of `goldens.json` flipped to a space | **5 passed, 2 FAILED, rc=101** | `golden_file_matches_its_committed_manifest` and the non-vacuity arm of `golden_a_single_flipped_byte_fails_the_manifest`; the VALUE comparison stayed green — exactly the split a whitespace edit should produce |
| **M2** | a probability `+1e-3` **and the manifest re-signed by the tamperer** | **6 passed, 1 FAILED, rc=101** — `batch result 0 class 0: probability moved 0.18997816249762542 -> 0.18897816249762542` | `golden_library_answers_are_unchanged` |
| — | reverted, manifest restored | 19 passed (then), rc=0 | — |

M2 is the one that matters: a manifest alone is defeated by re-signing, and the value
comparison alone is defeated by nothing — but only the pair covers both.

**The in-band negatives, which run in EVERY cargo test invocation** (Ph1 D-24 / Ph2 D-25 /
Ph3 D-08 / this plan):

| test | what it proves |
| ---- | -------------- |
| `golden_negative_the_comparator_rejects_a_skewed_probability` | a 10x-tolerance skew is REJECTED and NAMED (`matches!(…, Err(ParityMismatch::Probability { index: 0, class: 0, .. }))`), plus the same rejection through `assert_parity` captured as a panic. Non-vacuity first: `compare_parity(&good, &good)` must succeed. |
| `golden_negative_a_sub_tolerance_perturbation_is_still_parity` | a perturbation an order of magnitude BELOW the bound is ACCEPTED — without this, the rejection above is consistent with a comparator that refuses any two distinct floats |
| `golden_a_single_flipped_byte_fails_the_manifest` | the flipped byte, in memory, every run — plus a forged-manifest arm, so the check cannot be satisfied by rewriting the cheaper of the two files |
| `golden_negative_a_malformed_envelope_does_not_deserialize` | a `null` probability (serde's type check) AND a row summing to 1.5 (the VALIDATING `TryFrom`, which a plain type check waves through) both fail — with a legal row as the non-vacuity control (review M1) |
| `golden_negative_uses_the_public_validating_constructors` | source assertion: the skew is built with `ClassifyResponse::new` / `ClassifyResult::new`, and no `*_for_tests` / `*_unchecked` / `skip_validation` door appears anywhere |

**The negative needed no backdoor, and none was shipped.** The perturbation is finite and
**mass-preserving** (`+delta` on class 0, `-delta` on class 1), so `ClassifyResult::new`'s
finiteness AND probability-mass checks both accept it. That is the point: the gate under test
is the COMPARATOR, so the negative must be a value the constructor considers entirely
legitimate. (The naive one-sided skew would have been rejected by the constructor's mass
check at 1e-6 and would have tested the wrong thing.)

## The ONE tier3 spawned-serve smoke

```
$ cargo test -p apr-cli --features setfit,inference --test setfit_parity -- \
      --ignored --nocapture spawned_serve_smoke
spawned_serve_smoke: attempt 1, port 61252, ready after 317 ms (4 poll(s)),
                     8 rows matched the frozen goldens
test result: ok. 1 passed; 0 failed; 20 filtered out
```

**Five recorded green runs, no retry ever needed:**

| run | port | readiness | polls | attempt |
| --- | ---- | --------- | ----- | ------- |
| 1 | 60626 | 320 ms | 4 | 1 |
| 2 | 60791 | 334 ms | 4 | 1 |
| 3 | 60816 | 331 ms | 4 | 1 |
| 4 | 60842 | 325 ms | 4 | 1 |
| 5 (final, `CARGO_INCREMENTAL=0`) | 61252 | 317 ms | 4 | 1 |

Spread 317–334 ms, always the 4th poll of a 50 × 100 ms budget — roughly 15x headroom.
`pgrep -f "serve run"` returns rc=1 (no match) after every run: **no orphan process**.

**The port handoff protocol, as shipped** (the review found it underspecified):

1. The PARENT binds `127.0.0.1:0`, reads `local_addr().port()`, drops the listener, and
   passes that number to the child as `--port`. Not a fixed high port (collision-prone) and
   not port 0 (the parent could not then know where to connect).
2. The reserve-then-release race is **handled, not pretended away**: a child that exits
   before readiness has its captured output inspected for an address-in-use signature, and
   the whole reserve-spawn sequence retries up to 3 times. It has not yet fired in 5 runs.
3. Both child pipes are `Stdio::piped()` and DRAINED on their own threads into 64 KiB-capped
   buffers, so a chatty child cannot fill a pipe and deadlock a parent that only reads after
   `wait()`. The captured tail is included in every failure message.
4. `ServeChild` implements `Drop` (kill + wait), so no orphan survives a panic. The child's
   status is read from `try_wait()` / the reaped `ExitStatus` — never through a pipe.
5. Readiness: `GET /health/ready` polled to 200, then `classifier_artifact_sha256` compared
   to the fixture FILE's hash and `classifier_verified` asserted `true` (OPS-05).
6. One `POST /v1/classify` with the same committed document, parsed INTO `ClassifyResponse`,
   artifact hash re-checked, and the full frozen rows compared.

**No HTTP client package was added.** T-04-SC binds this file too, so the round trip is raw
`std::net::TcpStream` with `Connection: close`, plus an explicit de-chunker for the chunked
case rather than an assumption that it cannot happen.

### The mutation that changed the design

The plan says "assert labels match the golden". Mutating `golden_batch_labels()` to reverse
the row order **passed** — because the fixture's random head produces a CONSTANT argmax
(`neutral` for all 8 probes), so a labels-only comparison cannot distinguish a row
permutation. That is a near-vacuous gate, and review would not have caught it.

Fixed by comparing the **full frozen rows** through `assert_golden_rows()` — the SAME
function the library golden test uses, so "the served answers are the frozen answers" is one
claim rather than two that can drift. That function now carries a **non-vacuity assertion**
that the frozen rows are not all identical in leading probability and not all identical in
`token_count`, so the day a fixture change flattens the rows the suite says so instead of
quietly proving nothing.

Re-mutated after the fix:

| | mutation | result |
| - | -------- | ------ |
| — | baseline | 1 passed, rc=0 |
| **M3** | `golden_batch_rows()` reversed | **1 FAILED, rc=101** — `spawned batch result 0 class 0: probability moved 0.1885219021812737 -> 0.18897816249762542` |
| — | reverted | 1 passed, rc=0 |

The rows differ from each other by ~7e-4 in leading probability — 73x the 1e-5 tolerance —
and `token_count` ranges 2…256, so the comparison rests on values that genuinely vary.

## Deviations from Plan

### 1. [Rule 1 — the plan's premise was false] NO `realizar` dev-dependency was added

- **Plan:** Task 1 step 1 requires `realizar = { workspace = true, features = ["setfit"] }`
  in `[dev-dependencies]`, and spends a paragraph pre-recording the cost: Cargo forbids an
  optional dev-dep, so the entry would be unconditional and its features would UNIFY with the
  normal `realizar` dependency for any build with test targets — weakening the
  `cargo test --no-default-features` RUN leg SAFE-02 reads.
- **Measured:** it is not needed. `realizar` is already the optional NORMAL dependency behind
  `inference` (`Cargo.toml:146`), `setfit` already carries the weak `realizar?/setfit`
  (04-08), and an integration test can use its package's normal dependencies. So the test
  target's own `required-features = ["setfit", "inference"]` makes `realizar::api` reachable
  **with its setfit surface**, with no dev-dep. `cargo check -p apr-cli --features
  setfit,inference --test setfit_parity` → rc=0 on the first attempt, with no dev-dep
  present.
- **Consequence:** **T-04-61 does not arise.** There is nothing to mitigate, because the
  weakening does not occur. This is strictly better than the mitigation the plan designed.
- **The RULE is still recorded** beside the dev-dependency block, because the next person
  reaching for a dev-dep on a crate that is also a normal dep needs it: dev-dep features
  unify, and any such entry weakens the corresponding gating-by-absence `cargo test` leg.
- **The one dev-dep change:** `tower = "0.5"` → `features = ["util"]`, for
  `ServiceExt::oneshot`. `tower` is dev-only and appears in no shipped build. `Cargo.lock` is
  not in the diff.

### 2. [Rule 1 — bug in the plan's own verification instruction] the SAFE-02 leg the plan names does not compile

The plan requires the Cargo.toml note to name `cargo check -p apr-cli --no-default-features`
as the leg SAFE-02's gating evidence is read from. **Verifying that rather than transcribing
it found the leg is red**, and red at the base commit:

```
$ cargo check -p apr-cli --no-default-features > log 2>&1; echo "rc=$?"
rc=101   error: could not compile `apr-cli` (lib) due to 4 previous errors

$ git checkout f2824611a -- crates/apr-cli/Cargo.toml   # a TRUE base measurement:
$ cargo check -p apr-cli --no-default-features           # this plan touches no src/ file
rc=101   the same 4 errors
```

| file | error |
| ---- | ----- |
| `src/commands/explain.rs:231,344` | `realizar::safetensors::find_sibling_file` — unlinked crate |
| `src/commands/diff_05_aprt_stage.rs:100` | `realizar::inference_trace::save_tensor::read_tensor_file` |
| `src/lib.rs:63` | re-exports `commands::serve::auth::apply`, itself `#[cfg(feature = "inference")]` |

`inference`-gated code that is not `cfg`-gated. **Not fixed here** — three files this plan
does not own, the fix is a `cfg` decision on the `explain`/`diff` surface, and 04-15 is
editing `crates/apr-cli/` in the same wave. Recorded as **D-04-09-A** in `deferred-items.md`
and in the Cargo.toml note, which now names the leg that IS green:
`cargo check -p apr-cli --all-targets` with DEFAULT features (setfit OFF, inference ON) — the
leg 04-06 and 04-07 actually used.

### 3. [Deviation, argued] the smoke test compares full rows, not labels

See "The mutation that changed the design" above. A superset of what the plan asked for,
adopted because the plan's version was measured near-vacuous.

### 4. [Rule 2 — missing critical] the plan's negative would not have compiled as written

"one probability perturbed by 10x the contract tolerance" through `ClassifyResult::new` is
rejected by that constructor: it checks probability mass to 1e-6, and a one-sided 1e-4 skew
breaks it. The shipped skew is **mass-preserving**, which is what keeps the negative about
the comparator instead of about the constructor. Recorded because the distinction is the
whole point of an in-band negative.

### 5. [Rule 3 — blocking] the fixture is duplicated a third time

Core's `fixture_view_full_pin_shape()` is `#[cfg(test)]` (03-10 keeps it that way) and
`aprender-serve`'s copy is `#[cfg(all(test, feature = "setfit"))]` — both unreachable from an
integration test. The harness therefore carries a third copy of the SHAPE, built entirely
through core's public API, with a module header naming both ancestors. **No golden hash is
hand-written into the fixture builder**: the artifact hash in `goldens.json` is measured at
bless time, and every live identity assertion compares two values measured in the same run.

### 6. [Deviation, minor] `#[allow(clippy::disallowed_methods)]` on one function

`serde_json::json!`'s macro expansion calls `Result::unwrap` on an infallible construction —
5 diagnostics, none hand-written. The allow is scoped to `fixture_view_full_pin_shape()`, its
doc comment says exactly what it covers, and `parity_harness_writes_no_unwrap` asserts zero
`.unwrap()` on any code line of the file, so the allow cannot be silently widened.

---

**Total deviations:** 2 corrections of false premises in the plan, 1 missing-critical, 1
blocking, 2 argued. No new package, no `Cargo.lock` change, no `contracts/` change.

## Threat Register, as shipped

| Threat ID | Mitigation as shipped |
| --------- | --------------------- |
| T-04-27 (vacuous parity gate) | 5 in-band negatives running in every `cargo test`: the skewed rejection, its sub-tolerance mirror, the flipped byte, the forged manifest, and two malformed envelopes with a legal control. Plus 3 on-disk mutations (M1/M2/M3) each observed red BY NAME and reverted with the green baseline re-measured. |
| T-04-28 (golden drift) | SHA-256 manifest over `goldens.json`, proven able to fire by a real byte flip (M1) — and the value comparison proven able to fire against a tamperer who RE-SIGNS (M2), which the manifest alone cannot catch. |
| T-04-29 (stale/shadowed `apr`) | `env!("CARGO_BIN_EXE_apr")` only. Source-asserted: the assembled `Command::new("apr")` needles count 0, and the cargo needle counts >= 1. |
| T-04-53 (surfaces compared on different inputs) | ONE `ClassifyRequestDocument` serialized ONCE; the CLI leg reads those bytes from a file, the HTTP leg posts them. `parity_every_leg_returns_one_result_per_text` pins per-leg input fidelity with four non-vacuity assertions about the input set. |
| T-04-61 (a SAFE-02 run leg that proves less than it reads) | **Does not arise** — no dev-dep on `realizar` was added, so no feature unification occurs. The RULE is recorded in Cargo.toml regardless, and the leg the plan pointed at was found red and replaced (deviation 2). |
| T-04-SC (package installs) | Zero. `Cargo.lock` is not in `git diff --name-only`; the only manifest change is a feature on an existing dev-dep and a `[[test]]` block. The spawned smoke's HTTP client is raw `TcpStream` for exactly this reason. |

## Known Stubs

**None.** Every leg is wired end to end and every assertion compares measured values. The one
thing a reader might mistake for a stub is the `APR_BLESS_SETFIT_PARITY_GOLDENS` escape
hatch — it is not a stub but the blessing path, and it is deliberately env-gated so that
neither a plain `cargo test` nor a `--ignored` run can silently re-bless (which would make the
golden test vacuous). The control on a bless is the git diff, and M1/M2 show what a bless-less
edit costs.

## Threat Flags

**None.** No new network endpoint, no new auth path, no new file-access pattern in shipped
code: everything added is under `crates/apr-cli/tests/` plus a dev-dependency feature and a
`[[test]]` declaration. The spawned smoke binds only `127.0.0.1` on a port the parent
reserved.

## Notes for Later Plans

- **04-10 (gates).** Two counted filters, both requiring `--features setfit,inference`:
  `--test setfit_parity` (**20 passed, 1 ignored**) and
  `--test setfit_parity -- --ignored spawned_serve_smoke` (**1 passed**). The second is the
  tier3 leg. **Do NOT wire `cargo check -p apr-cli --no-default-features`** — it is red at
  base (D-04-09-A) and cannot distinguish a regression from the standing red; use
  `cargo check -p apr-cli --all-targets` with default features, which is what 04-06/04-07
  used and which passes. Clippy needs `--no-deps` (F-03). The feature-matrix comment should
  carry the dev-dep unification RULE (it still applies to any future dev-dep) but must NOT
  claim this plan incurred it — it did not.
- **04-11 (requirements audit).** **Nothing was flipped in REQUIREMENTS.md.** SAFE-01's
  parity detector exists and is proven able to fail; OPS-04's envelope is compared field by
  field across all three surfaces. But the parity is over a SYNTHETIC fixture, because F-10
  still blocks artifact production on this host — so any requirement whose text implies an
  end-to-end demonstration is NOT closed by this plan. The three-surface agreement is a real,
  citable fact; "training produces an artifact the three agree on" is not yet.
- **04-15 (spawned lifecycle).** No `[[test]]` entry was declared for your file, exactly as
  the plan required — `crates/apr-cli/tests/setfit_cli_lifecycle.rs` stays auto-discovered
  behind its own module cfg guard. My only Cargo.toml edits are the `tower` `util` feature,
  the `setfit_parity` `[[test]]` block and the note beside `[dev-dependencies]`.
- **Anyone adding a fixture-based golden here.** The fixture's head is random and its argmax
  is CONSTANT across all six probes. Any comparison that rests on labels alone is near-vacuous
  and will pass a permuted row order. `assert_golden_rows()` carries the non-vacuity
  assertions that catch this; keep them.
- **Anyone re-blessing the goldens.** `artifact_sha256` is compared EXACTLY, so a change to
  the fixture builder — even one that leaves every probability inside tolerance — turns
  `golden_library_answers_are_unchanged` red with a message that says exactly that and tells
  you to re-bless deliberately. That is intended.

## Self-Check: PASSED

Files claimed, checked on disk:

```
FOUND: crates/apr-cli/tests/setfit_parity.rs                          (~2.3k lines)
FOUND: crates/apr-cli/tests/fixtures/setfit_parity/goldens.json       (6.7K, 256 lines)
FOUND: crates/apr-cli/tests/fixtures/setfit_parity/goldens.sha256     (304B)
```

Commits claimed, checked in the log: `21fb17be1`, `671460737`, `ac5e1f9ed`, `1d3cfe39b`.

| Assertion | Criterion | Observed |
| --- | --- | --- |
| `git status --short --untracked-files=all` after each commit | empty | **empty** |
| `git diff --diff-filter=D --name-only` base..HEAD | empty | **empty** — zero deletions |
| files changed vs base `f2824611a` | 5 | **5**, no others |
| `STATE.md` / `ROADMAP.md` / `REQUIREMENTS.md` / `contracts/` | not modified | **not touched** |
| `Cargo.lock` in the diff | no | **not present** — zero new packages |
| goldens gitignored | no | `git check-ignore` exits 1 |
| default filter | >= 6 tests | **20 passed, 1 ignored, 0 failed**, rc=0 |
| `golden` filter | >= 4 tests | **7 passed**, rc=0 |
| `--ignored spawned_serve_smoke` | passes | **1 passed**, rc=0, ×5 runs |
| goldens committed WITH a manifest | yes | both files, `ee942f6c…` over `goldens.json` |
| manifest shown able to FAIL | by mutation | **M1**: 2 tests red, rc=101, reverted |
| golden VALUES shown able to fail against a re-signing tamperer | by mutation | **M2**: 1 test red naming both values, rc=101, reverted |
| spawned smoke shown able to fail | by mutation | **M3**: red naming both values, rc=101, reverted |
| fabricated artifact used to manufacture a green result | no | **no** — the fixture is a synthetic artifact through core's PUBLIC doors, the SUMMARY and the module header both say so, and no end-to-end claim is made |
| `latency_ms > 0` assertion anywhere | 0 | **0**, source-asserted |
| `.lines()` on any code line | 0 | **0**, source-asserted |
| `.unwrap()` on any code line | 0 | **0**, source-asserted |
| serve-spawning sites | exactly 1 | **1**, source-asserted |
| fixed port literal passed to the child | 0 | **0**, source-asserted |
| apr-cli lib regression | 6751 (04-07's number) | **6751 passed, 15 ignored, 0 failed** |
| orphan `apr serve` after the smoke runs | none | `pgrep -f "serve run"` rc=1 |

---
*Phase: 04-apr-artifact-and-production-parity*
*Plan: 09 — COMPLETE*
*Completed: 2026-08-15*
