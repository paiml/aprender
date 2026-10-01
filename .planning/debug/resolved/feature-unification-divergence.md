---
slug: feature-unification-divergence
status: resolved
trigger: "— systematic session on the feature-unification divergence, starting from the corrected candidate surface"
goal: find_and_fix
created: 2026-09-22
updated: 2026-09-22T-checkpoint2
---

# Debug Session: feature-unification-divergence

## Symptoms

**Expected behavior**
`invariance::every_baseline_case_reproduces_its_signature`
(`crates/aprender-forecast/src/invariance.rs:718`) reproduces every recorded baseline
signature byte-for-byte, under the build configuration CI actually runs
(`.github/workflows/ci.yml:289` → `--workspace --lib`). This gate is the proof of
D-19/SC2: a caller passing NO new argument gets a byte-identical answer.

**Actual behavior**
Under `--workspace` scope the gate is RED, deterministically:

```
case peyton/prophet/default on arch=aarch64 no longer reproduces its pre-change signature:
recorded aa669c2352dd376a, got e01f605419765a6a
```

Under `-p aprender-forecast --lib` scope it is GREEN, 9/9, repeatable. The baseline was
recorded in a scope neither CI nor a real consumer uses.

**Error messages**
`recorded aa669c2352dd376a, got e01f605419765a6a` — byte-identical on all retries in both
directions. rc=101 under the failing scope, rc=0 under the passing one.

**Timeline**
Surfaced 2026-09-21 during phase 06.1 plan 08 (gate sweep), by deviation 3 — a
workspace-wide base-vs-HEAD failing-set diff that replaced a per-crate probe. Narrowed
further the same day at `6c40821c3` (commit `41f6d7eb4`). Not known to have ever passed
under `--workspace`; the baseline predates the scope question being asked.

**Reproduction**
```bash
cargo nextest run --profile ci -p aprender-forecast --lib invariance::            # PASS 9/9
cargo nextest run --profile ci --workspace --lib  <filtered to this test>         # FAIL
cargo nextest run --profile ci -p aprender-forecast -p apr-cli --lib              # FAIL — one-variable flip
```
Independently reproduced by the orchestrator before the blocker was accepted.

## Prior Evidence (carried in — do not re-derive)

Source: `.planning/phases/06.1-forecast-exogenous-inputs-prophet-regressors-neuralprophet-e/deferred-items.md`
item 5 + its "Follow-up narrowing (orchestrator, 2026-09-21, at `6c40821c3`)" section, and
`06.1-08-SUMMARY.md` § "BLOCKER — the SC2 no-argument invariance gate is RED".

### Controls already run

| Control | Result | What it rules out |
|---|---|---|
| `-p aprender-forecast --lib invariance::` | PASS 9/9 repeatable | — baseline scope |
| `--workspace --lib` (filtered to this test) | FAIL, same hash ×3 | not flaky |
| `-p aprender-forecast --lib` under saturating 14-core busy-loop | PASS | CPU load; cooperative `FIT_BUDGET_SECS` |
| `-p aprender-forecast -p aprender-core --lib` | PASS | **aprender-core**'s `parallel`/`rayon` reduction order |
| `-p aprender-forecast -p apr-cli --lib` | FAIL, identical hash | isolates the flip to one added package |
| `-p aprender-forecast -p aprender-compute --features aprender-compute/parallel --lib` | PASS | **aprender-compute**'s `parallel`+`rayon` |
| `-p aprender-forecast -p aprender-quant --lib` | PASS | the `half` → `num-traits/libm` chain |

### Three corrections to the original mechanism paragraph (each measured)

1. The "23 aprender-core features" figure is a **`--no-dedupe` artifact**. Deduped,
   `apr-cli` pulls exactly **6** `aprender-core` edges: `default`, `format-compression`,
   `lz4_flex`, `parallel`, `rayon`, `zstd` — and since `default = ["parallel"]`,
   `parallel = ["rayon"]`, that is **two axes**: parallel and compression.
2. `safetensors-compare`, `setfit`, `format-encryption`, `format-quantize`, `half`,
   `hf-hub-integration` are **NOT in the delta**. Do not chase them.
3. The parallel/rayon exoneration was reached on the wrong crate. The crate whose feature
   set actually diverges is **`aprender-compute`** (`[lib] name = "trueno"`, SIMD/GPU):
   `default` alone in the scoped build vs
   `default, parallel, rayon, gpu, wgpu, bytemuck, futures-intrusive, pollster` under
   `apr-cli`. Two same-named features in two different crates.

### Dead leads (do not re-run)

- `num-traits/libm` via `apr-cli → aprender-quant(default) → half(default)`. The chain is
  real and would swap `exp`/`ln`/`powf` for non-bit-identical pure-Rust implementations —
  exactly this symptom's shape — but enabling it alone does not reproduce the flip.
- Rayon reduction order in `aprender-core`.

### Bisect rung that is unavailable

`apr-cli` does **not** build with `--no-default-features` on this tree (unresolved `axum` /
`realizar` imports), so "narrow inside apr-cli's defaults by turning its defaults off" is
not a usable step.

## Corrected Candidate Surface (the starting point for this session)

~**190 feature edges** across the **79 packages** `aprender-forecast` links, whose feature
set grows when `apr-cli` joins the build. Reproducible with two deduped
`cargo tree -e features` runs (scoped vs workspace) and a diff.

**Largest untested axis:** `gpu` / `wgpu` on `aprender-compute` — it can change kernel
dispatch, which is the one mechanism class that plausibly moves numerics.

## Constraints

- **Do NOT close this by re-recording the baseline under `--workspace`.** That inverts
  which scope is red without answering either real question.
- ~~Goal is **diagnose only**~~ — SUPERSEDED 2026-09-22. Diagnosis is complete (see
  Resolution). Goal is now **find_and_fix** with the key-sort fix selected by the human.
- The forbidden action is unchanged and still applies to the fix: **do NOT re-record the
  baseline under `--workspace`.** If the key-sort fix is correct, no re-record of any kind
  is needed — that is the fix's own acceptance test, not an inconvenience to work around.
- Budget: ~1 hour of compute on this host, then checkpoint with whatever is narrowed
  (CLAUDE.md escalation rule; this host is not lambda-vector).
- Verification discipline applies: capture `rc` before any pipe (never `$?` through a
  pipe); never label a run by intent — prove the feature actually engaged (`cargo tree -e
  features` on the exact scope, or a build-log/dispatch line), not what was requested.

## Open Questions (for the human, NOT for this session to decide)

(a) Which enabled feature moves the numerics? ← **this session's question**
(b) Should a consumer's forecast depend on the feature set of unrelated workspace members
    at all? ← policy decision, deferred; a "yes" makes the bisect moot, a "no" makes it
    mandatory.

## Current Focus

- hypothesis: CONFIRMED (and it is NOT a floating-point path). `pmcp v2.19.3` enables the
  cargo feature `serde_json/preserve_order`; feature unification applies it to
  `aprender-forecast`'s own `serde_json`, turning `serde_json::Map` from a key-sorted
  `BTreeMap` into an insertion-ordered `IndexMap`, which changes the order
  `invariance::json` hashes `components` / `diagnostics` in. Zero floats change.
- test: DONE. Both directions proven (evidence s6 / s8) plus an independent route (s9).
- expecting: MET. Exactly one edge — `serde_json/preserve_order` — flips
  `aa669c2352dd376a` -> `e01f605419765a6a`; enabling it alone FAILS, enabling everything
  else without it PASSES.
- next_action: CHECKPOINT — item (3)'s falsifiable acceptance test came back FALSE. Item (0)
  (instrument fix) is DONE and COMMITTED. Item (1) (blast radius) is DONE: all 8/8 cases
  diverge under `preserve_order` alone. Item (2) (key-sort fix) is APPLIED but UNCOMMITTED
  (working tree only, `crates/aprender-forecast/src/invariance.rs`) pending this decision —
  it is verified CORRECT for the `preserve_order`-only scope (scope A, s13: 9/9 pass) but
  verified INSUFFICIENT for the real consumer route (scope B, s14-s15: 5/8 Prophet-arm cases
  still fail, against a THIRD distinct hash — not the recorded baseline, not the pre-fix
  insertion-order hash either). Isolated the second cause to `serde_json/raw_value` (s15),
  enabled by both `axum v0.8.9` and `pmcp v2.19.3` (s16), mechanism UNCONFIRMED (s17 — no
  RawValue/arbitrary_precision usage in this crate's own code, so it is an internal
  serde_json behaviour difference gated by the feature, not yet pinned down). Items (4) and
  (5) are NOT started — held per the explicit STOP instruction until this decision resolves.
  DO NOT re-record the baseline. DO NOT assume the key-sort fix is wrong for what it does
  cover — it is confirmed correct and necessary for the preserve_order/ordering half; it is
  just not sufficient alone for the byte-identical acceptance criterion under the full
  real-world route.
- reasoning_checkpoint:
    hypothesis: "`pmcp v2.19.3` enables `serde_json/preserve_order`; cargo feature-unification applies it to aprender-forecast's own serde_json; `serde_json::Map` becomes an `IndexMap` instead of a `BTreeMap`; `invariance::json` walks `Value::Object` in MAP ITERATION ORDER, so `ForecastResponse.components` — filled from the ORDERED `Vec<(String, Vec<f64>)>` by `.insert()` in computation order — hashes in insertion order rather than key-sorted order, flipping the signature. No float changes."
    confirming_evidence:
      - "ENABLE: scoped build + ONLY `--features serde_json/preserve_order` -> rc=100, `recorded aa669c2352dd376a, got e01f605419765a6a` (3/3 retries, exact symptom hash)"
      - "DISABLE: the ENTIRE remaining delta (core 15 feats, compute gpu+wgpu+parallel+rayon, num-traits/libm, zstd, lz4, serde_json indexmap+raw_value+unbounded_depth) WITHOUT preserve_order -> rc=0, 9/9 PASS"
      - "INDEPENDENT ROUTE: `-p aprender-forecast -p aprender-mcp-forecast` (no apr-cli, numerics deltas all at baseline) -> same failure, same hash"
      - "`invariance.rs:22-25` states the assumption verbatim: '`serde_json::Map` is a `BTreeMap`, so object iteration is KEY-SORTED'. `types.rs:711` types `components` as `serde_json::Map`. `prophet.rs:810` types the source as an ordered `Vec<(String, Vec<f64>)>`."
    falsification_test: "If preserve_order were not the cause, enabling it alone in the passing scope would NOT reproduce the exact recorded/got hash pair, and/or enabling the whole remaining delta without it would still fail. Both were run; both came out the other way."
    blind_spots:
      - "The gate's `assert_eq!` is INSIDE the CASES loop, so it aborts at case 1 of 9. The true blast radius across the other 8 cases is unmeasured."
      - "24 of the 82 delta edges (bitflags/serde, rand/*, toml/*, syn/quote, serde_spanned, toml_datetime, serde_core, memchr, once_cell, chrono/serde, errno, libc/extra_traits, rustix, proc-macro2/span-locations, getrandom/wasm_js, hashbrown) were never individually tested. They are build-time or unrelated-IO edges and cannot reach the hashed response, but that is an argument, not a control."
      - "Only arch=aarch64 (this host) was exercised."
    candidate_causes:
      - "code: signature walk depends on map iteration order (CONFIRMED)"
      - "config/build: cargo feature unification makes a transitive dependency's cargo feature observable in an unrelated crate's behaviour (CONFIRMED — the delivery mechanism)"
      - "environment: kernel/SIMD dispatch, softfloat (ELIMINATED by control D)"
      - "data: input series shape (ELIMINATED — same inputs in both directions)"
    and_gate: "yes — two conditions must hold simultaneously. (1) `serde_json/preserve_order` is enabled somewhere in the build graph, AND (2) the signature hashes an insertion-ordered map whose insertion order differs from its key-sorted order. Neither alone is the defect: preserve_order is a legitimate choice by pmcp, and insertion-order-built maps are legitimate for the door. The defect is their conjunction, which is why the fix has two independent candidate sites."
- tdd_checkpoint:

## Evidence

- timestamp: 2026-09-22T-s1
  checked: Instrument confirmed on HEAD 41f6d7eb4. `cargo nextest run --profile ci -p aprender-forecast --lib invariance::`
  found: rc=0, `Summary [40.162s] 9 tests run: 9 passed, 245 skipped`
  implication: baseline PASS reproduces; the instrument is live on this tree state.

- timestamp: 2026-09-22T-s2
  checked: Deduped feature-edge delta re-derived with `cargo tree --format "{p}|{f}" --prefix none`, scoped vs `+apr-cli`, restricted to the 79 packages `aprender-forecast` links.
  found: **82 edges across 28 packages** — not ~190. CORRECTION 1 IN THIS FILE IS WRONG: `apr-cli` pulls **14** `aprender-core` edges (`default, dirs, format-compression, half, hf-hub, hf-hub-integration, lz4_flex, parallel, rayon, safetensors, safetensors-compare, sha2, ureq, zstd`), not 6. CORRECTION 2 IS WRONG: `safetensors-compare`, `half`, `hf-hub-integration` ARE in the delta. `aprender-core` is EMPTY in the scoped build (`default-features = false` in aprender-forecast's manifest), and a package whose scoped feature set is empty is dropped by a naive `{f}`-based package list — the likely source of both bad corrections.
  implication: the candidate surface is 82 edges, and `aprender-core` (the crate that actually runs the Prophet fit) has the largest jump.

- timestamp: 2026-09-22T-s3
  checked: Enablement of the ALREADY-RUN control `-p aprender-forecast -p aprender-core` (recorded PASS), via `cargo tree` on that exact scope.
  found: it enabled `aprender-core|default,parallel,rayon` AND `aprender-compute|default,parallel,rayon` AND `num-traits|default,i128,libm,std` — 35 of the 82 delta edges at once.
  implication: that single PASS exonerates far more than recorded: core's rayon reduction order, compute's rayon reduction order, AND the `num-traits/libm` softfloat swap. Remaining untested surface: **47 edges across 17 packages**.

- timestamp: 2026-09-22T-s4
  checked: `crates/aprender-forecast/src/invariance.rs` module docs (lines 22-25) and `fn json` (line 145).
  found: the signature walk hashes `serde_json::Value::Object` in MAP ITERATION ORDER, and the docs state the load-bearing assumption verbatim: "`serde_json::Map` is a `BTreeMap`, so object iteration is KEY-SORTED and the signature does not depend on insertion order. Both fields rely on that: the door builds `components` by inserting in computation order". `ForecastResponse.components` is `serde_json::Map<String, serde_json::Value>` (`types.rs:711`), filled from `prophet::Forecast.components: Vec<(String, Vec<f64>)>` (`prophet.rs:810`) — an ORDERED VEC — by `components.insert(...)` (`forecast.rs:979`).
  implication: `serde_json/preserve_order` swaps that Map's backing store from `BTreeMap` to `IndexMap`, making iteration INSERTION-ordered. That falsifies the stated assumption and changes the hash WITHOUT changing a single float. `serde_json: preserve_order` is in the remaining 47-edge delta. The "byte-level FP signature change" framing in this file's symptom section is a WRONG PREMISE — this is a serialisation-ordering defect, not a numerics defect, which is why every numerics control passed.

- timestamp: 2026-09-22T-s5
  checked: `cargo tree -e features -p aprender-forecast -p apr-cli -i serde_json`
  found: `serde_json feature "preserve_order"` has exactly one enabler in the delta scope: **`pmcp v2.19.3`** (the MCP SDK), reached through `apr-cli`.
  implication: the edge is `pmcp -> serde_json/preserve_order`, unified onto aprender-forecast's own serde_json by cargo's feature unification.

- timestamp: 2026-09-22T-s6
  checked: ONE-VARIABLE ENABLE CONTROL. `cargo nextest run --profile ci -p aprender-forecast --lib --features serde_json/preserve_order invariance::`. Enablement proven first by `cargo tree` on that exact scope: identical to baseline in every package EXCEPT `serde_json` gains `indexmap,preserve_order`; `aprender-core` stays EMPTY, `aprender-compute` stays `default`.
  found: **rc=100. `case peyton/prophet/default on arch=aarch64 no longer reproduces its pre-change signature: recorded aa669c2352dd376a, got e01f605419765a6a`** — the EXACT hash pair, the EXACT case, the EXACT arch from the `--workspace` symptom. Byte-identical across 3/3 nextest retries.
  implication: ROOT CAUSE. One feature edge, no numerics touched, reproduces the reported failure exactly.

- timestamp: 2026-09-22T-s7
  checked: `invariance.rs:709-725` — the gate body.
  found: the `assert_eq!` is INSIDE the `for (label, csv, horizon, shape) in CASES` loop, so the test panics at the FIRST mismatching case and never evaluates the remaining 8.
  implication: "only `peyton/prophet/default` flips" is NOT established by the evidence — it is the first case in `CASES` order. The blast radius is unknown and is very likely larger (the `holidays+windows` case inserts `trend, <seasonalities>, <event names>, holidays`, whose insertion order diverges from key-sorted order even more). Do not report this as a single-case defect.

- timestamp: 2026-09-22T-s8
  checked: ONE-VARIABLE DISABLE CONTROL ("everything else, without preserve_order"). `cargo nextest run --profile ci -p aprender-forecast -p aprender-core -p aprender-compute --features aprender-core/safetensors-compare,aprender-core/format-compression,aprender-core/format-quantize,aprender-compute/gpu,aprender-compute/wgpu,serde_json/indexmap,serde_json/raw_value,serde_json/unbounded_depth --lib invariance::`. Enablement proven by `cargo tree` on that exact scope: `aprender-compute|bytemuck,default,futures-intrusive,gpu,parallel,pollster,rayon,wgpu`, `wgpu v27.0.1|default,dx12,gles,metal,parking_lot,std,vulkan,web,webgpu,wgsl`, `aprender-core` at 15 features (superset of its 14 delta edges), `num-traits|default,i128,libm,std`, `half|num-traits`, `zstd|arrays,default,legacy,zdict_builder`, `serde_json|alloc,default,float_roundtrip,indexmap,raw_value,std,unbounded_depth` — `preserve_order` ABSENT.
  found: **rc=0, `9 tests run: 9 passed, 18071 skipped`.**
  implication: the ENTIRE remaining numerics + compression + serde surface is enabled and the gate is GREEN. `aprender-compute/gpu` and `/wgpu` — this file's leading hypothesis — are ELIMINATED with proven enablement. `serde_json/preserve_order` is not merely sufficient; within the delta it is necessary.

- timestamp: 2026-09-22T-s9
  checked: INDEPENDENT-ROUTE VARIATION (anti-anecdote). `cargo nextest run --profile ci -p aprender-forecast -p aprender-mcp-forecast --lib invariance::` — a thin pmcp server crate, NO `apr-cli` anywhere. `cargo tree` on that exact scope: `aprender-core|` (EMPTY, as baseline), `aprender-compute|default` (as baseline), `num-traits|std` (as baseline — NO libm), `serde_json|alloc,default,float_roundtrip,indexmap,preserve_order,raw_value,std`.
  found: **rc=100, `recorded aa669c2352dd376a, got e01f605419765a6a`** — the same hash pair, reached through a completely different package, with ZERO numerics-feature deltas.
  implication: the cause is not `apr-cli` and not any numerics feature. It is `pmcp -> serde_json/preserve_order`, reachable from every crate in this workspace that links the MCP SDK. The symptom title "feature-unification-divergence" is right; the mechanism paragraph's "FP path" premise was wrong.

- timestamp: 2026-09-22T-s10
  checked: FIX_AND_VERIFY item (0): restructured `every_baseline_case_reproduces_its_signature`
    (invariance.rs:709-733) to accumulate every case's outcome across the full loop and
    assert once at the end, instead of `assert_eq!` inside the loop. Control run afterward:
    `cargo nextest run --profile ci -p aprender-forecast --lib invariance::` (the PASSING
    scope, unchanged).
  found: rc=0, 9 tests run: 9 passed, 245 skipped. The instrument change does not alter the
    passing scope's outcome.
  implication: instrument repaired without regressing the known-good scope. Safe to measure
    blast radius under the failing scope next.

- timestamp: 2026-09-22T-s11
  checked: FIX_AND_VERIFY item (1): true blast radius under the failing scope with the
    repaired instrument. `cargo tree --format "{p}|{f}" -p aprender-forecast -e features
    --features serde_json/preserve_order` proved `serde_json|...,indexmap,preserve_order,...`
    engaged first. Then `cargo nextest run --profile ci -p aprender-forecast --lib --features
    serde_json/preserve_order invariance::` (3/3 retries, byte-identical).
  found: rc=100, **8 of 8 cases** diverge, not 1: peyton/prophet/default
    (aa669c2352dd376a->e01f605419765a6a), air/prophet/multiplicative
    (3053244dcb27492c->5c62b8dbc74c8eec), retail/prophet/default
    (73b171523eb3fa3b->a1b174048ca054a9), wp_log_R/prophet/logistic
    (331d67a8f8c924bb->47cbc9b90e12349f), peyton/prophet/holidays+windows
    (e0a13f9a10e3c3cc->9b781663ebda3f9c), peyton/neuralprophet/lag0
    (d5a583795591a3db->859053930e5b34e9), peyton/neuralprophet/lag7
    (726043759dde45dc->9117b619c2755be2), wp_log_R/neuralprophet/lag0
    (72528357591ad4d6->01d7e106ca4a6f16).
  implication: CONFIRMS the s7 blind spot was real and material — "only
    peyton/prophet/default is affected" was an early-termination artifact of the old
    in-loop assert, not a finding about scope. Every door case's `components`/`diagnostics`
    JSON walk is affected, including the NeuralProphet arm (which does not go through
    `prophet.rs` at all), meaning the divergence is general to `invariance::json`'s Object
    arm rather than specific to one shape.

- timestamp: 2026-09-22T-s12
  checked: FIX_AND_VERIFY item (2): applied the human-selected fix — explicit key-sort in
    `invariance::json`'s `Object` arm (collect entries, `sort_unstable_by` on the key,
    iterate sorted) — plus updated the module docs (lines 22-25) and the arm's own comment
    to state the property is now ENFORCED, not inherited. `cargo check -p aprender-forecast
    --lib` -> rc=0. `cargo clippy -p aprender-forecast --lib -- -D warnings` -> rc=101, but
    grep confirms ZERO findings mention `invariance.rs`; every reported error is in
    aprender-compute (pre-existing dead-code/unreachable-expression tech debt, untouched by
    this session, confirmed via `git status --short crates/aprender-compute`).
  found: clean compile, clean clippy on the changed file; pre-existing unrelated warnings
    elsewhere in the dependency graph are not new.
  implication: the fix as written is syntactically and lint-clean. Proceeding to item (3),
    the acceptance test.

- timestamp: 2026-09-22T-s13
  checked: FIX_AND_VERIFY item (3), SCOPE A — `cargo nextest run --profile ci -p
    aprender-forecast --lib invariance::` (the baseline/passing scope, preserve_order NOT
    engaged here).
  found: rc=0, `9 tests run: 9 passed (1 leaky), 245 skipped`.
  implication: PREDICTION HOLDS for scope A — sorting explicitly reproduces the same order
    BTreeMap already gave, so all 8 recorded hashes are unchanged.

- timestamp: 2026-09-22T-s14
  checked: FIX_AND_VERIFY item (3), SCOPE B — the real consumer route, `-p aprender-forecast
    -p aprender-mcp-forecast --lib invariance::`. Engagement proven FIRST: `cargo tree
    --format "{p}|{f}" -p aprender-forecast -p aprender-mcp-forecast -e features | grep
    serde_json` -> `serde_json v1.0.150|alloc,default,float_roundtrip,indexmap,
    preserve_order,raw_value,std` — note **`raw_value` is ALSO enabled here**, which no
    prior evidence entry (s5-s9) singled out or controlled for; every prior control isolated
    `preserve_order` alone.
  found: **rc=100, 5 of 8 cases fail** (all 5 Prophet-arm cases; all 3 NeuralProphet cases
    now PASS): peyton/prophet/default `aa669c2352dd376a`->`8cd3c2517e29c2dc`,
    air/prophet/multiplicative `3053244dcb27492c`->`b9ff8b72b4c01394`,
    retail/prophet/default `73b171523eb3fa3b`->`f360e41a2ede7f45`,
    wp_log_R/prophet/logistic `331d67a8f8c924bb`->`5b453bba2f386401`,
    peyton/prophet/holidays+windows `e0a13f9a10e3c3cc`->`1addb3f8b30e323a`. Every "got" value
    is a THIRD distinct hash — neither the recorded baseline nor the pre-fix
    (unsorted/insertion-order) "got" value from s6/s9.
  implication: **THE PREDICTION IS FALSIFIED.** The key-sort fix does not make scope B
    byte-identical to the recorded baseline. Something beyond map-ordering is now in play,
    isolated to the Prophet arm specifically (NeuralProphet is clean).

- timestamp: 2026-09-22T-s15
  checked: ISOLATION CONTROL, run to differentiate "it's raw_value" from "it's something
    else in aprender-mcp-forecast's dependency graph". `cargo tree --format "{p}|{f}" -p
    aprender-forecast -e features --features serde_json/preserve_order,serde_json/raw_value`
    proved ONLY `serde_json` gains `raw_value` on top of the already-controlled
    `preserve_order` (no apr-cli, no aprender-mcp-forecast, no other package touched). Then
    `cargo nextest run --profile ci -p aprender-forecast --lib --features
    serde_json/preserve_order,serde_json/raw_value invariance::`.
  found: rc=100, **byte-identical to s14** — same 5 cases, same exact "got" hash values
    (`8cd3c2517e29c2dc`, `b9ff8b72b4c01394`, `f360e41a2ede7f45`, `5b453bba2f386401`,
    `1addb3f8b30e323a`), same 3 NeuralProphet cases passing.
  implication: **ISOLATED.** `serde_json/raw_value` alone, combined with
    `serde_json/preserve_order`, fully reproduces scope B's failure with zero contribution
    from apr-cli, axum-the-HTTP-layer, or any other package. This is a SECOND, INDEPENDENT
    divergence mechanism the key-sort fix does not address.

- timestamp: 2026-09-22T-s16
  checked: enabler chain for the new edge. `cargo tree -e features -p aprender-forecast -p
    aprender-mcp-forecast -i serde_json | grep -A2 'feature "raw_value"'`.
  found: `serde_json feature "raw_value"` has TWO enablers in this scope: **`axum v0.8.9`**
    and **`pmcp v2.19.3`**. Both reach `aprender-mcp-forecast` directly (no `apr-cli` needed
    — same blast-radius shape as `preserve_order`).
  implication: a second transitively-unified serde_json feature, delivered by the same
    mechanism (cargo feature unification via the MCP/HTTP stack) as the first.

- timestamp: 2026-09-22T-s17
  checked: mechanism search for WHY `raw_value` would move a hash. `grep -rn
    "serde_json::from_str|from_str::<serde_json|RawValue|arbitrary_precision"
    crates/aprender-forecast/src/` (excluding test_support/invariance.rs).
  found: zero matches. `aprender-forecast`'s own Prophet path does not use `RawValue`,
    `arbitrary_precision`, or any `from_str` JSON round-trip.
  implication: the mechanism is NOT in this crate's own code — it must be an internal
    `serde_json` behaviour difference gated by the `raw_value` cargo feature itself (e.g. a
    different number/float encoding code path enabled internally alongside raw-value
    support), affecting values built directly via `Value::Number`/`json!` without any
    explicit RawValue usage. NOT YET CONFIRMED — this is inference from a negative grep, not
    a positive mechanism finding. Root cause of the SECOND divergence is UNRESOLVED.

## Eliminated

- hypothesis: rayon reduction order in `aprender-core` — control `-p aprender-forecast -p aprender-core --lib` PASS
- hypothesis: `aprender-compute` `parallel`+`rayon` — control with `--features aprender-compute/parallel` PASS
- hypothesis: `half` → `num-traits/libm` softfloat swap — control `-p aprender-forecast -p aprender-quant --lib` PASS
- hypothesis: CPU load / cooperative `FIT_BUDGET_SECS` timeout — PASS under saturating 14-core load
- hypothesis: test flakiness — byte-identical hash on 3/3 retries in both directions
- hypothesis: `aprender-compute/gpu` + `/wgpu` kernel dispatch (this file's leading candidate)
  evidence: control D enabled `gpu,wgpu,bytemuck,futures-intrusive,pollster` on aprender-compute AND `wgpu v27.0.1|...,metal,...` (enablement proven by `cargo tree` on that exact scope) — PASS 9/9, rc=0
  timestamp: 2026-09-22
- hypothesis: any numerics feature at all (reduction order, SIMD width, BLAS/GEMM path, softfloat)
  evidence: control D enabled the FULL numerics delta (`aprender-core` 15 features incl. parallel/rayon/half, `aprender-compute` all 8, `num-traits|default,i128,libm`) — PASS 9/9. Control F reproduced the FAILURE with the numerics delta at BASELINE values (`aprender-core|` empty, `aprender-compute|default`, `num-traits|std`).
  timestamp: 2026-09-22
- hypothesis: `aprender-core`'s 14-feature jump (safetensors-compare, hf-hub-integration, format-compression, half, ...)
  evidence: control D enabled a superset of all 14 — PASS 9/9
  timestamp: 2026-09-22

## Resolution

**RESOLVED 2026-09-22.** Two commits, both verified; no baseline re-recorded.

- root_cause: |
    `serde_json/preserve_order`, enabled transitively by `pmcp v2.19.3` and applied to
    `aprender-forecast`'s own `serde_json` by cargo feature unification, swaps
    `serde_json::Map`'s backing store from `BTreeMap` (key-sorted iteration) to `IndexMap`
    (insertion-ordered). The signature hashed maps in raw iteration order, so it inherited
    its ordering from a dependency's feature flags. NOT a numerics defect: no float changes,
    which is why all seven numerics controls passed.
    TWO sites needed the sort, and missing the second is what made a partial fix look like a
    second cause:
      1. `json`'s `Object` arm  -> covers `diagnostics` and nested objects.
      2. `signature_with`'s `components` loop -> walks the top-level `serde_json::Map`
         DIRECTLY, so it never inherited the `Object` arm. The door inserts those keys in
         prophet's computation order (`additive_terms`, `multiplicative_terms`, `holidays`,
         seasonality names), which is not sorted order. Fixing only site 1 left all five
         prophet cases diverging and repaired the three neuralprophet ones.
- fix: |
    `b7317854c` - instrument first: move `assert_eq!` OUT of the `for .. in CASES` loop so
                  every case is evaluated. This had to land before any hash changed.
    `e920b4e36` - sort object keys at BOTH sites. Sorting by `String`'s `Ord` reproduces
                  `BTreeMap`'s own iteration order, so no recorded baseline changes.
    `5ede4a0c4` - drop the unanchored `debug` rule from `.gitignore` (CB-510 shape; it hid
                  this session file and would have hidden any `crates/*/src/debug/`).
- verification: |
    Feature engagement proven per scope with `cargo tree --format "{p}|{f}"` BEFORE each run
    (`cargo tree -e features` is the WRONG instrument here: it renders only edge-enabled
    features, so a command-line `--features` is invisible to it and it reports an identical
    set with and without the flag).
      A  -p aprender-forecast                                    9/9 rc=0  (BTreeMap)
      B  -p aprender-forecast --features serde_json/preserve_order
                                                                 9/9 rc=0  (indexmap,preserve_order; raw_value ABSENT)
      C  -p aprender-forecast -p aprender-mcp-forecast           9/9 rc=0  (+raw_value engaged)
      D  -p aprender-forecast -p apr-cli                         9/9 rc=0  (the original flip)
      E  --workspace --lib, CI's own scope (ci.yml:289)          9/9 rc=0
      F  clippy --no-deps -D warnings rc=0; cargo fmt --check rc=0
    Blast radius, measured with the repaired instrument: 8 of 8 cases, not 1. The original
    "only peyton/prophet/default" was an early-termination artifact.
    Refuted mid-session: a `serde_json/raw_value` second cause. Same divergence reproduces
    with raw_value provably absent (B), and scope C passes with it engaged. It had been
    attributed on one isolation control where both features moved together.
- files_changed:
    - crates/aprender-forecast/src/invariance.rs
    - .gitignore
    - .planning/phases/06.1-forecast-exogenous-inputs-prophet-regressors-neuralprophet-e/deferred-items.md
- still_open: |
    Whether anything ELSE in the forecast path is feature-set-sensitive is untested. For this
    gate, question (b) is now answered structurally rather than by policy: the signature
    enforces its own ordering instead of inheriting one.
    The human declined a regression guard asserting the property; nothing stops a later
    revert of the sort.
