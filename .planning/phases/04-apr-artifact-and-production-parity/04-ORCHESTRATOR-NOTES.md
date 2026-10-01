# Phase 4 — Orchestrator Notes

Cross-plan findings surfaced during execution that no single plan owns. Each is verified,
not reported second-hand. Wave-8 (04-10) and wave-9 (04-11) executors MUST read this.

---

## F-01 — `contracts/aprender/binding.yaml` is an ORPHAN file (BLOCKING for 04-10/04-11)

**No plan declares it in `files_modified`; no plan even mentions it.** Verified by scanning the
frontmatter of all 16 plans. Yet:

- 04-01 had to edit it (undeclared, but measured as required: rc=1 with 15 `BIND-001` errors
  without the entries, rc=0 with 15 `BIND-004` warnings after).
- 04-02 deferred a status flip to it because it is shared across concurrently-running agents.

All 15 equations are `status: pending`. `make contract-audit-phase4` currently exits 0 with 15
`BIND-004` warnings (pending is a WARNING, not an error). As plans land, these statuses drift
further from reality, and 04-11's closing requirements audit would report 15 pending equations
for work that is largely done.

**Do NOT blind-flip these.** Verified as of wave 2 (`f91d79279`):

| equation | bound function | state |
|---|---|---|
| `artifact_storage_map` | `write_setfit_apr` | `pub fn`, artifact.rs:576 — real |
| `canonical_tensor_names` | `canonical_name_for_hf` | `pub fn`, artifact.rs:966 — real |
| `architecture_derived_tensor_set` | `expected_tensor_names` | `pub fn`, artifact.rs:950 — real |
| `nullable_path_allowlist` | `first_unallowed_null_path` | **PRIVATE** fn, artifact.rs:1083 |

The fourth is private, so flipping it to `implemented` asserts something a reverse-coverage gate
may not resolve. Its equation also spans two crates (the walk in 04-02, the completeness gate in
04-13), so a single binding row under-describes it.

**Action for 04-10:** add `contracts/aprender/binding.yaml` to `files_modified`, take ownership of
status flips, and decide per-equation whether the bound function needs `pub` (or `pub(crate)` plus
a `--crate-dir` reverse-coverage run) before claiming `implemented`.

---

## F-02 — `pv lint <file>` is a VACUOUS gate

Independently confirmed with the pinned binary (`./target/release/pv`):

- `pv lint contracts/setfit-apr-v1.yaml` → `Summary: 0 errors, 0 warnings, 0 suppressed` /
  `Result: PASS`, with gate 7 skipped and gate 8 reporting `0 edges`. It validates nothing.
- `pv lint contracts/` → `Summary: 0 errors, 1010 warnings, 0 suppressed, 73 new`.

`pv validate <file>` IS a real gate (`0 error(s), 0 warning(s) / Contract is valid.`).

**Never cite `pv lint <file>` as evidence.** Use `pv validate <file>` or `pv lint <dir>`.
`pv` is not on PATH — use `./target/release/pv`, pinned.

---

## F-03 — the plans' clippy command is a gate that can never pass

`cargo clippy -p aprender-train --features setfit -- -D warnings` → **rc=101**, with 23 warning
lines from `crates/aprender-compute/src/blis/*` and zero findings in the code under test.
It fails on a dependency's pre-existing debt, so it cannot distinguish "my code is clean" from
"never linted."

`cargo clippy -p aprender-train --features setfit --no-deps -- -D warnings` → **rc=0**.

04-13 proved `--no-deps` is still a working gate by planting a deliberate `useless_format` probe
and observing RED. Both 04-13 and 04-14 hit this independently.

**REFINEMENT (verified after wave 4, from 04-04's D-04-04-A):** `--no-deps` rescues
`aprender-train` but **NOT `aprender-core`**, which has a pre-existing error of its own:

```
cargo clippy -p aprender-core --features setfit --no-deps -- -D warnings   -> rc=101
error: unreachable expression
  --> crates/aprender-core/src/demo/reliable/performance.rs:126:5
```

It is **arm64-only and pre-existing**: on aarch64 the `#[cfg(target_arch = "aarch64")] { return
"NEON" }` makes the trailing `"Scalar".to_string()` unreachable. Last touched by `bb519c6eb`
(APR-MONO), **zero** phase-4 commits touch that file. So on this host `aprender-core` exits 101 on an
empty diff — the gate cannot distinguish clean from dirty there either.

04-04's workaround is the right shape: a **path-scoped** assertion over the setfit surface only,
proven non-vacuous by planting a `useless_format` (0 → 1). Do not scope by crate.

**Action for 04-10:** every clippy leg must carry `--no-deps`, AND `aprender-core` legs must
additionally be path-scoped to the setfit surface or they fail on unrelated pre-existing arm64 debt.
Do not "fix" either by removing `-D warnings`.

---

## F-04 — zero-match test filters exit 0 (CR-02 vacuity), hit three times

1. 04-13: `setfit::bundle::bundle_nullable` matched **zero** tests and exited 0. `bundle_tests.rs`
   is included via `#[path]` inside `bundle.rs`, so the real module path is
   `setfit::bundle::bundle_tests::…`. A mutation that should have turned the suite red reported success.
2. Orchestrator (me): `cargo test -p aprender-core --lib setfit::artifact::` returned
   `0 passed … 14185 filtered out`, rc=0 — because I omitted `--features setfit`. With the feature
   the same filter returns 41. **A feature-gated module yields a vacuous green when the feature is off.**
3. This is exactly the class 04-10's `assert_tests_ran` guards exist to catch.

**Action for 04-10:** every guarded target must assert a stated minimum count AND carry the feature
flag its module needs. A filter is only proven by observing a non-zero count.

---

## F-05 — a `skip_serializing_if` gate must match the attribute, not the token

The bare token count in the setfit directories moved 0 → 4 (04-13) and is 3 in `artifact.rs`
(04-02). **Every one is prose** — doc comments and an assertion message explaining that the
attribute is forbidden. The attribute form (`serde(skip_serializing_if`) is still 0.

A future gate on the bare token turns red on its own documentation. Match `serde(skip_serializing_if`.

---

## F-06 — contract amendment: six HF templates have no canonical name (from 04-01)

Six of 21 HF name templates have no `tensor-names-v1` canonical form — four have no role at all,
and `position_embedding` has a `bert:` alias with an **empty `_fallback`**. On the pinned model that
is **22 of 101 encoder tensors with no name to write**. The contract now RESERVES those six, with
non-collision enumerated against the complete `_fallback` set, recorded as a D-01 amendment plus a
deferred upstreaming item. Plans written before wave 1 do not know this — read the contract, not the
plan text, for the reserved set.

---

## F-07 — `bashrs` is not installed on this host

04-01 recorded `bashrs make lint Makefile` as an **unrun check, not a passed one**. CLAUDE.md
mandates bashrs over shellcheck for shell/Makefile linting. Any plan asserting a bashrs gate must
either install it or record it unrun — never report it green.

---

## F-08 — intermediate commit `ab7ace94a` does not compile in isolation

Verified against the committed blob, not assumed: `close` passes 6 args to the now-7-param
`from_run_parts` (E0061). A signature change and its call sites cannot be split across commits and
still compile; `59a4eca93` restores it. The two always land together. Relevant only to `git bisect`
across that pair.

---

## F-09 — there is NO active pre-commit hook; "hooks run by default" is a false assumption

04-03 reported "the active pre-commit hook ran a failing test suite and did not block, and one of
its lines is `command not found: --features`." I investigated from the main checkout. The reality
is stronger than the diagnosis:

```
git config core.hooksPath        -> unset
git rev-parse --git-path hooks   -> .git/hooks
.git/hooks/pre-commit            -> DOES NOT EXIST (only *.sample)
```

**No pre-commit hook is active in this repository at all.** The repo ships `.githooks/pre-commit`,
but it requires `git config core.hooksPath .githooks` to take effect, and that is unset. No Claude
Code hook intercepts commits either (the configured hooks are GSD SessionStart/PostToolUse infra;
none contain `--features`).

Consequence: the standing executor instruction — *"Run `git commit` normally — hooks run by default.
Do NOT pass `--no-verify`"* — has been providing **no gate whatsoever** for every commit in this
phase. Nothing was bypassed and nothing is wrong with the commits; the per-commit verification in
this phase came entirely from executors running their suites explicitly, which they did. But the
belief that a hook was also checking was unfounded. This is the "a guard that does not run is
theater" class from CLAUDE.md, in the workflow's own scaffolding.

**Do not simply activate it.** Per F-03, `.githooks/pre-commit` runs `cargo clippy -- -D warnings`
workspace-wide, which exits 101 on pre-existing `aprender-compute` debt — activating it today would
block every commit immediately. Activation requires fixing that debt or scoping the hook first.
This is a repo-level decision for the human, not a phase-4 change.

---

## F-10 — the phase-3 slice fixture CANNOT carry a setfit-apr-v1 artifact (BLOCKING for 04-12)

Measured by 04-05, not predicted:

```
ProbeComputation { probe: "probe_unicode",
    reason: "SetFitError::VocabOutOfSlice(canonical id 5915 is outside the slice closure)" }
```

The slice fixture's `vocab_remap` is a **97-row closure** and it declares **64 position rows** against
a **256-token** truncation probe. Neither is a production defect — the real pin has the full
vocabulary and 512 positions — but the probes in the contract exercise ranges the slice does not have.

Worse for anyone trying to route around it: **`tune_encoder` gates on
`encoder.architecture_fingerprint()`, so no synthetic encoder can reach `HeadFitted` through the
shipped transitions either.** Both obvious paths are closed.

04-05's Task 2 solved it by substituting **the encoder and head only**, keeping the dataset,
selection, config, evidence and the entire trusted verify policy real, and kept the finding
executable as a test rather than a comment.

**Action for 04-12 (OPS-01, wave 5):** a plain `fx::head_fitted_run(..)` **will not close to APR**.
Reuse 04-05's substitution shape. Read `04-05-SUMMARY.md` before writing the lifecycle test.

---

## F-11 — do NOT use `head`/`tail` to produce file content in this environment

04-05 hit this live: `head -N file > file2` produced a **682-line** file when **1021** lines were
requested. `head` output is filtered by the RTK proxy, so a redirect captures the *rendered view*,
not the bytes. Caught and reverted within one command; nothing committed was affected.

This generalizes the user's global CLAUDE.md guidance ("avoid using this tool to run cat/head/tail")
into a correctness hazard, not just a token-efficiency preference: **any `head`/`tail`/`cat` redirect
can silently truncate.** Use `Read`, or `sed -n '1,Np'`, or Python for byte-exact extraction.

---

## D-04-04-B — RESOLVED: the 24 `aprender-train` failures are pre-existing and unrelated

04-04 observed 24 failures and correctly declined to call them pre-existing without measuring.
Measured on the merged wave-4 tree (`e21f108dd` + recovery merge):

```
test result: FAILED. 7882 passed; 24 failed; 15 ignored
failure set = 21 x gpu::*  +  3 x prune::*   ->  ZERO setfit failures
```

04-05 independently `diff`ed the failure set against the phase-3 known-red baseline: `DIFF_RC=0`,
zero new. Confirmed pre-existing, in subsystems this phase does not touch.

---

## F-12 — ORCHESTRATOR SELF-CORRECTION: `--ff-only` after the base moves, plus unconditional cleanup

I captured `EXPECTED_BASE` for wave 4, then committed the F-03 refinement on top of it before the
agents returned. `git merge --ff-only <04-04 branch>` therefore exited **128** ("Not possible to
fast-forward") — correctly, since the branch had diverged — and my cleanup loop then deleted the
branch **without checking the merge status**, so 04-04's four commits were briefly unreferenced.

Recovered in full by merging the dangling commit `b943116e0` directly (7 files, 0 conflicts,
`classify.rs` and `04-04-SUMMARY.md` restored, verified reachable).

**Rules for the remaining waves:** capture `EXPECTED_BASE` and then do not commit to the branch
until the wave's agents have returned and merged; use `git merge --no-edit` (not `--ff-only`) for
wave merges; and never delete a worktree branch without first asserting `git merge-base --is-ancestor
<branch> HEAD`.

---

## F-13 — wave-4 regression: 04-05 broke the trybuild ui suite and did not run it

Found while verifying an unrelated `/simplify` pass, and proven pre-existing by
stashing those changes and re-running on the clean tree (`rc=101`, identical single
mismatch). **Not caused by the simplify work.**

**Cause.** 04-03 added `tests/ui/setfit_external_codec_impl.rs` and blessed its `.stderr`
when `SerdeJsonCodec` was the ONLY `sealed::Sealed` implementor. 04-05 then added
`impl sealed::Sealed for AprCodec` (`apr_codec.rs:86`), so rustc's help text changed from

```
help: the trait `verify::sealed::Sealed` is implemented for `SerdeJsonCodec`
```

to `help: the following other types implement trait …` listing both. The snapshot went stale.

**Why it escaped.** 04-05's SUMMARY lists every command it ran — `--lib`, `setfit::`,
`setfit::apr_codec::`, `::bijection`, `::round_trip`, clippy, fmt. **`--test ui` is not among
them.** 04-03 ran it (`8 of 8 cases`); 04-05 changed a type that the ui suite observes and
never re-ran it. A plan that adds an implementor of a sealed trait must re-run the suite that
snapshots that trait's diagnostics.

**Fix applied:** regenerated via `TRYBUILD=overwrite`. The diff is purely the implementor list;
the `error[E0277]: the trait bound 'MyCodec: verify::sealed::Sealed' is not satisfied` refusal is
unchanged, so the sealing property is still proven, not weakened. Verified 8 case files on disk
and 8 cases reported ok — not a vacuous harness pass.

**Action for 04-10 (wave 8):** the Make gates must include `-p aprender-train --features setfit
--test ui` with a case-count assertion (`>= 8`), not just a `rc=0` check. A trybuild harness
reports `1 passed` for the whole suite, so `test result: ok` alone cannot distinguish 8 cases
from 0.

---

## F-14 — code-review: five findings deferred to a phase-owner decision (BLOCKING for 04-10/04-11)

Ten `/code-review` findings were fixed in `bbc631ef7`. These five were correctly NOT fixed because
each needs a decision, not a patch. **04-10 and 04-11 own them.**

**1. The loader's rung numbering is off by one from the contract.** `contracts/setfit-apr-v1.yaml`
(`load_validation_ladder`, ~:983-991) defines EIGHT rungs — 1 bounded read, 2 raw length, 3
container/CRC, 4 typed tag+doc, 5 structural, 6 non-finite scan, 7 rebuild, 8 probe replay — and
its postcondition says *a refusal names the RUNG*. The code is `rung1_raw_length` ..
`rung7_replay_probes`, and the typed refusals say `InconsistentTensor(rung 4…)` for a corrupt
tensor index, which the contract calls typed-tag/document-parse. An operator reading a refusal and
looking it up gets the wrong subsystem. The contract also asserts `VerifiedSetFitModel exists only
after all EIGHT rungs passed` — unsatisfiable with seven. **Decide: amend the contract to seven, or
renumber the code to eight.** No CI gate compares them, which is why it survived. NOTE: my
`/simplify` pass relabelled the WRITER's helpers from "Rung N" to "Step N" for a different reason
(the writer and loader gave the same numbers to different checks); that change is unrelated to and
does not fix this one.

**2. All 15 `setfit-apr-v1` binding rows are still `status: pending`** — see F-01, now confirmed by
review. `binding.yaml:1352-1356` states the rule in its own words: *"A plan that lands a module and
leaves its binding `pending` has recorded a claim nothing checks."* `contract-audit-phase4`
tolerates BIND-004 (pending) and fails only on BIND-001 (missing), so it passes green over a
registry tracking nothing. **Two rows name symbols that do not exist**: `probe_policy -> function:
replay_probes` (the function is `rung7_replay_probes`, and it is private) and `backend_identity ->
aprender::setfit::classify::backend_identity` (no such item — identity comes from
`ExecutionBackend::identity` in `encoder.rs`). Flipping either today would FAIL the audit.

**3. `ClassifyResponse` derives `PartialEq`, which compares `latency_ms`.** The contract (yaml:805,
:822) makes it normative that latency *"participates in no equality comparison"*. 04-09's parity
harness is this envelope's intended consumer, and the obvious `assert_eq!(cli, http)` is a
guaranteed red; the workaround (field-by-field) puts the exclusion rule back in the harness where
the contract says it must not live. **Decide: a manual `PartialEq`, or a dedicated `agrees_with`
comparator.** Not auto-fixed because removing latency from `PartialEq` also weakens the existing
serde round-trip assertion at `classify.rs:1898`.

**4. `ClassifyRequestDocument` deserializes an unbounded `texts` vector.** `MAX_BATCH_TEXTS` is
checked only AFTER the whole document is parsed and allocated — the module's own T-04-11 rationale
("a batch bound checked after tokenization is not a bound on the work an attacker can request")
applies one layer up and is not honoured. A body with 10,000,000 strings is fully materialised
first. **04-08's HTTP surface will hand this type an attacker-controlled body.** The review added
and re-exported `MAX_REQUEST_BODY_BYTES` (the contract bound at yaml:807-809 that had NO code
binding at all), but **enforcement is still owed by 04-07 (`--input` reader) and 04-08 (body
extractor)** — before deserializing, not after.

**5. `read_setfit_apr_bytes_bounded` pre-reserves the caller-declared length clamped to 256 MiB.**
A source that lies upward (a one-byte FIFO, a sparse file, metadata reporting 268,435,456) makes
the process commit a quarter gigabyte before reading a byte — the exhaustion the cap exists to
prevent, arriving via the reservation instead of the payload. Fixing it means inventing a
reservation ceiling the contract does not fix. **Decide with the bound's owner.**


---

## F-14 UPDATE — three of the five closed in `fb7904bad`

**CLOSED (1) rung renumbering.** Code now follows the contract's eight rungs
(`rung2_raw_length` .. `rung8_replay_probes`), with contract rung 1 — the declared-length source
bound — documented as living in `read_setfit_apr_bytes_bounded` rather than left as a silent gap.
Pinned by `the_rung_numbering_matches_the_contracts_eight_rung_ladder`, which parses the contract's
own `rungs:` block via `include_str!` (not a hand-copied list) and was shown RED on a real rename
before being believed. It scans only the production half — its first version failed on its own
`fn rung1_` literal, the F-05 self-scan defect, caught and fixed.

**CLOSED (3) `latency_ms` out of equality.** Manual `PartialEq` over schema version, artifact hash,
backend and results. `Eq` deliberately not implemented (`f64` probabilities). The two round-trip
tests that had been leaning on equality to cover latency now assert it separately and BY BITS, so a
renormalized `-0.0` cannot pass. **04-09's parity harness can now use `assert_eq!` as intended.**

**PARTIALLY CLOSED (4).** Research first: `max_request_body_bytes` appears in NO other contract, and
no crate enforces a body limit anywhere (no `DefaultBodyLimit`, no body-limit layer). So this
constant ESTABLISHES the number rather than inheriting a pattern. `MAX_REQUEST_BODY_BYTES` is the
shared value; **enforcement before deserializing is still owed by 04-07 (`--input` reader) and
04-08 (HTTP body extractor)** — the type still parses an unbounded `texts` vector.

**STILL OPEN (2) binding statuses** — all 15 rows remain `pending`; `contract-audit-phase4` is
green over a registry tracking nothing. One of the two bad rows is now fixed: `probe_policy` named
`replay_probes`, a function that never existed on this branch, corrected to `rung8_replay_probes`.
The other remains: `backend_identity` names
`aprender::setfit::classify::backend_identity`, which does not exist — the identity comes from
`ExecutionBackend::identity` in `encoder.rs`.

**STILL OPEN (5) bounded-read pre-reservation** — `read_setfit_apr_bytes_bounded` still reserves the
caller-declared length clamped to 256 MiB, so a source that lies upward commits a quarter gigabyte
before reading a byte. Fixing it means inventing a reservation ceiling the contract does not fix.

## Wave 6 close — three open items (orchestrator-recorded, 2026-08-15)

**OPS-01 is NOT met, and 04-12 must not be read as meeting it.** 04-12 landed its file and 5
passing tests, and PROVED the wave-5 blocker (OPS-01-F1) is gone: `into_artifact_bytes` returns
1,824,298 bytes cross-crate whose SHA-256 equals the digest the trusted policy recorded. But the
full train->save->load->embed->classify->inspect chain still cannot close, blocked by **F-10**:
`CALIBRATED_REGIMES` has exactly one entry, architecture compared for exact equality, so the
phase-3 MiniLM slice is the only trainable encoder and its 97-row vocabulary closure cannot compute
`probe_unicode` (canonical id 5915). Measured on three independent routes. The plan's
`requirements-completed` is deliberately `[]`. Closing F-10 needs a calibration run plus a
deliberate edit to `contracts/setfit-train-lifecycle-v1.yaml` (D-10(c)) — a **Phase 5** item.

CORRECTION to this file's own earlier F-10 action item: it tells 04-12 to "reuse 04-05's
substitution shape". That shape is `E0451` out-of-crate — both `SetFitRun` and `HeadFittedEvidence`
refuse it by name. 04-05's remedy exists only at the `--lib` tier, which is exactly why OPS-01 is
the requirement F-10 blocks hardest.

**D-04-08-A — `aprender-serve` standing red, pre-existing, NOT caused by phase 4.**
Orchestrator-verified after the wave-6 merge: `cargo test -p aprender-serve --lib` =
15389 passed / 51 failed, and all 51 share ONE root cause: `attempt to multiply with overflow` at
`crates/aprender-serve/src/contract_gate.rs:428:21`. No phase-4 plan has touched `contract_gate.rs`.
Consequence for 04-10: do NOT add a whole-crate `aprender-serve` test leg to the Make gates; scope
any serve leg to the setfit surface, or the gate is red on arrival.

**Codec-reachability note (surfaced by 04-12, assessed by the orchestrator).** With G1 supplying
artifact bytes out-of-crate, a caller can reach `AprCodec::deserialize` and obtain a `SetFitBundle`
without going through the verify policy's probe replay and round-trip closure. Bounded, and NOT a
credential-forgery path: `pub trait SetFitCodec: sealed::Sealed` is sealed so no codec can be
forged; `AprCodec` is not re-exported from the crate root; and the lock/token/grant doors still
require `reload_verified_run_from_apr`, i.e. the full `load_setfit_apr` ladder plus the provenance
identity gate. The observed route died at probe validation. Recorded so 04-07/04-09/04-15 treat
"has a bundle" as strictly weaker than "has a verified model" — not as a phase-4 blocker.

## Wave 8 close — constraints 04-10 MUST respect (orchestrator-verified, 2026-08-15)

04-10 wires the Make gates. There are now **two legs it must NOT wire**, both pre-existing and
both re-measured by the orchestrator on the post-wave-8 tree:

**D-04-08-A — no whole-crate `aprender-serve` test leg.** `cargo test -p aprender-serve --lib` =
15389 passed / 51 failed; all 51 share ONE root cause, `attempt to multiply with overflow` at
`crates/aprender-serve/src/contract_gate.rs:428:21`. No phase-4 plan touched that file. Scope any
serve leg to the setfit surface, or the gate is red on arrival.

**D-04-09-A — no `cargo check -p apr-cli --no-default-features` leg.** Orchestrator-measured
rc=101 (4 errors + 1 summary line). 04-09 re-measured it with the BASE manifest restored and got
the same failure, so it predates phase 4. Cause: `inference`-gated code that is not `cfg`-gated at
`src/commands/explain.rs:231,344`, `src/commands/diff_05_aprt_stage.rs:100`, `src/lib.rs:63`.
04-09's plan named this leg as SAFE-02's gating evidence; that is wrong. **The green equivalent is
`cargo check -p apr-cli --all-targets` with default features.**

Known-green legs on this tree, all orchestrator-measured post-merge:
- `cargo test -p apr-cli --lib --features setfit` -> 6751 passed / 0 failed / 15 ignored
- `cargo test -p apr-cli --test setfit_parity --features setfit,inference` -> 20 passed / 1 ignored
- `cargo test -p apr-cli --test setfit_cli_lifecycle --features setfit` -> 1 passed / 3 ignored
- `cargo test -p aprender-train --lib --features setfit` -> 7920 passed / **24 failed = the
  known-red baseline exactly**. Any train leg MUST diff failing NAMES against
  `.planning/phases/03-faithful-two-stage-trainer-and-head/known-red-baseline.md`, never assert 0.

**T-04-61 does not arise.** 04-09 measured that the `realizar` dev-dependency its plan required is
unnecessary — `--features setfit,inference` already makes `realizar::api` reachable from an
integration test. There is no feature unification to mitigate.

**OPS-01 and OPS-02 remain UNMET** behind F-10 (a Phase 5 item). 04-12, 04-15 and 04-09 each
declined to route around it by synthesising an APR-capable encoder, which would have compiled and
satisfied their acceptance criteria while testing a model training never produced. 04-09's parity
fixture IS synthetic and says so in both its module header and SUMMARY — legitimate, because its
claim is about three *readers* of one artifact, not about the training chain.
