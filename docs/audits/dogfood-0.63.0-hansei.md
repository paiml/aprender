# Dogfood 0.63.0 — hansei

Reflection on the audit epic #2373 (201 findings, 37 clusters) and the six fix passes that
closed part of it. Ledger: [`dogfood-0.63.0-ledger.md`](dogfood-0.63.0-ledger.md).

This document does one thing: takes the audit's unifying observation, drives a 5 Whys on it
to something changeable, and proposes one mechanical guard. Everything asserted here is
cited to a file and line in this tree, or to `gh issue view 2373`.

---

## 1. The class

> **A verdict that does not depend on what was checked.**

Concretely, five shapes of the same defect:

| shape | example from the audit |
|---|---|
| gate that cannot fail | `apr qa --assert-tps 100000` passed at 4.1 tok/s — the assertion was rewritten to `max(10, N/10)` for GGUF and discarded entirely for APR/SafeTensors (#2380) |
| gate that cannot pass | `apr lint` exits 5 with "0 error(s)" on all 18 models tested, pristine ones included (#2389) |
| value fabricated from an index | `/v1/explain` SHAP values are `0.1 - (i * 0.02)` over the feature *positions*, and `prediction: 0.95` is a literal — output is independent of the feature *values* and of whether a model is loaded (`crates/aprender-serve/src/api/apr_handlers.rs:212-224,261`) |
| count that drops its evidence before comparing | `apr data audit` reports "Duplicates: 0 (0.0%) OK" for any ratio ≤ 1% — 50 identical rows in 10,000 read as zero (#2381) |
| checks reported as run that never ran | `apr validate` printed "✓ VALID 3/100 points" with 21 of 25 checks "Pending / Not implemented"; `apr cbtop --iterations 0` printed a full green 7/7 report from zero measurements (#2394, #2397) |

The shapes differ. The invariant is identical: **the output symbol and the observation are
computed independently, so no input change can move the symbol.**

That the class is real and not a narrative imposed after the fact is visible in the fixes.
Six independent agents, working separate clusters without a shared prescription, each
converged on the same remedy — make the verdict *carry* its evidence in the type:

- `FinishReason::from_generation(stopped, completion_tokens, max_tokens)` — a terminal SSE
  chunk cannot be built without the budget it was generated under (#2375).
- `ImplementedScore { passed, ran, declared }` — a caller cannot render the numerator
  without the denominator that was actually measured (closure-audit).
- `MeasuredQuant::from_qtype(u32)`, whose only constructor reads a tensor header — a
  `String` derived from a filename no longer compiles into that field (closure-audit).
- `Health`, one value that both renders the badge and derives the exit code — printing
  "CORRUPTED" and returning success now requires deliberately discarding a `#[must_use]`
  (closure-audit).
- Route rows carrying `MethodRouter` — a path cannot be advertised without supplying the
  handler that answers it (#2376).

Five people did not invent the same fix by coincidence. They were all looking at one defect.

---

## 2. Five Whys

Each answer is evidence in this tree, not a restatement of the previous line.

### Why 1 — Why did 201 verdicts ship that do not depend on what was checked?

**Because the verdict channel was never observed under a changed input. The tests covering
these lines assert that the code was *reached*, not what it *concluded*.**

The clearest specimen sits directly on top of the fabricated-SHAP defect.
`crates/aprender-serve/src/api/tests/gpu_warmup.rs:199-224`:

```rust
let response = app.oneshot(request).await.expect("test value should be present");
// Accept various status codes - handler is exercised either way
let status = response.status();
assert!(
    status == StatusCode::OK
        || status == StatusCode::BAD_REQUEST
        || status == StatusCode::UNPROCESSABLE_ENTITY
        || status == StatusCode::NOT_IMPLEMENTED,
    ...
);
```

This test POSTs to `/v1/explain` and accepts four outcomes spanning three status classes.
It never reads the body — which is the only place the fabricated `prediction: 0.95` is
visible. The comment states the goal in the author's own words: *the handler is exercised
either way*. The neighbouring test `test_deep_apicov_apr_explain_empty_features`
(`deep_apicov.rs:246-269`) admits five.

This is not two tests. Measured over `crates/`:

| | count | files |
|---|---|---|
| `assert!` disjunctions over an outcome token (`StatusCode`, `is_ok`, `success()`, `.code(`) | **396** | 99 |
| …of those, admitting **more than one status class** (2xx/4xx/5xx) | **286** | 45 |
| literal tautologies — `assert!(r.is_ok() \|\| r.is_err())` | **34** | 24 |

An assertion admitting both 2xx and 5xx cannot distinguish "the endpoint served the
request" from "the endpoint failed." One of the 34 tautologies ships its own rationale —
`crates/apr-cli/src/commands/rosetta_verification_report.rs:153`:

```rust
assert!(result.is_ok() || result.is_err()); // Platform-dependent
```

That is the product's defect, written in the test tree, in the same crate as
`apr rosetta verify` — the command the audit found "prints *Round-trip verification
FAILED* with Max Diff: inf and exits 0" (#2382). I am **not** claiming this assertion hid
that bug; it tests `FormatType::from_extension`. I am claiming the habit is the same one,
and it is house style, not an accident.

### Why 2 — Why were those tests written to assert reachability?

**Because they were commissioned by a coverage number, and coverage counts execution, not
conclusions.**

The modules say so. `crates/aprender-serve/src/api/tests/mod.rs:44`:

```rust
mod predict_request; // T-COV-95 APR handlers coverage (predict, explain, audit, serde, error paths)
```

and the file next to it is named `deep_apicov.rs`. A five-way status disjunction and an
`assert_eq!` earn **identical line coverage**. Coverage is mechanically enforced here —
`Makefile:287` sets `COV_FLOOR := 88` and `Makefile:367` fails the build below it, with a
documented target of ≥95%. Nothing anywhere enforces what an assertion must *exclude*.

Given a floor to clear and no counter-pressure, the cheapest satisfying assertion is the
one that excludes nothing. The test suite optimised correctly for the only metric pointed
at it.

### Why 3 — But mutation testing *is* blocking at `MUTANTS_MAX_MISSED=0`. Why did it not catch this?

**Because it is diff-scoped, so it never looks at code that is not being edited.**

`.github/workflows/ci.yml:493-602` computes `git diff "$MERGE_BASE"...HEAD > pr.diff` and
runs `cargo mutants --in-diff pr.diff -- --lib`. It mutates **only lines the PR adds or
changes**. A verdict fabricated once and never touched again is never in any diff, so it is
never mutated — permanently.

This is not a bug in the gate. The comment at `ci.yml:476-488` records why: full-tree
mutation was push-to-main only and non-blocking, so "new under-tested code merged
silently"; diff-scoping is what made mutation *blocking at all*. The design is right and
should stay. Its unavoidable consequence is that **the entire pre-existing surface is
exempt by construction** — and 0.63.0 is almost entirely pre-existing surface. (Two further
narrowings compound it: `-- --lib`, so integration tests never kill a mutant, and the
`gate` job at `ci.yml:466` treats `skipped` as pass.)

### Why 4 — Why did no other gate look at shipped behaviour?

**Because every mechanical gate runs the product on inputs it expects to succeed.**

`ci.yml` invokes the `apr` binary exactly zero times — the only grep hit in the file is a
comment at `:432`. The binary is reached solely through integration tests, and there the
balance is:

| assertion | occurrences in `crates/apr-cli/tests/` |
|---|---|
| `.success()` | **633** |
| `.failure()` or `.code(` | **55** |

92% of everything ever asserted about the shipped binary is that it worked. The negative
space — *what must this tool refuse* — is 8%, and is required by nothing. So a command that
can only say "pass" is, at every checkpoint in this repo, indistinguishable from a command
that is correct.

### Why 5 — Why is negative space not part of the work? *(root — and changeable)*

**Because "done" here is defined as *the product did the thing*, and never as *the product
refused the thing it must refuse*. No artifact in the repo states, per surface, what input
must be rejected and with what verdict — so no gate can check it, and the cheapest way to
satisfy every gate is an assertion that excludes nothing.**

The contract layer is where such statements would live, and it inherited the same habit
rather than correcting it. #2377's pass found `contracts/crux-B-07-v1.yaml:60-88`
deliberately discharging **both** the improvement gate and the leakage gate under a single
`FALSIFY-CRUX-B-07-001` — one falsifier standing in for two opposed claims — and
recommended reclassifying the finding rather than "fixing" the code into disagreement with
its own contract. Even the falsifiable-contract tree can hold a verdict that excludes
nothing.

This is the changeable thing. Not "write better tests" — that is vigilance, and this repo's
own record says vigilance loses: the `apr`-invocation guard regexes were wrong **five
times**, and every one was caught by a must-match/must-not-match table, none by review
(CLAUDE.md, Verification Discipline #7). The changeable thing is that **nothing mechanical
in this repo requires an assertion to exclude an outcome.** That is a rule a script can
enforce.

---

## 3. The standard-work change

### `scripts/check_assertions_exclude.sh`

One guard, in the shape that has stuck here (`scripts/check_*.sh` wired into `ci.yml`'s
guard block, self-testing, vacuity-proof). Seconds to run. No build, no models, no GPU.

**Rule.** An assertion is in scope if it is an `assert!` / `prop_assert!` whose expression
contains `||` and mentions an outcome token: `StatusCode::`, `.status()`, `is_ok()`,
`is_err()`, `.success()`, `.failure()`, `.code(`, `exit_code`. It **fails** if it admits
more than one *outcome class*:

| domain | classes | failing example | passing example |
|---|---|---|---|
| HTTP | 2xx / 4xx / 5xx | `OK \|\| BAD_REQUEST \|\| NOT_IMPLEMENTED` | `BAD_REQUEST \|\| UNPROCESSABLE_ENTITY` |
| `Result` | ok / err | `r.is_ok() \|\| r.is_err()` | `r.is_err()` |
| process | zero / non-zero exit | `code == 0 \|\| code == 3` | `code == 3 \|\| code == 5` |

A disjunction *within* one class stays legal and needs no exemption: "the client was told it
was wrong, and I don't care which way" is a real claim that excludes a real outcome. Only
"I don't care whether it worked" is banned. That is the narrowest rule that names the
defect, and it needs no judgement at the call site.

**Baseline ratchet.** 320 sites across ~60 files exist today; they cannot be fixed in one
PR, and pretending otherwise is how a guard gets `|| true`-d six weeks later. So:
`scripts/assertion_exclusion_baseline.txt` holds `path<TAB>count` per file. The guard fails
if any file exceeds its baseline, **or if a file absent from the baseline has any**. New
code is at zero from day one. A second check asserts the committed sum only ever decreases
and prints the delta, so the debt is visible and monotonic.

**Wiring.** `ci.yml` guard block, beside `check_pass_grep_anchored.sh` (`:404`), invoked as
`bash scripts/check_assertions_exclude.sh` plus a separate `--self-test` step in the style
of `check_sourced_libs_option_neutral.sh --self-test` (`:419`).

**Self-test case table** (shipped in the script, run in CI — per Verification Discipline #7,
re-run the table, don't re-read the pattern):

*Must turn RED:*
1. The verbatim four-way assertion from `gpu_warmup.rs:199-224` above — the test that
   covered `/v1/explain` while it returned a hardcoded 0.95.
2. `assert!(result.is_ok() || result.is_err());`
3. `assert!(out.status.code() == Some(0) || out.status.code() == Some(5));`

*Must stay GREEN:*
4. `assert!(status == StatusCode::BAD_REQUEST || status == StatusCode::UNPROCESSABLE_ENTITY);`
5. `assert_eq!(status, StatusCode::OK);`
6. `assert!(n == 1 || n == 2);` — no outcome token, out of scope
7. `assert!(msg.contains("a || b"));` — `||` inside a string literal

**Vacuity guard.** The script asserts it scanned ≥ 2000 `.rs` files and that it re-found at
least the baseline files by name, failing loudly on zero hits. A guard that silently matches
nothing must not pass as clean — the lesson from `raw_htod_symbol_has_exactly_one_call_site`
and from `check_pass_grep_anchored.sh`'s own `MIN_EXPECTED`.

**The mutation that proves it works** — to be run and its verbatim output recorded in the
implementing PR, not asserted:

1. Append one new test containing case (1) to a file **not** in the baseline
   (e.g. `crates/apr-cli/src/commands/`). Guard must exit 1, naming `file:line` and the
   classes admitted (`2,4,5`). Revert → exit 0, byte-identical tree.
2. Then the load-bearing half: tighten the real `gpu_warmup.rs:199-224` to
   `assert_eq!(status, StatusCode::OK)` and run it. **Either outcome is a result.** If it
   goes RED, the endpoint's true verdict is now pinned and a live defect just surfaced. If
   it stays GREEN, one line of baseline debt retires and the sum decrements. What must not
   happen is what happens today: nobody ever asks.

Step 2 is the point of the whole exercise. The guard's value is not that it deletes 320
weak assertions — it is that it forces someone to state, once per site, what the code must
*not* do. That question is what nobody was ever required to answer, and it is the question
that produced all six clusters' real fixes.

### Rejected alternatives

| candidate | why not |
|---|---|
| Full-tree mutation testing | Would catch this class directly, and was already tried: `ci.yml:476-488` records it as push-to-main-only and non-blocking because full-tree runtime is prohibitive. Re-proposing it ignores measured history. |
| A negative-space corpus running the real binary (`apr <cmd> <known-bad>` must exit non-zero) | Catches strictly **more** of the actual 201 — it is the right *second* step and I'd take it next. But it needs fixtures, a pinned build and models: minutes in `workspace-test`, not seconds in `gate`. The brief asked for the cheapest point; this is not it. |
| A review rule ("assert what must fail") | Vigilance. Five wrong guard regexes in this repo were caught by tables, none by review. |
| Banning `\|\|` in assertions outright | Over-broad; it would reject the legitimate one-class disjunction and get exempted into uselessness. |

---

## 4. Ratchets vs. instance removals

The six passes are **not** of uniform strength. Sorted honestly:

**Real ratchets — the defect is now unrepresentable, and the guard scans a whole surface:**

- `lint_family_guard.rs` (#2377) — scans all 29 `crates/apr-cli/src/commands/*_lint.rs`,
  bans `-> Result<(), String>`, bans any mention of `InvalidFormat`, and rejects
  `.map_err(CliError::Aprender)` within 15 lines of a `*_lint::run(` call. Vacuity-guarded
  (≥25 files, names two by name). Strongest artifact in the set: a class fix with a scanner,
  not sixteen point fixes.
- `raw_htod_symbol_has_exactly_one_call_site` (gpu-order-dependence) — whole-crate scan,
  fails on zero hits as loudly as on many.
- Thread-local transfer/cache counters — cross-test pollution unrepresentable at any
  parallelism, and it needed zero changes at the 87 call sites. Leak assertions became
  `assert_eq!` instead of carrying 10–50 MB tolerances.
- The route table carrying `MethodRouter` (#2376) — the two-hand-maintained-lists *shape*
  is gone, not just the `--no-metrics` drift it produced.
- The error-body middleware keying on response **shape** rather than an enumerated status
  set (#2376) — covers an axum rejection variant the day it appears. Enumeration is exactly
  what let 400 and 415 through for two releases.
- Type-level: `ChoiceCount`, `TurnPermit`, `MeasuredQuant`, `DiffMode`, `ImplementedScore`,
  `produces_qa_score` with no wildcard arm, `apr_decode_dense_float` with the `_ =>` arm
  deleted.

**Instance removals — the class survives elsewhere:**

- The `AppState::demo()` error-string scan (#2375) covers *those* constructor names in
  *that* crate. Any other internal identifier leaking into an HTTP body is unguarded.
- `ptx-map` `num_kv_heads` (2444-2) — the two hardcoded arms now read config, but the
  poka-yoke described is for `quant`. Nothing prevents the next hardcoded architecture arm.
  Note the reason it read healthy in the first place: the 1.5B model was correct **by luck**,
  matching one of the two hardcoded arms. A single-model probe would have confirmed health.
- `#2443` fixed BF16 1-D dtype dispatch — but the reporting agent flagged a *second*
  prompt-independent model (`qwen2.5-coder-1.5b-instruct-st.apr`, 339/339 F32) as a
  different root cause, untouched. **Prompt-independence as a class is not closed.**
- `AppState::model_format()` still reports `"gguf"` for any quantized backend
  (`mod_app_state_new.rs:182-195`). Finding 6 in #2376 stays *closed* because the two
  endpoints agree — they agree on a wrong value. That is the audit's own class, surviving
  inside a fix pass, correctly flagged by the agent and deliberately not fixed.

**Ratchets whose proof is weaker than it looks:**

- GPU-ORD-9 measured RED 10/10 on a busy box, **RED 0/10 an hour later on an idle GPU**,
  RED 6/10 again under load. It is a load-dependent race: a green run proves nothing, and
  only the RED reproduced under load is evidence. The fix is right; the falsifier is not a
  regression detector.
- The `.config/nextest.toml` contract ("a test claiming device capacity MUST take the
  in-process lock AND join the `gpu-exclusive` group") is **documentation, not a mechanism**.
  It is the one place in the six passes where the remedy is vigilance, and it is correctly
  labelled as covering two execution models that each defeat the other's guard.

---

## 5. What the audit did *not* establish

- **The count is a floor and has never been stable.** The ledger has said 190, 211, 202 and
  201; the epic body says 201/24 P0 while the working memory of the same audit says
  202/26 P0. One probe slice (`inspect`/`validate`/`tensors`/`diff`) never completed at all.
  Nothing here should be read as "201 defects exist" — only as "700+ invocations surfaced at
  least this many."
- **No user impact was measured.** Every finding is a divergence between documented and
  actual behaviour, reproduced ≥2 ways. Not one is tied to a user report, a support ticket,
  or a production incident. Severity is the auditor's judgement, unvalidated.
- **Regression vs. never-worked was not determined** for most findings. `--no-metrics` route
  drift is known to be a regression introduced by #2449 itself; for the rest, nobody
  bisected. "How long has this shipped?" is unanswered, so the rate at which the class is
  being *created* is unknown — and that rate is exactly what tells us whether the proposed
  guard is worth its cost.
- **The five mechanisms are a post-hoc grouping**, not a measured partition. I find them
  convincing because six independent fix passes converged on one remedy (§1). That is
  evidence, not proof, and no attempt was made to falsify the grouping by looking for
  findings that fit none of the five.
- **Coverage of the fix passes is partial and stated as such**: #2375 fixed 4 of 7
  (#4, #7 untouched); #2376 fixed 2 of 4, leaving both resource-lifecycle defects, on the
  explicit and correct reasoning that fixing four shallowly is worse than two well; #2377
  left six live findings. The clusters are *reduced*, not closed.
- **The guard proposed above is necessary, not sufficient.** It forces an assertion to
  exclude *something*; it cannot force it to exclude the *right* thing. `assert_eq!(status,
  StatusCode::OK)` against a handler that returns 200 unconditionally still passes. It would
  **not** have caught #2443 (no test existed at all), the GPU order-dependence cluster (a
  real race, not an assertion defect), or the #2376 route drift (which needed the two-list
  shape removed). It attacks the enabling condition of the two largest mechanisms at the
  cheapest checkpoint. The binary-level negative-space corpus is the honest second step, and
  should follow.
- **This document has not been tested.** The 5 Whys is reasoning over evidence, and the
  chain is only as strong as Why 3 — that diff-scoped mutation structurally exempts
  pre-existing code. That claim is checkable and should be checked: pick ten of the fabricated
  verdicts, confirm none appears in any PR diff since it was written. If several do, Why 3 is
  wrong and the proposal needs rethinking.
