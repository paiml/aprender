# APR-CODE-EVAL-001 — the `apr code` eval: our own coding-agent benchmark

| | |
|---|---|
| **Spec id** | APR-CODE-EVAL-001 |
| **Version** | 1.0.0 (2026-10-14) |
| **Status** | PROPOSED |
| **Epic** | #5281 (children #5282–#5288) |
| **Ontology** | `contracts/apr-code-eval-task-v1.yaml`, `contracts/apr-code-eval-sandbox-v1.yaml`, `contracts/apr-code-eval-verdict-v1.yaml`, `contracts/apr-code-eval-metric-v1.yaml` |
| **Language** | Rust only for the harness. Python appears only as a target language inside the sandbox. |
| **Check class** | LAB. Never on a PR, merge-queue or release path. |

## 1. Why we build our own

Every other `apr` verb has an outside reference: `apr run` and `apr serve` are measured
against Ollama and llama.cpp, and the CRUX contracts name a competitor for each surface.
`apr code` has none, because Ollama ships no coding agent. So there is no external number
to match, and today nothing checks whether `apr code` fixes code at all.

The answer is not to invent a new kind of benchmark. Five public benchmarks already settle
how a coding-agent eval should work, and each was written to fix a failure in the ones
before it. This spec takes the rule each one contributes, writes those rules down as pv
contracts (§3–§6), and builds a Rust harness that obeys them.

## 2. Industry practice and what we take from each

| Benchmark | Its rule | Failure it fixes | Where it lands here |
|---|---|---|---|
| **SWE-bench Verified** (Jimenez et al., ICLR 2024; OpenAI's human-validated 500-task subset, 2024) | Real repository at a pinned commit; hidden FAIL_TO_PASS and PASS_TO_PASS tests; resolved = all of both pass; every task validated against its gold patch | Toy problems that do not look like real work; tasks that cannot be solved or that pass with no change | Task kind `repo`; ACE-TASK-002/003; ACE-VER-001/007 |
| **EvalPlus** (HumanEval+ / MBPP+; Liu et al., NeurIPS 2023) | Extend the original tests by ~80× with generated inputs | Weak tests passing wrong code (HumanEval's ~7.7 tests per problem) | Task kind `function`; hidden tests are the EvalPlus `plus` set |
| **Aider polyglot** (2024) | 225 hard Exercism exercises in C++, Go, Java, JavaScript, Python and Rust; the model must edit through a declared edit format; two attempts, the second with test output | Python-only evals; harnesses that quietly repair malformed edits | Task kind `edit`; ACE-VER-005; format conformance ACE-MET-005; the two-attempt loop in §8 |
| **Terminal-Bench** (2025) | Agentic tasks in a container, judged only by tests of the container's final state | Judging the transcript instead of the result | Task kind `terminal`; §4 sandbox; the state check is the F2P test |
| **LiveCodeBench** (Jain et al., 2024) | Every problem carries its release date; report scores for problems released after the model's training cutoff | Contamination: models scoring on problems they were trained on | ACE-TASK-006; ACE-MET-006; §9 |

Two practices that apply to all of them:

- **pass@k via the unbiased estimator** (Chen et al., 2021, the Codex paper): `1 − C(n−c,k)/C(n,k)`
  per task, never computed with the problem count in the `n` slot. The repo already has this
  rule for `apr eval` (`contracts/eval-passk-single-sample-v1.yaml`); ACE-MET-002 reuses it.
- **Report an interval, not just a point.** Suites here are small (tens to a few hundred
  tasks), so ±7 points at N=164 is normal. The Wilson score interval (ACE-MET-003) behaves at
  0% and 100%, where the normal approximation does not.

What we do **not** take: leaderboards that let a submitter run their own harness (scores
are only comparable inside one harness version), LLM-as-judge verdicts (a test decides, never
a model), and "best of N reruns" (a flaky run is reported as a flake, §4).

## 3. Task (contract `apr-code-eval-task-v1`)

One record serves all four kinds:

```text
Task {
  id, kind ∈ {repo, function, edit, terminal}, release_date, language,
  prompt,                      // shown to the agent
  snapshot,                    // the files the agent may read and edit (repo at base commit, stub file, exercise dir, container setup)
  hidden_tests { fail_to_pass[], pass_to_pass[] },   // never shown
  gold,                        // reference edit, used only for validation
  edit_format ∈ {whole, search_replace, udiff, agentic},
  image_ref,                   // name@sha256:… (ACE-SBX-002)
  limits { wall_s, cpus, mem, pids, disk },
  sha256,
}
```

Rules (each one is an equation in the contract and has a falsification test):

1. **The set is non-empty and the ids are unique** (ACE-TASK-001).
2. **Every task can fail**: `|F2P| ≥ 1` (ACE-TASK-002).
3. **Gold validation** before a task enters a suite: the gold edit resolves it and the empty
   edit does not (ACE-TASK-003). This is SWE-bench Verified's filter. A task the empty edit
   resolves would score a model for doing nothing.
4. **Hidden tests are invisible** to the agent (ACE-TASK-004).
5. **The set digest** is order-independent and content-sensitive (ACE-TASK-005). A receipt
   names its tasks by this digest.
6. **Every task is dated** (ACE-TASK-006).

**Suites.** A suite is a manifest of task ids plus its digest:

| Suite | Kind | Size | Source | Purpose |
|---|---|---|---|---|
| `smoke` | all four | 8 (2 per kind) | hand-written in-repo, MIT | harness self-test; every nightly runs it first |
| `function` | function | HumanEval+ 164 + MBPP+ 378 | EvalPlus release, pinned version + sha256 | comparable to published numbers |
| `edit` | edit | Aider polyglot 225 | Aider benchmark repo at a pinned commit | multi-language edits + format conformance |
| `repo-lite` | repo | ≤ 50 from SWE-bench Verified | per-instance images rebuilt from their Dockerfiles, pinned by digest | real-repo fixes at a nightly-sized cost |
| `terminal` | terminal | ≤ 30 | Terminal-Bench tasks, pinned commit | agentic shell work |
| `fresh` | all | grows monthly | tasks written after the model cutoff | contamination-free score (§9) |

Third-party suites are imported by a loader at a pinned upstream commit and sha256, not
vendored, and each keeps its own license. A suite whose upstream digest changes is a new
suite version, never a silent update.

## 4. Sandbox (contract `apr-code-eval-sandbox-v1`)

Each (task, candidate edit, seed) runs once in a fresh container:

| Property | Rule | Equation |
|---|---|---|
| Network | `--network none`; a planted outbound call must fail | ACE-SBX-001 |
| Image | pinned by `@sha256:` digest; a tag alone is refused; the pulled digest is compared | ACE-SBX-002 |
| Limits | wall, CPUs, memory, pids and disk all finite; a breach is a status (`timeout`, `oom`, …), never a pass | ACE-SBX-003 |
| Filesystem | task dir read-only plus a tmpfs workdir; no container socket, no host home | ACE-SBX-004 |
| Environment | `SEED`, `TZ=UTC`, `LANG=C.UTF-8`, `SOURCE_DATE_EPOCH`, `PYTHONHASHSEED` fixed; host secrets never passed | ACE-SBX-005 |
| Determinism | same inputs ⇒ same report apart from wall time | ACE-SBX-006 |
| Test counts | read from the runner's structured report (JUnit XML, cargo/nextest JSON, pytest junit); no report ⇒ `sandbox_error` | ACE-SBX-007 |
| Clean slate | a new container per run; nothing survives | ACE-SBX-008 |

**Flakes.** The validation step runs the gold edit three times. A task whose report differs
across those runs is quarantined with a ticket and leaves the suite until fixed. A scored run
is never retried until it goes green.

**Runtime.** The runner drives the container engine through its CLI or socket API from Rust.
Runner jobs already have a container engine, so no new host service is needed.
Architecture: `function`, `edit` and `smoke` images are built multi-arch (amd64 + arm64).
Most SWE-bench instance images are amd64-only, so `repo-lite` runs on the X64 pool until
arm64 rebuilds of its images are validated (ACE-TASK-003 on arm64).

## 5. Verdict (contract `apr-code-eval-verdict-v1`)

`verdict(task, model_output, sandbox_report)` returns exactly one of
`RESOLVED, FAILED, FORMAT_ERROR, EMPTY, TIMEOUT, SANDBOX_ERROR, NO_TESTS`. The rules are
checked in this order and the first match wins (ACE-VER-006 is this function):

1. no edit in the output → `EMPTY` (ACE-VER-002);
2. the edit does not parse or does not apply under the task's edit format → `FORMAT_ERROR`
   (ACE-VER-005). The harness never fuzzy-applies or repairs the model's edits;
3. killed by the wall-clock limit → `TIMEOUT`; any other non-ok sandbox status (OOM, pids,
   disk, missing runner report) → `SANDBOX_ERROR` (ACE-VER-004);
4. not every F2P test executed → `NO_TESTS` (ACE-VER-003);
5. all F2P pass and all P2P pass → `RESOLVED`, else `FAILED` (ACE-VER-001/007).

Only `RESOLVED` counts as a pass. Every label stays in the denominator.

## 6. Metric (contract `apr-code-eval-metric-v1`)

| Metric | Definition | Equation |
|---|---|---|
| pass@1 | resolved samples over **all** `|T|·n`, with errors and timeouts included | ACE-MET-001 |
| pass@k | unbiased estimator, `k ≤ n`; `n = 1` ⇒ pass@k = pass@1 | ACE-MET-002 |
| 95% CI | Wilson score interval on pass@1 | ACE-MET-003 |
| format conformance | `1 − FORMAT_ERROR samples/(|T|·n)` | ACE-MET-005 |
| post-cutoff pass@1 | pass@1 over tasks dated after the model cutoff, with the count; `null` when there are none | ACE-MET-006 |
| counts by label | one count per verdict label; they sum to `|T|·n` | ACE-MET-008 |
| cost | prompt + completion tokens, turns, wall time per task (reported, not scored) | — |

**Refusals (ACE-MET-004).** The engine prints no number at all, returning `Err(Refused)`, when:

- the task set is empty;
- zero tests executed across the run; or
- every output is `EMPTY`. That last case is a broken driver, not a 0% model.

This is the rule the operator asked for: "refusing 0-task/empty-output passes".

Default sampling: greedy (`temperature = 0`), `n = 1`, fixed seed, so pass@k = pass@1.
pass@k with `k > 1` is an opt-in run with `n ≥ k` and recorded sampling settings.

## 7. Receipt

Every run writes one JSON receipt (ACE-MET-007). A metric without every field below is
refused:

```text
{ schema: "apr-code-eval-receipt/v1",
  suite, task_set_digest, model: { path, sha256, quant, backend: cpu|cuda },
  apr: { version, git_sha }, harness_git_sha,
  sandbox: { engine_version, image_digests{task_id → sha256} },
  seed, sampling: { temperature, top_p, n, max_tokens, max_turns },
  started_at, wall_s,
  per_task: [ { id, verdict, tests_executed, tests_passed, f2p_passed, p2p_failed,
                edit_format_ok, tokens_in, tokens_out, turns, wall_ms, report_sha256 } ],
  counts_by_label, metrics: { pass_at_1, wilson95, pass_at_k?, format_conformance,
                              post_cutoff: { pass_at_1|null, n } } }
```

Two receipts can be compared only if `suite`, `task_set_digest`, `harness_git_sha` and
`sampling` match. A comparison tool refuses any other pair.

## 8. The `apr code` driver

The driver puts `apr code` in front of a task the way a user would, with a bounded budget:

- **Input**: the `prompt` and a working copy of the `snapshot`. Tools (read, write, shell)
  run inside the task's sandbox, never on the host.
- **Budget**: `max_turns`, `max_tokens` and wall time come from the suite manifest and are
  recorded in the receipt.
- **Edit formats**: `whole` (the full file), `search_replace` (Aider's SEARCH/REPLACE
  blocks), `udiff` (unified diff), and `agentic` (direct writes through the tool loop). A
  block that does not apply is a `FORMAT_ERROR` (§5).
- **Two attempts** for `edit` tasks (Aider practice): after a failed first attempt, the
  visible test output (never the hidden tests) goes back to the model once. pass@1 is
  reported per attempt (`attempt_1`, `attempt_2`).
- **Backends**: the model runs through realizar on CPU or CUDA (`apr code` already does).
  The sandbox only runs tests, so it needs no GPU.
- **Determinism check**: two runs of `smoke` with the same seed must produce receipts that
  are byte-identical apart from timing fields.

## 9. Contamination control

- Every task carries `release_date`. Each receipt reports the post-cutoff score
  separately (ACE-MET-006), using the cutoff declared in the model card.
- **Canary**: every in-repo task file carries one fixed canary GUID string. If a model can
  complete that string, it has seen the suite in training.
- **Overlap check**: n-gram overlap (13-gram, as in the GPT-3 contamination
  analysis) between gold solutions and our own training-corpus manifests. A hit flags the
  task in the receipt. It is not dropped silently.
- The `fresh` suite adds tasks every month, written after the current model cutoffs.

## 10. Where it runs

- **Developer**: `apr` gets an eval entry point that takes a suite manifest and writes a
  receipt. The exact verb, and whether it is a new `apr code` subcommand or a benchmark of
  `apr eval`, is decided in #5286 and registered in `contracts/apr-cli-commands-v1.yaml`.
- **Nightly (LAB)**: once built, the `smoke` and `function` suites run on CPU and on GPU on
  the ARM64 CUDA edge runner through the fast-verb matrix of #5254, plus `edit` on the
  X64 pool. Each night posts the receipt and one ranked list of failing tasks to #5288.
- Per the GATE/LAB rule, it never runs in a PR, merge-queue or release job and never blocks
  one. A red night updates its ticket and stays silent. After 7 days red or unmeasured it
  stops until its owner brings it back. Promoting any of it to a GATE is an operator ruling.

## 11. Language rule

The harness, loader, sandbox runner, verdict, metric and receipt are Rust, in a proposed
workspace crate `aprender-code-eval` (#5283–#5285). Python is allowed only as a target
language inside the sandbox: HumanEval+ and MBPP+ tests run under a pinned Python image, the
same way Go or Java tests do for the `edit` suite. No harness step shells out to a Python
script on the host.

## 12. Falsification summary

| Contract | Equations | Falsification tests | Planted failure each must catch |
|---|---|---|---|
| task | 6 | 6 | empty set, empty F2P, task solved by the empty edit, hidden test in snapshot, digest drift, undated task |
| sandbox | 8 | 8 | outbound network, tag-only image, runaway loop / OOM, host read, leaked token, nondeterminism, exit 0 with no report, state between runs |
| verdict | 7 | 7 | partial F2P, empty output, zero executed, killed run with passes, unapplied edit, odd report shape, P2P regression |
| metric | 8 | 8 | errors dropped from denominator, biased pass@k, Wilson at 0/N, 0-task / 0-test / all-empty runs, hidden format errors, fake post-cutoff 0, missing receipt field, lost verdict |

Each falsification test is cited in its contract as `PLANNED (#ticket)` until the crate
exists, so `scripts/check_contract_test_binding.sh` stays at baseline 0. The ticket that
builds a module replaces each `PLANNED` line with its `cargo test` invocation in the same PR.

## 13. Tickets

| Ticket | Scope | Done when |
|---|---|---|
| #5282 E1 | this spec + the four contracts | `pv validate` green; C314 receipt on the PR head |
| #5283 E2 | task schema, loader, suite manifests, importers | FALSIFY-ACE-TASK-* green |
| #5284 E3 | sandbox runner | FALSIFY-ACE-SBX-* green, including the planted network call |
| #5285 E4 | verdict + metric engine | FALSIFY-ACE-VER-* and FALSIFY-ACE-MET-* green |
| #5286 E5 | `apr code` driver, edit-format adapters, CLI verb | `smoke` receipts byte-identical across two seeded runs |
| #5287 E6 | contamination: dates, canary, overlap | planted leaked task flagged |
| #5288 E7 | receipts + LAB nightly via #5254 | two consecutive nightly receipts and a ranked gap comment |

Order: E2 → E3 → E4 can proceed in parallel after E1; E5 needs E2–E4; E6 and E7 need E5.

## 14. Non-goals

- A public leaderboard, or a claim of parity with a hosted model's published number.
  Scores are comparable only inside one harness version (§7).
- LLM-as-judge verdicts. A test decides.
- Training on any suite. The canary and overlap check exist to detect that.
- Any gate. This is LAB until the operator rules otherwise.
