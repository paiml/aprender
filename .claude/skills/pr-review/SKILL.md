---
# EXPLICIT name (#2332 class, the same reason pre-release and apr-dogfood carry one).
# Without it a skill takes its name from its directory, and a user-scope skill at
# ~/.claude/skills/pr-review/ would shadow this file: it would never appear in the
# session's skill listing, could not be invoked, and nothing would warn. Edits would
# look effective and change nothing that runs. That is #2361, and it cost this repo a
# hardened release-certifying skill that never executed.
name: pr-review
allowed-tools: Bash(git:*), Bash(pmat:*), Bash(jq:*), Bash(sha256sum:*), Bash(minisign:*), Bash(check-jsonschema:*), Bash(cargo:*), Bash(bash:*), Bash(gh:*), Bash(pv:*), Bash(grep:*), Bash(sed:*), Bash(awk:*), Bash(cut:*), Bash(cat:*), Bash(head:*), Bash(tail:*), Bash(wc:*), Bash(mkdir:*), Bash(printf:*), Bash(command:*), Bash(base64:*), Read, Glob, Grep, mcp__nvidia-cuda-docs__search_cuda_docs
description: Adversarial PR review that must show HOW it knows — five consultations (one of them a reviewing agent from a different vendor), a three-state availability encoding no prose can fake, and a signed in-toto receipt whose own guard can reject it
# MACS F4: pinned. A review whose verdict can block a merge is not a place to let effort
# float; a cheap run that reports PASS is indistinguishable from a thorough one that does,
# and the whole point of the receipt is that they should not be.
effort: high
---

# pr-review — grounded adversarial review with a receipt that can be rejected

**Version**: 2.0.0 (implements `PR-REVIEW-SKILL-002 v2`; this is that spec's §9 step 5)
**Authority**: the spec. Where this file and the spec differ, **the spec wins** and the
difference is a defect in this file.
**Contract**: `contracts/pr-review-skill-v2.yaml` (§1 grounding, §7 blocking, §8 metrics)
**Guard**: `scripts/check_pr_review_receipt.sh` — it validates what you emit here, it has
its own positive controls, and its mutation set (`scripts/mutate-guard.sh`) reports
219/219. **Run it on your own receipt before you post anything.**

## Context

- Branch: !`git branch --show-current`
- Head: !`git rev-parse HEAD`
- Merge base vs origin/main: !`git merge-base origin/main HEAD 2>/dev/null || echo "NO MERGE BASE — fetch origin first"`
- Files in the diff: !`B=$(git merge-base origin/main HEAD 2>/dev/null) && git diff --name-only "$B" HEAD | wc -l || echo "UNKNOWN — no merge base"`
- pmat index present: !`test -f .pmat/context.db && echo yes || echo "no — see §3.A precondition"`
- Guard present: !`test -x scripts/check_pr_review_receipt.sh && echo yes || echo NO`
- Tools: !`for t in git jq sha256sum check-jsonschema minisign pmat cargo-mutants; do command -v "$t" >/dev/null 2>&1 || printf 'MISSING:%s ' "$t"; done; echo ok`

A `MISSING:` above is not a licence to skip a consultation. The guard treats an absent
tool as a **rejection, never a skip** — "a gate that cannot execute its own checks must
not report green". Install it, or emit `unreachable` and take the `DEGRADED`.

## Where this file sits among the review artifacts

Five things exist and each owns exactly one job. Restating another's job here is how a
runner comes to exist twice (aprender#2640).

| artifact | owns |
|---|---|
| `PR-REVIEW-SKILL-002 v2` (spec) | every rule. The authority. |
| `contracts/pr-review-skill-v2.yaml` | §1 / §7 / §8 as falsifiable equations |
| `schemas/` + `scripts/check_vendored_schemas.sh` | the offline schema gate |
| `scripts/check_pr_review_receipt.sh` | **whether a receipt is acceptable** |
| this file | **how to produce one honestly** |

The guard decides acceptance. This file cannot loosen it and must not try: if a rule here
is weaker than the guard, the guard wins and your receipt is rejected; if it is stronger,
say why in the receipt rather than encoding a private policy nothing tests.

---

## §0 The three sentences this skill exists to make impossible

1. *"I checked the CUDA docs and they didn't say anything."* — indistinguishable, in
   prose, from never asking. §3.0 and §3.B make the two different **artifacts**.
2. *"No issues found."* — indistinguishable, in prose, from a consultation that could not
   run. §3.0 row 3.
3. *"The receipt is signed."* — true, and it says nothing about whether the review was
   done. §4.3.

Every rule below is downstream of one of those three.

---

## §1 The grounding rule — three marks, and there is no fourth

Every claim this review makes about external reality carries exactly one mark:

| mark | means | required alongside it |
|---|---|---|
| `cited` | traced to a consultation **this run performed** | `source`, `excerpt` (≤400 chars), `excerpt_sha256` |
| `measured` | produced by a command **this run executed** | `command` (argv array), `exit_code`, `stdout_sha256` |
| `asserted` | reviewer judgement | `rationale`; **always** `precision_class: advisory` |

An unmarked claim is not a weaker claim. It is a **defect in the review**, and the guard
rejects the whole receipt that carries it (fixture row 15).

Four rules that are easy to get wrong and are checked mechanically:

- **`excerpt_sha256` is over the excerpt bytes AS STORED**, with no trailing newline.
  `printf '%s' "$excerpt" | sha256sum`. Using `echo` adds `\n`, the digest changes, and
  the guard rejects the citation as unverified (fixture row 12). This is the single most
  common way a well-intentioned receipt fails.
- **`asserted` never blocks.** A finding marked `asserted` with
  `precision_class: blocking` is rejected. Judgement is welcome; judgement with a merge
  block behind it is not.
- **`failure_scenario` is required and non-empty on every result.** A finding that cannot
  name the concrete failure it permits is a comment, and comments are not what this skill
  is for.
- **§1.2, contradiction is fatal to the claim, not to the run.** If an `asserted` claim
  contradicts a `measured` value elsewhere in the same receipt, **drop the claim** and
  record `finding.suppressed_by_measurement` in
  `predicate.consultations.<k>.suppressed[]`. This is the anti-hallucination-snowball
  rule: the measurement wins, and the fact that it had to win is itself recorded.

**§1.1 residual risk, stated rather than hidden.** The guard verifies that
`excerpt_sha256 = sha256(excerpt)` and that the excerpt is non-empty. It does **not**
verify that the excerpt supports the claim. Entailment checking is Phase 3. Until then a
well-formed citation of an irrelevant excerpt passes, and you are the only thing stopping
it. Do not cite an excerpt you would not be willing to have quoted back at you next to
your claim.

---

## §2 The diff boundary — merge-base, never `origin/main`

```bash
git fetch origin main --quiet          # do this FIRST; a stale origin/main moves BASE
BASE=$(git merge-base origin/main HEAD)
HEAD_SHA=$(git rev-parse HEAD)
git diff --name-only "$BASE" "$HEAD_SHA"      # the only diff you review
```

`BASE` goes into the receipt as `base_sha`; the guard recomputes it and rejects any other
value (fixture row 10). A floating `origin/main` pulls other agents' commits into your
review, inflates tokens, and manufactures false positives — this repo runs parallel
worktree agents, so that is the normal case, not the edge case.

**Blast radius.** Consultations run over the changed crates *and their reverse
dependencies*, recorded as `affected_crates[]`:

```bash
cargo metadata --format-version 1 --no-deps | jq -r '.packages[] | .name'   # 79 packages
```

A consultation that skipped a crate listed in `affected_crates` is `DEGRADED`, not silent.

**Baseline cache reuse** (complexity, TDG, semantic index) is admissible only when
`git merge-base --is-ancestor "$baseline_commit" HEAD` exits 0 *and* the file's last
modification is at or before `$baseline_commit`. Count the reuse into
`consultations.pmat.cache_hits` — an uncounted cache is an unfalsifiable one.

---

## §3 The five consultations

### §3.0 Unavailability is never silently green — and the encoding is the point

Four states. They are distinguished by the **artifact**, never by the prose:

| state | SARIF | predicate | verdict effect |
|---|---|---|---|
| consulted, found nothing | run present, `executionSuccessful: true`, `results: []`, `toolExecutionNotifications: []` | `status: consulted` | none |
| consulted, found something | run present, `executionSuccessful: true`, `results` populated | `status: consulted` | `FINDINGS` |
| **could not consult** | run present, `executionSuccessful: false`, ≥1 `error`-level `toolExecutionNotifications` | `status: unreachable` | **`DEGRADED`** — never `PASS` |
| not triggered | **run object omitted entirely** | `status: not-triggered` + `trigger_reason` | none |

Three rules that make this real rather than decorative:

**(a) "Could not consult" is itself `measured`, not `asserted`.** Record the probe you
ran and what it returned. `"the server seems down"` is an assertion about external
reality with no evidence, which is the exact thing this skill exists to stop:

```json
{ "level": "error",
  "message": { "text": "pmat MCP transport refused the connection; fell back to the CLI transport." },
  "properties": { "grounding": "measured",
                  "command": ["pmat", "query", "receipt guard", "--limit", "3"],
                  "exit_code": 0, "stdout_sha256": "…" } }
```

**(b) A working fallback does not erase a dead transport.** `pmat` is reachable two ways:
the CLI, and an MCP server (`pmat --mode mcp` starts the same analyzer). **The pmat MCP
server is `ConnectionRefused` in this environment right now.** If the CLI answers, the
consultation is honestly `consulted` — but record which transport answered and which one
did not:

```json
"pmat": { "status": "consulted",
          "transport": "cli",
          "transport_unavailable": ["mcp: ConnectionRefused"],
          … }
```

Without those two fields, "the MCP was refused and the CLI answered" and "everything was
fine" are the same receipt. That is the same defect one tier up from the one §3.0 is
about, and it is the reason this skill was written for the refused case rather than around
it.

**(c) You may not narrate around the encoding.** If you write "no CUDA concerns" in the PR
comment while `cuda.status` is `unreachable`, the artifact and the prose disagree and the
prose is the lie. Write the verdict the artifact supports.

### §3.A `pmat` — quality and duplication (**every PR, unconditional**)

**Precondition, non-negotiable.** Without an index, CB-200 is `Skip`, and **`Skip` is not
a pass** — it is `DEGRADED`:

```bash
pmat query "x" >/dev/null            # builds .pmat/context.db if absent
```

Measured on this repository from a cold worktree: **45.4 s, 87,592 functions in 10,317
files.** Budget for it; do not skip it.

**Index staleness is gating, not cosmetic.** Record the commit the index was built from
and prove the ancestry:

```bash
INDEX_COMMIT=$(git rev-parse HEAD)                 # at the moment the index was built
git merge-base --is-ancestor "$INDEX_COMMIT" "$HEAD_SHA" && A=true || A=false
```

`pmat` does not itself record which commit it indexed, so **you** stamp it, and you stamp
it from the worktree the index lives in. Two traps:

- Using another checkout's index. `/home/noah/src/aprender/.pmat/` belongs to whatever
  branch that checkout is on — measured today, **66 commits behind** and not an ancestor
  of this branch. An index built there answers about code that is not in this PR. That is
  the scar A4 exists for: `index_is_ancestor: false` with `verdict: PASS` is blocking
  class **B6** and the guard rejects it (fixture row 9).
- Recording ancestry you did not compute. The guard recomputes it and rejects a receipt
  whose recorded value disagrees, **whatever the verdict** — a receipt that misreports
  its own staleness is worse than a stale one.

If the tree was dirty when the index was built, the index describes uncommitted content;
record `index_worktree_dirty: true` beside it rather than pretending `HEAD` is exact.

Required output, all four arrays present even when empty:

```bash
pmat analyze complexity --format json --path .     # complexity_delta[] — INCREASES ONLY
pmat analyze tdg        --format json --path .     # tdg_delta[]        — A+…F per changed file
pmat analyze satd       --format json --path .     # satd_introduced[]  — markers added BY THIS DIFF
pmat query "<what the diff adds>" --limit 10       # duplication_hits[]
```

- `complexity_delta[]` is **increase-only** and comes from the AST/token walk `pmat`
  performs. Never compute it from a line scan: a formatter run would then read as a
  quality regression, and a gate that fires on `cargo fmt` gets routed around within a
  week.
- `satd_introduced[]` is markers **this diff added**, not markers the file already had.
  Diff the two SATD reports; do not report the file's standing debt as your finding.
- **`duplication_hits[]` is the highest-EV field in the receipt.** Before accepting that
  the diff adds something new, search for it: PERF-055 nearly re-implemented ~7,200 lines
  across 46 files that already existed. `pmat query "<the thing being added>"` and
  `pmat query "<same>" --duplicates` are two minutes that have paid for this entire skill
  once already.

**`pmat query` alone is HALF the search, and the receipt must say which half.** PRREV-007
measured it and PRREV-009 reproduced it on a second symbol:

| | measured |
|---|---|
| pmat's semantic index | **Rust-only.** `pmat query` for `arm_c_integrity` — a function *defined* at `scripts/perf_gate.sh:39` — returns 10 results, all `.rs`, and never that file. |
| the diff S3.A cites as its evidence | #2742: 46 files, 7,244 insertions, of which **3,533 (48.8%) are sh, py and yaml** — outside semantic reach entirely. |
| prior art on an unmerged sibling branch | **invisible by construction.** B6 requires `index_commit` to be an ancestor of HEAD, so the index can only hold this branch's history. #2781 found #2742's prior art because #2742 merged 17 hours earlier. Luck, not mechanism. |
| prior art that LANDED on `origin/main` after your merge base | **in neither region, and until F7 not even named.** Not on HEAD — your branch predates it. Not an unmerged sibling — it merged. #2781's blind region is exactly #2742: 1 commit, 46 files, 11 of them the prior art. One `git grep` over it costs **1 s** against 20 s for the 774-branch sweep, and it returns `crates/apr-cli/src/commands/test_llm_band.rs`. |

So run the second half as well, and record what each half reached:

```bash
scripts/pr_review_duplication_scan.sh --base "$BASE" --head "$HEAD_SHA" \
    --rust-semantic --json /tmp/dup.json      # --rust-semantic ONLY if you ran pmat query
jq -r '.duplication_coverage, .horizon_branches_scanned, .hits_total' /tmp/dup.json
```

It emits `duplication_hits[]`, `duplication_coverage{}`, `duplication_horizon[]`,
`horizon_branches_{total,scanned}`, `merge_base_to_main_files` and `symbols_searched` —
copy all of them into the `pmat` block verbatim.

**The horizon has THREE components and the receipt names all three** — `head=`,
`siblings=`, `merge_base_to_main=` — whether or not each was swept, because a region
that is absent from the horizon cannot be told apart from one that was searched and held
nothing. `duplication_coverage.merge_base_to_main` is the separate field that says which
of the three were actually reached, and `none` there cannot sit under a `PASS`. Measured cost on this repository: **18.6 s** over the full
772-branch horizon; 73 s on a 151-needle range. Put the wall time in `cost`.

Three rules the guard enforces on what you copy, all of them S3.0 applied one level down
— *"searched and found nothing" must not read the same as "could not search"*:

1. **Every surface carries a method**, from `{ semantic, lexical, none }`. A surface with
   no entry is REJECTED. `none` is honest and permitted.
2. **`none` anywhere ⇒ the verdict is not `PASS`.** Exactly the rule rows 5 and 6 apply
   to an unreachable consultation. Fixture rows 16/17 are the pair: the same coverage map
   is RED under `PASS` and GREEN under `DEGRADED`. Being honest costs you the PASS, never
   the receipt.
3. **A partial horizon is not a swept one.** `horizon_branches_scanned <
   horizon_branches_total` with `verdict: PASS` is REJECTED, and claiming the sibling
   branches with `scanned: 0` over a non-empty horizon is the `attempted: 0` shape.

**What the scan cannot do, and you must not imply otherwise.** It is an exact,
word-boundary name match. A re-implementation under a *different* name is invisible to
it; the sibling-branch half matches filenames only, not symbols. If you have reason to
think the diff re-implements something under a new name, say so as an `asserted` finding
with a rationale — never as `measured` off the back of this scan.

### §3.B NVIDIA CUDA documentation (triggered)

**Trigger** — any changed path matching `crates/aprender-gpu/**`,
`crates/aprender-serve/src/cuda/**`, `*cuda*`, `*ptx*`, `*cublas*`, `*fp8*`, `*nvrtc*`;
or a PR body / commit message matching `sm_\d+`, `cu[A-Z]\w+`, `cuda[A-Z]\w+`, or a GPU
architecture name.

**Do not evaluate the trigger by eye. Ask the guard**, which owns the patterns and has a
must-match / must-not-match case table behind them
(`tests/fixtures/pr-review/cuda-{path,message}-cases.tsv`). This repo's guard regexes have
been wrong six times; a table caught every one and review caught none:

```bash
git diff --name-only "$BASE" "$HEAD_SHA" | while IFS= read -r f; do
  bash scripts/check_pr_review_receipt.sh --match-path "$f" && echo "TRIGGER: $f"
done
MSG=$(git log --format=%B "$BASE..$HEAD_SHA")
bash scripts/check_pr_review_receipt.sh --match-message "$MSG" && echo "TRIGGER: message"
```

Read the status from the command, never from the tail of a pipeline — `$?` after a pipe
is the **last** command's status, and this repo has lost time to exactly that twice
(#2336, #2360).

**The trigger is deliberately over-broad and you will meet that.** It is a
case-insensitive match on `cuda` anywhere in the path, so a *fixture* named
`row-01-cuda-not-triggered-on-cuda-diff/` fires it — **eight of the 202 changed paths at
`0b7b876`**, all of them fixture filenames, recomputed with the guard's own predicate
(`check_pr_review_receipt.sh --match-path`) rather than counted by eye; the sentence this
replaces said "five", which was already wrong in the commit that shipped it. The number
names its commit, because a count of a moving diff is stale the moment the branch moves —
recompute it, do not read it:

```bash
BASE=$(git merge-base origin/main HEAD)
git diff --name-only "$BASE"..HEAD \
  | while IFS= read -r f; do
      bash scripts/check_pr_review_receipt.sh --match-path "$f" && echo "$f"
    done | wc -l
```

That is not a bug to route around. The correct response is `status: consulted` with
a `trigger_reason` naming the false-positive path and a `no-authority-found` query, **not**
`status: not-triggered`: the guard recomputes the trigger and rejects the receipt
(fixture row 1). A wide trigger costs one query; a narrowed one costs the 18% regression
that shipped on an ungrounded stream-ordering claim.

**Required output.** For **every device-behaviour claim**, one of:

- a `cited` entry — the query, the excerpt, the digest; or
- an explicit `no-authority-found` entry **naming the query that returned nothing**.

The second form is mandatory. Without it, *"the docs said nothing"* and *"I did not ask"*
are the same artifact — which is the whole of #2754, #2779, #2780 and #2790.

Consult via `mcp__nvidia-cuda-docs__search_cuda_docs`. If that server is unreachable,
that is row 3 of §3.0: `unreachable`, `executionSuccessful: false`, `DEGRADED`. It is not
a licence to answer from memory. #2765 (16-row alignment), #2789 (E4M3), #2771 (PTX
aliasing) and #2786 (GB10 Blackwell) were all asked of memory while the docs server sat
idle.

### §3.C CRUX — user-facing surface and competitive claims (triggered)

**Trigger**: the diff changes a CLI subcommand or flag, an HTTP route, an MCP tool, a
config key, or an output format.

**Scope is resolved (§10, 8.2 (a)): match on `category` + the changed surface. Do NOT run
semantic search over all 277 `contracts/crux-*.yaml`** — that is the resolution the spec
rejected, and it is also a 277-file read on every PR.

Required per surface: covering contracts, the competitor behaviour from each contract's
`competitor:` field, `gap_effect: closes | widens | none`, and — where nothing covers the
surface — `crux_coverage: none`, **which is itself a finding**, not an absence of one.

#### §3.C.1 Comparative performance claims — the `2.93× Ollama` rule

Any claim of the form "N× *competitor*" **anywhere** in the diff, the PR body, the docs or
the benchmark output must carry a complete comparator block:

```json
"comparative_claims": [
  { "claim": "1.21x llama.cpp on aarch64 Q4_K",
    "comparator": {
      "command": ["llama-cli", "-m", "…", "-n", "128", "-ngl", "99"],
      "version": "llama.cpp b4021",
      "env_sha256": "…", "artifact_sha256": "…",
      "log_path": "evidence/bench/<run-id>/comparator.log" } } ]
```

All five fields. **Absent any one of them the claim is reclassified `asserted`, marked
`unverified_comparative_claim`, and BLOCKS** (§7, class B4; fixture row 4). The book
published *"2.93× Ollama"* from a harness that never ran Ollama; this is the mechanism
that makes that unwriteable rather than merely discouraged.

Two things the guard checks that are easy to miss:

- A comparative ratio written into a **finding's message** while
  `comparative_claims` is empty is itself B4. You cannot state the ratio in prose and
  omit the provenance.
- B4 also reads the **diff**, not only your receipt, over the surface a user reads:
  `book/**.md` at any depth, and printed literals plus doc comments in shipped `.rs`.
  `book/src/examples/` is **in** that scope — it is 153 of the book's 441 published
  pages and the directory `851.8 tok/s = 2.93x Ollama` was actually published to.
- `version` and `artifact_sha256` are **captured, not remembered**. Run the comparator,
  read its version banner, hash the artifact you actually exercised. And never label a
  run by intent: `CUDA_VISIBLE_DEVICES` says what was *visible*, never what was *used*.

Before you record a ratio at all, ask whether one input produced it. **One failing input
is an anecdote** — four neighbouring prompts once inverted a diagnosis from "GPU
correctness defect" to "the gate sampled a near-tie" (#2359), and a first-reported 2.91×
on GB10 turned out to be a bimodal median whose honest value was 1.21× (#2567).

### §3.D Mutation — adversarial falsification (scoped)

| the diff touches | requirement | blocking |
|---|---|---|
| `scripts/check_*.sh`, `dogfood.sh`, `ci.yml` gate logic, or a `contracts/*.yaml` falsifier | **100% kill** on that guard's committed mutation set | **yes** |
| Rust source | `cargo mutants --in-diff <diff> --timeout 120 --jobs 4`, capped at `MUTANT_BUDGET=40` | advisory |
| docs / non-code | not triggered | — |

Bash guards are exercised with `bats-core` fixtures. For the receipt guard itself the
mutation set already exists and is a derivation, not a list:

```bash
bash scripts/mutate-guard.sh          # 219/219 on scripts/check_pr_review_receipt.sh
```

**`attempted: 0` with `status: consulted` is rejected** (fixture row 2). A mutation set
that matches nothing passes vacuously — the same shape as `pv lint <FILE>` returning PASS
over zero contracts. If nothing was attempted, the honest encoding is `unreachable` and
`DEGRADED`, not a clean run.

Record survivors as `{ "mutant": …, "file": …, "line": …, "killed": false }`. A surviving
mutant on a guard is not a scoring detail: it is a rule the guard *states* and nothing
*tests*.

---

### §3.E Antigravity — a second reviewer from a different vendor (**every PR**)

§3.A–§3.D ask *sources*. §3.E asks **a different reviewing agent**: different vendor,
different model family, its own process, its own tools. That is the only arm here whose
separation survives Huang et al. (ICLR'24) — a fresh session is a different *actor*, not a
different set of *weights*, and self-preference lives in the weights.

**It is ADVISORY. It cannot block anything, and nothing you write from it may say it does.**

#### Step 1 — resolve the binary, never invoke a bare `agy`

```bash
AGY=$(command -v agy) || AGY=""
if [ -z "$AGY" ]; then
  # UNAVAILABLE. Record it, verdict DEGRADED, and move on. This is the intended
  # behaviour on a box with no agy, not a rollout bug.
  arm_e_status=unreachable
  arm_e_reason="agy is not on PATH"
else
  # `sed -n 1p`, NOT `head -1`. head exits after the first line and hands the producer
  # SIGPIPE; under `set -o pipefail` the substitution then reports 141 for a command
  # that produced exactly the right answer. Four instances of that shape landed in this
  # repository in one day and one of them PASSED a safety check on the error. sed
  # without `q` reads to EOF and closes no pipe.
  AGY_VERSION=$("$AGY" --version 2>&1 | sed -n 1p)   # recorded verbatim
fi
```

`binary_path` in the receipt is `$AGY` — **what the resolution produced**, recorded per
run. That is the opposite of a hardcoded path, and it is the rule four coexisting `apr`
binaries taught this repository. Do not hardcode `/home/…/agy`; do not call bare `agy`.

#### Step 2 — PIN THE MODEL. This is the arm's correctness property, not a preference.

```bash
"$AGY" models        # the catalogue is the VENDOR's and it moves: 14 ids, then 11, hours apart on 2026-08-31
```

**Two of them are Claude** — `claude-sonnet-4-6` and `claude-opus-4-6-thinking` — beside
eleven Gemini ids and `gpt-oss-120b-medium`. **`agy` is a harness, not a model.** With
`--model` omitted, or pointed at a Claude id, §3.E is *you reviewing yourself* while every
field in the receipt still reads `antigravity`.

```bash
AGY_MODEL=gemini-3.1-pro-high     # or any non-Anthropic id `agy models` offers
```

The guard checks it:

```bash
scripts/check_pr_review_receipt.sh --match-arm-e-same-family "$AGY_MODEL"   # exit 1 = OK
```

Exit **0** from that predicate means the id is your own family and the receipt will be
**REJECTED [B1]**. This is the standing rule against labelling a run by intent: prove the
mechanism engaged. `model_family: google/gemini` written beside `--model
claude-opus-4-6-thinking` is `device: GPU` printed by a build with no CUDA in it.

#### Step 2b — what this arm can honestly claim about vendor identity. Read this before writing `model_family`.

**The pin records what was REQUESTED. Nothing in the artifact records what ANSWERED.**
Both output formats were checked, on 2026-08-31, and neither carries the model:

| format | what comes back | a model anywhere? |
|---|---|---|
| `--output-format json` | `conversation_id, status, response, duration_seconds, num_turns, json_schema, structured_output, usage` | **no** |
| `--output-format stream-json` | `init` (with `conversation_id`, `cwd`, `tools[]`, `permission_mode`), `step_update`, `result` | **no** |

So a silent server-side fallback — `--model gemini-3.1-pro-high` answered by something
else — leaves **no trace in anything the receipt can quote.** `stream-json` was checked
precisely because it looked like the place a model id would live; it is not.

**A self-report probe was measured too, and it is not a control.** One extra invocation,
same box, same day: `--model gemini-3.1-pro-high` answered *"Google Gemini 3.1 Pro"* and
`--model claude-sonnet-4-6` answered *"Anthropic Claude Sonnet 4.6"* — it discriminates.
It still cannot close the hole, for a reason no extra care fixes: it is a **claim by the
thing being identified**, about a **different invocation** than the review. Two calls, two
routes; the probe attests to the probe's route. Wiring it into the guard would put a
number in the column that shows which properties are checked, next to a property that is
not.

**Therefore, stated plainly and not to be softened later: §3.E's cross-vendor property is
UNVERIFIABLE FROM THE RECEIPT.** The pin is a real control — it stops the *accidental*
Claude route, which is the likely failure and the one the catalogue invites — and
`model_id` is a checkable argv value, which is why the guard checks it. What neither
delivers is evidence that a second vendor answered. **§13.1 makes vendor-distinctness the
load-bearing property of the quorum and encodes `|distinct vendor| ≥ 2`. That predicate
reads a producer's assertion, not an artifact, and §13's guarantee is weaker than the
argument for it assumes.** This is the same shape as a self-asserted `vendor` field failing
to establish cross-vendor identity: one key signs the whole document, and nothing in it is
evidence that a second vendor ever ran.

Closing it takes a field agy does not emit, or a second signature from a key bound to the
second vendor. Until one exists, write `model_family` as the label it is, and do not let
any sentence downstream read the receipt as proof the arm was cross-vendor.

#### Step 3 — run it in a DISPOSABLE tree, with the prompt and the diff as FILES

**This recipe was run. The one this file used to print was also run, and it reviewed
nothing while exiting 0.** Both failures below are measured, not anticipated.

```bash
# 1. A disposable copy of the PR head. `git archive | tar -x` gives agy a tree with no
#    .git, no remotes, no worktree, and no way back to your checkout.
REVIEW=$(mktemp -d)
git archive "$HEAD_SHA" | tar -x -C "$REVIEW"

# 2. The diff and the prompt go in as FILES. See "why not -p <the diff>" below.
git diff "$BASE_SHA" "$HEAD_SHA"                    > "$REVIEW/.pr-review-diff.patch"
cp .claude/skills/pr-review/agy-review-v1.schema.json "$REVIEW/.pr-review-schema.json"
cat > "$REVIEW/.pr-review-prompt.md" <<'EOF'
Review this pull request. The complete merge-base diff is in `.pr-review-diff.patch`
at the root of this workspace: read that file first, then read whatever source it
touches. Report findings under the required JSON schema. Every finding carries one of
the three grounding marks: cited, measured, or asserted.
EOF

# 3. $OUT MUST BE ABSOLUTE BEFORE THE cd. §4.1 defines it relative
#    ("evidence/pr-review/$PR/$HEAD_SHA"); redirecting to a relative $OUT after the cd
#    writes the transcript INTO the disposable tree, and step 4 then deletes the only
#    record of the run — silently, at rc 0. Same class as everything else on this page.
OUT=$(cd "$OUT" && pwd)

# 4. Run INSIDE the disposable tree. --dangerously-skip-permissions is safe HERE and
#    only here: the only directory agy can write is the copy.
cd "$REVIEW"
"$AGY" -p "$(cat .pr-review-prompt.md)" \
  --output-format json \
  --json-schema .pr-review-schema.json \
  --model "$AGY_MODEL" \
  --print-timeout 30m \
  --dangerously-skip-permissions \
  --add-dir "$PWD" > "$OUT/agy.json" 2> "$OUT/agy.err"
rc=$?                                  # ON ITS OWN LINE. Never through a pipe.
cd - >/dev/null && rm -rf "$REVIEW"
```

Measured end to end on #2803's real merge-base diff (144 325 bytes, 19 283 files
extracted): `rc 0`, `status SUCCESS`, `structured_output.reviewed true`,
`duration_seconds 350`, and `git status` in the real checkout **empty afterwards**.

**Why `--dangerously-skip-permissions`, and why the disposable tree is not optional.**
Print mode cannot prompt, so a tool needing permission is **auto-denied**. Measured with
this file's previous recipe, verbatim, on a real checkout:

```
rc=0                              <- clean
.status            = "SUCCESS"    <- clean
.response          = ""           <- empty
.structured_output   ABSENT       <- there was no review
usage                21 237 in / 544 out / 21 781 total   <- tokens really spent
stderr: "no output produced - a tool required the "command" permission that headless
         mode cannot prompt for, so it was auto-denied."
```

Only **stderr** told the truth. Note also which tools survive: `view_file` and
`grep_search` are auto-approved, so a read-only prompt returns a perfectly good review
and the recipe looks fine — until agy reaches for a command, which is exactly what a
finding marked `measured` requires it to do. **The flag without the disposable tree is
the wrong fix**: it hands a second agent write access to your working tree. The
`git archive` copy is what makes the flag safe, so the two ship together or neither does.

**Why not `-p "$(cat prompt-with-the-diff-inlined)"`.** It dies at `rc 127`,
`argument list too long`, and the limit is **not `ARG_MAX`** — this box reports
`getconf ARG_MAX` = 2 097 152 and still fails at 149 KB. The real cap is Linux's
`MAX_ARG_STRLEN`, 32 pages, on a **single** argument. Bisected here:

| argument bytes | result |
|---|---|
| 131 071 | rc 0 |
| **131 072** | **refused, `argument list too long`** |
| 144 325 (#2803's diff) | refused |

The exit code is the shell's, not the kernel's — zsh reports **127** (where `agy` itself
was measured), bash **126** for the same `E2BIG`. Assert the refusal, never the number.

No `ulimit` and no `ARG_MAX` raises it. Any diff past 128 KiB must arrive as a file the
agent reads — which is what step 2 above does, at a 310-byte prompt.

`--print-timeout` **defaults to `5m0s`** and a repository-scale review will exceed it.
**A timeout is `unreachable`, never "no findings."** agy fails slowly at least as readily
as it fails fast, and a slow failure is the one that looks like a clean run. Measured: a
*trivial* prompt took 44 s of `duration_seconds` and 54 s of wall clock.

**DO NOT TEST `.status`. IT IS NOT A LIVENESS SIGNAL** — measured, after the first draft
of this step said to test it:

```
rc=0  wall=54s
.status            = ERROR
.error             = "Your previous response contained an improperly formatted function
                      call. Please retry with a properly formatted function call.
                      Retries remaining: 3"
.structured_output = {"findings":[{…one well-formed finding…}],"reviewed":true}
```

`.status` reflects the last internal turn, so a **recovered** retry leaves it `ERROR` on a
run that finished and returned exactly what was asked for. A rule that treats
`.status != "SUCCESS"` as unavailable discards good reviews and reports `DEGRADED` for a
consultation that worked — a liveness check that fails closed on a healthy run, which is
the defect class this whole skill exists to catch, in the step that classifies it.

**AND THE OTHER POLARITY IS ALSO MEASURED, WHICH IS WHAT SETTLES IT.** The auto-denied run
above reported `.status: "SUCCESS"` over an empty response and no `structured_output` at
all. So `.status` has now been observed **wrong in both directions** — `ERROR` on a good
review, `SUCCESS` on no review — and no amount of reading it more carefully repairs that.
It is a diagnostic. It is not the test.

**The availability test is the artifact, not the label:**

```bash
if [ "$rc" -eq 0 ] && jq -e '.structured_output.reviewed == true' "$OUT/agy.json" >/dev/null 2>&1    && check-jsonschema --schemafile .claude/skills/pr-review/agy-review-v1.schema.json         <(jq '.structured_output' "$OUT/agy.json") >/dev/null 2>&1
then arm_e_status=consulted; else arm_e_status=unreachable; fi
```

Record `.status` and `.error` in the receipt as diagnostics — they are worth having — but
do not let them decide the three-state. Unavailable is: binary absent, `rc != 0`, missing
or schema-invalid `.structured_output`, `reviewed: false`, unparseable JSON, a permission
auto-denied in headless mode, or a timeout. **`rc == 0` is on neither side of that line.**

**Then RECORD THE TEST'S OWN RESULT, not the exit code.** The three conjuncts above go
into the receipt as three booleans, and the guard requires all three true under
`consulted`:

```yaml
output_check:
  structured_output_present: true    # the key existed at all
  reviewed: true                     # .structured_output.reviewed == true
  schema_valid: true                 # it validates against agy-review-v1.schema.json
```

A `consulted` receipt whose `output_check` is absent, non-boolean or false is **REJECTED
[B1]** (fixture row 36). The honest record of the auto-denied run is row 37: the same
`exit_code: 0`, the same `agy_status: "SUCCESS"`, the same duration — recorded
`unreachable`, verdict `DEGRADED`. Those two rows differ in nothing agy reported, only in
what the receipt claims about it.

The prompt gives agy the merge-base diff (§2) and asks for findings under
`.claude/skills/pr-review/agy-review-v1.schema.json`. Ask it for the **same three marks** §1 defines — a finding
with no mark is dropped, not guessed at.

#### Step 4 — read the record it produced, and record whose it is

```bash
jq -r '.usage | "\(.input_tokens) \(.output_tokens) \(.total_tokens)"' "$OUT/agy.json"
jq -r '.duration_seconds' "$OUT/agy.json"
```

agy's `usage` block is **real token accounting from the process that spent them**, which is
what §8's `cost_per_actionable` needs. Copy it into the receipt; the guard requires
`input_tokens`, `output_tokens` and `total_tokens` to be present and numeric.

**`total_tokens` is NOT the total.** Measured on the run above: `input 11163 + output 2327
= total 13490`, while `thinking_tokens 2176` and `cache_read_tokens 47698` sit **outside**
it — the cache reads alone are 3.5x the reported total. Copy all five fields, not three,
and when you divide by actionable findings say which numerator you used. A cost metric
built on a field named `total` that is not the total is a number that will be quoted for
years by people who never opened the receipt.

**`reverified_by_primary: false`.** agy's `measured` claims were measured by *agy*.
**You do not re-run them.** Re-running and adjudicating dissolves the independence the arm
exists to create — the disagreement disappears into your judgement, which is exactly what
step 5 exists to prevent. If you *did* re-run something, write `true` and carry the
commands as your own `measured` marks in the SARIF. What is forbidden is leaving it unsaid.

#### Step 5 — the divergence ledger. Disagreement is signal.

```json
"divergence": { "agreed": 0, "agy_only": 1, "primary_only": 2, "contradicted": 0 }
```

| column | means |
|---|---|
| `agreed` | agy raised it and so did you |
| `agy_only` | agy raised it, you did not |
| `primary_only` | you raised it, agy did not |
| `contradicted` | **you reached opposite conclusions on one subject** |

**`contradicted` is the column that matters and the one you will be tempted to leave at
zero.** A receipt that cannot represent two reviewers disagreeing is a receipt in which you
always win, and the disagreement leaves no trace anywhere.

The guard checks the arithmetic: `agreed + agy_only + contradicted == len(findings)`.
`primary_only` is outside the identity — it counts *your* findings, which are not in agy's
array. Do not "resolve" a contradiction by deleting agy's finding. Record both, mark yours
with your own grounding, and let §8 count it.

#### Step 6 — emit it advisory

Every `antigravity` result in the SARIF carries `"precision_class": "advisory"`. Level
`error` is fine — the arm is advisory about **authority**, not about volume. A `blocking`
class from this run is **REJECTED [B1]**: §7 admits a class to the blocking tier only while
its measured precision is ≥90% on a rolling sample, and §3.E has **zero** samples.

Nothing about this changes until `arm_e_actionable_rate` has 30 of them, and the change is
then a ticket editing `contracts/pr-review-skill-v2.yaml` — never a quiet edit here.

---

## §4 Emit the receipt

Two artifacts, under `evidence/pr-review/<pr>/<head_sha>/` — a branch with no PR number
yet uses `0000`, the convention PRREV-002's evidence already follows:

```
receipt.intoto.jsonl          # in-toto Statement v1, ONE JSON record on ONE line
receipt.intoto.jsonl.minisig  # detached minisign signature over it
findings.sarif                # SARIF 2.1.0, one run per CONSULTED consultation
```

> **Divergence from the spec, recorded rather than reconciled.** Spec §4 labels the
> receipt *"DSSE-wrapped"*, but §4.1 shows a **bare** in-toto Statement and §4.3 signs it
> with a **detached** `minisign` signature — a DSSE envelope carries its signatures
> inside, in `payload`/`payloadType`/`signatures`. The normative bodies (§4.1, §4.3) and
> the shipped guard both require the bare Statement: `check_pr_review_receipt.sh` reads
> `.predicateType` off the top level and validates the file against
> `schemas/in-toto-statement-v1.json`, so a DSSE envelope would be **rejected**. This
> file follows §4.1 + §4.3. If DSSE is wanted, it is a spec amendment plus a guard
> change, not a thing to improvise inside a receipt.

### §4.1 Order matters — SARIF first, then hash, then receipt

`findings_ref.sha256` must equal `sha256(findings.sarif)`, so the SARIF must be **final**
before the receipt is written. Editing the SARIF afterwards silently invalidates the
receipt; the guard catches it (positive control `findings-digest`), but only after you
have wasted the run.

```bash
OUT="evidence/pr-review/$PR/$HEAD_SHA"; mkdir -p "$OUT"
# 1. write $OUT/findings.sarif, complete.
# 2. hash it — read the status from sha256sum, not from a pipeline tail:
FINDINGS_SHA=$(sha256sum "$OUT/findings.sarif" | cut -d' ' -f1)
# 3. write $OUT/receipt.intoto.jsonl with that digest, as ONE line:
jq -c . receipt.pretty.json > "$OUT/receipt.intoto.jsonl"
```

`jq -c` is not cosmetic. The guard counts **records**, not newlines, and requires exactly
one: a pretty-printed Statement is many lines and is rejected as "holds N JSON records".

### §4.2 The receipt skeleton

```json
{ "_type": "https://in-toto.io/Statement/v1",
  "subject": [ { "name": "git+https://github.com/paiml/aprender",
                 "digest": { "sha1": "<head_sha>" } } ],
  "predicateType": "https://paiml.dev/attestations/pr-review/v2",
  "predicate": {
    "skill_version": "2.1.0",
    "attestation_level": "L1-self",
    "pr": 2783,
    "base_sha": "<merge-base>", "head_sha": "<head>",
    "author_actor":   { "kind": "agent", "id": "agent:<model>/<authoring-session>" },
    "reviewer_actor": { "kind": "agent", "id": "agent:<model>/<review-session>" },
    "affected_crates": [],
    "verdict": "PASS|FINDINGS|DEGRADED|BLOCK",
    "degraded_reason": "<one clause; required whenever verdict is DEGRADED — §9 reads it>",
    "consultations": {
      "pmat":     { "status": "consulted", "transport": "cli",
                    "transport_unavailable": ["mcp: ConnectionRefused"],
                    "index_commit": "…", "index_is_ancestor": true,
                    "complexity_delta": [], "tdg_delta": [],
                    "satd_introduced": [], "duplication_hits": [], "cache_hits": 0,
                    "duplication_coverage": { "rust": "semantic", "shell": "lexical",
                        "python": "lexical", "config": "lexical", "docs": "lexical",
                        "other": "lexical", "sibling_branches": "lexical",
                        "merge_base_to_main": "lexical" },
                    "duplication_horizon": ["head=<head_sha>",
                        "siblings=refs/remotes/origin/* unmerged into origin/main",
                        "merge_base_to_main=<base_sha>..refs/remotes/origin/main"],
                    "horizon_branches_total": 0, "horizon_branches_scanned": 0,
                    "merge_base_to_main_files": 0,
                    "symbols_searched": 0 },
      "cuda":     { "status": "…", "trigger_reason": "…", "queries": [] },
      "crux":     { "status": "…", "surfaces": [], "contracts": [],
                    "gap_effect": "none", "crux_coverage": "covered",
                    "comparative_claims": [] },
      "mutation": { "status": "…", "scope": "guard|in-diff|not-triggered",
                    "attempted": 0, "killed": 0, "survivors": [] },
      "antigravity": { "status": "consulted|unreachable", "attempted": 1,
                    "agy_version": "agy 1.1.22",
                    "binary_path": "<what `command -v agy` resolved to>",
                    "model_id": "gemini-3.1-pro-high",
                    "model_family": "google/gemini",
                    "exit_code": 0, "duration_seconds": 0,
                    "agy_status": "<.status, a DIAGNOSTIC - measured wrong BOTH ways>",
                    "usage": { "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 },
                    "output_check": { "structured_output_present": true,
                                      "reviewed": true, "schema_valid": true },
                    "reverified_by_primary": false,
                    "divergence": { "agreed": 0, "agy_only": 0,
                                    "primary_only": 0, "contradicted": 0 },
                    "findings": [] } },
    "findings_ref": { "path": "findings.sarif", "sha256": "<sha256 of findings.sarif>" },
    "cost": { "input_tokens": 0, "output_tokens": 0, "wall_seconds": 0 } } }
```

Fields the guard checks that are easy to forget:

- `subject[0].digest.sha1` **must equal** `predicate.head_sha`. A receipt whose subject is
  a different commit reviews a different commit.
- `attestation_level` is **`L1-self`**, always. A skill invoked by the agent that wrote
  the code is self-attestation. SLSA Build L3 requires an isolated builder the tenant
  cannot influence; claiming it here would be the enforcement theater this repo names as
  its dominant failure mode (spec R1).
- `cost` needs all three numeric fields. **Record-only is not unenforced** — §8's four
  continuous metrics can only be ratcheted from a measured baseline, and a metric nobody
  records is one nobody can ratchet.
- `reviewer_actor.id != author_actor.id` (§5, class B2, fixture row 8).
- `skill_version` is **no longer decoration**: it selects the rule set the receipt is judged
  by. At `2.1.0` and above the `antigravity` block is **required** (§3.E); a `2.0.0` receipt
  predates the arm and is judged by `2.0.0`'s rules, which is why this repository's one real
  receipt still validates instead of being back-filled with a consultation nobody performed.
  Writing `2.0.0` to skip §3.E is a **stated bypass**, owed to `PRREV-016` — do not use it.
- `antigravity.model_id` must not be your own model family (§3.E step 2, fixture row 35).
  `agy models` lists two Claude ids; agy is a harness, not a model.
- `antigravity.attempted: 0` under `status: consulted` is rejected (fixture row 31), exactly
  as `mutation.attempted: 0` and `cuda.queries: []` are.

### §4.3 Signing — and the sentence that must accompany it

```bash
minisign -S -s "$PR_REVIEW_SIGNING_KEY" -m "$OUT/receipt.intoto.jsonl"
minisign -V -m "$OUT/receipt.intoto.jsonl" -p .github/pr-review.pub    # verify your own
```

Repo-local key. **No Fulcio, no Rekor, no external transparency log** — an external SaaS
in the verification path violates the sovereign constraint (spec R2). The transparency log
is `evidence/pr-review/` under git.

> **The signature proves the receipt came from the CI environment. It does not prove the
> review was honest.** State this in the PR comment, every time. Cryptographic assurance
> of provenance is not assurance of diligence, and conflating them would be the most
> sophisticated form of theater this repository has yet produced — the spec says so in
> those words, and it is the reason `attestation_level` reads `L1-self` and not something
> more impressive.

A signed receipt whose consultations are all `not-triggered` on a 4,000-line GPU PR is
perfectly valid cryptography and a worthless review. The guard catches some of that
(fixture rows 1, 2, 11, 14). It cannot catch all of it. You are the rest.

### §4.4 Validate your own receipt before you post

Non-optional. The guard runs its positive controls first, so a green from it means
*"malformed receipts are still being rejected AND yours was accepted"*:

```bash
bash scripts/check_pr_review_receipt.sh "$OUT"      # exit 0 = ACCEPT
```

A `REJECT` line names exactly one blocking class (`B1`, `B2`, `B4`, `B6`) and the branch
that fired. Fix the receipt. **Do not fix the guard** — if you believe the guard is wrong,
that is a mutation-verified change to `scripts/check_pr_review_receipt.sh` plus a fixture,
in its own PR, not an edit made to get your own review through.

---

## §5 Author / reviewer separation

The review is a **separate invocation** from the authoring session: fresh context, an
adversarial stance, and no access to the author's reasoning trace.

| control | rule |
|---|---|
| distinct actor | `reviewer_actor.id != author_actor.id` — RED if equal |
| grounded findings | ≥1 `measured` or `cited` finding on any PR over 200 changed lines |
| independent audit | 10% of receipts re-reviewed, results under `evidence/pr-review/audit/` |

Self-review is worth close to nothing: intrinsic self-correction degrades reasoning
(Huang et al., ICLR'24), models self-prefer, and a same-model critic yields ≈0 gain. If
you wrote the code, **you are not the reviewer** — say so and stop, rather than emitting a
receipt whose two actor ids you had to invent to differ.

Audit divergence is recorded only. **No threshold until 30 audited receipts exist** — §8
is instrument-first, and a threshold invented before the measurement is the thing this
repo has most often mistaken for rigour.

---

## §6 Verdict

| condition | verdict |
|---|---|
| any consultation `unreachable`; or a stale index; or a skipped `affected_crates` entry; or `mutation.attempted: 0` | `DEGRADED` |
| any blocking class (§7) fires | `BLOCK` |
| findings present, none blocking | `FINDINGS` |
| all triggered consultations ran, nothing found | `PASS` |

`DEGRADED` and `PASS` are not interchangeable and the guard enforces it: `unreachable`
plus `PASS` is a rejection (fixture row 5), `unreachable` plus `DEGRADED` is accepted
(fixture row 6). **Rows 6, 7 and 14 are discrimination cases** — they exist so that
"reject every receipt" cannot read as a working guard, which is the over-reach that
already bit PERF-055 and the #2766 delta-gate work.

## §7 Blocking policy — six objective classes, and nothing else

| class | condition |
|---|---|
| B1 | missing / schema-invalid / unsigned / internally inconsistent receipt |
| B2 | `reviewer_actor == author_actor` |
| B3 | guard mutation score < 100% on a guard-touching PR |
| B4 | `unverified_comparative_claim` |
| B6 | `index_is_ancestor: false` with `verdict: PASS` |

Plus: breaking API surface with no semver bump (B5 — no consultation emits it today, and
the contract records that rather than pretending otherwise).

Every one is machine-decidable. **None is a judgement call**, so none can freeze an active
investigation the way #2757 and #2766 did. A `FINDINGS` verdict proceeds on a feature
branch and blocks on a release branch — the release branch is the last boundary and the
routed-around argument does not apply where the alternative is shipping.

**§3.E is NOT in this table, and cannot be.** The admission rule below needs a measured
precision on a rolling sample; §3.E has zero samples. An `antigravity` finding claiming
`precision_class: blocking` is refused as an internally inconsistent receipt — **B1**, not a
seventh class. Promotion happens after 30 samples, by editing
`contracts/pr-review-skill-v2.yaml`, and it is a ticket.

**Admission rule.** A class may block only while its measured precision on the rolling
sample is ≥90% (Tricorder's ≤10% effective-false-positive bar). A class that falls below
is demoted to advisory by **editing `contracts/pr-review-skill-v2.yaml`** — a ticket, not
a silent config change, and never by disabling the gate.

## §8 What to record, and what not to threshold

Four zeros and ones — the line stops on these:
`guard_mutation_score = 100%` · `receipt_presence = 100%` · `unmarked_claims = 0` ·
`vacuous_consultations = 0`.

Four instrument-first parameters — **record, do not threshold**, until 30 samples exist:
`actionable_rate` · `effective_fp_rate` · `audit_divergence` · `cost_per_actionable`.

Inventing a number for the second group is the failure this repo has repeated most: a
`3 × pooled stddev` bench threshold returned GREEN on the only regression on record and
its power *fell* as data accumulated (#2675). Fill `cost` honestly and let the baseline
come from measurement.

---

## §9 The PR comment (spec §12)

One line, exactly this shape:

```
pr-review v2.1.0 | verdict=<V> | consultations: pmat=<s> cuda=<s> crux=<s> mutation=<s> agy=<s>
| findings=<n> (cited=<a> measured=<b> asserted=<c>) | index=<sha7> ancestor=<bool>
| agy=<model_id> advisory | divergence: agreed=<a> agy-only=<b> primary-only=<c> contradicted=<d>
| receipt=evidence/pr-review/<pr>/<sha>/receipt.intoto.jsonl (L1-self, signed)
```

**Name the MODEL, not the tool.** "agy" alone does not even say which *family* was asked
for, which is the first thing a reader needs. It is the **requested** id and not proof of
what answered — §3.E.2b — so read the line as "this is what was asked for", never as
"a second vendor reviewed this". **Print `advisory` literally**, so nobody
reads an agy finding as a merge blocker. And put the divergence counts on the line: a
disagreement that has to be dug out of a JSON file is one that gets resolved in your favour
by default.

**`DEGRADED` puts the reason FIRST**, before anything else on the line — a reader who
stops after six words must still learn that this review did not fully run:

```
DEGRADED: agy timed out at --print-timeout 30m | pr-review v2.1.0 | verdict=DEGRADED | …
```

**Generate the line from the receipt. Do not type it.** A hand-written summary can
disagree with the artifact it summarises, and then the prose is the thing people read:

```bash
jq -r --slurpfile s "$OUT/findings.sarif" '
  .predicate as $p | ([$s[0].runs[]?.results[]?]) as $r |
  (if $p.verdict=="DEGRADED"
     then "DEGRADED: " + ($p.degraded_reason // "reason not recorded") + " | " else "" end) +
  "pr-review v" + $p.skill_version + " | verdict=" + $p.verdict +
  " | consultations: pmat=" + $p.consultations.pmat.status +
  " cuda=" + $p.consultations.cuda.status +
  " crux=" + $p.consultations.crux.status +
  " mutation=" + $p.consultations.mutation.status +
  " agy=" + ($p.consultations.antigravity.status // "absent") +
  " | findings=" + ($r|length|tostring) +
  " (cited="    + ([$r[]|select(.properties.grounding=="cited")]   |length|tostring) +
  " measured="  + ([$r[]|select(.properties.grounding=="measured")]|length|tostring) +
  " asserted="  + ([$r[]|select(.properties.grounding=="asserted")]|length|tostring) + ")" +
  " | index=" + ($p.consultations.pmat.index_commit // "none" | .[0:7]) +
  " ancestor=" + ($p.consultations.pmat.index_is_ancestor|tostring) +
  ($p.consultations.antigravity as $a
   | if $a == null or $a.status != "consulted" then "" else
     " | agy=" + ($a.model_id // "MODEL NOT RECORDED") + " advisory" +
     ($a.divergence as $d | if $d == null then "" else
        " | divergence: agreed=" + ($d.agreed|tostring) +
        " agy-only="    + ($d.agy_only|tostring) +
        " primary-only="+ ($d.primary_only|tostring) +
        " contradicted="+ ($d.contradicted|tostring) end) end) +
  " | receipt=evidence/pr-review/" + (($p.pr // "none")|tostring) + "/" + $p.head_sha +
    "/receipt.intoto.jsonl (L1-self, signed)"' "$OUT/receipt.intoto.jsonl"
```

**The §3.E half of the line names the MODEL, not the tool** — `agy` alone does not even say
which family was asked for, and that is the first thing a reader needs (§3.E.2). It is the
**requested** id: agy emits no model field in either output format, so the line records
what was asked for and **not** that a second vendor answered (§3.E.2b).
`advisory` is a literal, so nobody reads an agy finding as a merge blocker. The divergence
counts are on the line rather than only in the receipt: a disagreement that has to be dug
out of a JSON file is one that gets resolved in the primary's favour by default.

A receipt with no `antigravity` block prints `agy=absent` and no §3.E clause — that is a
`2.0.0` receipt (§3.E.8), and the line says so instead of failing to render. An
`unreachable` arm prints `agy=unreachable` in the consultation list and **no model clause
either**, because no model answered: printing `agy=<model> advisory` for a run that never
happened would name a reviewer that did not review. The `DEGRADED:` prefix already leads
the line in that case.

Note what the counts do: `findings=3 (cited=1 measured=2 asserted=0)` is read off the
SARIF, so a review that found things but grounded none of them cannot hide behind a
sentence. A line reading `asserted=n` with `cited=0 measured=0` is a review that consulted
nothing, and it says so in its own summary.

Immediately beneath the line, both sentences, every time:

> The signature proves this receipt was produced in the CI environment. It does **not**
> prove the review was honest or complete — `attestation_level` is `L1-self`.
> Verify it yourself: `bash scripts/check_pr_review_receipt.sh evidence/pr-review/<pr>/<sha>`

Then the findings, each with its grounding mark visible. A finding whose mark is not shown
to the human is an unmarked claim wearing a receipt.

---

## §10 Ways this review can be theater, and what stops each

| the failure | what stops it |
|---|---|
| A confident claim from stale memory | §1 — three marks, no fourth; the guard rejects an unmarked claim |
| "The docs said nothing" ≡ "I didn't ask" | §3.B `no-authority-found`, naming the query |
| An unreachable source reading clean | §3.0 row 3 + fixture rows 5/6 |
| A dead transport hidden by a live one | §3.0 (b) — `transport` / `transport_unavailable` |
| Re-implementing what already exists | §3.A `duplication_hits` — PERF-055 |
| A competitor ratio with no comparator | §3.C.1 + B4 — the 2.93× Ollama scar |
| An index answering about other code | §3.A ancestry + B6 — the 66-commit drift |
| Self-review flattering itself | §5 + B2 |
| A green run that attempted nothing | `attempted: 0` rejected; `Skip` is not a pass |
| A signed receipt read as an honest one | §4.3, stated in the comment, `L1-self` |
| **This skill being shadowed and never running** | the explicit `name:` in the frontmatter |

The last row is not hypothetical: a user-scope `~/.claude/skills/dogfood/` shadowed this
repo's release-certifying skill, and hardening it edited a file that never ran (#2361).
If a change to this file appears to have no effect, ask what else claims the name before
asking what else is wrong with the change.

## §11 Do not

- **Do not** write a verdict the artifact does not support. The artifact is the review;
  the prose is a rendering of it.
- **Do not** relax the guard to pass your own receipt.
- **Do not** answer a CUDA device-behaviour question from memory because the docs server
  was slow.
- **Do not** run semantic search over all 277 CRUX contracts (§10, 8.2 resolved as (a)).
- **Do not** compute `complexity_delta` from a line scan.
- **Do not** read an exit status through a pipe, and do not pipe a mutating command into
  a truncating filter: `producer | grep -q X` can return 141 on SIGPIPE **despite a
  match**, and a `git commit` piped into `head` dies silently with the file still staged
  and HEAD unmoved.
- **Do not** claim `attestation_level` above `L1-self`.
- **Do not** emit a receipt at all if you authored the diff. Say so, and hand it to a
  different invocation.

---

## §12 State of this skill on the commit that adds it

Written down because a skill that reads as operational while one prerequisite is missing
is the #2504 shape, and the remedy this repo settled on is to put the state where a reader
cannot miss it.

**The emit recipe above is executed, not described.** `evidence/prrev-005/` holds the
transcript: a receipt built by following §4 step for step, over the real repository at
`5928ec2a7`, was **ACCEPTED** by `scripts/check_pr_review_receipt.sh` with all four of the
guard's positive controls firing first. Four negative controls were then run against the
same real diff — not the synthetic fixture repo — each re-signed so it could only fire on
the branch it names:

| mutation of the accepted receipt | guard |
|---|---|
| `cuda.status: not-triggered` | REJECT — *"its S3.B trigger fires on this diff"* |
| `verdict: PASS` with `mutation.status: unreachable` | REJECT — *"unreachable but the verdict is PASS"* |
| a finding's `properties.grounding` deleted | REJECT — *"carries no properties.grounding"* |
| `excerpt_sha256` corrupted | REJECT — *"excerpt_sha256 … is not verified"* |

Those are §3.0's three-state rule, §1's grounding mark and §1.1's verified citation,
each shown to be load-bearing on this repository rather than on a fixture.

**Both of the things that used to be missing have landed.**

1. **`.github/pr-review.pub` is committed** (PRREV-013). The guard defaults
   `PR_REVIEW_PUBKEY` to it, and the receipt under
   `evidence/pr-review/2795/f5fe147.../` now **ACCEPTs under that default with no
   override** — which it did not when this section was first written. The conformance
   run described above passed only with `PR_REVIEW_PUBKEY` pointed at a throwaway key;
   against the repository default the same receipt was
   `REJECT [B1] public key .github/pr-review.pub is absent`. A default no receipt can
   satisfy is not a default, and the ownership of fixing that was orphaned across three
   files naming two different tickets.

   The secret half is **not in this repository and never will be**. It is held by whoever
   runs the reviewer, at the path `$PR_REVIEW_SIGNING_KEY` names, and a copy is escrowed
   in the repository secret `PR_REVIEW_SIGNING_KEY_B64` (base64 of the minisign
   secret-key *file* — §4.3's `minisign -S -s` takes a path, so a CI signer materialises
   it before use). Rotate with `minisign -G -W`, replace `.github/pr-review.pub`, re-set
   the secret, re-sign.

2. **`ci.yml` invokes the guard** — job `pr-review-receipt`: the stated-count guard, the
   receipt guard over one GREEN and one RED fixture, the bats fixture table, the guard's
   own mutation set, and this PR's own receipt. Job-level `if:`, no workflow-level
   `paths:` filter, with `check_pr_review_wiring.sh` checking both polarities of that rule
   mechanically rather than in a comment.

   Arm 4 — this PR's own receipt — is now **armed and able to fail**. It used to begin
   `if [ ! -f .github/pr-review.pub ]; then ... exit 0; fi` over a key nothing owned, so
   it exited 0 on every run it ever had: a gate that cannot fail, inside the job built to
   prevent gates that cannot fail, holding up `receipt_presence`, which §8 fixes at 100%
   with no ratchet. Its *armed* branch was no better — it looked for
   `evidence/pr-review/<pr>/<TIP SHA>`, which no pull request can produce, because
   committing the receipt changes the tip. Both are fixed in
   `scripts/check_pr_review_arm4.sh`, whose `--self-test` drives a hermetic case table —
   against the deterministic fixture repo, not against this repository's history, so a
   squash-merge cannot expire it — including the absent-key row that used to read green.

**The backtest ran, three times, and the first two failed.** §9 step 7 is the acceptance
test for the whole design, and it is recorded rather than summarised: PRREV-007 scored
**1 of 3**, PRREV-011 **2 of 3**, PRREV-012 **3 of 3**. Each transcript is committed
verbatim under `evidence/pr-review/backtest/` — `results.md`, `results-v2.md`,
`results-v3.md` — rather than replaced by the run that passed. The eight defects the two
failures found (F1–F8) are what the guard now enforces; F9 is measured, unfixed, and named
with its counterfactual. Genchi genbutsu: the verdicts came from running the guard against
the real merged commits, not from the spec's reasoning about them.

---

## §13 Autonomous merge on quorum (DESIGNED AND BUILT, **NOT ARMED**)

Spec §13. Operator instruction, 2026-08-31: PRs auto-merge once the review quorum passes.
The mechanism exists — `scripts/pr_review_quorum_arm.sh`, a table of 83 rows, a 134-mutant
set at 100% — and **it is reachable from no workflow that can merge anything.** §13.11 is
the arming ladder; rung 0 is where this file is written.

**Read this before doing anything else in this section.** Every other part of this skill
makes a review harder to fake. §13 lets a merge happen with nobody watching, so a
dishonest review stops being something a human reads and becomes something that ships.
§13 therefore **adds zero rows to §7**. When the arm script refuses, the pull request is
exactly as green as it was and a human merges it. The refusal classes are lettered
**Q1..Q10** so a log line cannot be mistaken for a B-class block.

### §13.1 What you record in the receipt

Two new blocks. Both are **optional**: a receipt without them is a perfectly good receipt
that does not authorise an unattended merge, and its absence is not a default to yes.

```json
"autonomy": {
  "requested": true,
  "main_sha_at_review": "<origin/main tip WHEN YOU REVIEWED>",
  "quorum": [
    { "role": "primary",      "vendor": "anthropic",
      "actor": { "kind": "agent", "id": "agent:claude-opus-5/session-review" },
      "verdict": "PASS", "refusal": null },
    { "role": "cross_vendor", "vendor": "google",
      "actor": { "kind": "agent", "id": "agent:agy-1.1.22/session-cross-vendor" },
      "verdict": "PASS", "refusal": null }
  ],
  "delta_sweep": { "status": "clean",
                   "region": "<main_sha_at_review>..refs/remotes/origin/main",
                   "needles_sha256": "<sha256 of the needle list, joined by \n>",
                   "hits": [] }
}
```

and, inside `consultations.pmat`:

```json
"duplication_needles": ["fused_kernel_launch", "stream_ordering_guard", "..."]
```

**Four things that are easy to get wrong and are refused if you do.**

1. `main_sha_at_review` is the tip of `origin/main` **at review time**, not at merge time.
   The queue runs at about one PR an hour with `max_entries_to_build: 1`, so hours pass
   and `main` moves; the region between the two is unswept, and before §13 it was also
   unnamed. If that region is non-empty you owe a **delta sweep**, and the sweep must
   **replay** the needle set — `needles_sha256 = sha256(join(duplication_needles, "\n"))`.
   Re-deriving needles from the diff is a second implementation of §3.A's derivation,
   each green against its own copy, which is the exact defect §3.A exists to catch.
2. `vendor` is what makes the quorum a quorum. Two `anthropic` members raise the count
   and not the independence, and the predicate reads `|distinct vendor| ≥ 2`.
3. No quorum member may carry the author's actor id — not merely `reviewer_actor`.
4. `refusal: null` means "no reservation". One non-null refusal ends it. No member may
   clear another member's refusal.

### §13.2 What the cross-vendor reviewer may do

`agy` is **advisory** under §7 and may not block. §13 gives it exactly one power: it may
refuse the unattended merge. Record that as a property on an advisory SARIF result:

```json
"properties": { "grounding": "asserted", "precision_class": "advisory",
                "autonomy_effect": "refuse", "rationale": "...", "failure_scenario": "..." }
```

Nothing about §7's tier changes. A cross-vendor reviewer that could neither block nor
refuse would be a consultation and not a member.

### §13.3 Running the arm script

```bash
# SHADOW MODE. Evaluates the §13.2 predicate, prints PERMIT or REFUSE [Qn], merges nothing.
bash scripts/pr_review_quorum_arm.sh --explain --pr 2795

# HERMETIC (what the fixture table does): everything from artifacts, no network.
bash scripts/pr_review_quorum_arm.sh --explain --pr 2783 \
     --receipt tests/fixtures/pr-review/q-52-permits-a-clean-quorum \
     --context tests/fixtures/pr-review/q-52-permits-a-clean-quorum/pr-context.json

# THE ARMING VERB. Runs `gh pr merge <N> --squash --auto` on PERMIT and nothing else.
bash scripts/pr_review_quorum_arm.sh --pr 2795
```

Exit codes: **0** permitted (and armed, unless `--explain`), **1** refused, **2** the box
cannot answer. That third code is not decoration — three guards in one day here named a
CODE cause for a broken box, and one of them blocked every PR.

It is **idempotent**: an already-armed PR prints `ALREADY-ARMED` and calls nothing.

### §13.4 DEGRADED does not auto-merge, and you do not work around it

A `DEGRADED` receipt refuses under **Q6**. Do not "fix" throughput by recording
`not-triggered` where you meant `unreachable` — §3.A's trigger is unconditional, the
receipt guard rejects a `not-triggered` pmat outright, the arm script refuses any pmat
status other than `consulted`, and an empty `trigger_reason` on any consultation is
refused too.

**Check the transport before you believe the source is down.** Measured on this box,
2026-08-31: `pmat 3.34.0` is on `PATH`, `.pmat/context.db` exists, and
`pmat query "receipt validation" --limit 3` exits 0 over 84,919 functions in 10,136 files
— while the `pmat` **MCP server** refuses connections in the same session. §3.A's
precondition is a shell command, not an MCP tool call. A DEGRADED verdict recorded because
one transport was down is a misattribution, and `PRREV-016` owes the `transport:` field
that would make it visible.

### §13.5 The refusal classes

| class | it refused because |
|---|---|
| **Q1** | the receipt is missing, unreadable, unsigned, invalid (delegated to the receipt guard), or carries no autonomy block |
| **Q2** | a consultation did not run, ran over nothing, left a surviving mutant, or a guard-touching diff has no `scope: "guard"` 100% run |
| **Q3** | a commit after the reviewed SHA touches something outside `evidence/pr-review/<pr>/**` |
| **Q4** | `main` moved and the delta region was not swept clean, or the sweep did not replay the needle set |
| **Q5** | single-vendor quorum, missing role, non-unanimous, or the author sits in it |
| **Q6** | the verdict is not PASS, a finding blocks, a claim carries no mark, or a tool did not execute |
| **Q7** | the cross-vendor reviewer set `autonomy_effect: refuse` |
| **Q8** | the kill switch on `origin/main`, the `autonomy-hold` label, or an open CHANGES_REQUESTED |
| **Q9** | the PR does not target `main`, or its diff touches `MECHANISM_PATHS` |
| **Q10** | `workspace-test` or the gate check is not `success` |

**A refusal is not a block.** Say so when you report one.

### §13.6 The kill switch

`.github/pr-review-autonomy.disabled` on `origin/main` refuses everything at every rung.
It is read from `origin/main` and **never** from the PR tree, so a branch that deletes it
does not turn autonomy back on. It is the first clause of the repository phase, because an
off switch consulted only after five other checks have passed costs a minute of compute to
use.

### §13.7 What §13 will not do for you

- It does not verify that a review was **honest**. `attestation_level: L1-self` still says
  the skill was invoked by the authoring agent, and a signature is a provenance claim.
- It does not read the **PR body**. `gh` produces `base_ref`, labels, reviews, checks and
  the tip, and nothing else.
- It does not check **entailment**. That is Phase 3 (§1.1).
- It has **zero samples**. `autonomous_merge_reverts` is `0/0`, which is **undefined, not
  clean** — the same shape as the coverage pipeline that read `0/0` for months.

### §13.8 This PR cannot auto-merge itself

`MECHANISM_PATHS` covers the skill, the key, the kill switch, `.github/workflows/`, the
specification, both contracts, `schemas/`, the six `check_pr_review_*`/`mutate*` scripts,
the arm script, the duplication scanner, the fixture tree and both bats files. The pull
request that introduces §13 touches every one of them, so §13 refuses it under **Q9**.
That is the first falsifiable property of the section, and `q-44` plus the
`MECHANISM_PATHS` bats row are what hold it.

### §13.9 Verifying the mechanism

```bash
bats tests/pr-review-quorum.bats            # 83 rows: one per refusal path, four that PERMIT
bash scripts/mutate_quorum_arm.sh           # 134/134 — §13.10 fixes this at one, no ratchet
bash scripts/mutate_quorum_arm.sh --list    # the catalogue, no mutants run
```

The mutation set is **derived** — one `drop` and one `flip` per `refuse Q<n>` site,
rescanned on every run — so a refusal added to the arm script is mutated the next time this
runs without anybody remembering to add it. Two branches were rewritten during the build
because the set proved them unreachable rather than because anybody read them.
