# PR-REVIEW-SKILL-001 — adversarial PR review with three mandatory consultations

**Status:** SUPERSEDED by PR-REVIEW-SKILL-002-v2.md (2026-08-30). Retained for the audit trail.
**Original status:** DRAFT for review
**Author:** @noah (requested), drafted 2026-08-30
**Supersedes:** nothing. Complements `/code-review`, `nightly-ux-crux`, and the 102 `check_*.sh` guards in `ci.yml`.

---

## §0 The failure this exists to prevent

A reviewer — human or model — reasons from memory about a domain where memory is stale, and ships a decision nobody can trace back to a source.

This is not hypothetical in this repository. During APR-PERF-GATE-001, over roughly 36 hours:

- An 18% throughput regression was shipped on a CUDA stream-ordering claim **asserted from memory**. The claim was correct — verified later against the CUDA Programming Guide §2.5.8 — but it was not grounded at the time of the decision.
- `cublas_prefill_fp8_gemm`'s 16-row alignment requirement (#2765), E4M3 semantics (#2789), and PTX register aliasing (#2771) were all reasoned about without consulting NVIDIA documentation once.
- A GB10 Blackwell deficit (#2786) was diagnosed as a "real deficit" without checking whether Blackwell has documented architectural differences on that path.

**Every one of those was a CUDA question, and the CUDA documentation server was available and unused.**

### §0.1 GROUNDING RULE

Every claim this skill makes about external reality is one of:

- **`cited`** — traced to a consultation this run performed, with the query and the returned excerpt recorded
- **`measured`** — produced by a command this run executed, with the command and output recorded
- **`asserted`** — the reviewer's judgement, explicitly marked as such

There is no fourth category. A claim about CUDA semantics, a competitor's behaviour, or code quality that is none of the three is a **defect in the review**, not a finding.

---

## §1 Scope

### §1.1 What this skill is

An **adversarial** review that runs on a pull request's diff and consults three sources before rendering a verdict.

"Adversarial" means something specific and testable, and it is **not** "read the code and comment":

> For each substantive change, construct a concrete failure the change would permit, and then determine whether anything in the tree would catch it.

This is the shape that worked repeatedly during APR-PERF-GATE-001. Verifier agents were briefed to *"independently construct a violation it must catch, and confirm it does — do not merely replay the author's own mutation."* That instruction caught a real coverage gap (a call-site bypass that left all three unit tests green) that replaying the author's mutations did not.

### §1.2 What this skill is NOT

- **Not a replacement for CI.** The 102 `check_*.sh` guards run mechanically and always. This skill reasons; it does not duplicate them.
- **Not a style reviewer.** `cargo fmt`, `clippy` and the PMAT gates own that.
- **Not a blocker on first landing.** See §6 — a gate whose failure mode is to suppress the work gets routed around, which this repo has now documented twice (#2757, #2766).

---

## §2 The three mandatory consultations

Each consultation has a **trigger**, a **required output**, and a defined **unavailability behaviour**. The last is the part most likely to be got wrong, so it is specified first:

### §2.0 Unavailability is never silently green

If a consultation's source is unreachable, the review verdict is **`DEGRADED`**, the receipt records `status: unreachable` with the error, and the PR comment says so in its first line.

**It must not be possible to distinguish "consulted and found nothing" from "could not consult" by reading the output.** That distinction is the entire failure mode of #2754 (`tokenization.method` asserting `server_usage` over client-counted tokens), #2779 (`wgpu` reporting a backend it does not enable), #2780 (a join key echoing a clap default), and #2790 (`--gpu` silently resolving to CPU).

This is live today: **the `pmat` MCP server is currently `ConnectionRefused`.** A design that treats that as "nothing to report" is already broken on the day it ships.

### §2.A NVIDIA CUDA documentation (`mcp__nvidia-cuda-docs__search_cuda_docs`)

**Trigger.** The diff touches any of: `crates/aprender-gpu/**`, `crates/aprender-serve/src/cuda/**`, any file matching `*cuda*`, `*ptx*`, `*cublas*`, `*fp8*`, `*nvrtc*`; or the PR body/commit messages mention CUDA, PTX, a compute capability (`sm_\d+`), a CUDA API symbol (`cu[A-Z]\w+`, `cuda[A-Z]\w+`), or a GPU architecture name.

**Required output.** For every device-behaviour claim the diff or its commit message makes, one of:
- the query issued and the excerpt returned, or
- an explicit `no-authority-found` entry naming the query that failed to find one

**The second form is required.** Without it, "the docs didn't say anything" is indistinguishable from "I didn't ask", which is §2.0's failure.

**Why this is first.** It is the consultation with the most evidence of being needed — §0's list is entirely CUDA.

### §2.B pmat (`pmat` MCP server)

**Trigger.** Every PR. There is no diff shape for which code quality is irrelevant.

**Required output.** At minimum:
- complexity delta for each changed function (this is what #2766 exists around — the gate must report *increases*, not absolute values in files it did not touch)
- TDG grade delta for changed files
- SATD introduced by the diff
- any `pmat query` result showing an existing implementation of what the diff adds — **duplication is the most expensive review miss**, and PERF-055 nearly re-implemented ~7,200 lines across 46 files that already existed on an unmerged branch

**Unavailability.** `DEGRADED`, per §2.0. Note the server is **down right now**, so this path will be exercised on day one.

**Open question for review (§8.1):** should `pmat query` run against `origin/main` or the PR head? The local checkout has been observed **66 commits behind** during this epic, and `pmat query` indexes the checkout. A stale index would silently answer questions about code that is not in the PR.

### §2.C CRUX — competitive research review

**Trigger.** The diff changes a **user-facing surface**: a CLI subcommand or flag, an HTTP route, an MCP tool, a config key, or an output format.

**Required output.** For each such surface:
- the CRUX contract(s) that cover it, if any (`contracts/crux-*.yaml`, 275 of them, keyed by `category` and `competitor`)
- how the nearest competitor behaves — `ollama`, `llama.cpp`, `vllm`, `huggingface` per each contract's `competitor:` field
- whether the change **closes**, **widens**, or **does not affect** a recorded gap

**On a surface with no CRUX contract:** say so explicitly and record `crux_coverage: none`. That is a finding in itself — an unaudited user-facing surface is exactly what `nightly-ux-crux` exists to enumerate.

**Do not invent competitor behaviour.** A claim about what ollama does is `cited` (to a CRUX contract or a command run against a local ollama) or it is `asserted` and marked so. This repository has a standing scar here: the book published *"2.93× Ollama"* from a harness that **never ran Ollama**.

---

## §3 The receipt

The skill emits a machine-checkable receipt to `evidence/pr-review/<pr-number>/<head-sha>.json`.

Rationale: prose reviews cannot be audited. This epic's central lesson is that **a claim is evidence only if something can prove how it was produced** — and the same standard has to apply to the reviewer.

```
{
  "pr": 2783,
  "head_sha": "06be2ec18...",
  "base_sha": "9d45b927d...",
  "skill_version": "1.0.0",
  "verdict": "PASS" | "FINDINGS" | "DEGRADED",
  "consultations": {
    "cuda":  { "status": "consulted|not-triggered|unreachable",
               "trigger_reason": "...", "queries": [ {"q": "...", "excerpt_sha256": "..."} ] },
    "pmat":  { "status": "...", "complexity_delta": [...], "duplication_hits": [...] },
    "crux":  { "status": "...", "surfaces": [...], "contracts": ["CRUX-A-08"], "gap_effect": "closes|widens|none" }
  },
  "findings": [ { "claim": "...", "grounding": "cited|measured|asserted",
                  "source": "...", "failure_scenario": "..." } ],
  "adversarial_cases": [ { "constructed_failure": "...", "caught_by": "..." | null } ]
}
```

**`adversarial_cases` is the load-bearing field.** A review with zero constructed failures did not perform §1.1's task, and the guard in §4 refuses it.

**Every finding carries a `grounding`.** A finding grounded `asserted` is permitted — reviewer judgement is legitimate — but it is visibly distinct from one grounded `cited`.

---

## §4 The guard — how we prove this skill can fail

**This section is the reason the spec is worth writing.** A review skill is the most theater-prone artifact this repository could add: it emits prose, prose always "succeeds", and nobody can tell whether it consulted anything.

`scripts/check_pr_review_receipt.sh` validates every receipt and **must be mutation-verified before the skill is enabled**, following the pattern `nightly-ux-crux` already uses — *proving every probe can fail against a committed defective fixture.*

Required case table, each row a committed fixture:

| # | fixture | required verdict |
|---|---|---|
| 1 | receipt with `cuda.status: not-triggered` on a diff touching `src/cuda/` | **RED** |
| 2 | receipt with `adversarial_cases: []` | **RED** |
| 3 | receipt with a finding grounded `cited` whose `source` is empty | **RED** |
| 4 | receipt claiming competitor behaviour with `grounding: cited` and no CRUX contract or command | **RED** |
| 5 | receipt with `pmat.status: unreachable` and `verdict: PASS` | **RED** (must be `DEGRADED`) |
| 6 | receipt with `pmat.status: unreachable` and `verdict: DEGRADED` | **GREEN** (discrimination) |
| 7 | a complete, honest receipt on a docs-only PR with all three `not-triggered` | **GREEN** (discrimination) |

Rows 6 and 7 are not optional. Without them the guard could be "refuse every receipt" and still read green — the exact over-reach that a discrimination case caught in PERF-055 and in the `#2766` delta-gate work.

**A missing receipt is RED, not skipped.** A skill that can be silently not-run is not a gate.

---

## §5 Where it runs

**Proposal:** a `pr-review` skill invoked by the agent opening or updating the PR, writing the receipt into the branch; `check_pr_review_receipt.sh` wired into `ci.yml` beside the existing guards.

**Rationale for not making it a CI job that shells to a model:** the three consultations need MCP servers that CI runners do not have, and a CI job that silently skips its consultations is §2.0's failure at the infrastructure level.

**Consequence, stated plainly:** the receipt is produced by the same actor that authored the change. That is a real weakness. Mitigation is that the receipt is *machine-checked* — an author who skips a consultation cannot produce a receipt that passes §4 — but it does not prevent a low-effort-but-well-formed review. See §8.3.

---

## §6 Blocking versus advisory

| verdict | merge queue | rationale |
|---|---|---|
| `PASS` | proceeds | |
| `FINDINGS` | **proceeds**, findings posted as a PR comment | see below |
| `DEGRADED` | **proceeds**, prominently marked | a down MCP server must not stop the repo |
| missing/invalid receipt | **BLOCKS** | a skill that can be skipped is not a gate |

**`FINDINGS` does not block, deliberately.** This repository has documented twice what happens when a gate's failure mode is to suppress the work rather than the defect: #2766 (a complexity gate scanning staged files froze four files at the centre of an active investigation) and #2757 (a ratchet that made a newly-discovered defect unrecordable, so the cheapest path to green was to not record the finding).

An adversarial reviewer that blocks on judgement calls will be routed around within a week. What blocks is **the absence of a review**, which is objective.

---

## §7 Failure modes this design deliberately prevents

Each is drawn from an actual defect found in this repository, cited so a future reader can check the reasoning rather than trust it:

| failure mode | prevented by | evidence |
|---|---|---|
| Reviewer asserts device semantics from stale memory | §2.A mandatory consultation | 18% regression shipped on an ungrounded stream claim |
| "Source said nothing" indistinguishable from "didn't ask" | §2.A `no-authority-found` requirement | #2754, #2779, #2780, #2790 |
| Unreachable source reads as clean | §2.0, §4 rows 5–6 | pmat is `ConnectionRefused` **today** |
| Re-implementing what already exists | §2.B duplication check | PERF-055 nearly duplicated ~7,200 lines across 46 files |
| Competitor claims with no source | §2.C, §0.1 grounding | the book's *"2.93× Ollama"* from a harness that never ran Ollama |
| Guard that cannot fail | §4 mutation-verified case table | the epic's single most common defect class |
| Guard that blocks the work instead of the defect | §6 | #2757, #2766 |
| Review that "passes" without doing anything | §4 rows 1–2 | — |

---

## §8 Open decisions — for your review

**§8.1 pmat index freshness.** `pmat query` indexes the working checkout, which has been observed 66 commits behind `origin/main` during this epic. Options: (a) require a fresh index and fail if stale, (b) query `origin/main` explicitly, (c) record the index's base SHA in the receipt and let the guard flag drift. **My recommendation: (c)** — it is the cheapest and it makes the staleness visible rather than fatal. But (a) is more honest and I can argue either.

**§8.2 CRUX scope.** 275 contracts is a lot to search per PR. Options: (a) match on the contract's `category` and changed surface, (b) full semantic search each time, (c) maintain a surface→contract index. **My recommendation: (a) now, (c) if it proves slow.**

**§8.3 Self-review weakness (§5).** The author produces their own review receipt. Options: (a) accept it, relying on the machine check; (b) require a *second* independent skill invocation for PRs above some size; (c) sample — every Nth PR gets an independent review. **I do not have a confident recommendation here** and would rather you decide. (b) doubles cost; (c) is weaker but cheap.

**§8.4 Does this run on every PR or only above a size threshold?** A one-line typo fix paying three consultations is waste. But a threshold is a bypass, and this repo's scars are full of bypasses that became load-bearing. **My recommendation: every PR, with `not-triggered` being cheap** — §2.A and §2.C skip on shape, and only §2.B always runs.

**§8.5 Should `FINDINGS` block on a release branch?** §6 says advisory always. An argument exists for blocking at release. Not decided.

---

## §9 Implementation order

1. `contracts/pr-review-skill-v1.yaml` — the contract, `pv validate` clean
2. `scripts/check_pr_review_receipt.sh` + the 7 fixtures — **mutation-verified before anything consumes it**
3. `.claude/skills/pr-review/SKILL.md` — the skill itself
4. Wire the guard into `ci.yml`
5. Run it against three already-merged PRs from this epic and check it would have caught something real. **If it would not have, the design is wrong and should change before it is enabled.**

Step 5 is the acceptance test for the spec itself.
