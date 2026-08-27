# Restart prompt — the receipt rule (APR-PERF-GATE-001)

Copy the fenced block below verbatim into a new session. It is deliberately
terse; everything it asserts is checkable from the repo, and the two
preconditions after it are the ones that have actually bitten.

Companion to [`APR-PERF-GATE-001-v2.2.md`](./APR-PERF-GATE-001-v2.2.md).
Epic: paiml/aprender#2706.

## The prompt

```
Work #2706 (the receipt rule / APR-PERF-GATE-001) autonomously, in parallel. Do not stop.

Spec:     docs/specifications/APR-PERF-GATE-001-v2.2.md
Roadmap:  pmat work list | grep -E 'PERF-|APR-PERF-GATE'   (25 tasks, github_issue 2706)

PARALLEL IS THE DEFAULT, on both axes:
- agents -> speed: fan out independent lanes with the Workflow tool, adversarially verify every "done" claim
- hosts  -> coverage: lambda (x86_64/4090), gx10 (aarch64/GB10), intel (x86_64/AVX-512), mini (arm64/M4)
  A result from one host is a result about one host. #2567 was invisible on x86_64 by construction.

RULES (violating these is the defect, not a style note):
- No claim without a receipt. Unmeasured is UNMEASURED with an owner and expiry, never silence, never an estimate.
- Every change ships its gate, with the named mutation observed RED and a discrimination case that stays GREEN.
- Capture rc directly. `cmd | head; echo $?` reads head's status.
- Prove the mechanism engaged. rc=0, `ldd`, and `apr gpu` have each lied.
- Never hand-assign a number a tool is supposed to measure.

START HERE (highest value first):
1. 8/8 perf-matrix cells are status: UNMEASURED -> the gate arms nothing. Measure one, honestly.
2. perf_gate.sh is invoked by NOTHING (git grep perf_gate -- .github Makefile -> rc=1).
   check_guards_are_wired.sh can't see it: it globs check_*.sh. Fix the universe, re-mutate in the widened scope.
   Do NOT add it to [package.metadata.dogfood] gates — dogfood.sh runs it with no args, it needs four, rc=2 forever.
3. PERF-001 is real but partial: serialization_index(2) = 2.45 lambda / 2.85 gx10, postcondition < 2.

Report progress as N/25 from the roadmap, not as a feeling.
```

## Two preconditions the prompt assumes

**`pmat serve` is not a persistent service.** The roadmap query needs it up:

```bash
pmat serve --transport http --port 8765     # prints the `claude mcp add` line
```

then `/mcp` to reconnect. A generated token dies with the process, so a restart
mints a new one and a client registered with the old one gets 401. Pin it with
`export PMAT_MCP_HTTP_TOKEN=$(pmat mcp token)` if you want it stable.

**Verify gx10 before trusting a number from it.** As of 2026-08-27 it had no
default route, no DNS, a checkout frozen at 0.60.0, and an installed `apr` with
zero CUDA symbols — any benchmark through it would have reported CPU numbers
from a GB10 box.

```bash
ssh gx10 'ip route show default | grep -q . && echo egress-ok'
ssh gx10 'strings -a <bin> | grep -c "libcuda\.so"'      # NOT ldd
```

`ldd` is the wrong probe: CUDA is `dlopen`'d at runtime, so it never appears
there. And `apr gpu` prints the same GPU id on a CUDA build and a CPU-only one —
it reads device presence, not binary capability, which is a defect in its own
right.

## Why the prompt names the failures rather than the goals

Each "START HERE" item is a thing that looks done and is not. The gate is
written, tested, and executed by nothing. The matrix is complete and every cell
is unmeasured. PERF-001 landed and its own contract records the postcondition
failing at c=2. A restart prompt that said "continue the perf gate work" would
walk straight past all three, because all three read as finished from the
outside.

That is the receipt rule applied to the project plan: **a task is done only if
something can prove it.**
