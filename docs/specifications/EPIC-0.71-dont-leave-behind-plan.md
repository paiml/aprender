# EPIC 0.71.0 "Don't Leave Behind" + MoE: plan (paiml/aprender#3994)

**Status:** plan for operator review. Nothing is applied: no child issues, milestone moves, or closes.
**Ticket:** PMAT-3994 · **kind:** docs · **Ratchet:** slice 2 of 5 of DEBT-RATCHET-001 (#3997, PR #4003)

Baselines come from §6's commands, run 2026-09-23 on `origin/main` @ `49fe19c28`. The operator's own decisions on
#3994 (verbatim): scope A, "lets put all into A"; "make .70 fast train" (this theme moves to 0.71); "MOE goes in .71"
(qwen35moe 35B-A3B #3977 and MoE across verbs are **in**).

## 1. Exit bar, made measurable

**Unit: the ladder cell** = (model file × verb × backend × host). The bar is **zero RED cells and zero DEFER cells**
over the cell universe that `contracts/model-capability-ladder-v1.yaml` declares at the 0.71 cut. The universe comes
from the contract, never from a list typed into a plan.

| Axis | Baseline in the contract at `49fe19c28` | 0.71 target in the contract |
|---|---|---|
| hosts | **2** (lambda, gx10) | + a Mac host (Apple Silicon); yoga as a pre-screen only (0.70 FT-4), not an evidence host |
| backends per rung | **cpu, cuda** only; **wgpu: 0 rungs, Metal: 0 rungs** | cpu, cuda, wgpu (Vulkan on lambda/gx10), Metal/wgpu + NEON on the Mac |
| rungs (declared) | **8**, all `*-q4km` (qwen2-1.5b, qwen3-1.7b/8b, qwen35-0.8b/2b/4b/9b/27b) | + every inventory model that fits (`inventory.dirs`/`patterns` already declare the scan), including IQ*/Q2_K/f16/.apr and **qwen3moe + qwen35moe** |
| verbs | **run, chat, serve, code** (`qa` runs per rung via `qa_gate`) | unchanged |
| tracked receipts | the newest on `main` are **0.68.2** (`evidence/dogfood/models/0.68.2/{lambda,gx10}.json`). 0.69.x receipts are not on `main` | a 0.71.0 receipt from every required host |

**The universe's size is `[U]`** until row R-1's probe reports. The CPU legs alone roughly double the sweep, which is
2.5–3 h CUDA-only today (#3994). 0.70's lock work (#3998) is the precondition that makes this affordable.

## 2. Rows

| Row | Item | done_when | Baseline (measured) | First-green proof |
|---|---|---|---|---|
| **R-1** | **Sizing probe** (first, after the 0.69.1 freeze): the 8 contract rungs through `--features wgpu` on lambda and on a Mac | a probe receipt with the RED count per backend | not run; no wgpu rung exists | the probe itself. Its RED count sizes R-3 and R-4 |
| **R-2** | CPU leg for **every** inventory model on lambda (x86) and gx10 (ARM) | the ladder contract declares `cpu` for every inventory rung, and both hosts' receipts are green | CPU on the 8 contract rungs only | the first full-inventory CPU receipt on each host; a planted wrong golden turns one cell RED |
| **R-3** | wgpu backend leg | a `wgpu` backend on the rungs; a cell that prints `falling back to CPU` is RED (existing T-2 Models rule) | 0 rungs | the first wgpu receipt; the fallback-text case turns RED on a CPU-only build |
| **R-4** | Mac ladder host (Metal/wgpu + NEON CPU) | a `mac` host in the contract with `required: true`, plus its receipt | no ladder host exists; #3205 (mini-m4 leg) OPEN in 0.70.0 | the first mac receipt, with its `apr --version` sha equal to the release SHA |
| **R-5** | Low-bit admission + CUDA GEMV: #3963 IQ3_XXS, #3953 IQ2_S, #3960 Q2_K | the rung cell green on lambda and gx10 | 3/3 OPEN, no CUDA GEMV | per rung, green on both hosts |
| **R-6** | #3951 IQ4_XS thinking never closes | the rung's golden output (think block closed) green on CUDA | OPEN | green on both hosts; Q4_K_M as the positive control |
| **R-7** | MoE: #3987 qwen3moe chat/serve/code (rc 8, 501/500, rc 1), **#3977 qwen35moe 35B-A3B CUDA forward** (new SSM+MoE arch) | the qwen3moe and qwen35moe rungs green through all 4 verbs, on every host they fit | both OPEN; #3977 has no CUDA forward | per verb, per host. For #3977, correctness is judged against llama.cpp on the **official template** (an oracle fed apr's own prompt inherits apr's template bugs) |
| **R-8** | GPU correctness underneath: #3973 (F2 fails open), #3976 (Q4_K GEMV empty PTX launched), #3975 (GPU/CPU f32 APR divergence at layer 0) | each issue's falsifier in CI or the cuda nightly | 3/3 OPEN | #3973: a planted CPU-reference failure must fail CLOSED |
| **R-9** | Verb surface: #3978 (`apr code` hardcodes `--gpu`), #3979 (`.apr` serve routes, SSE `[DONE]`) | `apr code` has a CPU lane; serve's `.apr` routers carry `GET /` and end SSE with `[DONE]` | 2/2 OPEN | the ladder's `code` and `serve` verbs green on a CPU-only rung |
| **R-10** | Re-bucket the milestone | every open 0.71.0 issue is judged against this bar: in / 0.72 / backlog | 0.71.0 holds **12** open issues today. #3994's "187" was counted when this theme was the 0.70.0 epic; 0.70.0 holds 176 today | step-2 triage, **operator approval before any move** |
| **R-11** | **Ratchet slice 2 of 5** | the DEBT-RATCHET-001 slice-2 gates | see #4003 | see #4003 |

**Overlap with 0.70:** R-5/R-6 and the #3987 part of R-7 are the same issues as 0.70's FT-11
(`docs/specifications/release-0.70-fast-train-plan.md`, branch `docs/3998-fast-train-plan`). Whichever release the 0.70
quorum's Q1 gives them, the other plan drops them. **They are never carried twice.** 0.70 only carries what 0.69.1 did
not close.

## 3. Ratchet slice 2 of 5 (from #4003 §3, proposed)

| Pillar | 0.71 floor |
|---|---|
| A: P₀ bp | ≥ 9,037 (0.75 continues: see #4003 §3.E); `P_cuda`'s first sharded measurement recorded as `B_cuda` (report-only) |
| B-1: E2 call sites | ≥ 243 (total ≥ 510) |
| B-2: contracts with no falsifier | ≤ 11 |
| C: ONT rows bound | ≥ 19 |
| D-1: issues outside a release milestone | ≤ 256 |
| D-2: stale PRs | ≤ 11 |
| D-3: remote branches with no PR | ≤ 177 |

## 4. Open questions for the quorum to decide

- **Q1. Which Mac is the ladder host?** The only candidate named in the tree is `mini-m4` (#3205). Recommendation:
  mini-m4, provisioned through forjar like the other hosts. A laptop cannot be a `required: true` host.
- **Q2. The 0.71 date.** The milestone is due 2026-09-29 (3 days after 0.70). Recommendation: no date until R-1's
  probe reports a RED count. The bar is "zero REDs", so the date follows the count, not the other way round.
- **Q3. qwen35moe (#3977) fit.** 35B-A3B Q4 ≈ 20 GB: it fits on gx10 (unified memory) and at 24 GB on lambda only with
  a small context. Recommendation: `required` on gx10, `required` on lambda at a declared context, and the contract
  states the context.

## 5. Out of scope

Performance (0.73) and new features.

## 6. Commands

```bash
python3 -c "import yaml;d=yaml.safe_load(open('contracts/model-capability-ladder-v1.yaml'))['ladder'];print(d['hosts'],d['cells']['verbs'],[(r['id'],r.get('backends')) for r in d['rungs']])"
git ls-tree -r --name-only origin/main evidence/dogfood/models | sort | tail
gh issue list -R paiml/aprender --state open --milestone 0.71.0 --limit 300 --json number --jq length   # 12
for i in 3963 3953 3960 3951 3987 3977 3973 3976 3975 3978 3979 3205; do gh issue view $i -R paiml/aprender --json state,milestone; done
```

## 7. Quorum record

_Filled after the quorum returns._
