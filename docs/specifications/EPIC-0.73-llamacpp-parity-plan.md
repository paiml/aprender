# EPIC 0.73.0 "llama.cpp Parity": plan (paiml/aprender#3999)

**Status:** plan for operator review. Nothing is applied: no child issues, milestone moves, or closes.
**Ticket:** PMAT-3999 · **kind:** docs · **Ratchet:** slice 4 of 5 of DEBT-RATCHET-001 (#3997, PR #4003)

The operator's scope, quoted on #3999: ".73 is about performance parity with llama.cpp". PP-QUANT/PP-ARCH/PP-TENSOR
moved to 0.74 (#3999 comment), so 0.73 is **pure performance parity**.

## 1. Baseline: the newest head-to-head in the tree

`evidence/parity-http/` at `49fe19c28`. Setup:
- host: lambda, RTX 4090 sm_89, 48 threads;
- model: qwen2.5-coder-7b-instruct Q4_K_M;
- one OpenAI client driving both servers (`apr test llm bench`), 128 in / 128 out, concurrency 1, streaming;
- 3 × 30 s after a 15 s warmup;
- pinned llama.cpp `39173bcac`.

| Metric | apr (`lambda-apr.json`, 08-25) | llama.cpp (`lambda-llamacpp.json`, 08-25) | apr ÷ llama.cpp |
|---|---|---|---|
| decode tok/s (`evidence/parity-http/lambda-apr.json`, `evidence/parity-http/lambda-llamacpp.json`) | 113.6 | 175.3 | **0.65×** |
| prefill tok/s (`evidence/parity-http/lambda-apr.json`, `evidence/parity-http/lambda-llamacpp.json`) | 3,067 | 11,290 | **0.27×** |
| TTFT p50 (`evidence/parity-http/lambda-apr.json`, `evidence/parity-http/lambda-llamacpp.json`) | 33.3 ms | 9.0 ms | **3.7× slower** |
| quiet re-run, decode (`evidence/parity-http/quiet-apr.json`, `evidence/parity-http/quiet-llamacpp.json`) | 104.1 | 159.5 | 0.65× |
| **gx10 GB10 sm_121** (`evidence/parity-http/findings.json` `gx10_gb10_sm121`, same protocol) decode / prefill / TTFT | 31.09 / 2,976 / 34.3 ms | 46.87 / 3,950 / 25.8 ms | **0.66× / 0.75× / 1.33× slower** |

The `findings.json` verdict: "decode ~0.65x on BOTH hosts, a consistent engine gap, not a host artifact". apr's
prefill is flat across hosts (3,067 vs 2,976), which is what a host-bound prefill looks like. Its declared floor was
0.80, and neither host met it on decode. (Quorum lane 2 cited these lines; aprender-cb re-read them.)

- **It is 29 days old and covers one cell** (one model, one quant, one host, CUDA, c=1). R-0 re-measures it on the
  release candidate before any threshold is set.
- **No `contracts/beat-*` gates against llama.cpp.** 16 beat contracts exist; the decode/throughput ones compare with
  **Ollama** (`beat_threshold: 0.9000`, a no-collapse floor). The one llama.cpp mention (`beat-claude-code-parity-v1`)
  is not a perf gate.
- The comparator pin is tracked in paiml/infra#911 (OPEN): "pinned llama.cpp on PATH (both hosts) + ollama 0.34.2,
  and the version gate that was grading itself".

## 2. Exit bar (from #3999), made measurable

**Unit: the parity cell** = (model × quant × backend × host × metric), over the matrix 0.71 certifies. The ratio
`r = apr ÷ llama.cpp` uses the median of N runs. Thresholds: decode and prefill need `r ≥ 1.0 − band`; TTFT and peak
memory need `apr ≤ llama.cpp × (1 + band)`. The band is **declared in the contract and derived from the measured
noise**: the spread of llama.cpp against itself over N runs on the same host, never chosen by hand. Both engines'
version and sha are in every receipt.

## 3. Rows

| Row | Item | done_when | Baseline (measured) | First-green proof |
|---|---|---|---|---|
| **R-0** | Re-measure the matrix before setting thresholds; derive the noise band from llama.cpp vs itself | a parity receipt per certified cell, with N runs per engine and the band in the receipt | 1 cell, 29 days old | the receipt itself. Positive control: apr vs apr must give `r` ≈ 1.0 within the band. **Negative control (quorum fix):** apr run with a planted 20% sleep per token must give `r` < 1 − band, i.e. RED |
| **R-1** | `contracts/beat-llamacpp-*`: re-baseline the beat contracts against llama.cpp (Ollama stays as a secondary reference) | contracts per metric with the declared band, gated by `pv` and the nightly on exclusive GPU time | 0 contracts reference llama.cpp for perf | the contract goes RED on today's numbers (0.65× decode, `evidence/parity-http/findings.json`). That RED is the proof it is not vacuous, and it goes GREEN only when R-2..R-4 land |
| **R-2** | Prefill: batched CPU prefill (#2801) and GPU prefill parity | prefill cells `r ≥ 1 − band` | 0.27× (the CUDA cell above, `evidence/parity-http/lambda-apr.json`); CPU prefill runs at decode rate (#2801, OPEN) | the prefill cell on lambda CUDA, and the CPU cell on lambda and gx10 |
| **R-3** | Decode and TTFT gap on CUDA sm_89 | decode and TTFT cells within band | 0.65× decode, 3.7× TTFT (`evidence/parity-http/lambda-apr.json`, `evidence/parity-http/lambda-llamacpp.json`) | per cell |
| **R-4** | GB10 (sm_121) shortfall (#2800) | gx10 cells within band | decode 0.66×, prefill 0.75×, TTFT 1.33× slower (`evidence/parity-http/findings.json`, 08-24) | per cell on gx10 |
| **R-5** | CPU x86/ARM and Apple Silicon cells | cells within band on every certified CPU/Metal host | not measured | per cell |
| **R-5b** | Peak memory (quorum fix: the exit bar named it and no row did) | peak RSS/VRAM cells `apr ≤ llama.cpp × (1 + band)`, sampled by the harness at 10 Hz | not measured in `evidence/parity-http/` | per cell; a planted 2× allocation in apr must go RED |
| **R-6** | Exclusive-time protocol | every parity run holds the exclusive GPU lock (benchmarks never share, per 0.70) and records `nvidia-smi --query-compute-apps` empty at start | protocol exists for the 08-25 run (it records mechanism lines) | a run started while a foreign GPU process is present must refuse (RED) |
| **R-7** | **Ratchet slice 4 of 5** | the DEBT-RATCHET-001 slice-4 gates | see #4003 | see #4003 |

## 4. Ratchet slice 4 of 5 (from #4003 §3, proposed)

| Pillar | 0.73 floor |
|---|---|
| A: P₀ bp | ≥ 9,255; `P_cuda` ≥ `B_cuda + 2·s_cuda` |
| B-1: E2 call sites | ≥ 371 (total ≥ 510) |
| B-2: contracts with no falsifier | ≤ 3 |
| C: ONT rows bound | ≥ 24 |
| D-1 / D-2 / D-3 | ≤ 84 / ≤ 3 / ≤ 57 |

## 5. Open questions for the quorum to DECIDE

- **Q1 (the epic's open decision): #3977 qwen35moe CUDA forward, 0.73 or 0.74?** **Superseded by the operator:** "MOE
  goes in .71" (#3994 comment, 2026-09-23). #3977 is carried by 0.71 (milestone 0.71.0). The quorum only confirms it
  is out of 0.73's scope. Its parity cell joins 0.73's matrix automatically once 0.71 certifies it.
- **Q2. N and the band.** Recommendation: N = 7 per engine (the Ollama beat's median-of-7). The band = the
  max(|r−1|) of llama.cpp vs itself over those 7, rounded up to the next 0.5%.
- **Q3. Parity per cell, or aggregate?** Recommendation: **per cell**. An aggregate (a geometric mean) lets a 2× win
  on one model hide a 0.5× loss on another, which is the opposite of "Don't Leave Behind".
- **Q4. Which cells are required at 0.73?** Recommendation: every cell 0.71 certifies, with "at parity" reported
  per cell. A cell below band is RED and follows the no-defer doctrine. There is no "not a perf target" exemption.

## 6. Commands

```bash
for f in $(git ls-tree --name-only origin/main evidence/parity-http/ | grep json); do jq -c '(.runs//[.])[0]|{runtime_name,decode_tok_per_sec,prefill_tok_per_sec,ttft_p50_ms,timestamp}' "$f"; done
ls contracts | grep -c '^beat-'                                                 # 16
grep -lE 'llama\.cpp|llamacpp' contracts/beat-*.yaml                            # only beat-claude-code-parity-v1
gh issue view 911 -R paiml/infra --json title,state
```


## Quorum record: decision quorum, 2026-09-23 (aprender-cb)

**Lanes (ADVISORY: single family, all gemini):** gemini-3.1-pro-high, gemini-3.8-flash-high, gemini-3.7-flash-high,
all returning PASS-with-changes. gpt-oss returned 429. 2/3 exited 3 on foreign ref motion (one exited 0), with every
clone byte-identical. Conversations: `fdd3f914`, `2185ec92`, `10c086b1`.

| Q | Decision (tally) | Applied as |
|---|---|---|
| Q1 | **#3977 is out of 0.73** (the operator: "MOE goes in .71"), 3/3 | its cell joins the matrix once 0.71 certifies it |
| Q2 | **N = 7 per engine; band = llama.cpp's self-noise**, 3/3 | R-0 |
| Q3 | **parity per cell**, 3/3 | §2 |
| Q4 | **every 0.71-certified cell, no exemption**, 2/3. Lane 1 would exempt the "PP-* cells deferred to 0.74". **Not adopted:** PP-QUANT/ARCH/TENSOR are code consolidation, not parity cells, so there is nothing to exempt | §2, R-3..R-5 |

**Must-fix items applied:**
- the GB10 baseline, re-read from `findings.json`;
- the R-4 baseline;
- a peak-memory row (R-5b);
- R-0's negative control.

**Must-fix items carried to step 2 as child-issue acceptance:**
- exact `done_when` commands per row;
- negative controls for R-2..R-5;
- the 0.71 certified matrix, enumerated from the ladder contract at the 0.71 tag (it does not exist before then);
- the slice-4 baselines and commands (#4003 §7; #4003 is a separate PR, so it is absent from this tree).

