# Fable Architectural Review: EV-Ordered, Falsifiable Engineering Roadmap

**Date:** 2026-07-05
**Auditor:** Claude Fable 5 (14-agent verification workflow, 406 tool calls, all claims artifact-grounded)
**Reference HEAD:** `origin/main` @ `9f0b89fb1c333756a4cf8df98e0426965a4ca38d` (proof-corpus climb wave 2, #2288)
**Prompt:** `~/Downloads/fable5-aprender-roadmap-prompt.md` (grounding contract §2: every claim VERIFIED / STALE / UNVERIFIED with a fetched artifact)

---

## 0. Operating assumption

**Live access: YES.** This audit used a local clone pinned to `origin/main` HEAD (fetched 2026-07-05), the `gh` CLI (issues, PRs, CI-run logs, branch-protection API), the crates.io API + sparse index, and web search for the competitive flank. Every claim cites an artifact fetched during the audit: path:line in the pinned worktree, commit SHA, `gh` output, or URL. Nothing is quoted from training data or from the (absent) vision-doc snapshots.

**Session flag:** branch `fix-pmat-737` tip `3a8c92f67` claims "implement ggml-style pre-interleaved Q4_K layout" but its diff vs main touches only `docs/roadmaps/roadmap.yaml` (flips PMAT-737 planned→completed; zero interleave code, no PR). Do not merge as-is — a status-flip without artifact is the exact theater class the doctrine bans.

---

## 7a. Verification ledger

| claim | snapshot value | HEAD value | source | verdict |
|---|---|---|---|---|
| Workspace crates | 82 | 82 dirs under `crates/` (README:43 ✓, FALSIFY-README-001 passes); actual workspace **members = 78** (77 + root facade; 3 excluded, `aprender-contracts-staging` has no Cargo.toml, `aprender-train-canary` orphaned) | `Cargo.toml` tomllib parse; README.md:43 | VERIFIED (with member nuance) |
| Provable contracts | 1,331 | **1,460** recursive (`git ls-tree HEAD` agrees); 1,217 top-level. 1,331 was exact at `7c0dc4388` (2026-05-23). README:44 still says 1331; CLAUDE.md says 1148 *and* 1134 | README.md:44; `find contracts -name '*.yaml'` | STALE |
| CLI commands | 103 | 103 — contract registry `apr-cli-commands-v1.yaml` = 103 entries; source enums = 104 defined − 1 dev-gated `mono`; book cli chapters = 103. (Contract prose line 24 still says "77" — rot) | commands_enum.rs:26 + awk counts; contract lines 40–686 | VERIFIED |
| Tests | 25,300+ | **~83,500 lib tests executed** on green main CI (79,994 nextest + 3,506 compute). 3.3× stale; not gated by any contract; ci.yml:246 comment repeats the stale figure | `gh run view 28718694787` log; ci.yml:272 | STALE |
| MSRV | Rust 1.89 | 1.89 (Cargo.toml:123,479) — but **never CI-verified**: rust-toolchain.toml pins 1.93.0, hosted jobs use `@stable`; overrides: aprender-train 1.87, aprender-graph 1.75 | Cargo.toml; rust-toolchain.toml; grep workflows → 0 MSRV refs | VERIFIED (value) / gate absent |
| P1 LinReg beat "2.0×, CI-gated" | 2.0× | Gate = ratio ≤0.90 (≥1.11×); pinned 1.78×; **measured 2.73×** (nightly 28734800170, 2026-07-05). Nightly **non-blocking only** ("never blocks PRs") | beat_sklearn_linreg_speed.rs:37,137; contract:31-32 | STALE (both number and "CI-gated") |
| P2 grad max\|Δ\|=5e-7 | 5e-7 | ≈5.0e-7 documented measured (contract:39); enforced assert 1e-4; **per-PR blocking** at ci.yml:317 | apr-pytorch-autograd-equivalence-beat-v1.yaml:36-43 | VERIFIED |
| P3 NF4 4.9e-7 / merge 1.5e-8 | 4.9e-7 / 1.5e-8 | ≈4.92e-7 (enforced 1e-3) and ≈1.49e-8 (enforced 1e-4); both per-PR at ci.yml:317; composed-QLoRA (#2266, ≈2.98e-8) merged, enforced via workspace-test `--lib` (its `ci_gate_name` appears in no workflow) | both beat contracts:39-43; nf4.rs:303; merge.rs:512 | VERIFIED |
| P4 broken-model rejection "10/10" | 10/10 | **11 classes** (7 weight + 4 embedding, `beat_threshold: 11`), per-PR at ci.yml:317; incumbent side = pinned 2026-06-13 manual measurement, not live CI; docs/BEATS.md:119 still says 10/10 | apr-fail-closed-garbage-beat-v1.yaml:48; beat_fail_closed_garbage.rs:75-151 | STALE (count) |
| P4 decode "1.2–1.37× enforced, best 1.43×" | 1.2–1.37× / 1.43× | Gate constant = **1.10×**; measured median 1.371× (best single run 1.523×); 1.43× is the *separate* apr-qa flat lane (440 vs 307). And the harness is **in ZERO workflows** — see T12 | beat_ollama_decode_throughput_speed.rs:67,195; contract:33-34; roadmap.yaml:10026 | STALE + gate is theater |
| Case files #153/#701/#1599/#1864 closed w/ mechanized fixes | all 4 mechanized | All 4 CLOSED ✓. Mechanized: **#701** (check_publish_safety.sh + FALSIFY-MONO-005) and **#1864** (FALSIFY-CPU-GPU-006 + multi-step gate + stop_tokens fix). **#153 & #1599: CLOSED-BUT-UNMECHANIZED** — zero regression artifacts; a refactor reintroducing either bug passes every gate today | gh issue/pr views; contract + source cites per issue | STALE (2/4) |
| GPU falsifiers "developer-side only" | not CI-enforced | **CLOSED at HEAD (good-stale)**: cuda-nightly.yml (01:30 UTC, ada-4090 sm_89 + blackwell-gb10 sm_121) runs all 5 FALSIFY-CUDA-* with `--include-ignored` + provisioned APR_PARITY_MODEL; 2026-07-05 log proves real run, 5/5 pass both legs. Caveat: yield-to-training makes any given green night possibly a no-op skip | cuda-nightly.yml:150-161; `gh run view 28730946549` log | STALE (gap closed) |
| Proof ladder "L4 small, L5 ~3-4" | L4≈0–5, L5 3-4 | **L5 = 6** (P1 metrics, P3 lora-merge, P4 transpose-roundtrip, P5 harness-ir + attention-kernel + softmax-kernel), **L4 = 11** with `--verify-bindings`; P2 cross-entropy honest L3 (9 proved + 1 N/A < 12). #2288 (=HEAD) superseded the snapshot same-day. (`~/.cargo/bin/pv` is stale 0.49.0 — measured with fresh pv 0.59.0 at `/mnt/nvme-raid0/targets/aprender/release/pv`) | `pv proof-status --binding contracts/binding.yaml --verify-bindings`; verification_summary blocks | STALE (higher) |
| Contract falsification coverage | n/a | falsification_tests non-empty: 961/1460 (65.8%); 93.0% counting legacy `falsification:` key; kani_harnesses 417 (28.6%); proof_obligations 807 (55.3%) | python+pyyaml scan over corpus | VERIFIED (fresh) |
| crates.io publish (T7) | lag suspected | **FIRED: 0.51.0 (2026-06-21) vs v0.59.0 tag (2026-07-04) = 8 versions / 14 days**; release.yml deleted 2026-04-14 ("use manual cargo publish"); CLAUDE.md's "release.yml: automated releases" is stale; binary-release.yml ships only pv Linux binaries | crates.io index; git tags; deletion commit deccac2e6 | VERIFIED |
| Roadmap PMAT-9xx private refs | assumed present | **ZERO** PMAT-9xx refs in ROADMAP.md or roadmap.yaml (max id = PMAT-741); premise doesn't hold at HEAD | grep exit 1 | VERIFIED (absence) |
| Roadmap state | 43 phantom inprogress | Reconciled ✓: 595 items — planned 301 / completed 275 / cancelled 15 / blocked 4 / **inprogress 0**. Stale standalone-repo refs confined to completed/cancelled items (only live ones are the intentional archive epics PMAT-501/502). roadmap.yaml header comment (lines 33-49) still calls the CUDA-CI gap open though the item is completed | roadmap.yaml parse; #2275/7e054b375 | VERIFIED |
| Required status checks | "ci/gate + workspace-test" (memory) | Ruleset requires **only `gate`**; workspace-test is transitively required via `needs:` | `gh api repos/paiml/aprender/rules/branches/main` | STALE (memory corrected) |
| T12 gate-theater watch signals | canonical CF-4 | **FIRED, multiple live instances at HEAD** — see §STEP-3 | H1/H2 agent cites | VERIFIED |

**New defects found during verification (not in any snapshot):**

1. `ci.yml:288` compute-step pass check `grep -q "test result.*0 failed"` **substring-matches "10 failed"** — a 10-failure run merges green.
2. sovereign-ci `ci / test` job is vacuous for aprender (root facade, 0 tests); fmt + cargo-deny are advisory (`continue-on-error`).
3. Coverage ≥95% is enforced **nowhere** in CI (coverage job vacuous on root facade; `coverage_min` unset; coverage-nightly is report-mode).
4. ~595 integration-test files in `crates/*/tests/` (the entire falsify_*/parity corpus) are never executed by any workflow — compile-checked only (clippy `--all-targets`).

### Frontier map (STEP 2)

- **P1**: 6 accuracy/parity beats per-PR blocking (ci.yml:317; all four v0.59 PRs #2267/#2271/#2272/#2273 verified merged + wired); 7 speed beats nightly-non-blocking. LinReg measured 2.73×. **Risk: GaussianNB speed beat decayed 4.9×→2.12× against a 2.0× gate; red 07-03/07-04 nightlies.**
- **P2**: autograd-grad beat per-PR; deploy-footprint 150 MiB ceiling (no ratchet); cross-entropy honest L3 (9 proved + 1 N/A < 12 obligations). Training throughput conceded (specific, sourced).
- **P3**: NF4/merge/composed beats enforced; **the 5-falsifier CUDA nightly on 2 silicon is live** — frontier moved from "enforce at all" to "enforce trajectories, not step-0."
- **P4**: fail-closed 11-class per-PR (headline intact); decode beat = theater (no workflow); 1.5× marquee = PMAT-739 planned; CPU decode 8.2× slower than llama.cpp (PMAT-737 planned; branch mislabels it completed); F2 probe multi-position but prefill-phase/serial-path only.
- **P5**: parity pillar; harness-ir contract at L5; apr-code function-scale 1.0 vs project-scale 0.20 gap (#2242); Antigravity/either-harness lane tops the planned queue; **flank ships Anthropic-compatible `/v1/messages` (mistral.rs) — aprender doesn't.**

### Gate-integrity audit (STEP 3, highest-leverage hits)

CF-4-signature gates (one point of a temporal/compounding system):

- **[A]** `apr qa` format_parity = one prefill forward, single final-position argmax, **zero decode steps** (forward_error.rs:313-416) — a literal #1864 clone inside the flagship diagnostic; the daily qwen-story never even reaches it (APR input → skip at :345-350).
- **[B]** The #1864 remediation itself (F2 probe, inference_result.rs:468) samples only the **serial prefill** path — generate_2.rs:246-268 documents in-code that the unsampled batched-prefill decode combination shipped PMAT-810's "CertainlyCertainly" corruption silently. BATCHED_PREFILL remains default on sm_89.
- **[C]** GPU training gated at **step 0 only** (parity_probe.rs:186 — no optimizer step ever taken) while CPU has an enforced 400-step trajectory test (train_to_loss_tests.rs:74) — the asymmetry is the gap; #2251-class backward/stream bugs are structurally invisible.
- **[D]** Serve sampled at exactly **one request/one turn** everywhere (ollama_http_compat: single stream:false request on a demo model; qwen-story B6: one 4-token curl); the only "multi-turn" test (FALSIFY-CHAT-008, falsification_chat_http_cli.rs:386) is a self-referential string tautology that never calls the template engine — and is in no enforced test list.
- **[E]** GQA cache-parity horizon = 2 positions (gqa_attention_parity.rs) vs PMAT-749's ~64-token failure onset; unenforced besides.
- **[F]** wgpu/CUDA AdamW parity = 1 step from zero moments (wgpu_cuda_parity.rs:131) — blind to bias-correction/moment-accumulation bugs by construction; unenforced.
- **[G]** `gpu_cpu_parity.rs` is assertion-free: the CUDA model is built as `_cuda_model` and never compared — nothing can turn it red.

### Threats (STEP 4)

- **T7 FIRED** — 8 versions/14 days crates.io lag, no automation (release.yml deleted 2026-04-14).
- **T12 FIRED** — readme-claims checker unwired despite `status: enforced` + live 1331≠1460 drift; "ENFORCED"-labeled decode beat in no workflow; the "0 failed" grep bug; vacuous ci/test job; advisory fmt/deny.
- **T5 partial** — nightly macOS *build* green (both darwin targets, run 28732648442) but zero macOS test execution; the Metal/wgpu path is never compiled on Apple hardware (the METAL assertion at device/mod.rs:500 has no runner) — while Ollama shipped MLX (0.19 preview → v0.31.1: Gemma4 +90% on Apple silicon via multi-token prediction).
- **T2 LIVE and accelerating** — mistral.rs v0.8.23 (weekly cadence, ~7.3k★) ships Anthropic-compatible `/v1/messages` + native agent loop + prebuilt binaries; rvLLM (753★) markets "parity-proven bit-identical" kernels via testing, no contracts; burn 0.21 presses the training pillar (distributed collectives); candle remains substrate. **No competitor claims contract-gated correctness — the wedge is intact but rhetorically contested.**
- **T9** guarded by doctrine (see 7c). **T1/T3/T4/T6/T8/T10/T11**: not enumerable — vision docs absent from repo and session (→7d).

---

## 7b. EV-ranked backlog

```yaml
- id: PMAT-CI-PASSGREP-001
  pillar: infra
  type: gate
  ev_rank: 1
  ev_rationale: The per-PR enforcement backbone is bypassable — ci.yml:288's pass check substring-matches "10 failed"; every other gate's credibility transits this step; one-line fix.
  definition_of_done: "ci.yml compute step pass check anchored (e.g. `grep -q '; 0 failed;'` + require `test result: ok.`). Mutation that must turn RED: add 10 `assert!(false)` tests to aprender-compute --lib on a scratch branch — workspace-test must fail (today it exits 0)."
  blocked_by: none
  artifact_on_completion: PR (one-line ci.yml) + RED-mutation evidence in PR body
  workflow_note: ticket → fix/ci-compute-pass-grep → PR → `ci / gate` (small-CI-edit class, pre-authorized)

- id: BEAT-OLLAMA-DECODE-CI-001   # exists at roadmap.yaml:183 (planned/medium) — promote + harden
  pillar: P4
  type: gate
  ev_rank: 2
  ev_rationale: The marquee 1.371× decode win has ZERO CI execution path (#[ignore], absent from all 11 workflows) AND a speed-only oracle — gibberish at 412 tok/s passes; Ollama v0.31.1 velocity decays the pinned 300.7 tok/s baseline silently.
  definition_of_done: "cuda-nightly.yml ada-4090 leg runs the beat with --ignored nightly; harness gains a coherence oracle (4-gram repetition ratio + no <|im_start|> re-emission over the 384-token window); contract baseline re-pinned against ollama v0.31.x with date+version. Mutations that must turn RED: (a) throttle apr decode 2× → ratio gate red; (b) full-speed repeat-collapse stub output → coherence red."
  blocked_by: none (runner + 1.5B Q4_K_M model already provisioned for cuda-nightly)
  artifact_on_completion: falsifier + re-pinned beat-ollama-decode-throughput-speed-v1.yaml
  workflow_note: ticket → branch → PR → `ci / gate`; workflow edit = small-CI-edit class

- id: PMAT-F2-DECODE-PHASE-001
  pillar: P4
  type: gate
  ev_rank: 3
  ev_rationale: The #1864 remediation probe validates only serial prefill; generate_2.rs:246-268 documents in-code that batched-prefill decode corruption (PMAT-810) ships through it silently, and BATCHED_PREFILL is still default on sm_89.
  definition_of_done: "validate_gpu_first_token extended with ≥8 greedy decode steps executed through the SAME prefill mode production selects, per-step argmax + cosine≥0.95 vs CPU; new FALSIFY-CPU-GPU-007 in apr-cpu-vs-gpu-output-parity-v1.yaml. Mutation that must turn RED: corrupt the batched-prefill KV scatter (or truncate cache at decode step 4) — probe must reject; today it accepts."
  blocked_by: none
  artifact_on_completion: falsifier + pv_contract amendment
  workflow_note: ticket → branch → PR → `ci / gate`; verify on cuda-nightly both silicon legs

- id: PMAT-QA-FMTPARITY-DECODE-001
  pillar: P4
  type: gate
  ev_rank: 4
  ev_rationale: apr qa's format_parity gate is a literal #1864 clone (one forward, final-position argmax, zero decode steps) inside the tool the doctrine says to reach for FIRST; cross-format cached-decode drift is invisible to it.
  definition_of_done: "Gate greedy-decodes ≥64 tokens per format via the cache path with per-step argmax equality (near-tie cosine ≥0.98 exemption); qwen-story B2 gains a GGUF leg so the gate actually executes daily. Mutation that must turn RED: zero the SafeTensors-side KV write at decode step 32 — apr qa exits 5; today it stays green."
  blocked_by: none
  artifact_on_completion: falsifier + qa-gate contract update
  workflow_note: ticket → branch → PR → `ci / gate`

- id: FALSIFY-CUDA-NF4-TRAIN-TRAJECTORY-001
  pillar: P3
  type: gate
  ev_rank: 5
  ev_rationale: GPU training is gated at step 0 only (no optimizer step ever taken) while CPU has an enforced 400-step trajectory — the #2251-class backward/stream bugs are structurally invisible; the cuda-nightly lane already exists so marginal cost is one test.
  definition_of_done: "≥20 real optimizer steps GPU vs NF4-matched CPU oracle (identical data/lr); per-step |Δloss| within band AND total decrease ≥ X nats by step 20; wired into cuda-nightly.yml. Mutation that must turn RED: freeze the CUDA Adam t counter at 1 — red by step 3; the current step-0 gate stays green forever."
  blocked_by: none
  artifact_on_completion: falsifier + contract (cuda-nf4-train-trajectory-v1.yaml)
  workflow_note: ticket → branch → PR → `ci / gate`; nightly proof on both silicon

- id: PMAT-SERVE-MULTITURN-001
  pillar: P4
  type: gate
  ev_rank: 6
  ev_rationale: Every serve gate samples one request/one turn; multi-turn KV/template accumulation — the #1864 gibberish vector — has no genuine gate, and the sole "multi-turn" test is a tautology that never calls the template engine.
  definition_of_done: "(a) ollama_http_compat gains a 3-turn /api/chat sequence + a stream:true request asserting ≥2 chunks then done:true (stays per-PR at ci.yml:317); (b) FALSIFY-CHAT-008 rewritten to invoke the real chat_template engine; (c) qwen-story B6b: 2 follow-up requests to the same server. Mutations that must turn RED: handler drops all-but-last message → (a); swap two turns pre-render → (b); kill KV reset between requests → (c)."
  blocked_by: none
  artifact_on_completion: falsifier ×3
  workflow_note: (a)/(b) per-PR; (c) rides qwen-story-daily; ci.yml:317 is ONE line — consolidate edits in one PR

- id: PMAT-DRIFT-GATES-001
  pillar: infra
  type: contract
  ev_rank: 7
  ev_rationale: The claims-as-contract layer is live-proven theater (README 1331 vs tree 1460, checker in zero workflows despite status:'enforced'; book/CLI parity path-filtered so exactly the drift it exists to catch bypasses it).
  definition_of_done: "check_readme_claims.sh (or extended readme_contract test) runs per-PR; book.yml + book-contracts.yml triggers extended to crates/apr-cli/src/**; README/CLAUDE.md counts corrected (1460 contracts, ~83.5k tests, 82 crates). Mutations that must turn RED: PR changing crates/ dir count without README edit; PR adding a clap subcommand without a book chapter — red on THAT PR, not the next book PR."
  blocked_by: none
  artifact_on_completion: PR + FALSIFY-README-001..004 status flipped to genuinely enforced
  workflow_note: ticket → branch → PR → `ci / gate`; small-CI-edit class

- id: PMAT-739   # exists (decode 1.5× marquee) — sequence behind rank 2
  pillar: P4
  type: beat
  ev_rank: 8
  ev_rationale: The one open falsifiable P4 speed beat (5-run median ≥460.7 tok/s, FUSION-004 DP4A lever); highest outward EV but multi-week — and pointless against a dead baseline, so it rides behind the re-pin in rank 2.
  definition_of_done: "5-run median ≥ 460.7 tok/s (or 1.5× the re-pinned ollama v0.31.x number, whichever is higher) on RTX 4090 1.5B Q4_K_M, measured by the rank-2 harness turning green at threshold 1.5; contract updated with dated measurement."
  blocked_by: BEAT-OLLAMA-DECODE-CI-001
  artifact_on_completion: benchmark + updated beat contract
  workflow_note: ticket → branch → PR → `ci / gate`; nightly harness is the arbiter

- id: APR-ANTHROPIC-MSGS-INTEGRITY-001
  pillar: P5
  type: beat
  ev_rank: 9
  ev_rationale: mistral.rs ships /v1/messages + agent loop with zero correctness contracts (web-verified); P5's only defensible ground is provable tool-call integrity — shipping the endpoint WITH a falsification-gated integrity contract is a beat nobody claims, and it feeds the APR-ANTIGRAVITY-PARITY-001 lane already atop the planned queue.
  definition_of_done: "apr serve exposes Anthropic-compatible /v1/messages (+count_tokens); contract apr-anthropic-messages-v1.yaml with falsifier: tool_call id/structure preserved byte-exact across ≥3 turns incl. the #2245 salvage-parser path. Mutation that must turn RED: drop tool_call id remapping on turn 2."
  blocked_by: none
  artifact_on_completion: pv_contract + falsifier + PR
  workflow_note: ticket → branch → PR → `ci / gate`

- id: PMAT-GNB-SPEED-DEFENSE-001
  pillar: P1
  type: beat
  ev_rank: 10
  ev_rationale: A LIVE beat decaying toward its own gate (pinned 4.9× → measured 2.12× vs 2.0× ceiling; red 07-03/07-04 nightlies) — an untriaged decay becomes a flake, then a deleted gate; five-whys before it flips.
  definition_of_done: "Regression bisected to a commit or attributed to environment (sklearn 1.9 speedup / allocator / ln-hoist erosion); margin restored to ≥3× measured OR contract honestly re-pinned with dated measurement; then 5 consecutive green nightlies."
  blocked_by: none
  artifact_on_completion: benchmark + contract re-pin PR
  workflow_note: ticket → branch → PR → `ci / gate`

- id: PMAT-CUDA-NIGHTLY-PROBATIVE-001
  pillar: P3
  type: gate
  ev_rank: 11
  ev_rationale: Yield-to-training turns busy-GPU nights into green no-ops (greens are non-probative), and the lane runs only 5 falsifiers while GQA parity (2-position horizon vs PMAT-749's ~64), single-step AdamW, and NF4 roundtrip sit unwired beside it.
  definition_of_done: "Skip nights emit a distinct conclusion (neutral/skipped + weekly skip-rate report, alert >50%); lane adopts gqa_attention_parity extended to ≥64 cached positions, ≥10-step AdamW parity, NF4 double-roundtrip idempotence. Mutations that must turn RED: stride GPU cache by q_dim (the PMAT-749 bug) → red by step 8; freeze bias correction → red by step 3; perturb absmax on roundtrip 2 → red."
  blocked_by: none
  artifact_on_completion: falsifier set + workflow PR
  workflow_note: small-CI-edit class; verify a real (non-skip) run on both legs

- id: PMAT-CASEFILE-MECH-001
  pillar: cross-cutting
  type: contract
  ev_rank: 12
  ev_rationale: Doctrine 7's corpus claim is overstated at HEAD — #153 and #1599 are CLOSED-BUT-UNMECHANIZED; a refactor reintroducing full-file format detection or a non-optional facade dep passes every gate today.
  definition_of_done: "#153: regression test asserting format detection reads ≤8 bytes + bind-before-ready ordering test; #1599: CI step asserting `cargo tree -p aprender --no-default-features -e normal` contains no apr-cli. Mutations that must turn RED: swap the 8-byte read for std::fs::read; make apr-cli non-optional."
  blocked_by: none
  artifact_on_completion: falsifier ×2 + contract entries naming both issues
  workflow_note: ticket → branch → PR → `ci / gate`

- id: PMAT-MACOS-WGPU-NIGHTLY-001
  pillar: infra
  type: gate
  ev_rank: 13
  ev_rationale: T5 — the incumbent's fastest 2026 platform (Ollama MLX: ~2× decode, Gemma4 +90%) vs zero macOS test execution here; the Metal/wgpu path is never compiled on Apple hardware (the METAL assertion at device/mod.rs:500 has no runner); the nightly macOS lane already exists.
  definition_of_done: "nightly.yml macOS aarch64 leg adds --features wgpu build + `cargo test -p aprender-compute --lib --features gpu`; the macOS METAL assertion executes. Mutation that must turn RED: cfg-gate out Metal backend registration → macOS test leg fails."
  blocked_by: none
  artifact_on_completion: workflow PR + green run link
  workflow_note: small-CI-edit class; GitHub-hosted macos-latest, no fleet dependency

- id: PMAT-MSRV-GATE-001
  pillar: infra
  type: gate
  ev_rank: 14
  ev_rationale: rust-version=1.89 is a published contract cargo enforces on every downstream install, but CI builds on 1.93/stable — any 1.90+ feature merges green and breaks `cargo install aprender` users; 4-line job.
  definition_of_done: "PR-blocking job: dtolnay/rust-toolchain@1.89 + `cargo check --workspace`; per-crate overrides reconciled (aprender-train 1.87, aprender-graph 1.75). Mutation that must turn RED: use a 1.93-only std API."
  blocked_by: none
  artifact_on_completion: workflow PR
  workflow_note: small-CI-edit class

- id: PMAT-PUBLISH-POLICY-001
  pillar: infra
  type: infra
  ev_rank: 15
  ev_rationale: T7 fired — doctrine 5 declares clean-room publishability a HARD release gate while crates.io is 8 versions/14 days behind with zero automation; doctrine and practice must reconverge; operator decision (publish skipped since v0.51 by choice).
  definition_of_done: "CLEAN-ROOM GATE FIRST: `cargo publish --dry-run --no-verify` cascade preflight (per the dev-dep-cycle rules) recorded green; THEN either cascade v0.59.x to crates.io OR a written policy in ROADMAP.md designating GH releases as the channel; CLAUDE.md's stale 'release.yml' claim fixed either way."
  blocked_by: operator decision (publish authorization)
  artifact_on_completion: PR (cascade or policy) + preflight output
  workflow_note: preflight → operator GO/NO-GO → cascade; never publish without the preflight artifact

- id: PMAT-MUTANTS-FULLTREE-001
  pillar: infra
  type: gate
  ev_rank: 16
  ev_rationale: Mutation is diff-scoped PR-only; full-tree has never been re-sampled since the push-to-main run was removed — mutation debt outside diffs compounds silently (T12 signal).
  definition_of_done: "Weekly scheduled sharded `cargo mutants -- --lib` (round-robin N crates/night within a time budget) auto-filing an issue listing surviving mutants; first report produced. The surviving-mutant list IS the RED signal."
  blocked_by: none
  artifact_on_completion: workflow + first surviving-mutant issue
  workflow_note: small-CI-edit class; clean-room runners

- id: PMAT-THEATER-TRIAGE-001
  pillar: infra
  type: gate
  ev_rank: 17
  ev_rationale: ~595 unexecuted integration-test files (incl. the falsify_*/parity corpus and the assertion-free gpu_cpu_parity.rs) read as coverage but enforce nothing; high leverage but high cost, hence below the surgical fixes it overlaps.
  definition_of_done: "Committed classification manifest (wire / lib-twin-verified / archive-labeled) for all crates/*/tests files; assertion-free tests completed or deleted (gpu_cpu_parity.rs `_cuda_model` never compared); top-10 falsifiers wired into cuda-nightly or ci.yml:317 in ONE consolidated PR (single-physical-line constraint)."
  blocked_by: PMAT-CUDA-NIGHTLY-PROBATIVE-001 (shares the lane)
  artifact_on_completion: manifest + workflow PR
  workflow_note: consolidate ALL ci.yml:317 edits into one PR to avoid merge-queue conflicts

- id: PMAT-COVERAGE-FLOOR-001
  pillar: infra
  type: gate
  ev_rank: 18
  ev_rationale: The ≥95% standard is enforced nowhere (vacuous root-facade coverage job, coverage_min unset, coverage-nightly report-mode) — ranked last because the coverage+contracts co-evolution rule makes a naive floor counterproductive without scoped work.
  definition_of_done: "sovereign-ci coverage_min + test_workspace set for aprender (or coverage-nightly promoted to fail below a committed baseline). Mutation that must turn RED: delete a tested module's tests → coverage job red."
  blocked_by: paiml/.github sovereign-ci reusable workflow (cross-repo change)
  artifact_on_completion: workflow PR + committed baseline
  workflow_note: cross-repo — coordinate with the sovereign-ci owner before edit
```

Note: `SVC-SMO-WSS-001` stays in the backlog as-is at roadmap.yaml:212 — planned/low is the right rank; ROADMAP.md:40 contradictorily calls it "in flight" — fix the label in the rank-7 drift PR.

---

## 7c. Do-not-do list

- **P1 estimator-breadth race** — EV rule; sklearn's pedagogy moat isn't attackable head-on; only contract-backed correctness/speed beats enter.
- **LAPACK-bound speed contests** (Ridge/Lasso/KMeans/PCA vs MKL-backed sklearn) — documented narrow losses (BEATS.md:57-58,82-85: PCA ~18.6×, Lasso ~19× slower); no falsifiable beat available.
- **P2 raw training-throughput race vs PyTorch/MKL** — conceded (~11×); burn's distributed push doesn't change the calculus.
- **P3 Triton GPU tok/s race vs Unsloth** — conceded; the correctness/trajectory gates are the P3 ground.
- **P4 datacenter-scale concurrent serving** (vLLM class) — conceded segment.
- **Apple-silicon decode-speed contest vs Ollama-MLX/M5 Neural Accelerators** — no winnable beat with the current fleet; ship the T5 build/test smoke (rank 13), not a speed war.
- **P5 feature-parity chasing** Claude Code or the mistral.rs agent loop (web search/Python exec/Skills breadth) — threat T9; only the through-line + tool-call-integrity contract (rank 9).
- **Prose rebuttals to rvLLM's "parity-proven" branding** — answer with L5 proof levels and CI-enforced falsifiers, not marketing.
- **APR-BOOK archive epics (PMAT-497..503) as near-term work** — deferred pre-pivot lane; zero beat value now.
- **Merging `fix-pmat-737` as-is** — it marks PMAT-737 completed with no implementation; status-flip-without-artifact is the exact theater class the doctrine bans.
- **Re-litigating APR-MONO** (crates.io trueno, standalone-repo issue refs) — settled; residues are cleanup items only (train-canary's absolute-path `[patch]`, excluded-manifest registry refs).

---

## 7d. UNVERIFIED / needs-live-access appendix

| gap | exact artifact a human must fetch |
|---|---|
| T1, T3, T4, T6, T8, T10, T11 definitions & watch-signals | The `aprender-pillar-vision*.md` snapshot docs (2026-07-02) — not in-repo (verified absent), not pasted; supply the files to complete STEP 4 |
| README↔binary CLI-count leg (103) | Run `bash scripts/check_readme_claims.sh --claim cli_command_count` (needs `cargo run`; source+contract+book all agree at 103, binary leg unexercised) |
| sovereign-ci benchmark job assertions | `paiml/.github` sovereign-ci.yml `run_benchmarks` section — does nightly-bench assert any regression threshold, or report-only? |
| Rust version inside `localhost:5000/sovereign-ci:stable` | Inspect the image on a clean-room runner (`docker run … rustc -V`) — Dockerfile is not in this repo; PR-blocking builds' toolchain is otherwise unknowable |
| cuda-nightly real-run rate | `gh run view <id> --log \| grep proceed=` across the last N scheduled runs — 2026-07-05 is proven real; earlier greens may be yield-to-training no-ops |
| llama.cpp 2026 perf specifics (FA2 30-41%, ~122-128 tok/s claims) | First-party ggml-org benchmarks/discussions — current figures are secondary-blog-only |
| Ollama MLX ≥32 GB unified-memory requirement | https://ollama.com/blog/mlx — the requirement appears only in a Medium post |
| Current CPU decode tok/s (PMAT-737 premise) | `apr bench <1.5B Q4_K_M>.apr --device cpu` on a fresh binary — the 9.9-vs-81 figure is dated 2026-06-11 |
| Clean-room publishability of 0.52–0.59 | `cargo publish --dry-run --no-verify` cascade per crate (blocks rank-15 GO/NO-GO) |
| One unparsed contract YAML | The single pyyaml parse error in the 1460-file scan — identify and fix the file |
| `pmat comply` DEBT tickets (2026-07-05, local) | Re-run with coverage data + exclusion patterns — DEBT-COV "0.0%" is a no-data artifact; DEBT-SIZE's 7727 count is inflated by lean `.lake` build output and agent worktrees |

---

## Definition of done for this analysis (§8)

Every 7b item carries a committable proof with its RED-turning mutation; none is marked UNDERSPECIFIED. The one deliberately gated item is rank 15 (publish), blocked on an operator decision by design, with the clean-room gate named first per doctrine 5. A reviewer can take any 7b item and open or reject the pmat ticket from `id + definition_of_done + source` alone.
