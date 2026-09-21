# PMAT-3715 implementation receipt: `release-readiness-v1` (aprender#3715)

**Read this before judging scope.** `pmat work status PMAT-3715` prints only the ticket's title. The ticket's scope
is the ISSUE, and the issue's author (the cop, aprender-04, posting as `noahgift`) widened it on the issue itself.
Below, every criterion this diff implements is **quoted verbatim, with its URL**, so each part of the diff can be
checked against the words that asked for it. Nothing below is a paraphrase standing in for a criterion.

## 1. The issue body's done_when (https://github.com/paiml/aprender/issues/3715)

> - [ ] **the evidence graph**: `pv extract` ingests the release receipts (per-host model inventory, ladder and matrix receipts from #3712/#3708, dogfood receipt) as nodes: `:Model` (sha256, quant, arch), `:Host`, `:Verb` (run/chat/serve/code), `:ContextRung` (token counts DERIVED from the consumer-traffic records on #3710, never invented), `:Receipt` (verdict, backend, fallback, apr version plus sha)
> - [ ] **the shape** `release-readiness-v1`: focus = every `:Model` in the inventory of every required `:Host`. For each (verb, context rung) exactly one `:Receipt` with `verdict=Pass`, `backend=cuda`, `fallback=false`, `apr_sha = release commit`. A missing, skipped, stale-sha or fallback receipt is a violation naming the cell.
> - [ ] **armed with a red-turning falsifier**: fixtures where each single cell is missing, skipped, falls back, or is stale, and each one turns the gate RED (the ONT-001 rule: a shape that has never gone red isn't armed)
> - [ ] **wired where the decision is made**: autopilot's T-1 step and `check_publish_preflight.sh` both run `pv lint --gate shapes --shape release-readiness-v1` on the release commit's evidence, and a violation STOPs before the tag and refuses at T-4
> - [ ] **proof**: on the v0.69.0 evidence the shape reports the actual leaks (qwen3-8b lambda, qwen2.5-coder lambda, qwen3moe both hosts, chat/serve/code cells absent). On 0.69.1 it passes with N/N cells named.

## 2. Scope ADDED on the issue by its author, each implemented here

**2a. Kernel differential**, https://github.com/paiml/aprender/issues/3715#issuecomment-5763396082:
> **Added cell (operator approved, verbatim "yes, to all", 2026-09-21): kernel differential per GPU architecture.**
> - [ ] The evidence graph gains `:KernelDiffReceipt` nodes: (kernel, sm arch ∈ {sm_89 lambda, sm_121 gx10}, reference ∈ {cpu, cuda-oxide twin}, max_err, bound, verdict). The bound is DERIVED from a measured known-good/known-bad pair (the C14 basis pattern), never picked.
> - [ ] `release-readiness-v1` requires a Pass receipt for every kernel on the release's dispatch path (the kernels `apr parity --per-op` reports for each inventory model) on EACH GPU arch. A kernel that behaves differently on sm_89 and sm_121 is a violation BEFORE any model-level cell is judged.
> - [ ] The falsifier fixture is a kernel receipt that passes on sm_121 and fails on sm_89, and it turns the gate RED.

→ `release:KernelCell` / `release:KernelDiffReceipt`, shape `release-readiness-v1.kernel`, reported before any cell. The fixture is `a_kernel_green_on_sm_121_and_red_on_sm_89_is_named_on_lambda_before_any_model_cell`. The (kernel, quant) key and the discrete-kernel `index_mismatch`/`tie_margin` rule are the producer's amendments, agreed on #3712 (issuecomment-5763633743).

**2b. Thinking × context dimensions**, https://github.com/paiml/aprender/issues/3715#issuecomment-5763490841:
> **Dimensions added (operator, verbatim):** "lets ensure this release has MORE than enough tokens for our needs (overdo it), and our verbs are dogfooded, i.e chat with and without thinking, ditto run, ditto code, etc,". Each cell is now (model, host, verb ∈ {run, chat, serve, code}, **thinking ∈ {on, off}**, **context rung ∈ {4k, 8k, 20k, 60k, 148k, the model's declared context_length}**). The declared length is READ from the GGUF, not picked. Thinking ON must close its block and answer correctly, or REPORT an unclosed block as an error, never an empty answer. Full statement on #3710.

The full statement, https://github.com/paiml/aprender/issues/3710#issuecomment-5763489958, adds:
> Each Qwen3.5 model must be CORRECT … at every rung up to the **model's own declared context length** (GGUF `*.context_length`, read from the file, not picked here).

→ the cell key (verb, `think-on|off`, rung); `thinkOk` / `answered` on every row; the `declared` rung sized by the model's own `context_length`; a rung above the model's length is not owed.

**2c. Which thinking modes a model owes**, https://github.com/paiml/aprender/issues/3723#issuecomment-5763858250 (the cop accepted this derivation there):
> - the template honours `enable_thinking` → **both** modes are supported (ON and OFF cells owed)
> - else the generation prompt ALWAYS opens `<think>` → **ON only** (an OFF request is refused by name)
> - else → **OFF only** (an ON request is refused by name)

→ `inventory[].thinking_modes`, and a `thinkingContradiction` whenever the markers disprove the modes.

**2d. Tokenizer parity**, #3726's own done_when, https://github.com/paiml/aprender/issues/3726:
> a **tokenizer-parity gate** … that fails on any id mismatch against the pinned llama.cpp … It becomes a required cell per model in the release shape (#3715).

→ `release:TokenizerCell`, shape `release-readiness-v1.tokenizer`; one mismatching id turns it RED.

**2e. The memory rule**, operator decision "a", https://github.com/paiml/aprender/issues/3710#issuecomment-5763753907:
> - A (model, host, context-rung) cell is **owed on a host only if it fits** by the declared memory arithmetic (weights + KV at the chosen precision + workspace ≤ that host's GPU memory, measured).
> - Every rung a model declares must be **owed and Pass on at least one required host.**
> - On a host where the cell doesn't fit, a DIFFERENT cell is owed: **an honest pre-load refusal** (non-zero, naming the arithmetic), never an OOM mid-run and never a silent truncation.

→ `release:RefusalCell` (shape `.refusal`) and `release:RungCoverage` (shape `.coverage`). The rule that a refusal is honest only against TOTAL memory, never free, is the producer's amendment on #3712 (issuecomment-5763795158 thread).

**2f. No waivers.** #3710 allows "a FAIL for the release unless the operator waives that cell by name", but every agent session posts as `noahgift`, so a comment-based waiver is forgeable by any of them. The cop's ruling, recorded in the body of https://github.com/paiml/aprender/issues/3722: *"(c). #3715 ships with no waiver path."* The diff therefore has no waiver input.

## 3. Points a reviewer may otherwise read as out of scope

- **"All models, not just Q4_K."** The universe is the host's measured inventory, exactly as done_when 2 says ("focus = every `:Model` in the inventory"). The inventory is defined by the ladder contract's `inventory.patterns`: `*q4_k*.gguf`, `*q4k*.gguf`, `*q4_k*.apr`, `*q4k*.apr` (#3712). That makes it Q4_K by construction. pv does not re-filter by quant, because that would be a second declaration of the universe. Ladder rungs join the inventory only when they claim `cuda`.
- **`receipts.rs` now reads v2.** #3712's `apr-model-ladder-receipt/v2` is the receipt done_when 1 ingests. Without this change, the first committed v2 receipt turns `pv lint --gate shapes` into exit 3 on every PR. A v3 is still refused by name (`a_v2_receipt_joins_its_rungs_like_v1_…`).
- **`--shape` forces arming.** done_when 4's invocation is `pv lint --gate shapes --shape release-readiness-v1`. A named shape that was only "computed and reported" would print its violations and exit 0 at T-1. In ordinary PR lint the family is unarmed and has 0 focus nodes, because there is no release subject.
- **The ont4b shape ratchet goes 9 → 18.** That test hard-codes the count so that a new shape must be named; the nine are named in it.

## 4. Measured at the PR head

| check | result |
|---|---|
| `cargo test -p aprender-contracts --lib` | 1695 passed, 0 failed |
| `cargo test -p aprender-contracts-cli` (every target) | all green; `ont_release_readiness` has 34 cases |
| `cargo clippy -p aprender-contracts --lib --tests -- -D warnings`, and the same for `-cli` | clean |
| `pv lint contracts` (full) | PASS, 0 errors; no warning from `release-readiness-v1.yaml` |
| `pv extract contracts --check` | fresh (`contracts.nt`, `shapes.ttl` regenerated) |
| `bash scripts/check_ont_ratchet.sh --check` | PASS, after the `--write` restamp |
| `bash scripts/check_explicit_test_commands.sh` | PASS (new `.cmd` 450) |
| mutant: `in: [pass, skip]` on the cell verdict | `a_skipped_cell_is_red_naming_it` FAILS, so it is caught |
| mutant: `fresh()` always true | 3 staleness cases FAIL, so it is caught |
| 0.69.0 proof (`evidence/release/proof-0.69.0/`) | exit 1, 795 findings, naming every leak in done_when 5 |

## 5. What this PR does NOT do (so it carries `Refs #3715`, not `Closes`)

- done_when 4's call sites: autopilot T-1 and `check_publish_preflight.sh` T-4. Those are aprender-f0's (#3717/#3708). This PR provides the exact invocation and exit contract they call, agreed on #3712 (issuecomment-5763344947): 0 Pass · 1 Fail · 2 decline · 3 caller error, and every non-zero STOPs.
- done_when 5's second half, "on 0.69.1 it passes with N/N cells named". It needs the producer (#3712 `cells[]`, the memory arithmetic, and kernel and tokenizer receipts) and the model fixes. `release.cell_names` is how it will name the cells.
