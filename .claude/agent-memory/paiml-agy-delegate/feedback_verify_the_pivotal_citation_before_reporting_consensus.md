---
name: verify-the-pivotal-citation-before-reporting-consensus
description: A 2-of-3 majority can build its whole root cause on a premise the tree contradicts — before writing "consensus", check the ONE citation every verdict hinges on
metadata:
  type: feedback
---

When lanes converge on a root cause, find the single factual premise the chain rests on
and read that file:line yourself. Report the majority AND the refutation; do not let lane
count decide.

**Why:** on PMAT-1065 (L0-1, width 3, `--mode plan`) lanes 2 and 3 both named "Q8_1
activation quantization in the fused gate+up+SwiGLU kernel" as the root cause of a
0.9508 first-token cosine, reasoning that the load-time gate passed on the fused path.
Lane 1 dissented: `auto_q4k` returns `Q4kVariant::Mwv` unconditionally. That was
checkable in two greps —
`crates/aprender-serve/src/cuda/gpu_profile.rs:238-244` really does read
`let _ = (has_dp4a, cc); Q4kVariant::Mwv` (FALSIFY-Q4K-ADA-PARITY-001), `fused_gate_up`
is derived from it at `gpu_profile.rs:137`, and the in-tree unit test at
`gpu_profile.rs:815` asserts `!detect_fused_gate_up(&Q4kVariant::Mwv, None)`. So the
fused kernel is OFF by default on every GPU and the 2-lane majority's mechanism could
not have been running. Reporting "consensus: Q8_1 fused FFN" would have sent the
orchestrator to edit a kernel that never executed.

**How to apply:** every quorum where lanes name a mechanism. Ask "what one fact makes
this chain true?", then `grep`/`sed` that exact line. This is citation-checking, not
running the acceptance commands (rule 6) — Fable still re-runs those. Put the check in
the receipt as a delegate-verified finding with its file:line so it can be re-run, and
put the majority in `dissent` when it loses to the tree. Corollary: `--mode plan` lanes
(`num_turns=1`) never open a file, so *every* citation they give is from the prompt you
staged or from memory — staging the pivotal source yourself is what makes the dissent
resolvable. See [[plan-mode-lanes-do-not-run-commands]] and
[[lanes-need-a-cd-wrapper-script]].

**Addendum (PMAT-966, SPEC-2.0 rescope, width 3, `--mode plan`).** Two distinct failure
modes, both fatal if passed through as consensus:

1. *The majority's premise was in the spec it was told to read.* Lanes 2 and 3 both ruled
   `scope-fails` because "criterion C3 moved to 0.67 leaves `--backend` resolution
   unenforced". The same spec says the opposite two ways —
   `docs/specifications/PP-066-release-spec.md:267` ("R-0b ships the resolution; the
   per-surface case table over every host is credited in 0.67") and `:323` (R-0b = "the
   resolution rewiring … zero cfg! reads in decisions"), with R-0b a kept row and a
   TAG-0.66.0 dependency at `:192`. A moved *criterion* is not a moved *capability*: read
   what the row defers, not what its id implies.
2. *A lane quoted a WITHDRAWN claim's own withdrawal record as a live fact and ruled
   "keep".* Lane 2 cited `docs/BEATS.md:38` as "✅ WON, apr **1.371×** ollama … gate ≥
   1.10×" and ruled it a receipted fact to keep. Line 38 is a table header; that text
   lives at `:245`, inside `### ⛔ … "apr 1.371× faster" (WITHDRAWN 2026-07-31)`, as the
   **Claimed** row of the withdrawal table. Acting on it would have re-published a
   headline the repo retracted.

**How to apply:** for any lane finding ruled *keep*, check the surrounding section header,
not just the line — withdrawal tables, archives and "what we used to claim" blocks quote
the claim verbatim. And when a lane says a guard's universe misses a surface, read the
guard's file-collection: `scripts/check_no_claim_literals.sh:1091` already unions
`book/**`, `docs/**` and root `*.md`, so "the ratchet misses BEATS.md and the book" was
false against the implementation while true against the spec's prose sentence — a
spec-text/implementation gap, not a coverage hole.

**Addendum (PMAT-1065 L0-1a, width 1, `--mode plan`).** A lane can be RIGHT in substance
and WRONG in citation, and the wrongness is invisible unless you check. Lane 1 correctly
concluded "diff_benchmark_report.rs silently defeats the printed-override claim" but
attached it to `crates/aprender-serve/src/gguf/cuda/mod.rs:410` — a different crate from
the defect (`crates/apr-cli/`). The real citation is that `override_line()` has exactly ONE
production call site, `comparison.rs:207`, while its own doc at
`parity_admission.rs:128-130` requires it "everywhere the env var is read … `comparison.rs`,
`diff_benchmark_report.rs`". Passing the lane's file:line through would have sent the
orchestrator to the wrong crate to fix a real bug.

Second failure mode: **a lane explains a missing artifact as a typo when it is a forward
reference.** Lane 1 found `contracts/apr-gpu-cpu-parity-v1.yaml:151` names test target
`backend_refusal_case_table` and proposed "rename it to reg15_admission.rs". But
`git ls-tree -r --name-only <sha> | grep -c backend_refusal_case_table` = 0 *files* while
`docs/specifications/PP-066-release-spec.md:150` specifies it as row C3/R-0b's acceptance —
so it is a not-yet-built target, and renaming would silently delete a real obligation.

**How to apply:** for every lane finding, (a) confirm the cited path is in the crate the
claim is about, and (b) when a lane calls a missing test/file a naming mistake, grep the
specs and roadmap for the name before accepting the rename — an absent target is usually
owed by a later row, not misspelled. Also: with `width=1` there is no dissent to arbitrate,
so this checking IS the quorum; budget for it. Staging the diff into the prompt (plan mode
opens no files) means lane line numbers are only as good as the hunk headers — treat every
lane `line` as approximate and re-derive it.

**Addendum (PMAT-1073, R-0b, width 1, `--mode goal`).** Three things the citation check
caught that the lane did not, all in a PR whose *conclusion* was right:

1. *The brief's own premise was the wrong mechanism.* Both the brief and the PR's receipt
   (`docs/audits/impl-PMAT-1073-receipt.md:78`) said "apr-cli's `wgpu` feature is
   `["inference"]` — an alias — so the wgpu inference path exists on every default build".
   Cargo features do not imply backwards: `wgpu = ["inference"]` means enabling `wgpu`
   enables `inference`, never the reverse, and a default build does not enable `wgpu` at
   all. The real mechanism is `crates/aprender-serve/Cargo.toml:49`
   (`trueno = { workspace = true, features = ["gpu"] }`, unconditional) plus its
   `default = ["server","cli","gpu"]` (:217) and `gpu = ["trueno/gpu"]` (:220), which
   compiles `WgpuFactory` into `trueno::registry::default_factories()`
   (`crates/aprender-compute/src/registry/mod.rs:643-651`, `#[cfg(feature = "gpu")]`).
   Right conclusion, wrong cause — and the wrong cause was about to be published in a
   receipt. **Read a feature-implication claim in the direction cargo actually resolves it.**
2. *A hypothesis I formed from one file died against a second.* `Request::wanted()`
   (`crates/apr-cli/src/registry.rs:67-83`) returns `Wanted::Cpu` whenever `no_gpu` is set,
   which looks like it inverts run/chat's GH-326 `--gpu`-beats-`--no-gpu` precedence. It
   does not: `crates/apr-cli/src/accel.rs:51-56` normalises (`no_gpu = no_gpu && !gpu`, and
   `--backend cpu` is dropped when `gpu`) *before* constructing the `Request`. Likewise
   "serve `--backend metal` is silently downgraded" died on
   `crates/apr-cli/src/commands_enum.rs:16`, `BACKEND_VALUES = ["cuda","cpu","wgpu"]` — clap
   refuses `metal` at parse time. Check the CALLER's normalisation and the clap value set
   before reporting a precedence or unreachable-input defect.
3. *Scope calibration changes the orchestrator's next move.* The lane ruled
   `do-not-implement-as-written` because realizar silently falls to CPU at
   `crates/aprender-serve/src/infer/gguf_gpu_generate.rs:375-391` (the `Err` arm prints only
   `if config.verbose`, and `log_cpu_backend` at
   `crates/aprender-serve/src/infer/inference_result.rs:517-520` returns early when
   `!verbose`). The defect is real — but the contract obligation the PR discharges
   (`contracts/apr-backend-registry-v1.yaml:89`, REG-OB-004) is scoped to *apr-cli*
   resolution, so the code satisfies its obligation and it is the row's prose claim that
   overreaches. "Reject the PR" and "add a caveat plus a follow-up row" are different
   actions; say which, and cite the obligation text that decides it.

**How to apply:** when a lane rules against a PR, read the contract/obligation the PR
actually claims to discharge before passing the verdict up. A lane grades the prose claim;
the obligation is what CI enforces.
