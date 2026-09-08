# L0-1a — review quorum record (one agy lane, review-only row; 2026-09-06)

Lane: agy 1.1.27, `--mode plan`, `--sandbox`, writes=false; conversation `8b5326d5-e9a7-433c-b185-0b3a75130b86`; family Gemini 3.1 Pro (one lane, as the row's quorum column says). `num_turns=1`: the lane ran nothing; the delegate citation-checked every finding against the tree; the orchestrator re-ran the acceptance (`.pr/L0-1/accept.log`, 10/10 legs).

Verdict: **implement-with-changes** → folded in `043a47801` and `25a8f8683`.

| # | finding (delegate-verified) | disposition |
|---|---|---|
| 1 | `diff_benchmark_report.rs:82` still set `SKIP_PARITY_GATE=1` silently, contradicting the contract's "printed override everywhere" | **fixed** `25a8f8683`: the GPU half extracted into `gpu_profile_or_none()` (the file was over the complexity gate), the set_var replaced by the printed override |
| 2 | `check_model_parity.sh` judge defaulted a missing `basis` to `[U]` — a threshold file without a basis passed | **fixed** `043a47801`: refused (exit 2), self-test row 7 |
| 3 | the manifest carries alias pairs (`…-1.5b` / `…-1.5b-instruct`) that prefix-glob to one `.gguf` — measured twice | **fixed**: names iterated longest-first; an alias of a measured file prints `ALIAS` and is counted once |
| 4 | `PAR-F-003` names `backend_refusal_case_table`, which does not exist (R-0b's) | **fixed**: the test is the admission level (`reg15_admission`); the CLI-level rows are R-0b's and the contract says so |
| 5 | C14 cannot pass on any host that lacks one README-cited model (per-host RED on UNMEASURED) | **fixed**: UNMEASURED is a per-host REPORT; a host that measured nothing is not a pass; the fleet-level rule (every README-cited model on ≥ 1 GPU host) is R-5's promotion |
| 6 | `--self-test` copied the twin into a tracked fixture on every run | **fixed**: written only when absent; the case table judges under its own threshold fixture, so the live file is the sentinels' business alone |
| 7 | effective-config `not-run (cpu residency)` cannot be told from a parity refusal | **recorded for R-0b**: a refusal never constructs the GPU model; the CPU-residency report after a refusal needs the resolution row's `selected:` plumbing |
| 8 | `forced=false` at the chat call site | **documented handoff**: `apr chat`/`apr run` carry no forced-GPU flag; R-0b's `--backend` passes `forced` and the admission-level test already refuses |
| 9 | the qwen-only regex admits prose/code-block matches and drops non-Qwen names | **recorded as the admission rule** in the script header: the families the tree ships GGUFs for (qwen); llama/mistral/phi/gemma mentions are templates and comparators; widening is one edit and a claim C14 must then carry |
| 10 | line numbers approximate (plan mode) | re-located before acting |
