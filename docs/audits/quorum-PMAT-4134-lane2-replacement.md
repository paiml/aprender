# quorum-PMAT-4134: lane 2 replacement

`quorum-PMAT-4134.json` (head `11fa390a7`) recorded lane 1 PASS (gemini-3.1-pro-high) and lane 3 PASS
(gemini-3.7-flash-high), both measured. Lane 2 (gemini-3.8-flash-high) was NO-VERDICT: agy 429 "Individual quota
reached", with every fallback step quota-exhausted. So the artifact reads `agreed: false`.

Per the operator rule "if agy quota is ever gone, simply use claude code itself" (author claude-opus-5-5, so a
**Sonnet 5** lane; `degraded: same-family`), lane 2 was re-run as a Claude Code lane on the same head, diff and
receipt.
- **Verdict: PASS.** No blocking findings.
- Measured by the lane: the guard refuses (exit 2) on x86_64, both plain and `--self-test`; actionlint,
  `check_workflow_cargo_packages.sh` and `check_runner_labels.sh` pass.
- Its one open point: the gx10 numbers were asserted, not reproducible from an x86 lane with no builds. It is
  answered by committing the gx10 logs themselves under `evidence/pmat-4134/`:
  - `dd-4134-st.log`: self-test RED on `unused_on_arm`, then PASS;
  - `dd-4134-run.log`: the real run, PASS;
  - `dd-4134-default.log` / `dd-4134-cuda.log`: the raw default and cuda-axis clippy runs.

Net: 3 PASS: 2 gemini (independent family) and 1 Sonnet 5 (same family as the author, degraded).
