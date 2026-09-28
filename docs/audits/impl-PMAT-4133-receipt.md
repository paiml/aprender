# PMAT-4133 receipt: workflow-called guards run under setsid (#4133)

- ticket PMAT-4133 (from #4133), kind code, branch `ci/4133-guard-steps-setsid` off `origin/main`; orchestrator opus-5-5. Follow-up to #4120, where a guard's test signalled pid 1 (Runner.Listener) inside the runner container.
- **Change:** every non-comment `bash scripts/check_*.sh` a workflow step calls directly (130 invocations, 13 workflows) is now `setsid --wait bash scripts/check_*.sh`. guard_tree.sh already isolates the guards it dispatches (#4120 commit 2); this covers guard-cargo, guards-nightly and the argument-taking guard-tree steps.
- **New guard** `scripts/check_guard_steps_isolated.sh`: refuses any unwrapped `bash`/`sh`/`./` or bare `run: scripts/check_…` invocation in `.github/workflows/*.y{a,}ml` and `.github/actions/*/action.y{a,}ml`. `--self-test` has 21 rows, including a mutant of the real ci.yml with one wrapper removed, which must go RED.
- **Measured:**
  - every guard-calling job runs on a self-hosted Linux runner, and the runner image has util-linux 2.39.3 `setsid --wait`;
  - from a non-interactive shell, setsid execs without forking (pid==pgid==sid), so a cancelled step still signals the guard itself;
  - before/after outputs of all 23 workflow-reading guards are identical, except a live runner count;
  - check_guards_are_wired, check_bashrs_gate, check_shell_lint_ratchet and check_explicit_test_commands PASS.
- **Quorum (ph4, 517edff93):** gemini-3.8 PASS (lane 1 VOIDED by cop ruling: a deleted foreign ref was attributed by inference only), gemini-3.7 PASS, haiku-4-5 PASS: 2/3 counted, not armable. A fresh full round follows. Record: `docs/audits/quorum-PMAT-4133.json`.
- Workflow edit: covered by the cop's standing yes (quorum + green CI, no runner-host/secret changes). Not armed (batching).
