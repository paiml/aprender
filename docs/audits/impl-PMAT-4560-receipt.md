# impl receipt — #4560 TRACE-001 TR-05 tool resolution (renacer)

Row: `scripts/renacer_bin.sh` (sourced, `return` never `exit`, mirrors `apr_bin.sh`) builds and resolves the in-tree
renacer from HEAD; `capture_golden_traces.sh` and `Makefile profile:` use it; the `cargo install` line is gone.
Contract `tool-resolution-v1`.

| Piece | What |
|---|---|
| `scripts/renacer_bin.sh` | Roots: cargo-metadata `target_directory` and `<workspace_root>/target`, release then debug. No PATH, no absolute path. A candidate must be executable, its `aprender-profile.d` must name THIS tree's `crates/aprender-profile/src/main.rs` (a foreign build is never returned), and `--version` must carry `(<git rev-parse --short HEAD>)`. If none qualifies, it runs `cargo build --release -p aprender-profile --bin aprender-profile` once and rescans; `RENACER_BIN_NO_BUILD=1` refuses instead. `RENACER_BIN` overrides the path but must still be fresh. Exports `$RENACER`. No file-scope `set`; failure is `return 1 2>/dev/null \|\| exit 1`; every `$(…)` assignment carries `\|\| x=""` |
| `scripts/capture_golden_traces.sh` | The `cargo install renacer --version 0.6.2` fallback is deleted. It sources the resolver, and the 8 bare `renacer` calls are now `"$RENACER"`. The fallback JSON `version` comes from the binary, not a hardcoded 0.6.2. `ANALYSIS.md`'s footer printed a literal `$(date)` (quoted heredoc) and "Renacer Version: 0.6.2"; it now prints real values after the heredoc. The doc snippets inside ANALYSIS.md show the resolver |
| `Makefile profile:` | `. scripts/renacer_bin.sh \|\| exit 1; "$$RENACER" --function-time --source -- cargo bench` (SHELL is /bin/bash) |
| `contracts/tool-resolution-v1.yaml` | 4 obligations, 4 falsifiers. `pv validate` (pv 0.70.0 817d63361): valid, 0 warnings |

**Name mismatch with the spec.** The row says `target/release/renacer`. No such file exists. Package `aprender-profile` has one `[[bin]]`, `aprender-profile` (`src/main.rs`), and clap names it `renacer`. The resolver returns `<target>/release/aprender-profile`. Adding a `[[bin]] name = "renacer"` would change what `cargo install aprender-profile` ships, so it is out of this row's scope.

Measured (worktree at `aca6f2d7f6`, lambda):
- `bash scripts/renacer_bin.sh --self-test`: 9 passed, 0 broken. Each row sources the resolver from a caller running `set -euo pipefail` and must print `survived rc=N`. The rows cover:
  - target absent, in bash and in zsh (the TR-05 done-when row)
  - own + fresh resolves, in bash and in zsh
  - foreign + fresh is refused
  - override fresh, override stale, and override not executable
  - sourcing is option-neutral
- Mutants, each run from a copy (worktree `git status` clean afterwards):
  - Tail `return 1 … \|\| exit 1` → `exit 1`: 6 rows BROKE, including target_absent in bash and zsh.
  - Origin check deleted: 1 row BROKE (foreign_fresh_is_refused).
  - `--version` match made unconditional: 1 row BROKE (override_stale_is_refused).
- Live, private target dir `/mnt/nvme-raid0/targets/aprender-5d-guard`:
  - First source, target absent: built (`Finished release in 1m 52s`), rc 0, `RENACER=…/release/aprender-profile`, `renacer 0.69.3 (aca6f2d7f6)`.
  - Second source with `RENACER_BIN_NO_BUILD=1`: rc 0, no build.
  - `"$RENACER" --summary -- /bin/true`: printed a syscall table, 32 calls.
- `git grep -nE 'cargo install renacer|^\s*renacer ' -- scripts Makefile`: no output, exit 1.
- `bashrs lint`: `renacer_bin.sh` has 0 errors, and so does `capture_golden_traces.sh`.
- `check_sourced_libs_option_neutral.sh`: OK, 13 libraries. Its discovery list includes `renacer_bin.sh`, which `capture_golden_traces.sh` sources.

[U] **`make chaos-test` still reaches renacer through PATH.** `crates/aprender-shell/scripts/chaos-baseline.sh` calls `command -v renacer`, and when that fails it builds `../renacer` (a pre-monorepo sibling checkout). The Makefile target also still suggests `cargo install --git …/renacer`. The done-when grep does not reach it: that path is outside `scripts/`, and the Makefile line is `command -v`, not `^\s*renacer `. Prepending the resolved dir to PATH would not work either, because the binary is named `aprender-profile`. Follow-up: make chaos-baseline.sh source `scripts/renacer_bin.sh` and call `"$RENACER"`.
[U] `capture_golden_traces.sh` was not run end to end. It builds three release examples and writes `golden_traces/`. The resolver and a real trace through `$RENACER` were measured on their own.
