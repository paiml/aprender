---
name: lanes-need-a-cd-wrapper-script
description: agy-lane.sh execs agy so lanes inherit cwd — fan-out must go through a bash wrapper that cd's to repo_root, never inline zsh
metadata:
  type: feedback
---

Launch quorum lanes from a written `launch.sh` run as `bash launch.sh`, not from an
inline Bash-tool command. The wrapper must `cd "$repo_root"` before the loop, and the
prompt should be staged in a separate `prompt.txt` read with `$(cat ...)`.

**Why:** two independent traps compound. `agy-lane.sh` ends in `exec "${CMD[@]}"`, so the
lane inherits the caller's cwd — a lane launched from the delegate's own worktree reviews
the wrong tree and still exits 0, which is a silent wrong answer, not a failure. And the
Bash tool is zsh, where unquoted `$var` does not word-split and the tool resets cwd between
calls, so inline `for i in 1 2 3; do ... & done; wait` with a multi-KB embedded prompt is
where the quoting breaks.

**How to apply:** every `lane=quorum` brief. Pattern that worked on PMAT-991 (width 3,
`--mode plan`, `--sandbox`): wrapper redirects each lane to `lane-$i.json` / `lane-$i.err`
and writes `echo $? > lane-$i.rc` inside the subshell — `wait` alone does not preserve
per-lane status. Check `lane-$i.err` is 0 bytes before trusting a verdict; a headless lane
that cannot run a shell command prints nothing and exits 0.

Related: when `writes=false` the brief may still point at a live worktree rather than an
exported one; `--sandbox` is the only thing preventing a lane from rewriting its `.git`,
so never drop it to "let the lane run one more check". See
[[quorum-schema-verdict-enum-mismatch]] for the mode/schema pairing.
