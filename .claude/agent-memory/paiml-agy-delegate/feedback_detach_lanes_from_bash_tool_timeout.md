---
name: detach-lanes-from-bash-tool-timeout
description: The Bash tool's 2-minute default timeout kills a foreground quorum launcher and agy reports status=ERROR "timeout waiting for response" — always launch lanes with setsid nohup + disown, then poll
metadata:
  type: feedback
---

Never run `bash launch.sh` in the foreground. Launch detached and poll:
`setsid nohup bash launch.sh > launcher.log 2>&1 & disown`, then a
`for i in $(seq 1 55); do ... sleep 10; done` poll with the Bash tool's `timeout`
raised (max 600000ms). Lanes take 5-25m; the tool's default is 120000ms.

**Why:** on PMAT-1062 the first launch died at the 2-minute mark (tool exit 143). All
three lane files existed and parsed, each with `"status":"ERROR"`,
`"error":"timeout waiting for response"`, `duration_seconds` ~110-117 — i.e. the agy
process was SIGTERM'd with the group, and the failure text *looks like* an agy-side
timeout rather than a harness kill. The tell is that `duration_seconds` equals the
Bash-tool timeout, not the `--print-timeout`; `usage.input_tokens` was 135-179k, so the
lanes were doing real work when they were killed. Relaunching detached, the same three
lanes finished in 307-377s.

**How to apply:** every `lane=quorum` and `lane=teamwork` brief. Also: `launch.sh`
ignores argv, so `bash launch.sh --dry-run` does NOT dry-run — it launches the real
fan-out. Dry-run the calling form by invoking `agy-lane.sh ... --dry-run` directly,
never through the wrapper. See [[lanes-need-a-cd-wrapper-script]].
