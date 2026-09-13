---
name: capture-per-lane-exit-codes
description: A quorum launcher must capture each lane's exit code into a file; bare `wait` discards them and the receipt can only infer PASS from agy's own status field
metadata:
  type: feedback
---

A quorum launcher must write **each lane's exit code to its own file** (`echo $? > lane-$i.rc`
per background job, or `wait "$pid"; rc=$?` per recorded pid). Bare `wait` at the end of the
loop returns only the last job's status, so every other lane's exit code is gone.

**Why:** on PMAT-1096 round 2 the launcher used `for i; do ... & done; wait`. All three lanes
produced valid JSON with `"status":"SUCCESS"` and 0-byte stderr, but the delegate receipt could
only report `exit: null` — the exit code had to be *inferred* from agy's own self-reported
status. A lane self-reporting SUCCESS is a claim by the thing being measured; the process exit
code is the independent signal, and it is the one the receipt schema asks for. This is the same
class as "never read `$?` through a pipe" (see the project's verification-discipline rules):
the status you print must be the status you actually measured.

**How to apply:** in any detached quorum launcher, per lane:
`bash "$LANE" ... > "$OUT/lane-$i.json" 2> "$OUT/lane-$i.err"; echo $? > "$OUT/lane-$i.rc" &`
Then the receipt's `results[].exit` is measured, not inferred. Also note that agy-lane.sh
reserves **exit 3** for an isolation violation on `--writes` lanes — an exit code that is never
captured is an isolation violation that can never be reported.

Related: [[quorum-lane-launch-form]]
