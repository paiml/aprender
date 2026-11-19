# Finding: why Measure hangs in the edge nightly (#7154, slice S0.1 of #7149)

Source: the `Measure` step of recent `edge-fast-verbs-nightly.yml` runs on `main`, read with `gh run view <id> --log`
and `--json jobs`. Nothing was run on the device.

| run | Measure | length | end |
|---|---|---|---|
| 42576144746 | 07:03:23 to 08:03:30 | 60 min | cancelled |
| 42776090566 | 07:48:15 to 08:48:03 | 60 min | cancelled |
| 43175475846 | 07:38:41 to 08:38:37 | 60 min | cancelled |
| 43820068891 | 07:04:33 to 08:05:28 | 61 min | `Killed`, exit 137, in the CRUX smoke loop (script line 89) |
| 44044745634 | 08:21:34 to 09:20:39 | 59 min | `##[error]The operation was canceled.` |
| 44261506568 | 07:56:02 to 10:46:47 | 170 min | the same, at job start + 180 min |

In the three runs whose logs were read (43820068891, 44044745634, 44261506568) the build ended green 9 min after
job start and the last log line before the end is the step header (`##[endgroup]` after `env:`).

1. Why red? `Measure` never ends by itself: five of six runs end near 60 min, one at the 180 min job limit.
2. Why is its position unknown? The script prints only at its end (`cat "$OUT/gaps.md"`); every per-measurement log
   (`crux-gpu.log`, `apr-bench.err`, ...) stays in the runner temp dir. Only run 43820068891 names a position, the
   CRUX loop (measurement 1), because bash reported the killed child.
3. Why can a measurement run an hour? No command has a bound but the CRUX per-cell `TMO=600`
   (`scripts/crux_inference_dogfood.sh`; 4 verbs x 2 engines x 2 devices); the 180 min job limit is the only outer one.
4. Why near 60 min and not 180? The stop is not the script's or the workflow's: 137 is SIGKILL and "operation was
   canceled" is the runner's message. Who sent it is the missing fact. Five ends at 60:07, 59:48, 59:56, 61:00, 59:05
   look like a fixed timer outside the job (device, runner service or host watchdog), not the OOM killer; that is a
   reading of the timings, not a proved cause.
5. Why not found sooner (55 runs, one success, per the S0.2 contract)? A run wrote one verdict for the job, no cell
   per measurement, so a hang could not be tied to a measurement (remedy: the S0.2 receipt, #7155).

Missing, and how to get it: on the device, `journalctl -k` for each run window (grep `oom|killed process`), the
runner service journal, and the runner `_diag/Worker_*.log` of one cancelled run. A kernel OOM line means memory; a
service restart or cancel request means a timer. Going forward each measurement is bounded by `--step-timeout` and
writes its own cell (#7156), and each workflow step has its own limit (#7157).
