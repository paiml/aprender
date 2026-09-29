# Finding: intel is over-subscribed during the 0.70 cut (2026-09-28)

Samples are `/proc/loadavg` 1-min on intel (mac-server, `nproc` = 32), taken at each la-73 pickup.

| UTC | 1-min load | load / core |
|-----|-----------:|------------:|
| 13:48 | 83 | 2.6 |
| 14:18 | 11 | 0.3 |
| 14:48 | 199 | 6.2 |
| 15:18 | 49 | 1.5 |
| 15:48 | 92 | 2.9 |
| 16:18 | 210 | 6.6 |
| 16:48 | 65 | 2.0 |
| 17:18 | 78 | 2.4 |
| 17:48 | 96 | 3.0 |
| 18:18 | 115 | 3.6 |
| 18:48 | 14 | 0.4 |
| 19:18 | 139 | 4.3 |
| 19:48 | 79 | 2.5 |
| 20:19 | 150 | 4.7 |
| 20:48 | 94 | 2.9 |
| 21:18 | 136 | 4.2 |
| 21:48 | 118 | 3.7 |
| 22:18 | 97 | 3.0 |
| 22:48 | 102 | 3.2 |
| 23:18 | 109 | 3.4 |
| 23:48 | 108 | 3.4 |
| 00:18 (09-29) | 48 | 1.5 |
| 00:48 (09-29) | 134 | 4.2 |
| 03:18 (09-29) | 11 | 0.3 |
| 03:48 (09-29) | 36 | 1.1 |
| 06:48 (09-29) | 211 | 6.6 |
| 14:24 (09-29) | 706 | 22.0 |

At 14:48Z the top processes were root `clippy-driver` and `ld.mold` (CI runners) plus two noah test binaries. Only 3 of 27 samples (14:18, 18:48, and 03:18 on 09-29) were under 1 load per core. The first two windows closed within 30 min: a serve re-run started at 18:48 was stopped by PID at 19:18 at load 139 (partial 7498 ok / 0 fail). The 03:18 window lasted long enough: re-run 3 finished in 1673 s (16149 passed / 0 failed / 62 ignored), and by 03:48 the load was back to 36. Before the power loss the peak was 211, at 06:48 on 09-29. The hosts lost power at about 14:00Z on 09-29. At 14:24Z, just after recovery, the 1-min load was 706 (5-min 475, 15-min 235), and it was still rising. The whole fleet was rebuilding and re-running at once, so this is a recovery load spike, not steady state. The cop called an ANDON above 400: no local test runs on intel until load1 ≤ 32.

## What follows
- **F3:** the aprender-serve lib suite finished only when started inside a quiet window (03:18Z on 09-29, load 11): 16149 passed / 0 failed / 62 ignored at fd02954a8. Two earlier starts were stopped by PID at load 139 and 199. See `F3-pr-draft.md`.
- **Local-gate step 2** (time the CI groups on fw16/intel, after 0.70 final): a timing taken on intel at this load measures contention, not the gate. Time on an idle host, and record `loadavg` before and after each group as a precondition (it must be < 1.0 per core, or the sample is `not_measured`).
