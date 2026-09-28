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

At 14:48Z the top processes were root `clippy-driver` and `ld.mold` (CI runners) plus two noah test binaries. Only 1 of 10 samples was under 1 load per core.

## What follows
- **F3:** the aprender-serve lib suite cannot finish on intel while the cut runs. A partial run was stopped at 16143 ok / 0 fail; see `F3-pr-draft.md`.
- **Local-gate step 2** (time the CI groups on fw16/intel, after 0.70 final): a timing taken on intel at this load measures contention, not the gate. Time on an idle host, and record `loadavg` before and after each group as a precondition (it must be < 1.0 per core, or the sample is `not_measured`).
