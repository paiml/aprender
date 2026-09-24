# GH-4145: the gx10 leg fetches the exact candidate sha, not only origin/main

**Branch.** `fix/4145-gx10-fetch-candidate-sha`, stacked on `fix/4117-release-scope-callers` @ 0a6c7ef62.

**Defect.**
- `models_t1.sh`'s remote leg ran `git fetch origin main && git cat-file -e <sha>`.
- The candidate watch (#4117, REAL row `models:release-scope`) calls `models_t1.sh` on a `release/*` branch's HEAD, which is not on main.
- So gx10 refused every correct candidate as FETCH-FAILED. Found writing #4132's runbook (trap (a)).

**Change.**
- The leg tries `origin/main` first (the autopilot T-1 case, the bump's merge commit), then fetches the exact sha by id (GitHub serves any reachable commit).
- A sha the origin does not have is refused by name, before any build.

**Measured** (`scripts/check_release_models_t1.sh`, 42/42 rows):
- `candidate-on-release-branch`: gx10's clone predates a `release/9.9.9`-only candidate, and main holds only its parent. gx10 fetches it by sha and measures it (`apr 9.9.9 (<sha9>)` on gx10).
- `candidate-nowhere`: a candidate never pushed is refused with `MODELS gx10 NO-GO: no receipt -- FETCH-FAILED`, and gx10 builds nothing.
- Both rows drive `models_t1.sh` directly, as the watch does. Through autopilot they would pass vacuously: autopilot refuses a non-main commit first ("merge commit … not on origin/main"). The first version of these rows did exactly that, and its mutant "kill" was vacuous.
- Mutant `fetch-main-only` (the by-id fetch removed) is killed by `candidate-on-release-branch`: gx10 then has no receipt.
- bashrs `--no-ignore --level error` on the two changed files: 0 gating findings.
