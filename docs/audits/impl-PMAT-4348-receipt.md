# impl receipt — #4348 (car-applicable half), branch 1c/4348-lean-scope @ car/0.70.0

Ticket: P0 Lean/leanchecker must not overload lambda. The original fix (c2efb7fa1, 1498cf9a0, 0a79b45f8) lives only on
GH-4347-dpo-on-nf4 and is in neither car/0.70.0 nor main. Its pv-discharge half (the thread cap and the 24G/800% scope
inside `pv discharge check --leanchecker`) cannot land on car: car has no crates/aprender-contracts-cli/src/commands/discharge.rs
and no lean/build.sh (provable-ladder SKIPs there). This PR ports the rest.

| Ask | This diff |
|---|---|
| 1 narrow imports | 2 bare `import Mathlib` removed. RegressionAnalytic.lean 96.8 s / 6.13 GB -> 19.6 s / 2.34 GB; HeadMapping.lean 7.4 s / 1.92 GB. rc 0, lean v4.29.0-rc4, LEAN_NUM_THREADS=2, 8G/200% scope |
| 2 tiers | forjar `lean-build` is `lake build` (trusts the Mathlib oleans); leanchecker only via pv (not on car) |
| 3 thread cap | forjar `lean-build` exports LEAN_NUM_THREADS=${LEAN_NUM_THREADS:-8} |
| 4 no native_decide | no change on car; `git grep native_decide` in lean/ = 0 |
| guard | scripts/check_leanchecker_scoped.sh (26/26 case rows) now also scans ci/*.yml, ci/*.cmd and forjar*.yaml. Planted bare leanchecker in ci/sections.yml: old guard rc=0, new rc=1. Also RED in a .cmd fragment, forjar.yaml and the Makefile. Auto-wired via guard_tree.sh --no-cargo |

Not done here (open on #4348): the "full `pv discharge check --leanchecker` within 24G/800%" acceptance needs the PVL
discharge stack in the release line first.
