# PMAT-4481: batch(ONT), #4480 + #4473 folded onto main

SPEED-BATCH ONT lane (cop aprender-77, 2026-09-26 16:36Z). B3 #4431 was not used: its head is already 3/3 (round 3 AGREED at 8b59fe3016; the head 7061eb6dd1 only adds the receipt), and the rule is never to fold into a quorumed head.

## What the diff is

| commit on this branch | cherry-picked from | origin PR | what it does |
|---|---|---|---|
| 7ffbcf6695 | c30936d9d3 | #4480 | `scripts/contracts_gate.sh`: `make contracts` runs every step, fails closed on shapes, and regenerates census.json / contracts.nt / shapes.ttl, then runs `git diff --exit-code` (#4475 steps 1-3) |
| 672fa47ddf | 91969973aa | #4473 | ONT-4c4: kernel receipts as focus nodes, `extract:kernel` over `extract:code`, and the first `#[kernel]` (#4069, #3522) |
| 7290582f90 | 09edd15f28 | #4473 | O-1: gated_rmsnorm cuda-oxide receipt on lambda (evidence + experiments/) |

All three picks applied cleanly onto main 57d7f8edd9 with no conflict, so their content is byte-identical to the originals. The picks, this receipt, the PMAT-4481 roadmap fragment and its regenerated aggregate made up the diff that was quorumed 3/3 at 367ae45bd6.

## Fix after CI (b757dd99e5, delta over c074ace09d)

On the combined tree, #4481 CI went red for two reasons that the fold itself introduced:

- **BINDING-006** (`real_binding_registries_validate`): #4473 binds the `#[kernel]` next to its CPU reference on the same (contract, equation), because the kernel shapes join on that pair. The fix adds `KernelBinding.kernel: bool` (default false). BINDING-006 now keys on (contract, equation, kernel), `find_binding` skips kernel entries, and `contracts/binding.yaml` marks the oxide entry `kernel: true`. infra-83 (ONT owner) agreed to this, on condition of the tests in `schema/artifact_tests.rs::a_reference_and_its_kernel_share_an_equation`: a reference+kernel pair passes, two kernels are RED, two references stay RED, and a kernel-only equation resolves to no reference. The other 18 edits add `kernel: false` to the test/fixture struct literals.
- **bashrs SC2171** in `scripts/contracts_gate.sh`: the jq program moves into a heredoc with no change in behaviour. The self-test still passes 16 of 16.

Checks: `cargo test -p aprender-contracts --lib binding` passes 131, and `pv lint contracts/ --gate shapes` passes. `lint::tests::lint_empty_dir` fails locally only, because PV-DUP-002 reads the repo baseline from the checkout. It is not in this PR's CI failures.

## Evidence (combined tree, HEAD-built pv, private CARGO_TARGET_DIR)

- `bash scripts/contracts_gate.sh`: `contracts gate: 6 of 6 step(s) RAN, 0 FAILED`, rc=0. This is the interaction that matters: #4480 makes the census / nt / shapes regeneration check real, and #4473 adds contracts and a new extractor. They pass together.
- `bash scripts/contracts_gate.sh --self-test`: `16 passed, 0 failed`, rc=0.
- Roadmap guards (ids-unique, sorted, fragment-required, diff-additive): see the commit that adds this file.

## Not claimed

- The #4475 coverage floor stays open (#4480 said the same).
- The per-origin PR CI runs are not re-cited here; this PR's own CI is the gate.
