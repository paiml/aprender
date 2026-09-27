# Receipt — PMAT-4130: four crates no longer ship tests that cannot compile from their .crate

Branch `fix/4130-shipped-tests-path-dev-deps` off `main@49fe19c28`. Author aprender-dd (claude-opus-5-5).
Commits:
- `526e898db`: package excludes (plus a build.rs cfg that was later replaced);
- `da71f449d`: the run-time read pattern, replacing that cfg;
- `d1f3c584a`: the aprender-db generated file.

The must-RED is f5's tarball gate, `scripts/package_tarball_build.sh` (#4114, branch `fix/4114-tarball-build-gate`).
It ran on lambda's RAID, build-only (no test is executed), with `CARGO_BUILD_JOBS=8` under the cargo memcap, in a
private target that was deleted afterwards (81G). That setup is the cop's ruling.

## Baseline (main 49fe19c28, gate @ccbeff0c0): the four crates RED
- aprender-core: 12 integration targets plus the lib test.
- aprender-orchestrate: `cuda_edge_cases`.
- aprender-cbtop: `tui_pixel_f301`.
- aprender-test-showcase: `probar_tests`, `gui_coverage_tests`.

Five more crates were RED. Each has its own ticket and owner, so they are out of scope here: train, contracts,
contracts-cli and present-terminal are #4129 (f5); aprender-serve is #4048.

## Two causes, and NOT re-versioning
PMAT-955 and preflight R6 require every sibling dev-dependency to stay path-only (the 0.65.0 publish cycle stuck
the cascade at 48/74). So giving `jugar-probar`/`provable-contracts`/`trueno-cuda-edge` a version is not an option.
The two causes are:
1. **Integration targets** that `use` a stripped path-only dev-dep, or `include_str!` a file outside the crate
   (`contracts/`, `CLAUDE.md`, `LICENSE`, apr-cli sources), cannot compile from the .crate at all.
   - They are excluded from the package, root-anchored `/tests/x.rs` (CB-510), and cargo drops the target.
   - They still run in the workspace.
   - Per crate: core 12 files plus `/tests/contracts/`, orchestrate 1, cbtop 1, test-showcase 2.
2. **aprender-core lib unit tests** that `include_str!("../../../../contracts/…")` are now read at run time.
   - This is the ONE pattern agreed with f5 at the cop's request: #4129's `schema::workspace_contract_or_skip`, which
     #4048 also uses.
   - The helper is `test_support::workspace_contract_or_skip`, with the same name and semantics.
   - IN TREE, a missing file PANICS.
   - With no `contracts/` at all (the .crate), it prints `SKIP <test>: out of tree …` at column 0 and returns.
   - Converted: ship_001/003/004/010, provenance, setfit `mod backend` (CONTRACT/BINDING become cached run-time
     readers, and `binding_row` takes the text), the rung-ladder test and the truncation-probe test.
   - The backend doc's reason for `include_str!` ("a run-time read would be a silent skip") is kept and answered: in
     tree it panics.
   - The first attempt (`526e898db`, a build.rs `cfg(aprender_monorepo)`) compiled the tests out SILENTLY and was
     replaced.

## Found while measuring
Every build left the tree dirty.
- `aprender-db/build.rs` writes `src/generated_contracts.rs`. It was untracked and NOT ignored (its comment says
  "gitignored"; `git check-ignore` matched nothing).
- As a result `cargo package` refused any tree that had been built in, and the gate printed "cannot check".
- The 25 other crates TRACK this file, so the exact stub the build writes is committed: no build-time change.
- f5 proposed a `.gitignore` entry. It was not used, because it would have made aprender-db the one exception.
- The existing `.gitignore` line `src/generated_contracts.rs` is root-anchored and never matched a crate's copy.

## Measured
- **In tree** (`cargo test -p aprender-core --lib`):
  - the converted tests pass: 5 (default) and 56 (`--features setfit`), with no SKIP line;
  - `test_support`'s three outcomes are unit-tested against a temp root (skip / read / missing → panic), plus one
    test that this build resolves in tree;
  - mutant (the in-tree-missing arm returns `None` instead of panicking): exactly `in_tree_missing_file_panics`
    FAILED; restored with `git checkout --` and a trap.
- **Gate, final tree**:
  - Run 1 (gate @ccbeff0c0): my four crates are absent from RED.
    - It also showed rows naming no file ("could not compile (no diagnostic attributed to a tarball path)") for 2 core
      tests and 4 train-* crates.
    - A `realizar-977871` test process (not this run: the gate executes nothing) was OOM-killed on lambda at 03:38
      CEST, inside that window.
  - Run 2 (gate @556994b7f, with f5's counting), same tree:
    - RED is exactly the five out-of-scope crates. aprender-core, aprender-orchestrate, aprender-cbtop and
      aprender-test-showcase are not RED.
    - The extra rows did not reproduce. They are recorded here as seen once and unexplained, not as explained.
  - Run 2 prints what the tarballs drop:
    - `NOT SHIPPED` lists aprender-core 13, orchestrate 1, cbtop 1, test-showcase 2;
    - `SHRINK: 390 integration test target(s) not shipped across 21 crate(s); 51 run-time skip site(s) across 5
      crate(s)`.
- `bash scripts/check_include_files.sh`: OK, 1795 files.

## Not done
- The five other RED crates: #4129 and #4048, their owners.
- Real `pv codegen` output for aprender-db's `generated_contracts.rs` (the committed file is the build's stub).
