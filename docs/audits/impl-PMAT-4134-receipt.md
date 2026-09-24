# Receipt — PMAT-4134: the 4 arm-only clippy findings fixed, and the first clippy gate that runs on arm

Branch `fix/4134-aarch64-clippy` off `main@49fe19c28`. Author aprender-dd (claude-opus-5-5). Commits:
- `a1d42003b`: the four fixes;
- `024318618` + `78418d7d5`: the guard;
- `0a2acd0d2`: the **workflow edit, its own commit**.

## Claim 1: the four findings are fixed, with no allow attribute
All four bindings are read only inside a `#[cfg(target_arch = "x86_64")]` AVX2 branch. Each is now consumed by
`#[cfg(not(target_arch = "x86_64"))] let _ = …;`, placed next to the branch that uses it:
- `blis/compute.rs` `dispatch_microkernel`: `(mr_block, nr_block)`;
- `brick/quant_ops/mod.rs` `DotQ5KOp::execute`: `backend`;
- `brick/quant_ops/mod.rs` `DotQ6KOp::execute`: `backend`.

Measured:
- **gx10** (aarch64, toolchain 1.93.0): `cargo clippy -p apr-cli --lib -- -D warnings` rc 0, at `a1d42003b`.
- **lambda** (x86_64): the same command, rc 0. The `let _` lines are cfg'd out there, so x86 is unchanged.

## Claim 2: an aarch64 clippy gate, with a first-green proof and a planted RED
`scripts/check_clippy_aarch64.sh` runs `cargo clippy -p apr-cli --lib -- -D warnings`. Its flags are an array, so
they reach cargo as separate words.
- On any architecture other than aarch64 it exits 2, "could not check". Measured on lambda: an x86 pass is never
  reported green for arm.
- `--self-test` appends `#[cfg(target_arch = "aarch64")] fn …{ let unused_on_arm = x; 0 }` to
  aprender-compute's `lib.rs`.
  - The restore is trapped before the plant, and the self-test refuses to plant over local edits.
  - It requires the axis RED and the finding to name `unused_on_arm`.
  - On gx10 at `024318618`: **self-test PASS**, `error: unused variable: unused_on_arm`, lib.rs:531. The real run
    is **PASS, 0 findings**, its first green on its target. The tree was clean afterwards.
- **The cuda axis is excluded by name, not by omission.** On main it is RED on EVERY architecture: `unused import
  GemmOp` in aprender-compute, measured on gx10, plus about 61 aprender-train findings behind it. All of them are
  #3837's findings, fixed by PR #4091, which is not merged. Add `cuda` to `AXES` when it lands. gx10's release binary
  is a cuda build.

## Claim 3: wiring (the workflow edit)
- No gx10-side script owns a command list a clippy run could join.
  - `silicon-nightly.yml`'s gx10 job writes its cargo commands inline.
  - `guards_nightly_manifest.txt` requires a matching step name in `guards-nightly.yml`, whose runners are x86.
  - `make fleet-verify`, cited in dogfood.sh, does not exist in the Makefile.
- So `0a2acd0d2` adds one step to `silicon-nightly.yml` job `aarch64-cuda-sm121`, after the NEON tests: the
  self-test, then the real run.
- `runs-on` is unchanged, and there is no runner, host, label or secret change.
- actionlint OK; `check_guards_are_wired.sh` PASS (ratchet did not grow); `check_workflow_cargo_packages.sh` OK;
  `check_runner_labels.sh` OK.

## Not done
- The cuda axis: #3837 / PR #4091.
- A PR-time arm clippy. This is nightly, so a finding reaches main and is caught within 24 h: rule E's 20-minute
  PR cap, decision D-1.
