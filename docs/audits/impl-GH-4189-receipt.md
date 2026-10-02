# GH-4189 receipt: G5, the RELEASE half (PR #4580)

## Scope: which half of #4189 this PR is
#4189 asks for two things: (a) the nightly builds every [[bin]] at main head and publishes a SHA
manifest for lambda, and (b) the 0.70 release gate G5, where every [[bin]] ships on the tag.

- **(a) is already on the base branch, and this PR does not touch it.** `.github/workflows/nightly.yml`
  derives every bin with `scripts/nightly_manifest.py bins` (l.129-130), builds them at the pushed
  sha, packages `<bin>-<target>.tar.gz` + `.sha256`, and publishes `nightly-manifest.json` (per-bin
  sha256, executable sha256, --version, build sha) on the `nightly` prerelease (l.24, l.63). That
  manifest is the lambda SHA manifest.
- **(b) is this PR**, per the cop's G5 brief: "Edit binary-release.yml to publish all 29 [[bin]]s,
  reusing nightly.yml's bin matrix and manifest rather than a second list. Include the
  asset_version_check.sh version-format fix."

## What the diff does
1. `build-all-bins` in binary-release.yml.
   - The bin set and the cargo selection are `nightly_manifest.py bins` of the tag tree (the nightly's
     own derivation).
   - Every bin must pass, before upload: `asset_version_check.sh` under its own name, `--help`, the
     GLIBC_2.31 floor, and `nightly_manifest.py smoke`. smoke is the nightly manifest's per-bin
     verdicts, written to the job summary.
   - After upload, every asset is read back from the release.
   - apr and pv keep their dedicated lanes.
2. `asset_version_check.sh` takes an optional BIN (default apr) and accepts pv/pv-sat's trailing
   ` (<description>)`.
3. `aprender-cgp` prints its [[bin]] name. Its clap name was `cgp`.

## Measured
- A stamped v0.70.0-rc.1 build at 9f5609568 (29 bins), with each bin's `--version` stdout checked
  under its own name: 29/29 ok. aprender-cgp was re-checked after the rename:
  `aprender-cgp 0.70.0 (9f5609568f)`.
- `asset_version_check.sh --self-test` PASS, with 12 new rows. Two mutants turn it red: ignoring the
  bin name gives 6 FAIL rows, and a greedy sha field gives 1.
- `cargo test -p aprender-cgp --lib --test integration`: 121 + 29 passed. `cargo fmt --check`: clean.
