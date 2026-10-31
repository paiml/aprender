# impl-PMAT-6422 receipt — ship upstream-reference-v1.json inside the aprender-serve crate

Measured 2026-10-26T21:23Z at merge head of branch fix/serve-package-upstream-ref-r1 (main merged in).

## AC1 — the file is in the published crate
`cargo package --list -p aprender-serve --allow-dirty | grep upstream-reference` ->
```
data/upstream-reference-v1.json
```

## AC4 — tests compile from the published tarball
`bash scripts/package_tarball_build.sh` (exit 0):
```
PASS  all 73 published tarball(s) compile their tests (cargo build --tests)
```
GREEN, 2026-10-26T21:23Z. This covers aprender-serve-0.72.0, whose two tests include_str! the crate-local path.

## RED on revert
Not re-measured here. The defect it reverts to is the class of #4114: an include_str! of a path outside the package
(docs/kernel-registry/...) fails `cargo test --lib --no-run` from the unpacked tarball.

IMPL-PMAT-6422-RECEIPT-END
