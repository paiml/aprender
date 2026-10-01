# SetFit / MiniLM conformance fixture generator

The hash-locked Python reference environment (D-12) that produces every frozen fixture
the Phase 1 conformance gates compare against.

**This is a deliberate developer workflow. It is never wired into CI** — no build script,
no test, and no Makefile tier invokes it. CI consumes the *committed* fixtures only.

## Regenerate the committed fixture corpus

```bash
cd scripts/setfit_fixtures
uv run python slice_model.py        # pinned download + digest check + slice -> slice_model.apr
uv run python generate_fixtures.py  # the whole ENC-01..06 corpus + manifest.sha256
```

`slice_model.py` must run first: it writes `slice_config.json`, `vocab_remap.json`,
`tokenizer.json` and the slice APR that the generator loads.

Any regeneration shows up as a diff in
`crates/aprender-core/tests/fixtures/setfit/manifest.sha256` (D-13). That is the point —
re-baselining a fixture to make Rust pass must be a visible, reviewable act, not a quiet
one.

## Full-weight (~90 MB) parity artifact — D-10, opt-in

```bash
uv run python fetch_full_weights.py
```

Writes `full_model.apr` plus `config.json` / `tokenizer.json` / `modules.json` /
`1_Pooling/config.json` and a `full_manifest.json` (source safetensors digest **and**
produced APR digest) into `$APRENDER_MINILM_DIR`
(default `~/.cache/aprender/minilm-l6-v2-1110a243`). The checkpoint is deliberately not
vendored in git. The script prints the exact `cargo test` command to run next.

## Pins

| Package | Version | Why |
|---------|---------|-----|
| `setfit` | 1.1.3 | D-12 reference pin |
| `sentence-transformers` | 5.7.0 | D-12 reference pin; pooling/normalize semantics of record |
| `torch` | 2.13.0 | D-12 reference pin; every fixture number is an artifact of it |
| `scikit-learn` | 1.9.0 | D-12 reference pin |
| `transformers` | `>=4.41.0,<5` | constraint, not a reference pin — see below |

Model: `sentence-transformers/all-MiniLM-L6-v2` at the immutable revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (never a branch name). Per-file sha256s are
recorded in `upstream_manifest.json` and re-verified fail-closed on every run.

### Two environment constraints that are not obvious

**`transformers` is capped below 5.** `setfit` 1.1.3 declares an unbounded
`transformers>=4.41.0`. A fresh resolve therefore picks `transformers` 5.x, where
`default_logdir` was removed from `transformers.training_args` — and `import setfit` dies
with `ImportError`. `sentence-transformers` 5.7.0 already caps at `<6.0.0`; setfit is the
one missing the cap, so this project supplies it. 4.57.x is also the exact series
`01-RESEARCH.md` verified BERT dropout placement against.

**`requires-python` is capped below 3.14**, because `transformers` 4.x has no 3.14
wheels and an open-ended `>=3.13` makes uv resolve an unsolvable 3.14 split.

**`.python-version` pins 3.13.7, not `3.13`.** On an arm64 machine a bare `3.13` request
can resolve to a stale *x86_64* managed CPython, and `torch` 2.13.0 publishes no
macOS-x86_64 wheel — install then fails with a platform error that reads like a bad
`torch` pin but is really a wrong-architecture interpreter. Pinning the patch version
disambiguates: uv selects the native-arch 3.13.7 (and downloads a native build if absent).
