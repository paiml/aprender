# Oracle fixtures for `aprender-forecast`

These 21 files are the phase's **correctness bar**. The parity ladder does not check that
the Rust port is self-consistent; it checks that the port reproduces a specific number
produced by a specific version of a specific Python library on a specific dataset. That is
what makes a green ladder evidence.

**That count is a GATED CLAIM, not a comment.** `test_support::tests::
every_committed_fixture_is_readable_and_the_readme_count_is_true` enumerates this directory,
parses the number out of the sentence above and asserts they agree. The counting rule is
every entry here EXCEPT `README.md` itself, which documents the fixtures rather than being
one. Re-derive the sentence when you add a file; do not increment it by hand — the old
hand-kept list went stale three times without any build noticing.

**They are committed on purpose, and their absence is a defect.** `test_support::load_json`
`expect`s — no test in this crate skips because a fixture is missing. A skipped parity test
is a parity test that proves nothing (CLAUDE.md Verification Discipline #5).

Every file here is **byte-identical** to its original under `.planning/spikes/NNN-*/fixtures/`;
`06-01` Task 2's `<verify>` re-establishes that with `cmp` on every run, so a drifted copy
turns the plan red rather than quietly changing a tolerance. Where a file appears in more
than one spike (`peyton_manning.csv` in 7, `air_passengers.csv` in 6, `wp_log_R.csv` in 2)
the copies were verified identical to each other before ONE was taken.

## Provenance

| File | Spike | Oracle environment | Regenerate with |
|---|---|---|---|
| `peyton_manning_prophet140.json` | `001-prophet-map-fit-lbfgs` | Python Prophet 1.4.0 | `.planning/spikes/001-prophet-map-fit-lbfgs/tools/` |
| `air_passengers_prophet140.json` | `001-prophet-map-fit-lbfgs` | Python Prophet 1.4.0 | `.planning/spikes/001-prophet-map-fit-lbfgs/tools/` |
| `retail_sales_prophet140.json` | `001-prophet-map-fit-lbfgs` | Python Prophet 1.4.0 | `.planning/spikes/001-prophet-map-fit-lbfgs/tools/` |
| `np_oracle_peyton.json` | `002-neuralprophet-autograd` | NeuralProphet 0.9.0 | `.planning/spikes/002-neuralprophet-autograd/tools/` |
| `peyton_default_prophet140.json` | `003-prophet-intervals-and-components` | Python Prophet 1.4.0 | `.planning/spikes/003-prophet-intervals-and-components/tools/` |
| `peyton_holidays_prophet140.json` | `003-prophet-intervals-and-components` | Python Prophet 1.4.0 | `.planning/spikes/003-prophet-intervals-and-components/tools/` |
| `wp_log_R_logistic_prophet140.json` | `003-prophet-intervals-and-components` | Python Prophet 1.4.0 | `.planning/spikes/003-prophet-intervals-and-components/tools/` |
| `air_multiplicative_prophet140.json` | `003-prophet-intervals-and-components` | Python Prophet 1.4.0 | `.planning/spikes/003-prophet-intervals-and-components/tools/` |
| `chronos_bolt_tiny_fixture.json` | `005-chronos-bolt-tiny-parity` | chronos-forecasting 2.3.1 | `.planning/spikes/005-chronos-bolt-tiny-parity/tools/` |
| `chronos_probes.json` | `005-chronos-bolt-tiny-parity` | chronos-forecasting 2.3.1 | `.planning/spikes/005-chronos-bolt-tiny-parity/tools/` |
| `weights_index.json` | `005-chronos-bolt-tiny-parity` | chronos-forecasting 2.3.1 | `.planning/spikes/005-chronos-bolt-tiny-parity/tools/` |
| `chronos_holdout_oracle.json` | `006-chronos-vs-prophet-holdout` | chronos-forecasting 2.3.1 | `.planning/spikes/006-chronos-vs-prophet-holdout/tools/` |
| `peyton_tiny_oracle.json` | `007-chronos-mcp-thin-server` | chronos-forecasting 2.3.1 | `.planning/spikes/007-chronos-mcp-thin-server/tools/` |
| `retail_regressors_prophet140.json` | `011-prophet-regressor-parity` | Python Prophet 1.4.0 under Python 3.12 | `.planning/spikes/011-prophet-regressor-parity/tools/oracle.py` |
| `retail_regressors_holidays_prophet140.json` | `011-prophet-regressor-parity` | Python Prophet 1.4.0 under Python 3.12 | `.planning/spikes/011-prophet-regressor-parity/tools/oracle_holidays.py` |
| `invariance_baseline.json` | `012-no-argument-invariance-gate` | not an oracle — see below | captured from this repo at `captured_at_commit`, never regenerated |
| `peyton_manning.csv` | `002` (7 identical copies across 001-007) | raw dataset (Prophet's example data) | — |
| `air_passengers.csv` | `004` (6 identical copies) | raw dataset | — |
| `wp_log_R.csv` | `003` (2 identical copies) | raw dataset (Prophet's logistic example) | — |
| `retail_sales.csv` | `006` | raw dataset | — |
| `chronos_bolt_tiny_config.json` | not a spike copy — see below | `amazon/chronos-bolt-tiny` @ `a0e552de8349…` | `just fetch-chronos-tiny`, then copy `models/chronos-bolt-tiny/f32/config.json` |

## Gotchas the readers must handle (`src/test_support.rs`)

- `wp_log_R.csv` is **not chronological** — sort and de-duplicate by `ds` before use.
- Two of the four CSVs (`peyton_manning`, `wp_log_R`) carry `"`-quoted fields, two
  (`air_passengers`, `retail_sales`) do not. All four are LF-terminated as committed
  (measured 2026-09-05); the reader strips a trailing `\r` anyway so a CRLF checkout on
  another platform cannot silently change a parsed value.
- Prophet's objective is stored at `log_posterior_at_map_unnormalized` as a **log
  posterior** (positive), so the Rust objective (a negative log posterior) is compared as
  `f_rust + log_posterior_at_map_unnormalized`, and the fit's `1/T` scale must be undone
  first.

## Which plan consumes what

| Fixture family | Plan |
|---|---|
| `peyton_manning_prophet140` (rung 2) | `06-01` (this plan's one ladder rung) |
| the four remaining `*_prophet140` | `06-03` (the seven-fixture ladder) |
| `np_oracle_peyton` | `06-04` (NeuralProphet-lite) |
| `chronos_*`, `weights_index`, `peyton_tiny_oracle` | `06-05` / `06-07` (Chronos) |
| `retail_regressors_prophet140` | `06.1-01` (the tracer, 24 columns) and `06.1-02` (the full rung ladder) |
| `retail_regressors_holidays_prophet140` | `06.1-02` (the 30-column rungs, where holidays and regressors coexist and the column sort is load-bearing) |
| `invariance_baseline` | `06.1-01` (the D-19 no-argument bitwise invariance gate, SC2) |

## `invariance_baseline.json` — the one fixture that is not an oracle at all

Every other file here is either a byte-identical spike copy or a pinned upstream artifact.
This one is **evidence captured from this repository**, at the commit its own
`captured_at_commit` field names, BEFORE the regressor splice existed — a baseline captured
after the change would be circular. It carries per-architecture `ForecastResponse` signatures
for eight pre-change door cases, keyed by `std::env::consts::ARCH`, and it is never
regenerated: regenerating it would replace the evidence with a restatement of current
behaviour.

## `chronos_bolt_tiny_config.json` — the one fixture that is not a spike copy

Every other file here is byte-identical to a `.planning/spikes/NNN-*/fixtures/` original.
This one is the 1.1 KB `config.json` that `amazon/chronos-bolt-tiny` serves at revision
`a0e552de83495b5c28c14c71c374f3e33280b340`, copied out of `models/chronos-bolt-tiny/f32/`
after `just fetch-chronos-tiny` verified it. Its sha256 is
`278f0086733031635fb1c861cb01c1bad6477420c7fcb19381a2993e335785e0` — the same pin the recipe
enforces, so a test can assert it and catch a swapped file.

It is committed because it is **hyper-parameters, not weights** (D-18 forbids the latter and
permits the former): `d_model`, `num_layers`, `chronos_config.quantiles`, `context_length`,
`prediction_length`. That is what lets the validation and shape tests build a real `Config`
and run with no weights on disk at all — the weight-*dependent* tests stay gated behind
`cfg(chronos_weights)` and report as counted skips instead.
