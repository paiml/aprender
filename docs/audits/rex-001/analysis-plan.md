# REX-001 analysis plan (pre-registered, REX-00; v2 2026-09-25)

Frozen together with spec §2–§5, `crates/aprender-review-experiment/src/stats.rs`
and prompt v1 by `docs/audits/rex-001/prereg.lock`. Any byte change to one of those
files is a new spec version (R-1); data gathered under the old version is `exploratory`.

## Units

- **Item**: one corpus diff (class P, R or G; stratum S/M/L; split dev/test).
- **Arm**: (lane, model) — `apr-4b` (primary), `apr-9b` (control), `haiku`, `agy`.
- **Receipt**: one `review-experiment-receipt-v1` row per (item, cell, arm).
- **Correct**: a parsed `FAIL` on a P/R item, a parsed `PASS` on a G item.
  `Unparsed`, `NotRun{…}`, `Refused` and `ContextOverflow` are never correct and never a
  FAIL (R-6); they are counted and reported per state.

## Tests (code: `stats.rs`)

| H | Statistic | Function | Rejection rule |
|---|---|---|---|
| H1 invariance | test items whose VERDICT differs from reference cell C2 (v2; byte divergence is descriptive) | `verdict_divergence`, `count_rule_p` | count > 0 |
| H2 determinism | divergent items on the 10 % same-cell re-run | `count_rule_p` | count > 0 |
| H3 size | McNemar exact, one-sided, `b` = 9B right & 4B wrong | `discordant`, `mcnemar_exact_one_sided` | Holm-adjusted p ≤ 0.05 |
| H4 usefulness | precision(4B) − min(precision(haiku), precision(agy)), min computed INSIDE each resample; paired bootstrap, 10 000 resamples, seed 4354 | `vs_voters_parsed`, `bootstrap_vs_min_voter(Rate::Precision)` | holds iff the Holm-adjusted one-sided p (share of resamples < 0) ≤ 0.05; the 95 % CI is report-only |
| H5 recall | recall(4B) − recall(Claude lane) on class-R items only, paired bootstrap as H4 | `paired_parsed_class(.., Class::R)`, `paired_bootstrap_diff(Rate::Recall)` | as H4. Class-P recall and precision are descriptive |
| H6 interference | mean(co-located) − mean(alone), 5 + 5 runs | `interference` | slowdown > 2 × sd(alone) |

Family H1–H6, Holm step-down (`holm`), α = 0.05. H1, H2 and H6 are deterministic rules and
enter the family as p = 0 (rejected) or p = 1 (not rejected).

For H4 (v2) there is no fixed comparison lane: each resample subtracts whichever voter has the
lower precision ON THAT RESAMPLE. Items where apr or either voter is not a parsed verdict are
dropped and counted. For H5 the comparison is the Claude lane (Haiku 4.5), pairwise-dropped.

## Descriptive

Per cell: p50/p95 review wall-clock (nearest rank, `percentile`) with a bootstrap CI of the
p95 (same seed), TTFT, prefill and decode tok/s, peak anon memory, parse rate, and the ratio
to llama.cpp `d1d3c3396` on the same cell. Recall/precision/false-refute/localization with
Wilson 95 % intervals (`wilson`).

Error overlap (v2, descriptive): Cohen's κ on per-item errors between apr and each voter
(`error_pairs`, `error_kappa`), measured before and after every silver-label training step;
`kappa_andon` fires when κ rises and recall does not.

Sensitivity: every per-cell number is reported twice — all runs (primary) and runs with
loadavg1 < 0.5 × physical cores (secondary). Decisions read the primary column only.

## Sample-size rule (§2.2)

After REX-05, project the test-split Wilson half-width on recall at the pilot's point estimate
(`wilson_half_width`). If > 0.12, the corpus grows; hypotheses do not change.

## Seeds

- Run order per cell: SplitMix64 seed 4354 (the epic number), Fisher–Yates.
- Determinism re-run subset: the first 10 % of the seed-4354 order of test items.
- Bootstrap: seed 4354.
