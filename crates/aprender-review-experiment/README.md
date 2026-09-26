# aprender-review-experiment

PROMETHEUS (PRM-001, was REX-001) — the review-lane experiment for the Qwen 3.5 4B
review lane on a released `apr`, across fleet device cells. Part of the
[aprender monorepo](https://github.com/paiml/aprender); epic paiml/aprender#4354.

- `prereg` — the REX-00 pre-registration lock (`contracts/rex-prereg-v1.yaml`).
  `docs/audits/rex-001/prereg.lock` freezes spec §2–§5, the analysis code, the
  analysis plan and prompt v1; every receipt carries the prereg sha.
- `stats` — the frozen analysis code: Wilson, McNemar exact, seeded bootstrap
  (SplitMix64), Holm, H6 interference.

```bash
cargo run -p aprender-review-experiment --example rex -- prereg-check
```

Not published (`publish = false`): it hashes repo files via `include_str!`.
