# aprender-contrastive-data

Deterministic, leakage-safe contrastive data construction — class buckets, balanced
few-shot selection, bounded pair sampling.

Part of the [Aprender](https://github.com/paiml/aprender) monorepo.

## What it does

Builds the pair dataset that contrastive/Siamese fine-tuning (SetFit) trains on: strict
JSONL ingest, typestate split roles, dataset attestation, cross-split deduplication,
balanced k-shot selection, bounded positive/negative pair sampling, and a canonical
manifest for every artifact along the way.

```toml
[dependencies]
aprender-contrastive-data = "0.64"
```

## Three constraints shape the API

**Bytes in, bytes out.** No filesystem access, no sockets, no path-shaped parameters —
not even in tests. `apr-cli` owns every filesystem adapter. The destination for these
artifacts is object storage, where a manifest is an object rather than a file; an API
that speaks `&Path` makes such a consumer a rewrite instead of a wrapper. Enforced by
`make contrastive-data-boundary`, which checks the resolved dependency closure against a
positive allowlist and bans `std::fs`/`std::net`/`std::path` under `src/` — including the
grouped-import spellings rustfmt produces.

**Determinism.** Every random decision is a pure function of its draw ordinal, taken from
the counter-based Philox generator in `aprender-rand`. Draw *i* cannot depend on how many
draws preceded it, because nothing precedes it. No `HashMap` or `HashSet` appears in
`src/`; ordered maps only.

**Typestate.** A `Split<Train>` cannot be built from validation bytes, a compatibility
dataset has no validation witness, and pairs cannot be built from raw ids. Five such
misuses are proven uncompilable with `trybuild` and committed `.stderr` snapshots.

## Contract

`contracts/contrastive-pair-protocol-v1.yaml` — 24 equations, 15 proof obligations,
20 falsification tests, 2 Kani harnesses.

```bash
pv validate contracts/contrastive-pair-protocol-v1.yaml
pv audit    contracts/contrastive-pair-protocol-v1.yaml --binding contracts/aprender/binding.yaml
```

## Links

- [Monorepo](https://github.com/paiml/aprender)
- [Contract](https://github.com/paiml/aprender/blob/main/contracts/contrastive-pair-protocol-v1.yaml)
