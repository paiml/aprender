# aprender-decide

**Method-neutral decision models** over `aprender-core`: answer one fixed task —
"which of these K criteria does this text belong to?" — with a calibrated
probability per criterion, in the task's criteria order.

Part of the [aprender](https://github.com/paiml/aprender) monorepo.

## The seam

`DecisionMethod` is designed for exactly one implementation today, with Kev (a
few-shot decoder classifier) as the known second:

| Method | Returns |
|--------|---------|
| `task()` | the parsed `task.json` (ordered labels) |
| `prepare(texts)` | one built row per text (tokenized once; a server prices its token budget from these) |
| `classify_prepared(rows)` | one `Decision { label_index, probabilities, tokens, truncated }` per row |
| `classify(texts)` | `prepare` then `classify_prepared` |

Probabilities are always an **array in criteria order**, never a map.

## Laya (the only method)

`aprender_decide::laya::Laya` is Laya's decision model on top of `aprender-core`'s
reusable ModernBERT encoder (`aprender::models::modernbert`): a type embedding, a
pre-norm `TransformerEncoderLayer` head with `nhead = max(1, d / 64)`, a per-marker
scorer MLP, Laya's request builder (`build_sequence`, including its option shrink
and right truncation) and its calibrated temperature buckets clamped to
`[0.5, 5.0]`. Head and scorer reuse core's `Linear`, `layer_norm`, `gelu_exact`
and `attention` — no numeric kernel is copied.

A task is refused once, at load, only when its built row would lose an option
marker (`LayaError::MarkersLost`); options that are merely long are shrunk exactly
as Laya does and served.

## task.json

`{"type": "choice", "instructions": "...", "criteria": {"name": "description" | null, ...}}`.
The **document order of `criteria` is the label index**. It is read from the raw
bytes by an order-preserving map visitor, so it is the same whether or not the
build unifies `serde_json/preserve_order` in (tested in both shapes). This crate
does not enable that feature.

## This crate is not

- a server — the thin decide MCP servers are the transport and call this crate;
- an encoder — that is `aprender-core` (D-13);
- a binary — library only.

## Contracts

- `contracts/laya-parity-v1.yaml` — torch -> `.apr` -> Rust parity ladder
  (ids and markers exact, logits <= 1e-4, probabilities <= 1e-5, argmax exact)
- `contracts/decide-apr-v1.yaml` — the task schema and the marker rule
- `contracts/laya-finetune-gate-v1.yaml` — the fine-tune gate the verifier decides

**Two different probability bars.** The FIXTURE bars are the parity ladder's: 1e-5 on
probabilities and 1e-4 on logits, applied to fixture rows only (the spike-025 rows and the
tiny fixtures). The PACK/VERIFY bar is separate: `pack_laya` re-scores every eval row of the
packed artifact (and of the declared base) and holds each set to
`max(1e-5, 4 x noise)`. Here `noise` is torch fp32's own distance from a float64 forward of the
same checkpoint. `verify::rescore_bounds` recomputes it from the hash-bound
`rescore-noise.json`, and a derived bound above the 1e-3 ceiling is refused. A run without that
record is held to the 1e-5 floor, so omitting it can only tighten the check. Argmax stays exact.
Every number behind the bar was measured on aarch64 (Apple M4). x86_64 is unmeasured
(laya-parity-v1 `qa_gate`).

**Seed selection.** A production run trains seeds 13, 17 and 23 and ships the median-ECE
seed: rank key `floor(ece_post x 10000)`, ties to the smaller seed. `verify` does not trust the
report's choice. `check_seed_selection` recomputes every seed's metrics from its hash-bound
`seeds/seed-<s>/eval-probs.json`, re-derives the median, and binds the shipped checkpoint and
eval file by sha256. A run without `seed_selection` (the 1.x rule) is never deploy-eligible, but
it is refused `SeedPolicyMissing` only after its gate, so a failing legacy run still reports
`GateFailed`.

A passing gate certifies margin and calibration on held-out data drawn like the tenant's
shots. It does not certify robustness to a shifted input population; see the gate contract's
`eval_set.claim`. The optional shift probe is reported and recomputed, never gated.

## Tests

```bash
cargo test -p aprender-decide --lib                              # all lib tests
cargo test -p aprender-decide --lib laya::tests::tiny_parity     # parity vs Laya's oracle
cargo test -p aprender-decide -p aprender-mcp-setfit --lib task:: -- --nocapture  # preserve_order=ON shape
cargo test -p aprender-decide --test ui                          # private-mint compile-fail proof (in CI)
just laya-verify-suite                                           # Python parity + real weights (LOCAL ONLY)
```

CI runs the lib tests and `--test ui` (plus `-p aprender-mcp-decide --test e2e_stdio`). The four
env-gated targets `laya_parity`, `fail_closed_vectors`, `demo_run` and `python_records` need the
Laya base snapshot, gitignored run dirs and the Python trainer. They are deliberately kept out of CI
(user decision, plan 08-12) and would only print `SKIP` there. `just laya-verify-suite` arms all four
and runs the Python self-tests; see `scripts/laya_train/README.md`.
