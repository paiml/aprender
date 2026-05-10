# SHIP MODEL-1 Discharge Scripts

LIVE dispatch scripts for the 5 MODEL-1 PARTIAL discharges that were
unblocked by SHIP-007 §22 / aprender PR #1550 (merged 2026-05-07).

Each script runs the canonical command from its contract's
`full_discharge_blocks_on:` clause, parses the output, writes evidence
to `evidence/ship-XXX-full-discharge/discharge-evidence-v1.json`, and
exits 0 (Pass) or 1 (Fail).

## Discharge matrix

| SHIP | Contract                              | Falsification ID         | Pass criterion                                |
|------|---------------------------------------|--------------------------|-----------------------------------------------|
| 002  | `qwen2-e2e-verification-v1.yaml`      | `FALSIFY-QW2E-SHIP-002`  | `def fib(n):` completion parses, 0 syntax errors |
| 005  | `qwen2-e2e-verification-v1.yaml`      | `FALSIFY-QW2E-SHIP-005`  | Median HumanEval pass@1 ≥ 86.00 (or ≥ 84.80 with 1.2 pp noise allowance) |
| 006  | `apr-model-qa-v1.yaml`                | `FALSIFY-QA-SHIP-006`    | All 8 `apr qa` gates report `"pass": true` |
| 007  | `qwen2-e2e-verification-v1.yaml`      | `FALSIFY-QW2E-SHIP-007`  | Median decode ≥ 30.0 tok/s (RTX 4090, 7B Q4_K) |
| 008  | `chat-template-v1.yaml`               | `GATE-CHAT-SHIP-008`     | Rendered ChatML prompt byte-exact to spec golden |

## Operator workflow

Run on the canonical RTX 4090 host (`noah-Lambda-Vector`, lambda-labs):

```bash
# Default: uses /mnt/nvme-raid0/targets/aprender/release/apr (must be built --features cuda)
bash scripts/ship-discharges/ship-002-discharge.sh
bash scripts/ship-discharges/ship-005-discharge.sh
bash scripts/ship-discharges/ship-006-discharge.sh
bash scripts/ship-discharges/ship-007-discharge.sh
bash scripts/ship-discharges/ship-008-discharge.sh

# Or run all five sequentially (any non-zero exit aborts):
for s in scripts/ship-discharges/ship-*.sh; do
    bash "$s" || { echo "STOP: $s failed"; exit 1; }
done
```

## Common arguments

All five scripts accept the same baseline flags:

- `--apr-binary <path>` — override the apr binary path
  (default: `/mnt/nvme-raid0/targets/aprender/release/apr`).
- `--model <path-or-hf-id>` — override the canonical 7B teacher
  (default: `paiml/qwen2.5-coder-7b-apache-q4k-v1[.safetensors]`).

Per-script flags:

- `ship-005-discharge.sh`: `--threshold <pct>`, `--runs <n>` (default 86.00, 3)
- `ship-007-discharge.sh`: `--iterations <n>`, `--max-tokens <n>`, `--threshold <tps>` (default 5, 128, 30.0)

## Evidence emission

Each run writes one canonical evidence file:

```
evidence/ship-002-full-discharge/discharge-evidence-v1.json
evidence/ship-005-full-discharge/discharge-evidence-v1.json
evidence/ship-006-full-discharge/discharge-evidence-v1.json
evidence/ship-007-full-discharge/discharge-evidence-v1.json
evidence/ship-008-full-discharge/discharge-evidence-v1.json
```

Schema mirrors the existing DISCHARGED evidence files (SHIP-001/003/004):

- `host.hostname`, `host.apr_binary`, `host.apr_version`
- `command` — exact canonical command run
- `verdict_from_<rule>` — the algorithm-level verdict-fn name
- `overall` — `"PASS"` or `"FAIL"`

After a Pass run, the operator updates the contract YAML's
`discharge_status: PARTIAL_ALGORITHM_LEVEL` → `DISCHARGED`, attaches the
JSON evidence file under `discharged_evidence:`, and amends the changelog.

## Prerequisites

| Tool        | Used by              | Purpose                                |
|-------------|----------------------|----------------------------------------|
| `jq`        | 005, 006, 007, 008   | JSON parsing                           |
| `cmp`       | 008                  | Byte-exact diff                        |
| `sha256sum` | 008                  | Golden / rendered SHAs in evidence     |
| `awk`       | 005, 007             | Numeric threshold comparison           |
| Python parser | 002                | `ruff`, `rustpython`, or `python3`     |

The 7B Q4_K teacher (`paiml/qwen2.5-coder-7b-apache-q4k-v1`) must be
fetched ahead of time via `apr pull` if running offline.

## Lint

```bash
bashrs lint scripts/ship-discharges/*.sh
```

## Why these 5

SHIP-007 §22 (apr-vs-gguf-forward-parity) was the upstream blocker: until
its bisection landed (PR #1550, merged 2026-05-07), the 7B teacher's
`apr run` path was numerically suspect, so any LIVE evaluation was
contaminated. With §22 closed, the 5 PARTIAL_ALGORITHM_LEVEL discharges
that depend on actually exercising the teacher are now LIVE-dispatch-ready.

Each script's purpose is to flip the contract from
`PARTIAL_ALGORITHM_LEVEL` → `DISCHARGED` by attaching live RTX 4090
evidence to the algorithm-level verdict fn that already exists.
