<!-- PCU: cli-canary | contract: contracts/apr-page-cli-canary-v1.yaml -->

# apr canary

Manage canary tests for regression

**Category**: Quality & Evaluation

## Synopsis

```text
apr canary [OPTIONS]
```

## Example

```bash
apr canary qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

## What this does

`apr canary` records and checks behavioral fixtures — a known input (audio,
text, JSON) paired with the model's reference output. Use it to guard against
silent regressions: if Qwen2.5-Coder used to translate `"def add(a, b):"` into
exactly `"\n    return a + b\n"`, a canary captures that, and any later change
that flips the answer fails CI.

## Key flags

| Subcommand | What it does | Example |
|-----------|-------------|---------|
| `canary create` | Record a new canary from a model + input | `--input prompt.txt --output ref.json` |
| `canary check` | Compare model against a saved canary | `--canary ref.json` |
| `--json` | Machine-readable output | `--json` |

## Common workflows

**Capture a regression fixture during golden-model bring-up.**

```bash
apr canary create qwen2.5-coder-0.5b.apr \
    --input prompts/fizzbuzz.txt \
    --output canaries/qwen-0.5b-fizzbuzz.json
```

**Gate every PR on the saved canary set.**

```bash
for c in canaries/*.json; do
    apr canary check qwen2.5-coder-0.5b.apr --canary "$c" --json || exit 1
done
```

## Troubleshooting

- **Canary fails after harmless refactor** — sampler stochasticity. Re-run with
  `--temperature 0.0 --seed 299792458` (the project's standard deterministic
  seed); regenerate the canary if the change is intentional.
- **`canary create` segfaults on audio input** — confirm the input file format
  matches the model's expected modality (WAV 16kHz mono for Whisper).
- **All canaries fail after a model swap** — that's the point. Canaries are
  per-model fixtures; recreate them when you intentionally change weights.

## See also

- Source: [`crates/apr-cli/src/commands/canary.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/canary.rs)
- Contract: [`contracts/apr-page-cli-canary-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-canary-v1.yaml)

