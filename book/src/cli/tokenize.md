<!-- PCU: cli-tokenize | contract: contracts/apr-page-cli-tokenize-v1.yaml -->

# apr tokenize

Tokenizer training pipeline (plan/apply) — BPE vocabulary learning

**Category**: Training

## Synopsis

```text
apr tokenize [OPTIONS]
```

## Example

```bash
apr tokenize "Hello world" --tokenizer qwen2.5-coder-1.5b
```

## What this does

`apr tokenize` trains BPE tokenizers from a JSONL corpus per
`contracts/tokenizer-bpe-v1.yaml`, imports HuggingFace `tokenizer.json` files
into the project's two-file `vocab.json` + `merges.txt` layout, and pretokenizes
corpora into `.bin` shards ready for training. It also serves as a one-shot
"tokenize this text" inspector — handy for verifying that a prompt round-trips
through the model's tokenizer.

## Key subcommands

| Subcommand | What it does | Example |
|-----------|-------------|---------|
| `tokenize plan` | Validate + estimate training time | `apr tokenize plan corpus.jsonl` |
| `tokenize apply` | Train BPE on the corpus | `apr tokenize apply corpus.jsonl` |
| `tokenize train` | Direct BPE train per MODEL-2 contract | `apr tokenize train corpus.jsonl` |
| `tokenize import-hf` | Import a HF `tokenizer.json` | `apr tokenize import-hf tokenizer.json` |
| `tokenize encode-corpus` | Pretokenize JSONL into `.bin` shards | `apr tokenize encode-corpus train.jsonl` |
| `tokenize repair-manifest` | Reconstruct manifest from shard files | `apr tokenize repair-manifest dir/` |

## Common workflows

**Train a 50k-vocab BPE on a code corpus.**

```bash
apr tokenize plan ./corpus/python-csn.jsonl --vocab-size 50257
apr tokenize apply ./corpus/python-csn.jsonl --vocab-size 50257 -o ./tok/python-50k/
```

**Pretokenize for pretrain consumption.**

```bash
apr tokenize encode-corpus ./corpus/train.jsonl \
    --tokenizer ./tok/python-50k/ \
    -o ./shards/python-train/
ls ./shards/python-train/        # shard-0000.bin, shard-0001.bin, manifest.json
```

## Troubleshooting

- **Vocab smaller than requested** — corpus too small; BPE merges run out before
  reaching the budget. Use a larger corpus or accept the resulting smaller
  vocab.
- **`import-hf` rejects `tokenizer.json`** — the layout must be
  HuggingFace-standard. Some old tokenizers use a different schema; convert
  via `transformers.AutoTokenizer.save_pretrained` first.
- **`.bin` shard size doesn't match `manifest.json`** — interrupted encode.
  Re-run `encode-corpus` or use `repair-manifest` to rebuild the index from
  existing shards.

## See also

- Source: [`crates/apr-cli/src/commands/tokenize.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/tokenize.rs)
- Contract: [`contracts/apr-page-cli-tokenize-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-tokenize-v1.yaml)

