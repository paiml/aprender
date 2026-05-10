# Live extraction smoke: Qwen2.5-Coder-0.5B-Instruct tokenizer.json → aprender format

**Date:** 2026-05-05
**Spec ref:** SPEC-SHIP-TWO-001 §50.4 step 5g.0 (this PR).
**Contract:** `contracts/apr-cli-tokenize-import-hf-v1.yaml` v1.0.0 PARTIAL_ALGORITHM_LEVEL.
**Falsifiers exercised:** FALSIFY-TOK-IMPORT-HF-001..005 (all 5 PASS in CI; FALSIFY-002..005 also PASS in this LIVE smoke).

## Command

```bash
apr tokenize import-hf \
  --input ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/.../tokenizer.json \
  --output /tmp/qwen-0.5b-tokenizer-extracted \
  --json
```

## Output

```json
{
  "added_tokens_count": 22,
  "bpe_vocab_count": 151643,
  "effective_vocab_count": 151643,
  "extraction_timestamp_utc": "2026-05-05T03:10:46.343818077+00:00",
  "include_added_tokens": false,
  "merges_count": 151387,
  "model_type": "BPE",
  "schema": "apr-cli-tokenize-import-hf-v1",
  "source": "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/ea3f2471cf1b1f0db85067f1ef93848e38e88c25/tokenizer.json",
  "source_sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
}
```

## Files written

```
$ ls -la /tmp/qwen-0.5b-tokenizer-extracted/
manifest.json     534 bytes
merges.txt   1671853 bytes (1.6 MiB)
vocab.json   3383406 bytes (3.2 MiB)
```

## Sample of merges.txt (first 16 lines)

```
#version: 0.2
Ġ Ġ
ĠĠ ĠĠ
i n
Ġ t
ĠĠĠĠ ĠĠĠĠ
e r
ĠĠ Ġ
o n
Ġ a
r e
a t
s t
e n
o r
Ġt h
```

This matches the GPT-2 BPE merges.txt convention exactly (#version header + space-separated merge per line in original order). Confirms FALSIFY-TOK-IMPORT-HF-004 LIVE.

## What this proves

- **FALSIFY-TOK-IMPORT-HF-002 LIVE**: BPE input produces non-empty vocab.json + merges.txt (151643 entries / 151387 lines).
- **FALSIFY-TOK-IMPORT-HF-003 LIVE**: vocab.json entry count == |tokenizer.json:model.vocab| (151643 = 151643).
- **FALSIFY-TOK-IMPORT-HF-004 LIVE**: merges.txt has one merge per line in original order, GPT-2 format.
- **Provenance**: source_sha256 + extraction_timestamp_utc captured for audit.
- **Non-BPE rejection**: FALSIFY-TOK-IMPORT-HF-005 covered by unit test (Unigram input rejects fail-fast).

## What this does NOT yet prove

- **`--include-added-tokens` mode** produces 151665 effective_vocab (151643 + 22 added). This DOES NOT match Qwen2.5's declared `vocab_size = 151936` because the gap (271 entries) is **reserved/special slots** that aren't materialized in tokenizer.json. The polymorphic preflight in `apr-pretrain-arch-polymorphic-v1` currently fails on this gap with `tokenizer vocab_size (151643/151665) != model vocab_size (151936)`. **This is a §55 follow-up finding** — either the preflight relaxes to `<=` semantics for reserved-slot tolerance, OR `import-hf` gains a `--pad-to <N>` flag that synthesizes reserved-token placeholders.

- **5g.1 (Qwen-tokenized corpus)** still pending. The import-hf step itself produces a syntactically-valid aprender tokenizer dir; downstream consumption (`apr tokenize encode-corpus --tokenizer <DIR>`) is not yet smoke-tested in this PR.

## Files referenced

- `crates/apr-cli/src/commands/tokenize.rs::run_import_hf` (this PR).
- `crates/apr-cli/src/tokenize_commands.rs::TokenizeCommands::ImportHf` (this PR).
- `crates/apr-cli/src/dispatch_analysis.rs` (this PR — dispatch wireup).
- `contracts/apr-cli-tokenize-import-hf-v1.yaml` (this PR).
