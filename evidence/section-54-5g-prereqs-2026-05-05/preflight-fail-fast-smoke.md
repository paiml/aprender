# Live preflight smoke: Qwen --init + legacy 50257-vocab tokenizer fail-fasts

**Date:** 2026-05-05
**Branch:** main (commit 92c7e237b — includes §53 + §50.4 step 5f.4 wireup)
**Binary:** `/mnt/nvme-raid0/targets/aprender/release/apr` rebuilt at 2026-05-05T04:31Z (CPU-only `cargo build --release -p apr-cli`).
**Spec reference:** SPEC-SHIP-TWO-001 §54 (this evidence informs the §54 amendment).
**Falsifiers exercised:** FALSIFY-APR-PRETRAIN-ARCH-005, FALSIFY-APR-PRETRAIN-ARCH-006, GATE-ARCH-370M-011 (INV-ARCH-370M-006).

## Command dispatched

```bash
/mnt/nvme-raid0/targets/aprender/release/apr pretrain \
  --dataset /mnt/nvme-raid0/data/codeparrot-python-permissive-shards \
  --tokenizer /mnt/nvme-raid0/models/model-2-tokenizer-v1 \
  --run-dir /tmp/apr-pretrain-5g-smoke/run-1 \
  --init /mnt/nvme-raid0/models/qwen2.5-coder-0.5b-instruct-fp16.apr \
  --mode finetune \
  --num-steps 10 \
  --device cpu \
  --seed 42 \
  --vocab-size 151936
```

## Inputs

- **Init APR:** `qwen2.5-coder-0.5b-instruct-fp16.apr` (942.31 MiB, APR v2, 290 tensors, checksum VALID).
  - Architecture: hidden=896, layers=24, heads=14, kv_heads=2 (GQA-7:1), ffn=4864, vocab=151936, model_type=qwen2.
  - Matches `TransformerConfig::qwen2_0_5b()` shape byte-for-byte.
- **Tokenizer dir:** `/mnt/nvme-raid0/models/model-2-tokenizer-v1` (vocab=50257, GPT-2-style BPE).
- **Corpus shards:** `/mnt/nvme-raid0/data/codeparrot-python-permissive-shards` (565.6M tokens, tokenized with the 50257-vocab tokenizer per `manifest.json:vocab_size`).

## Observed output

```
=== Configuration ===
    Dataset: /mnt/nvme-raid0/data/codeparrot-python-permissive-shards
    Tokenizer: /mnt/nvme-raid0/models/model-2-tokenizer-v1
    Run dir: /tmp/apr-pretrain-5g-smoke/run-1
    LR max: 5.00e-5
    Total steps: 10
    Warmup steps: 100
    Batch × seq: 16 × 1024
    Steps / epoch: 100
    Seed: 42
    Target val_loss: 2.20

    Device: cpu

error: Validation failed: GATE-ARCH-370M-011 (INV-ARCH-370M-006) violated:
    tokenizer vocab_size (50257) != model vocab_size (151936).
    See contracts/model-families/llama-370m-sovereign-v1.yaml and
    contracts/tokenizer-bpe-v1.yaml — retrain the tokenizer or amend
    both contracts in lockstep before resuming pretraining.
```

## What this proves

1. **The polymorphic preflight wired in PR #1476 + #1494 fires correctly in the user-facing CLI.** The `--init` path's extracted `vocab_size = 151936` is the gate's `target_vocab` (not the legacy hardcoded `Llama370MConfig::VOCAB_SIZE = 50257`). When the tokenizer's vocab.json contains 50257 entries, the preflight FAILS FAST with a clear contract-cited error.

2. **FALSIFY-APR-PRETRAIN-ARCH-006 is empirically reachable:** "Qwen tokenizer with --init absent fails pre-flight" was the unit-test framing. The dual case (legacy tokenizer with Qwen --init) is what an operator actually hits today, and it ALSO fails fast — proving the symmetry the contract pinned.

3. **5g LIVE has prerequisites the original §50 8-step decomposition did not enumerate.** A Qwen-vocab tokenizer dir + Qwen-tokenized corpus must exist before the polymorphic preflight passes for the Qwen --init path. The §50.4 cascade's CLI wireup is correct but a Qwen-tokenized corpus does not currently exist on this host.

## Files referenced

- `crates/apr-cli/src/commands/pretrain.rs::preflight_tokenizer_vocab_matches_target` — gate site.
- `contracts/apr-pretrain-arch-polymorphic-v1.yaml` v1.2.0 FUNCTIONAL — the polymorphic preflight invariant.
- `contracts/model-families/llama-370m-sovereign-v1.yaml`, `contracts/tokenizer-bpe-v1.yaml` — the cited cross-references in the error message.

## What this does NOT prove (still open)

- The Qwen-vocab fine-tune does NOT yet have a corpus + tokenizer pair that allows the preflight to pass. Re-tokenizing 565.6M tokens with the Qwen vocab takes multi-hour wall (the original codeparrot tokenization run took ~10 hours per `manifest.json:elapsed_seconds = 35979.9 = 9.99h`).
- val_loss < 9.38 evidence (the actual MODEL-2 ship-% gate) is NOT yet captured. Step 5g.0..5g.3 (re-scoped per §54) must complete before this measurement is taken.
