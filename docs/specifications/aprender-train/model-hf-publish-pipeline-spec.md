# Specification: HuggingFace Model Publish Pipeline

**Document ID:** SPEC-HF-PUBLISH-001
**Version:** 1.0.0
**Status:** Live — canonical workflow for shipping a Sovereign AI Stack model to HF Hub
**Parent:** [Ship Two Models Index](./ship-two-models-spec.md)
**First applied:** v0.34.0 / paiml/albor-370m-v1 (2026-05-18)

## Purpose

This spec defines the canonical workflow for publishing a model trained with the pure-Rust Sovereign AI Stack to HuggingFace Hub such that **all three usage paths work**:

1. **`apr run <repo>`** — native Rust stack via `aprender-serve` (realizar) inference engine
2. **`AutoModelForCausalLM.from_pretrained(<repo>)`** — HuggingFace Transformers cross-stack
3. **`llama-cli -m <repo>/...-q4k.gguf`** — llama.cpp / GGUF ecosystem

Each path imposes its own file + metadata requirements. Missing any one of them produces a published artifact that loads in one ecosystem but breaks in another. The pipeline below ensures all three.

## Required artifacts (12 files minimum)

Every shipped model MUST publish the following files to the HF repo root. The number in `()` is the count observed for `paiml/albor-370m-v1` (Qwen2 0.5B-architecture, 494M unique params).

| File | Size class | Format | Required by | Source |
|---|---|---|---|---|
| `README.md` | small | markdown + YAML front-matter | HF page render + tag inference | hand-authored model card |
| `LICENSE` | small | text | HF page badge + Apache compliance | upstream / `apr publish` |
| `config.json` | tiny | JSON | HF Transformers `AutoConfig`, llama.cpp arch detection | `apr export --format safetensors` (auto-generated) |
| `generation_config.json` | tiny | JSON | HF Transformers `model.generate()` defaults | upstream base or hand-authored |
| `tokenizer.json` | medium (~7MB) | JSON (HF fast tokenizer) | HF Transformers `AutoTokenizer` | upstream base (when fine-tuning from one) |
| `tokenizer_config.json` | tiny | JSON | `AutoTokenizer` metadata (chat_template, model_max_length, special tokens) | upstream base or hand-authored |
| `vocab.json` | small (~3MB) | JSON | legacy BPE tokenizer path + `apr tokenize` | upstream base / `apr stamp --tokenizer` source |
| `merges.txt` | small (~2MB) | text | legacy BPE merges | upstream base / `apr stamp --tokenizer` source |
| `model.safetensors` | LFS | SafeTensors | HF Transformers `AutoModelForCausalLM.from_pretrained` (canonical filename) | `apr export --format safetensors` + rename / LFS alias |
| `<name>.safetensors` | LFS | SafeTensors | descriptive named export referenced in README | `apr export --format safetensors` |
| `<name>.apr` | LFS | APR v2 | `apr run <repo>` (Rust stack canonical) | `apr stamp` |
| `<name>-q4k.gguf` | LFS | GGUF Q4_K | `llama-cli`, `ollama`, third-party GGUF tools | `apr export --format gguf --quantize int4` |

`<name>` is the model slug minus the org (e.g., `albor-370m-v1` for `paiml/albor-370m-v1`).

The `model.safetensors` alias must point at the **same LFS OID** as `<name>.safetensors` (HF deduplicates the blob storage — no actual disk cost). Achieve this by emitting a second NDJSON `lfsFile` commit op with the same `oid` and `size`. See "Publishing the alias" below.

## YAML front-matter (in README.md)

The `README.md` MUST start with a YAML front-matter block. Minimum keys:

```yaml
---
library_name: aprender             # NEVER use "transformers" — be honest about the framework
license: apache-2.0                # SPDX identifier
pipeline_tag: text-generation      # required for HF inference widget routing
language:
- en
tags:
- code                             # at minimum, declare the model's primary domain
- code-generation
- <other relevant tags>
- aprender                         # always include the framework tag
- rust                             # always include — distinguishes from PyTorch models
- pytorch-free                     # always include — marketing the sovereign claim
- stack-existence-proof            # only for §88-class compute-bounded ships
- sovereign-ai
base_model: <upstream/model>       # when fine-tuning from a base
datasets:
- <publisher/dataset-1>
- <publisher/dataset-2>
metrics:
- val_loss                         # whatever you actually measured
model-index:                       # OMIT ENTIRELY if no metrics — never emit empty results
- name: <model-slug-matching-repo-name>
  results:                         # MANDATORY when model-index is present (HF rejects with HTTP 400 otherwise)
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: <descriptive-corpus-name>
      type: <category>
    metrics:
    - type: val_loss
      value: 4.6227
      name: Validation Cross-Entropy
    - type: val_perplexity
      value: 101.78
      name: Validation Perplexity
    - type: throughput_tps
      value: 315.6
      name: Inference Throughput (tok/s, RTX 4090)
---
```

**Critical rules from PMAT-690 P3-C-prep defect 5c (2026-05-17):**
- If you emit `model-index:` you MUST emit `results:` (HF rejects HTTP 400 `"model-index[0].results" is required`).
- `ModelCard::to_huggingface` in `aprender-core` skips the entire block when `metrics` is empty (correct behavior).
- `pipeline_tag` must be set for the inference widget to render the right input form.

## File-source workflow

For a model fine-tuned from an upstream base (e.g., `Qwen/Qwen2.5-Coder-0.5B-Instruct`):

```bash
# Pull upstream tokenizer + LICENSE + companion files
TOKENIZER=/tmp/upstream-tok
mkdir -p $TOKENIZER
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json LICENSE; do
  curl -sLo $TOKENIZER/$f "https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/resolve/main/$f"
done

# Stamp the .apr with embedded tokenizer + provenance metadata
apr stamp ep49.apr \
  --architecture qwen2 --hf-architecture Qwen2ForCausalLM --hf-model-type qwen2 \
  --license Apache-2.0 \
  --data-source "huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct + bigcode/the-stack-dedup + codeparrot/codeparrot-clean" \
  --data-license "Apache-2.0 / permissive-aggregate" \
  --tokenizer $TOKENIZER \
  -o staging/<name>.apr

# Export to SafeTensors (auto-generates config.json)
apr export staging/<name>.apr --format safetensors -o staging/<name>.safetensors

# Export to GGUF Q4_K (with K-divisibility fallback per PMAT-690 defect 2)
apr export staging/<name>.apr --format gguf --quantize int4 -o staging/<name>-q4k.gguf

# Drop the upstream tokenizer + companion files into the staging dir
cp $TOKENIZER/{tokenizer.json,tokenizer_config.json,vocab.json,merges.txt,generation_config.json,LICENSE} staging/

# Author the README (hand-crafted) in staging/README.md per the YAML schema above
```

For a model trained from scratch (no upstream base), substitute hand-authored equivalents for `tokenizer.json` / `generation_config.json`.

## Publish

```bash
apr publish staging/ <org>/<slug> \
  --license Apache-2.0 \
  --library-name aprender \
  --tags "<comma-separated tags matching README YAML>" \
  --message "<commit message>"
```

**Known limitation (P3-C-prep defect 6, FOLLOW-UP):** `apr publish`'s `find_model_files` currently only picks `.apr` / `.safetensors` / `.gguf` extensions plus an auto-generated README. The companion files (`config.json`, `vocab.json`, `merges.txt`, user-authored README, LICENSE, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`) must currently be uploaded via a separate API call. Until the file-selection defect is fixed, use the inline NDJSON commit pattern documented in "Manual companion-file upload" below.

## Publishing the `model.safetensors` alias

After `apr publish` lands `<name>.safetensors` with a known OID, add the `model.safetensors` alias in one NDJSON commit. This is required for `AutoModelForCausalLM.from_pretrained(<repo>)` to find the weights without an explicit `weights_file` argument.

```bash
OID=$(curl -s "https://huggingface.co/api/models/<repo>/tree/main" | \
      python3 -c "import json,sys; t=json.load(sys.stdin); print(next(s['lfs']['oid'] for s in t if s['path']=='<name>.safetensors'))")
SIZE=$(curl -s "https://huggingface.co/api/models/<repo>/tree/main" | \
       python3 -c "import json,sys; t=json.load(sys.stdin); print(next(s['size'] for s in t if s['path']=='<name>.safetensors'))")

printf '{"key":"header","value":{"summary":"add model.safetensors alias for HF Transformers","description":""}}\n{"key":"lfsFile","value":{"path":"model.safetensors","algo":"sha256","oid":"%s","size":%s}}\n' \
  "$OID" "$SIZE" > /tmp/alias.ndjson

curl -s -X POST "https://huggingface.co/api/models/<repo>/commit/main" \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/x-ndjson" \
  --data-binary @/tmp/alias.ndjson
```

LFS deduplication makes this free — the blob storage is shared between both filenames.

## Manual companion-file upload (until publish CLI is fixed)

Until P3-C-prep defect 6 lands (extending `find_model_files`), companion files go via direct NDJSON commits:

```bash
# Build one NDJSON payload with all small files
{
  echo '{"key":"header","value":{"summary":"add HF integration files","description":""}}'
  for f in LICENSE config.json generation_config.json tokenizer.json tokenizer_config.json README.md; do
    B64=$(base64 -w0 staging/$f)
    printf '{"key":"file","value":{"path":"%s","content":"%s","encoding":"base64"}}\n' "$f" "$B64"
  done
} > /tmp/companion.ndjson

curl -s -X POST "https://huggingface.co/api/models/<repo>/commit/main" \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/x-ndjson" \
  --data-binary @/tmp/companion.ndjson
```

`tokenizer.json` at ~7MB fits within HF's NDJSON commit payload size budget. Files above ~10MB should use the LFS batch path (see `apr publish` source after PMAT-690 defect 5a fix).

## End-to-end verification protocol

After publish completes, run **all three** verification paths. Any failure means the publish is broken even if the HF page renders.

### Path 1: Rust stack (canonical)

```bash
cargo install aprender                                    # latest from crates.io
apr pull <repo>
apr run <repo> "def fibonacci(n):" --max-tokens 16        # must produce text (gibberish OK for §88 ships)
apr inspect <repo>/<name>.apr | grep "HAS_VOCAB"          # must show HAS_VOCAB flag
```

### Path 2: HuggingFace Transformers

```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained('<repo>')
model = AutoModelForCausalLM.from_pretrained('<repo>')
print(tok.decode(model.generate(**tok('def fib(n):', return_tensors='pt'), max_new_tokens=16)[0]))
"
```

Must succeed without `OSError: <repo> does not appear to have a file named pytorch_model.bin or model.safetensors`.

### Path 3: llama.cpp / GGUF

```bash
huggingface-cli download <repo> <name>-q4k.gguf --local-dir /tmp/
llama-cli -m /tmp/<name>-q4k.gguf -p "def fib(n):" -n 16 --simple-io < /dev/null
```

Must load and generate without `gguf_init_from_file_impl: tensor 'X' has offset N, expected M` (PMAT-690 defect 3) or `tensor 'X' of type 12 (q4_K) has N elements per row, not a multiple of block size (256)` (PMAT-690 defect 2).

### HF page-render audit

```bash
curl -s "https://huggingface.co/api/models/<repo>" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert d['pipeline_tag'] == 'text-generation', 'missing pipeline_tag'
assert d['library_name'] == 'aprender', 'wrong library_name'
assert d['cardData']['license'] == 'apache-2.0', 'missing/wrong license'
assert d['cardData']['base_model'], 'missing base_model'
assert d['cardData']['datasets'], 'missing datasets'
sib = {s['rfilename'] for s in d['siblings']}
required = {'README.md', 'LICENSE', 'config.json', 'generation_config.json',
            'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
            'model.safetensors'}
missing = required - sib
assert not missing, f'missing files: {missing}'
print('✅ All integrations present')
"
```

## crates.io release cascade (when shipping a new aprender version alongside a model)

When a model publish also requires shipping new `aprender` crate versions (e.g., defect fixes that landed during the publish dry-run), follow this cascade. Order matters — each crate must be on crates.io before any dependent crate publishes.

**Tier 1 (leaves, no internal deps):**
`aprender-contracts-macros, aprender-quant, aprender-gemm-codegen, aprender-sparse, aprender-solve, aprender-rand, aprender-fft, aprender-image, aprender-tensor, aprender-cupti`

**Tier 2 (depend on Tier 1):**
`aprender-contracts, aprender-core, aprender-profile-core, aprender-graph`

**Tier 3:**
`aprender-profile` (depends on aprender-core)

**Tier 4:**
`aprender-gpu` (depends on aprender-profile via renacer alias)

**Tier 5:**
`aprender-cuda-edge, aprender-cgp` (both depend on aprender-gpu)

**Tier 6:**
`aprender-compute` (depends on gpu + cuda-edge + quant + sparse + solve + gemm-codegen)

**Tier 7 (GPU-tier consumers):**
`aprender-cbtop, aprender-ptx-debug, aprender-explain`

**Tier 8 (workspace consumers):**
`aprender-common, aprender-train-common, aprender-train, aprender-train-lora, aprender-train-distill, aprender-serve, aprender-mcp, aprender-data, aprender-orchestrate`

**Tier 9 (present family, in order):**
`aprender-present-core, aprender-present-layout, aprender-present-yaml, aprender-present-widgets, aprender-present-terminal`

**Tier 10 (CLI consumer):**
`apr-cli` (needs all the above)

**Tier 11 (root facade):**
`aprender` (depends on apr-cli)

**Tier 12 (root-aprender consumers, must wait for root):**
`aprender-tsp, aprender-monte-carlo, aprender-shell`

**Tier 13 (satellites, any order):**
`aprender-test-{derive,lib,js-gen,cli,showcase}, aprender-zram-{core,adaptive,generator,cli}, aprender-zram, aprender-present-{test-macros,test,lib,cli}, aprender-db, aprender-rag, aprender-viz, aprender-registry, aprender-distribute, aprender-simulate, aprender-verify, aprender-verify-ml, aprender-train-{shell,inspect,bench,wasm}, aprender-contracts-cli`

Use the cascade script at `scripts/cascade-publish.sh` (committed as part of SPEC-HF-PUBLISH-001 v1.0.0) to walk this list. Verify each tier reached crates.io before starting the next.

**Per-crate verification:**
```bash
curl -s "https://crates.io/api/v1/crates/<crate>" | jq -r '.crate.max_version'
# Must equal the target version (e.g., 0.34.0)
```

**End-of-cascade verification:**
```bash
cargo install aprender --force
~/.cargo/bin/apr --version
# Must report the target version
```

## HF API gotchas (load-bearing rules)

1. **NDJSON, not JSON, for commits.** HF's commit endpoint MUST receive `application/x-ndjson` with two lines: `{key:"header"}` then `{key:"file"}` (small files) or `{key:"lfsFile"}` (LFS files). JSON `addOrUpdate` ops return 200 + `success: true` but silently drop the file. Memory rule: `feedback_hf_commit_ndjson_load_bearing.md` (2026-04-18). Fixed in v0.34.0 / `apr publish` per PMAT-690 defect 5b.
2. **LFS batch API for 5MB–5GB files.** When HF's `/preupload/main` returns `uploadMode: "lfs"` with no inline `uploadUrl` or `chunkUrls`, the client MUST call `POST /{repo}.git/info/lfs/objects/batch` to obtain a presigned S3 URL, then PUT the blob to that URL. Skipping this step leaves orphaned LFS pointers (commit succeeds, repo shows only `.gitattributes`). Fixed in v0.34.0 per PMAT-690 defect 5a.
3. **Xet for files > 5 GiB.** Files exceeding the HF Xet threshold (`HF_XET_THRESHOLD_BYTES = 5 * 1024 * 1024 * 1024`) must use the Xet content-addressable protocol, not LFS batch. `apr publish` dispatches via `should_use_xet(file_size_bytes)` in `aprender-core/src/hf_hub/xet.rs`.
4. **Empty model-index rejected.** A YAML `model-index:` block with a `name:` but no `results:` triggers HTTP 400 `"model-index[0].results" is required`. Either populate `results` with at least one metric, or omit the block entirely. Fixed in v0.34.0 per PMAT-690 defect 5c.
5. **GGUF Q4_K requires K % 256 == 0.** Tensors where the inner matmul dim (K = ne[0]) is not divisible by 256 must fall back to F32; otherwise `llama-cli` rejects with `tensor 'X' of type 12 (q4_K) has N elements per row, not a multiple of block size (256)`. Notable: Qwen2 0.5B (`hidden=896`) fails this on 7 tensors per layer; 1.5B (`hidden=1536`) and 7B (`hidden=3584`) are unaffected. Fixed in v0.34.0 per PMAT-690 defect 2.
6. **GGUF Q4_K shape must be APR-native.** Pass `[rows=out, cols=in=K]` to `quantize_q4_k_matrix`, not the swapped `[K, out]`. The swap pads the wrong axis and produces transposed bytes with the wrong byte count (350,208-byte excess on Qwen2 0.5B ffn_down, producing `gguf_init_from_file_impl: tensor 'X' has offset N, expected M`). Fixed in v0.34.0 per PMAT-690 defect 3.

## Reference implementations

| Concern | File |
|---|---|
| `apr stamp --tokenizer` | `crates/aprender-core/src/format/v2/stamp.rs`, `crates/apr-cli/src/commands/stamp.rs` |
| GGUF Q4_K divisibility + shape | `crates/aprender-core/src/format/converter/gguf_export_config.rs`, `fusion.rs`, `export_include_01.rs` |
| HF NDJSON LFS commit | `crates/aprender-core/src/hf_hub/upload.rs::upload_via_lfs`, `commit_lfs_pointer` |
| HF NDJSON small-file commit | `crates/aprender-core/src/hf_hub/client_impl.rs::upload_direct` |
| HF LFS batch upload | `crates/aprender-core/src/hf_hub/upload.rs::upload_via_lfs_batch` |
| HF Xet upload (>5 GiB) | `crates/aprender-core/src/hf_hub/xet.rs` |
| Model card YAML generation | `crates/aprender-core/src/format/model_card.rs::to_huggingface` |

## Lineage / first applied

- **paiml/albor-370m-v1** (MODEL-2 §88 stack-existence-proof, 2026-05-18) — first model shipped through this pipeline. 13 files. All three usage paths verified. Triggered the PMAT-690 P3-C-prep defect cascade (1+2+3+5a+5b+5c) that this spec encodes.
- **paiml/qwen2.5-coder-7b-apache-q4k-v1** (MODEL-1 teacher, 2026-04-18) — published BEFORE this spec was authored. Per `feedback_post_publish_qa_required.md`, that publish predates the SPEC-HF-PUBLISH-001 protocol; future MODEL-1 republishes should follow this spec.

## Changelog

- **1.0.0 (2026-05-18)** — Initial publish, derived from the paiml/albor-370m-v1 ship + PMAT-690 P3-C-prep defect cascade (defects 1, 2, 3, 5a, 5b, 5c).
