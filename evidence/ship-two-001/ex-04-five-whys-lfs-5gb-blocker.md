# SHIP-TWO-001 EX-04 — Five Whys: 5 GB HF Hub upload blocker

**Date:** 2026-04-18
**Trigger:** Real upload attempt (HF_TOKEN available) failed after all pre-flight
gates passed.
**Evidence:** `evidence/ship-two-001/ex-04-upload-v3.log` lines 1–60

## Observation

The canonical apr binary (ec60b5c9e, `--features cuda`) executed
`ex-04-upload-hf.sh` with live HF_TOKEN. All three pre-flight gates passed:

```
PRE-FLIGHT: FALSIFY-PM-001..007 on all three formats
  apr            PASS
  safetensors    PASS
  gguf           PASS
```

The first `apr publish` invocation then aborted with:

```
[LFS] ERROR: File qwen2.5-coder-7b-instruct-q4k.apr (8.0 GB)
      exceeds 5GB HuggingFace Hub limit for HTTP API uploads
[LFS] Files > 5GB require HuggingFace's multipart transfer agent.
```

The HF preupload API returned `200 OK` with `"uploadMode": "lfs"` but
**both** `upload_url` and `chunk_urls` fields were empty, causing
`reject_oversized_file` (`crates/aprender-core/src/hf_hub/upload.rs:283`)
to fire and `apr publish` to return a `NetworkError` non-zero exit.

## Five Whys

1. **Why did EX-04 fail?**
   `apr publish` rejected the 8.0 GiB teacher `.apr` because HF Hub's
   preupload response did not include the presigned multipart URLs
   needed for files > 5 GiB.

2. **Why no multipart URLs in the response?**
   HF Hub's `preupload/main` API returns pre-signed S3 chunk URLs only
   for files below the 5 GiB soft threshold. Beyond that, HF expects the
   client to negotiate upload via the **LFS batch API**
   (`POST /{repo}/.git/info/lfs/objects/batch` — git-lfs protocol) or
   the HF **custom transfer agent** (`hf_transfer`), neither of which
   `apr publish` currently implements.

3. **Why does `apr publish`'s error message recommend a non-existent fix?**
   The error at line 289 of `upload.rs` recommends
   `apr export --max-shard-size 4GB`, but `apr export --help` shows no
   such flag. The recommendation was written speculatively in anticipation
   of sharding support that was never implemented.

4. **Why didn't the pre-flight gates catch this before any network I/O?**
   PM-001..009 validate **manifest ↔ artifact binary layer** agreement
   (sha256, dtype, quantization, magic). They do not probe HF Hub's
   capabilities against the staged file sizes. The 5 GiB limit is a
   **destination-side** property that pre-flight gates (by design)
   never tested.

5. **Why is the teacher ~8 GiB in the first place?**
   The teacher is Qwen2.5-Coder-7B-Instruct quantized to Q4_K
   (`qwen2.5-coder-7b-instruct-q4k.*`). At Q4_K on a 7 B model, every
   format lands in the 8–15 GiB range:
   - `.apr`        8.0 GiB
   - `.gguf`       8.0 GiB
   - `.safetensors` 15.2 GiB (fp16 re-expansion)
   All three single-file uploads exceed HF's 5 GiB threshold.

## Root cause (from Why #2)

`apr publish` cannot complete single-file uploads > 5 GiB to HF Hub
regardless of filename or format. This is an architectural gap in
`crates/aprender-core/src/hf_hub/upload.rs`: the code path for
`uploadMode: lfs` without presigned URLs invokes
`reject_oversized_file` (a deliberate early-exit) because no fallback
transfer mechanism is wired in.

## What the pre-flight gates **did** catch (and still valuable)

Before this run, the script ran with `--commit-message` (unknown flag
to `apr publish`, which uses `--message`). That was the first real bug
surfaced by live HF_TOKEN — a trivial script fix
(`scripts/ship-two-001/ex-04-upload-hf.sh:119`).

## Ship decision (options)

| Option | Pros | Cons | Cost |
|--------|------|------|------|
| A) Implement sharded `apr export --max-shard-size` | Dogfood path; HF-native safetensors sharding convention | Only helps `.safetensors` (index.json based); `.apr` and `.gguf` need their own sharding scheme | 3–5 days |
| B) Add LFS batch-API / git-lfs subprocess to `apr publish` | Supports files up to HF's real limit (~50 GiB); dogfood | Pulls git-lfs as external dependency OR reimplements LFS protocol in Rust | 1–2 weeks |
| C) Use self-hosted S3 bucket only (skip HF Hub for >5 GiB formats) | Decouples SHIP-TWO-001 from HF limits; aligns with Sovereign AI Stack | Loses HF model page discovery; breaks AC-SHIP1-006 (`apr pull` from HF) | 1 day |
| D) Reduce teacher footprint | Keeps single-file path | Q4_K is already the smallest meaningful precision for 7 B; would require a smaller parent model | Respec |
| E) Ship the teacher `.apr` only (drop .safetensors/.gguf for now) | Fastest ship; 8 GiB still blocks | Doesn't actually fix the blocker | — |

None of the options is a trivial change. **Ship is blocked until
operator makes architectural call.**

## Recommended path (subject to user decision)

**Option A + C combined** —
1. Add `--max-shard-size 4G` to `apr export` (for .safetensors; leverages
   HF's native sharding index convention).
2. For `.apr` and `.gguf`, document in the spec that single-file
   teachers > 5 GiB are currently published only to the self-hosted S3
   bucket. `.safetensors` is published to HF Hub in sharded form.
3. Both sets of URLs appear in the per-format manifest (already supported
   — `artifact_url` and `artifact_url_mirror`).

This preserves the three-format promise while respecting the HF 5 GiB
limit without waiting for full LFS batch-API support.

## Immediate follow-up tasks

- Commit the `--commit-message` → `--message` script fix independently
  (it's a real bug that would have blocked upload even at 4 GiB).
- Draft `contracts/apr-cli-publish-lfs-multipart-v1.yaml` with
  `FALSIFY-PUB-LFS-001..005` (file-size dispatch, preupload-URL gate,
  sharded-safetensors happy path, LFS batch API retry, sovereign S3
  mirror path).
- Amend SHIP-TWO-001 spec §12.7 with this discovery as v2.8.0.

## Related

- `crates/aprender-core/src/hf_hub/upload.rs` — `reject_oversized_file`
- `crates/apr-cli/src/commands/publish.rs` — callsite
- HF docs: <https://huggingface.co/docs/hub/upload> (curl / git-lfs /
  `hf_transfer`)
- Existing precedent: `safetensors` crate and `transformers` library
  ship 70 B+ models as `model-00001-of-00015.safetensors` shards.
