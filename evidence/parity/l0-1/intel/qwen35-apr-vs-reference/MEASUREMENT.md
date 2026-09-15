# Qwen3.5-0.8B Q4_K_M — apr CPU forward (#3114) vs llama.cpp d1d3c3396 raw logits, all 78 positions (PMAT-3091)

First measurement of the 0.68 deliverable (#3091) against the raw-logit reference
(`../qwen35-cpu-reference/REFERENCE-RAW.md`, PMAT-3303). **Numbers only. There is no verdict
and no threshold.** The Qwen3.5 threshold row stays `[U]` and fail-closed, and the known-bad
half of its basis (B5) is still undecided.

## Tree (every measurement below)

| side | tree |
|------|------|
| evidence worktree (meas-3091), base of this commit | `9dd532fa41f971641f6d54b239feced673732c6e` (PMAT-3303-qwen35-raw-logits) |
| subject: draft PR #3114, branch `PMAT-1098-qwen35-cpu`, built detached and read-only | `5b055c0ca994ee3449f19e9641204789dcacf4d8` |
| reference: llama.cpp | `d1d3c3396` |
| model | `Qwen3.5-0.8B-Q4_K_M.gguf`, sha256 `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517` (same sha on intel and on lambda) |

## Inputs

- Reference: `intel:~/parity-ref/qwen35-0.8b-q4km-d1d3c3396/qwen35-0.8b-q4km-d1d3c3396.rawlogits.bin`,
  sha256 `1d5eaf129545aa6087cd6c49792f9774b61caa874fccefdd79e6164afa15bb93`, 77,476,172 bytes.
  Header read per `../qwen35-cpu-reference/producer/apr_raw_logits.cpp`: magic `APRRAWLG`,
  version 1, n_pos 78, n_vocab 248320. That is 20 + 4·78 + 4·78·248320 = 77,476,172 bytes, exact.
  Produced on intel with `-ngl 0 -t 8 -c 78 -b 78`, which is one batched `llama_decode` over all 78 positions.
- Token ids: `prompt_token_ids.txt`, the 78 ids from the brief, verbatim. The comparator checks
  them against the reference header's `token_ids[78]` and the subject's; all three are equal
  (the comparator exits 2 otherwise).
- **Tokenization: ids-direct.** apr's tokenizer is never called.

## Subject: how apr produces per-position logits

The apr CLI has no switch that dumps per-position logits for a Qwen3.5 GGUF from given ids.
`apr run` / `apr chat` → `realizar::gguf::forward_qwen35::run_qwen35_generate`
(#3114: `crates/apr-cli/src/commands/chat_generate_session_02.rs:297`,
`crates/aprender-serve/src/infer/inference_result.rs:248`) prefills with the loop
`for (pos, &token) in input_tokens.iter().enumerate() { logits = qwen.forward_single_qwen35(token, &mut state, pos)?; }`
and keeps only the last row. The same loop drives #3114's own `examples/qwen35_parity.rs`.

`qwen35_raw_logits.rs` (committed here) runs **that same loop** over the same load path
(`MappedGGUFModel::from_path` → `Qwen35Model::create_base_model` → `from_model_and_layers` →
`new_state`). It keeps the row from EVERY position and writes it in the reference's
APRRAWLG layout. It was copied, uncommitted, into
`crates/aprender-serve/examples/` of the #3114 worktree and built there. Nothing was committed
to or pushed on the #3114 branch.

```bash
# build (lambda, load 5.72 before the build; only the aprender-serve example is built)
cp evidence/parity/l0-1/intel/qwen35-apr-vs-reference/qwen35_raw_logits.rs \
   /mnt/nvme-raid0/agent-wt/apr-3114/crates/aprender-serve/examples/
cd /mnt/nvme-raid0/agent-wt/apr-3114
CARGO_BUILD_JOBS=16 CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/apr-3114-meas \
  timeout 3000 ~/.cargo/bin/cargo build --release -p aprender-serve --example qwen35_raw_logits < /dev/null
# rc 0, rustc 1.93.0 (254b59607 2026-01-19), default features, no target-cpu flags (#3114 tree has no .cargo/config.toml)
# binary sha256 ad4397d5efcc8aa64a7b4e15303b32e5b51de3180c2da80cbf247cdfe7253d8e

# run (identical command on both hosts)
timeout 1800 ./qwen35_raw_logits ~/models/Qwen3.5-0.8B-Q4_K_M.gguf prompt_token_ids.txt apr-<host>-run<i>.bin < /dev/null
```

| host | CPU | run | rc | wall | CPU% | load (1/5/15) before → after | output sha256 |
|------|-----|-----|----|------|------|------------------------------|---------------|
| intel (`mac-server`) | Xeon W-3245, 32 threads, avx512f | 1 | 0 | 16.84 s | 632% | 34.79/47.27/56.95 → 43.27/48.57/57.22 | `f6f792649014dcef6eec1afaf96852a293cc95c6cfceb1c88d62e16ee36c0252` |
| intel | same | 2 | 0 | 16.43 s | 644% | 43.27/48.57/57.22 → 46.25/48.98/57.22 | `f6f79264…0252` (identical) |
| lambda | Threadripper 7960X, 48 threads | 1 | 0 | 5.09 s | 1950% | 8.69/5.07/4.99 → 8.32/5.05/4.98 | `f6f79264…0252` (identical) |
| lambda | same | 2 | 0 | 5.15 s | 1933% | 8.32/5.05/4.98 → 7.50/4.98/4.96 | `f6f79264…0252` (identical) |

On both hosts the subject is deterministic, n = 2 per host, and **byte-identical across hosts**
(n = 4 in total, one sha). The subject bytes do not depend on which of these two CPUs ran them. The
comparison below uses the **intel** run, the same host that produced the reference, so there is
**no cross-host confound**.

Saved subject logits (not committed; 77 MB):
`intel:~/parity-ref/apr-3091-subject/apr-intel-run1.bin` (sha `f6f79264…0252`).

## Comparison

```bash
python3 evidence/parity/l0-1/intel/qwen35-apr-vs-reference/compare_raw_logits.py \
  <reference .rawlogits.bin> <apr-intel-run1.bin> --json comparison.json > comparison.tsv
```

The comparator runs in float64. It computes the cosine over the full 248,320-entry vectors, the
argmax of each vector, and the max |ref − apr|. It also reports near-tie context: the
reference's top-1 − top-2 logit gap, apr's rank of the reference argmax, and apr's logit gap
from its own argmax to the reference argmax.

It was run twice. The two `comparison.tsv` outputs are byte-identical, sha256
`40c81bd72a58ce03ade4c316955b48538a00e3b478cf6f116816979c132d1d15`:
- on lambda (numpy 2.3.5), reading the reference copied from intel (sha re-verified as `1d5eaf12…bb93`) and the intel subject file copied back (sha re-verified);
- on intel (numpy 2.2.6), reading both files where they sit.

`comparison.json` sha256 `fb55cf746588c1a3d6a427a1ee15ffc5a5bc331fb72c6391aaf0f8da1f523678`.

As a sanity row, the comparator was run on the reference against itself: rc 0, min cosine 1.0,
0 argmax mismatches, max |diff| 0.0.

### Summary (n = 78 positions, n_vocab = 248320, host intel for both sides)

| stat | value |
|------|-------|
| min cosine | **0.995905**, at position **4** (token 33075) |
| mean cosine | 0.998989 |
| median cosine | 0.999175 |
| positions with cosine < 0.98 | **0**, none |
| next-lowest cosines | pos 28 0.996796 · pos 22 0.996803 · pos 27 0.997801 · pos 37 0.997945 |
| argmax mismatches | **4 of 78**, at positions **2, 28, 36, 73** |
| max \|ref − apr\| over all 78×248320 logits | **1.342222**, at position 2 |

In each of the 4 argmax mismatches, apr ranks the reference's top-1 token **2nd**:

| pos | ref argmax | apr argmax | ref top1−top2 gap | apr gap (apr top1 − apr logit of ref argmax) | cosine |
|-----|-----------|-----------|-------------------|----------------------------------------------|--------|
| 2  | 627  | 37550 | 0.432076 | 0.197060 | 0.998216 |
| 28 | 795  | 28758 | 0.632557 | 0.216549 | 0.996796 |
| 36 | 5331 | 57695 | 0.013315 | 0.075449 | 0.998697 |
| 73 | 13   | 314   | 0.109070 | 0.119488 | 0.999186 |

### Context only, not a judgement

`evidence/parity/thresholds.yaml` records a dense-model basis with min cosine over ≥ 64
positions on the same 78-token corpus prompt, n = 5:
known-good qwen2.5-coder-7b Q4_K_M, apr CUDA vs apr CPU, @lambda **0.998607**; known-bad
qwen2.5-coder-1.5b @lambda **0.950827**. That basis measures a different pair (apr GPU against
apr CPU, dense architecture) with a different producer. It sits here as a scale reference and
nothing more. It sets no Qwen3.5 threshold.

## Confounds (named)

1. **Different backends, by design.** The reference is llama.cpp (ggml CPU kernels, `-t 8`). The
   subject is realizar's Qwen3.5 forward in its own Q4_K/Q6_K dequant+matmul and DeltaNet
   recurrence, using the default thread count. Some divergence at float32 is expected from kernel
   accumulation order alone. This measurement cannot split kernel noise from a forward defect.
2. **Batched vs sequential prefill.** The reference used one `llama_decode` over all 78 positions
   (`-b 78`). How llama.cpp evaluates the gated-DeltaNet layers over a 78-token ubatch was not read, so whether that differs from a per-token step is `[U]`. The subject calls
   `forward_single_qwen35` 78 times and takes the one-step recurrence. If the paths do differ they are meant to be mathematically
   equal and numerically different paths. `[U]`: llama.cpp run token by token (`-b 1 -ub 1`)
   through the same producer would isolate this:
   `apr_raw_logits -m <model> -f prompt.txt -ngl 0 -t 8 -c 78 -b 1 -ub 1 --raw-out r-b1.bin`
   then `compare_raw_logits.py r-b1.bin apr-intel-run1.bin`.
3. **Thread count differs.** The reference ran `-t 8`. The subject ran its default thread pools:
   632–644% CPU on intel, 1933–1950% on lambda. The subject bytes were identical at both host
   parallelisms. Reference determinism at other `-t` is `[U]` per REFERENCE-RAW.md.
4. **Harness, not the `apr` binary.** The subject is a small example program that calls the library
   loop `apr run` uses. It is not the pinned `apr` CLI (`. scripts/apr_bin.sh`), because
   that CLI has no per-position logit output. The load and forward calls are the same functions.
   Sampling, chat template and tokenizer are not exercised. `[U]` whether `apr run` on #3114 greedy-decodes these ids to
   the same tokens: `. scripts/apr_bin.sh && timeout 600 "$APR" run ~/models/Qwen3.5-0.8B-Q4_K_M.gguf --prompt "$(cat ../qwen35-cpu-reference/input-prompt-nl-prompt.txt)" --max-tokens 1 < /dev/null`
   (that route also needs apr's tokenization of the prompt proven equal to the 78 ids).
5. **Intel load.** intel is the CI fleet host, at load 35–46 on 32 threads during the subject runs.
   Wall time is affected. The bytes are not (identical to lambda's at load ≈ 8).
6. **Single prompt.** One 78-token natural-language prompt. Other prompts, longer contexts and
   decode steps past the prompt are `[U]`. `examples/qwen35_parity.rs` in #3114 covers 3 short
   prompts teacher-forced, top-1 plus near-tie only, and does not report cosine.
7. **B5 undecided.** There is no known-bad Qwen3.5 pair, so these numbers cannot be put on a
   good/bad axis for this architecture.

## Per-position table (n = 78; host intel both sides)

| pos | token_id | cosine | argmax_ref | argmax_apr | argmax_match | max_abs_diff | ref_top1_top2_gap | apr_rank_of_ref_argmax | apr_gap_to_ref_argmax |
|-----|----------|--------|-----------|-----------|--------------|--------------|-------------------|------------------------|-----------------------|
| 0 | 760 | 0.998376 | 2614 | 2614 | yes | 1.018627 | 1.407180 | 1 | 0.000000 |
| 1 | 3841 | 0.998317 | 321 | 321 | yes | 0.800874 | 0.593156 | 1 | 0.000000 |
| 2 | 13477 | 0.998216 | 627 | 37550 | **no** | 1.342222 | 0.432076 | 2 | 0.197060 |
| 3 | 37550 | 0.998583 | 321 | 321 | yes | 1.054795 | 0.137941 | 1 | 0.000000 |
| 4 | 33075 | 0.995905 | 888 | 888 | yes | 1.044012 | 0.745678 | 1 | 0.000000 |
| 5 | 888 | 0.999371 | 279 | 279 | yes | 0.726841 | 0.581070 | 1 | 0.000000 |
| 6 | 279 | 0.999520 | 15740 | 15740 | yes | 0.754851 | 0.534869 | 1 | 0.000000 |
| 7 | 15217 | 0.999529 | 5388 | 5388 | yes | 0.962131 | 0.575095 | 1 | 0.000000 |
| 8 | 5388 | 0.998932 | 13 | 13 | yes | 0.712698 | 1.329920 | 1 | 0.000000 |
| 9 | 1345 | 0.999490 | 279 | 279 | yes | 0.657455 | 2.534767 | 1 | 0.000000 |
| 10 | 279 | 0.999664 | 15217 | 15217 | yes | 0.492375 | 1.284418 | 1 | 0.000000 |
| 11 | 12433 | 0.998927 | 369 | 369 | yes | 0.553746 | 0.900751 | 1 | 0.000000 |
| 12 | 21250 | 0.999436 | 279 | 279 | yes | 0.553334 | 2.958714 | 1 | 0.000000 |
| 13 | 34227 | 0.999451 | 34129 | 34129 | yes | 0.545431 | 0.702033 | 1 | 0.000000 |
| 14 | 36165 | 0.999574 | 314 | 314 | yes | 0.519200 | 0.446839 | 1 | 0.000000 |
| 15 | 23497 | 0.999304 | 13 | 13 | yes | 0.513666 | 0.215031 | 1 | 0.000000 |
| 16 | 883 | 0.999587 | 279 | 279 | yes | 0.564519 | 1.200860 | 1 | 0.000000 |
| 17 | 31108 | 0.999497 | 4649 | 4649 | yes | 0.535324 | 4.276793 | 1 | 0.000000 |
| 18 | 4649 | 0.999454 | 13 | 13 | yes | 0.567844 | 1.246357 | 1 | 0.000000 |
| 19 | 5638 | 0.999111 | 13 | 13 | yes | 0.619223 | 1.258518 | 1 | 0.000000 |
| 20 | 11 | 0.999073 | 321 | 321 | yes | 0.564489 | 0.786652 | 1 | 0.000000 |
| 21 | 4098 | 0.998294 | 3808 | 3808 | yes | 0.833739 | 0.065741 | 1 | 0.000000 |
| 22 | 23122 | 0.996803 | 11 | 11 | yes | 1.133435 | 0.353788 | 1 | 0.000000 |
| 23 | 11 | 0.997968 | 321 | 321 | yes | 1.010476 | 3.715588 | 1 | 0.000000 |
| 24 | 15167 | 0.999010 | 3741 | 3741 | yes | 0.651882 | 0.347454 | 1 | 0.000000 |
| 25 | 47410 | 0.998615 | 11 | 11 | yes | 0.673544 | 2.470766 | 1 | 0.000000 |
| 26 | 11 | 0.999436 | 321 | 321 | yes | 0.489212 | 3.126120 | 1 | 0.000000 |
| 27 | 9966 | 0.997801 | 1954 | 1954 | yes | 0.645328 | 0.590347 | 1 | 0.000000 |
| 28 | 1452 | 0.996796 | 795 | 28758 | **no** | 1.201192 | 0.632557 | 2 | 0.216549 |
| 29 | 6326 | 0.998986 | 11 | 11 | yes | 0.710832 | 1.596657 | 1 | 0.000000 |
| 30 | 61369 | 0.998525 | 11 | 11 | yes | 0.584024 | 2.966839 | 1 | 0.000000 |
| 31 | 11 | 0.999428 | 321 | 321 | yes | 0.501533 | 2.666105 | 1 | 0.000000 |
| 32 | 321 | 0.999367 | 279 | 279 | yes | 0.549079 | 0.094513 | 1 | 0.000000 |
| 33 | 279 | 0.999418 | 491 | 491 | yes | 0.552459 | 0.879807 | 1 | 0.000000 |
| 34 | 26984 | 0.998275 | 314 | 314 | yes | 0.641287 | 5.691628 | 1 | 0.000000 |
| 35 | 314 | 0.999385 | 29144 | 29144 | yes | 0.572366 | 0.546744 | 1 | 0.000000 |
| 36 | 638 | 0.998697 | 5331 | 57695 | **no** | 0.779463 | 0.013315 | 2 | 0.075449 |
| 37 | 37309 | 0.997945 | 290 | 290 | yes | 0.724147 | 0.495329 | 1 | 0.000000 |
| 38 | 290 | 0.999405 | 14791 | 14791 | yes | 0.739064 | 0.874512 | 1 | 0.000000 |
| 39 | 42903 | 0.998776 | 13 | 13 | yes | 0.811644 | 2.345137 | 1 | 0.000000 |
| 40 | 3808 | 0.999563 | 279 | 279 | yes | 0.496683 | 0.658201 | 1 | 0.000000 |
| 41 | 2943 | 0.999303 | 3478 | 3478 | yes | 0.603065 | 0.010328 | 1 | 0.000000 |
| 42 | 94408 | 0.999102 | 795 | 795 | yes | 0.669424 | 0.522985 | 1 | 0.000000 |
| 43 | 17882 | 0.999036 | 13 | 13 | yes | 0.603948 | 1.551729 | 1 | 0.000000 |
| 44 | 26 | 0.998808 | 279 | 279 | yes | 0.658595 | 0.785696 | 1 | 0.000000 |
| 45 | 1396 | 0.998856 | 13340 | 13340 | yes | 0.759709 | 1.197609 | 1 | 0.000000 |
| 46 | 13901 | 0.999083 | 314 | 314 | yes | 0.593441 | 0.245081 | 1 | 0.000000 |
| 47 | 557 | 0.999377 | 5158 | 5158 | yes | 0.550978 | 0.401985 | 1 | 0.000000 |
| 48 | 47193 | 0.999443 | 11 | 11 | yes | 0.542975 | 0.224258 | 1 | 0.000000 |
| 49 | 11 | 0.998193 | 1396 | 1396 | yes | 0.594273 | 0.152668 | 1 | 0.000000 |
| 50 | 1396 | 0.999094 | 11316 | 11316 | yes | 0.568399 | 1.556353 | 1 | 0.000000 |
| 51 | 1898 | 0.998765 | 557 | 557 | yes | 0.633099 | 1.444838 | 1 | 0.000000 |
| 52 | 557 | 0.999413 | 28853 | 28853 | yes | 0.510546 | 0.485321 | 1 | 0.000000 |
| 53 | 21197 | 0.998932 | 11 | 11 | yes | 0.547749 | 2.163490 | 1 | 0.000000 |
| 54 | 11 | 0.999185 | 321 | 321 | yes | 0.528216 | 1.309868 | 1 | 0.000000 |
| 55 | 321 | 0.999291 | 1396 | 1396 | yes | 0.524079 | 0.574141 | 1 | 0.000000 |
| 56 | 279 | 0.999222 | 4307 | 4307 | yes | 0.578575 | 0.792404 | 1 | 0.000000 |
| 57 | 1830 | 0.998911 | 557 | 557 | yes | 0.638514 | 0.733721 | 1 | 0.000000 |
| 58 | 9202 | 0.999104 | 440 | 440 | yes | 0.518411 | 4.318123 | 1 | 0.000000 |
| 59 | 440 | 0.999400 | 264 | 264 | yes | 0.540972 | 1.813231 | 1 | 0.000000 |
| 60 | 264 | 0.999414 | 1562 | 1562 | yes | 0.615428 | 0.482038 | 1 | 0.000000 |
| 61 | 50802 | 0.999121 | 314 | 314 | yes | 0.612707 | 1.492762 | 1 | 0.000000 |
| 62 | 314 | 0.999333 | 279 | 279 | yes | 0.588779 | 0.335559 | 1 | 0.000000 |
| 63 | 50614 | 0.999261 | 34129 | 34129 | yes | 0.596232 | 0.321232 | 1 | 0.000000 |
| 64 | 39663 | 0.999375 | 3470 | 3470 | yes | 0.559120 | 1.725536 | 1 | 0.000000 |
| 65 | 22188 | 0.999253 | 7123 | 7123 | yes | 0.668263 | 0.552864 | 1 | 0.000000 |
| 66 | 7123 | 0.999104 | 13 | 13 | yes | 0.618345 | 1.755409 | 1 | 0.000000 |
| 67 | 421 | 0.999297 | 998 | 998 | yes | 0.630416 | 0.255558 | 1 | 0.000000 |
| 68 | 995 | 0.999148 | 978 | 978 | yes | 0.589481 | 1.514719 | 1 | 0.000000 |
| 69 | 310 | 0.999165 | 381 | 381 | yes | 0.644727 | 2.247143 | 1 | 0.000000 |
| 70 | 1440 | 0.999085 | 1518 | 1518 | yes | 0.689554 | 1.109022 | 1 | 0.000000 |
| 71 | 1518 | 0.999496 | 279 | 279 | yes | 0.568270 | 1.373180 | 1 | 0.000000 |
| 72 | 279 | 0.999385 | 1534 | 1534 | yes | 0.593024 | 0.258187 | 1 | 0.000000 |
| 73 | 4722 | 0.999186 | 13 | 314 | **no** | 0.724819 | 0.109070 | 2 | 0.119488 |
| 74 | 1362 | 0.999449 | 381 | 381 | yes | 0.565757 | 1.478558 | 1 | 0.000000 |
| 75 | 381 | 0.999367 | 11454 | 11454 | yes | 0.669799 | 1.763706 | 1 | 0.000000 |
| 76 | 35883 | 0.999451 | 430 | 430 | yes | 0.547609 | 0.818068 | 1 | 0.000000 |
| 77 | 13 | 0.998885 | 271 | 271 | yes | 0.626695 | 0.018402 | 1 | 0.000000 |

## Files

| file | what |
|------|------|
| `qwen35_raw_logits.rs` | subject harness (built inside the #3114 tree) |
| `compare_raw_logits.py` | comparator: re-run it with the command above, rc 0 |
| `prompt_token_ids.txt` | the 78 ids |
| `comparison.tsv` / `comparison.json` | comparator output behind every number above |
