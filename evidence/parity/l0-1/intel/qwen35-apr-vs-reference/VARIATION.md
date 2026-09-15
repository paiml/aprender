# Qwen3.5-0.8B Q4_K_M: argmax-flip variation across 5 prompts, plus the apr thread confound (PMAT-3091)

Verification-discipline rule 6: one failing input is an anecdote. Positions 28 and 73 of the #3091 prompt flip
against BOTH llama.cpp prefill modes (`PER-TOKEN.md`). This file varies the prompt and the apr thread count
before any cause gets named. **No threshold is set. The Qwen3.5 parity row stays [U], fail-closed.**

## Tree (every measurement below)

| side | tree |
|---|---|
| evidence worktree (flipvar-3091), base of this commit | `fe1c46b8fbc206eb9c9dd93282518dd4b6e137b4`, the merge of PMAT-3303-per-token-prefill `7647b4a41` and PMAT-3091-apr-vs-reference |
| apr subject: #3114 head `31448f6c3` (branch PMAT-1098-qwen35-cpu), detached checkout, `qwen35_raw_logits.rs` copied in uncommitted | binary sha256 `ad4397d5efcc8aa64a7b4e15303b32e5b51de3180c2da80cbf247cdfe7253d8e` |
| reference: llama.cpp | `d1d3c3396`. Producer `apr_raw_logits` sha256 `5ef2b60ac1761efcd0584cfdcf67363e2c7ebaf1b08a6bcb19cf9d933af403e3`; `llama-tokenize` sha256 `bce3a4b6ba153a0f65436c7f284ae6eb01ac455a3d86634950765965d30c0617` |
| model | `Qwen3.5-0.8B-Q4_K_M.gguf`, sha256 `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517` |
| comparator | `compare_raw_logits.py` (committed alongside this file), sha256 `f05cd87aef3bbe8396606f91334c974ddcb4f59d68186c35611405f74bbdd4ce`, the same bytes on intel |

**apr build.** `MEASUREMENT.md` built from `5b055c0ca`. The #3114 head is now `31448f6c3`. Between the two commits,
aprender-serve changed only under `#[cfg(test)]` (`forward_qwen35.rs` +6 lines, a `mod qhf_contract_tests` include).
It was rebuilt on lambda, reusing the earlier target dir (incremental, 24.55 s, load 0.87/4.02/3.74 at start):

```bash
git -C /mnt/nvme-raid0/agent-wt/apr-3114 worktree add --detach /tmp/flipvar3091/apr-31448 31448f6c3
cp evidence/parity/l0-1/intel/qwen35-apr-vs-reference/qwen35_raw_logits.rs /tmp/flipvar3091/apr-31448/crates/aprender-serve/examples/
cd /tmp/flipvar3091/apr-31448 && CARGO_BUILD_JOBS=8 CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/apr-3114-meas \
  nice -n 10 timeout 3000 ~/.cargo/bin/cargo build --release -p aprender-serve --example qwen35_raw_logits < /dev/null   # rc 0
```
The result is **byte-identical** to the `5b055c0ca` binary: same sha256 `ad4397d5…3d8e`, `cmp -l` 0 differing bytes, relinked at 21:34:58 local.
It was copied to `intel:~/parity-ref/variation-3091/qwen35_raw_logits`, where the sha was re-verified.

## Host

All runs were on intel (`mac-server`, Xeon W-3245, 32 threads), CPU only, one job at a time. Every binary ran under `timeout` with `< /dev/null`. Load (1/5/15):
- 19:32:14Z, first contact: 63.69 / 54.86 / 47.83
- 19:35:55Z, driver start: 32.53 / 45.56 / 45.83
- 19:47:30Z, after the last run: 46.24 / 45.31 / 45.45

Per-run load is in `variation/out/run_variation.transcript` and `variation/out/run_fix_p2p4.transcript`.

## 1. Thread confound

The original 78 ids went through the rebuilt binary three times: default pool, `RAYON_NUM_THREADS=8`, and `RAYON_NUM_THREADS=1`
(`variation/run_variation.sh`, phase T). realizar's CPU pools size themselves from `rayon::current_num_threads()`
(`quantize/gemv_pool.rs:128`, `quantize/spin_pool.rs:16`). **Proof that the pin engaged:** the driver read `Threads:` from
`/proc/<pid>/status` every 0.2 s and kept the maximum, and `/usr/bin/time -v` recorded CPU%.

| run | n | max `Threads:` sampled | CPU% | wall | output sha256 |
|---|---|---|---|---|---|
| default | 1 | 33 | 785% | 13.35 s | `f6f792649014dcef6eec1afaf96852a293cc95c6cfceb1c88d62e16ee36c0252` |
| `RAYON_NUM_THREADS=8` | 1 | 9 | 545% | 12.38 s | `f6f792649014dcef6eec1afaf96852a293cc95c6cfceb1c88d62e16ee36c0252` |
| `RAYON_NUM_THREADS=1` | 1 | 2 | 99% | 58.11 s | `f6f792649014dcef6eec1afaf96852a293cc95c6cfceb1c88d62e16ee36c0252` |

All three equal the recorded subject `f6f79264…0252`, which is also n=4 across intel and lambda in `MEASUREMENT.md`.
**Verdict: the thread confound is eliminated.** apr's logits for these ids do not depend on pool size (1, 8, or 32 workers), and they do not depend on the #3114 head (`5b055c0ca` or `31448f6c3`).

## 2. Prompts and tokenization

| k | kind | file (sha256) | ids (sha256) | n tokens |
|---|---|---|---|---|
| 0 | original #3091 prose | `/tmp/ref3303/prompt.txt` (`e6796dc1…0eaa`) | `../prompt_token_ids.txt` (`62e90835…5e72`) | 78 |
| 1 | English prose | `variation/prompt-1.txt` (`9a7cc3f840df7d56e0ac4bf3141f097e24e717a085486a54c1fc28cbc40204ee`) | `variation/prompt-1.ids` (`a4df9cdea5769e5e0ab81ba20e3a6bc890767b4cd668ae001b5c3b4034f380be`) | 83 |
| 2 | Rust code | `variation/prompt-2.txt` (`134e1832e1b3cf189fa70a6d678a47234e77ce718588ab9ec9f5cc3b9f321787`) | `variation/prompt-2.ids` (`77001b22094106e48dc38b37e49a816e0af36d03dceba93f1feb1373bce297d6`) | 117 |
| 3 | numeric/list-heavy | `variation/prompt-3.txt` (`c2a4c3daf5b69e7cce50ab7246fb1f805826090462c346bd58941363aee5ac98`) | `variation/prompt-3.ids` (`1b7d56c1ff086b7332d9ece65683797417f8237a34a99705c9ce396b04fd8ca2`) | 108 |
| 4 | chat-template turn (`<\|im_start\|>system … <\|im_start\|>assistant\n`) | `variation/prompt-4.txt` (`7873eb8de372472c24dfb713b6720fd551b60fd1821f26a50850a807751c8f23`) | `variation/prompt-4.ids` (`42a64dc1c78e3b4ae0ad7ece7dcebf4ee80550931866e53762944c118ae68b9e`) | 82 |

Tokenizer: `llama-tokenize -m MODEL --ids --no-parse-special --log-disable`. `--no-parse-special` matches the producer, which
calls `common_tokenize(ctx, prompt, true)` with `parse_special` defaulting to `false`. The comparator refuses any pair whose
header ids differ, and all 12 pairs over prompts 1–4 were accepted. The driver also string-compared every `prompt-k.ids` with the
`token_ids=` line the producer logged, in both modes, and all 8 were equal (`run_fix_p2p4.transcript`).

**Deviations, recorded rather than hidden:**
- **Prompt 3 was cut once, before any engine ran.** Its first text (5 inventory lines) tokenized to 146 ids, over the 120 limit.
  Two lines were removed, and the final text tokenizes to 108.
- **Prompts 2 and 4 were tokenized twice.** `llama-tokenize -f` keeps the file's trailing `\n`. The producer's common `-f`
  parser (`common/arg.cpp:1804`) pops exactly one. The first-attempt ids differed from the producer's in the last
  token only (p2: 118 ids ending `…,92,198` vs 117; p4: last id 271 `\n\n` vs 198 `\n`). The comparator refused B/C for both (rc 2).
  They were re-tokenized from the producer's exact text (the file minus one trailing `\n`, via `--stdin`; neither text contains a
  backslash, so escape handling is a no-op). apr was then re-run on p2/p4, and B/C were re-run. The first attempts are kept as
  `variation/prompt-{2,4}.ids.first-attempt` (`b30c7fda…336f`, `db24cf1f…c8d3`), with apr outputs `p2-apr.first-attempt-ids.bin`
  (`0453801180d66354a11edc522960f21c7a467ea74e5d3fe8ebaa83fd1d52a3c4`) and `p4-apr.first-attempt-ids.bin`
  (`c2cedf2feb1c37c208a16478bd2949708d05ac440d09e7f87477b8a424d45e76`). No number below comes from them.
- **Prompt 4's chat markup is plain text, not special tokens.** The producer cannot parse specials, so `<|im_start|>` is 6 ids,
  `'<' '|' 'im' '_start' '|' '>'`. A true chat turn (special-token ids) is **[U]**. It needs a producer that takes ids or
  `parse_special=true`, which is outside this ticket's scope.

## Commands (host intel unless stated)

```bash
# tokenize (once per final text; see deviations above)
timeout 120 ~/src/llama.cpp-d1d3c3396/build/bin/llama-tokenize -m ~/models/Qwen3.5-0.8B-Q4_K_M.gguf -f prompt-k.txt --ids --no-parse-special --log-disable < /dev/null   # k = 1, 3
timeout 120 llama-tokenize -m MODEL --stdin --ids --no-parse-special --log-disable < prompt-k.tok-input.txt                                                          # k = 2, 4
bash ~/parity-ref/variation-3091/run_variation.sh    # phase T, then P (k=1..4: batched, --per-token, apr), then C; transcript variation/out/run_variation.transcript
bash ~/parity-ref/variation-3091/run_fix_p2p4.sh     # p2/p4 re-tokenize, apr, B/C, then classify; transcript variation/out/run_fix_p2p4.transcript
python3 classify_flips.py classify_spec.json flips.json                       # rc 0, empty stderr
python3 context_top5.py <variation-dir> <orig-subject-dir> <orig-per-token-dir> <orig-batched.bin> MODEL <gguf-py>   # rc 0
```
Inside the drivers:
- llama: `timeout 900 apr_raw_logits -m MODEL -f prompt-k.txt -ngl 0 -t 8 -c N -b N [--per-token] --raw-out F < /dev/null`, where N is the prompt's token count.
- apr: `timeout 1800 qwen35_raw_logits MODEL prompt-k.ids F < /dev/null`, default thread pool, the same as `MEASUREMENT.md`.
- comparator: `timeout 600 python3 compare_raw_logits.py <ref> <sub> --json <label>.json > <label>.tsv`, with pairs A = per-token → batched, B = per-token → apr, C = batched → apr.

Helper sha256:
- `run_variation.sh` `5320425cd4020201b9dfa9f12b77efe3a1869ddbb0cdc9def72efad3869c48a9`
- `run_fix_p2p4.sh` `9e03afe4c95766ff42acce3dcb33e2557b57efbf814e3d0a3339e740cb70c761`
- `classify_flips.py` `44f3acf933b0cef5cacd455b764896a6ea9553cf50fb9b524b1ef8d3c4f57ce7` (committed version). The first run used `49c589e6…c4a9`. It was split into helpers for the pre-commit complexity gate, re-run on intel as `classify_flips_v2.py` (rc 0, empty stderr), and `cmp flips.json flips.v2.json` returned rc 0, so the committed script reproduces the analysed output byte for byte
- `classify_spec.json` `6db75f196f2ca9d57c42b7a3fd389e402aabf0c983e00b1891a4a44a1ed75890`
- `context_top5.py` `5841ba054ba847e1629745543da343ce43165e835f69b21e423fd81aab18969b`

## Outputs (sha256, intel `~/parity-ref/variation-3091/`; the .bin files are not committed)

**n = 1 per new output.** That is enough for these files because the producer's modes were already shown byte-deterministic on this
host: batched n=5 (`REFERENCE-RAW.md`) and per-token n=3 (`PER-TOKEN.md`). apr is n=7 byte-identical on prompt 0 (n=4 in `MEASUREMENT.md`, n=3 here).
Determinism on the new prompts themselves was not re-measured: **[U]**, rerun `run_variation.sh` phase P and `cmp`.

| k | llama batched | llama per-token | apr |
|---|---|---|---|
| 0 | `1d5eaf129545aa6087cd6c49792f9774b61caa874fccefdd79e6164afa15bb93` | `2401c11005e08bfac946a1de57af84338cc64a0d8e9a95d0941a6023b81bebc4` | `f6f792649014dcef6eec1afaf96852a293cc95c6cfceb1c88d62e16ee36c0252` |
| 1 | `04221c5689ba7a66edb57dfa9c8f78d82a7c181c754ba08636739f9cd4b686da` | `67dcc9e73382f240a2278cb66863499d882c8eb526ee25737792998da965f10a` | `b9815d9be8b14b255db54d22821ff63a306d7dd8ae94d7bdfe4c93328376b7f8` |
| 2 | `985dc34529154df7ed7c1a501c87abea4cbadd9ab4dc58eee213830bb53460c6` | `a0f87b14ba9f2dd7630dd60268ef323f977e5f88cb9e060cabbfafe4aae8b760` | `d5ffa2b73bc304b93c8461b103e66092c3f8a1d0850d596d3ee9b5006179a7fa` |
| 3 | `788f0ab15d6f2bfb33a18a00be2bb4c656be6fcaf348433e24afae274d76a382` | `df47beffcc18cf27caed695669ad749642ec3a859c3e3c1652068bfa2ac66ea5` | `b2a984d0bacc24c1e5963f16256194c172fa99dd90481f6b364fc22a206e3e91` |
| 4 | `cba9ec3eb910d1c251a01b74c701ec3d5ae41169437461acd1ca234686489273` | `0eca077ceae11c0ef182e09d74a9b019a91503a56e491644a437765b297e6f1e` | `92f1b54d504a53f758a56911c4779ce1900207bd89c78d58597e558ebc6d3ee9` |

The comparator TSV/JSON for prompts 1–4 are committed under `variation/out/`, with sha256 in the transcripts. The prompt 0 JSON is unchanged from
`PER-TOKEN.md`: A `fa65a138…d8f6`, B `bbe9c397…6119`, C `3b58b5d7…c4c6`.
`variation/out/flips.json` sha256 is `cd308866200f283d4e359282af99683fee8d2a6e8fe247bf28a69aace0f923a5`, and `variation/out/context_top5.out` is `cd41675e46de5d7c14980de754c1edd4aaf81d26a7018742329ca2c78acd30d3`.

## 3. Per-prompt flips

Classes are per position:
- **llama-self**: per-token argmax ≠ batched argmax.
- **persistent**: both llama modes agree and apr differs from them.
- **mode-specific**: apr equals exactly one llama mode.
- **three-way**: all three differ.

"Gap" is the top1 − top2 logit gap of the named engine. The per-token llama gap is the context value, because per-token is the mode that matches apr's one-token forward. Comparator run on intel, n=1 per pair.

| k | A argmax mismatches | B argmax mismatches | C argmax mismatches | min cos A | min cos B | min cos C | positions with cos B < cos A |
|---|---|---|---|---|---|---|---|
| 0 | 4 | 4 | 4 | 0.998197 | 0.994599 | 0.995905 | 52 / 78 |
| 1 | 8 | 4 | 8 | 0.998855 | 0.997448 | 0.997315 | 38 / 83 |
| 2 | 3 | 2 | 3 | 0.996228 | 0.995370 | 0.994944 | 83 / 117 |
| 3 | 6 | 5 | 7 | 0.997185 | 0.995687 | 0.995826 | 76 / 108 |
| 4 | 3 | 6 | 6 | 0.986225 | **0.961309** | **0.953493** | 67 / 82 |

### Prompt 0: original NL prose (78 ids, #3091) (n = 78 positions)

Min cosine: A (per-token vs batched llama) 0.998197 · **B (per-token llama vs apr) 0.994599** · C (batched llama vs apr) 0.995905

- llama-self flips (4): [2, 21, 36, 77]
- mode-specific flips (4): 2 (apr = per-token), 21 (apr = batched), 36 (apr = per-token), 77 (apr = batched)
- three-way (all three differ): none
- persistent flips (2):

| pos | input piece | llama argmax (both modes) | apr argmax | llama per-token top-2 gap | llama batched top-2 gap | apr rank of llama argmax | apr own top-2 gap | cosine B |
|---|---|---|---|---|---|---|---|---|
| 28 | 'ized' (1452) | 795 'Ġdata' | 28758 'Ġneural' | 0.31467 | 0.632557 | 2 | 0.216549 | 0.998569 |
| 73 | 'Ġrelease' (4722) | 13 '.' | 314 'Ġof' | 0.245474 | 0.10907 | 2 | 0.119488 | 0.999303 |

### Prompt 1: English prose (n = 83 positions)

Min cosine: A (per-token vs batched llama) 0.998855 · **B (per-token llama vs apr) 0.997448** · C (batched llama vs apr) 0.997315

- llama-self flips (8): [8, 11, 15, 25, 32, 43, 54, 81]
- mode-specific flips (8): 8 (apr = per-token), 11 (apr = per-token), 15 (apr = per-token), 25 (apr = per-token), 32 (apr = per-token), 43 (apr = per-token), 54 (apr = batched), 81 (apr = batched)
- three-way (all three differ): none
- persistent flips (2):

| pos | input piece | llama argmax (both modes) | apr argmax | llama per-token top-2 gap | llama batched top-2 gap | apr rank of llama argmax | apr own top-2 gap | cosine B |
|---|---|---|---|---|---|---|---|---|
| 1 | 'Ġthe' (279) | 1156 'Ġuser' | 1510 'Ġ`' | 0.38298 | 0.196401 | 2 | 0.017526 | 0.999168 |
| 70 | 'Ġsimply' (4777) | 2047 'Ġleft' | 20922 'Ġretired' | 0.099743 | 0.11167 | 2 | 0.069799 | 0.999587 |

### Prompt 2: Rust code (n = 117 positions)

Min cosine: A (per-token vs batched llama) 0.996228 · **B (per-token llama vs apr) 0.99537** · C (batched llama vs apr) 0.994944

- llama-self flips (3): [8, 12, 71]
- mode-specific flips (3): 8 (apr = batched), 12 (apr = per-token), 71 (apr = per-token)
- three-way (all three differ): none
- persistent flips (1):

| pos | input piece | llama argmax (both modes) | apr argmax | llama per-token top-2 gap | llama batched top-2 gap | apr rank of llama argmax | apr own top-2 gap | cosine B |
|---|---|---|---|---|---|---|---|---|
| 57 | 'ĠĠĠĠĠĠĠ' (285) | 1042 'Ġlet' | 413 'Ġif' | 0.138132 | 0.167967 | 2 | 0.036299 | 0.99866 |

### Prompt 3: numeric/list-heavy (n = 108 positions)

Min cosine: A (per-token vs batched llama) 0.997185 · **B (per-token llama vs apr) 0.995687** · C (batched llama vs apr) 0.995826

- llama-self flips (6): [11, 42, 43, 70, 94, 105]
- mode-specific flips (6): 11 (apr = batched), 42 (apr = per-token), 43 (apr = batched), 70 (apr = per-token), 94 (apr = per-token), 105 (apr = per-token)
- three-way (all three differ): none
- persistent flips (3):

| pos | input piece | llama argmax (both modes) | apr argmax | llama per-token top-2 gap | llama batched top-2 gap | apr rank of llama argmax | apr own top-2 gap | cosine B |
|---|---|---|---|---|---|---|---|---|
| 0 | 'Inventory' (21626) | 9268 'ĠManagement' | 25 ':' | 0.2586 | 0.611251 | 2 | 0.198638 | 0.995687 |
| 17 | '0' (15) | 15 '0' | 12 '-' | 0.052027 | 0.19916 | 2 | 0.02069 | 0.999314 |
| 74 | 'Ġfrom' (494) | 220 'Ġ' | 1483 'Ġlast' | 0.257064 | 0.027846 | 2 | 0.051846 | 0.999472 |

### Prompt 4: chat-template user turn (markup as plain text) (n = 82 positions)

Min cosine: A (per-token vs batched llama) 0.986225 · **B (per-token llama vs apr) 0.961309** · C (batched llama vs apr) 0.953493

- llama-self flips (3): [2, 24, 32]
- mode-specific flips (2): 24 (apr = per-token), 32 (apr = batched)
- three-way (all three differ): 2 (pt 91, bat 33319, apr 3330; pt gap 0.23905)
- persistent flips (4):

| pos | input piece | llama argmax (both modes) | apr argmax | llama per-token top-2 gap | llama batched top-2 gap | apr rank of llama argmax | apr own top-2 gap | cosine B |
|---|---|---|---|---|---|---|---|---|
| 4 | '|' (91) | 16 '1' | 29 '>' | 0.234961 | 0.228786 | 4 | 0.377001 | 0.976518 |
| 6 | 'system' (8678) | 91 '|' | 59348 '_prompt' | 0.945776 | 0.389656 | 3 | 0.074289 | 0.996164 |
| 20 | 'im' (316) | 4747 '_start' | 6018 '_end' | 0.755405 | 0.938725 | 2 | 0.27659 | 0.990532 |
| 52 | 'Ġit' (424) | 264 'Ġa' | 5902 'Ġsafe' | 0.029835 | 0.010665 | 2 | 0.010218 | 0.999749 |

## 4. Pooled distribution (prompts 0–4, intel, n = 1 comparison per pair)

| stat | value |
|---|---|
| positions | **468** (78 + 83 + 117 + 108 + 82) |
| persistent flips | **12** (per prompt 2, 2, 1, 3, 4) |
| llama-self flips | 24 · mode-specific 23 · three-way 1 (p4 pos 2) |
| llama per-token top-2 gap at every persistent flip, sorted | 0.029835, 0.052027, 0.099743, 0.138132, 0.234961, **0.245474 (p0 pos 73)**, 0.257064, 0.258600, **0.314670 (p0 pos 28)**, 0.382980, 0.755405, 0.945776 |
| share of ALL 468 positions whose per-token gap is below that gap | 0.0235, 0.0385, 0.0705, 0.0919, 0.1560, 0.1645, 0.1688, 0.1688, 0.2051, 0.2329, 0.3889, 0.4466 |
| llama per-token top-2 gap percentiles over all 468 positions (p5, p25, p50) | **0.069416, 0.409569, 1.148219** |
| apr rank of llama's argmax at the 12 persistent flips | 2 at 10 of them; 3 at p4 pos 6; 4 at p4 pos 4 |
| largest-gap persistent flip | **prompt 4, pos 6.** Input `'system'` (8678); llama argmax `'\|'` (91), per-token gap 0.945776 (batched 0.389656); apr argmax `'_prompt'` (59348), apr's rank of `'\|'` 3, apr own gap 0.074289 |
| defect candidates (persistent flip with llama top-2 gap ≥ 1.0) | **none** |

Top-5 at the largest-gap persistent flips and at prompt 4's minimum-cosine position (`variation/out/context_top5.out`; context only, not a candidate):

```
-- prompt-4 pos 6 input 8678 'system'
   llama per-token: 91 16.9612 '|' | 59348 16.0155 '_prompt' | 25 15.9387 ':' | 198 15.8244 'Ċ' | 510 15.1883 '</'
   llama batched  : 91 16.6105 '|' | 59348 16.2209 '_prompt' | 25 15.9943 ':' | 198 15.6131 'Ċ' | 53077 15.3180 '_instruction'
   apr            : 59348 16.0894 '_prompt' | 25 16.0151 ':' | 91 16.0151 '|' | 198 15.8895 'Ċ' | 27 15.2749 '<'
-- prompt-4 pos 20 input 316 'im'
   llama per-token: 4747 22.9617 '_start' | 6018 22.2063 '_end' | 18466 20.2217 '_stop' | 41560 16.7768 '_finish' | 1217 15.9468 '_st'
   llama batched  : 4747 23.3259 '_start' | 6018 22.3872 '_end' | 18466 20.4803 '_stop' | 41560 16.8031 '_finish' | 1217 16.0980 '_st'
   apr            : 6018 21.9417 '_end' | 4747 21.6651 '_start' | 18466 20.2692 '_stop' | 41560 16.2393 '_finish' | 2135 16.0904 '_e'
-- prompt-4 pos 1 input 91 '|'   (min cosine B 0.961309; argmax agrees)
   llama per-token: 16 14.7762 '1' | 15 14.3543 '0' | 2267 14.2553 'path' | 3112 14.1968 'vector' | 8678 14.0728 'system'
   llama batched  : 16 15.0272 '1' | 8678 14.4342 'system' | 3112 14.1905 'vector' | 2267 13.9741 'path' | 15 13.9205 '0'
   apr            : 16 15.5327 '1' | 8678 15.3913 'system' | 15 14.6306 '0' | 846 14.2298 'user' | 2267 14.0530 'path'
-- prompt-0 pos 28 input 1452 'ized'
   llama per-token: 795 15.6902 'Ġdata' | 28758 15.3755 'Ġneural' | 29144 14.2875 'Ġquantum' | 11639 14.1518 'Ġnoise' | 22525 14.0380 'Ġgravity'
   apr            : 28758 15.7937 'Ġneural' | 795 15.5771 'Ġdata' | 29144 14.1590 'Ġquantum' | 3983 14.0696 'Ġmodels' | 13914 14.0083 'Ġweights'
-- prompt-0 pos 73 input 4722 'Ġrelease'
   llama per-token: 13 18.7355 '.' | 314 18.4900 'Ġof' | 628 16.9782 'Ġcan' | 1362 16.8695 'Ġcould' | 557 16.5303 'Ġwas'
   apr            : 314 18.6635 'Ġof' | 13 18.5440 '.' | 628 16.9293 'Ġcan' | 1362 16.6781 'Ġcould' | 557 16.2018 'Ġwas'
```

Prompt 4, positions 0–8 and 19–21: cosine and per-token gap. Its markup tokens are where the cosine drops.

| pos | input | cos A | cos B | cos C | per-token gap | apr rank of per-token argmax | max \|diff\| B |
|---|---|---|---|---|---|---|---|
| 0 | `<` 27 | 0.998534 | 0.995365 | 0.995839 | 0.1921 | 1 | 1.3458 |
| 1 | `\|` 91 | 0.991359 | **0.961309** | **0.953493** | 0.4219 | 1 | **4.1140** |
| 2 | `im` 316 | 0.996107 | 0.973410 | 0.976287 | 0.2390 | 2 | 3.8916 |
| 3 | `_start` 4747 | 0.995901 | 0.981265 | 0.983914 | 0.5380 | 1 | 2.3491 |
| 4 | `\|` 91 | 0.993730 | 0.976518 | 0.984359 | 0.2350 | 4 | 2.3653 |
| 5 | `>` 29 | 0.995443 | 0.985978 | 0.989244 | 0.4892 | 1 | 2.0916 |
| 6 | `system` 8678 | 0.997770 | 0.996164 | 0.996292 | 0.9458 | 3 | 1.3431 |
| 7 | `\n` 198 | 0.997825 | 0.992285 | 0.992890 | 0.9150 | 1 | 1.4880 |
| 8 | `You` 2523 | 0.999363 | 0.998624 | 0.998467 | 3.1133 | 1 | 0.8040 |
| 19 | `\|` 91 | 0.996285 | 0.979056 | 0.987746 | 4.0983 | 1 | 1.9856 |
| 20 | `im` 316 | 0.997323 | 0.990532 | 0.991869 | 0.7554 | 2 | 2.0028 |
| 21 | `_end` 6018 | 0.991951 | 0.989973 | 0.986064 | 5.8160 | 1 | 1.6351 |

## 5. Verdict: only what the data supports

1. **Thread confound: eliminated.** apr's bytes are identical at 1, 8, and 32 pool workers, and the pin was proven through `/proc` thread counts and CPU%.
2. **No defect candidate.** Over 468 positions and 5 prompts, all 12 persistent flips occur where llama's own per-token
   top-2 gap is **below 1.0**. Every one sits under the pooled median gap of 1.148219: the largest, 0.945776, is at the 45th percentile.
   At 10 of 12, apr ranks llama's argmax 2nd.
3. **28 and 73 do not stand out among the persistent flips.** Their gaps (0.314670, 0.245474) rank 9th and 6th of 12, at the 21st and 16th percentiles
   of all positions. Other prompts produce persistent flips at both smaller and larger gaps (0.030 … 0.946). One pattern
   holds across all 5 prompts: persistent flips occur only at gaps below the median, and none reaches 1.0.
4. **apr's divergence is larger than llama's own two-mode spread on every prompt, measured by cosine.** Min cosine B < min cosine A on 5/5 prompts,
   and cos B < cos A at **316 of 468 positions (67.5%)**. So "apr is off by the same amount llama is off from itself" does
   **not** hold. This is a magnitude excess, not an argmax signal, and it names no cause.
5. **Prompt 4 is an outlier.** Its divergence concentrates on the literal chat-markup tokens: min cosine B **0.961309** at pos 1,
   4 positions below 0.98 in B (1, 2, 4, 19), and max |diff| 4.11. Llama's own A there is 0.986 minimum. The first two positions after `<`
   (`|`, `im`) diverge the most, and the cosine recovers to ≥ 0.998 by pos 8. The three non-chat new prompts have B ≥ 0.99537.

**Cause for 28/73: [U].** The prompt variation fits "argmax flips confined to sub-median gaps". It cannot
separate kernel-accumulation noise from a small systematic forward bias that surfaces only at sub-median gaps. The cosine excess (item 4) and the
prompt 4 markup outlier (item 5) argue that apr's deviation is not just llama's own noise level, but they do not locate it.
**The next discriminating measurement** is a layer-wise hidden-state diff, apr vs llama per-token, at **prompt 4 positions 1–2**. That is where
apr-vs-llama divergence is largest and llama's self-spread is smallest relative to it. Find the first layer whose output cosine
drops below llama's own per-token-vs-batched value at that layer, and check whether that layer is a gated-DeltaNet or a full-attention
layer. The mechanism: llama.cpp `llama-eval-callback` (or a `ggml_backend_sched` eval callback in the producer) dumping `l_out-<il>` per
token, set against a realizar per-layer dump of `forward_single_qwen35` for the same ids.

## Remaining unknowns

- [U] Layer at which apr first diverges (command above).
- [U] True special-token chat prompt. It needs a producer that accepts ids or `parse_special=true`.
- [U] Determinism of the new-prompt outputs at n > 1: rerun `run_variation.sh` phase P and `cmp`.
- [U] llama.cpp at other `-t`. Every llama run here is `-t 8`.
- [U] Decode steps past the prompt, and longer contexts (here ≤ 117 positions).
- [U] Whether the prompt 4 early-position excess comes from the token content (`<|` fragments right after BOS-less start) or from position (pos 1–2).
  Discriminator: move the same markup to mid-prompt.
