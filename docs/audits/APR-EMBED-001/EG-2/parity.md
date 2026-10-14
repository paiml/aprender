# EG-2 forward parity: provisional receipt (2026-10-10)

**Result: PASS (provisional): 840 of 840 I-8 cells at or above 0.98, 0 token mismatches; I-4 and I-6 PASS; every plant RED**

| Item | Value |
|------|-------|
| apr | release build of `feat/apr-embed-001-eg1` @ `f112cf6bc7` plus the EG-2 changes of this commit, sha256 `3979204bbe364bfa9c16252bffeaabd286cf7c1968294291247b20fd35e1d1c1` |
| Oracle | llama.cpp b11490 (`9c2e0e491`) `llama-embedding --embd-normalize 2`, CPU build, sha256 `cc4d0223cedccf489e245f4505b9bdeaae4ed807105964ca16881e46bf177709`. **PROVISIONAL** (§7) |
| Pinned oracle | `scripts/llama_pin.toml` build_commit `d1d3c3396` refuses the file: `unknown model architecture: 'gemma-embedding2'`. Moving the pin is D-3, a §7 pin-change STOP, so it was not moved |
| Files | unsloth BF16, F16, Q8_0, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL (6 of 6) |
| Prompts | the 7 query prompts of `corpus/prompts.tsv`, applied by `apr embed --task <name>` and by template for the oracle |
| Lengths | the 5 texts of `corpus/lengths.tsv`: 8, 128, 1024, 2000 and 8000 tokens. The local window is 1024 (512 each side), so the last three are the ones where the local and global masks differ |
| Dimensions | 768, 512, 256, 128: the cosine of the first d components of each vector. A cosine does not see scale, so this is the cosine of the re-normalised prefixes |
| Cells | 6 files × 7 prompts × 5 lengths × 4 dimensions = 840 |
| Threads | 4 for both, on one CPU host |

**Which tree the binaries came from.** Every binary, test and lint below came from one copy of the tree. Before
the first build, all 28,052 files of the branch were compared with that copy by checksum: the committed tree, plus
the EG-2 files this commit adds. They matched, except for the two EG-1 receipt files, which nothing builds. The
comparison was repeated before the last steps, after the plant patch had been reverted. The parity cells ran on
one release build, `apr 0.70.2 (f112cf6bc7-eg2fix)`, sha256 `3979204bbe364bfa9c16252bffeaabd286cf7c1968294291247b20fd35e1d1c1`.

**One move after the runs.** The commit hook refused `attention.rs` for the complexity of a function this
branch does not touch (`tiled_single_head_attention`). The windowed attention this branch added there moved,
unchanged, into its own file, `attention_windowed.rs`, and `attention.rs` is now as it was before the branch.
The moved tree was compared with the commit by checksum (28,055 files, all equal but the four receipt files,
which nothing builds). On it, the release build, the serve tests (1275 passed, 0 failed, 8 ignored) and both
serve clippy steps were run again and passed. Its `apr` (sha256
`18c6c38023b768e8610343135984bf70d8ed29989db8a88680901e210ae84606`) gave vectors bit-equal to the parity
build's on all 5 lengths, for BF16 with the classification prompt and UD-Q4_K_XL with the search-query prompt.
The other cells were not run again.

A first run of the cells, on a build of the forward before the shape fix (finding 8), is void. That build refused
every real file at load. Its records are kept, and none of its numbers are used here. The first oracle side of
the BF16 cells is void too: it ran before the oracle was given `--no-escape` (finding 9). The BF16 oracle cells
were run again with the flag against the same `apr` vectors, and only that second run is used.

## I-8: `apr` against the oracle

Each row is one file at one dimension, over 7 prompts × 5 lengths = 35 cells. A token mismatch is a cell where
`apr`'s token count differs from the oracle's `number of tokens in prompt`. The last column counts cells where
the prompt the oracle logged is not, byte for byte, the prompt it was given (finding 9). Every cell is in `i8-cells.tsv` beside this file.

| File | Dim | Cells | Min cosine | Max \|Δ\| | Token mismatches | Oracle width ≠ 768 | Oracle prompt ≠ given |
|------|-----|-------|------------|-----------|------------------|--------------------|-----------------------|
| BF16 | 768 | 35 | 0.9999902 | 0.0005569 | 0 | 0 | 0 |
| BF16 | 512 | 35 | 0.9999901 | 0.0006422 | 0 | 0 | 0 |
| BF16 | 256 | 35 | 0.9999909 | 0.0007242 | 0 | 0 | 0 |
| BF16 | 128 | 35 | 0.9999942 | 0.0008267 | 0 | 0 | 0 |
| F16 | 768 | 35 | 0.9999994 | 0.0001378 | 0 | 0 | 0 |
| F16 | 512 | 35 | 0.9999994 | 0.0001607 | 0 | 0 | 0 |
| F16 | 256 | 35 | 0.9999995 | 0.0002031 | 0 | 0 | 0 |
| F16 | 128 | 35 | 0.9999996 | 0.0002241 | 0 | 0 | 0 |
| Q8_0 | 768 | 35 | 0.9997716 | 0.0027673 | 0 | 0 | 0 |
| Q8_0 | 512 | 35 | 0.9997802 | 0.0031852 | 0 | 0 | 0 |
| Q8_0 | 256 | 35 | 0.9997934 | 0.0039852 | 0 | 0 | 0 |
| Q8_0 | 128 | 35 | 0.9998300 | 0.0045931 | 0 | 0 | 0 |
| UD-Q6_K_XL | 768 | 35 | 0.9992352 | 0.0045409 | 0 | 0 | 0 |
| UD-Q6_K_XL | 512 | 35 | 0.9992453 | 0.0052333 | 0 | 0 | 0 |
| UD-Q6_K_XL | 256 | 35 | 0.9993490 | 0.0063002 | 0 | 0 | 0 |
| UD-Q6_K_XL | 128 | 35 | 0.9995339 | 0.0073799 | 0 | 0 | 0 |
| UD-Q5_K_XL | 768 | 35 | 0.9988872 | 0.0050412 | 0 | 0 | 0 |
| UD-Q5_K_XL | 512 | 35 | 0.9988615 | 0.0058222 | 0 | 0 | 0 |
| UD-Q5_K_XL | 256 | 35 | 0.9989918 | 0.0072789 | 0 | 0 | 0 |
| UD-Q5_K_XL | 128 | 35 | 0.9993532 | 0.0081475 | 0 | 0 | 0 |
| UD-Q4_K_XL | 768 | 35 | 0.9992868 | 0.0050547 | 0 | 0 | 0 |
| UD-Q4_K_XL | 512 | 35 | 0.9992605 | 0.0058557 | 0 | 0 | 0 |
| UD-Q4_K_XL | 256 | 35 | 0.9993366 | 0.0073728 | 0 | 0 | 0 |
| UD-Q4_K_XL | 128 | 35 | 0.9995503 | 0.0086412 | 0 | 0 | 0 |

## The threshold, from known-good and known-bad pairs

| Pair | n | Min cosine | Max cosine |
|------|---|------------|------------|
| oracle vs itself (noise floor), BF16, SearchQuery | 5 lengths | 1.0000000 | 1.0000000 |
| `apr` vs oracle (known-good), all files | 840 cells | 0.9988615 | — |
| `apr` vs oracle (known-good), planted binary with no plant (`none`), BF16 | 5 lengths | 0.9999971 | 0.9999993 |
| known-bad: the document form in place of the query prompt | 5 lengths | 0.9630706 | 0.9946302 |
| known-bad: a causal mask (`d-causal`) | 5 lengths | 0.8028001 | 0.9677693 |
| known-bad: the SiLU gate in place of GELU (`e-silu`) | 5 lengths | 0.9391608 | 0.9755112 |
| known-bad: the local-to-global pattern shifted by one layer (`f-shift`) | 5 lengths | 0.9553109 | 0.9993064 |

The known-bad rows by length, at 768:

| Pair | 8 | 128 | 1024 | 2000 | 8000 |
|------|---|-----|------|------|------|
| document form | 0.9699312 | 0.9630706 | 0.9844369 | 0.9837832 | 0.9946302 |
| `d-causal` | 0.8028001 | 0.8525408 | 0.9119064 | 0.9308720 | 0.9677693 |
| `e-silu` | 0.9755112 | 0.9588616 | 0.9391608 | 0.9408457 | 0.9481853 |
| `f-shift` | 0.9993064 | 0.9953435 | 0.9722698 | 0.9553109 | 0.9750457 |

All 840 known-good cells are at or above 0.98, so I-8 passes at G11's floor. The two ranges overlap,
though (finding 10). The lowest known-good cell is 0.9988615 (UD-Q5_K_XL, FactChecking, 8 tokens, 512), and
the highest known-bad cell is 0.9993064 (`f-shift`, 8 tokens). 12 of the 840 known-good cells sit below that
known-bad maximum: 2 on UD-Q6_K_XL, 6 on UD-Q5_K_XL and 4 on UD-Q4_K_XL. On BF16, F16 and Q8_0 none do: their
lowest cell is 0.9997716. So no one floor separates known-good from known-bad across all six files. The
known-bad pairs were run on BF16 only, so a floor for a quantized file needs known-bad pairs on that file.

The receipt line in the spec's form (§8):

```text
EG-2 | PASS (provisional: oracle b11490 9c2e0e491, not the pin) | files 6/6 (BF16 F16 Q8_0 UD-Q6_K_XL UD-Q5_K_XL UD-Q4_K_XL)
parity: I-8 840/840 cells >= 0.98 (G11), min 0.9988615, token mismatches 0, oracle width != 768 0, oracle prompt != given 0; GPU vs CPU not_measured
invar: I-4 7/7 PASS, I-6 2/2 PASS (BF16, CLI); plants 7/7 as expected (none green, 6 RED); host thresholds.yaml entry not_measured (D6)
```

## I-4 and I-6 at the CLI (BF16)

| Cell | Case | Value | Verdict |
|------|------|-------|---------|
| I-4 | default: header dim, then each vector length | `768 768,768` | PASS |
| I-4 | `--dim 512` text 0: len 512, norm 1.0000000, max\|d\| vs re-normalised prefix | 1.02e-08 | PASS |
| I-4 | `--dim 512` text 1: len 512, norm 1.0000000, max\|d\| vs re-normalised prefix | 7.06e-09 | PASS |
| I-4 | `--dim 256` text 0: len 256, norm 1.0000000, max\|d\| vs re-normalised prefix | 8.59e-09 | PASS |
| I-4 | `--dim 256` text 1: len 256, norm 1.0000000, max\|d\| vs re-normalised prefix | 1.08e-08 | PASS |
| I-4 | `--dim 128` text 0: len 128, norm 1.0000000, max\|d\| vs re-normalised prefix | 8.77e-09 | PASS |
| I-4 | `--dim 128` text 1: len 128, norm 1.0000000, max\|d\| vs re-normalised prefix | 9.63e-09 | PASS |
| I-6 | forward vs reversed batch of 8, bit-equal | 8/8 | PASS |
| I-6 | text 3 alone vs at position 3, bit-equal (max\|d\|) | 0 | PASS |

The oracle calls of this step, for the noise floor and the document form, all logged the prompt they were given
(10 of 10).

## The plants, observed RED (§9)

One planted `apr` (sha256 `2eee6a0865bda30f8cd254b83bbcece78b3f33da9b391aec74d360208b83b8ab`) holds every plant,
chosen at run time by `APR_EG2_PLANT`. It was built from the same tree
with a patch that was reverted after the build. The three patched files were then compared with their pre-patch
checksums, byte for byte. The plants were never committed. The `none` row is the same planted binary with no
plant selected, and it must be green. Its I-8 cells first ran before the oracle had `--no-escape` (finding 9).
They were run again with the flag, and only that second run is used.

| Plant | Invariant | What it does | Cell that went RED | Value |
|-------|-----------|--------------|--------------------|-------|
| `none` | all | no plant selected | none: the 9 CLI checks PASS, and I-8 is 0.9999971 to 0.9999993 over the 5 lengths | green |
| `a-trunk` | I-4 | `--dim 512` answers the pooled trunk state | `--dim 512`, texts 0 and 1: max\|d\| vs the re-normalised prefix | 0.3755542, 0.3328974 |
| `b-norm` | I-4 | the truncated prefix is not re-normalised | all 6 `--dim` rows: the norm | 0.578 to 0.867 |
| `c-pad` | I-6 | every text is padded with token 0 to the batch's longest, and pooled over it | text 3 alone vs at position 3 | max\|d\| 0.0182079 |
| `d-causal` | I-8 | a causal mask on top of the window | all 5 lengths | 0.8028001 to 0.9677693 |
| `e-silu` | I-8 | the SiLU gate in place of GELU | all 5 lengths | 0.9391608 to 0.9755112 |
| `f-shift` | I-8 | each layer takes its window and RoPE base from the layer before | 1024, 2000 and 8000 | 0.9722698, 0.9553109, 0.9750457 |

Every plant went RED in at least one cell, and the oracle logged the prompt it was given in all 20 of the plants'
oracle calls. Two plants are not RED everywhere, which is why each check exists:

- `c-pad` passes the reversed-batch check (8/8 bit-equal): padding every text to the same longest text is
  symmetric under reversal. Only the check of one text alone against the same text inside the batch sees it.
- `f-shift` stays above the floor at 8 and 128 tokens (0.9993064, 0.9953435). On a text shorter than the half
  window of 512, the local and global masks are the same, so only the RoPE base moves, and at short range that
  moves the vector little. The lengths of 1024 tokens and over are the ones that catch it.

## Tests and lints

Each step ran with `-j4` on one CPU host, on the copy of the tree described above.

| Step | Command | Result |
|------|---------|--------|
| release build | `cargo build --release --bin apr` | exit 0 |
| core tests | `cargo test -p aprender-core --lib -- test_4900 test_3601 test_3656 test_3947` | 16 passed, 0 failed |
| core clippy | `cargo clippy -p aprender-core --lib -- -D warnings` | exit 0 |
| serve tests | `cargo test -p aprender-serve --lib -- eg2 matmul get_tensor_f32 dequant` | 1275 passed, 0 failed, 8 ignored |
| serve clippy | `cargo clippy -p aprender-serve --lib -- -D warnings` | exit 0 |
| serve clippy, test profile | `cargo clippy -p aprender-serve --lib --profile test -- -D warnings` | exit 0 |
| cli tests | `cargo test -p apr-cli --lib -- embed model_kind` (includes the BERT `apr embed` tests) | 178 passed, 0 failed |
| cli clippy | `cargo clippy -p apr-cli --lib -- -D warnings` | exit 0 |

The 8 ignored serve tests carry `#[ignore]` in files this branch does not touch (`apr/apr_tests*.rs`,
`gguf/loader`, `quantize/tests`): three dtype reads, one real-model load, and four timing tests. They did not
run, so they are not measured here. 54 of the passed serve tests are this branch's `eg2` tests.

Earlier runs on older copies of the tree failed on this branch's own code, and each failure was fixed before the
run above. The first serve run had 1271 passed and 2 failed, both in this branch's own tests:

- `every_tensor_reaches_the_embedding_eg2` stopped at the first tensor that did not move the output, and reported
  no number. It now collects every such tensor with its cosine, so one run shows all of them. It passes in the run above, after the fixture fix of finding 7.
- `truncation_keeps_the_prefix_direction_at_unit_length_eg2` compared the full-width truncation with the input
  exactly. At full width, the vector is scaled by 1 / |v|, which is 1 only to within rounding. The test now allows
  a change of at most 1e-6 per component.

The same run's clippy reported three `needless_for_each` findings in `forward_embedding_gemma2.rs`. Each closure
became a `for` loop (`*v *= inv` in both forms). Findings 7 to 10 below came from later runs.

## Findings

1. **The census FFN pattern cannot match a real call.** The first census command in
   `docs/specifications/EPIC-0.74-any-model-plan.md` §6 (one of the two D8 uses) counts FFN calls with
   `ffn_block::|adaptive_ffn::`. Those two files are pulled into `forward/results.rs` with `include!`, so no module of
   either name exists and no call can carry the prefix; the only match in the crate is a doc comment. Its FFN count is
   0 for every forward file, whatever the file calls.
2. **The contract's formulas were wrong in two places, fixed in `embeddinggemma2-forward-v1.yaml` v1.1.0.** The
   score scale is 1.0, not 1/√d_head, as in the reference forward. The local window is centred: |i − j| ≤ ⌊W/2⌋,
   not a one-sided window of W. The unit test checks the boundary pair ⌊W/2⌋ (inside) and ⌊W/2⌋ + 1 (outside).
3. **The existing Gemma FFN and dispatch cover Gemma 1 and 2 only** (`geglu_ffn`, `gemma_dispatch`), so the
   EmbeddingGemma 2 forward has its own GELU-gated FFN.
4. **A BERT APR file without `--vocab` now exits 5, not 2.** `--vocab` was a required flag, so clap refused its
   absence with a usage error (exit 2). A GGUF file carries its own tokenizer, so the flag is optional now, and the
   APR path refuses its absence itself as a validation failure (exit 5) with the reason in the message.
5. **The first I-4 check of the default length could not fail on a short vector.** It read the header's `dim`
   field only. It now also reads each vector's length, and passes only on `768 768,768`.
6. **Clippy under `--profile test` is not clean on two crates, for code this branch does not touch.** On
   `aprender-core` it reported 20 findings: 19 are in files this branch does not touch, so `main` has them too and
   the step cannot be a merge gate. The one in a file this branch touches (`verbose_bit_mask` in the BF16 test) is
   fixed in de03bbb402. On `apr-cli` it reported two `unused_mut` in `nf4_classifier.rs`, which this branch does not
   touch. For clippy on the lib targets, see the tests section.
7. **The fixture's weights were rank 2, so the reach test could not see a weight.** The fixture filled each weight
   with `sin(a·i + b)` along its flattened index. For a matrix that is `sin(A_row + B_col)`, which is rank 2. With
   every weight rank 2, the fixture's embeddings all lay on one circle, and a perturbed weight could move the vector
   by less than the cosine can resolve. The fixture now fills weights from an integer hash of the index, and
   `the_fixture_embeds_beyond_a_plane_eg2` requires three embeddings to have a Gram determinant above 1e-4.
8. **The forward read every weight's shape the wrong way round, and its fixture agreed with it.** The GGUF reader
   in `aprender-serve` reverses GGUF's `ne0, ne1`, so it holds a weight as `[out, in]`. The generative config reads
   shapes this way (C-13). The EG-2 forward read `[in, out]`, and the test fixture wrote its shapes the same wrong
   way, so every unit test passed. Only the real file showed it. Each `attn_q` read back with an output width of
   512, the hidden width, so the forward found no local/global split and refused the file at load. That voided the
   first run of the cells. Now the forward reads `[out, in]` and the fixture writes shapes as a real file holds them.
   `the_fixture_holds_shapes_as_a_real_file_does_eg2` checks the fixture against the reader. On the real file,
   `attn_q` is 1024 wide on the local layers and 2048 wide on the global layers (5, 11, 17, 23), so the split by
   width holds.
9. **The oracle turned escape sequences in the text into bytes, so it embedded a different text from `apr`.**
   `llama-embedding` reads `\x2c`, `\\`, `\r` and the like in `-p` as the bytes they name unless it is given
   `--no-escape`. The 8000-token text quotes a tokenizer's escape table and holds seven backslashes. The other four
   texts and the seven prompt templates hold none. On BF16 with the query prompt, `apr` counted 11402 tokens for that
   text and the oracle 11393. The cosine was still 0.99990 at 768, well above the floor, so the cosine alone would
   have passed a comparison of two different texts. The token count is what showed it. The prompt the oracle logged
   first differed from the one it was given at the first `\x2c`. The oracle now runs with `--no-escape`. Each cell
   also checks the prompt the oracle logs against the prompt it was given (the last column of the I-8 table). One
   short text run both ways shows the flag's effect. With escapes on, `a\x2cb \\ c` became 7 tokens and the oracle
   logged `a,b \ c`. With `--no-escape` it became 9 tokens and the oracle logged the text as written.
10. **At the 0.98 floor, 5 of the 20 known-bad I-8 cells pass.** The document form passes at 1024, 2000 and 8000
    tokens (0.9844369, 0.9837832, 0.9946302): the prompt is a few tokens at the head of a long text, and pooling
    over the whole text dilutes it. `f-shift` passes at 8 and 128 tokens (see the plants). Each known-bad pair
    still fails at some length, so the five lengths together catch all four, but no one length does. The
    known-good and known-bad ranges overlap. The lowest known-good cell is 0.9988615, and the highest known-bad
    cell is 0.9993064. 12 of the 840 known-good cells sit below it, all on the K-quant files (UD-Q6_K_XL 2,
    UD-Q5_K_XL 6, UD-Q4_K_XL 4), and all on the 8-token text, at 256 to 768 dimensions. On BF16, F16 and Q8_0 the lowest
    known-good cell is 0.9997716, above every known-bad cell. So no one floor separates the two across all six
    files: a floor would need known-bad pairs measured on each quantized file (D6). This receipt does not move
    G11: that is a gate change.

## Why this is not the gate receipt

- The oracle is b11490, not the pinned commit, which refuses the architecture. The same files, prompts, lengths and
  dimensions become the gate receipt once a pin that loads `gemma-embedding2` is approved (D-3).
- The spec sets the threshold by the thresholds file's own method, on each required host. This is one CPU host, so
  `evidence/parity/thresholds.yaml` gets no `models:` entry from it (D6). Until it has one, G11 holds the cosine
  floor at 0.98, fail-closed.
- GPU against CPU is not measured: both sides ran on the CPU.
