# APR-EMBED-001 EG-5 (#4904): named refusals and the `apr qa` golden-embedding gate

**Status: refusals and `apr qa` measured, 8 of 8 files. The ladder half of EG-5 is not in this change.**

| Item | Value |
|------|-------|
| Build | `apr` release, default features (no `cuda`), from this branch |
| Files | unsloth UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, Q8_0, BF16, F16; ggml-org BF16, Q8_0 |
| Unit tests | `cargo test -p apr-cli --lib -- eg5 model_kind_gate rerank gate_display gate_registry golden_embedding`: 33 passed |
| Lint | `cargo clippy -p apr-cli --lib --profile test -- -D warnings`: clean on the lines this change touches |

## The five refusals (I-3)

Each verb run on each of the 8 files exits 6 and names `apr embed`: 32 of 32 cells for the four verbs below,
plus `chat` (refused since EG-1) on UD-Q4_K_XL with the same build.

| Verb | Where it refuses | Message, UD-Q4_K_XL |
|------|------------------|---------------------|
| `run` | `run_model`, from the header, before any backend | `'gemma-embedding2' is an embedding model (ModelKind::Embedding): it has no language-model head, so `apr run` cannot serve it. Use `apr embed`.` |
| `chat` | EG-1, the chat session's load | same form, `apr chat` |
| `code` | `dispatch_code_command`, before a serve child is spawned | same form, `apr code` |
| `showcase` | `dispatch_showcase`, on the model file `--model-dir` names (a file, or the tier's file in a directory) | same form, `apr showcase` |
| `rerank` | first line of `rerank::run`, before the vocabulary or the APR magic | `... `apr rerank` takes a BERT cross-encoder APR file, not GGUF. Use `apr embed`.` |

Before EG-5, `run` refused only inside the load (exit 8), `rerank` failed on the vocabulary or the APR magic
and named no verb, and `showcase` ran its Qwen demo whatever `--model-dir` held.

The header is read without the tensor data: the prefix grows from 16 MiB by doubling to a 256 MiB cap (the
`model_header` reader, now shared by both readers). A generative file, a missing file and a non-GGUF file pass
through to the loaders: `apr run` on Qwen3.5-0.8B-Q4_K_M answers as before.

## `apr qa` on an embedding model

`golden_embedding` embeds three golden triples (anchor, paraphrase, unrelated) with the sentence-similarity
prompt, through the call `apr embed` and `apr serve` make. It checks the width the file declares, finite
values, a norm within 1e-3 of 1, a bit-equal repeat, and that on every triple the paraphrase is nearer the
anchor than the unrelated sentence. `golden_output`, throughput, Ollama parity, GPU speedup, format parity,
PTX parity and GPU state isolation skip by kind: an embedding model generates no text. On a generative model
`golden_embedding` skips by kind. Control, Qwen3.5-0.8B-Q4_K_M: `golden_embedding` skipped ("Not an embedding
model"), `golden_output` ran on the unchanged path and failed with its existing 0.8B thinking-on result (think
block unclosed within 2048 tokens; the gate names Qwen3.5-2B/4B as the certified models), so that `apr qa` exit
5 is not from this change.

| File | `apr qa` exit | golden_embedding | Min gap | Time |
|------|---------------|------------------|---------|------|
| UD-Q4_K_XL | 0 | PASS | 0.1548 | 4.5 s |
| UD-Q5_K_XL | 0 | PASS | 0.1530 | 13.8 s |
| UD-Q6_K_XL | 0 | PASS | 0.1517 | 6.3 s |
| Q8_0 | 0 | PASS | 0.1515 | 4.5 s |
| BF16 | 0 | PASS | 0.1508 | 2.5 s |
| F16 | 0 | PASS | 0.1508 | 3.2 s |
| ggml-org BF16 | 0 | PASS | 0.1508 | 2.2 s |
| ggml-org Q8_0 | 0 | PASS | 0.1515 | 3.3 s |

Min gap is the smallest cos(anchor, paraphrase) − cos(anchor, unrelated) over the three triples. The
threshold is 0: the gate claims an order, not a margin. A margin needs the thresholds file's method (known-good
and known-bad files, n = 5) before it can be a gate. Before EG-5 the same files failed `golden_output` and
`throughput` on the kind refusal, so `apr qa` exited non-zero on every embedding file.

## Not in this change: the ladder

The ladder contract, `scripts/model_ladder.sh` and `scripts/check_model_ladder.sh` judge every rung and every
inventory file on `run`, `chat`, `code` and `serve` (`/api/chat` required) and on `golden_output`. The required
hosts do not hold these files today. Order matters: the inventory pattern `*q4_k*.gguf` matches
`embeddinggemma-2-UD-Q4_K_XL.gguf`, so a file placed in a host's model directory before the harness is
kind-aware would be judged on the generative verbs and turn the host red. The kind-aware harness lands first,
then the files, then one sweep per required host.
