# CRUX first-run UX analysis (2026-11-27)

**Status:** analysis, measured 2026-11-27. It changes no code, contract or gate.
**Scope:** the first ten minutes of a new user with the released `apr` binary, compared with Ollama, llama.cpp, LM Studio, mistral.rs, vLLM and mlx-lm.
**Feeds:** `docs/roadmaps/releases/0.78.0-first-run.md` (its rows) and §0 of `docs/roadmaps/releases/0.77.0-serve-parity.md`.

The existing CRUX work (`crux-competitive-research-ux-workflows.md`, #6730) asks whether `apr` has a competitor's *feature*. This document asks a different question: what happens when someone who has never used `apr` follows the README. The two do not overlap; where a finding here is already a row in #6730 or #6570 it is linked there, not repeated.

## 1. Method

| Subject | Version | How it was obtained |
|---|---|---|
| `apr` | 0.75.3 (982ebfd36), then 0.75.4 (d44c64523) | `apr --version`, the installed release binary. It was updated during the session; every finding was re-run on 0.75.4 and is unchanged (`recheck-0754.txt`, `routes3.txt`, `routes4-0754.txt` in the review packet) |
| Source tree | `dd6e459ac` (`origin/main`) | `git rev-parse HEAD` |
| Ollama | 0.40.18 | `ollama --version` |
| llama.cpp | 1.1.0-dev, build 12808 (ed2153e70) | `llama-cli --version` |
| Host | x86_64, 24 threads, RTX 4090 (24 GB), CUDA | `apr devices` |

Rules followed:

- Every `apr` finding was produced by running the installed binary. The command is given in §6.
- "From nothing" means a fresh `HOME` and `XDG_CACHE_HOME`, so no earlier download can help.
- stdout and stderr were captured separately, with stdin closed.
- Competitor behaviour that was run on this host is marked **measured**. Behaviour taken from the vendor's documentation is marked **cited**, with the URL in §7. LM Studio, mistral.rs, vLLM and mlx-lm are not installed here, so every cell for them is cited.
- A cell with no receipt is written `not_measured`.

## 2. Findings

Severity is this document's proposal, on the repo's P0 to P3 scale.

### F-01 (P0) The first command a new user is likely to try fails

In a fresh `HOME`, `apr run hf://<safetensors repo> "<prompt>"` downloads `model.safetensors`, does not download `tokenizer.json`, and exits 8.

```text
$ apr run hf://Qwen/Qwen2.5-Coder-0.5B-Instruct "What is 2+2?"
Downloading hf://Qwen/Qwen2.5-Coder-0.5B-Instruct...
  Download complete!
[PMAT-172] ERROR: No tokenizer found for ~/.apr/cache/hf/Qwen/Qwen2.5-Coder-0.5B-Instruct/model.safetensors.
error: Inference failed: Inference error: Tokenizer encode failed for SafeTensors model (no tokenizer.json sibling?)
exit=8
```

- It reproduces on two repos: `Qwen/Qwen2.5-Coder-0.5B-Instruct`, which is the README's own quick-start model, and `HuggingFaceTB/SmolLM2-135M-Instruct`.
- The same command passes on a machine that has already run `apr pull` for that model, because `run` then finds the tokenizer in the `pull` cache. That is why a developer machine does not show the failure.
- GGUF sources pass from nothing: `apr run hf://Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF "What is 2+2?"` replies `2 + 2 equals 4.` in 27.7 s including the download.

### F-02 (P1) `run` and `pull` keep separate caches, and `list` sees only one

- `apr run hf://…` stores files under `~/.apr/cache/hf/<org>/<repo>/`.
- `apr pull hf://…` stores files under `~/.cache/pacha/models/<hash>.<ext>`.
- After a successful `apr run` from nothing, `apr list` prints `No cached models found.` while the downloaded model file sits on disk.
- Whether `apr rm` can remove a model that `run` downloaded: `not_measured`.

### F-03 (P1) Pulled models have no name

- `apr list` names every model by a 16-character hash and marks every row `(orphan)`. That includes a model pulled seconds earlier in a fresh `HOME`.
- #2408 reported this against 0.63.0 and is closed. It still reproduces on 0.75.3.
- `apr pull` ends by telling the user to run `apr run <absolute path to the hash file>`. The README quick start (L44-46) does the same.
- Name resolution does exist: `apr run Qwen/Qwen2.5-Coder-0.5B-Instruct "hi"` and `apr run hf://…` both work on a machine with the model pulled. Neither `pull` nor the README mentions it.
- `apr pull` also prints two `apr convert <absolute path to the hash file>` lines, and a warning `⚠ <hash>.tokenizer.model (not found in repo)` for a model whose `tokenizer.json` it has just downloaded.
- `apr registry aliases` lists 11 short names. One is a Qwen3 model (`qwen3-coder`, a 30B GGUF); none is a Qwen3.5 model or a model small enough for a first run on a laptop.

### F-04 (P1) Replies are cut off after 32 tokens by default

`apr run` defaults to `--max-tokens 32`. Asked to explain quicksort in three sentences, it stops partway through the first:

```text
Quicksort is a divide-and-conquer algorithm that works by selecting a 'pivot' element from the array and partitioning the other elements into two sub-arrays
```

Ollama, with the same model and prompt on this host, returned all three sentences (53 tokens) and stopped at end of turn (**measured**). `apr chat` defaults to 512.

### F-05 (P2) `pull` prints a next step that is not a valid command

`apr pull` prints `apr serve <path>` under "Usage". `apr serve <path>` fails with `unrecognized subcommand`; the working form is `apr serve run <path>`.

### F-06 (P2) Output carries internal debugging text

- A failing `apr run` writes the `=== APR Run ===` banner and the `Source:` line to stdout (39 bytes for the path first tried, 45 for `/no/such/model.gguf`) before the error. This is the nightly crux probe C2.
- A successful `apr run` on a GGUF model writes lines such as `[trueno#243] ✓ Manual graph: 518 kernels. first_args=Some(["0x7b59ed7c0e00", …` to stderr, with raw pointer values. Other lines carry ticket tags: `[GH-189]`, `[PMAT-053]`, `[PREFILL-PATH]`.
- The timing line reads `Completed in 4.79s (downloaded)` for a run that used a cached model, and `Completed in 22.27s (cached)` for a run that had just downloaded it. The two labels appear to be swapped.
- A successful `apr run` wraps the reply on stdout in `=== APR Run ===`, `Source:`, `Output:` and `Completed in …`. So `$(apr run …)` does not yield the reply alone. `crux-C-01-v1` says "decoded text lives on stdout" and puts telemetry on stderr.
- A failing `apr run` prints `[PMAT-172] ERROR:` on stderr.
- `apr serve run` prints `(PAR-111)` and `(PMAT-044)` on stdout and the same `[trueno#243]` pointer lines on stderr.
- No ANSI escape reaches a redirected stream (probe C3 passes).

### F-07 (P2) `apr run` does not say which backend it used

- `apr serve run` does this well. It prints `gpu-layers: requested=all resolved=24 total=24 (backend=cuda)`, and `/health` returns `"compute_mode":"gpu"`.
- `apr run` prints no backend line on stdout or stderr at default verbosity.
- `apr devices` on 0.75.3 still says "no native Metal backend in 0.66" and "no HIP backend in 0.66", and its `selected:` line ends in an internal note (`basis=[U] default until master row 6 measures vram_peak`).
- `apr devices` and `apr capability` already answer "what can this build use, and why not". A user will not find them: `apr doctor` returns `unrecognized subcommand` and suggests `monitor`.
- The nightly models-crux receipt lists "apr prints no TTFT" under `not_covered[]` (cited from that receipt; it is not in the review packet).
- This matters because of #7285 (P0): a silent CPU fallback cost two orders of magnitude in speed and nothing told the user.

### F-08 (P2) The help listing is hard for a newcomer to use

- `apr --help` prints 115 entries in one list with no grouping.
- 32 of them are `*-lint` verbs, and several say "no apr producer yet".
- 52 help lines contain an internal ticket identifier (`GH-243`, `PMAT-741`, `CRUX-K-11`).
- `apr run` calls its argument `<SOURCE>` and accepts `hf://`; `apr chat` calls it `<FILE>` and describes it as "Path to .apr model file".
- The typo hint is weak: `apr nonexistent` suggests `finetune`.

### F-09 (P2) Two agent-facing routes return 404 on the released binary

`apr serve run` (0.75.3, and again on 0.75.4) answers these correctly: `/health`, `/metrics`, `/v1/models`, `/v1/chat/completions` (plain, streaming, `json_schema`), `/v1/completions`, `/api/chat`, `/api/tags`. A request with `tools` returns 200, but on the 0.5B model the call arrives as text inside `content`, with no `tool_calls` array. `/api/tags` answers, but reports `Q5_0` for a `q4_k_m` file.

It returns 404 for:

- `POST /v1/responses`. #6652 and #6653 (the tickets for this route) are closed as not planned.
- `POST /v1/chat/completions` and `POST /v1/completions` with no `model` field return 422, `Invalid request body. Check that the JSON structure matches the expected schema.` The message names no field. With `model` present both return 200.
- `POST /v1/messages` (the Anthropic Messages shape).

Why the routes are absent from this binary was not determined. They may sit behind a flag, in a different serve mode, or have landed after 0.75.3.

APR-EPIC-001 assigned `/v1/messages` conformance to E5 "Agent Ready" (#4000), which is closed. By the operator ruling of 2026-11-27 both routes are rows G-01 and G-02 of the 0.78.0 plan; embeddings and `/api/tags` go to 0.77 (SP-13, SP-14).

The start-up banner's `Endpoints:` list shows five routes and omits `/v1/models`, `/api/chat` and `/api/tags`, which all answer.

Two smaller observations, both already inside #6570's scope and so not new rows:

- `POST /v1/embeddings` returns 503 with "needs a loaded model" while a model is loaded (#6570 C-13).
- `/api/tags` reports `quantization_level: Q5_0` for a file named `…q4_k_m.gguf`.

### F-10 (P2) The README and book lag the release

- The library snippet (README L334) imports `aprender::linear_regression::LinearRegression`. `crates/aprender-core/src/` has `linear_model`, and no `linear_regression`.
- The quick start (L44-46, L67, L78, L89) uses a hashed cache path, where `apr run hf://…` exists.
- `book/src/getting-started/installation.md:29` shows `# apr 0.4.17`.
- In `book/src/SUMMARY.md` the Getting Started pages are at L220-223, in alphabetical order, so "Installation" comes after "First Training".
- #7451 and #7326 already track the README's generated counts; #7450 marked the 0.69.1 matrix as history.

### F-11 (P3) Some mutating verbs have no preview

`rm`, `convert`, `import`, `stamp`, `compile`, `shard` and `unshard` advertise no `--dry-run` or `--plan`. This is nightly crux probe C7, first seen 2026-08-30. The Makefile has no `help:` target (probe M1, first seen 2026-08-24).

### F-12 (P3) Install reach

The README (L29) says the install script covers Linux x86_64 and aarch64 only, and (L41) that `cargo install aprender` builds with no GPU backend. Neither was re-tested here.

## 3. What `apr` already does well

These were measured and should be kept as they are:

- `apr serve run` reports the resolved GPU layer count and backend at start-up.
- `json_schema` structured output and streaming work over `/v1/chat/completions` (captured on 0.75.4). Tool calls do not come back as a `tool_calls` array on the 0.5B model (see F-09).
- The server speaks the Ollama API (`/api/chat`, `/api/tags`) as well as the OpenAI one.
- `/health` includes the binary's SHA-256 and compute mode.
- `apr devices` explains every backend, including the reason one is unavailable.
- An unknown route returns the list of valid routes.
- `apr pull --dry-run` and `--verify` exist and are documented in the help.

## 4. Competitor matrix

| Behaviour | apr 0.75.3 | Ollama | llama.cpp | LM Studio | mistral.rs | vLLM | mlx-lm |
|---|---|---|---|---|---|---|---|
| Commands from nothing to a reply | 2 with a GGUF source; **fails** with a safetensors source (F-01) | 2 (cited) | 2 (cited) | 2 (cited) | 2, then a web page (cited) | 3 or more (cited) | 2 (cited) |
| Run by short name or `org/repo` | yes, undocumented (F-03) | yes (measured) | `-hf org/repo[:quant]` (cited) | yes (cited) | yes (cited) | yes (cited) | yes (cited) |
| Listing shows names | no, hashes (F-03) | yes (measured) | not_measured | yes (cited) | not_measured | n/a | n/a |
| Reply runs to end of turn by default | no, 32 tokens (F-04) | yes (measured) | yes (cited) | yes (cited) | not_measured | n/a | not_measured |
| Says which backend ran | `serve` yes; `run` no (F-07) | only in `ollama ps` (cited) | start-up log (cited) | not_measured | `doctor --json` (cited) | start-up log (cited) | n/a |
| Explains why the GPU was not used | `apr devices` yes; at run time no | no (cited: three open issues) | no (cited) | not_measured | `doctor` (cited) | not_measured | n/a |
| OpenAI chat, streaming, tools, JSON schema | chat, streaming and JSON schema yes; a tool call comes back as text in `content` on the 0.5B model (measured) | tools yes, `tool_choice` no (cited) | yes (cited) | yes (cited) | yes (cited) | yes (cited) | chat only (cited) |
| OpenAI Responses | 404 (F-09) | yes, stateless (cited) | yes (cited) | yes (cited) | not_measured | via NIM (cited) | no (cited) |
| Anthropic Messages | 404 (F-09) | yes (cited) | yes (cited) | yes (cited) | yes (cited) | yes (cited) | no (cited) |
| `/health` and Prometheus metrics | yes (measured) | not_measured | yes (cited) | not_measured | yes (cited) | yes (cited) | no (cited) |
| Built-in web chat page | no | no | yes (cited) | desktop app | yes (cited) | no | no |
| Installer beyond Linux | no (README) | yes (cited) | yes (cited) | yes (cited) | yes (cited) | Linux only (cited) | macOS only (cited) |

One measured speed note, for context only: on this host Ollama generated at 421 tok/s on the 0.5B fp16 model, and `apr serve run` on 0.75.4 reported 393 to 432 tok/s (`timings.predicted_per_second`, three 55-token replies) on the 0.5B Q4_K_M file. The files differ, so this is not a parity number; `evidence/parity/LEDGER.md` remains the source for those.

## 5. Where `apr` could lead

No competitor does these well, and `apr` already holds most of the parts:

1. **Tell the user at run time why the GPU is not in use, and what to do.** Every tool surveyed falls back to CPU quietly. `apr devices` already holds the reasons; the work is to print one line of it in `run`, `chat` and `serve`.
2. **A machine-readable diagnosis.** Only mistral.rs has `doctor --json`. `apr devices` plus `apr capability` are most of an `apr doctor`.
3. **A published record of which models pass on which hardware.** No competitor ships one. The README's verified matrix is that record, provided it tracks the current release and covers `chat` and `serve` as well as `run`.
4. **A conformance report for the API.** Each competitor lists unsupported fields in prose. `apr serve` already returns its route list; a per-field report would be new.

## 5a. Where the CRUX registry disagrees with this measurement

`contracts/crux-competitive-research-ux-v1.yaml` and the subspec carry a status per story. Three disagree with the released binary:

| Story | Registry status | Measured |
|---|---|---|
| CRUX-C-01 `apr run` one-shot | supported | fails from nothing on a safetensors source (F-01); stops at 32 tokens (F-04); stdout carries a banner (F-06) |
| CRUX-A-02 pull by `hf://org/name` | supported | the pulled model is listed as a hash marked `(orphan)` (F-03) |
| CRUX-C-04 Ollama `/api/chat` | missing | `POST /api/chat` returns 200 |

The contract verifier does not catch this. `pv lint --gate shapes` passes on this tree (76 shapes, 3,786 focus nodes, 0 armed violations), because the shapes grade the contracts' structure and no shape runs the released binary from an empty cache.

## 6. Commands

Each block reproduces one finding. `A` is the `apr` binary and `H` is an empty directory.

```sh
# F-01, F-02: from nothing
HOME=$H XDG_CACHE_HOME=$H/.cache $A run hf://Qwen/Qwen2.5-Coder-0.5B-Instruct "What is 2+2?"; echo "exit=$?"
HOME=$H XDG_CACHE_HOME=$H/.cache $A list
find $H -type f -size +1M

# F-01 control: a GGUF source passes
HOME=$H2 XDG_CACHE_HOME=$H2/.cache $A run hf://Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF "What is 2+2?"

# F-03, F-05: names and the printed next step
HOME=$H3 XDG_CACHE_HOME=$H3/.cache $A pull hf://HuggingFaceTB/SmolLM2-135M-Instruct
HOME=$H3 XDG_CACHE_HOME=$H3/.cache $A list
$A registry aliases | wc -l
$A serve /no/such.safetensors

# F-04: default length, and the Ollama control
$A run hf://Qwen/Qwen2.5-Coder-0.5B-Instruct "Explain quicksort in three sentences."
ollama run qwen2.5-coder:0.5b-instruct-fp16 "Explain quicksort in three sentences." --verbose

# F-06: streams
$A run /no/such/file >out 2>err </dev/null; echo "exit=$? stdout_bytes=$(wc -c <out)"
grep -c -P '\x1b\[' out err

# F-07
$A devices; $A doctor

# F-08: help counts
$A --help | grep -c -E '^  [a-z]'
$A --help | grep -c -E '^  [a-z0-9-]+-lint '
$A --help | grep -c -E 'GH-[0-9]+|PMAT-[0-9]+|CRUX-[A-Z]-[0-9]+|#[0-9]{3,}'

# F-09: routes (server started with: $A serve run <gguf> --port 18431)
for r in /v1/responses /v1/messages; do curl -s -o /dev/null -w "$r %{http_code}\n" -X POST -H content-type:application/json -d '{}' http://127.0.0.1:18431$r; done

# F-10: README and book, at the source tree's HEAD
grep -n -E 'aprender = |use aprender::' README.md; ls crates/aprender-core/src | grep -i linear
grep -c 'pacha/models/' README.md
sed -n '29p' book/src/getting-started/installation.md
grep -n 'getting-started/' book/src/SUMMARY.md

# F-11: preview flags
for v in rm convert import stamp compile shard unshard; do printf '%s:%s ' $v "$($A $v --help 2>&1 | grep -c -E -- '--dry-run|--plan')"; done
grep -c '^help:' Makefile
```

## 7. Sources for cited cells

- Ollama: <https://docs.ollama.com/api/openai-compatibility>, <https://docs.ollama.com/api/anthropic-compatibility>, <https://docs.ollama.com/gpu>; silent CPU fallback: <https://github.com/ollama/ollama/issues/11849>, <https://github.com/ollama/ollama/issues/10883>, <https://github.com/ollama/ollama/issues/11676>
- llama.cpp: <https://raw.githubusercontent.com/ggml-org/llama.cpp/master/README.md>, <https://raw.githubusercontent.com/ggml-org/llama.cpp/HEAD/tools/server/README.md>, <https://huggingface.co/blog/ggml-org/anthropic-messages-api-in-llamacpp>
- LM Studio: <https://lmstudio.ai/docs/cli>, <https://lmstudio.ai/docs/developer>, <https://lmstudio.ai/docs/advanced/headless>
- mistral.rs: <https://ericlbuehler.github.io/mistral.rs/quickstart/>, <https://ericlbuehler.github.io/mistral.rs/reference/cli/doctor/>
- vLLM: <https://docs.vllm.ai/en/latest/getting_started/quickstart.html>, <https://github.com/vllm-project/vllm/pull/22627>, <https://docs.nvidia.com/nim/large-language-models/2.0.7/reference/api-reference.html>
- mlx-lm: <https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/SERVER.md>

## 8. Limits of this analysis

- One host, one GPU class (sm_89). Nothing here was run on gx10 or on a machine without a GPU.
- `apr chat` was not driven, because it is interactive. Its backend reporting is `not_measured`.
- The nightly UX crux (`audit.sh`) did not probe aprender on 2026-11-27: its build step was refused by the memory cap. Probes C1, C2, C3 and C7 were re-run by hand here; C4, C5 and C6 are `not_measured`. That instrument, and its Ollama flag table from ollama 0.5.7, belong to the infra repo.
- The cause of F-09 was not traced in source.
- The raw captures are not committed. §6 regenerates them.
- One independent review lane read rev 1 of this analysis and the plan and returned FAIL, a second lane read rev 2 and returned FAIL, and a third read rev 4 and returned FAIL. The plan's §7 lists what changed each time; this document gained §5a, the F-10 command block and the extra lines in F-03, F-06 and F-09.
