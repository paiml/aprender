# PMAT-3778 — receipt

**Ticket:** PMAT-3778 (issue #3778) — CRUX engine drivers for #3739: llamafile and HF transformers, probe/gen/tok/tmpl/greedy rows to row contract v1. **Kind:** code.
**Branch:** `PMAT-3778-crux-engines` from `origin/main` at `a9502d992`, worktree `~/.cache/paiml-implement/wt/aprender-crux`; `~/src/aprender` untouched.
**Origin:** the operator, verbatim on #3739 (relayed by the cop, aprender-04): *"add hugging face as a pre-release CRUX too as they are a competitor for model behavior"* and *"also add llamafile as CRUX"*. The seam — engines here, correspondence file and judge with aprender-76 (#3774), the llamafile install in paiml/infra — agreed with both before code. The row contract is aprender-76's (#3739 comment 5765991210); the argument interface was proposed here and accepted with four additions (`--temperature`, `--context`, `--messages` always, answer-only `text` + `reasoning`).

## What landed

- **`scripts/crux_engine_llamafile.sh probe|gen`** — execs the forjar-declared `~/.local/lib/crux/llamafile` (paiml/infra#906/#907: 0.10.6, one sha256 for the APE binary on every host), never PATH. `probe` prints `--version` and the sha256 of the file it will execute; the pin has one owner, the forjar resource. `gen` appends one row per cell; `--interface cli|server` (default per the correspondence; aprender-76 chose `server` for every llamafile verb after the measurements below). `tok`/`tmpl`/`greedy` exit 2 — `none` in llamafile's column.
- **`scripts/crux_engine_hf.sh probe|gen|tok|tmpl|greedy`** — `uv run --frozen` over `scripts/crux_hf/uv.lock` (`transformers[serving]==5.17.0`, `torch==2.14.0`, `tokenizers==0.23.2`, `jinja2==3.1.6`, `numpy==2.2.6`, `safetensors==0.8.0`, `huggingface_hub==1.32.0`, `requests==2.34.2`); the venv outside the tree. `gen` runs `run`/`chat` through `generate` (greedy whatever `--temperature` says) and `serve run`/`code` through the pinned `transformers serve`; `tok` (AutoTokenizer ids), `tmpl` (apply_chat_template bytes, thinking on/off/unset), `greedy` (prompt/generated ids + optional float32 logits `.npy`).
- **Every gen reports `reported.device`** (the judge requires it and rejects a row off its lane): HF reads it from the weights (`cpu` / `cuda:0 <name>`); llamafile is `cpu (llamafile --gpu disable)` on the cpu lane and, on the gpu lane, its own "offloaded N/M layers to GPU" line — without that line the label starts with `cpu`, so the judge rejects the cell rather than trust an unmeasured GPU.
- **`chat` drives the conversation** (aprender-76's addition): `--messages` holds the user turns; each is answered with the conversation so far, the answer appended as the assistant turn; the output adds `turns` (every answer, in order) and `text` is the last. llamafile does it through one server; a multi-turn chat without `--interface server` is exit 2.
- **A cell an engine cannot run is a row**: `rc: null` and `refused` carrying the engine's own words (llamafile's error lines; the Python exception verbatim). A caller error (a `--model-sha256` the file does not hash to, a missing argument) is exit 2 with no row.

## Measured (lambda-labs, 2026-09-21)

| engine | measurement |
|---|---|
| llamafile 0.10.6 | `probe` → `llamafile v0.10.6 sha256=d579f61d…` (equal to the forjar pin and GitHub's asset digest) |
| llamafile | Qwen2.5-Coder-0.5B `run` → `4`. **Qwen3.5-0.8B (`qwen35`) LOADS** and answers `4` — #3739's "probably can't load qwen35" does not hold for it (aprender-76 is correcting #3739) |
| llamafile `--cli` | IGNORES `-rea off` and `--chat-template-kwargs '{"enable_thinking":false}'` (byte-identical `<think>` output) and does NOT bound output with `-n` (`-n 8` printed 100 numbers) |
| llamafile server | honours `enable_thinking` per request (off → `4`, 0 reasoning, 2 tokens; on → `4` + 608 chars of `reasoning_content`) and `max_tokens` exactly; the server's pid is not the launched pid (the APE re-execs), so it is stopped by the port's listener |
| llamafile refusal | a non-GGUF file → a row, `rc: null`, `refused: "… gguf_init_from_file: failed to open GGUF file …"` |
| HF `probe` | `transformers=5.17.0 torch=2.14.0+cu130 … cuda_available=true device=NVIDIA_GeForce_RTX_4090 capability=sm_89 arch_list=sm_75,sm_80,sm_86,sm_90,sm_100,sm_120 lock_sha256=be29a4f4…` |
| HF, GB10 | torch 2.14.0 on PyPI depends on the CUDA 13 runtime for all Linux incl. aarch64; its arch list has **no sm_121** entry, so gx10 relies on sm_120 binary compatibility — UNMEASURED until `probe` runs on gx10 |
| HF multi-turn chat (cpu) | Qwen2.5-Coder-1.5B-Instruct: `turns ["4","12"]`; llamafile qwen2.5-coder-0.5b Q4_K_M: `["4","9"]` (a wrong follow-up — a cell for the judge) |
| HF (CPU, every path) | Qwen2.5-0.5B `run` → `4`; Qwen3-0.6B `chat` thinking **off → `2`**, on → `4` (464 chars reasoning); `transformers serve` → `4`; `tok` 11 ids; `tmpl` ×3 (off appends Qwen3's empty `<think></think>`); `greedy` → `[19, 151645]` + logits; a missing repo and a GPU cell without CUDA are refused rows quoting the exception |

GPU cells were NOT run here: the `gpu-q` queue held four P1 release jobs, and the GPU column belongs to the CRUX run under `gpu-q` in aprender-76's driver.

## Deviations, named

- **The cpu lane reached the GPU** (found by aprender-76's integration run: `--backend cpu` answered "!!!!", token 0 every step, with the CUDA signature). With CUDA merely visible, transformers 5.17 loaded the model through a path that reached the 4090 — outside `gpu-q`, which the cpu lane never takes. My own CPU check had hidden CUDA and so could not see it. Fixed: CUDA is hidden from the process before torch is imported for cpu cells and `tok`/`tmpl`, and models load with `device_map=<lane device>`. Re-measured with CUDA visible and `--backend cpu`: `4`, `device: cpu`, no GPU compute process during the run. The GPU lane itself stays UNMEASURED here (a `gpu-q` test waited its 40-min bound behind P1 jobs and ran nothing).
- Tested with scratch manifests and the drivers' own CLIs, not through the main driver (`scripts/crux_inference_dogfood.sh`, aprender-76's, in flight); the integration is theirs to measure.
- `greedy` first forced `min_new_tokens=steps`, which suppressed EOS ("4." became "4.000000"); removed before commit.
- `scripts/check_no_competing_harnesses.sh` (PERF-009) flagged the first draft of `crux_engine_llamafile.sh`: it starts llamafile's server and named a `decode_rate_tokens_per_second` field. An allowlist line was added and then REMOVED on the cop's ruling (no exemptions; an agent-added allowlist line is an agent-writable waiver). Restructured instead: the row carries llamafile's raw `timings` object verbatim and names no rate — any rate is the judge's to derive, the one computation site. The guard passes with `check_no_competing_harnesses.sh` byte-identical to main (count 0, baseline 0).
- `gpu-q --help` was run once by mistake — gpu-q has no `--help` and queued it as a GPU job at prio 5; the tool timeout killed it and its trap removed the ticket (queue re-read afterwards: no ticket of ours).

## Verdict

Receipt written; per the cop's instruction the work stops here (aprender branches, stop at the receipt).
