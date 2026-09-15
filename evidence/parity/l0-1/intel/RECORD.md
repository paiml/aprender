# L0-1 CPU leg — intel (2026-09-15T14:4x–14:5xZ). STRUCTURE AND COMMANDS ONLY; NOTHING MEASURED ON qwen35.

The third host leg of L0-1 (`../lambda/RECORD.md`, `../gx10/RECORD.md`), the first with
`compute_class: cpu`, and the first whose `[X]` comparator is **llama.cpp on CPU** instead
of apr's own CPU reference. Issue #3303, child of the 0.68 Qwen3.5 epic #3038; the
measurement it will carry is the long pole of #3091.

**Read this first.** Every qwen35 cell below is `[U]`. The two GPU legs each carry a
number because both sides of their comparison ran; on this leg **neither side runs the
model today** and both refusals are measured and reproduced below. A `[U]` here is the
result, not a placeholder someone forgot to fill: the operator's ruling on #3303 is that
evidence introduced by the PR under test is not evidence, and the only qwen35 parity
numbers in existence today are #3114's own.

## Declaration (what the run will be, when it can run)

| field | value | where it comes from |
|---|---|---|
| host | `intel` (`mac-server`) | `scripts/perf-matrix.yaml:138` |
| `compute_class` | **`cpu`** | `scripts/perf-matrix.yaml:140` — the only host in the matrix with `compute_class: cpu` AND `ci_runner: self-hosted` (lambda/gx10/yoga are `cuda`; `mini` is `status: NA`, #2841) |
| comparator | `llamacpp`, CPU build, **pin `39173bcac`** | `scripts/perf-matrix.yaml:141`, `scripts/llama_pin.toml` `build_commit`; `build_flags_intel = "cmake -B build -DGGML_NATIVE=ON"` |
| model | `Qwen3.5-0.8B-Q4_K_M.gguf` (`general.architecture = qwen35`) | `evidence/models/supported.yaml` names `qwen3.5-0.8b`, family `qwen3.5` |
| corpus prompt | the 78-token paragraph, byte-identical to both GPU legs | `scripts/check_model_parity.sh` `$PROMPT` |
| n | **5** per lane, as on both GPU legs | `../gx10/RECORD.md` |
| `min_positions` | **64** | `evidence/parity/thresholds.yaml` `min_positions` (I8 / CF-4, #1864) |
| judged by | min over positions of per-position cosine, subject-vs-comparator | `scripts/check_model_parity.sh` `judge()` |
| threshold | `models["qwen3.5-0.8b"].min_cosine` — **absent, `basis: [U]`** | `evidence/parity/thresholds.yaml`; `judge()` exits **2** on a rule with no `min_cosine`, so this leg is fail-closed and cannot inherit the GPU `0.98` |

### The measurement host is under CI load

`ssh intel uptime`, two samples 5.5 minutes apart on 32 threads (Intel Xeon W-3245 @ 3.20 GHz,
283 GiB): `load average: 60.48, 79.50, 93.93` (14:46:59Z) and `49.74, 61.38, 81.70` (14:52:33Z).
A CPU throughput number taken at this load is a measurement of the CI fleet, not of apr.
The cosine comparison this leg judges is **load-insensitive** (it compares logits, not rates),
which is why the CPU leg is specified as a *correctness* leg first; any tok/s on this host
belongs in `../../LEDGER.md` with its load recorded, never here.

## The command — TWO lanes, because `apr parity` does not exist on a CPU host

On a GPU host one command produces both sides: `apr parity` runs apr's CUDA kernel and apr's
own CPU reference and compares them. That command **cannot** be the CPU leg's command, and
this is measured, not assumed:

```
$ ssh intel                          # apr 0.65.2, /usr/local/bin/apr,
$ cd ~/models && apr parity ./qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --prompt "Hi" --json
error: Feature not enabled: cuda feature required for parity check
rc=9
```

So the CPU leg is **two lanes and a comparator step**, and the `[X]` side is llama.cpp:

```bash
# ── lane A: the SUBJECT — apr's CPU forward, per-position logits ────────────
#    binary pinned from HEAD, never a bare `apr` (CLAUDE.md step 0):
. scripts/apr_bin.sh || exit 1
cd ~/models && for i in 1 2 3 4 5; do
  "$APR" <PRODUCER> ./Qwen3.5-0.8B-Q4_K_M.gguf \
      --prompt "<the 78-token corpus prompt, $PROMPT in scripts/check_model_parity.sh>" \
      --json > apr-qwen35.$i.json 2> apr-qwen35.$i.err
done

# ── lane B: the COMPARATOR [X] — llama.cpp CPU at the pin, per-position logits
#    NON-INTERACTIVE. `llama-cli` on master is chat-only and waits on stdin; the
#    logits producer is llama-perplexity, which takes none.
cd ~/src/llama.cpp && ./build/bin/llama-perplexity \
      -m ~/models/Qwen3.5-0.8B-Q4_K_M.gguf -ngl 0 -t 8 \
      -f corpus.txt --save-all-logits llamacpp-qwen35.logits < /dev/null

# ── judge: min over >= 64 positions of cosine(subject, comparator) ──────────
bash scripts/check_model_parity.sh --judge apr-qwen35.1.json --model qwen3.5-0.8b
```

`<PRODUCER>` is deliberately a hole, not a guess. **No apr subcommand on `main` emits
apr-CPU logits against an external reference.** `apr parity` is apr-CUDA-vs-apr-CPU and
is cuda-gated (rc 9 above); `apr kernel parity` is kernel-level (CRUX-L-02), not
whole-model. The only such producer in existence is
`crates/aprender-serve/examples/qwen35_parity.rs` in PR #3114 — which is precisely the
artifact the #3303 ruling forbids as evidence for #3114. Naming the producer is a
decision for the epic, not something this record may invent.

`-ngl 0` on lane B is a lane property, not a protocol knob (see the `llama_bin.sh` header):
a cpu-class subject may never be scored against a CUDA comparator.

## What blocks the measurement today — four findings, each with its command

### B1 — the comparator BINARY exists on intel, at the pin, CPU-built. NOT a blocker.

```
$ ssh intel 'cd ~/src/llama.cpp && git log -1 --format=%H'
  39173bcacb67329850b9ff3108dd036eafb680f0        # == llama_pin.toml build_commit 39173bcac
$ ./build/bin/llama-cli --version
  version: 7746 (39173bcac)
  built with GNU 11.4.0 for Linux x86_64
$ grep -E '^(GGML_CUDA|GGML_NATIVE|CMAKE_BUILD_TYPE):' build/CMakeCache.txt
  GGML_CUDA:BOOL=OFF   GGML_NATIVE:BOOL=ON   CMAKE_BUILD_TYPE:STRING=Release
$ sha256sum build/bin/llama-cli
  1d9050b5e8be51032763ab93bb71d94a09034f7829ad99cc6772c735af9206fe
```

`GGML_NATIVE=ON` + `GGML_CUDA=OFF` is exactly `llama_pin.toml` `build_flags_intel`, and only
`libggml-cpu.so` is present beside it (no `libggml-cuda.so`). Item 4 of #3303 — "a CPU-only
llama.cpp build of `39173bcac` on intel" — **is already satisfied**; the pin needs no bump for
this and must not get one.

### B2 — BLOCKER: the logits producer `llama-perplexity` is not built on intel.

```
$ ssh intel 'ls ~/src/llama.cpp/build/bin/ | wc -l'   -> 18 files, 3 executables:
  llama-bench  llama-cli  llama-server
$ ls build/bin/llama-perplexity   -> No such file or directory
$ grep -n 'kl.divergence.base' common/arg.cpp
  2032:  "computes KL-divergence to logits provided via --kl-divergence-base"
  2038:  {"--save-all-logits", "--kl-divergence-base"}, "FNAME",
```

The source and the flags exist at the pin (`tools/perplexity`, configured in `build/tools/`);
only the target was never built. `cmake --build ~/src/llama.cpp/build --target llama-perplexity`
fixes it. **Not done here**: building on a host is a host mutation, and this phase is read-only.
Owner: whoever provisions the intel comparator (perf-gate).

### B3 — HARD BLOCKER: llama.cpp at the pin cannot load a `qwen35` GGUF at all.

```
$ ./build/bin/llama-cli -m ~/models/Qwen3.5-0.8B-Q4_K_M.gguf -p hi -n 1 -no-cnv --no-warmup < /dev/null
  llama_model_load: error loading model: error loading model architecture: unknown model architecture: 'qwen35'
  rc=1
$ ssh intel 'cd ~/src/llama.cpp && grep -c "\"qwen35\"" src/llama-arch.cpp'   -> 0
$ ssh intel 'cd ~/src/llama.cpp && grep -rn "qwen35" src/ | wc -l'            -> 0
$ ssh intel 'cd ~/src/llama.cpp && grep -n QWEN3NEXT src/llama-arch.cpp | head -1'
  37:    { LLM_ARCH_QWEN3NEXT,        "qwen3next"        },
```

The pin knows `qwen3next`; the Unsloth GGUF declares `general.architecture = qwen35`. Zero
occurrences of the string anywhere in the pinned `src/`, so this is a load-time refusal on
every build of this commit, CPU or CUDA. **The `[X]` comparator baseline for qwen35 does not
exist at the pinned commit.** This is the decision the epic owes, and none of the three
options is free:

| option | cost |
|---|---|
| bump `build_commit` past qwen35 support | `llama_pin.toml` states a bump invalidates **every** ratio measured against `39173bcac`; the first release after it has no usable trend |
| a SECOND pin, scoped to qwen35 only | a new denominator series that may never be compared with the `39173bcac` series; needs its own expiry and decider |
| re-convert the model to `qwen3next` | not a rename: different kv keys and tensor names. Would also change what is being measured — the comparator would no longer be "the llama.cpp a user runs on this file" |

The refusal above was reproduced with the **lambda** checkout of the same commit `39173bcac`
(the model is on lambda and not on intel, and copying it there is a host mutation this phase
does not make). The arch table lives in `libllama` and is identical at a fixed commit; intel's
own source was grepped for the same string and has none, which is the same fact from the other
side.

### B4 — HARD BLOCKER: apr on `main` refuses qwen35 too, by name, on BOTH backends.

```
$ cd ~/models && apr run ./Qwen3.5-0.8B-Q4_K_M.gguf --prompt "hi" --max-tokens 1
  error: Inference failed: Format error: Architecture 'qwen35' uses SSM/Gated Delta Net layers
  (detected tensor 'blk.0.ssm_a'). NEITHER the CPU nor the GPU backend implements Gated
  DeltaNet/SSM layers yet. Tracking issues: #3090 (GPU) and #3091 (CPU).
  rc=8
```

Binary: `apr 0.67.0 (v0.67.0+no-git)`, sha256 `dd0feaad6eacd750883c4c6164af68b36ed1468400dbf99148420a899e6c3694`
— the published 0.67.0, **not** a HEAD-attributed build (`scripts/apr_bin.sh` would have had to
compile one; this probe did not need it). The refusal it prints is the one still on `main` at
`10bd25400`: `unsupported_architecture_reason` in
`crates/aprender-serve/src/gguf/transformer.rs`, with `crates/aprender-serve/src/gguf/config.rs:207`
listing `qwen35`. So the SUBJECT side is blocked by #3091 itself, which is the caveat #3303
already states: this reference can be **built** independently of #3114 but cannot be
**exercised** until a CPU qwen35 forward lands.

### B5 — the manifest name and the file on disk do not glob-match (found while probing).

```
$ bash -c 'ls ~/models/qwen3.5-0.8b*.gguf'
  ls: cannot access '<$HOME>/models/qwen3.5-0.8b*.gguf': No such file or directory   rc=2
$ ls ~/models/Qwen3.5-0.8B*.gguf
  <$HOME>/models/Qwen3.5-0.8B-Q4_K_M.gguf
```

`check_model_parity.sh` finds a model with `ls "$MODELS_DIR"/"$name"*.gguf` under `bash`, which
is case-sensitive. `evidence/models/supported.yaml` names the model `qwen3.5-0.8b`; the file
on disk is `Qwen3.5-0.8B-Q4_K_M.gguf`. C14 therefore prints `UNMEASURED qwen3.5-0.8b` on a host
that **holds the file** — an absence that reads as "no model here" rather than as a defect.
(An interactive `zsh` with `nocaseglob` matches it, which is how this nearly went unnoticed:
the same glob answered differently in the two shells.) `evidence/models/supported.yaml` and
`scripts/check_model_parity.sh` are outside this ticket's scope; recorded here so the next
leg does not re-derive it. (`<$HOME>` above is an elision: the shell printed the expanded
home, and a named user's home in a committed file is the defect `check_hardcoded_paths.sh`
exists to stop. Nothing else about the two lines is changed.)

## Model provenance (measured; the file is on lambda, not on intel)

| field | value | command |
|---|---|---|
| path | `~/models/Qwen3.5-0.8B-Q4_K_M.gguf` (lambda) | `ls ~/models` on intel returns **0** `qwen3*` files |
| size | 532517120 bytes | `stat -c %s` |
| sha256 | `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517` | `sha256sum` |
| `general.architecture` | `qwen35` | GGUF kv, read directly from the header |
| `general.name` / `size_label` | `Qwen3.5-0.8B` / `0.8B`, `quantized_by: Unsloth`, `file_type 15` (Q4_K_M) | GGUF kv |
| shape | `block_count 24`, `embedding_length 1024`, `feed_forward_length 3584`, `head_count 8`, `head_count_kv 2`, `key_length/value_length 256` | GGUF kv |
| hybrid | `ssm.conv_kernel 4`, `ssm.state_size 128`, `ssm.group_count 16`, `ssm.time_step_rank 16`, `ssm.inner_size 2048`, `full_attention_interval 4` | GGUF kv — this is the Gated Delta Net half B4 refuses |
| tokenizer | `tokenizer.ggml.model = gpt2`, `tokenizer.ggml.pre = qwen35` | GGUF kv |

The 78-token corpus prompt was counted under the **qwen2.5** tokenizer. Its position count
under **this** tokenizer is `[U]` and cannot be measured today: every tool that would tokenize
it loads the model first and hits B3/B4. Until it is counted, `min_positions: 64` is a
declaration this leg has not yet shown it can satisfy — the prompt is ~370 characters, so 64+
positions is expected, and "expected" is not "measured".

## What this leg does NOT do

- **It writes no `LEDGER.md` row.** A ledger row is one *cell run* and spends its PP-9 key.
  Nothing ran. A row for a run that did not happen would spend a key on nothing, and PP-9
  makes that irreversible.
- **It writes no `min_cosine`.** See the `models:` entry added to
  `evidence/parity/thresholds.yaml` by this commit: `basis` is an explicit `[U]` naming the
  measurement that would replace it, and `min_cosine` is **absent**, so
  `check_model_parity.sh --judge` exits 2 (`a threshold without a basis is not a threshold`)
  rather than silently inheriting the GPU population's `0.98`.
- **It touches nothing on any host.** No build, no copy, no install. Every probe above is a
  read.

## What would replace the `[U]`

Exactly what lifted the `[U]` on the GPU default: **two known-good and two known-bad pairs,
n=5 each, on this host, with the separating margin stated.**

1. Unblock B2: `cmake --build ~/src/llama.cpp/build --target llama-perplexity` on intel.
2. Decide B3 (the comparator pin for `qwen35`) — three options above, each with a named cost.
3. Land the subject side (#3091 CPU qwen35 forward) and name the `<PRODUCER>` subcommand.
4. Known-good: `Qwen3.5-0.8B-Q4_K_M` apr-CPU vs llama.cpp-CPU, n=5, min cosine over >= 64
   positions. Known-bad: **no deliberately-broken qwen35 GGUF exists** — #3303's own open
   question, and it still has no decider. The GPU basis used a *different model*
   (qwen2.5-coder-1.5b) as its known-bad; whether that substitution is available here is
   itself a decision, because on this leg the two sides are different *implementations*,
   not two backends of one implementation.
5. Only then: write `min_cosine` with the four numbers, their stdevs, and the margin.
