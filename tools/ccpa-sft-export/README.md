# ccpa-sft-export

Convert [claude-code-parity-apr](../../../claude-code-parity-apr) **teacher.stream.ndjson**
captures (real Claude Code trajectories) into **apr-code `<tool_call>` SFT JSONL** suitable
for `apr finetune` (entrenar `InstructSample`).

This is the SFT export pipeline for apr-code distillation — the previously-missing piece
that lets us fine-tune a small student to emit apr-code tool_calls instead of Markdown/prose.

## What it does

Walks every `teacher.stream.ndjson` under the CCPA `evidence/` tree (138 captures, 40 unique
fixtures, 6,569 raw Claude Code tool_use calls) and:

1. **Remaps** the Anthropic-native tool schema to the apr-code schema:
   | Anthropic | apr-code  | field remap |
   |-----------|-----------|-------------|
   | `Read`    | `file_read`  | `file_path`→`path` |
   | `Write`   | `file_write` | `file_path`→`path`, `content` |
   | `Edit`    | `file_edit`  | `file_path`→`path`, `old_string`→`old`, `new_string`→`new` |
   | `Bash`    | `shell`      | `command` |
   | `Grep`    | `grep`       | `pattern`, `path` |
   | `Glob`    | `glob`       | `pattern` |
   (Task*/ToolSearch/Agent/AskUserQuestion have no apr-code equivalent → dropped: 127 calls.)

2. Emits each assistant `tool_use` turn as an `entrenar` `InstructSample`:
   - `system`      = apr-code `CODE_SYSTEM_PROMPT`
   - `instruction` = running observation transcript (prior tool_calls + tool_results)
   - `response`    = the literal `<tool_call>{"name":..,"input":..}</tool_call>` envelope

   (`InstructSample` has no tool_calls field — the response *string* IS the tool_call JSON.)

## Usage

```bash
cargo build --release
TARGET=$(cargo metadata --no-deps --format-version 1 | python3 -c 'import json,sys;print(json.load(sys.stdin)["target_directory"])')
BIN=$TARGET/release/ccpa-sft-export

# Stratified spike set (recommended): 40 samples per apr-code tool, with context
$BIN --balanced --per-tool 40 --out ../../datasets/apr_code_sft_balanced.jsonl

# First-action-only curated set (124 samples)
$BIN --curated --out ../../datasets/apr_code_sft_curated.jsonl

# Full corpus (5,773 samples)
$BIN --full --out apr_code_sft_full.jsonl
```

## Corpus stats (measured)

| Mode      | Samples | Notes |
|-----------|---------|-------|
| `--balanced --per-tool 40` | 200 | 40 each: file_read/file_edit/file_write/shell/grep; 184 with context |
| `--curated` | 124 | one first-action tool_call per trajectory, deduped |
| `--full`    | 5,773 | every remappable turn (6,442 remapped − 669 dedup) |

Raw corpus: 138 streams, 6,569 tool_use calls (Bash 4190, Write 939, Edit 599, Read 579,
Grep 135, + 132 unmappable). 100% of emitted responses are valid parseable apr-code tool_calls.

## Flip test runbook (Part 2)

The falsifiable claim: base model emits **0** apr-code tool_calls (Markdown/prose prior);
after SFT on this corpus the student emits **≥1** parseable `<tool_call>`.

```bash
# 0) Build a real training binary (WITHOUT this, apr finetune -> execute_training_stub
#    = UNTRAINED random adapter). Use cuda on NVIDIA for speed.
cargo build --release --bin apr --features cuda,training   # or --features training (CPU)

# 1) Base baseline (oracle = parse_tool_calls in agent/driver/realizar.rs)
apr code --model qwen2.5-coder-1.5b-instruct-q4k.apr --project HELDOUT \
    --max-turns 1 --emit-trace base_trace.jsonl \
    -p "The subtract fn in src/lib.rs adds instead of subtracts. Read the file and fix it."
#  -> measured: 0 tool_use blocks (prose: "Sure, please provide the code…")

# 2) Train the LoRA (bare finetune — NOT --task instruct, which is stubbed)
apr finetune qwen2.5-coder-1.5b-instruct-q4k.apr --method lora --rank 16 \
    --data datasets/apr_code_sft_balanced.jsonl --output adapter.apr --epochs 2 \
    --checkpoint-format safetensors
#  -> trained LoRA written to <output-dir>/checkpoints/best/model.safetensors

# 3) Bridge the trainer checkpoint names -> apr finetune --merge names
uv run tools/ccpa-sft-export/remap_adapter.py \
    checkpoints/best/model.safetensors adapter_remapped.safetensors --rank 16 --alpha 32

# 4) Merge the trained delta into the base -> runnable merged.apr
apr finetune qwen2.5-coder-1.5b-instruct-q4k.apr --merge \
    --adapter adapter_remapped.safetensors -o merged.apr

# 5) LoRA'd model on the SAME held-out prompt
apr code --model merged.apr --project HELDOUT --max-turns 1 --emit-trace lora_trace.jsonl \
    -p "The subtract fn in src/lib.rs adds instead of subtracts. Read the file and fix it."
#  SUCCESS = tool_use_count(lora_trace) >= 1  vs  tool_use_count(base_trace) == 0
```

### Flip-test status (measured, this spike)

- **BASE = 0 tool_calls** (measured): `apr code` on the held-out fix task emits prose
  *"Sure, please provide the code for the `src/lib.rs` file."* — no `<tool_call>` block.
- **Merge→run leg = mechanically verified**: a synthetic remapped adapter merged
  (`Layers merged: 56/339` = 28 layers × q+v) into a runnable `merged.apr` that loads
  and generates in `apr code`. So `safetensors → remap → merge → run → trace` all work.
- **LoRA training did NOT complete** in the spike window: on this host the CUDA
  `InstructPipeline::from_apr` for the 1.5B model stayed in F32-dequant + PTX-JIT
  pre-warm for 15+ minutes **without reaching a single training step** (GPU never
  sustained load; the one-time JIT pre-warm is the wall). The CPU `--features training`
  path has the same InstructPipeline-construction bottleneck. So the measured flip is
  **BASE=0 / LoRA=blocked-on-training-build**, not a completed 0→1 flip. The dataset +
  full runbook are ready to run once the build cost is addressed (pre-compiled cubins /
  smaller base / longer GPU budget).

### Known pipeline blockers (measured, in the existing apr finetune/merge/run plumbing — NOT this converter)

1. `apr code`/`apr run` have **no inference-time LoRA adapter flag** — the trained adapter must
   be **merged** into the base first (step 3-4).
2. The `--features training,wgpu` combo **does not compile** (entrenar `WgpuInstructPipeline` /
   `autograd::wgpu_training` are `cfg`'d out — apr-cli's `wgpu` passthrough doesn't enable
   entrenar's gpu feature). Only the CUDA/CPU `execute_training` path builds.
3. The CUDA/CPU trainer writes **only** a SafeTensors checkpoint (`checkpoints/best/model.safetensors`)
   named `lora.{L}.{q,v}_proj.lora_{a,b}` with no rank/alpha header metadata; `apr finetune --merge`
   expects `blk.{L}.attn_{q,v}.weight.lora_{a,b}` + header `lora_rank`/`lora_alpha`. `remap_adapter.py`
   bridges this naming + metadata gap.
