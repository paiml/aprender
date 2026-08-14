<!-- PCU: cli-train | contract: contracts/apr-page-cli-train-v1.yaml -->

# apr train

Training pipeline (plan/apply) — forjar-style pre-flight validation — plus the
five training tools that used to ship as their own binaries.

**Category**: Training

## Synopsis

```text
apr train <SUBCOMMAND> [OPTIONS]
```

## Example

<!-- example-cost: trivial -->
```bash
apr train --help
```

## Pipeline subcommands

| Subcommand | Purpose |
|------------|---------|
| `plan` | Generate a training plan without touching the GPU |
| `apply` | Execute a plan (allocate GPU, run trials) |
| `watch` | Run a training config with crash restart and hang detection |
| `sweep` | Generate hyperparameter sweep configs from a base YAML |
| `halving` | Successive-halving HPO over sweep configs |
| `archive` | Package a checkpoint into a release bundle |
| `submit` | Submit multi-adapter jobs to a cluster |
| `cluster-status` | Show cluster nodes, GPUs and adapter capacity |

## Rehomed tools (APR-MONO Rule 1)

Five crates each shipped a standalone binary and were published to crates.io
under their own name. `apr` is the only user-facing binary, so those `[[bin]]`
targets were deleted and the capability moved under `apr train`. Nothing was
dropped: each subcommand calls the same library entry point the deleted `main`
called, and every argument is still reachable.

| Was | Is now | What it does |
|-----|--------|--------------|
| `aprender-train-bench` | `apr train bench` | Distillation hyperparameter sweeps, strategy comparison, cost analysis |
| `aprender-train-distill` | `apr train distill` | End-to-end distillation from a distill config file |
| `aprender-train-inspect` | `apr train inspect` | SafeTensors checkpoint inspection, memory estimates, conversion |
| `aprender-train-lora` | `apr train lora` | LoRA/QLoRA planning, method comparison, adapter merge |
| `aprender-train-shell` | `apr train shell` | Interactive REPL for model exploration and distillation |

These are **not** duplicates of the similarly named top-level commands:

* `apr bench` measures **inference** throughput of a model file (tok/s).
  `apr train bench` sweeps **distillation hyperparameters** (temperature, alpha).
* `apr inspect` reads `.apr` model metadata. `apr train inspect` reads
  **SafeTensors training checkpoints** and answers training questions
  (per-layer parameter counts, optimizer-state memory at a given batch size).
* `apr distill` is a flag-driven teacher/student run over apr's own YAML schema.
  `apr train distill` reads entrenar's native `DistillConfig` schema and adds
  `estimate`, `validate` and `export`.
* `apr finetune --merge` merges an adapter into an `.apr` base at scale 1.0.
  `apr train lora merge` is the SafeTensors/PEFT merge path with `--scale`.

### apr train bench

```bash
apr train bench temperature --start 1.0 --end 8.0 --step 0.5 --runs 3
apr train bench alpha --start 0.1 --end 0.9 --step 0.1 --runs 3
apr train bench compare --strategies kd,progressive,attention,combined
apr train bench ablation
apr train bench cost-performance --gpu a100-80gb
apr train bench recommend --max-cost 50 --min-accuracy 0.9 --gpu t4
```

`--gpu` accepts `a100-80gb`, `a100-40gb`, `v100`, `t4` and refuses anything else.

`compare --runs`, `ablation --config` and `cost-performance --results` are
accepted for compatibility and ignored — they were already ignored by the
binary this command replaces (the harness is deterministic and the ablation
ladder is fixed in code).

### apr train distill

```bash
apr train distill run --config distill.yaml --output ./out --dry-run
apr train distill estimate --teacher Qwen/Qwen2.5-7B --student Qwen/Qwen2.5-0.5B \
    --batch-size 32 --seq-len 512
apr train distill validate --config distill.yaml
apr train distill export -i student.safetensors -f gguf -o student.gguf --quantize q4_0
```

`export -f/--format` names the **model** format (`safetensors`, `gguf`, `apr`).
It is the one migrated subcommand that does not take the display `--format`
flag, because the two would collide at the same level; use the global
`apr --json` instead. GGUF export needs the crate's `hub` feature and refuses
with a clear message otherwise.

### apr train inspect

```bash
apr train inspect info model.safetensors
apr train inspect layers model.safetensors --verbose
apr train inspect memory model.safetensors -b 32 -s 512
apr train inspect validate model.safetensors --strict
apr train inspect convert in.safetensors -t gguf -o out.gguf --quantize q4_0
apr train inspect compare a.safetensors b.safetensors
```

`layers --verbose` lists every tensor name and shape. That is `apr`'s global
`-v` / `--verbose` — same two spellings and same meaning as the deleted
binary's own flag, which could not be redeclared without colliding with the
global one.

`validate` exits non-zero when the integrity checker reports issues.

### apr train lora

```bash
apr train lora plan --model 7B --vram 24 --method qlora
apr train lora compare --model 13B --vram 24
apr train lora merge -b base.safetensors -a adapter.safetensors -o merged.safetensors -s 0.5
apr train lora inspect adapter.safetensors
```

Two short flags could not be carried over from the deleted binary:

* `-m` was declared **twice** there (auto-derived for `--model`, explicit for
  `--method`), which clap rejects outright. Here `-m` is `--method`, matching
  `apr tune -m` and `apr finetune -m`; `--model` is long-only.
* `-v` meant `--vram` there. On `apr`, `-v` is the global `--verbose`, so
  `--vram` is long-only.

`--method` accepts `full`, `lora`, `qlora`, `auto` and refuses anything else.

### apr train shell

```bash
apr train shell
apr train shell --session session.json
apr train shell --command "help"
```

Without `--command`, enters the interactive REPL. With it, runs one command and
exits. `--session` pre-loads a saved session; a session file that fails to load
falls back to an empty session with a message on stderr, as before.

## Shared options

Every rehomed subcommand except `apr train distill export` accepts the two
output-shaping flags the deleted binaries carried:

| Flag | Meaning |
|------|---------|
| `--format <table\|json\|compact>` | Output shape (aliases: `text` → table, `line` → compact) |
| `--no-color` | Disable colored output |

The other two flags from that set — `--quiet` and `--verbose` — are already
global on `apr` and are folded in automatically. The global `apr --json`
overrides `--format`.

## Full help

Run `apr train --help` for the complete option list, and
`apr train <subcommand> --help` for each tool.

## See also

- Source: [`crates/apr-cli/src/commands/train.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/train.rs)
- Rehomed-tool dispatch: [`crates/apr-cli/src/commands/train_tools.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/train_tools.rs)
- Contract: [`contracts/apr-page-cli-train-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-train-v1.yaml)
- Binary rule: [`contracts/apr-mono-binary-rule-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-mono-binary-rule-v1.yaml)
