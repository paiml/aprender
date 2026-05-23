<!-- PCU: cli-distill | contract: contracts/apr-page-cli-distill-v1.yaml -->

# apr distill

Knowledge distillation (teacher -> student) (GH-247, ALB-011)

**Category**: Training

## Synopsis

```text
apr distill [OPTIONS]
```

## Example

```bash
apr distill --teacher 7b.apr --student 0.5b.apr --data train.jsonl
```

## What this does

`apr distill` trains a small "student" model to imitate a large "teacher" by
matching the teacher's softmax distribution over a corpus. The result is a model
with ~5-10x fewer parameters that retains most of the teacher's quality. Two-stage
mode (`--config two_stage.yaml`) precomputes teacher logits to disk, then trains
the student offline — much faster than running both forward passes per step.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `<TEACHER>` | Teacher model (positional) | `qwen2.5-coder-7b.apr` |
| `--student FILE` | Student model | `--student qwen2.5-coder-0.5b.apr` |
| `-d, --data FILE` | Training JSONL | `--data train.jsonl` |
| `--strategy S` | `standard`, `progressive`, `ensemble` | `--strategy progressive` |
| `--temperature T` | Softmax scaling (default 3.0) | `--temperature 4.0` |
| `--alpha A` | KL vs task loss weight (default 0.7) | `--alpha 0.5` |
| `--stage S` | `precompute`, `train`, `generate` | `--stage precompute` |
| `--config FILE` | Two-stage YAML config | `--config distill.yaml` |
| `--backend B` | `fixture` (default), `gpu` | `--backend gpu` |

## Common workflows

**Standard one-stage distill (small datasets, GPU memory permitting).**

```bash
apr distill qwen2.5-coder-7b.apr \
    --student qwen2.5-coder-0.5b.apr \
    --data ./data/distill.jsonl \
    --strategy standard --epochs 3 -o ./student-out/
```

**Two-stage: precompute teacher logits, then train the student offline.**

```bash
apr distill qwen2.5-coder-7b.apr --stage precompute \
    --data ./data/distill.jsonl -o ./teacher-logits/
apr distill --stage train --config two_stage.yaml -o ./student-final/
```

## Troubleshooting

- **Student plateaus at the teacher's loss** — that's the ceiling; distillation
  cannot exceed the teacher. Try `--strategy progressive` (curriculum) or use a
  stronger teacher.
- **Teacher OOM in single-stage mode** — switch to `--stage precompute` first,
  then run `--stage train` with only the student on the GPU. See
  [PMAT-701](https://github.com/paiml/aprender/issues/701) for the
  TEACHER==STUDENT==0.5B smoke-defaults-leak lesson.
- **NaN losses early in training** — drop `--temperature` to 2.0; T=3.0 can
  cause flat distributions that produce vanishing gradients.

## See also

- Source: [`crates/apr-cli/src/commands/distill.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/distill.rs)
- Contract: [`contracts/apr-page-cli-distill-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-distill-v1.yaml)

