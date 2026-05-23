<!-- PCU: cli-chat | contract: contracts/apr-page-cli-chat-v1.yaml -->

# apr chat

Interactive chat with language model

**Category**: Inference

## Synopsis

```text
apr chat [OPTIONS]
```

## Example

```bash
apr chat qwen2.5-coder-1.5b
```

## What this does

`apr chat` opens a multi-turn REPL against a local model. Each turn is wrapped in
the model's ChatML template, so Instruct-tuned checkpoints (Qwen, LLaMA, Mistral)
respond conversationally instead of completing your text. Conversation history is
kept in-process — exit the REPL and history is gone. For persisted sessions use
`apr code --resume`.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--system MSG` | System prompt that frames every turn | `--system "You are a Rust expert."` |
| `--temperature T` | Sampling temperature (0.7 is the default sweet spot) | `--temperature 0.2` |
| `--top-p P` | Nucleus sampling cutoff | `--top-p 0.95` |
| `--max-tokens N` | Cap per response (default 512) | `--max-tokens 1024` |
| `--inspect` | Show top-k probs + tok/s per turn | `--inspect` |
| `--backend B` | Force backend (`cuda`/`cpu`/`wgpu`) | `--backend cpu` |

## Common workflows

**Sandbox a system prompt before promoting it to production.**

```bash
apr chat qwen2.5-coder-7b.apr \
    --system "Reply only with valid Python. No prose." \
    --temperature 0.1
```

**Profile per-turn latency while you converse.**

```bash
apr chat qwen2.5-coder-1.5b.apr --inspect --backend cuda
# After each turn you'll see: 87 tok @ 412 tok/s, ttft 38ms
```

## Troubleshooting

- **Model echoes the prompt or returns gibberish** — the chat template wasn't
  applied. `apr chat` auto-detects Instruct models but a base model will need
  `apr run --chat` instead. Confirm with `apr inspect <model> | grep template`.
- **CUDA OOM on a 7B model** — drop `--max-tokens` (KV cache scales linearly with
  context), or pass `--backend cpu` to fall back to Trueno SIMD.
- **Responses get cut mid-sentence** — bump `--max-tokens`; the default 512 is
  conservative. For long-form generation, 1024-2048 is standard.

## See also

- Source: [`crates/apr-cli/src/commands/chat.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/chat.rs)
- Contract: [`contracts/apr-page-cli-chat-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-chat-v1.yaml)

