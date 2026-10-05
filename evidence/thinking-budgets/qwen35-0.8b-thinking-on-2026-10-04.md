# Qwen3.5-0.8B thinking-ON leg: budget sweep and engine cross-check (2026-10-04)

Cited by `contracts/thinking-budgets-v1.yaml` (entries `Qwen3.5-0.8B-IQ4_XS.gguf`,
aprender#4663, and `Qwen3.5-0.8B-Q4_K_M.gguf`, aprender#4666).

Binary: released apr v0.70.1 x86_64 cuda (`apr 0.70.1 (04332f838)`, release asset
sha256 verified), RTX 4090, the ladder's qa flags. Reference engine: llama.cpp
`llama-completion` build 11382 (11fe02151). Prompt: the official Qwen3.5 template with
thinking on, `What is 2+2?`, 17 prompt tokens on both engines.

## 1. Budget sweep, apr greedy (the golden ON leg): no budget closes

`apr qa` golden_output_thinking_on, one run per cell; every cell is rc=5,
"think block unclosed within N tokens".

| model | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|
| IQ4_XS | unclosed, 8,910 chars | 17,795 | 35,574 | 71,140 |
| Q4_K_M | unclosed, 6,723 chars | 13,225 | 26,243 | 52,290 |

Output grows linearly with the budget (about 4.35 chars per token for IQ4_XS, 3.2
to 3.3 for Q4_K_M). The text is
coherent, reaches "2 + 2 = 4" early, then loops ("Wait, I should check...").

## 2. The loop is the model's, not apr's

- Greedy decoding hits a near-tie early ("2. Identify the Core Task: ..."): apr GPU,
  apr CPU, llama.cpp GPU and llama.cpp CPU each pick a different token there.
  llama.cpp's two greedy trajectories happen to close (870 / 632 chars); apr's two loop.
- Qwen's recommended sampling (T 0.6, top-k 20, top-p 0.95), 8 seeds, budget 2048:
  - Q4_K_M: llama.cpp closed 2/8, apr closed 5/8.
  - IQ4_XS: llama.cpp closed 4/8.
- The unclosed llama.cpp runs loop the same way ("Wait, I should check if...").
- apr 0.70.1 already warns on this model: "Qwen3.5-0.8B is not CRUX-certified. Known
  issue #4030: with thinking on it may reason past the token budget".

## 3. What follows for the contract

No budget makes the greedy ON leg close on apr, so raising a budget cannot turn this
leg green. IQ4_XS keeps no `budget` (fail-closed); Q4_K_M keeps 2048 and stays FAIL,
known_red. Making the leg pass would mean changing the leg itself (for example a
sampled closure rate), which is a gate change for its own ticket.
