# CRUX inference dogfood: 0.69.1 on lambda (gpu lane)

apr `apr 0.69.1 (70d20d44c)` · llama.cpp `version: 0.4.1-dev (build 10987, commit d1d3c3396)` · ollama `None` · hf `None` · llamafile `None` · vllm `None` · judged 2026-09-23T10:36Z

**RED**: 12 cells, 12 RED (12 of them ALL_WRONG), 0 GREEN.

| model | verb | thinking | prompt | verdict | apr | llama.cpp | ollama | hf | llamafile | vllm | token parity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B-IQ4_XS.gguf | run | off | golden-2plus2 | **RED** | ❌ 2+2 = 4 | ❌ 2 + 2 = **4** | ⛔ not requested | ⛔ not requested | ⛔ not requested | ⛔ not requested | ≠ (apr 19 vs 17; first diff 17) |
| Qwen3.5-0.8B-IQ4_XS.gguf | run | off | golden-greeting | **RED** | ❌ Hello there! I'm doing well, thank you for aski… | ❌ Hello there too! How about you? I'm doing well,… | ⛔ not requested | ⛔ not requested | ⛔ not requested | ⛔ not requested | ≠ (apr 23 vs 21; first diff 21) |
| Qwen3.5-0.8B-IQ4_XS.gguf | run | off | golden-paris | **RED** | ❌ The capital of France is **Paris**. | ❌ The capital of France is **Paris**. It is the l… | ⛔ not requested | ⛔ not requested | ⛔ not requested | ⛔ not requested | ≠ (apr 19 vs 17; first diff 17) |
| Qwen3.5-0.8B-Q4_K_M.gguf | run | off | golden-2plus2 | **RED** | ❌ 2+2 = 4 | ❌ $2 + 2 = 4$. | ⛔ not requested | ⛔ not requested | ⛔ not requested | ⛔ not requested | ≠ (apr 19 vs 17; first diff 17) |
| Qwen3.5-0.8B-Q4_K_M.gguf | run | off | golden-greeting | **RED** | ❌ Hello there! I'm doing well, thank you for aski… | ❌ Hello there! I'm doing well, thank you for aski… | ⛔ not requested | ⛔ not requested | ⛔ not requested | ⛔ not requested | ≠ (apr 23 vs 21; first diff 21) |
| Qwen3.5-0.8B-Q4_K_M.gguf | run | off | golden-paris | **RED** | ❌ The capital of France is **Paris**. Paris is th… | ❌ The capital of France is **Paris**. Located in … | ⛔ not requested | ⛔ not requested | ⛔ not requested | ⛔ not requested | ≠ (apr 19 vs 17; first diff 17) |
| Qwen3.5-2B-Q4_K_M.gguf | run | off | golden-2plus2 | **RED** | ❌ 2 + 2 = 4 | ❌ 2 + 2 = **4**. | ⛔ not requested | ⛔ not requested | ⛔ not requested | ⛔ not requested | ≠ (apr 19 vs 17; first diff 17) |
| Qwen3.5-2B-Q4_K_M.gguf | run | off | golden-greeting | **RED** | ❌ Hello! I'm doing well, thank you for asking. Ho… | ❌ Hello! I'm doing well, thanks for asking. How a… | ⛔ not requested | ⛔ not requested | ⛔ not requested | ⛔ not requested | ≠ (apr 23 vs 21; first diff 21) |
| Qwen3.5-2B-Q4_K_M.gguf | run | off | golden-paris | **RED** | ❌ The capital of France is **Paris**. | ❌ The capital of France is **Paris**. Located in … | ⛔ not requested | ⛔ not requested | ⛔ not requested | ⛔ not requested | ≠ (apr 19 vs 17; first diff 17) |
| Qwen3.5-4B-Q4_K_M.gguf | run | off | golden-2plus2 | **RED** | ❌ 2 + 2 = 4. | ❌ The sum of 2 and 2 is **4**. | ⛔ not requested | ⛔ not requested | ⛔ not requested | ⛔ not requested | ≠ (apr 19 vs 17; first diff 17) |
| Qwen3.5-4B-Q4_K_M.gguf | run | off | golden-greeting | **RED** | ❌ Hello! I'm doing well, thank you for asking! Ho… | ❌ Hello! I'm doing well, thank you for asking. Ho… | ⛔ not requested | ⛔ not requested | ⛔ not requested | ⛔ not requested | ≠ (apr 23 vs 21; first diff 21) |
| Qwen3.5-4B-Q4_K_M.gguf | run | off | golden-paris | **RED** | ❌ The capital of France is **Paris**. | ❌ The capital of France is **Paris**. Located in … | ⛔ not requested | ⛔ not requested | ⛔ not requested | ⛔ not requested | ≠ (apr 19 vs 17; first diff 17) |

Greedy divergence (REPORTED, not judged): [{"key": {"model_sha256": "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4", "host": "lambda", "prompt_id": "golden-2plus2"}, "engines": ["apr", "llama.cpp"], "llama.cpp": {"first_divergence": 0, "steps_compared": 9}}, {"key": {"model_sha256": "619917ae92f61eb6515a7070e944a0a7c2b198a2cf6536386f475485188a36ff", "host": "lambda", "prompt_id": "golden-2plus2"}, "engines": ["apr", "llama.cpp"], "llama.cpp": {"first_divergence": 1, "steps_compared": 7}}, {"key": {"model_sha256": "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223", "host": "lambda", "prompt_id": "gold

✅ correct · ❌ answered, wrong · ⛔ did not answer (reason shown).
Rates and token counts are in the JSON, as each engine reported them, and are not judged.

Not covered by this run (the issue requires them): verbs not run here: chat, serve, code; thinking ON until #3723 adds an apr toggle; consumer-brief context rungs (#3716) and each engine's max accepted context; TTFT and decode rate: the serve verb's measurement goes through `apr test llm bench` (PERF-009), the next increment.
