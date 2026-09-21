# CRUX inference dogfood: 0.69.0 on lambda (gpu lane)

apr `apr 0.69.0 (afb05a3de)` · llama.cpp `version: 0.4.1-dev (build 10987, commit d1d3c3396)` · ollama `0.5.7` · judged 2026-09-21T18:15Z

**RED**: 3 cells, 2 RED, 1 GREEN, 0 ALL_WRONG, 0 UNJUDGED.

| model | verb | thinking | prompt | verdict | apr | llama.cpp | ollama | token parity |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-coder-1.5b-instruct-q4_k_m.gguf | run | off | golden-2plus2 | **RED** | ⛔ exit 14 | ✅ 2 + 2 equals 4. | ⛔ refused: ollama 0.5.7 imported the GGUF with no chat templa… | ≠ (apr 46 vs 36; first diff 1) |
| qwen2.5-coder-1.5b-instruct-q4_k_m.gguf | run | off | golden-greeting | **GREEN** | ✅ Hello! I'm just a computer program, so I don't … | ✅ Hello! I'm doing well, thank you. How about you… | ⛔ refused: ollama 0.5.7 imported the GGUF with no chat templa… | ≠ (apr 50 vs 40; first diff 1) |
| qwen2.5-coder-1.5b-instruct-q4_k_m.gguf | run | off | golden-paris | **RED** | ⛔ exit 14 | ✅ The capital of France is Paris. | ⛔ refused: ollama 0.5.7 imported the GGUF with no chat templa… | ≠ (apr 46 vs 36; first diff 1) |

✅ correct · ❌ answered, wrong · ⛔ did not answer (reason shown).
Rates and token counts are in the JSON, as each engine reported them, and are not judged.

Not covered by this run (the issue requires them): verbs chat, serve and code (the next slices of #3739); thinking ON (apr has no toggle until #3723); consumer-brief context rungs (#3716) and each engine's max accepted context; TTFT, which belongs to the serve verb's single OpenAI client.
