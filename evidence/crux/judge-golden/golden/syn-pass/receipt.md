# CRUX inference dogfood: 0.0.0-fixture on syn (gpu lane)

apr `apr 0.0.0-fixture (000000000)` · llama.cpp `fixture` · ollama `fixture` · hf `None` · llamafile `None` · vllm `None` · judged 2026-10-04T17:02Z

**PASS**: 1 cells, 0 RED (0 of them ALL_WRONG), 1 GREEN.

| model | verb | thinking | prompt | verdict | apr | llama.cpp | ollama | hf | llamafile | vllm | token parity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-model-q4_k_m.gguf | run | off | ctl-2plus2 | **GREEN** | ✅ <answer>4</answer> | ✅ <answer>4</answer> | ✅ <answer>4</answer> | ✅ <answer>4</answer> | ⛔ not requested | ⛔ not requested | = (apr 3 vs 3; first diff None) |

✅ correct · ❌ answered, wrong · ⛔ did not answer (reason shown).
Rates and token counts are in the JSON, as each engine reported them, and are not judged.
