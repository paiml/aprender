# CRUX inference dogfood: 0.0.0-fixture on syn (gpu lane)

apr `apr 0.0.0-fixture (000000000)` · llama.cpp `fixture` · ollama `fixture` · hf `None` · llamafile `None` · vllm `None` · judged 2026-10-04T17:02Z

**RED**: 1 cells, 1 RED (0 of them ALL_WRONG), 0 GREEN.

| model | verb | thinking | prompt | verdict | apr | llama.cpp | ollama | hf | llamafile | vllm | token parity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-model-q4_k_m.gguf | run | off | ctl-2plus2 | **RED** | ✅ <answer>4</answer> | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ not requested | ⛔ not requested | unmeasured: no llama.cpp tokenization row |

✅ correct · ❌ answered, wrong · ⛔ did not answer (reason shown).
Rates and token counts are in the JSON, as each engine reported them, and are not judged.
