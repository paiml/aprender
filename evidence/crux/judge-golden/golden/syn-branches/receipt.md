# CRUX inference dogfood: 0.0.0-fixture on syn (gpu lane)

apr `apr 0.0.0-fixture (000000000)` · llama.cpp `fixture` · ollama `fixture` · hf `None` · llamafile `None` · vllm `None` · judged 2026-10-04T17:02Z

**RED**: 23 cells, 23 RED (1 of them ALL_WRONG), 0 GREEN.

| model | verb | thinking | prompt | verdict | apr | llama.cpp | ollama | hf | llamafile | vllm | token parity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-model-q4_k_m.gguf | run | off | ctl-2plus2 | **RED** | ✅ <answer>4</answer> | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | run | off | p-refused | **RED** | ⛔ refused: model not pulled | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | run | off | p-timeout | **RED** | ⛔ refused: timed out after 600 s | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | run | off | p-exit | **RED** | ⛔ exit 1 | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | run | off | p-nojson | **RED** | ⛔ no JSON object on stdout | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | run | off | p-badjson | **RED** | ⛔ stdout JSON unparseable: Expecting property name enclosed i… | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | run | off | p-notext | **RED** | ⛔ stdout JSON has no text field | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | run | off | p-degenerate | **RED** | ⛔ degenerate output (one character is >=90% of it): 'aaaaaaaa… | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | run | off | p-fellback | **RED** | ⛔ backend: asked gpu, ran cpu (fell_back=True) | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | run | off | p-wrong | **RED** | ❌ <answer>5</answer> | ✅ <answer>4</answer> | ✅ <answer>4</answer> | ✅ <answer>4</answer> | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | run | on | p-think-unknown | **RED** | ❌ <think> The answer is <answer>4</answer> | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | chat | off | p-chat | **RED** | ✅ <answer>4</answer> | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: not measured for the chat verb |
| syn-model-q4_k_m.gguf | chat | off | p-chat-noturn | **RED** | ⛔ no `Assistant:` turn in the transcript | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: not measured for the chat verb |
| syn-model-q4_k_m.gguf | run | off | p-noecho | **RED** | ⛔ missing: no row for this engine | ⛔ the echoed prompt was not found in stdout | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | run | off | p-notiming | **RED** | ⛔ missing: no row for this engine | ⛔ no end-of-turn timing line after the echoed prompt | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | run | off | p-think | **RED** | ⛔ missing: no row for this engine | ✅ <think>checking</think><answer>4</answer> | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | run | off | p-ollama-empty | **RED** | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ empty stdout | ⛔ missing: no row for this engine | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | run | off | p-hf-nodevice | **RED** | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ no reported.device: the gpu lane cannot be verified | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | run | off | p-hf-cpu | **RED** | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ device 'cpu' is not the gpu lane | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | run | off | p-hf-protocol | **RED** | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ protocol fault: frame 3 truncated | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | run | off | p-hf-notjson | **RED** | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ stdout is not the contract's JSON: Expecting value: line 1 … | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | run | off | p-vllm-unpinned | **RED** | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ not requested | ⛔ unpinned: the run's meta records no version for vllm, so a … | unmeasured: no llama.cpp tokenization row |
| syn-model-q4_k_m.gguf | run | off | p-missing-stdout | **RED** | ⛔ no JSON object on stdout | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ missing: no row for this engine | ⛔ not requested | ⛔ missing: no row for this engine | unmeasured: no llama.cpp tokenization row |

Deterministic rows (byte-equal or RED): {'RED': 3, 'GREEN': 2}

| kind | model | prompt | thinking | verdict | apr | references |
|---|---|---|---|---|---|---|
| tok | syn-model-q4_k_m.gguf | ctl-2plus2 |  | **GREEN** | produced | llama.cpp: =  |
| tok | syn-model-q4_k_m.gguf | p-exit |  | **RED** | missing: no row for this engine | llama.cpp: ≠ at None |
| tok | syn-model-q4_k_m.gguf | p-wrong |  | **RED** | produced | llama.cpp: ≠ at 2 |
| tmpl | syn-model-q4_k_m.gguf | ctl-2plus2 | off | **GREEN** | produced | llama.cpp: =  |
| tmpl | syn-model-q4_k_m.gguf | p-think-unknown | on | **RED** | refused: apr prints no rendering for this verb | llama.cpp: ≠ at None |

✅ correct · ❌ answered, wrong · ⛔ did not answer (reason shown).
Rates and token counts are in the JSON, as each engine reported them, and are not judged.
