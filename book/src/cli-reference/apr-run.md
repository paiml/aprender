<!-- PCU: cli-reference-apr-run | contract: contracts/apr-page-cli-reference-apr-run-v1.yaml -->
<!-- Example: cargo run -p aprender-core --example none -->
<!-- Status: enforced -->

# apr run

```
Run model directly (auto-download, cache, execute)

Usage: apr run [OPTIONS] <SOURCE> [PROMPT]

Arguments:
  <SOURCE>
          Model source: local path, hf://org/repo, or URL

  [PROMPT]
          Text prompt (positional): `apr run model.gguf "What is 2+2?"`

Options:
  -i, --input <INPUT>
          Input file (audio, text, etc.)

  -p, --prompt <PROMPT>
          Text prompt for generation (for LLM models)

  -n, --max-tokens <MAX_TOKENS>
          Maximum tokens to generate (default: 32)
          
          [default: 32]

      --stream
          Enable streaming output

  -l, --language <LANGUAGE>
          Language code (for ASR models)

  -t, --task <TASK>
          Task (transcribe, translate)

  -f, --format <FORMAT>
          Output format (text, json, srt, vtt)
          
          [default: text]

      --no-gpu
          Disable GPU acceleration (force CPU-only inference)

      --gpu
          Force GPU acceleration

      --offline
          Offline mode: block all network access (Sovereign AI compliance)

      --benchmark
          Benchmark mode: output performance metrics (tok/s, latency)

      --trace
          Enable inference tracing (APR-TRACE-001)

      --trace-steps <TRACE_STEPS>
          Trace specific steps only (comma-separated)

      --trace-verbose
          Verbose tracing (show tensor values)

      --trace-output <FILE>
          Save trace output to JSON file

      --trace-level <LEVEL>
          Trace detail level (none, basic, layer, payload, chrome) "chrome" outputs chrome://tracing
          JSON integrating layer trace + brick profile. F-CLIPARITY-01 / PMAT-386 /
          paiml/aprender#574
          
          [default: basic]

      --trace-payload
          Shorthand for --trace --trace-level payload (tensor value inspection)

      --profile
          Enable inline Roofline profiling (PMAT-SHOWCASE-METHODOLOGY-001)

      --chat
          Apply chat template for Instruct models (GAP-UX-001)
          
          Wraps prompt in ChatML format for Qwen2, LLaMA, Mistral Instruct models. Format:
          <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n

      --temperature <TEMPERATURE>
          Sampling temperature (0.0 = greedy, default: 0.0)
          
          [default: 0.0]

      --top-k <TOP_K>
          Top-k sampling (default: 1 = greedy)
          
          [default: 1]

      --top-p <TOP_P>
          Top-p nucleus sampling (0.0 = disabled). When set with --top-k, applies top-k first then
          top-p. F-CLIPARITY-01 / PMAT-381 / paiml/aprender#569

      --seed <SEED>
          RNG seed for deterministic sampling (default: 299792458, matching Candle) F-CLIPARITY-01 /
          PMAT-382 / paiml/aprender#570
          
          [default: 299792458]

      --repeat-penalty <REPEAT_PENALTY>
          Repetition penalty (1.0 = no penalty, >1.0 penalizes repeats) F-CLIPARITY-01 / PMAT-383 /
          paiml/aprender#571
          
          [default: 1.0]

      --repeat-last-n <REPEAT_LAST_N>
          Context window for repetition penalty (number of recent tokens to check) F-CLIPARITY-01 /
          PMAT-384 / paiml/aprender#571
          
          [default: 64]

      --split-prompt
          Process prompt tokens one-by-one instead of batched prefill. Useful for debugging prefill
          correctness (comparing per-token attention). F-CLIPARITY-01 / PMAT-385 /
          paiml/aprender#572

      --batch-jsonl <FILE>
          Batch mode: read prompts from JSONL, output results as JSONL. Model loads once, processes
          all prompts sequentially. Each input line: {"prompt": "...", "task_id": "..."} Chat
          template is applied automatically

  -v, --verbose
          Show verbose output (model loading, backend info)

      --backend <BACKEND>
          PMAT-488: Compute backend override (cuda, cpu, wgpu)

      --json
          Output as JSON

  -q, --quiet
          Quiet mode (errors only)

      --skip-contract
          Skip tensor contract validation (PMAT-237: use with diagnostic tooling)

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```
