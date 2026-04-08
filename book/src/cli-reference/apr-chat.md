<!-- PCU: cli-reference-apr-chat | contract: contracts/apr-page-cli-reference-apr-chat-v1.yaml -->
<!-- Example: cargo run -p aprender-core --example none -->
<!-- Status: enforced -->

# apr chat

```
Interactive chat with language model

Usage: apr chat [OPTIONS] <FILE>

Arguments:
  <FILE>  Path to .apr model file

Options:
      --temperature <TEMPERATURE>  Sampling temperature (0 = greedy, higher = more random) [default:
                                   0.7]
      --top-p <TOP_P>              Nucleus sampling threshold [default: 0.9]
      --max-tokens <MAX_TOKENS>    Maximum tokens to generate per response [default: 512]
      --system <SYSTEM>            System prompt to set model behavior
      --inspect                    Show inspection info (top-k probs, tokens/sec)
      --no-gpu                     Disable GPU acceleration (use CPU)
      --gpu                        Force GPU acceleration (requires CUDA)
      --trace                      Enable inference tracing (APR-TRACE-001)
      --trace-steps <TRACE_STEPS>  Trace specific steps only (comma-separated)
      --trace-verbose              Verbose tracing
      --trace-output <FILE>        Save trace output to JSON file
      --trace-level <LEVEL>        Trace detail level (none, basic, layer, payload) [default: basic]
      --profile                    Enable inline Roofline profiling (PMAT-SHOWCASE-METHODOLOGY-001)
      --backend <BACKEND>          PMAT-488: Compute backend override (cuda, cpu, wgpu)
      --json                       Output as JSON
  -v, --verbose                    Verbose output
  -q, --quiet                      Quiet mode (errors only)
      --offline                    Disable network access (Sovereign AI compliance, Section 9)
      --skip-contract              Skip tensor contract validation (PMAT-237: use with diagnostic
                                   tooling)
  -h, --help                       Print help
  -V, --version                    Print version
```
