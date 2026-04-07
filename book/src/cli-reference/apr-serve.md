# apr serve

```
Inference server (plan/run)

Usage: apr serve [OPTIONS] <COMMAND>

Commands:
  plan  Pre-flight inference capacity plan (VRAM budget, roofline, contracts)
  run   Start inference server (REST API, streaming, metrics)
  help  Print this message or the help of the given subcommand(s)

Options:
      --json           Output as JSON
  -v, --verbose        Verbose output
  -q, --quiet          Quiet mode (errors only)
      --offline        Disable network access (Sovereign AI compliance, Section 9)
      --skip-contract  Skip tensor contract validation (PMAT-237: use with diagnostic tooling)
  -h, --help           Print help
  -V, --version        Print version
```
