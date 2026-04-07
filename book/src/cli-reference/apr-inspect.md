# apr inspect

```
Inspect model metadata, vocab, and structure

Usage: apr inspect [OPTIONS] <FILE>

Arguments:
  <FILE>  Path to .apr model file

Options:
      --vocab          Show vocabulary details
      --filters        Show filter/security details
      --weights        Show weight statistics
      --json           Output as JSON
  -v, --verbose        Verbose output
  -q, --quiet          Quiet mode (errors only)
      --offline        Disable network access (Sovereign AI compliance, Section 9)
      --skip-contract  Skip tensor contract validation (PMAT-237: use with diagnostic tooling)
  -h, --help           Print help
  -V, --version        Print version
```
