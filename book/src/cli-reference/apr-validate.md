# apr validate

```
Validate model integrity and quality

Usage: apr validate [OPTIONS] <FILE>

Arguments:
  <FILE>  Path to .apr model file

Options:
      --quality                Show 100-point quality assessment
      --strict                 Strict validation (fail on warnings)
      --min-score <MIN_SCORE>  Minimum score to pass (0-100)
      --json                   Output as JSON
  -v, --verbose                Verbose output
  -q, --quiet                  Quiet mode (errors only)
      --offline                Disable network access (Sovereign AI compliance, Section 9)
      --skip-contract          Skip tensor contract validation (PMAT-237: use with diagnostic
                               tooling)
  -h, --help                   Print help
  -V, --version                Print version
```
