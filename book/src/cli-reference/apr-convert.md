<!-- PCU: cli-reference-apr-convert | contract: contracts/apr-page-cli-reference-apr-convert-v1.yaml -->
<!-- Example: cargo run -p aprender-core --example none -->
<!-- Status: enforced -->

# apr convert

```
Convert/optimize model

Usage: apr convert [OPTIONS] --output <OUTPUT> <FILE>

Arguments:
  <FILE>  Path to .apr model file

Options:
      --quantize <QUANTIZE>  Quantize to format (int8, int4, fp16, q4k)
      --compress <COMPRESS>  Compress output (none, zstd, zstd-max, lz4)
  -o, --output <OUTPUT>      Output file path
  -f, --force                Force overwrite existing files
      --json                 Output as JSON
  -v, --verbose              Verbose output
  -q, --quiet                Quiet mode (errors only)
      --offline              Disable network access (Sovereign AI compliance, Section 9)
      --skip-contract        Skip tensor contract validation (PMAT-237: use with diagnostic tooling)
  -h, --help                 Print help
  -V, --version              Print version
```
