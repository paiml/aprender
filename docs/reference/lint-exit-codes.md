# `apr *-lint` exit codes

Every `apr *-lint` command reads an already-captured observation — a JSON body,
an NDJSON stream, a CSV, a Prometheus exposition, or a trace directory — runs
pure classifiers over it, and reports whether the contract gates held. They are
documented identically and are meant to be driven from one CI harness, so they
use one exit-code convention.

| exit | meaning | who should look at it |
|-----:|---------|-----------------------|
| `0` | every gate the observation exercised passed | nobody |
| `3` | the named input does not exist | whoever wrote the capture step |
| `4` | the input exists but is not a usable observation: wrong kind of path, empty, unparseable, or containing none of the sections the gates need | whoever wrote the capture step |
| `5` | the observation was usable and a contract gate rejected it | whoever owns the system under test |
| `7` | the input exists but could not be read (OS error) | whoever owns the machine |

The distinction that matters is **4 versus 5**. Exit 4 means *your capture is
broken*; exit 5 means *the thing you captured violated its contract*. A CI
wrapper that cannot tell those apart will page the wrong person.

```bash
apr awq-lint --observation-file capture.json
case $? in
  0) echo "gates passed" ;;
  3|4|7) echo "capture step is broken — fix the harness"; exit 1 ;;
  5) echo "contract violated — fix the model/runtime"; exit 1 ;;
esac
```

## Diagnostics

* `3` prints `File not found: <path>`. It does **not** carry a `FALSIFY-…`
  stamp: no gate ran, so no contract was falsified. Stamps appear on `4` and `5`
  where a contract actually wanted something.
* `4` prints `Invalid input: <detail>`. It never says "Invalid APR format" —
  these commands read a captured observation, not an APR model file.
* `5` prints `Validation failed: <FALSIFY-… detail>`, one line per failed gate.

## Implementation

`crates/apr-cli/src/commands/lint_error.rs` defines `LintError` and its mapping
onto `CliError`; `CliError::exit_code_value()` is the single numeric mapping.
`crates/apr-cli/src/commands/lint_exit_convention_tests.rs` walks the whole
family and fails if any member disagrees.

Prior to this (through 0.63.0) the family shipped five codes for the same
conditions: the ten linters returning `Result<(), String>` collapsed
missing-input, unparseable-input and gate-rejection all onto `1`, the rest used
`3`/`4`/`5`, and `hang-trace-lint` returned `7` when `--trace-dir` named a
regular file. See issue #2404.
