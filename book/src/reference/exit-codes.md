# Exit Codes

Renacer uses distinct exit codes to indicate different outcomes, enabling programmatic response in scripts and CI/CD pipelines.

## Tracer Exit Codes

When running `renacer -- <command>`:

| Code | Meaning |
|------|---------|
| 0 | Success (traced program exited successfully) |
| 1 | Renacer error (invalid arguments, ptrace failure, etc.) |
| N | Traced program exit code (if program exited with code N) |

### Examples

```bash
# Success - ls exits with 0
renacer -- ls
echo $?  # 0

# Command fails - ls on nonexistent file
renacer -- ls /nonexistent
echo $?  # 2 (ls exit code)

# Renacer error - invalid option
renacer --invalid-option
echo $?  # 1
```

## Validate Subcommand Exit Codes

When running `renacer validate`:

| Code | Constant | Meaning | CI Response |
|------|----------|---------|-------------|
| 0 | `Passed` | Validation passed | Continue |
| 1 | `Failed` | Regression detected | Block deployment |
| 2 | `BaselineNotFound` | Baseline directory not found | Check paths |
| 3 | `InvalidBaseline` | Invalid/corrupt baseline | Regenerate baseline |
| 4 | `CommandError` | Command execution failed | Check command |
| 5 | `ConfigError` | Invalid configuration | Check options |

### CI/CD Usage

```bash
#!/bin/bash
set -e

renacer validate --baseline ./golden -- ./myapp
status=$?

case $status in
    0) echo "Validation passed" ;;
    1) echo "REGRESSION DETECTED - blocking deployment"; exit 1 ;;
    2) echo "Baseline not found - generating...";
       renacer validate --generate ./golden -- ./myapp ;;
    3) echo "Invalid baseline - regenerating...";
       rm -rf ./golden
       renacer validate --generate ./golden -- ./myapp ;;
    4) echo "Command failed to execute"; exit 1 ;;
    5) echo "Configuration error"; exit 1 ;;
esac
```

### GitHub Actions

```yaml
- name: Validate traces
  run: renacer validate --baseline ./golden -- ./myapp
  continue-on-error: false  # Exit 1 fails the workflow
```

## Signal-Based Exit Codes

When the traced program is terminated by a signal:

| Exit Code | Signal | Meaning |
|-----------|--------|---------|
| 128 + N | SIGTERM (15) | 143 = Terminated |
| 128 + N | SIGKILL (9) | 137 = Killed |
| 128 + N | SIGSEGV (11) | 139 = Segmentation fault |
| 128 + N | SIGABRT (6) | 134 = Aborted |

### Example

```bash
# Program crashes with SIGSEGV
renacer -- ./crash_program
echo $?  # 139 (128 + 11)
```

## Exit Code Constants (Rust API)

```rust,ignore
use renacer::validate::ValidateExitCode;

fn handle_validation_result(code: ValidateExitCode) {
    match code {
        ValidateExitCode::Passed => println!("OK"),
        ValidateExitCode::Failed => {
            eprintln!("Regression detected!");
            std::process::exit(1);
        }
        ValidateExitCode::BaselineNotFound => {
            eprintln!("Baseline not found");
            std::process::exit(2);
        }
        ValidateExitCode::InvalidBaseline => {
            eprintln!("Invalid baseline");
            std::process::exit(3);
        }
        ValidateExitCode::CommandError => {
            eprintln!("Command error");
            std::process::exit(4);
        }
        ValidateExitCode::ConfigError => {
            eprintln!("Config error");
            std::process::exit(5);
        }
    }
}
```

## Best Practices

### 1. Always Check Exit Codes

```bash
if ! renacer validate --baseline ./golden -- ./myapp; then
    echo "Validation failed with code $?"
    exit 1
fi
```

### 2. Distinguish Tracer vs Program Failures

Exit code 1 could mean:
- Renacer internal error (when tracing)
- Regression detected (when validating)

Use distinct handling:

```bash
# Tracer mode
renacer -- ./myapp
if [ $? -eq 1 ]; then
    echo "Renacer or program error"
fi

# Validate mode - exit 1 is specifically regression
renacer validate --baseline ./golden -- ./myapp
if [ $? -eq 1 ]; then
    echo "REGRESSION DETECTED"
fi
```

### 3. Log Exit Codes for Debugging

```bash
renacer validate --baseline ./golden -- ./myapp
echo "Validate exit code: $?" >> validation.log
```

## Related

- [CLI Reference](./cli.md) - Full command documentation
- [Golden Trace Validation](../advanced/golden-trace-validation.md) - Validation details
