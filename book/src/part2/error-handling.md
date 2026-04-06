# Error Handling

Batuta applies the Toyota Production System principle of Jidoka (autonomation) to its pipeline: when an error is detected, the pipeline stops immediately rather than propagating broken state to downstream phases.

## Validation Strategies

The `TranspilationPipeline` supports three error handling modes:

```rust
pub enum ValidationStrategy {
    StopOnError,      // Jidoka: halt on first failure
    ContinueOnError,  // Collect all errors, report at end
    None,             // Skip validation entirely
}
```

The default is `StopOnError`, which ensures no phase operates on invalid input.

## Stop-on-Error Flow

Each pipeline stage is validated after execution. If validation fails under `StopOnError`, the pipeline bails immediately:

```rust
if !validation_result.passed
    && self.validation == ValidationStrategy::StopOnError
{
    anyhow::bail!(
        "Validation failed for stage '{}': {}",
        stage.name(),
        validation_result.message
    );
}
```

This prevents a cascade of errors where Phase 3 tries to optimize code that Phase 2 failed to transpile correctly.

## Structured Error Types

Pipeline errors are wrapped with context using `anyhow::Context`:

```rust
ctx = stage
    .execute(ctx)
    .await
    .with_context(|| format!("Stage '{}' failed", stage.name()))?;
```

This produces error chains that trace back to the root cause:

```
Error: Stage 'Transpilation' failed
  Caused by: Tool 'depyler' failed with exit code 1
    stderr: Unsupported feature at line 42: async generators
```

## Validation Results

Each stage produces a `ValidationResult` that is accumulated in the pipeline context:

```rust
pub struct ValidationResult {
    pub stage: String,
    pub passed: bool,
    pub message: String,
    pub details: Option<serde_json::Value>,
}
```

The final `PipelineOutput` checks all results: `validation_passed` is true only if every stage passed.

## Workflow State on Failure

When a phase fails, `WorkflowState::fail_phase()` records the error and keeps `current_phase` pointed at the failed phase. The workflow does not advance. Downstream phases refuse to start until the prerequisite completes.

## Recovery Pattern

```bash
# Phase fails
$ batuta transpile
Error: Transpilation failed for auth.py

# Fix the issue, then retry (incremental)
$ batuta transpile
Success: All files transpiled

# Now Phase 3 will accept
$ batuta optimize
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
