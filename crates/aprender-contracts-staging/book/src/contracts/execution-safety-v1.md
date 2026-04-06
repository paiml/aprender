# execution-safety-v1

**Version:** 1.0.0

Execution safety — atomic writes and jidoka failure policy

## References

- Lampson & Sturgis (1979) Crash Recovery in a Distributed Data Storage System
- Ohno (1988) Toyota Production System — Jidoka (autonomation)

## Equations

### atomic_write

$$
save_lock(dir, lock) = write(tmp) ∘ rename(tmp, target)
$$

**Domain:** $state_dir \in Path, lock \in StateLock$

**Codomain:** $Result<(), String>$

**Invariants:**

- $No temp file remains after successful save$
- $Target file exists after successful save$
- $Parent directories are created if absent$

### jidoka_stop

$$
on_failure(policy, error) = if policy = StopOnFirst then halt else continue
$$

**Domain:** $policy \in {StopOnFirst, ContinueIndependent}, error \in String$

**Codomain:** $bool (should_stop)$

**Invariants:**

- $StopOnFirst policy returns true on failure$
- $ContinueIndependent policy returns false on failure$
- $Failed resource is recorded in lock regardless of policy$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Atomic write leaves no temp file | $\forall save_lock(d, l) = Ok(()): ¬exists(d/l.machine/state.lock.yaml.tmp)$ |
| 2 | invariant | Jidoka dispatches correctly | $record_failure(StopOnFirst, ...) = true ∧ record_failure(Continue, ...) = false$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-ES-001 | Atomic write | After save_lock succeeds, no .tmp file exists | Temp file cleanup failed or rename not atomic |
| FALSIFY-ES-002 | Jidoka stop | record_failure with StopOnFirst returns true | Jidoka policy dispatch broken |
| FALSIFY-ES-003 | Jidoka continue | record_failure with ContinueIndependent returns false | Continue policy incorrectly stops |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-ES-001 | Atomic write leaves no temp file | 8 | bounded_int |
| KANI-ES-002 | Jidoka dispatches correctly | 2 | exhaustive |

## QA Gate

**Execution Safety Contract** (F-ES-001)

Atomic writes and jidoka policy quality gate

**Checks:** atomic_write, jidoka_stop, jidoka_continue

**Pass criteria:** All 3 falsification tests pass

