# Section 32 (continued): PMAT Infrastructure Contracts

Version: 1.0
Status: proposed
Date: 2026-04-09

> Parent: [pv-spec.md](../pv-spec.md) §32
>
> See also: [pmat-infrastructure-contracts.md](pmat-infrastructure-contracts.md) (sections 32.1-32.9).

## 32.10 Configuration Schema (configuration-schema-v1)

### Equations

**unknown_key_rejection**: Unknown YAML/TOML keys are rejected, not silently ignored.

```
parse_config: (Input, Schema) -> Result<Config, UnknownKeyError>
  ∀ key in input: key ∈ schema.known_keys ∨ Err(UnknownKeyError(key))
```

**threshold_invariants**: Configuration values satisfy domain constraints.

```
validate: Config -> bool
  min <= max (for all range pairs)
  percentages ∈ [0, 100]
  timeouts > 0
  RUST_MIN_STACK >= 8388608
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| soundness | No unknown keys accepted | unknown key → error |
| precondition | Threshold invariants | min ≤ max, pct ∈ [0,100], timeout > 0 |

### Falsification Tests

- **FALSIFY-CFG-001**: YAML with typo key `complxity` (vs `complexity`) returns error
- **FALSIFY-CFG-002**: `--timeout 0` returns validation error, not division by zero
- **FALSIFY-CFG-003**: `RUST_MIN_STACK=1024` in CI config triggers warning

## 32.11 Compression Roundtrip (compression-roundtrip-v1)

### Equations

**lz4_roundtrip**: LZ4 compress then decompress is identity.

```
lz4: Vec<u8> -> bool
  decompress(compress(data)) = data
  len(compressed) <= len(data) + header_overhead
```

**sqlite_migration**: Schema migration preserves all data.

```
migrate: (DB_v1, Schema_v2) -> DB_v2
  ∀ row in DB_v1: row ∈ DB_v2
  new_columns have default values
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| roundtrip | LZ4 identity | decompress(compress(x)) = x |
| conservation | Migration lossless | row_count(v1) = row_count(v2) |

### Falsification Tests

- **FALSIFY-CMP-001**: Empty data LZ4 roundtrip produces empty data
- **FALSIFY-CMP-002**: 50MB index LZ4 roundtrip matches byte-for-byte
- **FALSIFY-CMP-003**: SQLite v1→v2 migration preserves function count exactly

## 32.12 Summary Matrix

| Contract | Equations | Obligations | Falsification | Kani |
|----------|-----------|-------------|---------------|------|
| cli-interface-v1 | 4 | 5 | 6 | 5 |
| mcp-protocol-v1 | 4 | 5 | 5 | 5 |
| graph-index-v1 | 5 | 6 | 7 | 6 |
| concurrency-safety-v1 | 3 | 3 | 3 | 3 |
| tracing-observability-v1 | 3 | 3 | 3 | 3 |
| memory-safety-v1 | 3 | 3 | 3 | 3 |
| state-machine-v1 | 3 | 3 | 4 | 3 |
| configuration-schema-v1 | 2 | 2 | 3 | 2 |
| compression-roundtrip-v1 | 2 | 2 | 3 | 2 |
| **Total** | **29** | **32** | **37** | **32** |

## 32.13 Work DBC (work-dbc-v1)

### Equations

**work_lifecycle**: Work item lifecycle state machine.

```
lifecycle: (WorkItem, Event) -> Result<WorkItem, LifecycleError>
  Draft → Active → Review → Merged
  Draft → Cancelled
  Active → Blocked → Active (recoverable)
  Review → Active (rework)
  No skip: Draft → Merged is INVALID
  Terminal: Merged, Cancelled are final
```

**falsifiable_claim**: Popperian falsification of quality claims.

```
claim: (Claim, Evidence) -> Verdict
  evidence matches prediction → Verified
  evidence contradicts → Falsified
  inconclusive → Blocked
  Falsified claim blocks work completion
```

**contract_profile**: 5-dimension quality scoring.

```
profile: (WorkItem, QualityConfig) -> ContractProfile
  score = w1*complexity + w2*coverage + w3*satd + w4*lint + w5*tdg
  Weights sum to 1.0, score in [0.0, 100.0]
  any_claim_falsified → Fail
```

**rescue_protocol**: Meyer §11 rescue strategies.

```
rescue: (WorkItem, Failure) -> RescueStrategy
  {Retry, Fallback, Escalate, Abandon}
  retries ≤ max_retries (bounded)
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| state_machine | Lifecycle valid transitions | No skip, terminal final |
| determinism | Claim verdict deterministic | same evidence → same verdict |
| bound | Profile score bounded | 0.0 ≤ score ≤ 100.0 |
| postcondition | Falsified blocks completion | falsified → Fail |
| termination | Rescue retry bounded | retries ≤ max_retries |
| conservation | Weights sum to unity | w1+w2+w3+w4+w5 = 1.0 |

### Falsification Tests

- **FALSIFY-WDB-001**: Draft→Merged skip returns LifecycleError
- **FALSIFY-WDB-002**: Falsified claim blocks Merged transition
- **FALSIFY-WDB-003**: Extreme inputs produce score in [0.0, 100.0]
- **FALSIFY-WDB-004**: After max_retries, rescue escalates (not infinite loop)
- **FALSIFY-WDB-005**: Same evidence produces same verdict twice
- **FALSIFY-WDB-006**: Merged item cannot transition to any state

## 32.14 Summary Matrix

| Contract | Equations | Obligations | Falsification | Kani |
|----------|-----------|-------------|---------------|------|
| cli-interface-v1 | 4 | 5 | 6 | 5 |
| mcp-protocol-v1 | 4 | 5 | 5 | 5 |
| graph-index-v1 | 5 | 6 | 7 | 6 |
| concurrency-safety-v1 | 3 | 3 | 3 | 3 |
| tracing-observability-v1 | 3 | 3 | 3 | 3 |
| memory-safety-v1 | 3 | 3 | 3 | 3 |
| state-machine-v1 | 3 | 3 | 4 | 3 |
| configuration-schema-v1 | 2 | 2 | 3 | 2 |
| compression-roundtrip-v1 | 2 | 2 | 3 | 2 |
| work-dbc-v1 | 4 | 6 | 6 | 6 |
| **Total** | **33** | **38** | **43** | **38** |

This brings PMAT from 4 contracts (TDG, comply, score, context) to **14 contracts** covering all
critical infrastructure boundaries including the work DBC system itself.
