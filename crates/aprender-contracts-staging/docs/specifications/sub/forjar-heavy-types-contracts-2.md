# Section 33: Forjar Heavy Types Contracts (continued)

> Parent: [pv-spec.md](../pv-spec.md) §33
>
> See also [forjar-heavy-types-contracts.md](forjar-heavy-types-contracts.md) (sections 33.1-33.7)

## 33.8 Secret Provider (secret-provider-v1)

### Type Surface

```rust
enum SecretProvider { Env, File, Sops, OnePassword }
struct SecretRef { provider: SecretProvider, key: String }
struct ResolvedSecret { value: String, ephemeral: bool }
```

### Equations

**provider_dispatch**: Every provider kind resolves or fails explicitly.

```
resolve: SecretRef -> Result<ResolvedSecret, SecretError>
  Env → env::var(key).map_err(|_| SecretError::NotFound)
  File → fs::read_to_string(key).map_err(|_| SecretError::NotFound)
  Sops → sops_decrypt(key).map_err(|e| SecretError::DecryptFailed(e))
  OnePassword → op_read(key).map_err(|e| SecretError::ProviderFailed(e))
```

**ephemeral_cleanup**: Ephemeral secrets are zeroed after use.

```
ephemeral: ResolvedSecret -> ()
  drop(secret) when ephemeral=true → memory zeroed (zeroize crate)
  Secret does not appear in: logs, error messages, debug output
```

**drift_detection**: Secret hash change detected between runs.

```
drift: (SecretRef, StoredHash, CurrentHash) -> DriftStatus
  stored_hash = current_hash → NoDrift
  stored_hash ≠ current_hash → Drifted(old, new)
  stored_hash = None → NewSecret
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| completeness | All providers handled | ∀ provider: resolve(provider, key) returns Result |
| frame | Ephemeral cleanup | drop(secret) zeroes memory, no log leakage |
| determinism | Drift detection | same secret → NoDrift, changed → Drifted |

### Falsification Tests

- **FALSIFY-SEC-001**: Env provider returns NotFound for absent variable
- **FALSIFY-SEC-002**: Ephemeral secret not present in panic backtrace
- **FALSIFY-SEC-003**: Changed secret file detected as Drifted on next run
- **FALSIFY-SEC-004**: Sops provider with wrong key returns DecryptFailed, not garbage

## 33.9 Copia Delta Sync (copia-delta-v1)

### Type Surface

```rust
struct BlockIndex { block_size: usize, hashes: Vec<Blake3Hash> }
struct Delta { new_blocks: Vec<(usize, Vec<u8>)>, removed_blocks: Vec<usize> }
struct SyncResult { bytes_transferred: u64, blocks_reused: u64, blocks_new: u64 }
```

### Equations

**delta_correctness**: Applying delta to old file produces new file.

```
apply_delta: (OldFile, Delta) -> NewFile
  apply(old, compute_delta(old, new)) = new
  byte-for-byte equality
```

**block_reuse**: Unchanged blocks are not retransferred.

```
reuse: (BlockIndex_old, BlockIndex_new) -> SyncResult
  ∀ i: hash_old[i] = hash_new[i] → block[i] reused (0 bytes transferred)
  bytes_transferred = sum(size(block) for block in delta.new_blocks)
```

**transfer_minimality**: Delta is minimal (no redundant blocks).

```
minimal: Delta -> bool
  ∀ (i, data) in new_blocks: hash(data) ≠ old_hashes[i]
  No block is included in new_blocks if it already matches
```

**identity_sync**: Identical files produce empty delta.

```
identity: (File, File) -> Delta
  compute_delta(f, f) = Delta { new_blocks: [], removed_blocks: [] }
  bytes_transferred = 0
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| equivalence | Delta correctness | apply(old, delta(old, new)) = new |
| conservation | Block reuse | unchanged blocks → 0 transfer |
| bound | Transfer minimality | no redundant blocks in delta |
| idempotency | Identity sync | delta(f, f) = empty |

### Falsification Tests

- **FALSIFY-COPIA-001**: Single changed block in 1MB file transfers only that block
- **FALSIFY-COPIA-002**: Identical files produce 0 bytes transferred
- **FALSIFY-COPIA-003**: Appended data produces delta with new blocks only at end
- **FALSIFY-COPIA-004**: Delta apply on 100MB file matches byte-for-byte
- **FALSIFY-COPIA-005**: Empty file delta to 1MB file transfers all blocks
- **FALSIFY-COPIA-006**: Block size change forces full resync (no hash reuse)

## 33.10 Sandbox Isolation (sandbox-isolation-v1)

### Type Surface

```rust
enum SandboxBackend { Seccomp, Overlayfs, Netns, Combined }
struct SandboxConfig { backend: SandboxBackend, allowed_paths: Vec<PathBuf>, network: bool }
struct SandboxResult { exit_code: i32, stdout: String, stderr: String, fs_changes: Vec<FsChange> }
enum FsChange { Created(PathBuf), Modified(PathBuf), Deleted(PathBuf) }
```

### Equations

**filesystem_isolation**: Sandboxed process cannot write outside allowed paths.

```
isolate: (SandboxConfig, Process) -> SandboxResult
  ∀ write in process.writes: write.path ∈ config.allowed_paths ∨ write.path ∈ overlay
  Host filesystem unmodified outside allowed_paths
```

**network_isolation**: Sandboxed process without network permission cannot connect.

```
network: (SandboxConfig, Process) -> bool
  config.network = false → all connect() calls fail with ENETUNREACH
  config.network = true → normal network access
```

**overlay_capture**: All filesystem mutations captured in overlay.

```
overlay: (Process, OverlayFs) -> Vec<FsChange>
  ∀ mutation by process: mutation ∈ overlay.upper_dir
  overlay.lower_dir unchanged (read-only)
  merge(lower, upper) = final filesystem state
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| frame | FS isolation | writes only to allowed_paths ∪ overlay |
| precondition | Network isolation | network=false → no outbound connections |
| completeness | Overlay captures all mutations | every write in overlay.upper |
| conservation | Lower dir read-only | overlay.lower unchanged after execution |

### Falsification Tests

- **FALSIFY-SBX-001**: Sandboxed `touch /etc/test` fails (outside allowed paths)
- **FALSIFY-SBX-002**: Sandboxed `curl` with network=false gets ENETUNREACH
- **FALSIFY-SBX-003**: Sandboxed file creation appears in overlay upper dir
- **FALSIFY-SBX-004**: Host /tmp unmodified after sandboxed write to /tmp
- **FALSIFY-SBX-005**: Overlay lower dir is byte-identical before and after execution

## 33.11 Summary Matrix

| Contract | Equations | Obligations | Falsification | Kani |
|----------|-----------|-------------|---------------|------|
| store-cas-v1 | 5 | 6 | 9 | 6 |
| oci-manifest-v1 | 4 | 5 | 5 | 5 |
| task-pipeline-v1 | 4 | 5 | 7 | 5 |
| event-rulebook-v1 | 3 | 4 | 5 | 4 |
| plugin-lifecycle-v1 | 3 | 3 | 4 | 3 |
| secret-provider-v1 | 3 | 3 | 4 | 3 |
| copia-delta-v1 | 4 | 4 | 6 | 4 |
| sandbox-isolation-v1 | 3 | 4 | 5 | 4 |
| **Total** | **29** | **34** | **45** | **34** |

This brings Forjar from 5 contracts (blake3, dag, execution, recipe, codegen) to **13 contracts**
covering all heavy type domains. Combined with the existing contracts, Forjar would have formal
verification across its entire critical path: hash → store → build → execute → sync → isolate.

## 33.12 Cross-Domain Contract Dependencies

```
blake3-state-v1 ←── store-cas-v1 (store paths use BLAKE3)
                ←── copia-delta-v1 (block hashes use BLAKE3)
                ←── oci-manifest-v1 (layer digests)

dag-ordering-v1 ←── task-pipeline-v1 (stage dependency order)

execution-safety-v1 ←── sandbox-isolation-v1 (sandboxed execution)
                    ←── task-pipeline-v1 (jidoka failure policy)

recipe-determinism-v1 ←── store-cas-v1 (recipe expansion → derivation)

codegen-dispatch-v1 ←── sandbox-isolation-v1 (generated scripts run in sandbox)
```
