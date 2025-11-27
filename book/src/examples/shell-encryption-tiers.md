# Case Study: Model Encryption Tiers (Plain → Compressed → At-Rest → Homomorphic)

Four protection levels for shell completion models, each with distinct security/performance tradeoffs.

## The Four Tiers

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Model Protection Tiers                                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Tier 1: Plain (.apr)                                              │
│  ├─ Security: None (weights readable)                              │
│  ├─ Performance: Baseline                                          │
│  └─ Use: Development, open-source models                           │
│                                                                     │
│  Tier 2: Compressed (.apr + zstd)                                  │
│  ├─ Security: Obfuscation only                                     │
│  ├─ Performance: FASTER (smaller I/O, better cache)                │
│  └─ Use: Distribution, CDN deployment                              │
│                                                                     │
│  Tier 3: At-Rest Encrypted (.apr + AES-256-GCM)                    │
│  ├─ Security: Protected on disk                                    │
│  ├─ Performance: ~10ms decrypt overhead                            │
│  └─ Use: Commercial IP, compliance (HIPAA/SOC2)                    │
│                                                                     │
│  Tier 4: Homomorphic (.apr + CKKS/BFV)                             │
│  ├─ Security: Protected during computation                         │
│  ├─ Performance: ~100x overhead                                    │
│  └─ Use: Zero-trust inference, untrusted servers                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Comparison

| Tier | Size | Load Time | Inference | Weights Exposed | Query Exposed |
|------|------|-----------|-----------|-----------------|---------------|
| Plain | 7.0 MB | 45ms | 0.5ms | Yes | Yes |
| Compressed | 503 KB | 35ms | 0.5ms | Yes | Yes |
| At-Rest | 503 KB | 55ms | 0.5ms | No (on disk) | Yes (in RAM) |
| Homomorphic | 2.5 GB | 3s | 50ms | No | No |

## Tier 1: Plain Model

Default format. Fast, no protection.

```bash
# Train and save plain model
aprender-shell train --history ~/.bash_history --output model.apr

# Inspect
aprender-shell inspect model.apr
# Format: .apr v1 (plain)
# Size: 7.0 MB
# Encryption: None
```

```rust
use aprender_shell::NgramModel;

let model = NgramModel::train(&history, 3)?;
model.save("model.apr")?;

// Load - direct deserialization
let loaded = NgramModel::load("model.apr")?;
```

**When to use:**
- Development and testing
- Open-source model sharing
- Maximum performance required

## Tier 2: Compressed Model

14x smaller. **Faster in practice** due to I/O reduction.

```bash
# Train with compression
aprender-shell train --history ~/.bash_history --output model.apr --compress

# Inspect
aprender-shell inspect model.apr
# Format: .apr v1 (compressed)
# Size: 503 KB (14x reduction)
# Compression: zstd level 3
```

### Real-World Benchmarks (depyler)

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Performance: Plain vs Compressed (503KB model)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Metric          │ Plain (7MB) │ Compressed (503KB) │ Winner       │
│  ─────────────────┼─────────────┼────────────────────┼─────────────│
│  Disk read       │ 45ms        │ 25ms               │ Compressed  │
│  Decompress      │ 0ms         │ 10-20ms            │ Plain       │
│  Total load      │ 45ms        │ 35ms               │ Compressed  │
│  Predictions/sec │ 3,800       │ 4,140              │ Compressed  │
│                                                                     │
│  Why compressed wins:                                               │
│  • Smaller file = faster disk reads                                │
│  • Fits in CPU L3 cache (503KB < 8MB typical L3)                   │
│  • Less memory bandwidth pressure                                   │
│  • SSD/NVMe still I/O bound at these sizes                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```rust
use aprender::format::{Compression, SaveOptions};

let options = SaveOptions::default()
    .with_compression(Compression::ZstdDefault);

model.save_with_options("model.apr", options)?;
```

**When to use:**
- Production deployment (default choice)
- CDN distribution
- Embedded in binaries (`include_bytes!`)
- Mobile/edge devices

## Tier 3: At-Rest Encryption

AES-256-GCM with Argon2id key derivation. Protects IP on disk.

```bash
# Train with encryption
aprender-shell train --history ~/.bash_history --output model.apr --password
# Enter password: ********
# Confirm password: ********

# Or via environment variable (CI/CD)
APRENDER_PASSWORD=secret aprender-shell train --output model.apr --password

# Inspect (no password needed for metadata)
aprender-shell inspect model.apr
# Format: .apr v2 (encrypted)
# Size: 503 KB
# Encryption: AES-256-GCM + Argon2id
# Encrypted: Yes

# Load requires password
aprender-shell suggest --password "git com"
# Enter password: ********
# → commit, checkout, clone
```

```rust
use aprender_shell::NgramModel;

// Save encrypted
model.save_encrypted("model.apr", "my-strong-password")?;

// Load encrypted
let loaded = NgramModel::load_encrypted("model.apr", "my-strong-password")?;

// Check if encrypted without loading
if NgramModel::is_encrypted("model.apr")? {
    println!("Password required");
}
```

### Security Properties

| Property | Value |
|----------|-------|
| Key derivation | Argon2id (memory-hard, GPU-resistant) |
| Cipher | AES-256-GCM (authenticated) |
| Salt | 16 bytes random per file |
| Nonce | 12 bytes random per encryption |
| Tag | 16 bytes (integrity verification) |

**Threat model:**
- ✅ Protects against disk theft
- ✅ Protects against unauthorized file access
- ✅ Detects tampering (authenticated encryption)
- ❌ Weights exposed in RAM during inference
- ❌ Query patterns visible to process with RAM access

**When to use:**
- Commercial model distribution
- Compliance requirements (SOC2, HIPAA data-at-rest)
- Shared storage environments

## Tier 4: Homomorphic Encryption

Compute on encrypted data. **Model weights never decrypted.**

```bash
# Generate HE keys (one-time setup)
aprender-shell keygen --output ~/.config/aprender/
# Generated: public.key, secret.key, relin.key

# Train with homomorphic encryption
aprender-shell train --history ~/.bash_history --output model.apr \
    --homomorphic --public-key ~/.config/aprender/public.key

# Inspect
aprender-shell inspect model.apr
# Format: .apr v3 (homomorphic)
# Size: 2.5 GB
# Encryption: CKKS/BFV hybrid (128-bit security)
# HE Parameters: N=8192, Q=218 bits

# Suggest (encrypted inference)
aprender-shell suggest --homomorphic "git com"
# → commit, checkout, clone
# (inference performed on ciphertext, decrypted client-side)
```

### Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Homomorphic Inference Flow                                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Client (trusted)              Server (untrusted)                  │
│  ┌─────────────────┐          ┌─────────────────┐                 │
│  │ secret.key      │          │ public.key      │                 │
│  │ (never shared)  │          │ model.apr (HE)  │                 │
│  └────────┬────────┘          └────────┬────────┘                 │
│           │                            │                           │
│  Step 1: Encrypt query                 │                           │
│  ┌─────────────────┐                   │                           │
│  │ E("git com")    │ ─────────────────►│                           │
│  │ (256 KB)        │                   │                           │
│  └─────────────────┘                   │                           │
│                                        ▼                           │
│                            Step 2: HE Inference                    │
│                            ┌─────────────────┐                     │
│                            │ N-gram lookup   │                     │
│                            │ Score compute   │                     │
│                            │ (on ciphertext) │                     │
│                            └────────┬────────┘                     │
│                                     │                              │
│  Step 3: Decrypt result            │                              │
│  ┌─────────────────┐               │                              │
│  │ D(E(results))   │◄──────────────┘                              │
│  │ → [commit,      │  E(["commit", "checkout", "clone"])          │
│  │    checkout,    │  (encrypted suggestions)                      │
│  │    clone]       │                                               │
│  └─────────────────┘                                               │
│                                                                     │
│  What server sees: Random-looking ciphertext                       │
│  What server learns: Nothing (IND-CPA secure)                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Performance Reality

```text
┌─────────────────────────────────────────────────────────────────────┐
│ HE Performance Breakdown                                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Operation          │ Time      │ Notes                            │
│  ───────────────────┼───────────┼──────────────────────────────────│
│  Key generation     │ 5s        │ One-time setup                   │
│  Model encryption   │ 60s       │ One-time per model               │
│  Query encryption   │ 15ms      │ Per query (client)               │
│  HE inference       │ 50ms      │ Per query (server)               │
│  Result decryption  │ 5ms       │ Per query (client)               │
│  ───────────────────┼───────────┼──────────────────────────────────│
│  Total per query    │ ~70ms     │ vs 0.5ms plaintext (140x)        │
│                                                                     │
│  Memory:                                                            │
│  • Public key: 1.6 MB                                              │
│  • Relin keys: 50 MB                                               │
│  • Model (HE): 2.5 GB (vs 503KB compressed)                        │
│  • Query ciphertext: 256 KB                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### API

```rust
use aprender_shell::{NgramModel, HeContext, SecurityLevel};

// Setup (one-time)
let context = HeContext::new(SecurityLevel::Bit128)?;
let (public_key, secret_key) = context.generate_keys()?;

// Encrypt model (one-time)
let model = NgramModel::train(&history, 3)?;
let he_model = model.to_homomorphic(&public_key)?;
he_model.save("model.apr")?;

// Inference (per query)
let encrypted_query = context.encrypt_query("git com", &public_key)?;
let encrypted_result = he_model.suggest_encrypted(&encrypted_query)?;
let suggestions = context.decrypt_result(&encrypted_result, &secret_key)?;
```

**When to use:**
- Zero-trust cloud deployment
- Model IP protection on untrusted servers
- Privacy-preserving ML-as-a-Service
- Regulatory requirements (query privacy)

## Choosing a Tier

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Decision Tree                                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Is model IP sensitive?                                            │
│  ├─ No → Is distribution size important?                           │
│  │       ├─ No → Tier 1 (Plain)                                    │
│  │       └─ Yes → Tier 2 (Compressed) ← DEFAULT                    │
│  │                                                                  │
│  └─ Yes → Do you trust the inference environment?                  │
│           ├─ Yes (your servers) → Tier 3 (At-Rest)                 │
│           └─ No (cloud/third-party) → Tier 4 (Homomorphic)         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

| Requirement | Recommended Tier |
|-------------|------------------|
| Open-source distribution | Tier 2 (Compressed) |
| Commercial CLI tool | Tier 3 (At-Rest) |
| SaaS model serving | Tier 3 (At-Rest) |
| Untrusted cloud inference | Tier 4 (Homomorphic) |
| Privacy-preserving API | Tier 4 (Homomorphic) |
| Maximum performance | Tier 2 (Compressed) |

## CLI Reference

```bash
# Tier 1: Plain
aprender-shell train -o model.apr

# Tier 2: Compressed (recommended default)
aprender-shell train -o model.apr --compress

# Tier 3: At-Rest Encrypted
aprender-shell train -o model.apr --compress --password

# Tier 4: Homomorphic
aprender-shell keygen -o ~/.config/aprender/
aprender-shell train -o model.apr --homomorphic --public-key ~/.config/aprender/public.key

# Inspect any tier
aprender-shell inspect model.apr

# Convert between tiers
aprender-shell convert model-plain.apr model-encrypted.apr --password
aprender-shell convert model-plain.apr model-he.apr --homomorphic --public-key key.pub
```

## Toyota Way Alignment

| Principle | Implementation |
|-----------|----------------|
| **Jidoka** | Each tier builds in quality (checksums, authenticated encryption, HE proofs) |
| **Kaizen** | Progressive security: start simple, upgrade as needed |
| **Genchi Genbutsu** | Benchmarks from real workloads (depyler 4,140 pred/sec) |
| **Poka-yoke** | Type system prevents mixing tiers (`Plaintext<T>` vs `Ciphertext<T>`) |
| **Heijunka** | Tier 2 compression smooths I/O load |

## Further Reading

- [Homomorphic Encryption Specification](../../../docs/specifications/homomorphic-encryption-spec.md)
- [.apr Format Deep Dive](./apr-format-deep-dive.md)
- [Model Format Case Study](./model-format.md)
