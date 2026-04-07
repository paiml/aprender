# Sovereign Dataset Sharing Specification

**Version**: 1.2
**Status**: Draft
**Author**: alimentar team
**Date**: 2024-11

## Abstract

This specification defines a sovereign-first dataset sharing system for alimentar that operates without mandatory external services. All functionality must work in air-gapped environments, with optional federation for connected deployments.

## 1. Design Principles

### 1.1 Sovereignty Requirements

1. **No mandatory cloud dependency** - All features work offline
2. **Self-hostable** - Single binary deployment, no external databases
3. **Pure Rust** - Zero C/C++ dependencies (WASM-compatible)
4. **Federated optional** - Peer-to-peer sharing when connected
5. **Cryptographic integrity** - Content-addressed storage with signatures

### 1.2 Non-Goals

- Centralized hub (HuggingFace model)
- OAuth/external authentication providers
- Cloud-only features
- Native code dependencies

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Sovereign Registry                        │
├─────────────────────────────────────────────────────────────┤
│  Local Index      │  Content Store    │  Federation Layer   │
│  ───────────      │  ─────────────    │  ─────────────────  │
│  • SQLite/redb    │  • CDC Chunking   │  • mDNS discovery   │
│  • Metadata       │  • Deduplication  │  • P2P sync         │
│  • Search index   │  • Compression    │  • Signed manifests │
└─────────────────────────────────────────────────────────────┘
```

## 3. Content-Addressed Storage

### 3.1 Dataset Identifiers & Deduplication

Datasets are identified by cryptographic hash. To eliminate the **Muda (waste)** of redundant storage, files are split using **Content-Defined Chunking (CDC)** (e.g., FastCDC) rather than fixed-size blocks. This ensures that small insertions do not shift boundaries and invalidate downstream chunks [21].

```rust
/// Content identifier using BLAKE3 hash
pub struct ContentId([u8; 32]);

impl ContentId {
    pub fn from_bytes(data: &[u8]) -> Self {
        Self(blake3::hash(data).into())
    }

    pub fn to_string(&self) -> String {
        format!("alimentar:{}", hex::encode(self.0))
    }
}

/// Chunk definition for deduplication
pub struct Chunk {
    pub offset: u64,
    pub length: u32,
    pub hash: ContentId,
}
```

### 3.2 Manifest Format

```rust
/// Dataset manifest with cryptographic signatures
#[derive(Serialize, Deserialize)]
pub struct DatasetManifest {
    /// Schema version
    pub version: u32,
    /// Human-readable name
    pub name: String,
    /// Semantic version
    pub dataset_version: semver::Version,
    /// Content hash of data files
    pub content_id: ContentId,
    /// Root Merkle hash of the chunk tree
    pub chunk_tree_root: ContentId,
    /// Arrow schema serialized
    pub schema: Vec<u8>,
    /// Number of rows
    pub num_rows: u64,
    /// Compressed size in bytes
    pub size_bytes: u64,
    /// Optional description (Markdown)
    pub description: Option<String>,
    /// License identifier (SPDX)
    pub license: Option<String>,
    /// Creation timestamp (RFC 3339)
    pub created_at: String,
    /// Ed25519 signature of manifest
    pub signature: Option<[u8; 64]>,
    /// Public key of signer
    pub signer: Option<[u8; 32]>,
}
```

## 4. Local Registry

### 4.1 Embedded Database

Use redb (pure Rust) for local metadata storage [2][11]. No external database required.

```rust
pub struct LocalRegistry {
    db: redb::Database,
    content_dir: PathBuf,
}

impl LocalRegistry {
    /// Index a local dataset
    pub fn index(&self, path: &Path) -> Result<ContentId>;

    /// Search by name, tags, or description
    pub fn search(&self, query: &str) -> Result<Vec<DatasetManifest>>;

    /// Get dataset by content ID
    pub fn get(&self, id: &ContentId) -> Result<Option<DatasetManifest>>;

    /// List all indexed datasets
    pub fn list(&self) -> Result<Vec<DatasetManifest>>;

    /// Remove from index (not content)
    pub fn unindex(&self, id: &ContentId) -> Result<()>;
}
```

### 4.2 Full-Text Search

Implement inverted index for dataset discovery using tantivy (pure Rust) [3].

```rust
pub struct SearchIndex {
    index: tantivy::Index,
}

impl SearchIndex {
    pub fn add(&mut self, manifest: &DatasetManifest) -> Result<()>;
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<ContentId>>;
    pub fn search_by_schema(&self, field_name: &str) -> Result<Vec<ContentId>>;
}
```

## 5. Dataset Cards

### 5.1 Markdown Documentation

Each dataset includes optional documentation in a standardized format [17]:

```markdown
---
name: my-dataset
version: 1.0.0
license: MIT
tags: [nlp, classification, english]
---

# My Dataset

## Description

Brief description of the dataset.

## Schema

| Column | Type | Description |
|--------|------|-------------|
| text | string | Input text |
| label | int32 | Class label |

## Usage

```rust
let dataset = registry.pull("alimentar:abc123...")?;
```

## Citation

```bibtex
@dataset{...}
```
```

### 5.2 Card Parser

```rust
pub struct DatasetCard {
    pub frontmatter: CardFrontmatter,
    pub content: String,
}

#[derive(Deserialize)]
pub struct CardFrontmatter {
    pub name: String,
    pub version: String,
    pub license: Option<String>,
    pub tags: Vec<String>,
}

impl DatasetCard {
    pub fn parse(markdown: &str) -> Result<Self>;
    pub fn render_html(&self) -> String;
}
```

## 6. Cryptographic Signing

### 6.1 Key Management

Use Ed25519 for signing (via ed25519-dalek, pure Rust) [4].

```rust
pub struct KeyPair {
    signing_key: ed25519_dalek::SigningKey,
}

impl KeyPair {
    /// Generate new keypair
    pub fn generate() -> Self;

    /// Load from file (encrypted with passphrase)
    pub fn load(path: &Path, passphrase: &str) -> Result<Self>;

    /// Save to file (encrypted)
    pub fn save(&self, path: &Path, passphrase: &str) -> Result<()>;

    /// Sign a manifest
    pub fn sign(&self, manifest: &mut DatasetManifest);

    /// Get public key
    pub fn public_key(&self) -> [u8; 32];
}

/// Verify manifest signature
pub fn verify_signature(manifest: &DatasetManifest) -> Result<bool>;
```

### 6.2 Trust Model

**Poka-Yoke:** To prevent accidental acceptance of unknown keys, the system enforces a safe default for `Unknown` trust levels.

```rust
/// Trust store for known publishers [12]
pub struct TrustStore {
    trusted_keys: HashMap<[u8; 32], TrustedPublisher>,
}

pub struct TrustedPublisher {
    pub name: String,
    pub public_key: [u8; 32],
    pub trust_level: TrustLevel,
}

pub enum TrustLevel {
    /// Fully trusted - auto-accept
    Full,
    /// Prompt before accepting
    Prompt,
    /// Signature valid but unknown publisher - DEFAULT: Treat as untrusted/sandbox
    Unknown,
}
```

## 7. Federation Protocol

### 7.1 Local Network Discovery

Use mDNS for zero-configuration peer discovery (pure Rust mdns-sd) [5]. To eliminate the **waste of futile queries**, peer announcements include a compressed **Cuckoo Filter** [22] of available content.

```rust
pub struct PeerDiscovery {
    mdns: mdns_sd::ServiceDaemon,
}

impl PeerDiscovery {
    /// Announce this node as a registry
    pub fn announce(&self, port: u16) -> Result<()>;

    /// Discover peers on local network
    pub fn discover(&self) -> Result<Vec<PeerInfo>>;

    /// Stop announcing
    pub fn shutdown(&self);
}

pub struct PeerInfo {
    pub address: SocketAddr,
    pub node_id: [u8; 32],
    /// Compressed approximate set membership filter
    pub content_filter: Vec<u8>,
    pub datasets_available: u64,
}
```

### 7.2 Sync Protocol

Uses efficient set reconciliation (e.g., IBLT) [14] and **chunk-level synchronization** to transfer only modified parts of datasets, adhering to Just-In-Time principles.

```rust
pub enum SyncMessage {
    /// Request manifest reconciliation (e.g. IBLT or Merkle summary)
    ReconcileRequest { summary: Vec<u8> },
    /// Response with differences
    ReconcileResponse { missing: Vec<ContentId>, new: Vec<DatasetManifest> },
    /// Request specific chunk (CDC addressed)
    PullChunkRequest { chunk_id: ContentId },
    /// Response with chunk data [23]
    PullChunkResponse { data: Vec<u8> },
    /// Announce new dataset
    Announce { manifest: DatasetManifest },
}

pub struct SyncClient {
    connection: quinn::Connection,
}

impl SyncClient {
    pub async fn list_remote(&self) -> Result<Vec<DatasetManifest>>;
    pub async fn pull(&self, id: &ContentId, dest: &Path) -> Result<()>;
    pub async fn announce(&self, manifest: &DatasetManifest) -> Result<()>;
}
```

### 7.3 Conflict Resolution

Content-addressed storage ensures no conflicts - same content = same ID. For metadata (names, tags), use last-writer-wins with vector clocks [7][20].

## 8. Export/Import

### 8.1 Portable Archives

Export datasets as self-contained archives for sneakernet transfer:

```rust
pub struct DatasetArchive;

impl DatasetArchive {
    /// Create portable archive (.alimentar)
    pub fn export(
        registry: &LocalRegistry,
        content_id: &ContentId,
        dest: &Path,
    ) -> Result<()>;

    /// Import from archive
    pub fn import(
        registry: &mut LocalRegistry,
        archive: &Path,
    ) -> Result<ContentId>;
}
```

Archive format (zstd-compressed tar):
```
dataset.alimentar
├── manifest.json       # Signed manifest
├── card.md            # Dataset documentation
├── data/
│   └── *.chunk        # Deduplicated chunks
└── signature.bin      # Detached signature
```

### 8.2 Batch Operations

```rust
impl LocalRegistry {
    /// Export multiple datasets
    pub fn export_batch(
        &self,
        ids: &[ContentId],
        dest: &Path,
    ) -> Result<PathBuf>;

    /// Import directory of archives
    pub fn import_batch(&mut self, dir: &Path) -> Result<Vec<ContentId>>;
}
```

## 9. CLI Interface

```bash
# Local operations
alimentar registry index ./my-dataset.parquet
alimentar registry list
alimentar registry search "sentiment classification"
alimentar registry info alimentar:abc123...
alimentar registry card alimentar:abc123...

# Export/import
alimentar registry export alimentar:abc123... -o dataset.alimentar
alimentar registry import dataset.alimentar

# Signing
alimentar registry keygen -o ~/.alimentar/key.enc
alimentar registry sign alimentar:abc123...
alimentar registry verify alimentar:abc123...

# Federation (optional)
alimentar registry serve --port 8765
alimentar registry peers
alimentar registry sync --peer 192.168.1.100:8765
alimentar registry pull alimentar:abc123... --peer 192.168.1.100:8765
```

## 10. Implementation Plan

### Phase 1: Local Registry
- [ ] Content-addressed storage with **CDC**
- [ ] Manifest format
- [ ] redb-based index
- [ ] Basic search

### Phase 2: Documentation
- [ ] Dataset cards
- [ ] Markdown parser
- [ ] Schema introspection

### Phase 3: Signing
- [ ] Key generation
- [ ] Manifest signing
- [ ] Trust store

### Phase 4: Export/Import
- [ ] Archive format
- [ ] Batch operations
- [ ] Integrity verification

### Phase 5: Federation
- [ ] mDNS discovery with **Cuckoo Filters**
- [ ] QUIC transport
- [ ] Sync protocol (Set Reconciliation)

## 11. Security Considerations

1. **Content integrity**: BLAKE3 hashes prevent tampering [8]
2. **Authentication**: Ed25519 signatures verify publisher identity
3. **Transport security**: QUIC provides TLS 1.3 encryption [15]
4. **Key storage**: Encrypted at rest with Argon2 KDF [9]
5. **Trust boundaries**: Explicit trust store, no implicit trust [12]

## 12. References

[1] Benet, J. (2014). "IPFS - Content Addressed, Versioned, P2P File System." arXiv:1407.3561.

[2] Lennon, C. (2023). "redb: A simple, portable, high-performance, ACID, embedded key-value store."

[3] Clément, P. (2019). "Tantivy: A Full-Text Search Engine Library Written in Rust." SIGIR '19.

[4] Bernstein, D.J., et al. (2012). "High-speed high-security signatures." Journal of Cryptographic Engineering.

[5] Cheshire, S. & Krochmal, M. (2013). "Multicast DNS." RFC 6762.

[6] Iyengar, J. & Thomson, M. (2021). "QUIC: A UDP-Based Multiplexed and Secure Transport." RFC 9000.

[7] Shapiro, M., et al. (2011). "Conflict-Free Replicated Data Types." SSS '11.

[8] O'Connor, J., et al. (2020). "BLAKE3: One Function, Fast Everywhere."

[9] Biryukov, A., Dinu, D., & Khovratovich, D. (2016). "Argon2: New Generation of Memory-Hard Functions for Password Hashing." Euro S&P '16.

[10] Kleppmann, M. & Howard, H. (2020). "Byzantine Eventual Consistency and the Fundamental Limits of Peer-to-Peer Databases." arXiv:2012.00472.

[11] O'Neil, P., et al. (1996). "The Log-Structured Merge-Tree (LSM-Tree)." *Acta Informatica*.

[12] Castro, M., & Liskov, B. (1999). "Practical Byzantine Fault Tolerance." *OSDI*.

[13] Merkle, R. C. (1987). "A Digital Signature Based on a Conventional Encryption Function." *CRYPTO '87*.

[14] Eppstein, D., et al. (2011). "What's the Difference? Efficient Set Reconciliation without Prior Context." *SIGCOMM '11*.

[15] Langley, A., et al. (2017). "The QUIC Transport Protocol: Design and Internet-Scale Deployment." *SIGCOMM '17*.

[16] Abadi, D. J., et al. (2006). "Integrating Compression and Execution in Column-Oriented Database Systems." *SIGMOD*.

[17] Gebru, T., et al. (2021). "Datasheets for Datasets." *Communications of the ACM*.

[18] Demers, A., et al. (1987). "Epidemic Algorithms for Replicated Database Maintenance." *PODC*.

[19] Stoica, I., et al. (2001). "Chord: A Scalable Peer-to-peer Lookup Service for Internet Applications." *SIGCOMM '01*.

[20] Lamport, L. (1978). "Time, Clocks, and the Ordering of Events in a Distributed System." *Communications of the ACM*.

[21] Muthitacharoen, A., Chen, B., & Mazieres, D. (2001). "A Low-bandwidth Network File System." *SOSP '01*. (Foundational paper for Content-Defined Chunking / LBFS)

[22] Fan, B., Andersen, D. G., Kaminsky, M., & Mitzenmacher, M. (2014). "Cuckoo Filter: Practically Better Than Bloom." *CoNEXT '14*. (Efficient set membership for peer discovery)

[23] Jacobson, V., et al. (2009). "Networking Named Content." *CoNEXT '09*. (Validates the efficiency of fetching data by chunk ID/Name)

[24] Rhea, S., et al. (2003). "Value-Based Web Caching." *WWW '03*. (Redundancy elimination via content addressing)

[25] Broder, A., & Mitzenmacher, M. (2004). "Network Applications of Bloom Filters: A Survey." *Internet Mathematics*.

[26] Ongaro, D., & Ousterhout, J. (2014). "In Search of an Understandable Consensus Algorithm." *USENIX ATC '14*. (Raft - Context for consistency choices)

[27] Maymounkov, P., & Mazieres, D. (2002). "Kademlia: A Peer-to-peer Information System Based on the XOR Metric." *IPTPS '02*. (Routing efficiency)

[28] Brewer, E. A. (2012). "CAP Twelve Years Later: How the 'Rules' Have Changed." *Computer*. (Trade-offs in distributed design)

[29] Helland, P. (2012). "Idempotence Is Not a Medical Condition." *Queue*. (Importance of idempotent operations in distributed CLI/API)

[30] Boncz, P. A., Zukowski, M., & Nes, N. (2005). "MonetDB/X100: Hyper-Pipelining Query Execution." *CIDR*. (Vectorized execution relevance to Arrow/chunk processing)

## Appendix A: Dependency Audit

All dependencies must be pure Rust (no C/C++ bindings):

| Crate | Purpose | C-free |
|-------|---------|--------|
| blake3 | Hashing | ✓ |
| ed25519-dalek | Signing | ✓ |
| redb | Database | ✓ |
| tantivy | Search | ✓ |
| quinn | QUIC transport | ✓ |
| mdns-sd | Discovery | ✓ |
| zstd (pure) | Compression | ✓ (pure feature) |
| argon2 | KDF | ✓ |
| fastcdc | Chunking | ✓ |
| cuckoofilter | Membership | ✓ |

## Appendix B: WASM Compatibility

Federation features require network access unavailable in WASM. The following subset works in browser:

- Local registry (IndexedDB backend)
- Manifest parsing/validation
- Signature verification
- Dataset cards
- Search (in-memory index)

Export/import uses browser File API instead of filesystem.
