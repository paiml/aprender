# HuggingFace Integration Specification v1.1.0

## Overview

Unified HuggingFace Hub integration for querying, pulling, and publishing models, datasets, and spaces within the PAIML stack.

```
[REVIEW-001] @alfredo 2024-12-05
Toyota Principle: Genchi Genbutsu (Go and See)
Direct Hub API integration eliminates third-party abstraction layers.
We query the source, not cached mirrors or outdated indices.
Status: APPROVED
```

## CLI Interface

### Query Commands

```bash
# Search models
batuta hf search models "llama 7b" --task text-generation --library transformers

# Search datasets
batuta hf search datasets "common voice" --language en --size ">1GB"

# Search spaces
batuta hf search spaces "gradio" --sdk gradio

# Get model info
batuta hf info model "meta-llama/Llama-2-7b-hf"

# Get dataset info
batuta hf info dataset "mozilla-foundation/common_voice_13_0"
```

### Pull Commands

```bash
# Pull model (auto-selects best format)
batuta hf pull model "TheBloke/Llama-2-7B-GGUF" --quantization Q4_K_M

# Pull to realizar format
batuta hf pull model "mistralai/Mistral-7B-v0.1" --format gguf --output ./models/

# Pull dataset
batuta hf pull dataset "squad" --split train --output ./data/

# Pull specific files
batuta hf pull files "meta-llama/Llama-2-7b-hf" "tokenizer.json" "config.json"
```

### Publish Commands

```bash
# Publish model trained with aprender
batuta hf push model ./my-model --repo "myorg/my-classifier" --format safetensors

# Publish dataset
batuta hf push dataset ./data/processed --repo "myorg/my-dataset"

# Publish space (presentar app)
batuta hf push space ./my-app --repo "myorg/my-demo" --sdk gradio
```

```
[REVIEW-002] @noah 2024-12-05
Toyota Principle: Jidoka (Automation with Human Touch)
CLI provides full automation while --interactive flag enables human review
before publish operations. Prevents accidental public releases.
Status: APPROVED
```

## Security & Resilience (v1.1.0)

### SafeTensors Enforcement (Poka-Yoke)

```bash
# Default: SafeOnly policy blocks pickle-based formats
batuta hf pull model "repo/model"  # Blocks .bin, .pkl, .pt files

# Explicit override for unsafe formats
batuta hf pull model "repo/model" --allow-unsafe
```

```rust
pub enum SafetyPolicy {
    SafeOnly,    // Default - blocks pickle
    AllowUnsafe, // Explicit consent required
}

pub fn classify_file_safety(filename: &str) -> FileSafety {
    // Safe: .safetensors, .gguf, .json, .yaml
    // Unsafe: .bin, .pkl, .pt, .pth, .pickle
}
```

### Rate Limit Handling (Andon)

```rust
pub struct RateLimitConfig {
    pub initial_backoff: Duration,  // 1s
    pub max_backoff: Duration,      // 60s
    pub max_retries: u32,           // 5
    pub multiplier: f64,            // 2.0
}

// Exponential backoff: 1s → 2s → 4s → 8s → 16s
// Respects Retry-After header from API
```

### Secret Scanning (Poka-Yoke)

```bash
# Automatic scan before push
batuta hf push model ./my-model --repo "org/model"

# Blocked if secrets detected:
# - .env files
# - Private keys (.pem, id_rsa)
# - Credential files
```

### Model Card Auto-Generation

```rust
pub fn generate_model_card(metadata: &ModelCardMetadata) -> String {
    // Auto-populates:
    // - YAML frontmatter (license, tags, library)
    // - Training metrics from certeza
    // - PAIML stack attribution
}
```

### Differential Uploads (Muda Elimination)

```rust
pub struct UploadManifest {
    pub files: HashMap<String, FileHash>,
}

impl UploadManifest {
    // Only upload changed files
    pub fn diff(&self, remote: &UploadManifest) -> Vec<String>;
}
```

```
[REVIEW-011] @security-team 2024-12-05
Toyota Principle: Poka-Yoke (Mistake Proofing)
SafeTensors-by-default prevents pickle deserialization attacks.
Secret scanning prevents accidental credential exposure.
Status: APPROVED
```

```
[REVIEW-012] @reliability-team 2024-12-05
Toyota Principle: Andon (Stop and Fix)
Rate limit handling with visual countdown enables graceful degradation.
Exponential backoff prevents thundering herd on API recovery.
Status: APPROVED
```

## Data Model

```rust
/// HuggingFace Hub asset types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HfAssetType {
    Model,
    Dataset,
    Space,
}

/// Model information from Hub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModelInfo {
    pub id: String,
    pub author: String,
    pub sha: String,
    pub pipeline_tag: Option<String>,
    pub tags: Vec<String>,
    pub downloads: u64,
    pub likes: u64,
    pub library_name: Option<String>,
    pub model_index: Option<Vec<ModelIndexEntry>>,
    pub safetensors: Option<SafetensorsInfo>,
    pub gguf: Option<GgufInfo>,
}

/// Dataset information from Hub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfDatasetInfo {
    pub id: String,
    pub author: String,
    pub description: Option<String>,
    pub citation: Option<String>,
    pub splits: Vec<DatasetSplit>,
    pub features: HashMap<String, FeatureType>,
    pub size_bytes: u64,
    pub download_size: u64,
}

/// Space information from Hub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfSpaceInfo {
    pub id: String,
    pub author: String,
    pub sdk: SpaceSdk,
    pub hardware: Option<String>,
    pub status: SpaceStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpaceSdk {
    Gradio,
    Streamlit,
    Docker,
    Static,
}
```

```
[REVIEW-003] @maria 2024-12-05
Toyota Principle: Standardized Work
Uniform data model across all asset types enables consistent tooling.
Same patterns for query, pull, push regardless of Model/Dataset/Space.
Status: APPROVED
```

## HuggingFace Stack Tree

Visual representation of HuggingFace ecosystem components.

```bash
batuta hf tree
```

```
HuggingFace Ecosystem (8 categories)
├── hub
│   ├── models         (700K+ models)
│   ├── datasets       (100K+ datasets)
│   ├── spaces         (300K+ spaces)
│   └── papers         (Research papers)
├── libraries
│   ├── transformers   (Model architectures)
│   ├── diffusers      (Diffusion models)
│   ├── accelerate     (Distributed training)
│   ├── peft           (Parameter-efficient fine-tuning)
│   ├── trl            (Reinforcement learning)
│   ├── optimum        (Hardware optimization)
│   ├── datasets       (Dataset loading)
│   ├── tokenizers     (Fast tokenization)
│   ├── safetensors    (Safe serialization)
│   └── huggingface_hub (Hub API client)
├── inference
│   ├── inference-api  (Serverless inference)
│   ├── inference-endpoints (Dedicated endpoints)
│   └── text-generation-inference (TGI server)
├── training
│   ├── autotrain      (AutoML training)
│   ├── trainer        (Training loops)
│   └── evaluate       (Metrics & evaluation)
├── deployment
│   ├── spaces         (Gradio/Streamlit hosting)
│   ├── endpoints      (Production serving)
│   └── enterprise-hub (Private deployments)
├── formats
│   ├── safetensors    (Safe tensor format)
│   ├── gguf           (Quantized format)
│   ├── onnx           (Cross-platform)
│   └── pytorch        (Native PyTorch)
├── tasks
│   ├── text-generation
│   ├── text-classification
│   ├── token-classification
│   ├── question-answering
│   ├── translation
│   ├── summarization
│   ├── image-classification
│   ├── object-detection
│   ├── image-segmentation
│   ├── text-to-image
│   ├── image-to-text
│   ├── speech-recognition
│   ├── text-to-speech
│   └── ...40+ more tasks
└── community
    ├── discussions
    ├── model-cards
    └── dataset-cards
```

```
[REVIEW-004] @carlos 2024-12-05
Toyota Principle: Visual Management
Tree view provides instant ecosystem comprehension.
Engineers see full HF landscape without documentation diving.
Status: APPROVED
```

## PAIML-HuggingFace Integration Tree

Shows how PAIML stack integrates with or provides alternatives to HuggingFace.

```bash
batuta hf tree --integration
```

```
PAIML ↔ HuggingFace Integration Map
═══════════════════════════════════════════════════════════════

┌─────────────────┬────────────────────┬────────────────────────┐
│ PAIML Component │ HF Equivalent      │ Integration Type       │
├─────────────────┼────────────────────┼────────────────────────┤
│ FORMATS         │                    │                        │
├─────────────────┼────────────────────┼────────────────────────┤
│ realizar/gguf   │ gguf (llama.cpp)   │ ✓ COMPATIBLE (parse)   │
│ realizar/safetensors │ safetensors   │ ✓ COMPATIBLE (r/w)     │
│ aprender/safetensors │ safetensors   │ ✓ COMPATIBLE (r/w)     │
├─────────────────┼────────────────────┼────────────────────────┤
│ HUB ACCESS      │                    │                        │
├─────────────────┼────────────────────┼────────────────────────┤
│ aprender/hf_hub │ huggingface_hub    │ ✓ USES (hf-hub crate)  │
│ batuta/hf       │ huggingface_hub    │ ✓ ORCHESTRATES         │
├─────────────────┼────────────────────┼────────────────────────┤
│ INFERENCE       │                    │                        │
├─────────────────┼────────────────────┼────────────────────────┤
│ realizar        │ transformers       │ ⚡ ALTERNATIVE (Rust)   │
│ realizar        │ TGI                │ ⚡ ALTERNATIVE (native) │
│ realizar/moe    │ optimum            │ ⚡ ALTERNATIVE (MoE)    │
├─────────────────┼────────────────────┼────────────────────────┤
│ TRAINING        │                    │                        │
├─────────────────┼────────────────────┼────────────────────────┤
│ aprender        │ transformers       │ ⚡ ALTERNATIVE (Rust)   │
│ entrenar        │ trainer/accelerate │ ⚡ ALTERNATIVE (native) │
│ alimentar       │ datasets           │ ⚡ ALTERNATIVE (Rust)   │
├─────────────────┼────────────────────┼────────────────────────┤
│ COMPUTE         │                    │                        │
├─────────────────┼────────────────────┼────────────────────────┤
│ trueno          │ (PyTorch backend)  │ ⚡ ALTERNATIVE (SIMD)   │
│ trueno-gpu      │ (CUDA backend)     │ ⚡ ALTERNATIVE (native) │
│ repartir        │ accelerate         │ ⚡ ALTERNATIVE (dist)   │
├─────────────────┼────────────────────┼────────────────────────┤
│ TOKENIZATION    │                    │                        │
├─────────────────┼────────────────────┼────────────────────────┤
│ realizar/tokenizer │ tokenizers      │ ✓ COMPATIBLE (load)    │
│ trueno-rag      │ tokenizers         │ ✓ COMPATIBLE (load)    │
├─────────────────┼────────────────────┼────────────────────────┤
│ APPS            │                    │                        │
├─────────────────┼────────────────────┼────────────────────────┤
│ presentar       │ gradio/streamlit   │ ⚡ ALTERNATIVE (Rust)   │
│ trueno-viz      │ (visualization)    │ ⚡ ALTERNATIVE (GPU)    │
├─────────────────┼────────────────────┼────────────────────────┤
│ QUALITY         │                    │                        │
├─────────────────┼────────────────────┼────────────────────────┤
│ certeza         │ evaluate           │ ⚡ ALTERNATIVE (Rust)   │
└─────────────────┴────────────────────┴────────────────────────┘

Legend:
  ✓ COMPATIBLE  - Interoperates with HF format/API
  ⚡ ALTERNATIVE - PAIML native replacement (pure Rust)
  ✗ INCOMPATIBLE - Not interoperable
```

```
[REVIEW-005] @elena 2024-12-05
Toyota Principle: Nemawashi (Consensus Building)
Integration map clarifies "compete vs cooperate" decisions.
Teams understand when to use HF vs PAIML for each capability.
Status: APPROVED
```

## API Client

```rust
/// HuggingFace Hub client for batuta
pub struct HfClient {
    /// API token (from HF_TOKEN env or ~/.huggingface/token)
    token: Option<String>,
    /// Base API URL
    api_url: String,
    /// HTTP client
    client: reqwest::Client,
    /// Cache directory
    cache_dir: PathBuf,
}

impl HfClient {
    /// Create new client (auto-loads token)
    pub fn new() -> Result<Self>;

    /// Search models
    pub async fn search_models(&self, query: &ModelQuery) -> Result<Vec<HfModelInfo>>;

    /// Search datasets
    pub async fn search_datasets(&self, query: &DatasetQuery) -> Result<Vec<HfDatasetInfo>>;

    /// Get model info
    pub async fn model_info(&self, repo_id: &str) -> Result<HfModelInfo>;

    /// Download model files
    pub async fn download_model(
        &self,
        repo_id: &str,
        files: Option<Vec<&str>>,
        revision: Option<&str>,
    ) -> Result<PathBuf>;

    /// Upload model to Hub
    pub async fn upload_model(
        &self,
        local_path: &Path,
        repo_id: &str,
        commit_message: &str,
    ) -> Result<String>;

    /// Create model card
    pub fn generate_model_card(&self, info: &ModelCardInfo) -> String;
}
```

```
[REVIEW-006] @david 2024-12-05
Toyota Principle: Pull System
Downloads happen on-demand, not pre-cached.
Only fetch what's needed, when needed.
Status: APPROVED
```

## Format Conversion Pipeline

```rust
/// Convert between model formats
pub enum ModelFormat {
    PyTorch,
    SafeTensors,
    GGUF { quantization: GGUFQuantization },
    ONNX,
}

/// Conversion pipeline
pub struct FormatConverter {
    /// Source format
    source: ModelFormat,
    /// Target format
    target: ModelFormat,
}

impl FormatConverter {
    /// PyTorch/SafeTensors → GGUF (for realizar serving)
    pub fn to_gguf(&self, input: &Path, quantization: GGUFQuantization) -> Result<PathBuf>;

    /// GGUF → SafeTensors (for aprender training)
    pub fn to_safetensors(&self, input: &Path) -> Result<PathBuf>;

    /// Auto-convert based on target runtime
    pub fn auto_convert(&self, input: &Path, target: Runtime) -> Result<PathBuf>;
}

#[derive(Clone, Copy)]
pub enum GGUFQuantization {
    F32,
    F16,
    Q8_0,
    Q4_0,
    Q4_K_M,
    Q5_K_M,
    Q6_K,
}
```

```
[REVIEW-007] @sofia 2024-12-05
Toyota Principle: Heijunka (Level Loading)
Format conversion balances precision vs performance.
Q4_K_M default provides good quality at 4-bit efficiency.
Status: APPROVED
```

## Caching Strategy

```rust
/// Cache configuration
pub struct HfCache {
    /// Cache root (~/.cache/batuta/hf or HF_HOME)
    root: PathBuf,
    /// Max cache size in bytes
    max_size: u64,
    /// TTL for metadata cache
    metadata_ttl: Duration,
}

impl HfCache {
    /// Get cached model path (or None)
    pub fn get_model(&self, repo_id: &str, revision: &str) -> Option<PathBuf>;

    /// Store model in cache
    pub fn put_model(&self, repo_id: &str, revision: &str, path: &Path) -> Result<PathBuf>;

    /// Prune cache to stay under max_size
    pub fn prune(&self) -> Result<u64>;

    /// Clear entire cache
    pub fn clear(&self) -> Result<()>;
}
```

```
[REVIEW-008] @miguel 2024-12-05
Toyota Principle: Muda Elimination (Waste)
Cache prevents redundant downloads. Pruning prevents disk waste.
LRU eviction keeps frequently-used models hot.
Status: APPROVED
```

## Authentication

```rust
/// Authentication methods
pub enum HfAuth {
    /// Token from environment (HF_TOKEN)
    EnvToken,
    /// Token from file (~/.huggingface/token)
    FileToken,
    /// Explicit token
    Token(String),
    /// No authentication (public repos only)
    None,
}

impl HfClient {
    /// Login and store token
    pub fn login(&self, token: &str) -> Result<()>;

    /// Check if authenticated
    pub fn is_authenticated(&self) -> bool;

    /// Get current user info
    pub async fn whoami(&self) -> Result<UserInfo>;
}
```

## Implementation Plan

| Phase | Component | Description |
|-------|-----------|-------------|
| 1 | `src/hf/mod.rs` | Module structure |
| 2 | `src/hf/client.rs` | HTTP client with auth |
| 3 | `src/hf/search.rs` | Search API |
| 4 | `src/hf/download.rs` | Download with progress |
| 5 | `src/hf/upload.rs` | Upload with validation |
| 6 | `src/hf/cache.rs` | Caching layer |
| 7 | `src/hf/tree.rs` | Tree visualization |
| 8 | CLI commands | `batuta hf *` |
| 9 | Tests | 95%+ coverage |

```
[REVIEW-009] @ana 2024-12-05
Toyota Principle: Hansei (Reflection)
Phased implementation allows course correction.
Each phase delivers testable, shippable value.
Status: APPROVED
```

## Success Criteria

- [ ] `batuta hf search models "llama"` returns results in <2s
- [ ] `batuta hf pull model` shows progress bar, resumes on failure
- [ ] `batuta hf push model` validates format before upload
- [ ] `batuta hf tree` displays full ecosystem in <100ms
- [ ] `batuta hf tree --integration` shows all PAIML mappings
- [ ] Cache hit rate >80% for repeated operations
- [ ] Works offline with cached models
- [ ] 95% test coverage on hf module

```
[REVIEW-010] @jorge 2024-12-05
Toyota Principle: Challenge (Long-term Vision)
Full HF integration positions PAIML as interoperable, not isolated.
Users migrate incrementally: HF models → PAIML inference → full stack.
Status: APPROVED
```

## Dependencies

```toml
[dependencies]
# HuggingFace Hub API (already in aprender)
hf-hub = { version = "0.4", features = ["ureq"] }

# Async HTTP
reqwest = { version = "0.11", features = ["json", "stream"] }

# Progress indication
indicatif = "0.17"

# File hashing for cache validation
sha2 = "0.10"
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | API token | None |
| `HF_HOME` | Cache directory | `~/.cache/huggingface` |
| `HF_ENDPOINT` | API endpoint | `https://huggingface.co` |
| `HF_HUB_OFFLINE` | Offline mode | `false` |

## References

- HuggingFace Hub API: https://huggingface.co/docs/hub/api
- hf-hub crate: https://docs.rs/hf-hub
- SafeTensors spec: https://github.com/huggingface/safetensors
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Toyota Production System (Ohno, 1988)
- The Toyota Way (Liker, 2004)
