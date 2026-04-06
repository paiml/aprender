# `batuta hf`

HuggingFace Hub integration commands.

## Synopsis

```bash
batuta hf <COMMAND>
```

## Commands

| Command | Description |
|---------|-------------|
| `catalog` | Query 50+ HuggingFace ecosystem components |
| `course` | Query by Coursera course alignment |
| `tree` | Display HuggingFace ecosystem tree |
| `search` | Search models, datasets, spaces |
| `info` | Get info about a Hub asset |
| `pull` | Download from HuggingFace Hub |
| `push` | Upload to HuggingFace Hub |

---

## `batuta hf catalog`

Query the HuggingFace ecosystem catalog with 51 components across 6 categories.

### Usage

```bash
batuta hf catalog [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--component <ID>` | Get details for a specific component |
| `--category <CAT>` | Filter by category (hub, deployment, library, training, collaboration, community) |
| `--tag <TAG>` | Filter by tag (e.g., rlhf, lora, quantization) |
| `--list` | List all available components |
| `--categories` | List all categories with component counts |
| `--tags` | List all available tags |
| `--format <FORMAT>` | Output format: `table` (default), `json` |

### Examples

```bash
# List all training components
batuta hf catalog --category training

# Output:
# 📦 HuggingFace Components
# ════════════════════════════════════════════════════════════
#   peft        PEFT           Training & Optimization
#   trl         TRL            Training & Optimization
#   bitsandbytes Bitsandbytes  Training & Optimization
#   ...

# Get component details
batuta hf catalog --component peft

# Output:
# 📦 PEFT
# ════════════════════════════════════════════════════════════
# ID:          peft
# Category:    Training & Optimization
# Description: Parameter-efficient finetuning for large language models
# Docs:        https://huggingface.co/docs/peft
# Repository:  https://github.com/huggingface/peft
# PyPI:        peft
# Tags:        finetuning, lora, qlora, efficient
# Dependencies: transformers, bitsandbytes
# Course Alignments:
#   Course 4, Week 1: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8

# Search by tag
batuta hf catalog --tag rlhf
batuta hf catalog --tag quantization
```

### Component Categories

| Category | Components | Description |
|----------|------------|-------------|
| Hub | 7 | Hub & client libraries (models, datasets, spaces) |
| Deployment | 7 | Inference & deployment (TGI, TEI, endpoints) |
| Library | 10 | Core ML libraries (transformers, diffusers, datasets) |
| Training | 10 | Training & optimization (PEFT, TRL, bitsandbytes) |
| Collaboration | 11 | Tools & integrations (Gradio, Argilla, agents) |
| Community | 6 | Community resources (blog, forum, leaderboards) |

---

## `batuta hf course`

Query HuggingFace components aligned to Coursera specialization courses.

### Usage

```bash
batuta hf course [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--list` | List all 5 courses with component counts |
| `--course <N>` | Show components for course N (1-5) |
| `--week <N>` | Filter by week (requires --course) |

### Examples

```bash
# List all courses
batuta hf course --list

# Output:
# 📚 Pragmatic AI Labs HuggingFace Specialization
# ════════════════════════════════════════════════════════════
# 5 Courses | 15 Weeks | 60 Hours
#
#   Course 1: Foundations of HuggingFace (9 components)
#   Course 2: Fine-Tuning and Datasets (5 components)
#   Course 3: RAG and Retrieval (3 components)
#   Course 4: Advanced Training (RLHF, DPO, PPO) (3 components)
#   Course 5: Production Deployment (8 components)

# Get Course 4 (Advanced Fine-Tuning)
batuta hf course --course 4

# Output:
# 📚 Course 4 - Advanced Training (RLHF, DPO, PPO)
# ════════════════════════════════════════════════════════════
#   peft           Week 1
#   bitsandbytes   Week 1
#   trl            Week 2, Week 3
```

### Course Curriculum

| Course | Topic | Key Components |
|--------|-------|----------------|
| 1 | Foundations | transformers, tokenizers, safetensors, hub |
| 2 | Datasets & Fine-Tuning | datasets, trainer, evaluate |
| 3 | RAG & Retrieval | sentence-transformers, faiss, outlines |
| 4 | RLHF/DPO/PPO | peft, trl, bitsandbytes |
| 5 | Production | tgi, gradio, optimum, inference-endpoints |

---

## `batuta hf tree`

Display hierarchical view of HuggingFace ecosystem or PAIML integration map.

### Usage

```bash
batuta hf tree [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--integration` | Show PAIML↔HuggingFace integration map |
| `--format <FORMAT>` | Output format: `ascii` (default), `json` |

### Examples

```bash
# HuggingFace ecosystem tree
batuta hf tree

# Output:
# HuggingFace Ecosystem (6 categories)
# ├── hub
# │   ├── models         (700K+ models)
# │   ├── datasets       (100K+ datasets)
# │   └── spaces         (300K+ spaces)
# ├── libraries
# │   ├── transformers   (Model architectures)
# │   └── ...

# PAIML-HuggingFace integration map
batuta hf tree --integration

# Output shows:
# ✓ COMPATIBLE  - Interoperates with HF format/API
# ⚡ ALTERNATIVE - PAIML native replacement (pure Rust)
# 🔄 ORCHESTRATES - PAIML wraps/orchestrates HF
# 📦 USES        - PAIML uses HF library directly
```

---

## `batuta hf search`

Search HuggingFace Hub for models, datasets, or spaces.

### Usage

```bash
batuta hf search <ASSET_TYPE> <QUERY> [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<ASSET_TYPE>` | Type: `model`, `dataset`, `space` |
| `<QUERY>` | Search query string |

### Options

| Option | Description |
|--------|-------------|
| `--task <TASK>` | Filter by task (for models) |
| `--limit <N>` | Limit results (default: 10) |

### Examples

```bash
# Search for Llama models
batuta hf search model "llama 7b" --task text-generation

# Search for speech datasets
batuta hf search dataset "common voice" --limit 5

# Search for Gradio spaces
batuta hf search space "image classifier"
```

---

## `batuta hf info`

Get detailed information about a HuggingFace asset.

### Usage

```bash
batuta hf info <ASSET_TYPE> <REPO_ID>
```

### Examples

```bash
# Get model info
batuta hf info model "meta-llama/Llama-2-7b-hf"

# Get dataset info
batuta hf info dataset "mozilla-foundation/common_voice_13_0"

# Get space info
batuta hf info space "gradio/chatbot"
```

---

## `batuta hf pull`

Download models, datasets, or spaces from HuggingFace Hub.

### Usage

```bash
batuta hf pull <ASSET_TYPE> <REPO_ID> [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `-o, --output <PATH>` | Output directory |
| `--quantization <Q>` | Model quantization (Q4_K_M, Q5_K_M, etc.) |

### Examples

```bash
# Pull GGUF model with quantization
batuta hf pull model "TheBloke/Llama-2-7B-GGUF" --quantization Q4_K_M

# Pull to specific directory
batuta hf pull model "mistralai/Mistral-7B-v0.1" -o ./models/

# Pull dataset
batuta hf pull dataset "squad" -o ./data/
```

---

## `batuta hf push`

Upload models, datasets, or spaces to HuggingFace Hub.

### Usage

```bash
batuta hf push <ASSET_TYPE> <PATH> --repo <REPO_ID> [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--repo <REPO_ID>` | Target repository (required) |
| `--message <MSG>` | Commit message |

### Examples

```bash
# Push trained model
batuta hf push model ./my-model --repo "myorg/my-classifier"

# Push dataset
batuta hf push dataset ./data/processed --repo "myorg/my-dataset"

# Push Presentar app as Space
batuta hf push space ./my-app --repo "myorg/demo" --message "Initial release"
```

---

## PAIML-HuggingFace Integration

The integration map shows how PAIML stack components relate to HuggingFace (28 mappings):

| Category | PAIML | HuggingFace | Type |
|----------|-------|-------------|------|
| **Formats** | `.apr` | pickle/.joblib, safetensors, gguf | ⚡ Alternative |
| | realizar/gguf | gguf | ✓ Compatible |
| | realizar/safetensors | safetensors | ✓ Compatible |
| **Data Formats** | `.ald` | parquet/arrow, json/csv | ⚡ Alternative |
| **Hub Access** | aprender/hf_hub | huggingface_hub | 📦 Uses |
| | batuta/hf | huggingface_hub | 🔄 Orchestrates |
| **Registry** | pacha | HF Hub registry, MLflow/W&B | ⚡ Alternative |
| **Inference** | realizar | transformers, TGI | ⚡ Alternative |
| | realizar/moe | optimum | ⚡ Alternative |
| **Classical ML** | aprender | sklearn, xgboost/lightgbm | ⚡ Alternative |
| **Deep Learning** | entrenar | PyTorch training | ⚡ Alternative |
| | alimentar | datasets | ⚡ Alternative |
| **Compute** | trueno | NumPy/PyTorch tensors | ⚡ Alternative |
| | repartir | accelerate | ⚡ Alternative |
| **Tokenization** | realizar/tokenizer | tokenizers | ✓ Compatible |
| | trueno-rag | tokenizers | ✓ Compatible |
| **Apps** | presentar | gradio | ⚡ Alternative |
| | trueno-viz | visualization | ⚡ Alternative |
| **Quality** | certeza | evaluate | ⚡ Alternative |
| **MCP Tooling** | pforge | LangChain Tools | ⚡ Alternative |
| | pmat | code analysis tools | ⚡ Alternative |
| | pmcp | mcp-sdk | ⚡ Alternative |

**Legend:**
- ✓ COMPATIBLE - Interoperates with HF format/API
- ⚡ ALTERNATIVE - PAIML native replacement (pure Rust)
- 🔄 ORCHESTRATES - PAIML wraps/orchestrates HF
- 📦 USES - PAIML uses HF library directly

### Compatible Formats

PAIML can load and save HuggingFace formats:

```rust
// Load GGUF model (realizar)
let model = GGUFModel::from_file("model.gguf")?;

// Load SafeTensors (aprender)
let weights = SafeTensors::load("model.safetensors")?;

// Load HF tokenizer (realizar)
let tokenizer = Tokenizer::from_pretrained("meta-llama/Llama-2-7b-hf")?;
```

### Security Features (v1.1.0)

### SafeTensors Enforcement

By default, `batuta hf pull` blocks unsafe pickle-based formats:

```bash
# Default: blocks .bin, .pkl, .pt files
batuta hf pull model "repo/model"

# Explicit override for unsafe formats
batuta hf pull model "repo/model" --allow-unsafe
```

| Extension | Safety | Notes |
|-----------|--------|-------|
| `.safetensors` | ✓ Safe | Recommended |
| `.gguf` | ✓ Safe | Quantized |
| `.json` | ✓ Safe | Config |
| `.bin` | ✗ Unsafe | Pickle-based |
| `.pkl` | ✗ Unsafe | Pickle |
| `.pt` | ✗ Unsafe | PyTorch |

### Secret Scanning

Automatic scan before push blocks accidental credential exposure:

```bash
# Blocked if secrets detected
batuta hf push model ./my-model --repo "org/model"

# Detected patterns:
# - .env files
# - Private keys (.pem, id_rsa)
# - Credential files
```

### Rate Limit Handling

Automatic exponential backoff for API rate limits (429):

- Initial: 1s → 2s → 4s → 8s → 16s
- Max backoff: 60s
- Max retries: 5
- Respects `Retry-After` header

### Model Card Auto-Generation

```bash
# Auto-generates README.md if missing
batuta hf push model ./my-model --repo "org/model"
```

Generated card includes:
- YAML frontmatter (license, tags)
- Training metrics from certeza
- PAIML stack attribution

### Differential Uploads

Only uploads changed files using content-addressable hashing:

```bash
# Only uploads modified files
batuta hf push model ./my-model --repo "org/model"
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token |
| `HF_HOME` | Cache directory |
| `HF_HUB_OFFLINE` | Offline mode |
