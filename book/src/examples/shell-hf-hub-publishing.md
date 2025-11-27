# Case Study: Publishing Shell Models to Hugging Face Hub

Share your trained shell completion models with the community via Hugging Face Hub.

## Overview

The `publish` command uploads your model to Hugging Face Hub, automatically generating:
- Model card (README.md) with metadata
- Training statistics
- Usage instructions
- License information

## Quick Start

```bash
# 1. Train a model
aprender-shell train -f ~/.zsh_history -o my-shell.model

# 2. Set your HF token
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# 3. Publish
aprender-shell publish -m my-shell.model -r username/my-shell-completions
```

## Getting a Hugging Face Token

1. Create account at [huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens
3. Create token with "Write" permission
4. Export: `export HF_TOKEN=hf_xxx`

## Publish Command

```bash
aprender-shell publish [OPTIONS] -m <MODEL> -r <REPO>

Options:
  -m, --model <MODEL>    Model file to publish
  -r, --repo <REPO>      Repository ID (username/repo-name)
  -c, --commit <MSG>     Commit message (default: "Upload model")
      --create           Create repository if it doesn't exist
      --private          Make repository private
```

### Examples

```bash
# Basic publish
aprender-shell publish -m model.apr -r paiml/devops-completions

# Create new repo with custom message
aprender-shell publish -m model.apr -r alice/k8s-model --create -c "Initial release"

# Private repository
aprender-shell publish -m model.apr -r company/internal-model --create --private
```

## Generated Model Card

The publish command generates a README.md with:

```markdown
---
license: mit
pipeline_tag: text-generation
tags:
  - aprender
  - shell-completion
  - markov-model
  - rust
---

# Shell Completion Model

AI-powered shell command completion trained on real history.

## Model Details

| Property | Value |
|----------|-------|
| Architecture | MarkovModel |
| N-gram Size | 3 |
| Vocabulary | 16,100 |
| Training Commands | 21,729 |

## Usage

\`\`\`bash
# Download
huggingface-cli download username/model model.apr

# Use with aprender-shell
aprender-shell suggest "git " -m model.apr
\`\`\`
```

## Without Token (Offline Mode)

If `HF_TOKEN` is not set, publish generates files locally:

```bash
$ aprender-shell publish -m model.apr -r paiml/test

⚠️  HF_TOKEN not set. Cannot upload to Hugging Face Hub.

📝 Model card saved to: README.md

To upload manually:
  1. Set HF_TOKEN: export HF_TOKEN=hf_xxx
  2. Run: huggingface-cli upload paiml/test model.apr README.md
```

## Model Inspection

Before publishing, inspect your model:

```bash
# Text format
aprender-shell inspect -m model.apr

# JSON format (programmatic)
aprender-shell inspect -m model.apr --format json

# Hugging Face YAML (model card preview)
aprender-shell inspect -m model.apr --format huggingface
```

### JSON Output

```json
{
  "model_id": "aprender-shell-markov-3gram-20251127",
  "name": "Shell Completion Model",
  "version": "1.0.0",
  "created_at": "2025-11-27T12:00:00Z",
  "framework_version": "aprender 0.10.0",
  "architecture": "MarkovModel",
  "hyperparameters": {
    "ngram_size": 3
  },
  "metrics": {
    "vocab_size": 16100,
    "ngram_count": 40848
  }
}
```

## Use Cases

### Team-Specific Models

Share DevOps patterns with your team:

```bash
# Train on team history
cat ~/.zsh_history ~/.bash_history team/*.history > combined.history
aprender-shell train -f combined.history -o devops.model

# Publish to org
aprender-shell publish -m devops.model -r myorg/devops-completions --create
```

### Domain-Specific Models

Curate models for specific domains:

| Domain | Example Commands |
|--------|------------------|
| Kubernetes | `kubectl`, `helm`, `k9s` |
| AWS | `aws`, `sam`, `cdk` |
| Docker | `docker`, `docker-compose` |
| Git | `git`, `gh`, `glab` |

### Community Models

Browse community models:

```bash
# Search HF Hub
huggingface-cli search aprender shell-completion

# Download
huggingface-cli download paiml/devops-completions model.apr
aprender-shell suggest "kubectl " -m model.apr
```

## Best Practices

### Privacy

Before publishing, verify no secrets in model:

```bash
# Check for sensitive patterns
strings model.apr | grep -iE 'password|secret|token|key'

# The model stores n-grams, not raw commands
# But verify training data was filtered
```

### Versioning

Use semantic versioning in commit messages:

```bash
aprender-shell publish -m model.apr -r user/model -c "v1.0.0: Initial release"
aprender-shell publish -m model.apr -r user/model -c "v1.1.0: Add kubectl patterns"
```

### Documentation

Add context in your model card:

```bash
# Edit generated README.md before upload
vim README.md

# Then upload with huggingface-cli
huggingface-cli upload user/model model.apr README.md
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  aprender-shell │────▶│   hf-hub crate   │────▶│  Hugging Face   │
│    publish      │     │  (official API)  │     │      Hub        │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │
        ▼
┌─────────────────┐
│   ModelCard     │
│  (README.md)    │
└─────────────────┘
```

The implementation uses the official `hf-hub` crate by Hugging Face for API compatibility.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "401 Unauthorized" | Check HF_TOKEN is valid and has write permission |
| "404 Not Found" | Use `--create` flag for new repositories |
| "Repository exists" | Repository already exists, will update files |
| "Model too large" | Use Git LFS for models >10MB |

## Related

- [Shell Completion](./shell-completion.md) - Training and usage
- [Model Cards](../ml-fundamentals/model-cards.md) - Metadata specification
- [Model Format (.apr)](./model-format.md) - Binary format details
