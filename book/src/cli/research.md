# Research Commands

The `entrenar research` command provides tools for academic research workflows, including artifact initialization,
pre-registration, citation generation, and repository deposits.

## Commands Overview

```bash
entrenar research <COMMAND>

Commands:
  init          Initialize a research artifact
  preregister   Create cryptographic pre-registration
  cite          Generate citation in various formats
  export        Export artifact metadata
  deposit       Deposit to repository (Zenodo/Figshare)
  bundle        Create RO-Crate package
  verify        Verify artifact integrity
```

## init

Initialize a new research artifact with metadata.

```bash
entrenar research init [OPTIONS]
```

### Options

| Option | Required | Description |
|--------|----------|-------------|
| `--id <ID>` | Yes | Unique artifact identifier |
| `--title <TITLE>` | Yes | Artifact title |
| `--type <TYPE>` | No | Type: dataset, model, code, paper (default: dataset) |
| `--author <AUTHOR>` | Yes | Author name (can specify multiple times) |
| `--orcid <ORCID>` | No | Author ORCID (can specify multiple times) |
| `--affiliation <AFF>` | No | Author affiliation (can specify multiple times) |
| `--license <LICENSE>` | No | License: mit, apache2, cc0, cc-by, cc-by-sa, gpl3 (default: cc-by) |
| `--description <DESC>` | No | Artifact description |
| `--output <PATH>` | No | Output file path (default: artifact.yaml) |

### Example

```bash
# Initialize a dataset artifact
entrenar research init \
  --id my-training-dataset \
  --title "ImageNet Subset for LoRA Training" \
  --type dataset \
  --author "Alice Smith" \
  --orcid "0000-0001-2345-6789" \
  --affiliation "Stanford University" \
  --license cc-by \
  --description "A curated subset of ImageNet for efficient LoRA fine-tuning experiments"

# Initialize a model artifact
entrenar research init \
  --id llama-lora-adapter \
  --title "LLaMA LoRA Adapter for Code Generation" \
  --type model \
  --author "Bob Jones" \
  --author "Carol White" \
  --output model-artifact.yaml
```

## preregister

Create a cryptographically signed pre-registration for reproducibility.

```bash
entrenar research preregister [OPTIONS] <ARTIFACT>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<ARTIFACT>` | Path to artifact YAML file |

### Options

| Option | Description |
|--------|-------------|
| `--hypothesis <TEXT>` | Pre-registered hypothesis |
| `--methods <TEXT>` | Pre-registered methods |
| `--output <PATH>` | Output file path |

### Example

```bash
# Create pre-registration with hypothesis
entrenar research preregister artifact.yaml \
  --hypothesis "LoRA rank 64 will achieve equivalent accuracy to full fine-tuning" \
  --methods "Train on 10K samples with AdamW, lr=1e-4, 3 epochs" \
  --output preregistration.yaml
```

The pre-registration includes:
- Git commit hash for reproducibility
- Ed25519 cryptographic signature
- Timestamp proof
- Hypothesis and methods locked at registration time

## cite

Generate citations in various academic formats.

```bash
entrenar research cite [OPTIONS] <ARTIFACT>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<ARTIFACT>` | Path to artifact YAML file |

### Options

| Option | Description |
|--------|-------------|
| `--format <FORMAT>` | Citation format: bibtex, apa, mla, chicago (default: bibtex) |
| `--year <YEAR>` | Publication year (default: current year) |

### Example

```bash
# Generate BibTeX citation
entrenar research cite artifact.yaml --format bibtex --year 2024

# Output:
# @misc{my-training-dataset,
#   author = {Alice Smith},
#   title = {ImageNet Subset for LoRA Training},
#   year = {2024},
#   howpublished = {\url{https://example.com/artifact}}
# }

# Generate APA citation
entrenar research cite artifact.yaml --format apa --year 2024

# Generate MLA citation
entrenar research cite artifact.yaml --format mla
```

## export

Export artifact metadata to different formats.

```bash
entrenar research export [OPTIONS] <ARTIFACT>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<ARTIFACT>` | Path to artifact YAML file |

### Options

| Option | Description |
|--------|-------------|
| `--format <FORMAT>` | Export format: json, yaml, datacite, schema-org (default: json) |
| `--output <PATH>` | Output file path |

### Example

```bash
# Export as JSON
entrenar research export artifact.yaml --format json --output metadata.json

# Export as DataCite XML
entrenar research export artifact.yaml --format datacite --output datacite.xml

# Export as Schema.org JSON-LD
entrenar research export artifact.yaml --format schema-org --output schema.jsonld
```

## deposit

Deposit artifact to a repository (Zenodo or Figshare).

```bash
entrenar research deposit [OPTIONS] <ARTIFACT>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<ARTIFACT>` | Path to artifact YAML file |

### Options

| Option | Description |
|--------|-------------|
| `--provider <PROVIDER>` | Repository: zenodo, figshare (default: zenodo) |
| `--sandbox` | Use sandbox/test environment |
| `--publish` | Publish immediately (otherwise draft) |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ZENODO_TOKEN` | API token for Zenodo |
| `FIGSHARE_TOKEN` | API token for Figshare |

### Example

```bash
# Deposit to Zenodo sandbox (for testing)
export ZENODO_TOKEN="your-api-token"
entrenar research deposit artifact.yaml --provider zenodo --sandbox

# Deposit and publish to Figshare
export FIGSHARE_TOKEN="your-api-token"
entrenar research deposit artifact.yaml --provider figshare --publish
```

## bundle

Create an RO-Crate package for FAIR data sharing.

```bash
entrenar research bundle [OPTIONS] <ARTIFACT>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<ARTIFACT>` | Path to artifact YAML file |

### Options

| Option | Description |
|--------|-------------|
| `--output <PATH>` | Output directory (default: ./ro-crate) |
| `--zip` | Create ZIP archive |

### Example

```bash
# Create RO-Crate directory
entrenar research bundle artifact.yaml --output ./my-crate

# Create ZIP archive
entrenar research bundle artifact.yaml --output ./package --zip
```

The RO-Crate bundle includes:
- `ro-crate-metadata.json` - JSON-LD metadata
- All referenced data files
- README with citation information
- License file

## verify

Verify artifact integrity and signatures.

```bash
entrenar research verify [OPTIONS] <ARTIFACT>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<ARTIFACT>` | Path to artifact YAML file |

### Options

| Option | Description |
|--------|-------------|
| `--deep` | Perform deep verification (check all referenced files) |

### Example

```bash
# Quick verification
entrenar research verify artifact.yaml

# Deep verification with file checksums
entrenar research verify artifact.yaml --deep
```

Verification checks:
- YAML schema validity
- Required metadata fields
- Pre-registration signatures (if present)
- File checksums (with `--deep`)
- Git commit existence (if timestamp proof present)

## Workflow Example

Complete research artifact workflow:

```bash
# 1. Initialize artifact
entrenar research init \
  --id experiment-2024 \
  --title "Temperature Scaling Ablation Study" \
  --type dataset \
  --author "Research Team" \
  --license cc-by

# 2. Pre-register hypothesis before running experiment
entrenar research preregister artifact.yaml \
  --hypothesis "T=4.0 is optimal for knowledge distillation" \
  --methods "Grid search T in [1.0, 8.0], step 0.5"

# 3. Run experiment (using entrenar-bench)
entrenar-bench temperature --start 1.0 --end 8.0 --step 0.5

# 4. Generate citation for paper
entrenar research cite artifact.yaml --format bibtex --year 2024

# 5. Create RO-Crate package
entrenar research bundle artifact.yaml --zip

# 6. Deposit to Zenodo
entrenar research deposit artifact.yaml --provider zenodo --publish

# 7. Verify final artifact
entrenar research verify artifact.yaml --deep
```

## See Also

- [CLI Overview](./overview.md) - General CLI reference
- [Benchmark Commands](./benchmark.md) - Benchmarking CLI reference
- [Academic Research Overview](../research/overview.md) - Research module documentation
