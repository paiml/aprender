# aprender-shell

AI-powered shell completion trained on your command history.

## Installation

```bash
cargo install aprender-shell
```

## Pre-trained Base Model

A base model trained on synthetic developer commands is available on Hugging Face Hub:

```bash
# Download base model
huggingface-cli download paiml/aprender-shell-base model.apr --local-dir ~/.aprender

# Use with aprender-shell
aprender-shell suggest "git " -m ~/.aprender/model.apr
```

**Model**: [paiml/aprender-shell-base](https://huggingface.co/paiml/aprender-shell-base)

The base model includes 401 common developer commands (git, cargo, docker, kubectl, npm, python, aws, terraform) and contains no personal data.

## Usage

```bash
# Train on your zsh history (creates personalized model)
aprender-shell train

# Get completions for a prefix
aprender-shell suggest "git ch"

# Install zsh widget for tab completion
aprender-shell zsh-widget >> ~/.zshrc
source ~/.zshrc
```

## Features

- Sub-10ms suggestion latency
- Context-aware N-gram completions
- Privacy-safe: sensitive commands filtered automatically
- Supports zsh and fish shell history formats
- Incremental training on new commands

## Fine-tuning

Start with the base model and fine-tune on your history:

```bash
# Download base model
huggingface-cli download paiml/aprender-shell-base model.apr -o base.apr

# Train incrementally on your history
aprender-shell train --base base.apr
```

## License

MIT
