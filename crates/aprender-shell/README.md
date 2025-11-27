# aprender-shell

AI-powered shell completion trained on your command history.

## Installation

```bash
cargo install aprender-shell
```

## Usage

```bash
# Train on your zsh history
aprender-shell train

# Get completions for a prefix
aprender-shell suggest "git ch"

# Interactive completion
aprender-shell complete "docker "
```

## Features

- Learns from your shell history patterns
- Sub-10ms suggestion latency
- Context-aware completions
- Supports zsh history format

## License

MIT
