# Case Study: AI Shell Completion

Train a personalized autocomplete on your shell history in 5 seconds. 100% local, private, fast.

## Quick Start

```bash
# Install
cargo install --path crates/aprender-shell

# Train on your history
aprender-shell train

# Test
aprender-shell suggest "git "
```

## How It Works

```
~/.zsh_history → Parser → N-gram Model → Trie Index → Suggestions
     │                         │              │
  21,729 cmds            40,848 n-grams    <1ms lookup
```

**Algorithm:** Markov chain with trigram context + prefix trie for O(1) lookup.

## Training

```bash
$ aprender-shell train

🚀 aprender-shell: Training model...

📂 History file: /home/user/.zsh_history
📊 Commands loaded: 21729
🧠 Training 3-gram model... done!

✅ Model saved to: ~/.aprender-shell.model

📈 Model Statistics:
   Unique n-grams: 40848
   Vocabulary size: 16100
   Model size: 2016.4 KB
```

## Suggestions

```bash
$ aprender-shell suggest "git "
git commit    0.505
git clone     0.065
git add       0.059
git push      0.035
git checkout  0.031

$ aprender-shell suggest "cargo "
cargo run      0.413
cargo install  0.069
cargo test     0.059
cargo clippy   0.045
```

Scores are frequency-based probabilities from your actual usage.

## Incremental Updates

Don't retrain from scratch—append new commands:

```bash
$ aprender-shell update
📊 Found 15 new commands
✅ Model updated (21744 total commands)

$ aprender-shell update
✓ Model is up to date (no new commands)
```

**Performance:**
- 0ms when no new commands
- ~10ms per 100 new commands
- Tracks position in history file

## ZSH Integration

Generate the widget:

```bash
aprender-shell zsh-widget >> ~/.zshrc
source ~/.zshrc
```

This adds:
- Ghost text suggestions as you type (gray)
- Tab or Right Arrow to accept
- Updates on every keystroke

## Auto-Retrain

```zsh
# Add to ~/.zshrc

# Option 1: Update after every command (~10ms)
precmd() { aprender-shell update -q & }

# Option 2: Update on shell exit
zshexit() { aprender-shell update -q }
```

## Model Statistics

```bash
$ aprender-shell stats

📊 Model Statistics:
   N-gram size: 3
   Unique n-grams: 40848
   Vocabulary size: 16100
   Model size: 2016.4 KB

🔝 Top commands:
    340x  git status
    245x  cargo build
    198x  cd ..
```

## Sharing Models

Export your model for teammates:

```bash
# Export
aprender-shell export -m ~/.aprender-shell.model team-model.json

# Import (on another machine)
aprender-shell import team-model.json
```

Use case: Share team-specific command patterns (deployment scripts, project aliases).

## Privacy & Security

**Filtered automatically:**
- Commands containing `password`, `secret`, `token`, `API_KEY`
- AWS credentials, GitHub tokens
- History manipulation commands (`history`, `fc`)

**100% local:**
- No network requests
- No telemetry
- Model stays on your machine

## Architecture

```
crates/aprender-shell/
├── src/
│   ├── main.rs      # CLI (clap)
│   ├── history.rs   # ZSH/Bash/Fish parser
│   ├── model.rs     # Markov n-gram model
│   └── trie.rs      # Prefix index
```

### History Parser

Handles multiple formats:

```rust
// ZSH extended: ": 1699900000:0;git status"
// Bash plain: "git status"
// Fish: "- cmd: git status"
```

### N-gram Model

Trigram Markov chain:

```
Context         → Next Token (count)
""              → "git" (340), "cargo" (245), "cd" (198)
"git"           → "commit" (89), "push" (45), "status" (340)
"git commit"    → "-m" (67), "--amend" (12)
```

### Trie Index

O(k) prefix lookup where k = prefix length:

```
g─i─t─ ─s─t─a─t─u─s (count: 340)
      └─c─o─m─m─i─t (count: 89)
      └─p─u─s─h     (count: 45)
```

## Why N-gram Beats Neural

For shell completion:

| Factor | N-gram | Neural (RNN/Transformer) |
|--------|--------|--------------------------|
| Training time | <1s | Minutes |
| Inference | <1ms | 10-50ms |
| Model size | 2MB | 50MB+ |
| Accuracy on shell | 70%+ | 75%+ |
| Cold start | Instant | GPU warmup |

Shell commands are repetitive patterns. N-gram captures this perfectly.

## CLI Reference

```
aprender-shell <COMMAND>

Commands:
  train       Full retrain from history
  update      Incremental update (fast)
  suggest     Get completions for prefix
  stats       Show model statistics
  export      Export model for sharing
  import      Import a shared model
  zsh-widget  Generate ZSH integration code

Options:
  -h, --help     Print help
  -V, --version  Print version
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Could not find history file" | Specify path: `-f ~/.bash_history` |
| Suggestions too generic | Increase n-gram: `-n 4` |
| Model too large | Decrease n-gram: `-n 2` |
| Slow suggestions | Check model size with `stats` |
