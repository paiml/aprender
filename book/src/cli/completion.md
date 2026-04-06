# Completion Command

The `entrenar completion` command generates shell completion scripts for bash, zsh, fish, and PowerShell.

## Usage

```bash
entrenar completion <SHELL>
```

## Arguments

| Argument | Description |
|----------|-------------|
| `<SHELL>` | Target shell: bash, zsh, fish, powershell |

## Installation

### Bash

```bash
# Generate and install completions
entrenar completion bash > ~/.local/share/bash-completion/completions/entrenar

# Or for system-wide installation (requires sudo)
entrenar completion bash | sudo tee /etc/bash_completion.d/entrenar > /dev/null

# Reload completions
source ~/.bashrc
```

### Zsh

```bash
# Generate completions
entrenar completion zsh > ~/.zsh/completions/_entrenar

# Add to fpath in ~/.zshrc
fpath=(~/.zsh/completions $fpath)

# Rebuild completion cache
rm -f ~/.zcompdump && compinit
```

### Fish

```bash
# Generate and install completions
entrenar completion fish > ~/.config/fish/completions/entrenar.fish

# Completions are automatically loaded
```

### PowerShell

```powershell
# Generate completions
entrenar completion powershell | Out-File -Encoding utf8 $PROFILE.CurrentUserAllHosts

# Or append to existing profile
entrenar completion powershell | Add-Content $PROFILE
```

## Completion Features

The generated completions provide:

- **Command completion** - All subcommands (train, validate, info, etc.)
- **Option completion** - All flags and options
- **File path completion** - For file arguments (.yaml, .safetensors, etc.)
- **Value completion** - For enum options (format, method, etc.)

## Example Usage

After installation, type `entrenar ` and press Tab:

```bash
$ entrenar <TAB>
audit       completion  info        inspect     merge       monitor
quantize    research    train       validate

$ entrenar train <TAB>
config.yaml  examples/    mnist.yaml

$ entrenar train config.yaml --<TAB>
--batch-size  --dry-run     --epochs      --help
--lr          --output-dir  --quiet       --verbose
```

## Verification

Verify completions are working:

```bash
# Bash
complete -p entrenar

# Zsh
which _entrenar

# Fish
complete -c entrenar
```

## Troubleshooting

### Completions Not Working

1. **Check installation path**:
   ```bash
   # Bash
   ls ~/.local/share/bash-completion/completions/entrenar

   # Zsh
   echo $fpath | grep completions
   ```

2. **Reload shell**:
   ```bash
   exec $SHELL
   ```

3. **Regenerate completions**:
   ```bash
   entrenar completion bash > /tmp/entrenar.bash
   source /tmp/entrenar.bash
   ```

### Permission Issues

```bash
# Use sudo for system-wide installation
sudo entrenar completion bash > /etc/bash_completion.d/entrenar
```

## See Also

- [CLI Overview](./overview.md) - General CLI reference
- [Installation](../getting-started/installation.md) - Installation guide
