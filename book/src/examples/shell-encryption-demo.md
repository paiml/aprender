# Shell Model Encryption Demo

Demonstrates encrypted and unencrypted model formats in aprender-shell.

## Overview

This example shows:
1. Creating and training a shell completion model
2. Saving as unencrypted `.apr` file
3. Saving as encrypted `.apr` file (AES-256-GCM with Argon2id)

## Running

```bash
cargo run --example shell_encryption_demo --features format-encryption
```

## Code

See `examples/shell_encryption_demo.rs` for the full implementation.
