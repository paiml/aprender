#!/bin/bash
set -euo pipefail

# Sync test models for falsification tests
# Creates artifacts/models/ with required model files

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly ARTIFACTS_DIR="$PROJECT_ROOT/artifacts/models"

# HuggingFace cache paths
readonly HF_CACHE="${HF_HOME:-"$HOME/.cache/huggingface"}/hub"

# SEC010: Path traversal validation
validate_path() {
    local path="$1"
    local context="$2"
    if [[ "$path" == *".."* ]]; then
        echo "ERROR: Path traversal detected in $context: $path"
        exit 1
    fi
}

echo "=== Realizar Model Sync ==="
echo "Artifacts: $ARTIFACTS_DIR"
echo ""

# Validate artifacts dir is within project and no traversal
validate_path "$ARTIFACTS_DIR" "ARTIFACTS_DIR"
if [[ "$ARTIFACTS_DIR" != "$PROJECT_ROOT"* ]]; then
    echo "ERROR: Artifacts directory must be within project root"
    exit 1
fi

# bashrs: SEC010 - path validated above
mkdir -p "$ARTIFACTS_DIR"

# Model definitions: name|source_type|source_path|target_name
declare -a MODELS=(
    "qwen2-0.5b-q4_0|gguf|$HOME/src/HF-Advanced-Fine-Tuning/corpus/models/qwen2-0.5b-instruct-q4_0.gguf|qwen2-0.5b-q4_0.gguf"
    "qwen2-0.5b-st|safetensors|$HF_CACHE/models--Qwen--Qwen2-0.5B-Instruct/snapshots/*/model.safetensors|qwen2-0.5b.safetensors"
    "tinyllama-1.1b-st|safetensors|$HF_CACHE/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/*/model.safetensors|tinyllama-1.1b.safetensors"
)

sync_model() {
    local name="$1"
    local source_type="$2"
    local source_pattern="$3"
    local target_name="$4"
    local target_path="$ARTIFACTS_DIR/$target_name"

    # SEC010: Validate no path traversal in target
    validate_path "$target_name" "target_name"
    validate_path "$target_path" "target_path"

    # Validate target path is within artifacts dir
    if [[ "$target_path" != "$ARTIFACTS_DIR"* ]]; then
        echo "  [ERROR] Invalid target path: $target_path"
        return 1
    fi

    echo "[$name] Checking..."

    if [[ -f "$target_path" ]]; then
        local size
        size="$(stat -c%s "$target_path" 2>/dev/null || stat -f%z "$target_path")"
        local human_size
        human_size="$(numfmt --to=iec "$size" 2>/dev/null || echo "${size} bytes")"
        echo "  [OK] Already synced ($human_size)"
        return 0
    fi

    # Find source file (expand glob) - source_pattern intentionally unquoted for glob
    local source_file=""
    # shellcheck disable=SC2086
    source_file="$(ls -1 $source_pattern 2>/dev/null | head -1)" || true

    if [[ -z "$source_file" || ! -f "$source_file" ]]; then
        echo "  [SKIP] Source not found: $source_pattern"
        return 0
    fi

    # Validate source file exists and is readable
    if [[ ! -r "$source_file" ]]; then
        echo "  [ERROR] Source not readable: $source_file"
        return 1
    fi

    echo "  Syncing from: $source_file"

    case "$source_type" in
        gguf|safetensors)
            # bashrs: SEC010 - paths validated above via validate_path
            ln -sf "$source_file" "$target_path"
            echo "  [OK] Linked"
            ;;
        *)
            echo "  [ERROR] Unknown source type: $source_type"
            return 1
            ;;
    esac
}

for model_def in "${MODELS[@]}"; do
    # Parse pipe-delimited fields
    name="${model_def%%|*}"
    rest="${model_def#*|}"
    source_type="${rest%%|*}"
    rest="${rest#*|}"
    source_path="${rest%%|*}"
    target_name="${rest#*|}"

    sync_model "$name" "$source_type" "$source_path" "$target_name"
done

echo ""
echo "=== Artifacts Status ==="
ls -la "$ARTIFACTS_DIR/" 2>/dev/null || echo "(empty)"

echo ""
echo "=== Test Readiness ==="
echo "Run: cargo test --lib falsification -- --nocapture"
