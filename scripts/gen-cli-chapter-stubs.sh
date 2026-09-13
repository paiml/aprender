#!/usr/bin/env bash
# Generate book chapter stubs for every `apr <cmd>` that doesn't already
# have one. Per BOOK-CLOSEOUT-001 spec § Phase 2.
#
# Constraint: every stub MUST include at least one runnable example.
# The example is seeded from `apr <cmd> --help` synopsis + a known-safe
# invocation pattern keyed by command category (inspection, transform,
# inference, registry, training, ui, misc).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CLI_DIR="book/src/cli"
mkdir -p "$CLI_DIR"

# This generator writes book chapters from `apr --help`. Hardcoding one
# developer's ~/.cargo/bin meant the generated chapters documented whatever was
# installed there, on any machine that happened to have it (#2358).
. scripts/apr_bin.sh || exit 2

# Categories drive the example. Default is `apr <cmd> --help`.
declare -A EXAMPLE
EXAMPLE[inspect]='apr inspect qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'
EXAMPLE[debug]='apr debug qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'
EXAMPLE[validate]='apr validate qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --quality'
EXAMPLE[lint]='apr lint qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'
EXAMPLE[tensors]='apr tensors qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --json | jq length'
EXAMPLE[trace]='apr trace qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --prompt "What is 2+2?" --max-tokens 4'
EXAMPLE[diff]='apr diff qwen2.5-coder-1.5b-instruct-q4_k_m.gguf qwen2.5-coder-1.5b-instruct-q4k.apr'
EXAMPLE[hex]='apr hex qwen2.5-coder-1.5b-instruct-q4_k_m.gguf | head -20'
EXAMPLE[tree]='apr tree qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'
EXAMPLE[flow]='apr flow qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'
EXAMPLE[explain]='apr explain qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'
EXAMPLE[qa]='apr qa qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'
EXAMPLE[qualify]='apr qualify qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'
EXAMPLE[run]='apr run qwen2.5-coder-1.5b "What is 2+2?" --max-tokens 16'
EXAMPLE[chat]='apr chat qwen2.5-coder-1.5b'
EXAMPLE[serve]='apr serve run qwen2.5-coder-1.5b --port 8080'
EXAMPLE[pull]='apr pull hf://Qwen/Qwen2.5-Coder-0.5B-Instruct'
EXAMPLE[list]='apr list --json | jq length'
EXAMPLE[rm]='apr rm <model-id>          # from \`apr list\` output'
EXAMPLE[gpu]='apr gpu --json'
EXAMPLE[convert]='apr convert model.safetensors --quantize q4_k -o model-q4k.apr'
EXAMPLE[export]='apr export model.apr --format gguf -o model.gguf'
EXAMPLE[import]='apr import hf://openai/whisper-tiny -o whisper.apr --arch whisper'
EXAMPLE[quantize]='apr quantize model.apr --to q4_k -o model-q4k.apr'
EXAMPLE[merge]='apr merge model1.apr model2.apr --strategy weighted --weights 0.7,0.3 -o merged.apr'
EXAMPLE[prune]='apr prune model.apr --ratio 0.5 -o pruned.apr'
EXAMPLE[compile]='apr compile model.apr --target cuda -o compiled.apr'
EXAMPLE[bench]='apr bench qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --iterations 10'
EXAMPLE[profile]='apr profile qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'
EXAMPLE[eval]='apr eval qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --suite humaneval --limit 5'
EXAMPLE[canary]='apr canary qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'
EXAMPLE[finetune]='apr finetune qwen2.5-coder-0.5b --data train.jsonl --epochs 3'
EXAMPLE[distill]='apr distill --teacher 7b.apr --student 0.5b.apr --data train.jsonl'
EXAMPLE[train]='apr train config.yaml'
EXAMPLE[pretrain]='apr pretrain config.yaml'
EXAMPLE[tokenize]='apr tokenize "Hello world" --tokenizer qwen2.5-coder-1.5b'
EXAMPLE[tune]='apr tune qwen2.5-coder-0.5b --data train.jsonl'
EXAMPLE[data]='apr data inspect train.jsonl'
EXAMPLE[code]='apr code -p "review this Python function" --max-turns 1'
EXAMPLE[probar]='apr probar test-suite.yaml'
EXAMPLE[cbtop]='apr cbtop'
EXAMPLE[tui]='apr tui'
EXAMPLE[monitor]='apr monitor'
EXAMPLE[runs]='apr runs list'
EXAMPLE[experiment]='apr experiment list'
EXAMPLE[pipeline]='apr pipeline run config.yaml'
EXAMPLE[diagnose]='apr diagnose'
EXAMPLE[showcase]='apr showcase'
EXAMPLE[rosetta]='apr rosetta'
EXAMPLE[publish]='apr publish staging/ paiml/my-model-v1 --library-name aprender --license MIT'
EXAMPLE[oracle]='apr oracle --rag "your question here"'
EXAMPLE[ptx]='apr ptx --gpu-arch sm_89 kernel.ptx'
EXAMPLE[ptx-map]='apr ptx-map model.apr --gpu-arch sm_89'
EXAMPLE[encrypt]='apr encrypt model.apr --key my-key -o model.enc'
EXAMPLE[decrypt]='apr decrypt model.enc --key my-key -o model.apr'
EXAMPLE[compare-hf]='apr compare-hf model.apr --hf-repo Qwen/Qwen2.5-Coder-1.5B-Instruct'
EXAMPLE[parity]='apr parity model.gguf --backends cpu,gpu'
EXAMPLE[check]='apr check qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'
EXAMPLE[registry]='apr registry status'
EXAMPLE[mcp]='apr mcp serve'
EXAMPLE[help]='apr help <command>'

# Default fallback example for commands not in the table above
default_example() {
  local cmd="$1"
  echo "apr $cmd --help"
}

# Categorize a command for the SUMMARY.md section
# (Each category maps to a # heading in SUMMARY.md.)
categorize() {
  local cmd="$1"
  case "$cmd" in
    inspect|debug|validate|lint|tensors|trace|diff|hex|tree|flow|explain) echo "Inspection" ;;
    qa|qualify|bench|eval|canary|compare-hf|parity|check|profile) echo "Quality & Evaluation" ;;
    convert|export|import|quantize|merge|prune|compile|encrypt|decrypt) echo "Model Transform" ;;
    run|chat|serve|code) echo "Inference" ;;
    pull|list|rm|gpu|registry) echo "Registry & Resources" ;;
    finetune|distill|train|pretrain|tokenize|tune|data) echo "Training" ;;
    tui|monitor|runs|experiment|pipeline|diagnose|showcase|cbtop) echo "Observability & Pipeline" ;;
    rosetta|publish|oracle|probar|ptx*|mcp|help) echo "Tools & Integration" ;;
    *-lint) echo "Linters" ;;
    *) echo "Other" ;;
  esac
}

# Get all 103 commands
cmds=$("$APR" --help 2>&1 | awk '/^Commands:/{f=1; next} f && /^  [a-z]/{print $1}')

NEW_STUBS=0
SKIPPED=0
for cmd in $cmds; do
  # $cmd is parsed from `apr --help` output; validate it before it is used to
  # build a file path below (cat > "$STUB_PATH") so a malformed help line can
  # never turn into a path-traversal write.
  case "$cmd" in
    *..*|/*) echo "skipping malformed command name: $cmd" >&2; continue ;;
  esac
  # Skip if a chapter for this command already exists anywhere in book/src
  if grep -rEq "(^|/)cli/${cmd}\.md|\b${cmd}\.md" book/src/SUMMARY.md 2>/dev/null; then
    : # found in SUMMARY — but the path could be wrong; check actual file
  fi
  STUB_PATH="$CLI_DIR/${cmd}.md"
  if [ -f "$STUB_PATH" ]; then
    SKIPPED=$((SKIPPED+1))
    continue
  fi

  # Get help text
  HELP=$("$APR" "$cmd" --help 2>&1 || true)
  DESC=$(echo "$HELP" | head -1 | tr -d '\r')
  if [ -z "$DESC" ]; then
    DESC="The \`apr $cmd\` command."
  fi
  USAGE=$(echo "$HELP" | grep -A1 "^Usage:" | tail -1 | tr -d '\r' | sed 's/^[[:space:]]*//')

  # Example: looked up by category-driven table; fall back to --help.
  EX="${EXAMPLE[$cmd]:-$(default_example "$cmd")}"

  CATEGORY=$(categorize "$cmd")

  # Generate stub
  cat > "$STUB_PATH" <<MARKDOWN
<!-- PCU: cli-${cmd} | contract: contracts/apr-page-cli-${cmd}-v1.yaml -->

# apr ${cmd}

${DESC}

**Category**: ${CATEGORY}

## Synopsis

\`\`\`text
${USAGE:-apr ${cmd} [OPTIONS]}
\`\`\`

## Example

\`\`\`bash
${EX}
\`\`\`

## Full help

Run \`apr ${cmd} --help\` for the complete option list.

## See also

- Source: [\`crates/apr-cli/src/commands/${cmd//-/_}.rs\`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/${cmd//-/_}.rs)
- Contract: [\`contracts/apr-page-cli-${cmd}-v1.yaml\`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-${cmd}-v1.yaml)

MARKDOWN
  NEW_STUBS=$((NEW_STUBS+1))
done

echo "Generated ${NEW_STUBS} stubs (${SKIPPED} skipped — already exist)"
echo "Stubs in: $CLI_DIR"
