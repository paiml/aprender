#!/usr/bin/env bash
# check_head_dim_single_source.sh -- LAB (nightly only; cannot block a merge or a release).
#
# head_dim has ONE source: the config accessor `config.head_dim()` (gguf/config.rs, GH-305,
# which reads the explicit `key_length` / attn_q shape). A site that derives it as
# `hidden / heads` is wrong for every model whose heads * head_dim != hidden. Qwen3-0.6B is
# one (16 x 128 = 2048 vs hidden 1024), and #7285 (CUDA logits differ from CPU, cosine 0.453)
# has this as a candidate cause.
#
# Red on any non-test site under crates/aprender-serve/src that divides a hidden width by a
# head count. Out of scope by construction: files whose path has a tests/ directory or whose
# basename contains "test", everything from the first `#[cfg(test)]` line of a file down,
# comment text, and gguf/config.rs (the accessor itself).
#
#   bash scripts/check_head_dim_single_source.sh [--root DIR] [--allow FILE]
#   bash scripts/check_head_dim_single_source.sh --self-test
#
# --allow FILE: lines "<repo-relative path> <ticket>". A hit in a listed path is reported as
#   ALLOWED, and a listed path with no hit is red (stale entry). None is shipped.
# Exit: 0 no hits; 1 hits or stale allow entry; 2 usage; 3 not_measured (no files scanned).
# Not wired into any workflow, PR path or guard tree.

set -euo pipefail

SCAN_DIR="crates/aprender-serve/src"
ACCESSOR_FILE="crates/aprender-serve/src/gguf/config.rs"

# The matcher. Reads one file's text on stdin; FNAME names it. Prints "path:line: text".
# Identifier boundaries are explicit (awk ERE has no \b), and the denominator is anchored to
# the start of an identifier path so `num_kv_heads` can never match `heads`.
AWK_PROG='
BEGIN {
  num = "(hidden_dim|hidden_size|hidden|embed_dim|embedding_dim|d_model|embedding_length)"
  den = "(num_heads|n_heads|n_head|num_attention_heads|head_count|heads)"
  pre = "(^|[^A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*[.])*"
  post = "([^A-Za-z0-9_]|$)"
  re_div = pre num "[ ]*(\\(\\))?[ ]*/[ ]*([A-Za-z_][A-Za-z0-9_]*[.])*" den post
  re_chk = pre num "[ ]*(\\(\\))?[.]checked_div[(][ ]*([A-Za-z_][A-Za-z0-9_]*[.])*" den "[ ]*[)]"
}
/#\[cfg\(test\)\]/ { exit }
{
  line = $0
  sub(/\/\/.*$/, "", line)
  if (line ~ re_div || line ~ re_chk) printf "%s:%d: %s\n", FNAME, NR, $0
}
'

# is_test_path PATH: 0 when the path is test code.
is_test_path() {
  local p="$1" base="${1##*/}"
  [[ "$p" == */tests/* || "$p" == tests/* || "$base" == *test* ]]
}

# scan_text FNAME TEXT: prints hits for one file's text.
scan_text() {
  awk -v FNAME="$1" "$AWK_PROG" <<<"$2"
}

# --- self-test --------------------------------------------------------------------

self_test() {
  local pass=0 fail=0
  hit() { # name text   (must match)
    local out; out="$(scan_text f.rs "$2")"
    if [[ -n "$out" ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL must-match $1: $2"; fi
  }
  miss() { # name text   (must NOT match)
    local out; out="$(scan_text f.rs "$2")"
    if [[ -z "$out" ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL must-not-match $1: $2"; fi
  }

  # must-match: the real shapes on origin/main and their neighbours
  hit let-hidden-over-heads        'let head_dim = model.config.hidden_dim / num_heads;'
  hit let-config-both              'let head_dim = model.config.hidden_dim / model.config.num_heads;'
  hit bare-names                   '        hidden_dim / num_heads'
  hit parenthesised-in-product     '            * (self.model.config.hidden_dim / self.model.config.num_heads);'
  hit hidden-size                  'let hd = cfg.hidden_size / cfg.num_attention_heads;'
  hit n-heads                      'let head_dim = hidden / n_heads;'
  hit n-head                       'let d = hidden_dim / n_head;'
  hit no-spaces                    'let d = hidden_dim/num_heads;'
  hit method-form                  'let d = self.hidden_dim() / self.num_heads();'
  hit checked-div                  'let default_head_dim = hidden_dim.checked_div(num_heads).unwrap_or(0);'
  hit checked-div-config           'let h = config.hidden_dim.checked_div(config.num_heads).unwrap_or(0);'
  hit checked-div-bare             'hidden.checked_div(heads)'
  hit in-struct-field              'head_dim: hidden_dim / num_heads,'
  hit trailing-comment-after-code  'let d = hidden_dim / num_heads; // derive'
  hit d-model                      'let d = d_model / num_heads;'

  # must-not-match
  miss accessor-call               'let head_dim = model.config.head_dim();'
  miss kv-heads                    'let g = hidden_dim / num_kv_heads;'
  miss kv-heads-config             'let g = config.hidden_dim / config.num_kv_heads;'
  miss group-size                  'let group_size = num_heads / num_kv_heads;'
  miss ffn-width                   'let x = ffn_hidden_dim / num_heads;'
  miss intermediate                'let x = intermediate_size / num_heads;'
  miss num-heads-numerator         'let x = num_heads / hidden_dim;'
  miss comment-only                '// head_dim = hidden_dim / num_heads is wrong for Qwen3'
  miss doc-comment                 '/// NOT `hidden_dim / num_heads`.'
  miss unrelated-division          'let per = total_tokens / batch_size;'
  miss heads-suffix-ident          'let x = hidden_dim / total_heads_seen;'
  miss hidden-prefix-ident         'let x = prehidden_dim / num_heads;'
  miss after-cfg-test              $'fn real() {}\n#[cfg(test)]\nmod tests { fn t() { let d = hidden_dim / num_heads; } }'

  # a hit before #[cfg(test)] still counts
  hit before-cfg-test              $'let d = hidden_dim / num_heads;\n#[cfg(test)]\nmod tests {}'

  # path classification
  tp() { # name want path
    local got=no; if is_test_path "$3"; then got=yes; fi
    if [[ "$got" == "$2" ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL path $1: $3 want $2"; fi
  }
  tp tests-dir        yes crates/aprender-serve/src/gguf/tests/tests_15.rs
  tp basename-tests   yes crates/aprender-serve/src/gguf/inference/attention_gqa_tests.rs
  tp mid-name-tests   yes crates/aprender-serve/src/gguf/inference/forward/single_tests_q8k.rs
  tp tests-prefix     yes crates/aprender-serve/src/gguf/inference/cached/sync_tests_constructors.rs
  tp layers-tests-dir yes crates/aprender-serve/src/layers/tests/imp_086.rs
  tp prod-cuda        no  crates/aprender-serve/src/gguf/cuda/mod.rs
  tp prod-parity-gate no  crates/aprender-serve/src/gguf/cuda/mod_parity_gate.rs
  tp prod-attention   no  crates/aprender-serve/src/gguf/inference/attention_gqa.rs

  # hit reporting carries path and line number
  local out
  out="$(scan_text a/b.rs $'fn x() {}\nlet d = hidden_dim / num_heads;')"
  if [[ "$out" == "a/b.rs:2: let d = hidden_dim / num_heads;" ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL report-format: $out"; fi

  echo "self-test: ${pass} passed, ${fail} failed"
  [[ "$fail" -eq 0 ]]
}

# --- main ---------------------------------------------------------------------------

main() {
  local root allow=""
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  if [[ "${1:-}" == "--self-test" ]]; then self_test; exit $?; fi
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --root) root="${2:-}"; shift 2 ;;
      --allow) allow="${2:-}"; shift 2 ;;
      *) echo "usage: $0 [--root DIR] [--allow FILE] | --self-test" >&2; exit 2 ;;
    esac
  done

  local files scanned=0 hits=0 f text out allowed_text="" a_path a_rest
  if ! files="$(git -C "$root" ls-files -- "$SCAN_DIR")" || [[ -z "$files" ]]; then
    echo "not_measured: no files listed under ${SCAN_DIR} in ${root}"; exit 3
  fi
  if [[ -n "$allow" ]]; then allowed_text="$(<"$allow")"; fi

  local seen_allowed=" "
  while IFS= read -r f; do
    [[ "$f" == *.rs ]] || continue
    [[ "$f" == "$ACCESSOR_FILE" ]] && continue
    is_test_path "$f" && continue
    scanned=$((scanned + 1))
    text="$(<"${root}/${f}")"
    out="$(scan_text "$f" "$text")"
    [[ -z "$out" ]] && continue
    local is_allowed=0
    while read -r a_path a_rest; do
      [[ -z "$a_path" || "$a_path" == \#* ]] && continue
      if [[ "$a_path" == "$f" ]]; then is_allowed=1; seen_allowed="${seen_allowed}${f} "; fi
    done <<<"$allowed_text"
    if [[ "$is_allowed" -eq 1 ]]; then
      while IFS= read -r line; do echo "ALLOWED ${line}"; done <<<"$out"
    else
      while IFS= read -r line; do echo "HIT ${line}"; hits=$((hits + 1)); done <<<"$out"
    fi
  done <<<"$files"

  if [[ "$scanned" -eq 0 ]]; then echo "not_measured: scanned 0 non-test .rs files"; exit 3; fi

  local stale=0
  while read -r a_path a_rest; do
    [[ -z "$a_path" || "$a_path" == \#* ]] && continue
    if [[ "$seen_allowed" != *" ${a_path} "* ]]; then echo "STALE allow entry (no hit): ${a_path}"; stale=1; fi
  done <<<"$allowed_text"

  echo "scanned ${scanned} files, ${hits} hits"
  if [[ "$hits" -gt 0 || "$stale" -eq 1 ]]; then exit 1; fi
  exit 0
}

main "$@"
