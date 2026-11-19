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
#   bash scripts/lab/check_head_dim_single_source.sh [--root DIR] [--allow FILE]
#   bash scripts/lab/check_head_dim_single_source.sh --self-test
#
# --allow FILE: one entry per line, TAB-separated: <repo-relative path> <normalized line text> <status>.
#   status is "legacy" or "known-defect #NNNN". An entry matches a hit on path plus the line text with
#   whitespace collapsed (never a line number); each entry consumes one hit. A hit with no entry is
#   red (NEW site); an entry with no hit is red ("remove the entry"). Default: the allowlist beside
#   this script, check_head_dim_single_source.allow.
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

  # allowlist judgement: match on path plus normalized text, never a line number
  local T=$'\t' H A
  H='a.rs:10: let d = hidden_dim / num_heads;'
  A="a.rs${T}let d = hidden_dim / num_heads;${T}legacy"
  jc() { # name want_new want_stale want_known want_legacy hits allow
    local got; judge "$6" "$7" >/dev/null
    got="${J_NEW}/${J_STALE}/${J_KNOWN}/${J_LEGACY}"
    if [[ "$got" == "$2/$3/$4/$5" ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL judge $1: got $got want $2/$3/$4/$5"; fi
  }
  jc listed-site-green          0 0 0 1 "$H" "$A"
  jc moved-line-number-green    0 0 0 1 'a.rs:99: let d = hidden_dim / num_heads;' "$A"
  jc reindented-green           0 0 0 1 'a.rs:10:        let   d = hidden_dim / num_heads;' "$A"
  jc new-site-red               1 0 0 1 "$H"$'\n''a.rs:20: let e = hidden_dim / n_heads;' "$A"
  jc new-file-red               1 1 0 0 'b.rs:10: let d = hidden_dim / num_heads;' "$A"
  jc removed-site-stale-red     0 1 0 0 '' "$A"
  jc stale-text-changed-red     1 1 0 0 'a.rs:10: let d = hidden_dim / num_heads + 1;' "$A"
  jc known-defect-green         0 0 1 0 "$H" "a.rs${T}let d = hidden_dim / num_heads;${T}known-defect #7285"
  jc duplicate-needs-two        1 0 0 1 "$H"$'\n''a.rs:12: let d = hidden_dim / num_heads;' "$A"
  jc comment-and-blank-ignored  0 0 0 1 "$H" $'# note\n\n'"$A"
  local jout
  jout="$(judge '' "$A")"
  if [[ "$jout" == *"remove the entry"* ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL stale message: $jout"; fi
  jout="$(judge "$H" '')"
  if [[ "$jout" == NEW\ * ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL new message: $jout"; fi

  echo "self-test: ${pass} passed, ${fail} failed"
  [[ "$fail" -eq 0 ]]
}

# --- allowlist judgement --------------------------------------------------------------

# norm: collapse runs of whitespace and trim, so an entry survives re-indentation and line moves.
norm() { sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//' <<<"$1"; }

# judge HITS ALLOW: HITS is "path:line: text" lines, ALLOW the allowlist text. Prints one row per
# hit and per stale entry; sets J_NEW, J_STALE, J_KNOWN, J_LEGACY (globals).
judge() {
  local hits="$1" allow="$2" line path text key a_path a_text a_status
  local -A avail=() status=()
  J_NEW=0 J_STALE=0 J_KNOWN=0 J_LEGACY=0
  while IFS=$'\t' read -r a_path a_text a_status; do
    [[ -z "$a_path" || "$a_path" == \#* ]] && continue
    key="${a_path}"$'\t'"$(norm "$a_text")"
    avail[$key]=$(( ${avail[$key]:-0} + 1 ))
    status[$key]="$a_status"
  done <<<"$allow"
  local -A used=()
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    path="${line%%:*}"; text="$(norm "${line#*: }")"; key="${path}"$'\t'"${text}"
    if (( ${avail[$key]:-0} > ${used[$key]:-0} )); then
      used[$key]=$(( ${used[$key]:-0} + 1 ))
      if [[ "${status[$key]}" == known-defect* ]]; then
        J_KNOWN=$((J_KNOWN + 1)); echo "KNOWN-DEFECT (${status[$key]}) ${line}"
      else
        J_LEGACY=$((J_LEGACY + 1)); echo "ALLOWED ${line}"
      fi
    else
      J_NEW=$((J_NEW + 1)); echo "NEW ${line}"
    fi
  done <<<"$hits"
  for key in "${!avail[@]}"; do
    if (( ${avail[$key]} > ${used[$key]:-0} )); then
      J_STALE=$((J_STALE + 1)); echo "STALE allow entry, remove the entry: ${key//$'\t'/ :: }"
    fi
  done
}

# --- main ---------------------------------------------------------------------------

main() {
  local root allow="" allow_given=0
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  if [[ "${1:-}" == "--self-test" ]]; then self_test; exit $?; fi
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --root) root="${2:-}"; shift 2 ;;
      --allow) allow="${2:-}"; allow_given=1; shift 2 ;;
      *) echo "usage: $0 [--root DIR] [--allow FILE] | --self-test" >&2; exit 2 ;;
    esac
  done
  if [[ "$allow_given" -eq 0 ]]; then allow="$(dirname "${BASH_SOURCE[0]}")/check_head_dim_single_source.allow"; fi
  if [[ ! -r "$allow" ]]; then echo "not_measured: allowlist unreadable: ${allow}"; exit 3; fi

  local files scanned=0 f text out all_hits=""
  if ! files="$(git -C "$root" ls-files -- "$SCAN_DIR")" || [[ -z "$files" ]]; then
    echo "not_measured: no files listed under ${SCAN_DIR} in ${root}"; exit 3
  fi
  while IFS= read -r f; do
    [[ "$f" == *.rs ]] || continue
    [[ "$f" == "$ACCESSOR_FILE" ]] && continue
    is_test_path "$f" && continue
    scanned=$((scanned + 1))
    text="$(<"${root}/${f}")"
    out="$(scan_text "$f" "$text")"
    [[ -z "$out" ]] && continue
    all_hits+="${out}"$'\n'
  done <<<"$files"

  if [[ "$scanned" -eq 0 ]]; then echo "not_measured: scanned 0 non-test .rs files"; exit 3; fi

  judge "$all_hits" "$(<"$allow")"
  echo "scanned ${scanned} files: ${J_NEW} new, ${J_STALE} stale, ${J_KNOWN} known-defect, ${J_LEGACY} legacy"
  if [[ "$J_NEW" -gt 0 || "$J_STALE" -gt 0 ]]; then exit 1; fi
  exit 0
}

main "$@"
