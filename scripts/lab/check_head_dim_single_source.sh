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
# basename contains "test", the BODY of an item marked `#[cfg(test)]` (production code before
# and after it is still scanned), comment text, string literals, and gguf/config.rs (the
# accessor itself).
#
# The matcher reads comment- and string-stripped code, joins a statement that rustfmt split
# over up to three lines, and accepts `as` casts, parenthesised operands and `()` method
# receivers on either side. A hit on a split statement is reported at its first line.
#
#   bash scripts/lab/check_head_dim_single_source.sh [--root DIR] [--allow FILE]
#   bash scripts/lab/check_head_dim_single_source.sh --self-test
#
# --allow FILE: one entry per line, TAB-separated:
#     <repo-relative path> <normalized line text> <status> [<count>]
#   status is "legacy" or "known-defect #NNNN" (anything else is an error). count is a positive
#   integer, default 1: how many identical sites the entry covers. A (path, text) pair may appear
#   on ONE row only; a repeated row is an error. An entry matches a hit on path plus the line
#   text with whitespace collapsed (never a line number). A hit with no entry is red (NEW
#   site); entry capacity with no hit is red ("remove the entry"). Default: the allowlist
#   beside this script, check_head_dim_single_source.allow.
# Exit: 0 no hits; 1 hits or stale allow entry; 2 usage (unknown option, option missing its
#   value); 3 not_measured (no files scanned, unreadable or non-file allowlist, malformed
#   allowlist, root not a directory, a listed source file unreadable or deleted).
# Not wired into any workflow, PR path or guard tree.

set -euo pipefail

SCAN_DIR="crates/aprender-serve/src"
ACCESSOR_FILE="crates/aprender-serve/src/gguf/config.rs"

# The matcher. Reads one file's text on stdin; FNAME names it. Prints "path:line: text".
# Identifier boundaries are explicit (awk ERE has no \b), and the denominator is anchored to
# the start of an identifier path so `num_kv_heads` can never match `heads`.
AWK_PROG='
function strip(s,   out, i, n, c, c2, j) {
  out = ""; n = length(s); i = 1
  while (i <= n) {
    c = substr(s, i, 1); c2 = substr(s, i, 2)
    if (inblk) { if (c2 == "*/") { inblk = 0; i += 2 } else i++; continue }
    if (c2 == "//") break
    if (c2 == "/*") { inblk = 1; i += 2; out = out " "; continue }
    if (c == "\"") {
      i++
      while (i <= n) {
        c = substr(s, i, 1)
        if (c == "\\") { i += 2; continue }
        if (c == "\"") { i++; break }
        i++
      }
      out = out "\"\""; continue
    }
    if (c == SQ) {
      if (substr(s, i + 2, 1) == SQ) { i += 3; out = out " "; continue }
      if (substr(s, i + 1, 1) == "\\") {
        j = index(substr(s, i + 2), SQ)
        if (j > 0 && j <= 4) { i += 2 + j; out = out " "; continue }
      }
    }
    out = out c; i++
  }
  return out
}
function cnt(s, ch,   t) { t = s; return gsub(ch, "", t) }
function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s); return s }
BEGIN {
  SQ = sprintf("%c", 39)
  num = "(hidden_dim|hidden_size|hidden|embed_dim|embedding_dim|d_model|embedding_length)"
  den = "(num_heads|n_heads|n_head|num_attention_heads|head_count|heads)"
  recv = "([A-Za-z_][A-Za-z0-9_]*(\\(\\))?[.])*"
  pre = "(^|[^A-Za-z0-9_])" recv
  post = "([^A-Za-z0-9_]|$)"
  cast = "([ ]+as[ ]+[A-Za-z0-9_]+)?"
  numtail = "[ ]*(\\(\\))?" cast "[ ]*\\)?[ ]*"
  re_div = pre num numtail "/[ ]*\\(?[ ]*" recv den "(\\(\\))?" post
  # a test-only attribute: cfg(test), or cfg(all(.., test, ..)) with test a bare member
  re_cfgt = "^ *#\\[cfg\\((all\\(([^()]*,)? *)?test *(,[^()]*)?\\)?\\)\\]"
  re_chk = pre num numtail "[.]checked_div[(][ ]*\\(?[ ]*" recv den "(\\(\\))?" cast "[ ]*\\)?[ ]*[)]"
}
{ c = strip($0); gsub(/\t/, " ", c); code[NR] = c; orig[NR] = $0 }
END {
  # blank the body of every #[cfg(test)] item, and only that
  mode = 0; depth = 0
  for (i = 1; i <= NR; i++) {
    c = code[i]
    if (mode == 0) {
      if (c !~ re_cfgt) continue
      sub(re_cfgt " *", "", c); mode = 1
    }
    if (mode == 1) {
      code[i] = ""
      if (c ~ /^ *$/ || c ~ /^ *#\[/) continue
      po = index(c, "{"); ps = index(c, ";")
      if (po && (!ps || po < ps)) { depth = cnt(c, "[{]") - cnt(c, "[}]"); mode = (depth > 0) ? 2 : 0; continue }
      if (ps) { mode = 0; continue }
      continue
    }
    if (mode == 2) { depth += cnt(c, "[{]") - cnt(c, "[}]"); code[i] = ""; if (depth <= 0) mode = 0 }
  }
  for (i = 1; i <= NR; i++) {
    if (code[i] ~ /^ *$/) continue
    L = length(code[i]); w = code[i]; used = 0; lastj = i; j = i
    while (used < 2 && ++j <= NR) { if (code[j] ~ /^ *$/) continue; w = w " " code[j]; used++; lastj = j }
    hit = 0; end = 0
    if (match(w, re_div)) {
      es = RSTART + ((substr(w, RSTART, 1) ~ /[^A-Za-z0-9_]/) ? 1 : 0)
      if (es <= L) { hit = 1; end = RSTART + RLENGTH - 1 }
    }
    if (!hit && match(w, re_chk)) {
      es = RSTART + ((substr(w, RSTART, 1) ~ /[^A-Za-z0-9_]/) ? 1 : 0)
      if (es <= L) { hit = 1; end = RSTART + RLENGTH - 1 }
    }
    if (!hit) continue
    text = orig[i]
    if (end > L + 1) {   # L+1 is the joining space the trailing boundary may consume
      for (j = i + 1; j <= lastj; j++) if (code[j] !~ /^ *$/) text = text " " trim(orig[j])
    }
    printf "%s:%d: %s\n", FNAME, i, text
  }
}
'

# is_test_path PATH: 0 when the path is test code.
is_test_path() {
  local p="$1" base="${1##*/}"
  [[ "$p" == */tests/* || "$p" == tests/* || "$p" == */testing/* || "$base" == *test* ]]   # testing/ is `#[cfg(test)] pub mod testing` in lib.rs
}

# scan_text FNAME TEXT: prints hits for one file's text.
scan_text() {
  awk -v FNAME="$1" "$AWK_PROG" <<<"$2"
}

# --- allowlist ------------------------------------------------------------------------

# norm: collapse runs of whitespace and trim, so an entry survives re-indentation and line moves.
norm() { sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//' <<<"$1"; }

# allow_problems ALLOW: prints one line per malformed row (bad status, bad count, extra column,
# missing column) and per repeated (path, text) pair.
allow_problems() {
  local a_path a_text a_status a_count a_extra key n=0
  local -A seen=()
  while IFS=$'\t' read -r a_path a_text a_status a_count a_extra || [[ -n "${a_path:-}" ]]; do
    n=$((n + 1))
    [[ -z "${a_path:-}" || "$a_path" == \#* ]] && continue
    if [[ -z "${a_text:-}" ]]; then echo "row ${n}: missing text column"; continue; fi
    if ! [[ "${a_status:-}" =~ ^(legacy|known-defect\ \#[0-9]+)$ ]]; then
      echo "row ${n}: status '${a_status:-}' is not 'legacy' or 'known-defect #NNNN'"
    fi
    if [[ -n "${a_count:-}" ]] && ! [[ "$a_count" =~ ^[1-9][0-9]*$ ]]; then
      echo "row ${n}: count '${a_count}' is not a positive integer"
    fi
    if [[ -n "${a_extra:-}" ]]; then echo "row ${n}: more than four columns"; fi
    key="${a_path}"$'\t'"$(norm "$a_text")"
    if [[ -n "${seen[$key]:-}" ]]; then echo "row ${n}: repeated (path, text) pair, first on row ${seen[$key]}; use the count column"; fi
    seen[$key]="${seen[$key]:-$n}"
  done <<<"$1"
}

# judge HITS ALLOW: HITS is "path:line: text" lines, ALLOW the allowlist text (already
# validated). Prints one row per hit and per stale entry; sets J_NEW, J_STALE, J_KNOWN,
# J_LEGACY (globals).
judge() {
  local hits="$1" allow="$2" line path text key a_path a_text a_status a_count a_extra left
  local -A avail=() status=()
  J_NEW=0 J_STALE=0 J_KNOWN=0 J_LEGACY=0
  while IFS=$'\t' read -r a_path a_text a_status a_count a_extra || [[ -n "${a_path:-}" ]]; do
    [[ -z "${a_path:-}" || "$a_path" == \#* ]] && continue
    key="${a_path}"$'\t'"$(norm "$a_text")"
    avail[$key]=$(( ${avail[$key]:-0} + ${a_count:-1} ))
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
    left=$(( ${avail[$key]} - ${used[$key]:-0} ))
    if (( left > 0 )); then
      J_STALE=$((J_STALE + left)); echo "STALE allow entry (${left} unused), remove the entry: ${key//$'\t'/ :: }"
    fi
  done
}

# --- main ---------------------------------------------------------------------------

usage() {
  echo "usage: $0 [--root DIR] [--allow FILE] | --self-test" >&2
  exit 2
}

main() {
  local root allow="" allow_given=0
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  if [[ "${1:-}" == "--self-test" ]]; then self_test; exit $?; fi
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --root | --allow)
        [[ $# -ge 2 ]] || usage
        if [[ "$1" == --root ]]; then root="$2"; else allow="$2"; allow_given=1; fi
        shift 2 ;;
      *) usage ;;
    esac
  done
  if [[ "$allow_given" -eq 0 ]]; then allow="$(dirname "${BASH_SOURCE[0]}")/check_head_dim_single_source.allow"; fi
  if [[ ! -f "$allow" || ! -r "$allow" ]]; then echo "not_measured: allowlist missing, unreadable or not a file: ${allow}"; exit 3; fi
  if [[ ! -d "$root" ]]; then echo "not_measured: root is not a directory: ${root}"; exit 3; fi
  local allow_text problems
  if ! allow_text="$(<"$allow")"; then echo "not_measured: cannot read allowlist ${allow}"; exit 3; fi
  problems="$(allow_problems "$allow_text")"
  if [[ -n "$problems" ]]; then echo "not_measured: malformed allowlist ${allow}:"; echo "$problems"; exit 3; fi

  local files scanned=0 f text out all_hits=""
  if ! files="$(git -C "$root" ls-files -co --exclude-standard -- "$SCAN_DIR")" || [[ -z "$files" ]]; then
    echo "not_measured: no files listed under ${SCAN_DIR} in ${root}"; exit 3
  fi
  while IFS= read -r f; do
    [[ "$f" == *.rs ]] || continue
    [[ "$f" == "$ACCESSOR_FILE" ]] && continue
    is_test_path "$f" && continue
    if [[ ! -f "${root}/${f}" || ! -r "${root}/${f}" ]]; then
      echo "not_measured: listed source file is deleted or unreadable: ${f}"; exit 3
    fi
    scanned=$((scanned + 1))
    if ! text="$(<"${root}/${f}")"; then echo "not_measured: cannot read ${f}"; exit 3; fi
    out="$(scan_text "$f" "$text")"
    [[ -z "$out" ]] && continue
    all_hits+="${out}"$'\n'
  done <<<"$files"

  if [[ "$scanned" -eq 0 ]]; then echo "not_measured: scanned 0 non-test .rs files"; exit 3; fi

  judge "$all_hits" "$allow_text"
  echo "scanned ${scanned} files: ${J_NEW} new, ${J_STALE} stale, ${J_KNOWN} known-defect, ${J_LEGACY} legacy"
  if [[ "$J_NEW" -gt 0 || "$J_STALE" -gt 0 ]]; then exit 1; fi
  exit 0
}

# --- self-test --------------------------------------------------------------------

self_test() {
  local pass=0 fail=0 tmp T=$'\t'
  tmp="$(mktemp -d)"
  trap '[[ -n "${tmp:-}" ]] && rm -rf "${tmp:?}"' RETURN
  hit() { # name text   (must match)
    local out; out="$(scan_text f.rs "$2")"
    if [[ -n "$out" ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL must-match $1: $2"; fi
  }
  miss() { # name text   (must NOT match)
    local out; out="$(scan_text f.rs "$2")"
    if [[ -z "$out" ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL must-not-match $1: $2 -> $out"; fi
  }
  t() { # name want got
    if [[ "$2" == "$3" ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL $1: want '$2' got '$3'"; fi
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
  hit embed-dim                    'let d = embed_dim / num_heads;'
  hit embedding-dim                'let d = embedding_dim / num_heads;'
  hit embedding-length             'let d = embedding_length / num_heads;'
  hit attention-heads-name         'let d = hidden_size / num_attention_heads;'
  hit head-count                   'let d = hidden_size / head_count;'
  hit plain-heads                  'let d = hidden_size / heads;'

  # must-match: forms the first version missed (rustfmt splits, casts, parentheses, methods)
  hit split-slash-leading          $'let head_dim = self.model.config.hidden_dim\n    / self.model.config.num_heads;'
  hit split-slash-trailing         $'let head_dim = hidden_dim /\n    num_heads;'
  hit split-checked-div            $'let h = hidden_dim\n    .checked_div(num_heads)\n    .unwrap_or(0);'
  hit split-with-comment-between   $'let h = hidden_dim\n    // why\n    / num_heads;'
  hit cast-both                    'let d = hidden_dim as usize / num_heads as usize;'
  hit cast-numerator               'let d = hidden_dim as usize / num_heads;'
  hit paren-numerator              'let d = (hidden_dim) / num_heads;'
  hit paren-cast-numerator         'let d = (hidden_dim as usize) / num_heads;'
  hit paren-denominator            'let d = hidden_dim / (num_heads);'
  hit paren-cast-denominator       'let d = hidden_dim / (num_heads as usize);'
  hit method-denominator-chain     'let d = hidden_dim / self.config().num_heads;'
  hit method-numerator-chain       'let d = self.config().hidden_dim / num_heads;'
  hit method-both-chains           'let d = self.config().hidden_dim() / self.config().num_heads();'
  hit checked-div-method-den       'let d = self.hidden_dim().checked_div(self.num_heads());'
  hit checked-div-paren-cast       'let d = (hidden_dim as usize).checked_div(num_heads as usize);'
  hit checked-div-paren-den        'let d = hidden_dim.checked_div((num_heads));'
  hit block-comment-between        'let d = hidden_dim /* x */ / num_heads;'
  hit cfg-test-mentioned-in-comment $'// helpers live in #[cfg(test)] at bottom\nfn f(hidden_dim: usize, num_heads: usize) -> usize { hidden_dim / num_heads }'
  hit cfg-test-in-doc-comment      $'/// see #[cfg(test)]\nfn f() { let d = hidden_dim / num_heads; }'
  hit cfg-test-in-block-comment    $'/* #[cfg(test)] */\nfn f() { let d = hidden_dim / num_heads; }'
  hit cfg-test-in-string           $'let s = "#[cfg(test)]";\nlet d = hidden_dim / num_heads;'
  hit after-cfg-test-use           $'#[cfg(test)]\nuse super::*;\nfn f() { let d = hidden_dim / num_heads; }'
  hit after-cfg-test-mod-closes    $'#[cfg(test)]\nmod tests {\n    fn t() { if x { y } }\n}\nfn f() { let d = hidden_dim / num_heads; }'
  hit after-cfg-test-mod-same-line $'#[cfg(test)] mod tests { fn t() {} }\nfn f() { let d = hidden_dim / num_heads; }'
  hit after-cfg-test-extern-mod    $'#[cfg(test)]\nmod tests;\nfn f() { let d = hidden_dim / num_heads; }'
  hit after-cfg-test-fn            $'#[cfg(test)]\nfn helper() {\n    let x = 1;\n}\nfn f() { let d = hidden_dim / num_heads; }'
  hit after-cfg-test-with-allow    $'#[cfg(test)]\n#[allow(dead_code)]\nmod tests {\n}\nfn f() { let d = hidden_dim / num_heads; }'
  hit cfg-not-test-is-production   $'#[cfg(not(test))]\nfn f() { let d = hidden_dim / num_heads; }'
  hit slashes-in-string-before     'let u = "http://x"; let d = hidden_dim / num_heads;'
  hit char-literal-before          $'let c = \'/\'; let d = hidden_dim / num_heads;'
  hit before-cfg-test              $'let d = hidden_dim / num_heads;\n#[cfg(test)]\nmod tests {}'

  # must-not-match
  miss accessor-call               'let head_dim = model.config.head_dim();'
  miss kv-heads                    'let g = hidden_dim / num_kv_heads;'
  miss kv-heads-config             'let g = config.hidden_dim / config.num_kv_heads;'
  miss kv-heads-paren              'let g = hidden_dim / (num_kv_heads as usize);'
  miss group-size                  'let group_size = num_heads / num_kv_heads;'
  miss ffn-width                   'let x = ffn_hidden_dim / num_heads;'
  miss intermediate                'let x = intermediate_size / num_heads;'
  miss num-heads-numerator         'let x = num_heads / hidden_dim;'
  miss comment-only                '// head_dim = hidden_dim / num_heads is wrong for Qwen3'
  miss doc-comment                 '/// NOT `hidden_dim / num_heads`.'
  miss block-comment-inline        'let d = 1; /* hidden_dim / num_heads */'
  miss block-comment-multiline     $'/*\n hidden_dim / num_heads\n*/\nlet d = 1;'
  miss string-literal              'let m = "hidden_dim / num_heads";'
  miss unrelated-division          'let per = total_tokens / batch_size;'
  miss heads-suffix-ident          'let x = hidden_dim / total_heads_seen;'
  miss hidden-prefix-ident         'let x = prehidden_dim / num_heads;'
  miss after-cfg-test              $'fn real() {}\n#[cfg(test)]\nmod tests { fn t() { let d = hidden_dim / num_heads; } }'
  miss inside-cfg-test-nested      $'#[cfg(test)]\nmod tests {\n    mod inner {\n        fn t() { let d = hidden_dim / num_heads; }\n    }\n    fn u() { let d = hidden_dim / num_heads; }\n}'
  miss inside-cfg-test-fn          $'#[cfg(test)]\nfn helper() {\n    let d = hidden_dim / num_heads;\n}'
  miss inside-cfg-test-brace-string $'#[cfg(test)]\nmod tests {\n    fn t() { let s = "}"; let d = hidden_dim / num_heads; }\n}'
  miss cfg-test-use-item           $'#[cfg(test)] use x::hidden_dim / num_heads;'
  miss inside-cfg-all-test-first   $'#[cfg(all(test, feature = "cuda"))]\nmod t { fn f() { let d = hidden_dim / num_heads; } }'
  miss inside-cfg-all-test-last    $'#[cfg(all(feature = "cuda", test))]\nmod t { fn f() { let d = hidden_dim / num_heads; } }'
  hit cfg-any-test-is-production  $'#[cfg(any(test, feature = "x"))]\nmod t { fn f() { let d = hidden_dim / num_heads; } }'
  hit cfg-all-not-test-is-production $'#[cfg(all(not(test), feature = "x"))]\nfn f() { let d = hidden_dim / num_heads; }'
  hit cfg-testing-feature-is-production $'#[cfg(testing)]\nfn f() { let d = hidden_dim / num_heads; }'
  miss separate-statements         $'let a = hidden_dim;\nlet b = x / num_heads;'
  miss next-line-kv-heads          $'let a = hidden_dim\n    / num_kv_heads;'
  miss next-line-unrelated         $'let a = hidden_dim;\n\nlet b = other / num_heads;'

  # a split hit is reported once, at its first line, with both lines of text
  local out
  out="$(scan_text f.rs $'fn x() {}\nlet d = hidden_dim\n    / num_heads;')"
  t split-report-once-first-line "f.rs:2: let d = hidden_dim / num_heads;" "$out"
  out="$(scan_text f.rs $'foo(\nhidden_dim / num_heads)')"
  t boundary-char-not-double-counted "f.rs:2: hidden_dim / num_heads)" "$out"
  out="$(scan_text f.rs $'let a = (\n    hidden_dim\n    / num_heads\n);')"
  t three-line-single-hit 1 "$(grep -c . <<<"$out")"
  out="$(scan_text f.rs $'let h = if n > 0 {\n    hidden_dim / num_heads\n} else { 0 };')"
  t boundary-space-not-a-span "f.rs:2:     hidden_dim / num_heads" "$out"

  # path classification
  tp() { # name want path
    local got=no; if is_test_path "$3"; then got=yes; fi
    t "path-$1" "$2" "$got"
  }
  tp tests-dir        yes crates/aprender-serve/src/gguf/tests/tests_15.rs
  tp basename-tests   yes crates/aprender-serve/src/gguf/inference/attention_gqa_tests.rs
  tp mid-name-tests   yes crates/aprender-serve/src/gguf/inference/forward/single_tests_q8k.rs
  tp tests-prefix     yes crates/aprender-serve/src/gguf/inference/cached/sync_tests_constructors.rs
  tp layers-tests-dir yes crates/aprender-serve/src/layers/tests/imp_086.rs
  tp top-tests-dir    yes tests/x.rs
  tp testing-dir      yes crates/aprender-serve/src/testing/mod.rs
  tp prod-cuda        no  crates/aprender-serve/src/gguf/cuda/mod.rs
  tp prod-parity-gate no  crates/aprender-serve/src/gguf/cuda/mod_parity_gate.rs
  tp prod-attention   no  crates/aprender-serve/src/gguf/inference/attention_gqa.rs

  # hit reporting carries path and line number
  out="$(scan_text a/b.rs $'fn x() {}\nlet d = hidden_dim / num_heads;')"
  t report-format "a/b.rs:2: let d = hidden_dim / num_heads;" "$out"

  # allowlist judgement: match on path plus normalized text, never a line number
  local H A
  H='a.rs:10: let d = hidden_dim / num_heads;'
  A="a.rs${T}let d = hidden_dim / num_heads;${T}legacy"
  jc() { # name want_new want_stale want_known want_legacy hits allow
    local got; judge "$6" "$7" >/dev/null
    got="${J_NEW}/${J_STALE}/${J_KNOWN}/${J_LEGACY}"
    t "judge-$1" "$2/$3/$4/$5" "$got"
  }
  jc listed-site-green          0 0 0 1 "$H" "$A"
  jc moved-line-number-green    0 0 0 1 'a.rs:99: let d = hidden_dim / num_heads;' "$A"
  jc reindented-green           0 0 0 1 'a.rs:10:        let   d = hidden_dim / num_heads;' "$A"
  jc new-site-red               1 0 0 1 "$H"$'\n''a.rs:20: let e = hidden_dim / n_heads;' "$A"
  jc new-file-red               1 1 0 0 'b.rs:10: let d = hidden_dim / num_heads;' "$A"
  jc removed-site-stale-red     0 1 0 0 '' "$A"
  jc stale-text-changed-red     1 1 0 0 'a.rs:10: let d = hidden_dim / num_heads + 1;' "$A"
  jc known-defect-green         0 0 1 0 "$H" "a.rs${T}let d = hidden_dim / num_heads;${T}known-defect #7285"
  jc second-site-needs-count    1 0 0 1 "$H"$'\n''a.rs:12: let d = hidden_dim / num_heads;' "$A"
  jc comment-and-blank-ignored  0 0 0 1 "$H" $'# note\n\n'"$A"
  local A3="${A}${T}3" H3
  H3="$H"$'\n''a.rs:12: let d = hidden_dim / num_heads;'$'\n''a.rs:14: let d = hidden_dim / num_heads;'
  jc count-3-all-used           0 0 0 3 "$H3" "$A3"
  jc count-3-fourth-is-new      1 0 0 3 "$H3"$'\n''a.rs:16: let d = hidden_dim / num_heads;' "$A3"
  jc count-3-one-used-two-stale 0 2 0 1 "$H" "$A3"
  jc count-3-none-used-three-stale 0 3 0 0 '' "$A3"
  local jout
  jout="$(judge '' "$A")"
  if [[ "$jout" == *"remove the entry"* ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL stale message: $jout"; fi
  jout="$(judge '' "$A3")"
  if [[ "$jout" == *"(3 unused)"* ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL stale count message: $jout"; fi
  jout="$(judge "$H" '')"
  if [[ "$jout" == NEW\ * ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL new message: $jout"; fi

  # allowlist validation: a missing or bogus status, a bad count, an extra column, a repeated pair
  ap() { t "allow-$1" "$2" "$(allow_problems "$3" | sed -E 's/^row [0-9]+: //' | head -1)"; }
  ap valid-legacy '' "a.rs${T}x / y${T}legacy"
  ap valid-known-defect '' "a.rs${T}x / y${T}known-defect #7285"
  ap valid-with-count '' "a.rs${T}x / y${T}legacy${T}4"
  ap valid-comments-only '' $'# c\n\n'
  ap missing-status "status '' is not 'legacy' or 'known-defect #NNNN'" "a.rs${T}x / y"
  ap empty-status "status '' is not 'legacy' or 'known-defect #NNNN'" "a.rs${T}x / y${T}"
  ap misspelled-status "status 'known_defect #7285' is not 'legacy' or 'known-defect #NNNN'" "a.rs${T}x / y${T}known_defect #7285"
  ap known-defect-without-issue "status 'known-defect' is not 'legacy' or 'known-defect #NNNN'" "a.rs${T}x / y${T}known-defect"
  ap legacy-with-suffix "status 'legacy x' is not 'legacy' or 'known-defect #NNNN'" "a.rs${T}x / y${T}legacy x"
  ap zero-count "count '0' is not a positive integer" "a.rs${T}x / y${T}legacy${T}0"
  ap word-count "count 'two' is not a positive integer" "a.rs${T}x / y${T}legacy${T}two"
  ap five-columns "more than four columns" "a.rs${T}x / y${T}legacy${T}1${T}junk"
  ap missing-text "missing text column" "a.rs"
  ap repeated-pair "repeated (path, text) pair, first on row 1; use the count column" "a.rs${T}x / y${T}legacy"$'\n'"a.rs${T}x  /  y${T}legacy"
  ap repeated-pair-mixed-status "repeated (path, text) pair, first on row 1; use the count column" "a.rs${T}x / y${T}legacy"$'\n'"a.rs${T}x / y${T}known-defect #1"
  ap same-text-other-path-ok '' "a.rs${T}x / y${T}legacy"$'\n'"b.rs${T}x / y${T}legacy"

  # the shipped allowlist is itself valid
  local shipped
  shipped="$(allow_problems "$(<"$(dirname "${BASH_SOURCE[0]}")/check_head_dim_single_source.allow")")"
  t shipped-allowlist-valid '' "$shipped"

  # main(): argument handling, input refusals and the exit code it forwards. A throwaway
  # git repo stands in for the tree; untracked files count, so no commit is needed.
  mc() { # name want args...
    local name="$1" want="$2" r=0
    shift 2
    ( main "$@" ) >/dev/null 2>&1 || r=$?
    t "main-$name" "$want" "$r"
  }
  local repo="${tmp}/repo" src="${tmp}/repo/crates/aprender-serve/src"
  mkdir -p "${src}/gguf" "${src}/tests"
  git -C "$repo" init -q
  printf 'fn ok() { let d = cfg.head_dim(); }\n' >"${src}/ok.rs"
  printf 'fn acc() { let d = hidden_dim / num_heads; }\n' >"${src}/gguf/config.rs"
  printf 'fn t() { let d = hidden_dim / num_heads; }\n' >"${src}/tests/t.rs"
  printf '# header\n' >"${tmp}/empty.allow"
  printf 'crates/aprender-serve/src/hit.rs\tfn bad() { let d = hidden_dim / num_heads; }\tlegacy\n' >"${tmp}/hit.allow"
  printf 'crates/aprender-serve/src/hit.rs\tlet d = hidden_dim / num_heads;\tbogus\n' >"${tmp}/bad.allow"
  mc clean-tree-green 0 --root "$repo" --allow "${tmp}/empty.allow"
  mc unknown-option 2 --bogus
  mc root-missing-value 2 --root
  mc allow-missing-value 2 --allow
  mc allow-missing-value-after-root 2 --root "$repo" --allow
  mc root-not-a-dir 3 --root "${tmp}/nope" --allow "${tmp}/empty.allow"
  mc allow-missing 3 --root "$repo" --allow "${tmp}/nope.allow"
  mc allow-is-directory 3 --root "$repo" --allow "$tmp"
  mc allow-bogus-status 3 --root "$repo" --allow "${tmp}/bad.allow"
  mc stale-entry-red 1 --root "$repo" --allow "${tmp}/hit.allow"
  printf 'fn bad() { let d = hidden_dim / num_heads; }\n' >"${src}/hit.rs"
  mc listed-hit-green 0 --root "$repo" --allow "${tmp}/hit.allow"
  mc unlisted-hit-red 1 --root "$repo" --allow "${tmp}/empty.allow"
  printf 'fn bad() {\n    let d = hidden_dim\n        / num_heads;\n}\n' >"${src}/hit.rs"
  mc split-hit-red 1 --root "$repo" --allow "${tmp}/empty.allow"
  rm -f "${src:?}/hit.rs"
  git -C "$repo" add -A
  rm -f "${src:?}/ok.rs"
  mc deleted-tracked-file 3 --root "$repo" --allow "${tmp}/empty.allow"
  git -C "$repo" rm -q --cached -r -f . >/dev/null
  rm -rf "${src:?}"
  mc no-files 3 --root "$repo" --allow "${tmp}/empty.allow"
  mkdir -p "$src"
  printf 'fn t() { let d = hidden_dim / num_heads; }\n' >"${src}/only_tests.rs"
  mc only-test-files-scanned-zero 3 --root "$repo" --allow "${tmp}/empty.allow"

  echo "self-test: ${pass} passed, ${fail} failed"
  [[ "$fail" -eq 0 ]]
}

main "$@"
