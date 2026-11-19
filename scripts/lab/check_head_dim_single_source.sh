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
# basename contains "test", the body of a `#[cfg(test)]` `mod NAME { .. }` block (end found by
# brace count on stripped text), a `#[cfg(test)]` `mod NAME;` or `use ..;`, a leading inner
# `#![cfg(test)]` (whole file), comment text, string literals, and gguf/config.rs (the accessor
# itself).
#
# KNOWN LIMITS (never guessed at): a `#[cfg(test)]` on any other form (fn, impl, const, static,
# field, match arm, statement, brace macro, `include!`), an inner `#![cfg(test)]` below the top
# of a file, an unbalanced cfg attribute, and an unterminated or unparseable literal or comment
# make the FILE unjudged: it is named and the run exits 3 (not_measured), never silently blanked.
# Macro-generated code is not seen at all.
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
#   allowlist (including an empty path column), root not a directory, a listed source file
#   unreadable or deleted, or any file unjudged (see KNOWN LIMITS; a red elsewhere still exits 1).
# Not wired into any workflow, PR path or guard tree.

set -euo pipefail

SCAN_DIR="crates/aprender-serve/src"
ACCESSOR_FILE="crates/aprender-serve/src/gguf/config.rs"
declare -A UNJ_PATHS=()   # files the scan could not judge; their allowlist rows are not evaluated

# The matcher. Reads one file's text on stdin; FNAME names it. Prints "path:line: text".
# Identifier boundaries are explicit (awk ERE has no \b), and the denominator is anchored to
# the start of an identifier path so `num_kv_heads` can never match `heads`.
AWK_PROG='
# blanks every char of s except newlines (keeps offsets and line numbers)
function blank(s) { gsub(/[^\n]/, " ", s); return s }
function isid(c) { return c ~ /[A-Za-z0-9_]/ }
# strip_all: T -> same-length text with comments, string and char literals blanked.
# State crosses lines: nested block comments, multi-line and raw strings (r, br, cr). A literal
# or comment it cannot close or parse sets UNJ (the file is unjudged, never guessed at).
function strip_all(T,   n, i, c, c2, out, d, j, h, k, raw, p, closed) {
  n = length(T); i = 1; out = ""
  while (i <= n) {
    c = substr(T, i, 1); c2 = substr(T, i, 2)
    if (c2 == "//") {
      j = index(substr(T, i), "\n"); if (j == 0) j = n - i + 2
      out = out blank(substr(T, i, j - 1)); i += j - 1; continue
    }
    if (c2 == "/*") {
      d = 1; j = i + 2
      while (j <= n && d > 0) {
        if (substr(T, j, 2) == "/*") { d++; j += 2 } else if (substr(T, j, 2) == "*/") { d--; j += 2 } else j++
      }
      if (d > 0 && UNJ == "") UNJ = "unterminated block comment"
      out = out blank(substr(T, i, j - i)); i = j; continue
    }
    if ((c == "r" || ((c == "b" || c == "c") && substr(T, i + 1, 1) == "r")) && (i == 1 || !isid(substr(T, i - 1, 1)))) {
      p = i + (c == "r" ? 1 : 2); h = 0
      while (substr(T, p + h, 1) == "#") h++
      if (substr(T, p + h, 1) == "\"") {
        raw = "\""; for (k = 0; k < h; k++) raw = raw "#"
        j = index(substr(T, p + h + 1), raw)
        if (j == 0 && UNJ == "") UNJ = "unterminated raw string"
        j = (j == 0) ? n + 1 : p + h + j + length(raw)
        out = out blank(substr(T, i, j - i)); i = j; continue
      }
    }
    if (c == "\"") {
      j = i + 1; closed = 0
      while (j <= n) {
        k = substr(T, j, 1)
        if (k == "\\") { j += 2; continue }
        if (k == "\"") { j++; closed = 1; break }
        j++
      }
      if (!closed && UNJ == "") UNJ = "unterminated string literal"
      if (j > n + 1) j = n + 1
      out = out blank(substr(T, i, j - i)); i = j; continue
    }
    if (c == SQ) {
      k = substr(T, i + 1, 1)
      if (k == "\\") {
        j = index(substr(T, i + 3), SQ)
        if (j > 0 && j <= 10) { out = out blank(substr(T, i, 3 + j)); i += 3 + j; continue }
        if (UNJ == "") UNJ = "unparseable char literal"
      } else if (substr(T, i + 2, 1) == SQ) { out = out "   "; i += 3; continue }
      else if (k != "" && index(PR, k) == 0 && k != "\n") {   # multibyte char literal: the closing quote is within 4 bytes
        j = index(substr(T, i + 1), SQ)
        if (j > 0 && j <= 5) { out = out blank(substr(T, i, 1 + j)); i += 1 + j; continue }
        if (UNJ == "") UNJ = "unparseable char literal"
      }
      else if (index(PR, k) > 0 && k !~ /[A-Za-z_]/ && UNJ == "") UNJ = "unparseable quote"
    }
    out = out c; i++
  }
  return out
}
# close_of S POS: index of the bracket closing the one opened at POS, or 0
function close_of(S, pos,   n, i, d, o, c, cl) {
  n = length(S); o = substr(S, pos, 1)
  cl = (o == "(") ? ")" : ((o == "[") ? "]" : "}")
  d = 0
  for (i = pos; i <= n; i++) {
    c = substr(S, i, 1)
    if (c == o) d++; else if (c == cl) { d--; if (d == 0) return i }
  }
  return 0
}
# pred_is_test: a cfg predicate that holds only under test: bare `test`, or all(.., test, ..)
# with the test arm at any nesting depth of all(..)
function pred_is_test(p,   n, i, d, c, arg, inner) {
  gsub(/^[ \n\t]+|[ \n\t]+$/, "", p)
  if (p == "test") return 1
  if (p !~ /^all[ \n\t]*\(/) return 0
  inner = substr(p, index(p, "(") + 1); sub(/\)[ \n\t]*$/, "", inner)
  n = length(inner); d = 0; arg = ""
  for (i = 1; i <= n + 1; i++) {
    c = (i <= n) ? substr(inner, i, 1) : ","
    if (c == "(" || c == "[") d++
    if (c == ")" || c == "]") d--
    if (c == "," && d == 0) { if (pred_is_test(arg)) return 1; arg = "" } else arg = arg c
  }
  return 0
}
function nl_count(s,   t) { t = s; return gsub(/\n/, "", t) }
# blank_cfg_test: blank a `#[cfg(test)]` (test-only predicate) `mod NAME { .. }` block, a
# `mod NAME;` declaration or a `use ..;` item, attribute included, and nothing else. The end of
# a mod block is found by brace count on the stripped text. Any other test-gated form (fn, impl,
# const, struct field, match arm, statement, macro, ...) or an attribute it cannot parse sets
# UNJ: the file is unjudged, never silently blanked. A leading inner `#![cfg(test)]` (nothing
# but inner attributes before it) gates the whole file; one anywhere else sets UNJ.
function blank_cfg_test(S,   pos, rest, o, e, pred, q, a, b, st, en, out, last, inner, k, w, lead) {
  out = ""; last = 1; pos = 1
  while (1) {
    rest = substr(S, pos)
    if (!match(rest, /#!?\[[ \n\t]*cfg[ \n\t]*\(/)) break
    a = pos + RSTART - 1
    inner = (substr(S, a + 1, 1) == "!")
    o = a + RLENGTH - 1
    e = close_of(S, o)
    if (e == 0) { UNJ = "line " (nl_count(substr(S, 1, a)) + 1) ": unbalanced cfg attribute"; return S }
    q = e + 1
    while (substr(S, q, 1) ~ /[ \n\t]/) q++
    if (substr(S, q, 1) != "]") { UNJ = "line " (nl_count(substr(S, 1, a)) + 1) ": malformed cfg attribute"; return S }
    pred = substr(S, o + 1, e - o - 1)
    if (!pred_is_test(pred)) { pos = q + 1; continue }
    if (inner) {
      lead = substr(S, 1, a - 1)
      while (match(lead, /^[ \n\t]*#!\[[^]]*\]/)) lead = substr(lead, RLENGTH + 1)
      if (lead ~ /^[ \n\t]*$/) return blank(S)
      UNJ = "line " (nl_count(substr(S, 1, a)) + 1) ": inner #![cfg(test)] below the top of the file"
      return S
    }
    st = q + 1
    while (1) {
      while (substr(S, st, 1) ~ /[ \n\t]/) st++
      if (substr(S, st, 2) == "#[") {
        b = close_of(S, st + 1)
        if (b == 0) { UNJ = "line " (nl_count(substr(S, 1, st)) + 1) ": unbalanced attribute"; return S }
        st = b + 1
      } else break
    }
    rest = substr(S, st, 400)
    if (match(rest, /^pub[ \n\t]*\([^)]*\)[ \n\t]*/) || match(rest, /^pub[ \n\t]+/)) { st += RLENGTH; rest = substr(S, st, 400) }
    if (match(rest, /^mod[ \n\t]+[A-Za-z_][A-Za-z0-9_]*[ \n\t]*[{;]/)) {
      en = st + RLENGTH - 1
      if (substr(S, en, 1) == "{") {
        en = close_of(S, en)
        if (en == 0) { UNJ = "line " (nl_count(substr(S, 1, a)) + 1) ": unbalanced cfg(test) mod block"; return S }
      }
    } else if (match(rest, /^use[ \n\t]/)) {
      k = index(substr(S, st), ";")
      if (k == 0) { UNJ = "line " (nl_count(substr(S, 1, a)) + 1) ": cfg(test) use without a semicolon"; return S }
      en = st + k - 1
    } else {
      w = substr(rest, 1, 28); gsub(/[ \n\t]+/, " ", w)
      UNJ = "line " (nl_count(substr(S, 1, a)) + 1) ": cfg(test) on a form that is not a mod block (`" w "`)"
      return S
    }
    out = out substr(S, last, a - last) blank(substr(S, a, en - a + 1))
    last = en + 1; pos = en + 1
  }
  return out substr(S, last)
}
function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s); return s }
BEGIN {
  SQ = sprintf("%c", 39)
  for (ci = 32; ci < 127; ci++) PR = PR sprintf("%c", ci)   # printable ASCII; a quote followed by anything else is a multibyte char literal
  num = "(hidden_dim|hidden_size|hidden|embed_dim|embedding_dim|d_model|embedding_length)"
  den = "(num_heads|n_heads|n_head|num_attention_heads|head_count|heads)"
  recv = "([A-Za-z_][A-Za-z0-9_]*(\\(\\))?[.])*"
  pre = "(^|[^A-Za-z0-9_])" recv
  post = "([^A-Za-z0-9_]|$)"
  cast = "([ ]+as[ ]+[A-Za-z0-9_]+)?"
  numtail = "[ ]*(\\(\\))?[ ]*\\)?" cast "[ ]*\\)?[ ]*"
  re_div = pre num numtail "/[ ]*\\(?[ ]*" recv den "(\\(\\))?" post
  re_chk = pre num numtail "[.]checked_div[(][ ]*\\(?[ ]*" recv den "(\\(\\))?" post
}
{ T = T $0 "\n"; orig[NR] = $0 }
END {
  S = blank_cfg_test(strip_all(T))
  if (UNJ != "") { printf "@@UNJUDGED@@ %s: %s\n", FNAME, UNJ; exit }
  gsub(/\t/, " ", S)
  split(S, code, "\n")
  for (i = 1; i <= NR; i++) {
    if (code[i] ~ /^ *$/) continue
    L = length(code[i]); w = code[i]; used = 0; j = i; delete jl
    while (used < 2 && ++j <= NR) { if (code[j] ~ /^ *$/) continue; w = w " " code[j]; used++; jl[used] = j }
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
    text = orig[i]; e = L
    for (k = 1; k <= used && end > e + 1; k++) {   # e+1 is the joining space the boundary may consume
      text = text " " trim(orig[jl[k]]); e += 1 + length(code[jl[k]])
    }
    printf "%s:%d: %s\n", FNAME, i, text
  }
}
'

# is_test_path PATH: 0 when the path is test code.
is_test_path() {
  local p="$1" base="${1##*/}"
  base="${base%.rs}"
  [[ "$p" == */tests/* || "$p" == tests/* || "$p" == */testing/* ]] && return 0   # testing/ is `#[cfg(test)] pub mod testing` in lib.rs
  # a test file has the token test(s) at an end of its name, or tests anywhere (the include!d
  # chunks are `<name>_tests_<topic>.rs`), or is a proptests file; "test" inside a word or in the
  # middle (latest_kv, load_testing, convoy_test_result) is production code
  [[ "$base" == test || "$base" == test_* || "$base" == *_test || "$base" == tests || "$base" == tests_* || "$base" == *_tests || "$base" == *_tests_* || "$base" == proptests* || "$base" == *_proptests* ]]
}

# scan_text FNAME TEXT: prints hits for one file's text.
scan_text() {
  LC_ALL=C awk -v FNAME="$1" "$AWK_PROG" <<<"$2"
}

# --- allowlist ------------------------------------------------------------------------

# norm: collapse runs of whitespace and trim, so an entry survives re-indentation and line moves.
norm() { sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//' <<<"$1"; }

# split_cols LINE: splits on every TAB without collapsing empty columns (bash `read` with a
# whitespace IFS would shift an empty column left). Sets the array COLS.
split_cols() {
  local s="$1"
  COLS=()
  while [[ "$s" == *$'\t'* ]]; do COLS+=("${s%%$'\t'*}"); s="${s#*$'\t'}"; done
  COLS+=("$s")
}

# allow_problems ALLOW: prints one line per malformed row (bad status, bad count, extra column,
# missing column, empty column) and per repeated (path, text) pair.
allow_problems() {
  local line n=0 key
  local -A seen=()
  while IFS= read -r line || [[ -n "$line" ]]; do
    n=$((n + 1))
    [[ -z "$line" || "$line" == \#* ]] && continue
    split_cols "$line"
    if [[ -z "$(norm "${COLS[0]}")" ]]; then echo "row ${n}: empty path column"; continue; fi
    if [[ -z "$(norm "${COLS[1]:-}")" ]]; then echo "row ${n}: missing text column"; continue; fi
    if ! [[ "${COLS[2]:-}" =~ ^(legacy|known-defect\ \#[0-9]+)$ ]]; then
      echo "row ${n}: status '${COLS[2]:-}' is not 'legacy' or 'known-defect #NNNN'"
    fi
    if [[ ${#COLS[@]} -ge 4 ]] && ! [[ "${COLS[3]}" =~ ^[1-9][0-9]*$ ]]; then
      echo "row ${n}: count '${COLS[3]}' is not a positive integer"
    fi
    if [[ ${#COLS[@]} -gt 4 ]]; then echo "row ${n}: more than four columns"; fi
    key="${COLS[0]}"$'\t'"$(norm "${COLS[1]}")"
    if [[ -n "${seen[$key]:-}" ]]; then echo "row ${n}: repeated (path, text) pair, first on row ${seen[$key]}; use the count column"; fi
    seen[$key]="${seen[$key]:-$n}"
  done <<<"$1"
}

# judge HITS ALLOW: HITS is "path:line: text" lines, ALLOW the allowlist text (already
# validated). Prints one row per hit and per stale entry; sets J_NEW, J_STALE, J_KNOWN,
# J_LEGACY (globals).
judge() {
  local hits="$1" allow="$2" line path text key left
  local -A avail=() status=()
  J_NEW=0 J_STALE=0 J_KNOWN=0 J_LEGACY=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    split_cols "$line"
    key="${COLS[0]}"$'\t'"$(norm "${COLS[1]}")"
    avail[$key]=$(( ${avail[$key]:-0} + ${COLS[3]:-1} ))
    status[$key]="${COLS[2]}"
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
    [[ -n "${UNJ_PATHS[${key%%$'\t'*}]:-}" ]] && continue
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
  if [[ "${1:-}" == "--self-test" ]]; then [[ $# -eq 1 ]] || usage; self_test; exit $?; fi
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

  local files scanned=0 f text out all_hits="" unjudged=""
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
    if [[ "$out" == "@@UNJUDGED@@ "* ]]; then unjudged+="UNJUDGED ${out#@@UNJUDGED@@ }"$'\n'; UNJ_PATHS[$f]=1; continue; fi
    [[ -z "$out" ]] && continue
    all_hits+="${out}"$'\n'
  done <<<"$files"

  if [[ "$scanned" -eq 0 ]]; then echo "not_measured: scanned 0 non-test .rs files"; exit 3; fi
  [[ -n "$unjudged" ]] && printf '%s' "$unjudged"
  judge "$all_hits" "$allow_text"
  echo "scanned ${scanned} files, ${#UNJ_PATHS[@]} unjudged: ${J_NEW} new, ${J_STALE} stale, ${J_KNOWN} known-defect, ${J_LEGACY} legacy"
  if [[ "$J_NEW" -gt 0 || "$J_STALE" -gt 0 ]]; then exit 1; fi
  if [[ -n "$unjudged" ]]; then echo "not_measured: ${#UNJ_PATHS[@]} file(s) could not be judged (listed above); a clean result on the rest is not a pass"; exit 3; fi
  exit 0
}

# --- self-test --------------------------------------------------------------------

self_test() {
  local pass=0 fail=0 tmp T=$'\t'
  tmp="$(mktemp -d)"
  trap '[[ -n "${tmp:-}" ]] && rm -rf "${tmp:?}"' RETURN
  hit() { # name text   (must match)
    local out; out="$(scan_text f.rs "$2")"
    if [[ -n "$out" && "$out" != "@@UNJUDGED@@ "* ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL must-match $1: $2"; fi
  }
  miss() { # name text   (must NOT match)
    local out; out="$(scan_text f.rs "$2")"
    if [[ -z "$out" ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL must-not-match $1: $2 -> $out"; fi
  }
  unj() { # name text   (must be reported unjudged, never a hit and never silently scanned)
    local out; out="$(scan_text f.rs "$2")"
    if [[ "$out" == "@@UNJUDGED@@ f.rs: "* ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL must-be-unjudged $1: $2 -> $out"; fi
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
  unj after-cfg-test-fn            $'#[cfg(test)]\nfn helper() {\n    let x = 1;\n}\nfn f() { let d = hidden_dim / num_heads; }'
  hit after-cfg-test-with-allow    $'#[cfg(test)]\n#[allow(dead_code)]\nmod tests {\n}\nfn f() { let d = hidden_dim / num_heads; }'
  hit cfg-not-test-is-production   $'#[cfg(not(test))]\nfn f() { let d = hidden_dim / num_heads; }'
  hit slashes-in-string-before     'let u = "http://x"; let d = hidden_dim / num_heads;'
  hit char-literal-before          $'let c = \'/\'; let d = hidden_dim / num_heads;'
  hit before-cfg-test              $'let d = hidden_dim / num_heads;\n#[cfg(test)]\nmod tests {}'

  # round 2: cfg(test) blanking is item-exact
  hit cfg-test-stacked-attr-same-line-use $'#[cfg(test)] #[allow(dead_code)] use x::y;\nfn f() { let d = hidden_dim / num_heads; }'
  hit cfg-test-stacked-attr-same-line-mod $'#[cfg(test)] #[allow(dead_code)] mod t { fn g() {} }\nfn f() { let d = hidden_dim / num_heads; }'
  miss cfg-test-stacked-attr-body  $'#[cfg(test)] #[allow(dead_code)] mod t { fn g() { let d = hidden_dim / num_heads; } }'
  unj cfg-test-struct-field-comma  $'struct S {\n    #[cfg(test)]\n    a: u8,\n    b: usize,\n}\nfn f() { let d = hidden_dim / num_heads; }'
  unj cfg-test-enum-variant-comma  $'enum E {\n    #[cfg(test)]\n    A,\n    B,\n}\nfn f() { let d = hidden_dim / num_heads; }'
  unj cfg-test-match-arm           $'fn f(x: u8) -> usize {\n    match x {\n        #[cfg(test)]\n        0 => 1,\n        _ => hidden_dim / num_heads,\n    }\n}'
  unj cfg-test-match-arm-body     $'fn f(x: u8) -> usize {\n    match x {\n        #[cfg(test)]\n        0 => hidden_dim / num_heads,\n        _ => 1,\n    }\n}'
  unj cfg-test-array-semicolon-sig $'#[cfg(test)]\nfn t(a: [usize; 2]) {\n}\nfn f() { let d = hidden_dim / num_heads; }'
  unj cfg-test-array-semicolon-body $'#[cfg(test)]\nfn t(a: [usize; 2]) {\n    let d = hidden_dim / num_heads;\n}'
  unj cfg-test-const-then-code     $'#[cfg(test)]\nconst X: [u8; 2] = [1, 2];\nfn f() { let d = hidden_dim / num_heads; }'
  unj cfg-test-item-then-prod-same-line $'#[cfg(test)] fn t() {} fn f() { let d = hidden_dim / num_heads; }'
  unj cfg-test-statement-then-prod $'fn f() {\n    #[cfg(test)]\n    let a = 1;\n    let d = hidden_dim / num_heads;\n}'
  miss cfg-test-nested-predicate   $'#[cfg(all(test, not(target_arch = "wasm32")))]\nmod t { fn f() { let d = hidden_dim / num_heads; } }'
  miss cfg-test-spaced-attribute   $'#[ cfg ( test ) ]\nmod t { fn f() { let d = hidden_dim / num_heads; } }'
  miss cfg-test-inner-attribute    $'#![cfg(test)]\nfn f() { let d = hidden_dim / num_heads; }'
  unj cfg-test-pub-crate-fn       $'#[cfg(test)]\npub(crate) fn t() { let d = hidden_dim / num_heads; }'
  unj cfg-test-impl-block         $'#[cfg(test)]\nimpl X {\n    fn t(&self) { let d = hidden_dim / num_heads; }\n}'
  unj cfg-test-generic-fn-then-prod $'#[cfg(test)]\nfn t<A, B>(a: A, b: B) {}\nfn f() { let d = hidden_dim / num_heads; }'

  # round 2: stripping carries state across lines
  miss raw-string-braces           $'let s = r#"{"msg": "{"}"#;\nlet t = "hidden_dim / num_heads";'
  hit  raw-string-then-code        $'let s = r#"a " b"#; let d = hidden_dim / num_heads;'
  miss raw-string-plain            'let s = r"hidden_dim / num_heads";'
  miss raw-string-backslash-end    $'let s = r"foo\\"; // hidden_dim / num_heads'
  miss raw-string-multiline        $'let s = r#"\nhidden_dim / num_heads\n"#;'
  miss string-multiline            $'let s = "line one\nhidden_dim / num_heads\nend";'
  miss string-escaped-quote        'let s = "a \" hidden_dim / num_heads \" b";'
  hit  string-escaped-backslash    'let s = "a \\"; let d = hidden_dim / num_heads;'
  hit  string-after-escape-quote   $'let s = "q\\"";\nlet d = hidden_dim / num_heads;'
  miss block-comment-nested        $'/* a /* b */ hidden_dim / num_heads */\nlet d = 1;'
  hit  block-comment-nested-ends   $'/* a /* b */ c */ let d = hidden_dim / num_heads;'
  hit  byte-string-then-code       $'let s = b"x"; let d = hidden_dim / num_heads;'
  hit  lifetime-not-char           $'fn f<\'a>(x: &\'a str) { let d = hidden_dim / num_heads; }'
  miss char-quote-literal          $'let c = \'"\'; // hidden_dim / num_heads'

  # round 3: literals. Live code follows every literal, so a stripper that swallows the rest
  # of the line (a quote it did not close) turns the row red.
  hit char-quote-then-code         $'let c = \'"\'; let d = hidden_dim / num_heads;'
  hit char-escaped-quote-then-code $'let c = \'\\\'\'; let d = hidden_dim / num_heads;'
  hit char-escaped-backslash       $'let c = \'\\\\\'; let d = hidden_dim / num_heads;'
  hit char-escape-width-exact      $'let c = \'\\u{1}\'hidden_dim / num_heads;'
  hit char-escape-width-exact-simple $'let c = \'\\n\'hidden_dim / num_heads;'
  unj  ident-ending-in-r-is-not-a-raw-prefix $'let a = xr"\\"; let d = hidden_dim / num_heads;'
  hit char-unicode-escape          $'let c = \'\\u{FFFD}\'; let d = hidden_dim / num_heads;'
  hit char-multibyte               $'let c = \'\xe2\x96\x88\'; let d = hidden_dim / num_heads;'
  hit char-multibyte-4-byte        $'let c = \'\xf0\x9f\x8e\x89\'; let d = hidden_dim / num_heads;'
  hit lifetime-then-code           $'fn f<\'a>(x: &\'a str) { let d = hidden_dim / num_heads; }'
  hit byte-char-quote              $'let c = b\'"\'; let d = hidden_dim / num_heads;'
  hit raw-string-0-hash-backslash  $'let s = r"\\"; let d = hidden_dim / num_heads;'
  hit raw-byte-string-0-hash       $'let s = br"\\"; let d = hidden_dim / num_heads;'
  hit raw-c-string-0-hash          $'let s = cr"\\"; let d = hidden_dim / num_heads;'
  hit raw-string-1-hash-inner-quote $'let s = r#"a"b"#; let d = hidden_dim / num_heads;'
  hit raw-byte-string-1-hash       $'let s = br#"a"b"#; let d = hidden_dim / num_heads;'
  hit raw-c-string-1-hash          $'let s = cr#"a"b"#; let d = hidden_dim / num_heads;'
  hit raw-string-2-hash            $'let s = r##"a"#b"##; let d = hidden_dim / num_heads;'
  miss raw-c-string-body           $'let s = cr#"hidden_dim / num_heads"#;'
  miss raw-byte-string-body        $'let s = br#"hidden_dim / num_heads"#;'
  hit raw-ident-not-a-string       $'let r#type = 1; let d = hidden_dim / num_heads;'
  hit ident-ending-in-r-then-string $'let bar = foo"x"; let d = hidden_dim / num_heads;'
  unj  unterminated-string         $'let s = "abc;\nlet d = hidden_dim / num_heads;'
  unj  unterminated-raw-string     $'let s = r#"abc"; let d = hidden_dim / num_heads;'
  unj  unterminated-block-comment  $'/* never closed\nlet d = hidden_dim / num_heads;'
  unj  unterminated-escaped-char   $'let c = \'\\u{1F600; let d = hidden_dim / num_heads;'
  unj  quote-before-punctuation    $'let c = \'(; let d = hidden_dim / num_heads;'

  # round 3: cfg(test). Only a `mod NAME { .. }` block (end by brace count), a `mod NAME;` and a
  # `use ..;` are blanked. Every other test-gated form is unjudged, never silently blanked.
  miss cfg-test-inner-attr-whole-file  $'#![cfg(test)]\nfn a() {}\nfn f() { let d = hidden_dim / num_heads; }'
  miss cfg-test-inner-attr-after-other $'#![allow(dead_code)]\n#![cfg(test)]\nfn a() {}\nfn f() { let d = hidden_dim / num_heads; }'
  unj  cfg-test-inner-attr-in-submodule $'mod m {\n    #![cfg(test)]\n    fn t() {}\n}\nfn f() { let d = hidden_dim / num_heads; }'
  unj  cfg-test-inner-attr-after-code  $'fn a() {}\n#![cfg(test)]\nfn f() { let d = hidden_dim / num_heads; }'
  miss cfg-test-nested-all             $'#[cfg(all(unix, all(test, feature = "a")))]\nmod t { fn f() { let d = hidden_dim / num_heads; } }'
  miss cfg-test-nested-all-deep        $'#[cfg(all(all(all(test))))]\nmod t { fn f() { let d = hidden_dim / num_heads; } }'
  hit  cfg-test-all-without-test       $'#[cfg(all(unix, all(feature = "a", not(test))))]\nmod t { fn f() { let d = hidden_dim / num_heads; } }'
  unj  cfg-test-generic-comma-const    $'#[cfg(test)]\nconst X: HashMap<A, B> = M;\nfn f() { let d = hidden_dim / num_heads; }'
  unj  cfg-test-brace-macro            $'#[cfg(test)]\nthread_local! { static X: u8 = 0; }\nfn f() { let d = hidden_dim / num_heads; }'
  unj  cfg-test-const-fn               $'#[cfg(test)]\nconst fn t() -> usize { 1 }\nfn f() { let d = hidden_dim / num_heads; }'
  unj  cfg-test-pub-const-fn           $'#[cfg(test)]\npub const fn t() -> usize { 1 }\nfn f() { let d = hidden_dim / num_heads; }'
  unj  cfg-test-if-statement           $'fn f() {\n    #[cfg(test)]\n    if x { y(); }\n    let d = hidden_dim / num_heads;\n}'
  unj  cfg-test-match-arm-block        $'fn f(x: u8) -> usize {\n    match x {\n        #[cfg(test)]\n        0 => { 1 }\n        _ => hidden_dim / num_heads,\n    }\n}'
  unj  cfg-test-impl-no-body-split     $'#[cfg(test)]\nimpl X for Y {}\nfn f() { let d = hidden_dim / num_heads; }'
  unj  cfg-test-unbalanced-attr        $'#[cfg(test\nmod t {}\nfn f() { let d = hidden_dim / num_heads; }'
  unj  cfg-test-mod-unbalanced-block   $'#[cfg(test)]\nmod t {\n    fn g() {\nfn f() { let d = hidden_dim / num_heads; }'
  miss cfg-test-pub-crate-mod          $'#[cfg(test)]\npub(crate) mod t { fn g() { let d = hidden_dim / num_heads; } }'
  miss cfg-test-pub-mod                $'#[cfg(test)]\npub mod t { fn g() { let d = hidden_dim / num_heads; } }'
  hit  cfg-test-pub-mod-declaration    $'#[cfg(test)]\npub mod t;\nfn f() { let d = hidden_dim / num_heads; }'
  hit  cfg-test-use-group              $'#[cfg(test)]\nuse super::{a, b};\nfn f() { let d = hidden_dim / num_heads; }'
  hit  cfg-test-mod-ends-at-brace-count $'#[cfg(test)]\nmod t { fn a() { if x { y } } }\nfn f() { let d = hidden_dim / num_heads; }'
  miss cfg-test-mod-brace-in-string    $'#[cfg(test)]\nmod t { fn a() { let s = "}"; let c = \'}\'; let d = hidden_dim / num_heads; } }'
  hit  cfg-test-two-mods               $'#[cfg(test)]\nmod a { fn g() {} }\nfn f() { let d = hidden_dim / num_heads; }\n#[cfg(test)]\nmod b { fn g() {} }'

  # round 2: matcher shapes
  hit checked-div-trailing-comma   $'foo(hidden_dim.checked_div(num_heads), 1)'
  hit checked-div-arg-comma        $'let d = hidden_dim.checked_div(num_heads,);'
  hit checked-div-max              'let d = hidden_dim.checked_div(num_heads.max(1));'
  hit checked-div-unwrap-or        'let d = hidden_dim.checked_div(num_heads).unwrap_or(1);'
  miss checked-div-heads-suffix    'let d = hidden_dim.checked_div(num_heads_total);'
  hit paren-numerator-then-cast    'let d = (hidden_dim) as usize / num_heads;'
  hit paren-numerator-cast-inside  'let d = (self.hidden_dim) as usize / self.num_heads as usize;'
  miss cast-then-other             'let d = (hidden_dim) as usize / num_kv_heads;'

  # round 2: the report glues only the lines the match spans
  out="$(scan_text f.rs $'let d = hidden_dim\n    / num_heads;\nlet e = head_dim;')"
  t report-two-line-span-not-three "f.rs:1: let d = hidden_dim / num_heads;" "$out"
  out="$(scan_text f.rs $'let d = hidden_dim\n    / num_heads;\nlet e = head_dim;\nlet g = 1;')"
  t report-two-line-span-fourth-line "f.rs:1: let d = hidden_dim / num_heads;" "$out"
  out="$(scan_text f.rs $'let a = (\n    hidden_dim\n    / num_heads\n);\nlet z = 1;')"
  t report-three-line-span "f.rs:2:     hidden_dim / num_heads" "$out"
  out="$(scan_text f.rs $'let d = hidden_dim /\n    num_heads;')"
  t report-trailing-slash-span "f.rs:1: let d = hidden_dim / num_heads;" "$out"
  out="$(scan_text f.rs $'let d = hidden_dim / num_heads;\nlet e = 2;')"
  t report-one-line-no-glue "f.rs:1: let d = hidden_dim / num_heads;" "$out"

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
  unj inside-cfg-test-fn          $'#[cfg(test)]\nfn helper() {\n    let d = hidden_dim / num_heads;\n}'
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
  tp latest-kv-prod   no  crates/aprender-serve/src/gguf/inference/latest_kv.rs
  tp load-testing-prod no crates/aprender-serve/src/bench/load_testing.rs
  tp mid-test-prod    no  crates/aprender-serve/src/bench/convoy_test_result.rs
  tp test-prefix      yes crates/aprender-serve/src/quantize/test_helpers.rs
  tp test-suffix      yes crates/aprender-serve/src/quantize/parity_test.rs
  tp proptests-file   yes crates/aprender-serve/src/cuda/executor/proptests_kernels.rs
  tp bare-tests-file  yes crates/aprender-serve/src/tests.rs
  tp tests-named-dir-prod no crates/aprender-serve/src/contests/mod.rs

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
  ap empty-status-middle "status '' is not 'legacy' or 'known-defect #NNNN'" "a.rs${T}x / y${T}${T}legacy"
  ap empty-count "count '' is not a positive integer" "a.rs${T}x / y${T}legacy${T}"
  ap empty-count-then-extra "count '' is not a positive integer" "a.rs${T}x / y${T}legacy${T}${T}2"
  ap empty-text-middle "missing text column" "a.rs${T}${T}legacy"
  ap empty-path "empty path column" "${T}x / y${T}legacy"
  ap blank-path "empty path column" "   ${T}x / y${T}legacy"
  ap blank-text "missing text column" "a.rs${T}   ${T}legacy"
  ap trailing-tab-after-status "count '' is not a positive integer" "a.rs${T}x / y${T}legacy${T}"
  jc count-2-unused-stale 0 2 0 0 '' "a.rs${T}x / y${T}legacy${T}2"

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
  local repo="${tmp}/repo" src="${tmp}/repo/crates/aprender-serve/src" unjout
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
  mc self-test-extra-arg 2 --self-test --bogus
  # the default allowlist (no --allow) is the shipped one: a clean temp repo makes every shipped
  # row stale (red); a corrupted default path would be not_measured
  mc default-allowlist-used-stale 1 --root "$repo"
  if [[ "$(id -u)" -ne 0 ]]; then
    printf '# x\n' >"${tmp}/noread.allow"; chmod 000 "${tmp}/noread.allow"
    mc allow-unreadable 3 --root "$repo" --allow "${tmp}/noread.allow"
  fi
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

  # an allowlist row with an empty path column is malformed (not_measured), not a stale row (red)
  printf '\tlet d = hidden_dim / num_heads;\tlegacy\n' >"${tmp}/emptypath.allow"
  printf 'crates/aprender-serve/src/\tlet d = hidden_dim / num_heads;\tlegacy\n' >"${tmp}/dirpath.allow"
  mc allow-empty-path 3 --root "$repo" --allow "${tmp}/emptypath.allow"
  mkdir -p "$src"
  printf 'fn ok() { let d = cfg.head_dim(); }\n' >"${src}/ok.rs"
  mc allow-empty-path-clean-tree 3 --root "$repo" --allow "${tmp}/emptypath.allow"
  # a file with a form the script does not judge: not_measured (exit 3, named), never green
  printf '#[cfg(test)]\nfn t() {}\nfn p() { let d = hidden_dim / num_heads; }\n' >"${src}/unj.rs"
  unjout="$( (main --root "$repo" --allow "${tmp}/empty.allow") 2>&1 )" || true
  t main-unjudged-names-the-file yes "$(if [[ "$unjout" == *"UNJUDGED crates/aprender-serve/src/unj.rs: line 1: cfg(test) on a form that is not a mod block"* ]]; then echo yes; else echo no; fi)"
  mc unjudged-file-not-measured 3 --root "$repo" --allow "${tmp}/empty.allow"
  # allowlist rows for an unjudged file are not evaluated, so they are not reported stale
  printf 'crates/aprender-serve/src/unj.rs\tlet d = hidden_dim / num_heads;\tlegacy\n' >"${tmp}/unj.allow"
  mc unjudged-file-allow-row-not-stale 3 --root "$repo" --allow "${tmp}/unj.allow"
  # a red in a judged file still wins over an unjudged one
  printf 'fn bad() { let d = hidden_dim / num_heads; }\n' >"${src}/hit.rs"
  mc red-beats-unjudged 1 --root "$repo" --allow "${tmp}/empty.allow"
  rm -f "${src:?}/hit.rs" "${src:?}/unj.rs"
  mc judged-again-green 0 --root "$repo" --allow "${tmp}/empty.allow"
  rm -f "${src:?}/ok.rs"

  echo "self-test: ${pass} passed, ${fail} failed"
  [[ "$fail" -eq 0 ]]
}

main "$@"
