#!/usr/bin/env bash
# surface_audit_cites_test.sh — case table + mutants for
# scripts/check_surface_audit_cites.sh (#7370).
#
# Each row mutates ONE cite of a scratch ledger committed on `main` and states
# the verdict the checker must give. Each mutant then breaks ONE rule inside a
# copy of the checker and the row that rule guards must flip, so a green table
# cannot come from rules that never fire.
#
# Usage: bash scripts/tests/surface_audit_cites_test.sh   (0 all pass, 1 a row failed)
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECK="$SRC/scripts/check_surface_audit_cites.sh"
TD="$(mktemp -d)"
trap 'rm -rf "${TD:?}"' EXIT
export GIT_CEILING_DIRECTORIES="${TD%/*}"

mkdir -p "$TD/scripts" "$TD/docs/audits" "$TD/crates/demo/src"
HDR='binary,feature,quality_1_10,verified_hardware,top_competitor,in_dogfood_skill,cluster_id,cluster_label,evidence_path,confidence'
{
  printf 'pub fn pad() {}\n'
  printf 'Alpha {\n'
  printf 'Beta {\n'
  printf '.route("/v1/models", get(models))\n'
  printf 'fn legacy_field_line() {}\n'
  printf '    dump_config: bool,\n'
  printf '    other_field: u8,\n'
} > "$TD/crates/demo/src/cli.rs"
printf 'a\nb\nc\nd\n' > "$TD/crates/demo/src/other.rs"

ledger() { # ledger <cite for alpha> <cite for the comma row> <cite for legacy row>
  printf '%s\n' "$HDR"
  printf 'demo,demo alpha,6,UNKNOWN,none,no,0,c,%s,high\n' "$1"
  printf 'demo,demo beta,6,UNKNOWN,none,no,0,c,crates/demo/src/cli.rs:3,high\n'
  printf '"demo","GET /v1/models, with comma",6,UNKNOWN,none,no,0,c,"%s",high\n' "$2"
  printf 'demo,demo gamma,6,UNKNOWN,none,no,0,c,%s,high\n' "$3"
  [ -z "${FLAGCITE:-}" ] || printf 'demo,demo --dump-config,6,UNKNOWN,none,no,0,c,%s,high\n' "$FLAGCITE"
}
G_ALPHA=crates/demo/src/cli.rs:2
G_COMMA='crates/demo/src/cli.rs:4 (routes 1,2 and 5)'
G_LEGACY=crates/demo/src/cli.rs:5   # names nothing called gamma: a legacy loose cite

git -C "$TD" init -q -b main --template=
git -C "$TD" config user.email t@t; git -C "$TD" config user.name t
git -C "$TD" config commit.gpgsign false; git -C "$TD" config core.hooksPath /dev/null
FLAGCITE=crates/demo/src/cli.rs:7 ledger "$G_ALPHA" "$G_COMMA" "$G_LEGACY" > "$TD/docs/audits/surface_audit.csv"
git -C "$TD" add -A
git -C "$TD" commit -q -m base

fails=0 rows=0
run() { # run <script> -> exit code in $RC, output in $OUT
  RC=0; OUT="$(bash "$1" --root "$TD" --base main 2>&1)" || RC=$?
}
row() { # row <name> <GREEN|RED> <script> <alpha> <comma> <legacy>
  local name=$1 want=$2 script=$3
  ledger "$4" "$5" "$6" > "$TD/docs/audits/surface_audit.csv"
  run "$script"
  local got=GREEN; [ "$RC" -eq 0 ] || got=RED
  rows=$((rows + 1))
  if [ "$got" = "$want" ]; then printf 'ok    %s (%s)\n' "$name" "$got"
  else printf 'FAIL  %s: want %s got %s\n%s\n' "$name" "$want" "$got" "$OUT" >&2; fails=$((fails + 1)); fi
}

# must-pass rows
row "clean ledger"                          GREEN "$CHECK" "$G_ALPHA" "$G_COMMA" "$G_LEGACY"
row "re-cite to the correct line"           GREEN "$CHECK" "$G_ALPHA" "$G_COMMA" "$G_LEGACY"
row "comma-quoted field, valid"             GREEN "$CHECK" "$G_ALPHA" "$G_COMMA" "$G_LEGACY"
row "unchanged legacy row that names nothing" GREEN "$CHECK" "$G_ALPHA" "$G_COMMA" "$G_LEGACY"
row "range cite in bounds"                  GREEN "$CHECK" crates/demo/src/cli.rs:2-3 "$G_COMMA" "$G_LEGACY"

# must-fail rows (the issue's three mutations, plus file, parse and comma)
row "line shift 2 -> 3 (wrong symbol line)" RED   "$CHECK" crates/demo/src/cli.rs:3 "$G_COMMA" "$G_LEGACY"
row "wrong symbol 2 -> 1"                   RED   "$CHECK" crates/demo/src/cli.rs:1 "$G_COMMA" "$G_LEGACY"
row "out of bounds 99999"                   RED   "$CHECK" crates/demo/src/cli.rs:99999 "$G_COMMA" "$G_LEGACY"
row "line 0"                                RED   "$CHECK" crates/demo/src/cli.rs:0 "$G_COMMA" "$G_LEGACY"
row "range ending out of bounds"            RED   "$CHECK" crates/demo/src/cli.rs:2-99 "$G_COMMA" "$G_LEGACY"
row "reversed range"                        RED   "$CHECK" crates/demo/src/cli.rs:3-2 "$G_COMMA" "$G_LEGACY"
row "cited file missing"                    RED   "$CHECK" crates/demo/src/nope.rs:1 "$G_COMMA" "$G_LEGACY"
row "malformed cite"                        RED   "$CHECK" crates/demo/src/cli.rs:2x "$G_COMMA" "$G_LEGACY"
row "comma field: later list item OOB"      RED   "$CHECK" "$G_ALPHA" 'crates/demo/src/cli.rs:4,99999 (x)' "$G_LEGACY"
row "comma field: shifted cite"             RED   "$CHECK" "$G_ALPHA" 'crates/demo/src/cli.rs:3 (routes 1,2)' "$G_LEGACY"
row "legacy row pushed out of range"        RED   "$CHECK" "$G_ALPHA" "$G_COMMA" crates/demo/src/other.rs:50

# L1 (#7437 lane 1): a --flag feature names its flag, not the binary before it
FLAGCITE=crates/demo/src/cli.rs:6 row "flag feature: re-cite to the line naming the flag" GREEN "$CHECK" "$G_ALPHA" "$G_COMMA" "$G_LEGACY"
FLAGCITE=crates/demo/src/cli.rs:5 row "flag feature: re-cite to a wrong line" RED "$CHECK" "$G_ALPHA" "$G_COMMA" "$G_LEGACY"
# L2 (#7437 lane 2): leading space in evidence_path must not skip C2/C3/C4
row "leading-space cite, file missing" RED "$CHECK" " crates/demo/src/nope.rs:1" "$G_COMMA" "$G_LEGACY"
row "leading-space cite, out of range" RED "$CHECK" " crates/demo/src/cli.rs:99999" "$G_COMMA" "$G_LEGACY"
row "leading-space cite, shifted line" RED "$CHECK" " crates/demo/src/cli.rs:3" "$G_COMMA" "$G_LEGACY"
row "leading-space cite, valid" GREEN "$CHECK" " $G_ALPHA" "$G_COMMA" "$G_LEGACY"

# mutants: break one rule in a copy of the checker; a row the rule guards must flip
mutant() { # mutant <name> <sed expr> <want-under-mutant> <row name> <alpha> <comma> <legacy>
  local m="$TD/mutant.sh"
  sed "$2" "$CHECK" > "$m"
  if cmp -s "$CHECK" "$m"; then printf 'FAIL  mutant %s did not engage\n' "$1" >&2; fails=$((fails + 1)); rows=$((rows + 1)); return; fi
  row "mutant [$1] flips '$4'" "$3" "$m" "$5" "$6" "$7"
}
# legacy rows are held to C2/C3 only, so each of these isolates one rule
mutant "no range check" 's/hi > lines\[file\]/0/' GREEN "range ending out of bounds" crates/demo/src/cli.rs:2-99 "$G_COMMA" "$G_LEGACY"
mutant "no symbol check" 's/index(norm(txt\[file, first\]), sym) == 0/0/' GREEN "line shift" crates/demo/src/cli.rs:3 "$G_COMMA" "$G_LEGACY"
mutant "no existence check" 's/lines\[file\] < 0/0/' GREEN "legacy file missing, no line" "$G_ALPHA" "$G_COMMA" crates/demo/src/nope.rs
mutant "naive comma split (no quote state)" 's/else if (c == "\\"") q = 1/else if (0) q = 1/' RED "valid comma-quoted field" "$G_ALPHA" "$G_COMMA" "$G_LEGACY"


# --- last cited line per file (--caps), read by check_dogfood_coverage.sh G2.1 ---
# The freshness check used to split the ledger with `awk -F,`: a quoted
# evidence_path holding commas lost its row, so a hunk inside its cited range
# could be judged "after the last cited line". One quote-aware parser now.
caps() { # caps <name> <want rc> <want sorted "file=cap;..."> <script> <alpha> <comma> <legacy>
  local name=$1 wantrc=$2 want=$3 script=$4 got rc=0
  ledger "$5" "$6" "$7" > "$TD/docs/audits/surface_audit.csv"
  got="$(bash "$script" --root "$TD" --caps 2>/dev/null | sort | tr '\t\n' '=;')" || rc=$?
  [ "$wantrc" -eq 0 ] || got=""
  rows=$((rows + 1))
  if [ "$rc" -eq "$wantrc" ] && [ "$got" = "$want" ]; then printf 'ok    caps: %s\n' "$name"
  else printf 'FAIL  caps: %s: want rc=%s %s got rc=%s %s\n' "$name" "$wantrc" "$want" "$rc" "$got" >&2; fails=$((fails + 1)); fi
}
Q_COMMA='crates/demo/src/other.rs:1-2,4 (see 1,2 and crates/demo/src/cli.rs:99)'
caps "quoted comma cite keeps its row, last list item wins" 0 \
  'crates/demo/src/cli.rs=5;crates/demo/src/other.rs=4;' "$CHECK" "$G_ALPHA" "$Q_COMMA" "$G_LEGACY"
caps "a cite with no line is whole-file evidence" 0 \
  'crates/demo/src/cli.rs=5;crates/demo/src/other.rs=999999999;' "$CHECK" "$G_ALPHA" crates/demo/src/other.rs "$G_LEGACY"
caps "range cite: the end line is the cap" 0 \
  'crates/demo/src/cli.rs=5;crates/demo/src/other.rs=3;' "$CHECK" "$G_ALPHA" 'crates/demo/src/other.rs:2-3 (a, b)' "$G_LEGACY"
caps "malformed cite fails closed (exit 2)" 2 '' "$CHECK" crates/demo/src/cli.rs:2x "$G_COMMA" "$G_LEGACY"
caps "reversed range fails closed (exit 2)" 2 '' "$CHECK" "$G_ALPHA" 'crates/demo/src/other.rs:3-2,4 (x)' "$G_LEGACY"
caps "leading-space cite is still capped" 0 \
  'crates/demo/src/cli.rs=5;crates/demo/src/other.rs=2;' "$CHECK" "$G_ALPHA" " crates/demo/src/other.rs:1-2" "$G_LEGACY"
sed 's/else if (c == "\\"") q = 1/else if (0) q = 1/' "$CHECK" > "$TD/mutant.sh"
caps "mutant [naive comma split] loses the quoted row" 0 \
  'crates/demo/src/cli.rs=5;' "$TD/mutant.sh" "$G_ALPHA" "$Q_COMMA" "$G_LEGACY"
sed 's/if (h > hi) hi = h/if (i == 1) hi = h/' "$CHECK" > "$TD/mutant.sh"
caps "mutant [first list item only] under-reports the cap" 0 \
  'crates/demo/src/cli.rs=5;crates/demo/src/other.rs=2;' "$TD/mutant.sh" "$G_ALPHA" "$Q_COMMA" "$G_LEGACY"
# the gate must read its caps from this parser, not from a second comma split
rows=$((rows + 1))
# shellcheck disable=SC2016  # literal text searched for, not expanded
if grep -q 'check_surface_audit_cites.sh" --root "$root" --caps' "$SRC/scripts/check_dogfood_coverage.sh" \
   && ! grep -q "awk -F, 'NR > 1 {" "$SRC/scripts/check_dogfood_coverage.sh"; then
  printf 'ok    caps: check_dogfood_coverage.sh reads --caps (no second comma split)\n'
else printf 'FAIL  caps: check_dogfood_coverage.sh still splits evidence_path on bare commas\n' >&2; fails=$((fails + 1)); fi

printf '\n%s rows, %s failed\n' "$rows" "$fails"
[ "$fails" -eq 0 ]
