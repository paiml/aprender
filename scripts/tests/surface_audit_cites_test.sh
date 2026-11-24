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
} > "$TD/crates/demo/src/cli.rs"
printf 'a\nb\nc\nd\n' > "$TD/crates/demo/src/other.rs"

ledger() { # ledger <cite for alpha> <cite for the comma row> <cite for legacy row>
  printf '%s\n' "$HDR"
  printf 'demo,demo alpha,6,UNKNOWN,none,no,0,c,%s,high\n' "$1"
  printf 'demo,demo beta,6,UNKNOWN,none,no,0,c,crates/demo/src/cli.rs:3,high\n'
  printf '"demo","GET /v1/models, with comma",6,UNKNOWN,none,no,0,c,"%s",high\n' "$2"
  printf 'demo,demo gamma,6,UNKNOWN,none,no,0,c,%s,high\n' "$3"
}
G_ALPHA=crates/demo/src/cli.rs:2
G_COMMA='crates/demo/src/cli.rs:4 (routes 1,2 and 5)'
G_LEGACY=crates/demo/src/cli.rs:5   # names nothing called gamma: a legacy loose cite

git -C "$TD" init -q -b main --template=
git -C "$TD" config user.email t@t; git -C "$TD" config user.name t
git -C "$TD" config commit.gpgsign false; git -C "$TD" config core.hooksPath /dev/null
ledger "$G_ALPHA" "$G_COMMA" "$G_LEGACY" > "$TD/docs/audits/surface_audit.csv"
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

printf '\n%s rows, %s failed\n' "$rows" "$fails"
[ "$fails" -eq 0 ]
