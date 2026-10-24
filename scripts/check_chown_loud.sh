#!/usr/bin/env bash
# check_chown_loud.sh -- an ownership restore in ci/sections.yml may not fail silently or name a fixed uid.
#   bash scripts/check_chown_loud.sh [--self-test]
# gx10 runners mix uid 1000 and 1001, so `chown 1000:1000` locks the runner user out of its own .git, and
# `chown ... 2>/dev/null || true` hid that: the pre-checkout step reported success while the tree stayed root-owned
# (#6235). A non-comment line that runs chown must pass "$OWNER" (the runner's own uid:gid), and must not discard
# its stderr or its exit status. Exit 0 = clean.
set -uo pipefail
FILE=${CHOWN_LOUD_FILE:-ci/sections.yml}

# violations FILE -> "line: text" for every non-comment line that runs chown badly
violations() {
  awk '
    /^[[:space:]]*#/ { next }
    /chown/ && (/1000:1000/ || /2>\/dev\/null/ || /\|\| true/ || /\|\| :/) { printf "%d: %s\n", NR, $0 }
  ' "$1"
}

check() {
  local v
  [[ -f $1 ]] || { echo "RED   $1 is missing: nothing was judged"; return 2; }
  grep -q 'chown' "$1" || { echo "RED   no chown in $1: the guard has gone blind"; return 2; }
  v=$(violations "$1")
  if [[ -n $v ]]; then echo "RED   chown that is silent or names a fixed uid in $1:"; printf '%s\n' "$v" | sed 's/^/  /' | cut -c1-200; return 1; fi
  echo "ok    every chown in $1 passes the runner's own uid and reports failure"
}

self_test() {
  local w rc=0 m
  w=$(mktemp -d); trap 'rm -rf "${w:?}"' RETURN
  check "$FILE" || rc=1
  mutant() { # NAME SED-EXPR
    cp "$FILE" "$w/m.yml"; sed -i "$2" "$w/m.yml"
    if cmp -s "$FILE" "$w/m.yml"; then echo "FAIL  mutant $1 did not apply"; rc=1; return; fi
    if check "$w/m.yml" > /dev/null 2>&1; then echo "FAIL  mutant $1 survived"; rc=1; else echo "ok    mutant $1 turns it red"; fi
  }
  mutant fixed-uid     '0,/chown -R "\$OWNER"/s//chown -R 1000:1000/'
  mutant silent-stderr '0,/chown -R "\$OWNER" \/workspace |/s//chown -R "$OWNER" \/workspace 2>\/dev\/null |/'
  mutant swallowed-rc  '0,/ || echo "::warning::ownership/s// || true # ::warning::ownership/'
  printf '# a comment about chown 1000:1000 || true\n' > "$w/c.yml"; printf 'run: chown -R "$OWNER" /w || echo warn\n' >> "$w/c.yml"
  m=$(check "$w/c.yml") && echo "ok    a comment naming 1000:1000 is not a violation" || { echo "FAIL  comment judged: $m"; rc=1; }
  printf 'run: echo hello\n' > "$w/n.yml"
  check "$w/n.yml" > /dev/null 2>&1 && { echo "FAIL  a file with no chown passed"; rc=1; } || echo "ok    a file with no chown is red (blind guard)"
  return "$rc"
}

case ${1:-} in
  "") check "$FILE" ;;
  --self-test) self_test ;;
  *) echo "usage: check_chown_loud.sh [--self-test]" >&2; exit 2 ;;
esac
