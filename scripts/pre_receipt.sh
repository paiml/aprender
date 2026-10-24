#!/usr/bin/env bash
# pre_receipt.sh -- the PRE receipt (FLOW-011 rev6 section 3): no review starts without one bound to the exact head.
#
#   bash scripts/pre_receipt.sh BASE HEAD --self-review FILE [--out FILE] [--max-lines N]
#        [--gate-cmd CMD] [--mutants-list-cmd CMD] [--mutants-run-cmd CMD]
#   bash scripts/pre_receipt.sh --self-test
#
# Writes a JSON receipt {head, base, verdict, reasons, gate, mutants:{state,listed}, selftest, self_review, size, solo_class, tier} and
# exits 0 only when every check passed. CHECKS (each failure is a named reason, none is skipped):
#   head      the working tree IS HEAD and is clean, so what ran is what the receipt names.
#   gate      the repo's own gate (default `make gate`) passes on that head.
#   mutants   a change to executable .rs (comments and tests stripped) must LIST >= 1 mutant, then the in-diff run
#             must pass. 0 mutants on executable .rs is a FAIL (anti-vacuity); mutants_diff_gate.sh alone passes it.
#   selftest  a change to .sh/.yaml/.yml/Makefile needs a changed script with a --self-test that passes, and every
#             changed .sh passes `bash -n` and `shellcheck -S error`.
#   review    --self-review FILE exists, is non-empty and has a line `open: 0` (findings fixed or answered).
#   solo      a `PMAT-Solo: docs|code|gate` trailer on any commit must equal the class derived from the paths
#             (review_tier.sh: light=docs, standard=code, sensitive=gate). No trailer is fine.
#   size      changed lines <= --max-lines (default 400), generated files (roadmap.yaml, Cargo.lock, witnesses) out.
# The two mutants commands see the diff as $PRE_DIFF.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GATE_CMD='make gate'
LIST_CMD='cargo mutants --workspace --in-diff "$PRE_DIFF" --list'
RUN_CMD='bash scripts/mutants_diff_gate.sh "$PRE_DIFF"'
MAX=400
reasons=()
fail() { reasons+=("$1"); }

# exec_rs BASE HEAD -> changed .rs files with a non-comment, non-test change
exec_rs() {
  local f
  git diff --name-only --no-renames --diff-filter=AMR "$1" "$2" -- '*.rs' | while IFS= read -r f; do
    case $f in */tests/*|*_tests.rs|*/tests.rs|*/benches/*) continue ;; esac
    [[ $(git diff -U0 "$1" "$2" -- "$f" | grep -E '^[+-]' | grep -Ev '^(\+\+\+|---)' | grep -Ev '^[+-][[:space:]]*(//.*)?$' | head -1) ]] && echo "$f"
  done
}

check_head() {
  [[ $(git rev-parse HEAD) == "$(git rev-parse "$2")" ]] || fail "head: the checkout is $(git rev-parse --short HEAD), not $2"
  [[ -z $(git status --porcelain --untracked-files=no) ]] || fail "head: the working tree is not clean"
}

check_gate() {
  if bash -c "$GATE_CMD" > "$W/gate.log" 2>&1; then GATE=pass; else GATE=fail; fail "gate: \`$GATE_CMD\` failed: $(tail -1 "$W/gate.log")"; fi
}

check_mutants() { # BASE HEAD
  local rs n
  rs=$(exec_rs "$1" "$2")
  if [[ -z $rs ]]; then MUT_STATE=none; return; fi
  git diff "$1" "$2" > "$W/pr.diff"; export PRE_DIFF=$W/pr.diff
  if ! bash -c "$LIST_CMD" > "$W/list.txt" 2> "$W/list.err"; then MUT_STATE=fail; fail "mutants: listing failed, so nothing can be said: $(tail -1 "$W/list.err")"; return; fi
  n=$(grep -c . "$W/list.txt" || true); MUT_LISTED=$n
  if [[ $n -lt 1 ]]; then MUT_STATE=fail; fail "mutants: 0 mutants listed for executable .rs change (${rs//$'\n'/ }): a vacuous pass is a fail"; return; fi
  if bash -c "$RUN_CMD" > "$W/mut.log" 2>&1; then MUT_STATE=pass; else MUT_STATE=fail; fail "mutants: in-diff run failed: $(tail -1 "$W/mut.log")"; fi
}

check_selftest() { # BASE HEAD
  local f need=0 have=0 sh
  git diff --name-only --no-renames --diff-filter=AMR "$1" "$2" | grep -E '\.(sh|ya?ml)$|(^|/)Makefile$' > "$W/ne.txt" || true
  [[ -s $W/ne.txt ]] && need=1
  sh=$(git diff --name-only --no-renames --diff-filter=AMR "$1" "$2" -- '*.sh' || true)
  for f in $sh; do
    bash -n "$f" 2>/dev/null || fail "selftest: bash -n fails on $f"
    shellcheck -S error "$f" > /dev/null 2>&1 || fail "selftest: shellcheck -S error fails on $f"
    if grep -q -- '--self-test' "$f"; then
      if bash "$f" --self-test > "$W/st.log" 2>&1; then have=1; else fail "selftest: $f --self-test failed: $(tail -1 "$W/st.log")"; fi
    fi
  done
  [[ $need -eq 1 && $have -eq 0 ]] && fail "selftest: .sh/.yaml/Makefile changed but no changed script has a passing --self-test"
  SELFTEST=$have
}

check_review() {
  if [[ -z ${REVIEW:-} || ! -s ${REVIEW:-/nonexistent} ]]; then fail "review: --self-review FILE is missing or empty"; return; fi
  grep -qE '^open: 0[[:space:]]*$' "$REVIEW" || fail "review: $REVIEW has no 'open: 0' line (findings unfixed or unanswered)"
}

check_solo() { # BASE HEAD
  local t want got
  t=$(bash "$HERE/review_tier.sh" "$1" "$2" 2>/dev/null) || { fail "solo: review_tier.sh could not derive a tier"; return; }
  TIER=$t
  case $t in light) want=docs ;; standard) want=code ;; *) want=gate ;; esac
  SOLO=$want
  git log --format=%B "$1..$2" | grep -oE '^PMAT-Solo:[[:space:]]*[a-z]+' | sed 's/^PMAT-Solo:[[:space:]]*//' | sort -u > "$W/solo.txt" || true
  while IFS= read -r got; do
    [[ -z $got || $got == "$want" ]] || fail "solo: trailer says $got, the paths say $want"
  done < "$W/solo.txt"
}

check_size() { # BASE HEAD
  local n
  n=$(git diff --numstat --no-renames "$1" "$2" -- . ':!docs/roadmaps/roadmap.yaml' ':!Cargo.lock' ':!contracts/witness' | awk '{ if ($1 != "-") s += $1 + $2 } END { print s + 0 }')
  SIZE=$n
  [[ $n -le $MAX ]] || fail "size: $n changed lines > $MAX: split the change"
}

emit() { # BASE HEAD
  local verdict=pass r
  [[ ${#reasons[@]} -eq 0 ]] || verdict=fail
  r=$(printf '%s\n' "${reasons[@]:-}" | jq -R . | jq -s 'map(select(. != ""))')
  jq -n --arg head "$(git rev-parse "$2")" --arg base "$(git rev-parse "$1")" --arg v "$verdict" --argjson reasons "$r" \
     --arg gate "${GATE:-}" --arg ms "${MUT_STATE:-}" --argjson ml "${MUT_LISTED:-0}" --argjson st "${SELFTEST:-0}" \
     --arg sr "${REVIEW:-}" --argjson size "${SIZE:-0}" --arg solo_class "${SOLO:-}" --arg tier "${TIER:-}" \
     '{head:$head, base:$base, verdict:$v, reasons:$reasons, gate:$gate, mutants:{state:$ms, listed:$ml}, selftest:($st==1), self_review:$sr, size:$size, solo_class:$solo_class, tier:$tier}'
  [[ $verdict == pass ]]
}

run() { # BASE HEAD
  W=$(mktemp -d); trap 'rm -rf "${W:?}"' RETURN
  check_head "$1" "$2"; check_gate; check_mutants "$1" "$2"; check_selftest "$1" "$2"
  check_review; check_solo "$1" "$2"; check_size "$1" "$2"
  if [[ -n ${OUT:-} ]]; then emit "$1" "$2" > "$OUT"; local rc=$?; cat "$OUT"; return $rc; else emit "$1" "$2"; fi
}

self_test() {
  local d pass=0 fail_n=0 base self out
  self=$(realpath "${BASH_SOURCE[0]}")
  d=$(mktemp -d); trap 'rm -rf "${d:?}" "${d:?}.rv"' RETURN
  git -C "$d" init -q; git -C "$d" config user.email t@t; git -C "$d" config user.name t
  mkdir -p "$d/scripts" "$d/crates/x/src" "$d/docs" "$d/.pmat"
  cp "$HERE/review_tier.sh" "$d/scripts/"; cp "$HERE/../.pmat/review-tiers.yaml" "$d/.pmat/"
  printf 'fn a() {}\n' > "$d/crates/x/src/lib.rs"; printf '# d\n' > "$d/docs/a.md"
  git -C "$d" add -A; git -C "$d" commit -qm base; base=$(git -C "$d" rev-parse HEAD)
  mkdir -p "$d.rv"; printf 'open: 0\n' > "$d.rv/review.txt"; printf 'open: 2\n' > "$d.rv/review-open.txt"
  expect() { # NAME WANT(pass|fail:substring) EXTRA-ARGS...
    local name=$1 want=$2; shift 2
    git -C "$d" add -A; git -C "$d" commit -qm "$name${TRAILER:+

$TRAILER}" --allow-empty
    out=$(cd "$d" && bash "$self" "$base" HEAD --self-review "$d.rv/${RV:-review.txt}" --gate-cmd "${GATE:-true}" \
          --mutants-list-cmd "${LIST:-echo m1}" --mutants-run-cmd "${RUN:-true}" "$@" 2>&1)
    local rc=$?
    if [[ $want == pass ]]; then [[ $rc -eq 0 ]] && pass=$((pass + 1)) || { fail_n=$((fail_n + 1)); echo "FAIL $name: want pass, rc=$rc: $(printf '%s' "$out" | jq -r '.reasons|join("; ")' 2>/dev/null)"; }
    else [[ $rc -ne 0 && $out == *"${want#fail:}"* ]] && pass=$((pass + 1)) || { fail_n=$((fail_n + 1)); echo "FAIL $name: want fail:${want#fail:}, rc=$rc"; }
    fi
    git -C "$d" reset -q --hard "$base"
  }
  printf 'fn b() {}\n' >> "$d/crates/x/src/lib.rs";                                         expect code-change-passes pass
  printf 'fn b() {}\n' >> "$d/crates/x/src/lib.rs"; GATE=false;                              expect gate-red "fail:gate:"; unset GATE
  printf 'fn b() {}\n' >> "$d/crates/x/src/lib.rs"; LIST=true;                              expect zero-mutants-is-fail "fail:0 mutants listed"; unset LIST
  printf 'fn b() {}\n' >> "$d/crates/x/src/lib.rs"; LIST=false;                             expect list-failure-is-fail "fail:listing failed"; unset LIST
  printf 'fn b() {}\n' >> "$d/crates/x/src/lib.rs"; RUN=false;                              expect survivor-is-fail "fail:in-diff run failed"; unset RUN
  printf '// only a comment\n' >> "$d/crates/x/src/lib.rs"; LIST=true;                      expect comment-only-needs-no-mutants pass; unset LIST
  printf 'fn b() {}\n' >> "$d/crates/x/src/lib.rs"; RV=review-open.txt;                      expect open-findings-fail "fail:open: 0"; unset RV
  printf 'fn b() {}\n' >> "$d/crates/x/src/lib.rs"; RV=missing.txt;                          expect no-self-review-fails "fail:--self-review"; unset RV
  printf 'fn b() {}\n' >> "$d/crates/x/src/lib.rs"; TRAILER='PMAT-Solo: docs';               expect mislabelled-docs-fails "fail:trailer says docs"; unset TRAILER
  printf 'fn b() {}\n' >> "$d/crates/x/src/lib.rs"; TRAILER='PMAT-Solo: code';               expect honest-solo-passes pass; unset TRAILER
  seq 1 450 | sed 's/^/# line /' > "$d/docs/big.md";                                        expect oversize-fails "fail:changed lines"
  printf '#!/usr/bin/env bash\necho hi\n' > "$d/scripts/new.sh";                                                  expect script-without-selftest-fails "fail:no changed script has a passing --self-test"
  printf '#!/usr/bin/env bash\nif [ "${1:-}" = --self-test ]; then exit 0; fi\n' > "$d/scripts/new.sh";             expect script-with-selftest-passes pass
  printf '#!/usr/bin/env bash\nif [ "${1:-}" = --self-test ]; then exit 1; fi\n' > "$d/scripts/new.sh";             expect failing-selftest-fails "fail:--self-test failed"
  printf '#!/usr/bin/env bash\nif then\n' > "$d/scripts/bad.sh";                                                  expect bash-syntax-fails "fail:bash -n"
  printf '# d2\n' >> "$d/docs/a.md";                                                          expect docs-only-passes pass
  # the head check: a checkout that is not the named head
  printf 'fn b() {}\n' >> "$d/crates/x/src/lib.rs"; git -C "$d" add -A; git -C "$d" commit -qm one
  out=$(cd "$d" && bash "$self" "$base" "$base" --self-review "$d.rv/review.txt" --gate-cmd true 2>&1); [[ $out == *"head:"* ]] && pass=$((pass + 1)) || { fail_n=$((fail_n + 1)); echo "FAIL head-mismatch not caught"; }
  echo "pre_receipt self-test: $pass passed, $fail_n failed"
  [[ $fail_n -eq 0 ]]
}

[[ ${1:-} == --self-test ]] && { self_test; exit $?; }
BASE=${1:?usage: pre_receipt.sh BASE HEAD --self-review FILE [...]}; HEADREF=${2:?usage}; shift 2
OUT=""; REVIEW=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --out) OUT=${2:?}; shift 2 ;;
    --self-review) REVIEW=${2:?}; shift 2 ;;
    --max-lines) MAX=${2:?}; shift 2 ;;
    --gate-cmd) GATE_CMD=${2:?}; shift 2 ;;
    --mutants-list-cmd) LIST_CMD=${2:?}; shift 2 ;;
    --mutants-run-cmd) RUN_CMD=${2:?}; shift 2 ;;
    *) echo "pre_receipt: unknown argument $1" >&2; exit 2 ;;
  esac
done
run "$BASE" "$HEADREF"
