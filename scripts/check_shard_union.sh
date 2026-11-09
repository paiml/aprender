#!/usr/bin/env bash
# check_shard_union.sh -- the union of the workspace-test shards (at each event's count) == the pre-split list.
# RED when a test is dropped (or run twice). Two selectors are shared by every shard:
#  1. explicit fragments: the runner's own round-robin (EXPLICIT_LIST_SHARD=1 prints a
#     shard's selection through the REAL code path); union over 1..M must equal --list.
#  2. nextest: scripts/shard_union_nextest.sh (kept out of this cargo-free guard).
# M is read from ci/sections.yml matrix-pins, never typed here.
# Usage: check_shard_union.sh [--fragments|--self-test]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${SHARD_UNION_RUNNER:-scripts/ci_run_explicit_test_commands.sh}
# The count is keyed by event: `workspace-test-shard` pins are pull_request's, the optional
# `workspace-test-shard-merge_group` pins are the merge queue's. The union must hold at EACH event's M.
pins_m() { # $1 = pins key; prints the number of pins under it (0 if absent)
  awk -v k="$1" '/^matrix-pins:/{p=1;next} p&&$0==("  " k ":"){w=1;next} w&&/^    - \{shard:/{n++;next} w&&!/^    - /{exit} END{print n+0}' ci/sections.yml
}
M="$(pins_m workspace-test-shard)"
[ "$M" -ge 1 ] || { echo "FAIL cannot read the shard count from ci/sections.yml"; exit 1; }
MG="$(pins_m workspace-test-shard-merge_group)"

fragments() { # $1 = fragment dir  $2 = shard count M  $3 = event label
  local d="$1" M="$2" ev="$3" full un i sh_out
  # The runner's exit code is a verdict: it exits 2 on an empty dir ("zero commands is broken wiring"). A
  # pipeline in $( ) hides it, so capture the raw output first and fail closed (FLAKE-001 C3, paiml/infra#3479).
  full="$(bash "$RUN" --list "$d")" || { echo "FAIL fragments: runner --list failed on $d (UNMEASURED)"; return 1; }
  [ -n "$full" ] || { echo "FAIL fragments: runner --list printed no commands for $d (UNMEASURED)"; return 1; }
  full="$(printf '%s\n' "$full" | sort)"
  un=""
  for i in $(seq 1 "$M"); do
    sh_out="$(EXPLICIT_LIST_SHARD=1 bash "$RUN" --run "$d" --shard "$i/$M")" || { echo "FAIL fragments: runner --run shard $i/$M failed (UNMEASURED)"; return 1; }
    un+="$(printf '%s\n' "$sh_out" | grep -v '^shard ' || true)"$'\n'
  done
  un="$(printf '%s' "$un" | sed '/^$/d' | sort)"
  [ "$(printf '%s\n' "$full" | wc -l)" = "$(printf '%s\n' "$un" | wc -l)" ] || { echo "FAIL fragments: count pre-split=$(printf '%s\n' "$full" | wc -l) union=$(printf '%s\n' "$un" | wc -l)"; return 1; }
  [ "$full" = "$un" ] || { echo "FAIL fragments: union != pre-split list"; diff <(printf '%s\n' "$full") <(printf '%s\n' "$un") | head -5; return 1; }
  echo "ok    fragments ($ev): union of $M shards == pre-split list ($(printf '%s\n' "$full" | wc -l) commands)"
}


fragments_all() { # $1 = fragment dir: every event that has pins
  fragments "$1" "$M" pull_request || return 1
  if [ "$MG" -ge 1 ]; then fragments "$1" "$MG" merge_group || return 1; fi
}

case "${1:-}" in
  --fragments|"") fragments_all ci/explicit-test-commands.d ;;
  --self-test)
    t="$(mktemp -d)"; trap 'rm -rf "${t:?}"' EXIT
    # mutant: the selector drops the last command of the round-robin (a test goes missing)
    sed 's/if \[ "\$sel" -eq "\$n" \]; then mine+=("\$cmd"); fi/if [ "$sel" -eq "$n" ] \&\& [ "$k" -lt "$(\( \${#cmds[@]} \))" ]; then mine+=("$cmd"); fi/' "$RUN" > "$t/mut.sh"
    cmp -s "$RUN" "$t/mut.sh" && { echo "self-test: mutation did not apply"; exit 1; }
    # RED must be THIS guard's own verdict: rc 1 and its count-mismatch line, not any crash of the mutant runner
    rc=0; out="$(SHARD_UNION_RUNNER="$t/mut.sh" bash "$0" --fragments 2>&1)" || rc=$?
    [ "$rc" = 1 ] || { echo "self-test: dropped command must exit 1, got $rc"; exit 1; }
    case "$out" in *"FAIL fragments: count pre-split="*) ;; *) echo "self-test: dropped command must print the count-mismatch FAIL line, got: $out" | head -c 400; echo; exit 1 ;; esac
    fragments_all ci/explicit-test-commands.d >/dev/null || { echo "self-test: clean must pass"; exit 1; }
    [ "$MG" -ge 1 ] && { out="$(fragments_all ci/explicit-test-commands.d)"; case "$out" in *"(merge_group)"*) ;; *) echo "self-test: merge_group pins exist but its union was not checked"; exit 1 ;; esac; }
    # known-bad (FLAKE-001 C3): an EMPTY fragment dir must be RED. The runner exits 2 ("zero commands is
    # broken wiring"), and a guard that swallows that exit code reports `ok ... (1 commands)` -- a vacuous pass.
    mkdir "$t/empty"; rc=0; out="$(fragments "$t/empty" "$M" pull_request 2>&1)" || rc=$?
    [ "$rc" = 1 ] || { echo "self-test: empty fragment dir must exit 1, got $rc: $out" | head -c 400; echo; exit 1; }
    case "$out" in *"FAIL fragments: runner"*) ;; *) echo "self-test: empty dir must print the runner FAIL line, got: $out" | head -c 400; echo; exit 1 ;; esac
    # each fail-closed branch ALONE (lane f20-a6): a stub runner fails exactly one call, so neither branch can mask the other
    printf '#!/usr/bin/env bash\ncase "$1" in --list) exit 2 ;; *) exit 0 ;; esac\n' > "$t/stub-list.sh"
    printf '#!/usr/bin/env bash\ncase "$1" in --list) echo cmd1 ;; *) exit 2 ;; esac\n' > "$t/stub-run.sh"
    for pair in "stub-list.sh:runner --list failed" "stub-run.sh:runner --run shard"; do
      rc=0; out="$( RUN="$t/${pair%%:*}"; fragments "$t/empty" 2 pull_request 2>&1 )" || rc=$?
      [ "$rc" = 1 ] || { echo "self-test: ${pair%%:*} must exit 1, got $rc: $out" | head -c 400; echo; exit 1; }
      case "$out" in *"FAIL fragments: ${pair#*:}"*) ;; *) echo "self-test: ${pair%%:*} must print '${pair#*:}', got: $out" | head -c 400; echo; exit 1 ;; esac
    done
    echo "PASS  check_shard_union self-test: clean ok, dropped-command mutant RED, empty dir RED, each fail-closed branch RED alone" ;;
  *) echo "usage: $0 [--fragments|--self-test]" >&2; exit 2 ;;
esac
