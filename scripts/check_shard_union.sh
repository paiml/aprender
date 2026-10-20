#!/usr/bin/env bash
# check_shard_union.sh -- the union of the 12 workspace-test shards == the pre-split list.
# RED when a test is dropped (or run twice). Two selectors are shared by every shard:
#  1. explicit fragments: the runner's own round-robin (EXPLICIT_LIST_SHARD=1 prints a
#     shard's selection through the REAL code path); union over 1..M must equal --list.
#  2. nextest: `--partition hash:i/M` over a probe package; union over 1..M must equal
#     the unpartitioned list (same count, none dropped, none twice).
# M is read from ci/sections.yml matrix-pins, never typed here.
# Usage: check_shard_union.sh [--fragments|--nextest|--self-test]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${SHARD_UNION_RUNNER:-scripts/ci_run_explicit_test_commands.sh}
M="$(grep -c '^    - {shard:' <(awk '/^matrix-pins:/{p=1;next} p&&/^  workspace-test-shard:/{w=1;next} w&&/^    - \{shard:/{print;next} w&&!/^    - /{exit}' ci/sections.yml))"
[ "$M" -ge 1 ] || { echo "FAIL cannot read the shard count from ci/sections.yml"; exit 1; }

fragments() { # $1 = fragment dir
  local d="$1" full un i
  full="$(bash "$RUN" --list "$d" | sort)"
  un="$(for i in $(seq 1 "$M"); do EXPLICIT_LIST_SHARD=1 bash "$RUN" --run "$d" --shard "$i/$M" | grep -v '^shard '; done | sort)"
  [ "$(printf '%s\n' "$full" | wc -l)" = "$(printf '%s\n' "$un" | wc -l)" ] || { echo "FAIL fragments: count pre-split=$(printf '%s\n' "$full" | wc -l) union=$(printf '%s\n' "$un" | wc -l)"; return 1; }
  [ "$full" = "$un" ] || { echo "FAIL fragments: union != pre-split list"; diff <(printf '%s\n' "$full") <(printf '%s\n' "$un") | head -5; return 1; }
  echo "ok    fragments: union of $M shards == pre-split list ($(printf '%s\n' "$full" | wc -l) commands)"
}

nextest() { # $1 = probe package
  local pkg="$1" full un i
  full="$(cargo nextest list -p "$pkg" --lib --message-format oneline 2>/dev/null | sort)"
  [ -n "$full" ] || { echo "FAIL nextest: empty pre-split list for $pkg"; return 1; }
  un="$(for i in $(seq 1 "$M"); do cargo nextest list -p "$pkg" --lib --message-format oneline --partition "hash:$i/$M" 2>/dev/null; done | sort)"
  [ "$(printf '%s\n' "$full" | wc -l)" = "$(printf '%s\n' "$un" | wc -l)" ] || { echo "FAIL nextest: count pre-split=$(printf '%s\n' "$full" | wc -l) union=$(printf '%s\n' "$un" | wc -l)"; return 1; }
  [ "$full" = "$un" ] || { echo "FAIL nextest: union != pre-split list"; return 1; }
  echo "ok    nextest: union of $M hash shards == pre-split list for $pkg ($(printf '%s\n' "$full" | wc -l) tests)"
}

case "${1:-}" in
  --fragments|"") fragments ci/explicit-test-commands.d ;;
  --nextest) nextest "${2:-aprender-contracts}" ;;
  --self-test)
    t="$(mktemp -d)"; trap 'rm -rf "${t:?}"' EXIT
    # mutant: the selector drops the last command of the round-robin (a test goes missing)
    sed 's/if \[ "\$sel" -eq "\$n" \]; then mine+=("\$cmd"); fi/if [ "$sel" -eq "$n" ] \&\& [ "$k" -lt "$(\( \${#cmds[@]} \))" ]; then mine+=("$cmd"); fi/' "$RUN" > "$t/mut.sh"
    cmp -s "$RUN" "$t/mut.sh" && { echo "self-test: mutation did not apply"; exit 1; }
    SHARD_UNION_RUNNER="$t/mut.sh" bash "$0" --fragments >/dev/null 2>&1 && { echo "self-test: dropped command must be RED"; exit 1; }
    fragments ci/explicit-test-commands.d >/dev/null || { echo "self-test: clean must pass"; exit 1; }
    echo "PASS  check_shard_union self-test: clean ok, dropped-command mutant RED" ;;
  *) echo "usage: $0 [--fragments|--nextest [pkg]|--self-test]" >&2; exit 2 ;;
esac
