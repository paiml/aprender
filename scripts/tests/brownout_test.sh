#!/usr/bin/env bash
# brownout_test.sh -- falsifier for aprender#5751 (PMAT-5751): a WAN brownout must not fail a merge job.
#   R1  a cold per-PR registry is seeded from the host-local seed: crates are on disk before cargo runs
#   R2  a warm registry is left alone (never clobbered by the seed)
#   R3  no seed present: exit 0, registry stays cold (a missing seed is a cold start, not a failure)
#   R4  publish refuses a registry with no cached crates (an empty one must not poison the seed)
#   R5  publish makes a cold registry seedable (round trip)
#   P1  pin_build_base never touches the network when the build base is already in the checkout
#   P2  pin_build_base still fetches when the base is absent (fallback unchanged)
# Usage: bash scripts/tests/brownout_test.sh   exit 0 all green, 1 any red
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SEED="$ROOT/scripts/ci/seed_registry.sh"
PIN="$ROOT/scripts/ci/pin_build_base.sh"
TD=$(mktemp -d); trap 'rm -rf "${TD:?}"' EXIT
export GIT_CEILING_DIRECTORIES="$TD"   # #5025: a slipped -C must fail, not write into the shared checkout
ok=0; bad=0
row() { if [ "$2" = "$3" ]; then ok=$((ok+1)); printf 'ok    %s\n' "$1"; else bad=$((bad+1)); printf 'RED   %s (got %s, want %s)\n' "$1" "$2" "$3"; fi; }
[ -f "$SEED" ] || { printf 'RED   %s is missing\n' "$SEED"; bad=$((bad+1)); }

mkreg() { mkdir -p "$1/cache/idx" "$1/index/idx"; printf 'crate\n' > "$1/cache/idx/linux-raw-sys-0.4.14.crate"; printf 'i\n' > "$1/index/idx/.cache"; }
S="$TD/seed"; mkreg "$S"

R1="$TD/r1"; mkdir -p "$R1"
bash "$SEED" seed "$R1" "$S" >/dev/null 2>&1; rc=$?
row "R1 cold registry gets the crate" "$rc $([ -f "$R1/cache/idx/linux-raw-sys-0.4.14.crate" ] && echo present || echo absent)" "0 present"

R2="$TD/r2"; mkreg "$R2"; printf 'mine\n' > "$R2/cache/idx/own.crate"
bash "$SEED" seed "$R2" "$S" >/dev/null 2>&1
row "R2 warm registry untouched" "$([ -f "$R2/cache/idx/own.crate" ] && echo kept || echo clobbered)" kept

R3="$TD/r3"; mkdir -p "$R3"
bash "$SEED" seed "$R3" "$TD/no-such-seed" >/dev/null 2>&1; rc=$?
row "R3 missing seed is a cold start" "$rc $(ls -A "$R3" | wc -l)" "0 0"

E="$TD/empty"; mkdir -p "$E"
bash "$SEED" publish "$E" "$TD/seed2" >/dev/null 2>&1; rc=$?
row "R4 empty registry is not published" "$([ "$rc" -ne 0 ] && [ ! -d "$TD/seed2" ] && echo refused || echo published)" refused

bash "$SEED" publish "$R2" "$TD/seed2" >/dev/null 2>&1; rc=$?
R5="$TD/r5"; mkdir -p "$R5"; bash "$SEED" seed "$R5" "$TD/seed2" >/dev/null 2>&1
row "R5 publish then seed round trip" "$rc $([ -f "$R5/cache/idx/own.crate" ] && echo present || echo absent)" "0 present"

# --- pin_build_base: a PR merge commit whose first parent is the base, checkout deep enough to hold it.
o="$TD/origin.git"; w="$TD/w"
git init -q --template= --bare -b main "$o"; git init -q --template= -b main "$w"
g() { git -C "$w" -c user.name=t -c user.email=t@example.invalid -c commit.gpgsign=false "$@"; }
printf a > "$w/a"; g add -A; g commit -q -m c0; c0=$(g rev-parse HEAD)
g checkout -q -b pr; printf b > "$w/b"; g add -A; g commit -q -m pr; prh=$(g rev-parse HEAD)
g checkout -q --detach "$c0"; g merge -q --no-ff --no-edit pr; m=$(g rev-parse HEAD)
# Network is DOWN: a fetch_p6 stand-in that fails and records that it was called.
printf '#!/usr/bin/env bash\necho called >> "%s/net-calls"\nexit 1\n' "$TD" > "$TD/dead_fetch.sh"
pinrun() { ( cd "$1" && GITHUB_EVENT_NAME=pull_request PR_HEAD_SHA="$prh" P6_OVERRIDE="$TD/dead_fetch.sh" bash "$PIN" ) >"$TD/pin.log" 2>&1; }
rc=0; pinrun "$w" || rc=$?
row "P1 base already local: no network, pinned" "$rc $(git -C "$w" rev-parse origin/main 2>/dev/null | cut -c1-7) $(ls "$TD/net-calls" 2>/dev/null | wc -l)" "0 $(printf %s "$c0" | cut -c1-7) 0"

# P2: a depth-1 checkout of the merge commit holds no parent.
d="$TD/shallow"; git init -q --template= "$d"; git -C "$d" remote add origin "file://$w"
git -C "$d" fetch -q --depth=1 origin "$m" && git -C "$d" checkout -q --detach FETCH_HEAD
rm -f "${TD:?}/net-calls"
rc=0; pinrun "$d" || rc=$?
row "P2 base absent: fetch attempted, refuses when net is down" "$rc $(ls "$TD/net-calls" 2>/dev/null | wc -l)" "1 1"

printf 'brownout_test: %s ok, %s red\n' "$ok" "$bad"
[ "$bad" -eq 0 ]
