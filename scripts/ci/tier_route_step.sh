#!/usr/bin/env bash
# FLOW-003 QM-11 (#4529) SHADOW: the tier router's decision beside today's tier.
#
# TELEMETRY. It gates nothing: no step, job or check reads what it writes. It exists so
# the ruling that decides HOW the router joins today's tiers (docs/audits/qm11-design-note.md,
# options A/B/C) is made on real PRs and the real input set, not on a model of them.
#
# It runs once per CI run, in the workspace-test-build section, after the tier is decided:
#   1. the newest unexpired `ci-input-set` artifact of a nightly on main (QM-10);
#   2. `input_set.sh check` — a set the checker refuses (incl. < 7 nightlies) is an ABSTENTION;
#   3. INV-BASE: `input_set.sh base` exit 0 -> --base-valid; exit 10 -> the n..base paths are --stale;
#   4. owners from rustc dep-info when this run compiled (QM11_TARGET_DIR), else NAMED absent;
#   5. `tier_router.py route` over the PR's touched paths.
# An abstention is NAMED on the one line this prints and in <out.json>, never a silent
# default: a shadow that reports nothing looks exactly like a shadow that is not wired.
#
# No caller-supplied string is ever pasted into a shell script: the base ref reaches the
# child shell as $QM11_BASE (quorum finding, 2026-09-29 -- a ref may contain a quote).
#
# usage: tier_route_step.sh <repo> <today-tier.txt> <changed.txt> <base-ref> <out.json>
#        tier_route_step.sh --self-test
# env:   QM11_INPUT_SET  an input-set.json on disk: skip the fetch (the self-test, a local run)
#        QM11_METADATA   a `cargo metadata --no-deps` JSON on disk: skip cargo
#        QM11_TARGET_DIR a cargo target dir this run compiled into: owners from its dep-info
#        QM11_BIN        where tier_router.py and input_set.sh live (default: beside this file)
#        IMAGE           run the fetch, cargo, jq and git steps in this image (the clean-room
#                        host has no cargo); unset -> run them on this host
#        GH_TOKEN, GITHUB_REPOSITORY   the fetch
# Prints ONE line `qm11-shadow: today=<tier> router=<tier>|abstained ...`.
# exit: 0 a decision or a named abstention · 1 a fault · 64 usage
set -euo pipefail

HERE=${QM11_BIN:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
ROUTER="$HERE/tier_router.py"
SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")

usage() { sed -n '/^# usage:/,/^# exit:/p' "$0" >&2; exit 64; }

# in_env <repo> <work> <script>: run a FIXED bash script with $W = the work dir, $BIN = the
# helper dir, $QM11_BASE = the base ref, and the repo as cwd; in $IMAGE when set.
in_env() {
  local repo=$1 work=$2 script=$3
  if [ -n "${IMAGE:-}" ]; then
    docker run --rm --user "$(id -u):$(id -g)" \
      -v "$repo:/workspace" -v "$HERE:/qm11-bin:ro" -v "$work:/w" -w /workspace \
      -e W=/w -e BIN=/qm11-bin -e HOME=/w -e QM11_BASE -e GH_TOKEN -e GITHUB_REPOSITORY \
      -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0='*' \
      "$IMAGE" bash -c "$script"
  else
    (cd "$repo" && W="$work" BIN="$HERE" bash -c "$script")
  fi
}

emit() { # emit <out.json> <today> <abstained|""> [route.json] [owners-source]
  local out=$1 today=$2 why=$3 route=${4:-} owners=${5:-}
  python3 - "$out" "$today" "$why" "$route" "$owners" <<'PY'
import json, sys
out, today, why, route, owners = sys.argv[1:6]
doc = {"schema": "qm11-shadow-v1", "today": today, "abstained": why or None,
       "router": None, "owners": owners or None}
if route:
    doc["router"] = json.load(open(route))
json.dump(doc, open(out, "w"), indent=1)
if why:
    print(f"qm11-shadow: today={today} router=abstained why={why}")
else:
    r = doc["router"]
    print(f"qm11-shadow: today={today} router={r['tier']} packages={len(r['packages'])}"
          f" causes={','.join(r['causes']) or '-'} owners={owners}")
PY
}

run() {
  [ $# -eq 5 ] || usage
  local repo today_f changed out today work rc
  repo=$(cd "$1" && pwd); today_f=$2; changed=$3; out=$5
  export QM11_BASE=$4
  today=$(sed -n 's/^tier=//p' "$today_f" | head -1)
  [ -n "$today" ] || { echo "tier_route_step: no tier= line in $today_f" >&2; return 1; }
  [ -r "$changed" ] || { echo "tier_route_step: $changed unreadable" >&2; return 1; }
  work=$(mktemp -d); trap 'rm -rf "${work:?}"' RETURN
  cp "$changed" "$work/changed.txt"

  if [ -n "${QM11_INPUT_SET:-}" ]; then
    cp "$QM11_INPUT_SET" "$work/input-set.json"
  else
    [ -n "${GH_TOKEN:-}" ] && [ -n "${GITHUB_REPOSITORY:-}" ] \
      || { emit "$out" "$today" "no GH_TOKEN/GITHUB_REPOSITORY to fetch the nightly input set"; return 0; }
    # shellcheck disable=SC2016 # expands inside in_env
    in_env "$repo" "$work" '
      set -euo pipefail
      api="https://api.github.com/repos/${GITHUB_REPOSITORY}"
      # Bounded: a stalled API must not hold the job to its timeout (quorum finding).
      hdr=(--connect-timeout 10 --max-time 60 -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json")
      # GH-1: two REST calls per run (the list, the download), never a walk over runs.
      url=$(curl -fsS "${hdr[@]}" "$api/actions/artifacts?name=ci-input-set&per_page=20" \
        | jq -r "[.artifacts[] | select((.expired | not) and .workflow_run.head_branch == \"main\")][0].archive_download_url // empty")
      [ -n "$url" ] || exit 3
      # unzip -p: one named member to stdout, so no archive path is ever written to disk.
      curl -fsSL "${hdr[@]}" -o "$W/is.zip" "$url" || exit 3
      unzip -p "$W/is.zip" input-set.json > "$W/input-set.json" || exit 3' || { emit "$out" "$today" "no unexpired ci-input-set artifact from a main nightly"; return 0; }
  fi

  if [ -n "${QM11_METADATA:-}" ]; then
    cp "$QM11_METADATA" "$work/metadata.json"
  else
    # shellcheck disable=SC2016
    in_env "$repo" "$work" 'cargo metadata --format-version 1 --no-deps > "$W/metadata.json"' \
      || { echo "tier_route_step: cargo metadata failed" >&2; return 1; }
  fi

  local why
  # shellcheck disable=SC2016
  if ! why=$(in_env "$repo" "$work" 'bash "$BIN/input_set.sh" check "$W/input-set.json" 2>&1'); then
    emit "$out" "$today" "input set refused: $(printf '%s' "$why" | tail -1 | tr -d '\n')"; return 0
  fi

  local args=(route --input-set "$work/input-set.json" --metadata "$work/metadata.json" --changed "$work/changed.txt")
  set +e
  # shellcheck disable=SC2016
  in_env "$repo" "$work" 'bash "$BIN/input_set.sh" base "$W/input-set.json" . "$QM11_BASE" > "$W/base.log" 2>&1'
  rc=$?
  set -e
  case $rc in
    0) args+=(--base-valid) ;;
    10)
      # INV-BASE failed: the paths changed between the nightly's base and this base join R'.
      # shellcheck disable=SC2016
      in_env "$repo" "$work" 'git diff --no-renames --name-only "$(jq -r .baseSha "$W/input-set.json")" "$QM11_BASE" > "$W/stale.txt"' \
        || { emit "$out" "$today" "INV-BASE failed and the stale range could not be listed"; return 0; }
      args+=(--stale "$work/stale.txt") ;;
    *) emit "$out" "$today" "INV-BASE check faulted (rc=$rc): $(tail -1 "$work/base.log" | tr -d '\n')"; return 0 ;;
  esac
  # Owners (quorum finding): without dep-info an I_dep path outside every package's own
  # dirs is unowned and routes T3, so the line names which regime produced the decision.
  local owners="absent" n wsroot
  if [ -n "${QM11_TARGET_DIR:-}" ] && [ -d "$QM11_TARGET_DIR" ]; then
    wsroot=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["workspace_root"])' "$work/metadata.json")
    python3 "$ROUTER" owners "$wsroot" "$QM11_TARGET_DIR" --metadata "$work/metadata.json" > "$work/owners.json"
    n=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$work/owners.json")
    if [ "$n" -gt 0 ]; then args+=(--owners "$work/owners.json"); owners="depinfo:$n"
    else owners="absent:no-depinfo-in-target-dir"; fi
  fi
  python3 "$ROUTER" "${args[@]}" > "$work/route.json"
  emit "$out" "$today" "" "$work/route.json" "$owners"
}

# ---------------------------------------------------------------------------
# self-test: a fixture repo, input set and metadata; every row names the line it must print.
# ---------------------------------------------------------------------------
self_test() {
  local t fail=0 n=0
  t=$(mktemp -d); trap 'rm -rf "${t:?}"' RETURN
  git -C "$t" init -q -b main repo
  local r="$t/repo"
  # Two members, app -> core, so the R' closure has an edge to walk (quorum finding: one
  # package made the closure row vacuous).
  mkdir -p "$r/crates/core/src" "$r/crates/app/src" "$r/docs"
  echo 'pub fn f() {}' > "$r/crates/core/src/lib.rs"; echo 'fn main() {}' > "$r/crates/app/src/main.rs"
  echo doc > "$r/docs/guide.md"; echo shared > "$r/crates/core/SHARED.txt"
  git -C "$r" add -A; git -C "$r" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm base
  local sha; sha=$(git -C "$r" rev-parse HEAD)
  cat > "$t/meta.json" <<JSON
{"workspace_root":"$r","packages":[
 {"name":"core","manifest_path":"$r/crates/core/Cargo.toml","targets":[{"name":"core"}],"dependencies":[]},
 {"name":"app","manifest_path":"$r/crates/app/Cargo.toml","targets":[{"name":"app"}],"dependencies":[{"name":"core","kind":null}]}]}
JSON
  # SHARED.txt is in I_dep but outside every package's autodiscovery dirs: unowned unless
  # dep-info says who reads it.
  local good='{"schema":"ci-input-set-v1","baseSha":"'"$sha"'","dep":["crates/core/src/lib.rs","crates/app/src/main.rs","crates/core/SHARED.txt"],"read":["crates/core/tests/data/x"],"dirPrefix":[],"absent":[],"config":["Cargo.toml"],"traceSyscalls":["openat","getdents64"],"nightlies":7,"nightlyShas":[]}'
  printf '%s\n' "$good" > "$t/good.json"
  printf '%s\n' "${good/\"nightlies\":7/\"nightlies\":3}" > "$t/few.json"
  printf 'tier=quick\ncrates=core\n' > "$t/today.txt"
  printf 'docs/guide.md\n' > "$t/docs.txt"; printf 'crates/core/src/lib.rs\n' > "$t/code.txt"
  printf 'crates/app/src/main.rs\n' > "$t/leaf.txt"; printf 'Cargo.toml\n' > "$t/cfg.txt"
  printf 'crates/core/SHARED.txt\n' > "$t/shared.txt"
  mkdir -p "$t/target/debug/deps"
  printf '%s: %s %s\n' "$t/target/debug/deps/app-0123abcd" "$r/crates/app/src/main.rs" "$r/crates/core/SHARED.txt" \
    > "$t/target/debug/deps/app-0123abcd.d"

  row() { # row <name> <want-regex> <input-set|""> <changed> [base] [target-dir]
    local name=$1 want=$2 is=$3 ch=$4 base=${5:-main} td=${6:-} got
    n=$((n + 1))
    got=$(cd "$t" && env -u IMAGE -u GH_TOKEN -u QM11_TARGET_DIR QM11_METADATA="$t/meta.json" \
          ${is:+QM11_INPUT_SET=$is} ${td:+QM11_TARGET_DIR=$td} \
          bash "$SELF" "$r" "$t/today.txt" "$ch" "$base" "$t/out.json" 2>&1) || got="rc=$? $got"
    if printf '%s' "$got" | grep -Eq "$want"; then echo "PASS  $name"
    else echo "FAIL  $name  (got: $got)"; fail=$((fail + 1)); fi
  }
  row "docs-only diff, base valid -> router T0" 'today=quick router=T0 packages=0' "$t/good.json" "$t/docs.txt"
  row "core code -> selective over R' = {core, app}: the closure walked the rdep" \
      'router=selective packages=2 causes=- owners=absent$' "$t/good.json" "$t/code.txt"
  row "leaf code -> selective over {app} alone: the closure does not over-walk" \
      'router=selective packages=1 ' "$t/good.json" "$t/leaf.txt"
  row "a C path -> T3 naming cause C" 'router=T3 .* causes=C ' "$t/good.json" "$t/cfg.txt"
  row "an unowned I_dep path without dep-info -> T3, owners named absent" \
      'router=T3 .* causes=unowned owners=absent$' "$t/good.json" "$t/shared.txt"
  row "the same path WITH dep-info -> selective over its reader, owners named depinfo" \
      'router=selective packages=1 causes=- owners=depinfo:[0-9]+$' "$t/good.json" "$t/shared.txt" main "$t/target"
  row "a target dir with no dep-info is named, never silently absent" \
      'owners=absent:no-depinfo-in-target-dir$' "$t/good.json" "$t/code.txt" main "$t/repo/docs"
  row "fewer than 7 nightlies -> a NAMED abstention, never a route" \
      'router=abstained why=input set refused: .*nightlies 3 < 7' "$t/few.json" "$t/code.txt"
  row "no input set and no token -> a named abstention" 'router=abstained why=no GH_TOKEN' "" "$t/code.txt"
  # A base ref is data, never code: a quote and a command substitution in it run nothing.
  row "a hostile base ref is a named abstention" \
      'router=abstained why=INV-BASE' "$t/good.json" "$t/code.txt" "main'\$(touch $t/PWNED)'\"\$(touch $t/PWNED)\""
  n=$((n + 1))
  if [ -e "$t/PWNED" ]; then echo "FAIL  the hostile base ref executed"; fail=$((fail + 1))
  else echo "PASS  the hostile base ref executed nothing"; fi
  # INV-BASE: a commit after the nightly base that touches I makes the base stale; its path joins R'.
  echo 'pub fn g() {}' >> "$r/crates/core/src/lib.rs"
  git -C "$r" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qam moved
  row "INV-BASE fails -> docs diff is NOT T0; the stale core path decides R' = {core, app}" \
      'router=selective packages=2' "$t/good.json" "$t/docs.txt"
  # The out.json is what a reader of the artifact gets: it must carry the same decision.
  n=$((n + 1))
  if python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["router"]["tier"]=="selective" and d["abstained"] is None and d["owners"]=="absent" else 1)' "$t/out.json"; then
    echo "PASS  out.json carries the decision the line printed"
  else echo "FAIL  out.json carries the decision the line printed"; fail=$((fail + 1)); fi
  n=$((n + 1))
  if bash "$SELF" "$r" "$t/nonexistent" "$t/docs.txt" main "$t/out.json" >/dev/null 2>&1; then
    echo "FAIL  a missing today-tier file is a fault, not a route"; fail=$((fail + 1))
  else echo "PASS  a missing today-tier file is a fault, not a route"; fi
  echo "tier_route_step self-test: $((n - fail))/$n rows"
  [ "$fail" -eq 0 ]
}

case "${1:-}" in
  --self-test) self_test ;;
  -h|--help|"") usage ;;
  *) run "$@" ;;
esac
