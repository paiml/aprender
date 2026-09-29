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
#   4. `tier_router.py route` over the PR's touched paths.
# An abstention is NAMED on the one line this prints and in <out.json>, never a silent
# default: a shadow that reports nothing looks exactly like a shadow that is not wired.
#
# usage: tier_route_step.sh <repo> <today-tier.txt> <changed.txt> <base-ref> <out.json>
#        tier_route_step.sh --self-test
# env:   QM11_INPUT_SET  an input-set.json on disk: skip the fetch (the self-test, a local run)
#        QM11_METADATA   a `cargo metadata --no-deps` JSON on disk: skip cargo
#        IMAGE           run the fetch, cargo, jq and git steps in this image (the clean-room
#                        host has no cargo); unset -> run them on this host
#        GH_TOKEN, GITHUB_REPOSITORY   the fetch
# Prints ONE line `qm11-shadow: today=<tier> router=<tier>|abstained ...`.
# exit: 0 a decision or a named abstention · 1 a fault · 64 usage
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROUTER="$HERE/tier_router.py"
INPUT_SET_SH="$HERE/input_set.sh"

usage() { sed -n '/^# usage:/,/^# exit:/p' "$0" >&2; exit 64; }

# in_env <repo> <work> <script>: run a bash script with $W = the work dir and the repo as cwd,
# in $IMAGE when set.
in_env() {
  local repo=$1 work=$2 script=$3
  if [ -n "${IMAGE:-}" ]; then
    docker run --rm --user "$(id -u):$(id -g)" \
      -v "$repo:/workspace" -v "$HERE:/qm11-bin:ro" -v "$work:/w" -w /workspace \
      -e W=/w -e BIN=/qm11-bin -e HOME=/w -e GH_TOKEN -e GITHUB_REPOSITORY \
      -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0='*' \
      "$IMAGE" bash -c "$script"
  else
    (cd "$repo" && W="$work" BIN="$HERE" bash -c "$script")
  fi
}

emit() { # emit <out.json> <today> <abstained|""> [route.json]
  local out=$1 today=$2 why=$3 route=${4:-}
  python3 - "$out" "$today" "$why" "$route" <<'PY'
import json, sys
out, today, why, route = sys.argv[1:5]
doc = {"schema": "qm11-shadow-v1", "today": today, "abstained": why or None, "router": None}
if route:
    doc["router"] = json.load(open(route))
json.dump(doc, open(out, "w"), indent=1)
if why:
    print(f"qm11-shadow: today={today} router=abstained why={why}")
else:
    r = doc["router"]
    print(f"qm11-shadow: today={today} router={r['tier']} packages={len(r['packages'])}"
          f" causes={','.join(r['causes']) or '-'}")
PY
}

run() {
  [ $# -eq 5 ] || usage
  local repo today_f changed base out today work rc
  repo=$(cd "$1" && pwd); today_f=$2; changed=$3; base=$4; out=$5
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
      hdr=(-H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json")
      # GH-1: two REST calls per run (the list, the download), never a walk over runs.
      url=$(curl -fsS "${hdr[@]}" "$api/actions/artifacts?name=ci-input-set&per_page=20" \
        | jq -r "[.artifacts[] | select((.expired | not) and .workflow_run.head_branch == \"main\")][0].archive_download_url // empty")
      [ -n "$url" ] || exit 3
      # unzip -p: one named member to stdout, so no archive path is ever written to disk.
      curl -fsSL "${hdr[@]}" -o "$W/is.zip" "$url" && unzip -p "$W/is.zip" input-set.json > "$W/input-set.json" && exit 0
      exit 3' || { emit "$out" "$today" "no unexpired ci-input-set artifact from a main nightly"; return 0; }
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
  in_env "$repo" "$work" "bash \"\$BIN/input_set.sh\" base \"\$W/input-set.json\" . '$base' > \"\$W/base.log\" 2>&1"
  rc=$?
  set -e
  case $rc in
    0) args+=(--base-valid) ;;
    10)
      # INV-BASE failed: the paths changed between the nightly's base and this base join R'.
      # shellcheck disable=SC2016
      in_env "$repo" "$work" "git diff --no-renames --name-only \"\$(jq -r .baseSha \"\$W/input-set.json\")\" '$base' > \"\$W/stale.txt\"" \
        || { emit "$out" "$today" "INV-BASE failed and the stale range could not be listed"; return 0; }
      args+=(--stale "$work/stale.txt") ;;
    *) emit "$out" "$today" "INV-BASE check faulted (rc=$rc): $(tail -1 "$work/base.log" | tr -d '\n')"; return 0 ;;
  esac
  python3 "$ROUTER" "${args[@]}" > "$work/route.json"
  emit "$out" "$today" "" "$work/route.json"
}

# ---------------------------------------------------------------------------
# self-test: a fixture repo, input set and metadata; every row names the line it must print.
# ---------------------------------------------------------------------------
self_test() {
  local t fail=0 n=0
  t=$(mktemp -d); trap 'rm -rf "${t:?}"' RETURN
  git -C "$t" init -q -b main repo
  local r="$t/repo"
  mkdir -p "$r/crates/core/src" "$r/docs"
  echo 'pub fn f() {}' > "$r/crates/core/src/lib.rs"; echo doc > "$r/docs/guide.md"
  git -C "$r" add -A; git -C "$r" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm base
  local sha; sha=$(git -C "$r" rev-parse HEAD)
  cat > "$t/meta.json" <<EOF
{"workspace_root":"$r","packages":[{"name":"core","manifest_path":"$r/crates/core/Cargo.toml","targets":[{"name":"core"}],"dependencies":[]}]}
EOF
  local good='{"schema":"ci-input-set-v1","baseSha":"'"$sha"'","dep":["crates/core/src/lib.rs"],"read":["crates/core/tests/data/x"],"dirPrefix":[],"absent":[],"config":["Cargo.toml"],"traceSyscalls":["openat","getdents64"],"nightlies":7,"nightlyShas":[]}'
  printf '%s\n' "$good" > "$t/good.json"
  printf '%s\n' "${good/\"nightlies\":7/\"nightlies\":3}" > "$t/few.json"
  printf 'tier=quick\ncrates=core\n' > "$t/today.txt"
  printf 'docs/guide.md\n' > "$t/docs.txt"; printf 'crates/core/src/lib.rs\n' > "$t/code.txt"; printf 'Cargo.toml\n' > "$t/cfg.txt"

  row() { # row <name> <want-regex> <input-set|-> <changed> [extra env]
    local name=$1 want=$2 is=$3 ch=$4 got
    n=$((n + 1))
    got=$(env -u IMAGE -u GH_TOKEN QM11_METADATA="$t/meta.json" ${is:+QM11_INPUT_SET=$is} ${5:-} \
          bash "$0" "$r" "$t/today.txt" "$ch" main "$t/out.json" 2>&1) || got="rc=$? $got"
    if printf '%s' "$got" | grep -Eq "$want"; then echo "PASS  $name"
    else echo "FAIL  $name  (got: $got)"; fail=$((fail + 1)); fi
  }
  row "docs-only diff, base valid -> router T0" 'today=quick router=T0 packages=0' "$t/good.json" "$t/docs.txt"
  row "crate code -> selective over R'" 'router=selective packages=1 causes=-$' "$t/good.json" "$t/code.txt"
  row "a C path -> T3 naming cause C" 'router=T3 .* causes=C$' "$t/good.json" "$t/cfg.txt"
  row "fewer than 7 nightlies -> a NAMED abstention, never a route" 'router=abstained why=input set refused: .*nightlies 3 < 7' "$t/few.json" "$t/code.txt"
  row "no input set and no token -> a named abstention" 'router=abstained why=no GH_TOKEN' "" "$t/code.txt"
  # INV-BASE: a commit after the nightly base that touches I makes the base stale; its path joins R'.
  echo 'pub fn g() {}' >> "$r/crates/core/src/lib.rs"
  git -C "$r" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qam moved
  row "INV-BASE fails -> docs diff is NOT T0; the stale path decides R'" 'router=selective packages=1' "$t/good.json" "$t/docs.txt"
  # The out.json is what a reader of the artifact gets: it must carry the same decision.
  n=$((n + 1))
  if python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["router"]["tier"]=="selective" and d["abstained"] is None else 1)' "$t/out.json"; then
    echo "PASS  out.json carries the decision the line printed"
  else echo "FAIL  out.json carries the decision the line printed"; fail=$((fail + 1)); fi
  n=$((n + 1))
  if bash "$0" "$r" "$t/nonexistent" "$t/docs.txt" main "$t/out.json" >/dev/null 2>&1; then
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
