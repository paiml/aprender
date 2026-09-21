#!/usr/bin/env bash
# SPEC-HF-PUBLISH-001 v1.0.0 — crates.io cascade publish in dep order.
#
# Usage:
#   scripts/cascade-publish.sh                # run full cascade for current workspace version
#   scripts/cascade-publish.sh --check        # report which crates are still behind
#   scripts/cascade-publish.sh --print-order  # print the publish order this run walks ("ORDER <crate>" lines); no network
#   (--tier N is gone: there are no tiers. The order is DERIVED -- see THE PUBLISH ORDER below.)
#
# Prerequisites:
#   - Workspace version already bumped (Cargo.toml + per-crate Cargo.toml refs)
#   - Cargo.lock regenerated (cargo check --workspace)
#   - All workspace tests pass
#   - A VALID crates.io token in ~/.cargo/credentials.toml (`cargo login`).
#
# AUTH GOTCHA (v0.60.0, ~2h lost): if $CARGO_REGISTRY_TOKEN is exported with a
# STALE value, cargo uses it over the valid credentials file and every upload
# fails "403 authentication failed" (dry-runs pass — they skip upload). Run this
# with `unset CARGO_REGISTRY_TOKEN` unless you KNOW the env token is fresh.
#
# CONFIG GOTCHA (v0.60.0): publish from a tree with NO .cargo/config.toml. The
# dev-only [patch.crates-io] there points siblings at ../<repo> paths that don't
# exist in a git worktree → "failed to load source". Remove it first; the
# consolidated monorepo resolves siblings via in-tree path deps.
#
# This script bypasses `make publish` (which has a known issue with .cargo/config.toml
# stub interaction on some crates per the v0.34.0 cascade observation) and uses
# direct `cargo publish` calls. It also tolerates "already at target version" as success.

set +e  # don't exit on individual crate failures — track each and report at end

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)


TARGET_VERSION=$(grep -E '^version = "' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
[ -z "$TARGET_VERSION" ] && { echo "ERROR: could not detect target version from Cargo.toml"; exit 1; }
echo "Target version: $TARGET_VERSION"

# --------------------------------------------------------------------------
# THE UNIVERSE — read from cargo, across EVERY workspace, never hand-written.
#
# Two facts about each crate that a bare name cannot carry, and that this
# cascade got wrong for the facades in both directions:
#
#   MANIFEST[c]  `cargo publish -p provable-contracts` is not merely wrong from
#                the repo root, it is IMPOSSIBLE — MEASURED, rc=101, "package ID
#                specification `provable-contracts` did not match any packages".
#                An excluded crate must be published by --manifest-path.
#
#   EXPECT[c]    the facades version INDEPENDENTLY of the aprender version line
#                (0.4.0 vs 0.63.0) and that is deliberate (aprender#2546 — these
#                names have no 0.63.0 history). Comparing them against one
#                $TARGET_VERSION would mark them permanently behind and make
#                FINAL VERIFICATION unreachable — a false failure on an
#                append-only registry.
# --------------------------------------------------------------------------
declare -A MANIFEST
declare -A EXPECT
declare -A ROOTWS
UNIVERSE_N=0
while IFS=$'\t' read -r _u_name _u_ver _u_manifest _u_ws; do
  [ -n "$_u_name" ] || continue
  EXPECT[$_u_name]="$_u_ver"
  MANIFEST[$_u_name]="$_u_manifest"
  ROOTWS[$_u_name]="$_u_ws"
  UNIVERSE_N=$((UNIVERSE_N + 1))
done < <(python3 "$REPO_ROOT/scripts/lib/cascade_universe.py" "$REPO_ROOT")
# Vacuity: an enumeration that saw nothing would publish nothing and verify
# nothing, and every row below would read as a clean pass. Make it RED.
if [ "$UNIVERSE_N" -lt 70 ]; then
  echo "ERROR (vacuity): cascade_universe.py enumerated $UNIVERSE_N crate(s), expected 70+."
  echo "The ENUMERATION is broken. Refusing to publish against an unknown universe."
  exit 1
fi
echo "Universe: $UNIVERSE_N publishable crate(s) across all workspaces"

# What version this crate is expected to reach. Falls back to the workspace
# version for anything cargo did not enumerate, so an unknown name still gets a
# defined answer rather than an empty string that matches nothing.
expected_version() {
  echo "${EXPECT[$1]:-$TARGET_VERSION}"
}

# --------------------------------------------------------------------------
# THE PUBLISH ORDER (#3462) -- DERIVED from cargo, never hand-written.
#
# This used to be TIERS[], a hand-maintained table. MEASURED at v0.68.1 it was NOT a dependency
# order: 47 (crate -> non-dev workspace dep) pairs had the dep in a LATER tier (aprender-core in T2
# needs aprender-compute in T6; apr-cli in T10 needs eight T13 crates), and at 225b2a9ab 44 non-dev
# pairs plus 1 versioned dev-dep still were. It only ever published because cascade-drain.sh
# re-ran it until the deferrals stopped ("pass 1 exiting 1 is normal"); under a
# stop-on-first-non-zero rule (operator, 0.68.1) it stops at crate 13.
#
# scripts/lib/cascade_universe.py --order walks the SAME metadata the universe comes from: B before
# A when A needs B through a normal or build dep or a VERSIONED dev-dep (kept in the published
# manifest, resolved on the registry -- PMAT-955). It is acyclic or it refuses (exit 2), and the
# facades land LAST. scripts/check_cascade_covers_all_crates.sh checks the sequence this script
# prints with --print-order, crate by crate, against the graph.
#
# FACADES are the excluded workspaces' crates (crates/facades): a re-export facade resolves its
# upstream FROM THE REGISTRY, not through its path dep, so publishing it first yields a crate that
# cannot compile for anyone. Being after its upstream in ORDER makes that true by construction;
# `facade_upstream_ready` below still refuses to upload a facade until the exact upstream version
# it requires is live on the sparse index.
# --------------------------------------------------------------------------
ORDER_OUT=$(python3 "$REPO_ROOT/scripts/lib/cascade_universe.py" --order --names "$REPO_ROOT") || {
  echo "ERROR: cascade_universe.py --order found no publish order (a dependency cycle, or the enumeration broke). Refusing to publish."
  exit 1
}
mapfile -t ORDER <<< "$ORDER_OUT"
FACADES=()
for _c in "${ORDER[@]}"; do [ "${ROOTWS[$_c]:-}" = "$REPO_ROOT" ] || FACADES+=("$_c"); done
if [ "${#ORDER[@]}" -ne "$UNIVERSE_N" ]; then
  echo "ERROR: the publish order names ${#ORDER[@]} crate(s), the universe $UNIVERSE_N. Refusing to publish."
  exit 1
fi

MODE="${1:-publish}"
# An ALLOWLIST, before anything can upload (#3462). An unrecognized argument used to fall through to
# the publish loop, so a typo (`--chek`) or a removed flag (`--tier 1`) started a real cascade.
case "$MODE" in
  publish|--check|--order-check) : ;;
  --print-order) printf 'ORDER %s\n' "${ORDER[@]}"; exit 0 ;;
  --tier) echo "ERROR: --tier is gone: the publish order is derived, not tiered (#3462). Use --print-order to see it." >&2; exit 2 ;;
  *) echo "ERROR: unknown argument '$MODE' (no argument | --check | --order-check | --print-order). Nothing was published." >&2; exit 2 ;;
esac

# scripts/release/publish-order.txt is the ROOT workspace's part of ORDER, GENERATED and committed:
# publish_strict.sh walks it at T-4, and infra#898's release sensor reads it AT THE TAG as the set
# this cascade publishes (infra-3c, option A). A file that is not this ORDER is stale; publishing
# anyway would make the sensor judge a different set than the one uploaded, so it refuses here,
# before any network call. (--print-order above still prints the derivation, so the guard can
# name the difference.)
ROOT_ORDER=()
for _c in "${ORDER[@]}"; do [ "${ROOTWS[$_c]:-}" = "$REPO_ROOT" ] && ROOT_ORDER+=("$_c"); done
if [ "$(printf '%s\n' "${ROOT_ORDER[@]}")" != "$(cat "$REPO_ROOT/scripts/release/publish-order.txt" 2> /dev/null)" ]; then
  echo "ERROR: scripts/release/publish-order.txt is not the derived root publish order (#3462). Regenerate it: make publish-order" >&2
  echo "       (python3 scripts/lib/cascade_universe.py --order --names --root-only . > scripts/release/publish-order.txt)" >&2
  echo "       Nothing was published." >&2
  exit 1
fi

check_version() {
  local crate=$1
  # crates.io REJECTS requests without a User-Agent (returns an error page, not
  # JSON) — omitting it made this always print "none", which silently broke the
  # --check STATUS report AND the FINAL VERIFICATION gate (every crate looked
  # unpublished). Always send a UA. (v0.60.0 cascade lesson.)
  curl -s -H "User-Agent: aprender-cascade-publish (release automation)" \
    "https://crates.io/api/v1/crates/$crate" 2>/dev/null \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('crate',{}).get('max_version','none'))" 2>/dev/null
}

# Is EXACTLY this version of this crate live? `check_version` answers
# max_version, which cannot express "0.64.0 is up" once 0.65.0 exists, and the
# publish-ORDER precondition below needs the exact question. Uses the sparse
# index (a static CDN, what cargo itself resolves against) rather than the JSON
# API, which rate-limited the v0.61.0 cascade into reporting 0/70.
version_live() {
  local crate=$1 want=$2 n p
  n=${#crate}
  if   [ "$n" -eq 1 ]; then p="1/${crate}"
  elif [ "$n" -eq 2 ]; then p="2/${crate}"
  elif [ "$n" -eq 3 ]; then p="3/${crate:0:1}/${crate}"
  else p="${crate:0:2}/${crate:2:2}/${crate}"
  fi
  local idx
  idx=$(curl -s --retry 3 --retry-delay 2 \
    -H "User-Agent: aprender-cascade-publish (release automation)" \
    "https://index.crates.io/${p}" 2>/dev/null) || idx=''
  grep -qF "\"vers\":\"${want}\"" <<< "$idx"
}

# PUBLISH ORDER AS A PRECONDITION, NOT AS A COMMENT.
#
# `scripts/check_facade_compat.sh` already ends with a `note` saying the order
# is a hard constraint. A note is not a mechanism: nothing stopped tier 14 being
# uploaded first, and nothing would have noticed. Inside THIS tree `upstream`
# resolves through its `path` and every build is green; a consumer installing
# from crates.io resolves it from the REGISTRY. Only the second is what
# `cargo install provable-contracts` does, and a facade published ahead of its
# upstream is a crate that cannot compile for anyone.
#
# The requirement is READ FROM THE FACADE'S OWN MANIFEST rather than assumed to
# equal $TARGET_VERSION, so it stays true if the pin ever legitimately differs.
# A crate with no `upstream =` line (the lib-only signpost facade, which has no
# dependencies at all) is ready by construction — that independence is stated in
# its manifest and is what makes it publishable at any point in the cascade.
# WHICH FACADES MUST HAVE AN UPSTREAM (aprender#2628).
#
# The previous implementation read the requirement with a line-shaped `sed` and
# treated "I could not parse one" as "there is none to order against" -- it
# returned READY. MEASURED: `cargo` accepts the inline and the multi-line
# dependency table as IDENTICAL (`cargo metadata` returns the same
# `('aprender-contracts','upstream','^0.63.0')` for both), while the sed parses
# only the inline spelling. So rewriting
#
#   upstream = { path = "...", version = "0.63.0", package = "aprender-contracts" }
# as
#   [dependencies.upstream]
#   path = "..."
#   version = "0.63.0"
#   package = "aprender-contracts"
#
# -- a semantically null edit that `cargo add` itself can produce and that no
# reviewer would flag -- silently DISARMED this gate, and the cascade would
# upload a facade before its upstream: a crate nobody can compile, on an
# append-only registry.
#
# Two changes close it, and the second matters more than the first:
#   1. Resolve the requirement with `cargo metadata`, the authority on what a
#      manifest MEANS, instead of a regex over one of its spellings.
#   2. Assert POSITIVELY. Absence of evidence must not read as evidence of
#      absence: a facade named here that resolves to no upstream is a FAILURE,
#      not a pass. `provable-contracts-cli` is deliberately absent from the list
#      -- it is the lib-only signpost facade with no [dependencies] at all, which
#      check_facade_compat.sh row R3 enforces from the other direction.
FACADE_EXPECTS_UPSTREAM="provable-contracts provable-contracts-macros"

# Resolve a facade's upstream (name and required version) from cargo itself.
# Echoes "<name> <version>" or nothing. The dependency is identified by its
# RENAME (`upstream`), which is how the manifest refers to it, so this does not
# depend on the dependency's spelling or position in the file.
facade_upstream_of() {
  local manifest=$1 pkg=$2
  cargo metadata --format-version 1 --no-deps --manifest-path "$manifest" 2>/dev/null \
    | PKG="$pkg" python3 -c '
import json, os, sys
try:
    meta = json.load(sys.stdin)
except Exception:
    sys.exit(0)
want = os.environ["PKG"]
for p in meta.get("packages", []):
    if p.get("name") != want:
        continue
    for d in p.get("dependencies", []):
        if d.get("rename") == "upstream":
            req = (d.get("req") or "").lstrip("^~=v ").strip()
            if d.get("name") and req:
                print(d["name"], req)
            sys.exit(0)
' 2>/dev/null
}

facade_upstream_ready() {
  local crate=$1 manifest=${MANIFEST[$1]:-} resolved up_name up_ver expected=0
  [ -n "$manifest" ] || return 0

  case " $FACADE_EXPECTS_UPSTREAM " in *" $crate "*) expected=1 ;; esac

  resolved=$(facade_upstream_of "$manifest" "$crate")
  up_name=${resolved%% *}
  up_ver=${resolved##* }

  if [ -z "$resolved" ] || [ -z "$up_name" ] || [ -z "$up_ver" ]; then
    if [ "$expected" -eq 1 ]; then
      # The gate could not answer the question it exists to answer. Refuse.
      echo "ORDER-FAIL ($crate is declared to require an upstream, but cargo"
      echo "            resolved none from $manifest -- refusing to publish"
      echo "            rather than assuming there is nothing to order against;"
      echo "            see aprender#2628)"
      return 1
    fi
    # Not expected to have one (the lib-only signpost facade): ready by design.
    return 0
  fi

  if version_live "$up_name" "$up_ver"; then
    return 0
  fi
  echo "ORDER-WAIT (needs $up_name $up_ver on crates.io first)"
  return 1
}

publish_crate() {
  local crate=$1
  local want=$(expected_version "$crate")
  local cur=$(check_version "$crate")
  if [ "$cur" = "$want" ]; then
    echo "  ✓ $crate (already $want)"
    return 0
  fi
  echo -n "  → $crate ($cur → $want): "
  # ORDER PRECONDITION. Deliberately a DEFER, not a FATAL: the drain re-runs the
  # cascade, so a facade whose upstream lands in pass N becomes publishable in
  # pass N+1 — the same mechanism every other dependency layer uses. Failing
  # hard here would abort a release that is still making forward progress.
  if ! facade_upstream_ready "$crate"; then
    return 1
  fi
  # An excluded crate has no package ID in the root workspace; `-p` cannot name
  # it at all (rc=101, "did not match any packages"). Select by manifest path
  # for anything cargo enumerated from a workspace other than the root one.
  local sel=(-p "$crate")
  if [ -n "${MANIFEST[$crate]:-}" ] && [ "${ROOTWS[$crate]:-}" != "$REPO_ROOT" ]; then
    sel=(--manifest-path "${MANIFEST[$crate]}")
  fi
  local out
  # No dirty-tree override here: scripts/check_publish_preflight.sh proved the tree
  # clean before the first upload (F-9, PMAT-745); a dirty tree stops the cascade there.
  # cargo's OWN exit status, read from the command: `cmd | tail -6` handed back
  # tail's status, so the verdict below came from six lines of text alone. Two
  # independent review lanes on #2859 reached this line. A zero exit without the
  # `Published` line is not counted as published either -- it is deferred and
  # named, so a drain pass asks again.
  local log rc
  log=$(mktemp "${TMPDIR:-/tmp}/cascade-publish.XXXXXX") || { echo "FATAL-ENV (mktemp failed; nothing uploaded for $crate)"; return 1; }
  cargo publish "${sel[@]}" --locked > "$log" 2>&1; rc=$?
  out=$(tail -6 "$log"); rm -f "$log"
  if [ "$rc" -eq 0 ] && grep -q "Published $crate" <<< "$out" ; then
    echo "✓ PUBLISHED"
    sleep 10  # let crates.io index settle before dependents try to fetch
    return 0
  elif grep -qE "already.*upload|already exists" <<< "$out" ; then
    echo "(already on registry)"
    return 0
  elif [ "$rc" -eq 0 ]; then
    echo "DEFER (cargo publish exited 0 but printed no 'Published $crate' line — not counted as published)"
    return 1
  # Surface the two FATAL classes that are NOT dep-ordering deferrals — a bare
  # "Caused by:" truncation hid both for ~2h in the v0.60.0 cascade:
  #   1. 403 authentication failed — a STALE $CARGO_REGISTRY_TOKEN env var
  #      overrides a valid ~/.cargo/credentials.toml. Fix: `unset
  #      CARGO_REGISTRY_TOKEN` so the file token is used (or refresh the env one).
  #   2. failed to load source for dependency `<sibling>` — a dev-only
  #      [patch.crates-io] in .cargo/config.toml points at ../<repo> paths that
  #      don't exist in a worktree. Fix: remove .cargo/config.toml before publish
  #      (the consolidated monorepo resolves siblings via in-tree path deps).
  elif grep -qiE "403|authentication failed" <<< "$out" ; then
    echo "FATAL-AUTH (403 — unset stale \$CARGO_REGISTRY_TOKEN; use ~/.cargo/credentials.toml)"
    return 1
  elif grep -qiE "failed to load source|no such file or directory" <<< "$out" ; then
    echo "FATAL-CONFIG (dev [patch.crates-io] in .cargo/config.toml — remove it before publish)"
    return 1
  else
    local err
    err=$(echo "$out" | grep -oE "candidate versions found which didn't match:.*$" | head -1)
    [ -z "$err" ] && err=$(echo "$out" | grep -oiE "(error|caused by)[:].*$" | head -1)
    [ -z "$err" ] && err=$(echo "$out" | head -1)
    echo "DEFER ($err)"
    return 1
  fi
}

# ==========================================================================
# THE CLEAN-ROOM GATE (PMAT-3318). The cascade refuses to start unless
# `clean-room.yml` is green on EXACTLY the tag's commit. Fail-closed, no flag,
# no environment bypass.
#
# WHY: clean-room was the first gate named in the release doctrine and was
# enforced nowhere -- none of the publish/cascade scripts referenced it. It ran
# red 8/8 from 2026-09-08 to 2026-09-15 and v0.66.0 and v0.67.0 both shipped
# over it.
#
# WHAT A RUN ACTUALLY TESTED. The workflow lives in paiml/infra, and the job
# does `git clone --depth 1 git@github.com:paiml/aprender.git`: it tests
# aprender's main HEAD AT CLONE TIME. A run's `headSha` is an INFRA commit, so
# comparing it to an aprender tag would compare two repositories. MEASURED
# (infra run 34915682258, job 104212693804): the only record of the aprender
# commit is one line printed by infra's Makefile `_copy-source`,
#
#     2026-09-15T01:06:22.6420228Z     commit:  030d9b14
#
# (`git rev-parse --short HEAD` of the clone). The result CSV has no sha column
# and no per-repo artifact is uploaded. So the gate reads that line from the
# job log, and it reads it STRICTLY: exactly one such line, 7-40 hex chars,
# which must resolve in THIS repository to exactly the tag commit
# (`git rev-parse --verify <abbrev>^{commit}` refuses an ambiguous prefix).
# Zero lines, two different lines, or a changed format all REFUSE -- if infra
# rewords the line, the release stops; it never silently passes.
#
# THE STRUCTURED RECORD (infra#621/#622, PMAT-3318) is that durable fix, and it
# is PREFERRED over the log line wherever it exists. infra's clean-room job now
# runs `Assert the commit under test` immediately after the clone: it refuses to
# build anything that is not the dispatched ref, and it records the tested
# commit as a full 40-char sha in three places -- the results.csv `tested_sha`
# column, the step summary, and the line
#
#     2026-09-16T00:52:43.0000000Z     tested-sha: <40 hex>
#
# in the JOB LOG. The gate reads the log copy, and only that copy, because it is
# the one that is BOTH per-run and reachable: results.csv holds one row per
# repo (the latest run, not this run) and the step summary text is not exposed
# by any REST endpoint. Reading it costs no new API surface -- it is the same
# `actions/jobs/<id>/logs` response the abbreviation is parsed from.
#
# PRECEDENCE, and what each path still has to prove:
#   structured present -> it must be a full 40-char LOWERCASE sha (an
#     abbreviation in that field is a refusal: being unabbreviated is the whole
#     point of the column), the `Assert the commit under test` step must itself
#     have concluded success (a recorded sha whose assertion did not pass is
#     not evidence), and if the old log line is there too the two must agree --
#     a disagreement REFUSES and prints both.
#   structured absent -> the strict log-line parse above, unchanged.
#   neither -> REFUSE, exactly as before.
# No path is looser than the one it replaces: all of them still require exactly
# one `clean-room (aprender)` job, conclusion `success`, and a tested commit
# EQUAL to the tag commit.
#
# Everything the gate cannot prove is a refusal: no run, a run still queued or
# in progress, cancelled or failed, a different sha, gh unauthenticated or
# erroring, any output it cannot parse. The repository, workflow and job name
# are literals, not parameters: an override would be a bypass.
# ==========================================================================

# stdin: `gh run list --json databaseId,status,conclusion,createdAt`.
# $1: the tag commit's committer time (epoch). A run CREATED before the commit
# existed cannot have cloned it. Prints one TSV row per run:
#   <id> <status> <conclusion|none> <createdAt> <after|before>
# Exits 3 on anything that is not that shape.
clean_room_parse_runs() {
  python3 -c '
import json, sys
from datetime import datetime, timezone
try:
    since = int(sys.argv[1])
    runs = json.load(sys.stdin)
    if not isinstance(runs, list):
        raise ValueError("run list is not a JSON array")
    rows = []
    for r in runs:
        rid = r["databaseId"]
        if isinstance(rid, bool) or not isinstance(rid, int):
            raise ValueError("databaseId is not an integer")
        created = datetime.strptime(r["createdAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        side = "after" if created.timestamp() >= since else "before"
        rows.append("%d\t%s\t%s\t%s\t%s" % (rid, str(r["status"]), r.get("conclusion") or "none", r["createdAt"], side))
except Exception as e:
    print("clean-room: unparseable run list: %s" % e, file=sys.stderr)
    sys.exit(3)
for row in rows:
    print(row)
' "$1"
}

# stdin: `gh run view <id> --json jobs`. $1: the exact job name, $2: the exact
# name of the step that asserts the tested commit.
# Prints "<job id> <status> <conclusion|none> <assert state>" for that job, or
# NONE. The assert state is "<status>/<conclusion>", or `absent` when the job
# has no such step (every run before infra#622), or `ambiguous` when it has
# more than one -- both of which the gate refuses to read a structured sha
# through. Exits 3 on malformed input or on more than one job of that name.
clean_room_parse_job() {
  python3 -c '
import json, sys
try:
    jobs = json.load(sys.stdin)["jobs"]
    if not isinstance(jobs, list):
        raise ValueError("jobs is not a JSON array")
    hits = [j for j in jobs if j.get("name") == sys.argv[1]]
    if len(hits) > 1:
        raise ValueError("%d jobs named %r" % (len(hits), sys.argv[1]))
    row = None
    if hits:
        jid = hits[0]["databaseId"]
        if isinstance(jid, bool) or not isinstance(jid, int):
            raise ValueError("job databaseId is not an integer")
        steps = hits[0].get("steps")
        if steps is None:
            state = "absent"
        elif not isinstance(steps, list):
            raise ValueError("steps is not a JSON array")
        else:
            hit = [st for st in steps if st.get("name") == sys.argv[2]]
            if len(hit) > 1:
                state = "ambiguous"
            elif not hit:
                state = "absent"
            else:
                state = "%s/%s" % (str(hit[0].get("status")), hit[0].get("conclusion") or "none")
        row = "%d\t%s\t%s\t%s" % (jid, str(hits[0]["status"]), hits[0].get("conclusion") or "none", state)
except Exception as e:
    print("clean-room: unparseable jobs: %s" % e, file=sys.stderr)
    sys.exit(3)
print(row if row else "NONE")
' "$1" "$2"
}

# stdin: a clean-room job log. Prints the DISTINCT aprender commits the log
# says it copied into the container -- the `_copy-source` line, optionally
# preceded by the Actions timestamp. Nothing else in the log is trusted.
clean_room_tested_abbrevs() {
  tr -d '\r' \
    | { grep -E '^([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z )?    commit:  [0-9a-f]{7,40}$' || true; } \
    | sed -E 's/^.*    commit:  //' \
    | sort -u
}

# stdin: a clean-room job log. Prints the DISTINCT values infra's
# `Assert the commit under test` step recorded as the tested commit -- the
# structured `tested-sha:` field (infra#621/#622), whatever it says. The value
# is captured LOOSELY and validated by the caller ON PURPOSE: a truncated or
# uppercased field has to reach the gate as a malformed record it can refuse by
# name, not vanish and read as "this run predates the structured record".
clean_room_tested_shas() {
  tr -d '\r' \
    | { grep -E '^([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z )?[[:space:]]*tested-sha: .+$' || true; } \
    | sed -E 's/^.*tested-sha: //' \
    | sed -E 's/[[:space:]]+$//' \
    | sort -u
}

# clean_room_gate ROOT TAG -- 0 only when a completed `clean-room (aprender)`
# job, whose log records exactly one tested commit resolving to TAG's commit,
# concluded `success`. Every other outcome prints a REFUSE line and returns 1.
clean_room_gate() {
  local root=$1 tag=$2
  local repo="paiml/infra" workflow="clean-room.yml" job_name="clean-room (aprender)"
  local assert_step="Assert the commit under test"
  local want epoch runs_json runs examined=0 seen=""
  local rid rstatus rconcl rcreated side jobs_json job jid jstatus jconcl jassert
  local log abbrevs n shas ns malformed agrees tested tdisp tsource resolved

  want=$(git -C "$root" rev-parse --verify --quiet "refs/tags/${tag}^{commit}" 2>/dev/null) || want=""
  if [ -z "$want" ]; then
    echo "CLEAN-ROOM REFUSE: tag $tag does not resolve to a commit in $root -- cannot prove clean-room ran on it"
    return 1
  fi
  epoch=$(git -C "$root" log -1 --format=%ct "$want" 2>/dev/null) || epoch=""
  case "$epoch" in
    ''|*[!0-9]*) echo "CLEAN-ROOM REFUSE: looked for $want (tag $tag); could not read its commit time"; return 1 ;;
  esac
  if ! gh auth status >/dev/null 2>&1; then
    echo "CLEAN-ROOM REFUSE: looked for $want (tag $tag); gh is unauthenticated or erroring -- cannot prove clean-room ran on it"
    return 1
  fi
  if ! runs_json=$(gh run list --repo "$repo" --workflow "$workflow" --limit 30 \
        --json databaseId,status,conclusion,createdAt 2>/dev/null); then
    echo "CLEAN-ROOM REFUSE: looked for $want (tag $tag); gh run list --repo $repo --workflow $workflow failed"
    return 1
  fi
  if ! runs=$(clean_room_parse_runs "$epoch" <<< "$runs_json" 2>/dev/null); then
    echo "CLEAN-ROOM REFUSE: looked for $want (tag $tag); the $workflow run list could not be parsed"
    return 1
  fi

  while IFS=$'\t' read -r rid rstatus rconcl rcreated side; do
    [ -n "$rid" ] || continue
    [ "$side" = "after" ] || continue
    examined=$((examined + 1))
    if ! jobs_json=$(gh run view "$rid" --repo "$repo" --json jobs 2>/dev/null); then
      echo "CLEAN-ROOM REFUSE: looked for $want (tag $tag); gh run view $rid failed"
      return 1
    fi
    if ! job=$(clean_room_parse_job "$job_name" "$assert_step" <<< "$jobs_json" 2>/dev/null); then
      echo "CLEAN-ROOM REFUSE: looked for $want (tag $tag); the jobs of run $rid could not be parsed"
      return 1
    fi
    if [ "$job" = "NONE" ]; then
      seen="$seen"$'\n'"  - run $rid ($rstatus/$rconcl, $rcreated): no '$job_name' job"
      continue
    fi
    IFS=$'\t' read -r jid jstatus jconcl jassert <<< "$job"
    if [ "$jstatus" != "completed" ]; then
      seen="$seen"$'\n'"  - run $rid job $jid: $jstatus -- tested sha unknown, conclusion=$jconcl"
      continue
    fi
    if ! log=$(gh api "repos/$repo/actions/jobs/$jid/logs" 2>/dev/null); then
      echo "CLEAN-ROOM REFUSE: looked for $want (tag $tag); could not read the log of run $rid job $jid"
      return 1
    fi
    shas=$(clean_room_tested_shas <<< "$log")
    ns=$(grep -c . <<< "$shas" || true); ns=${ns:-0}
    abbrevs=$(clean_room_tested_abbrevs <<< "$log")
    n=$(grep -c . <<< "$abbrevs" || true); n=${n:-0}
    tested=""; tdisp=""; tsource=""; resolved=""

    if [ "$ns" -gt 1 ]; then
      seen="$seen"$'\n'"  - run $rid job $jid: $ns structured tested-sha record(s) in the log (need exactly 1), conclusion=$jconcl"
      continue
    fi
    if [ "$ns" -eq 1 ]; then
      # The structured record wins -- after it proves it is what it claims.
      malformed=0
      case "${#shas}" in 40) : ;; *) malformed=1 ;; esac
      case "$shas" in *[!0-9a-f]*) malformed=1 ;; esac
      if [ "$malformed" -ne 0 ]; then
        seen="$seen"$'\n'"  - run $rid job $jid: structured tested-sha '$shas' is not a full 40-char lowercase sha, conclusion=$jconcl"
        continue
      fi
      if [ "$jassert" != "completed/success" ]; then
        seen="$seen"$'\n'"  - run $rid job $jid: structured tested-sha $shas but the '$assert_step' step is $jassert (need completed/success), conclusion=$jconcl"
        continue
      fi
      if [ "$n" -gt 1 ]; then
        seen="$seen"$'\n'"  - run $rid job $jid: $n tested-commit record(s) in the log beside structured tested-sha $shas, conclusion=$jconcl"
        continue
      fi
      # The abbreviation is `git rev-parse --short HEAD` of the SAME clone, so
      # agreement is exactly "the log line is a prefix of the structured sha".
      # A quoted glob, not a substring expansion: the latter is SC2299.
      agrees=0
      case "$shas" in "$abbrevs"*) agrees=1 ;; esac
      if [ "$n" -eq 1 ] && [ "$agrees" -ne 1 ]; then
        seen="$seen"$'\n'"  - run $rid job $jid: the two records disagree -- structured tested-sha says $shas, the log line says $abbrevs, conclusion=$jconcl"
        continue
      fi
      tested=$shas; tdisp=$shas; resolved=$shas; tsource="structured"
    else
      if [ "$n" -ne 1 ]; then
        seen="$seen"$'\n'"  - run $rid job $jid: $n tested-commit record(s) in the log (need exactly 1), conclusion=$jconcl"
        continue
      fi
      resolved=$(git -C "$root" rev-parse --verify --quiet "${abbrevs}^{commit}" 2>/dev/null) || resolved=""
      tested=$abbrevs; tdisp="$abbrevs${resolved:+ ($resolved)}"; tsource="log-line"
    fi

    if [ "$resolved" != "$want" ]; then
      seen="$seen"$'\n'"  - run $rid job $jid: tested $tdisp, conclusion=$jconcl [$tsource]"
      continue
    fi
    if [ "$jconcl" = "success" ]; then
      echo "CLEAN-ROOM PROCEED: run $rid job $jid '$job_name' tested $tested = $want (tag $tag), conclusion=success [tested-sha source: $tsource]"
      return 0
    fi
    seen="$seen"$'\n'"  - run $rid job $jid: tested $tested = the tag commit, conclusion=$jconcl [$tsource]"
  done <<< "$runs"

  echo "CLEAN-ROOM REFUSE: looked for $want (tag $tag); no green '$job_name' run tested it ($examined $workflow run(s) created after that commit examined)${seen:- -- found none}"
  return 1
}

# THE GATE (F-9, PMAT-745). Every mode that uploads passes through
# scripts/check_publish_preflight.sh first: clean tree, version from cargo
# metadata, tag at HEAD, HEAD on origin/main, dogfood receipt GO for this commit
# and version. --check and --order-check upload nothing and are not gated. The
# drain re-runs this script per pass, so the gate is re-asked before every pass.
#
# The clean-room gate runs FIRST, before the preflight and before any upload.
# There is no mode, flag or variable that skips it for a publishing run;
# --check REPORTS its verdict (it uploads nothing, so there is nothing to bypass).
case "$MODE" in
  --check|--order-check) : ;;
  *)
    if ! clean_room_gate "$REPO_ROOT" "v$TARGET_VERSION"; then
      echo "⛔ clean-room gate refused (clean-room.yml is not green on exactly the v$TARGET_VERSION commit); nothing was published." >&2
      exit 1
    fi
    if ! bash "$REPO_ROOT/scripts/check_publish_preflight.sh"; then
      echo "⛔ check_publish_preflight.sh refused; nothing was published." >&2
      exit 1
    fi
    ;;
esac

# Backup .cargo/config.toml once (publish needs a clean one without [patch.crates-io]).
# The backup lives OUTSIDE the tree. Beside the config it was an untracked file
# inside the root crate's package directory -- `.cargo/config.toml` is ignored,
# `.cargo/config.toml.cascade-backup` was not -- and with the dirty-tree override
# gone (F-9) `cargo publish` of the root crate would have refused on the file this
# script itself created, after the preflight had already passed R1. Found by the
# cross-vendor review of #2859. Measured: with the backup beside the config,
# `git status --porcelain --untracked-files=all` lists it; with mktemp, nothing.
# A backup that cannot be created is a refusal, not an empty string: with
# CASCADE_CONFIG_BACKUP="" the config would be overwritten and never restored
# (second review of #2859, mktemp-data-loss).
CASCADE_CONFIG_BACKUP=""
if [ -f .cargo/config.toml ]; then
  CASCADE_CONFIG_BACKUP=$(mktemp "${TMPDIR:-/tmp}/cascade-config-backup.XXXXXX") || CASCADE_CONFIG_BACKUP=""
  if [ -z "$CASCADE_CONFIG_BACKUP" ] || [ ! -f "$CASCADE_CONFIG_BACKUP" ]; then
    echo "⛔ could not create a backup of .cargo/config.toml (mktemp failed under ${TMPDIR:-/tmp}); nothing was published." >&2
    exit 2
  fi
  # A failed copy leaves a partial backup that must not be restored later and
  # must not be left behind: it is removed before the refusal.
  cp .cargo/config.toml "$CASCADE_CONFIG_BACKUP" || { rm -f "$CASCADE_CONFIG_BACKUP"; echo "⛔ could not back up .cargo/config.toml; nothing was published." >&2; exit 2; }
  # The restore trap is armed BEFORE the overwrite: between the two there was a
  # window in which an interrupt lost the config (sixth review of #2859).
  trap 'if [ -n "$CASCADE_CONFIG_BACKUP" ] && [ -f "$CASCADE_CONFIG_BACKUP" ]; then cp "$CASCADE_CONFIG_BACKUP" .cargo/config.toml && rm -f "$CASCADE_CONFIG_BACKUP"; fi' EXIT
  echo "# Clean config for cascade publishing" > .cargo/config.toml
fi

# --order-check: run ONLY the publish-order precondition, against the live
# registry, and publish nothing. Two reasons this mode exists rather than the
# order being checked implicitly at publish time:
#
#   1. PRE-FLIGHT. It answers "is it safe to run tier 14 right now" before any
#      upload, which is the question an operator actually has.
#   2. IT MAKES THE PRECONDITION TESTABLE. A branch that only ever runs in the
#      middle of a real cascade cannot be exercised, so it cannot be shown to
#      turn RED — and an ordering rule that has only ever been green is
#      indistinguishable from one that never ran. This mode drives the SAME
#      `facade_upstream_ready` the publish path calls; there is no second copy.
if [ "$MODE" = "--order-check" ]; then
  echo ""
  echo "=== PUBLISH ORDER PRECONDITION (not publishing) ==="
  order_rc=0
  for crate in "${FACADES[@]}"; do
    echo -n "  $crate: "
    if facade_upstream_ready "$crate"; then
      echo "READY"
    else
      order_rc=1
    fi
  done
  if [ $order_rc -eq 0 ]; then
    echo "  ✅ every facade has its upstream live at the version it requires"
  else
    echo "  ⛔ publish the upstream crates FIRST — a facade published ahead of its"
    echo "     upstream resolves the upstream from the REGISTRY and cannot compile."
  fi
  exit $order_rc
fi

if [ "$MODE" = "--check" ]; then
  echo ""
  echo "=== CLEAN-ROOM GATE (verdict reported; a publishing run refuses unless PROCEED) ==="
  clean_room_gate "$REPO_ROOT" "v$TARGET_VERSION" | sed 's/^/  /'
  echo ""
  echo "=== STATUS REPORT (not publishing) ==="
  any_behind=0
  i=0
  for crate in "${ORDER[@]}"; do
    i=$((i + 1))
    want=$(expected_version "$crate")
    cur=$(check_version "$crate")
    if [ "$cur" != "$want" ]; then
      echo "  #$i  $crate: $cur (want $want)"
      any_behind=1
    fi
  done
  [ $any_behind -eq 0 ] && echo "  ✅ ALL $UNIVERSE_N crates at their target version"
  exit 0
fi

# Walk the derived order: every crate's dependencies are already on the registry when its turn
# comes, so a deferral here is a real failure (or a transient), not the order working itself out.
DEFERRED=""
echo ""
echo "=== PUBLISH ORDER (${#ORDER[@]} crates, derived; #3462) ==="
for crate in "${ORDER[@]}"; do
  publish_crate "$crate" || DEFERRED="$DEFERRED $crate"
done

if [ -n "$DEFERRED" ]; then
  echo ""
  echo "=== RETRY ROUND ==="
  STILL_DEFERRED=""
  for crate in $DEFERRED; do
    publish_crate "$crate" || STILL_DEFERRED="$STILL_DEFERRED $crate"
  done
  if [ -n "$STILL_DEFERRED" ]; then
    echo ""
    echo "❌ FAILED to publish:$STILL_DEFERRED"
    exit 1
  fi
fi

echo ""
echo "=== FINAL VERIFICATION ==="
# Iterates the UNIVERSE, not the publish loop's list. Verifying the same list you published is
# circular: when the list was the hand-written TIERS[], a crate missing from it was skipped by the
# publish loop AND by the verify loop, so it could not be reported. That is precisely how the three
# facades went unshipped under a green "✅ ALL crates at $TARGET" -- the check and the omission
# shared one list. The order is now derived from the universe (#3462), and
# scripts/check_cascade_covers_all_crates.sh fails CI when the two disagree.
ALL_OK=1
VERIFIED=0
for crate in $(printf '%s\n' "${!EXPECT[@]}" | sort); do
  want=$(expected_version "$crate")
  cur=$(check_version "$crate")
  VERIFIED=$((VERIFIED + 1))
  if [ "$cur" != "$want" ]; then
    echo "  ✗ $crate at $cur (expected $want)"
    ALL_OK=0
  fi
done

if [ $ALL_OK -eq 1 ]; then
  echo "  ✅ ALL $VERIFIED crates at their target version (root $TARGET_VERSION)"
  echo ""
  echo "Run: cargo install aprender --force && apr --version"
  exit 0
else
  exit 1
fi
