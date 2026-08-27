#!/usr/bin/env bash
# SPEC-HF-PUBLISH-001 v1.0.0 — crates.io cascade publish in dep order.
#
# Usage:
#   scripts/cascade-publish.sh                # run full cascade for current workspace version
#   scripts/cascade-publish.sh --check        # report which crates are still behind
#   scripts/cascade-publish.sh --tier 1       # publish only Tier 1 (leaves)
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

# Tier definitions per SPEC-HF-PUBLISH-001 § crates.io release cascade.
declare -A TIERS
TIERS[1]="apr-format aprender-contracts-macros aprender-quant aprender-gemm-codegen aprender-sparse aprender-solve aprender-rand aprender-fft aprender-image aprender-tensor aprender-cupti"
TIERS[2]="aprender-contracts aprender-core aprender-profile-core aprender-graph aprender-contrastive-data"
TIERS[3]="aprender-profile"
TIERS[4]="aprender-gpu"
TIERS[5]="aprender-cuda-edge aprender-cgp"
TIERS[6]="aprender-compute"
TIERS[7]="aprender-cbtop aprender-ptx-debug aprender-explain"
TIERS[8]="aprender-common aprender-train-common aprender-train aprender-train-lora aprender-train-distill aprender-serve aprender-mcp aprender-data aprender-orchestrate"
TIERS[9]="aprender-present-core aprender-present-layout aprender-present-yaml aprender-present-widgets aprender-present-terminal"
TIERS[10]="apr-cli"
TIERS[11]="aprender"
TIERS[12]="aprender-tsp aprender-monte-carlo aprender-shell"
TIERS[13]="aprender-test-derive aprender-test-lib aprender-test-js-gen aprender-test-cli aprender-test-showcase aprender-zram-core aprender-zram-adaptive aprender-zram-generator aprender-zram-cli aprender-zram aprender-present-test-macros aprender-present-test aprender-present-lib aprender-present-cli aprender-db aprender-rag aprender-viz aprender-registry aprender-distribute aprender-simulate aprender-verify aprender-verify-ml aprender-train-shell aprender-train-inspect aprender-train-bench aprender-train-wasm aprender-contracts-cli aprender-rag-cli"
# TIER 14 — the crates.io compatibility facades (aprender#2559).
#
# THESE WERE NOT HERE, AND THE CASCADE STILL REPORTED SUCCESS. Tiers 1-13 are
# EXACTLY the 70 publishable crates of the root workspace — MEASURED, zero drift
# in either direction — so the table looked complete. `crates/facades/` is a
# second workspace, `exclude`d from the root, and FINAL VERIFICATION below
# iterates TIERS[]: three publishable crates were absent from the loop, so their
# absence read as "✅ ALL crates at $TARGET". See scripts/lib/cascade_universe.py.
#
# LAST TIER ON PURPOSE, AND THE ORDER IS ENFORCED, NOT ASSUMED. A re-export
# facade resolves `aprender-contracts*` FROM THE REGISTRY, not through its path
# dep, so publishing a facade before its upstream yields a crate that cannot
# compile for anyone. Being last makes the order LIKELY; `facade_upstream_ready`
# below makes it TRUE — it refuses to upload a facade until the exact upstream
# version that facade requires is live on the sparse index.
TIERS[14]="provable-contracts provable-contracts-macros provable-contracts-cli"

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

MODE="${1:-publish}"
ONLY_TIER="${2:-}"

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
  curl -s --retry 3 --retry-delay 2 \
    -H "User-Agent: aprender-cascade-publish (release automation)" \
    "https://index.crates.io/${p}" 2>/dev/null \
    | grep -qF "\"vers\":\"${want}\""
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
  out=$(cargo publish "${sel[@]}" --allow-dirty --locked 2>&1 | tail -6)
  if echo "$out" | grep -q "Published $crate"; then
    echo "✓ PUBLISHED"
    sleep 10  # let crates.io index settle before dependents try to fetch
    return 0
  elif echo "$out" | grep -qE "already.*upload|already exists"; then
    echo "(already on registry)"
    return 0
  # Surface the two FATAL classes that are NOT dep-ordering deferrals — a bare
  # "Caused by:" truncation hid both for ~2h in the v0.60.0 cascade:
  #   1. 403 authentication failed — a STALE $CARGO_REGISTRY_TOKEN env var
  #      overrides a valid ~/.cargo/credentials.toml. Fix: `unset
  #      CARGO_REGISTRY_TOKEN` so the file token is used (or refresh the env one).
  #   2. failed to load source for dependency `<sibling>` — a dev-only
  #      [patch.crates-io] in .cargo/config.toml points at ../<repo> paths that
  #      don't exist in a worktree. Fix: remove .cargo/config.toml before publish
  #      (the consolidated monorepo resolves siblings via in-tree path deps).
  elif echo "$out" | grep -qiE "403|authentication failed"; then
    echo "FATAL-AUTH (403 — unset stale \$CARGO_REGISTRY_TOKEN; use ~/.cargo/credentials.toml)"
    return 1
  elif echo "$out" | grep -qiE "failed to load source|no such file or directory"; then
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

# Backup .cargo/config.toml once (publish needs a clean one without [patch.crates-io])
if [ -f .cargo/config.toml ]; then
  cp .cargo/config.toml .cargo/config.toml.cascade-backup
  echo "# Clean config for cascade publishing" > .cargo/config.toml
fi
trap 'if [ -f .cargo/config.toml.cascade-backup ]; then cp .cargo/config.toml.cascade-backup .cargo/config.toml && rm -f .cargo/config.toml.cascade-backup; fi' EXIT

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
  for crate in ${TIERS[14]}; do
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
  echo "=== STATUS REPORT (not publishing) ==="
  any_behind=0
  for tier in $(echo "${!TIERS[@]}" | tr ' ' '\n' | sort -n); do
    for crate in ${TIERS[$tier]}; do
      want=$(expected_version "$crate")
      cur=$(check_version "$crate")
      if [ "$cur" != "$want" ]; then
        echo "  T$tier  $crate: $cur (want $want)"
        any_behind=1
      fi
    done
  done
  [ $any_behind -eq 0 ] && echo "  ✅ ALL $UNIVERSE_N crates at their target version"
  exit 0
fi

# Walk tiers in order
DEFERRED=""
for tier in $(echo "${!TIERS[@]}" | tr ' ' '\n' | sort -n); do
  if [ -n "$ONLY_TIER" ] && [ "$tier" != "$ONLY_TIER" ]; then
    continue
  fi
  echo ""
  echo "=== TIER $tier ==="
  for crate in ${TIERS[$tier]}; do
    publish_crate "$crate" || DEFERRED="$DEFERRED $crate"
  done
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
# Iterates the UNIVERSE, not TIERS[]. Verifying the same list you published is
# circular: a crate missing from TIERS[] was skipped by the publish loop AND by
# the verify loop, so it could not be reported. That is precisely how the three
# facades went unshipped under a green "✅ ALL crates at $TARGET" — the check
# and the omission shared one list. scripts/check_cascade_covers_all_crates.sh
# closes the other half by failing CI when TIERS[] and the universe disagree.
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
