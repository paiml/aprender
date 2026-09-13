#!/usr/bin/env bash
# check_pv_bin_resolution.sh - scripts/pv_bin.sh must hand back the `pv` THIS
# checkout builds, and must REFUSE anything else -- including another
# worktree's build of the same commit, which reports the same version string
# (#2745).
#
# WHY A SECOND GUARD, WHEN check_apr_bin_pinned.sh ALREADY COVERS pv.
# That guard is a text scan: since #2552 it carries BARE_PV/PATHRES_PV classes
# and proves no surface SPELLS a bare `pv`. It says nothing about whether the
# resolver every surface now depends on actually works. A repo can be 100%
# pinned to a resolver that silently returns the wrong binary and every text
# scan stays green.
#
# The repo already draws exactly this line for `apr` -- check_apr_bin_pinned.sh
# scans, check_apr_bin_resolution.sh runs the resolver -- and pv had only the
# scanning half. pv_bin.sh is now on the release-certification path
# (scripts/dogfood_surfaces.sh, Makefile `contracts`), so the half that runs it
# is the half that matters: a stale pv reported 253 test refs / 51 missing where
# the HEAD build reported 371 / 27 on the same tree in the same second (#2552).
#
# Every assertion is paired with a control in the opposite direction. A refusal
# check that refuses everything is worse than none, so each REFUSE case is
# followed by an ACCEPT case differing in exactly the property under test.
#
# The stale binary is SYNTHESIZED, never found. Writing this against the dev
# box's real ~/.cargo/bin/pv (0.49.0 today) would pass here and vacuously pass
# everywhere else, CI included, where no stale pv exists.

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

# Absolute, because the fixture rows in section 4 source this file from a
# throwaway checkout in /tmp.
PV_BIN_SH="$(pwd)/scripts/pv_bin.sh"

fails=0
TMP=$(mktemp -d)
cleanup() {
    if [ -n "${TMP:-}" ] && [ "$TMP" != / ] && [ -d "$TMP" ]; then
        rm -rf "$TMP"
    fi
}
trap cleanup EXIT

note() { printf 'OK  %s\n' "$*"; }
bad()  { printf 'FAIL: %s\n' "$*" >&2; fails=$((fails + 1)); }

printf '=== pv_bin.sh must resolve the HEAD-built pv (check_pv_bin_resolution.sh) ===\n'

# The identity marker `pv --version` must carry (#2559). Not decoration: the
# operator settled 2026-08-21 that this binary KEEPS the name `pv` even though
# pv(1) the pipe viewer, the crates.io `pv` crate and (until #2553) the aprender
# facade all claim it, which makes this string the whole mitigation.
PV_IDENTITY='(aprender provable-contracts verifier)'

# 0. The marker this file tests must be the one the binary prints and the one
#    pv_bin.sh decides on -- retyping a literal into a guard is how a guard ends
#    up proving a string nobody emits. Same defence as EXTRACTOR_MISMATCH in
#    scripts/check_pv_version_parse.sh.
if ! grep -qF -- "$PV_IDENTITY" crates/aprender-contracts-cli/src/lib.rs; then
    bad "IDENTITY_MISMATCH: this file tests '$PV_IDENTITY' but aprender-contracts-cli does not print it"
fi
# Captured first, then matched from a here-string. `grep -v FILE | grep -qF`
# is the SIGPIPE trap: the second grep exits on its first match, the first
# takes SIGPIPE, and this file's own `pipefail` reports 141 -- read as NO
# MATCH -- although it matched. It is input-size dependent, so it passes until
# pv_bin.sh grows past the pipe buffer, which #2745 nearly did.
PV_BIN_CODE=$(grep -v '^[[:space:]]*#' scripts/pv_bin.sh) || PV_BIN_CODE=''
if ! grep -qF -- "$PV_IDENTITY" <<< "$PV_BIN_CODE"; then
    bad "IDENTITY_MISMATCH: pv_bin.sh does not decide on '$PV_IDENTITY' -- the resolver is back to semver-only"
fi

# ---------------------------------------------------------------------------
# 1. It resolves at all -- under `set -euo pipefail`, deliberately.
#
#    A resolver is sourced by scripts that set their own options;
#    scripts/dogfood-book.sh runs under `set -u`. A resolution test run beneath
#    a lax shell would report success for a library that dies on an unbound
#    variable in its caller.
# Split across lines rather than `if ! ( ...; [ -x "$PV" ] )`: bashrs reads a
# `(` on the same line as a `[` as SC1028, and this repo's shell-lint ratchet is
# shrink-only.
resolve_rc=0
(
    set -euo pipefail
    . scripts/pv_bin.sh || exit 1
    test -x "$PV"
) >"$TMP/resolve.log" 2>&1 || resolve_rc=$?
if [ "$resolve_rc" -ne 0 ]; then
    bad "pv_bin.sh did not resolve an executable pv under 'set -euo pipefail'"
    sed 's/^/      /' "$TMP/resolve.log" >&2
else
    note "resolves under set -euo pipefail"
fi

RESOLVED=$( set -euo pipefail; . scripts/pv_bin.sh >/dev/null 2>&1 || exit 1; printf '%s' "$PV" ) || RESOLVED=''
[ -n "$RESOLVED" ] || bad "pv_bin.sh exported no \$PV"

# The version comes from cargo, never hardcoded, or this file becomes one more
# thing to bump on every release.
CRATE_VERSION=$(cargo metadata --no-deps --format-version 1 2>/dev/null \
    | jq -r '.packages[] | select(.name=="aprender-contracts-cli") | .version')
if [ -z "$CRATE_VERSION" ] || [ "$CRATE_VERSION" = null ]; then
    bad "cargo metadata reported no version for aprender-contracts-cli"
    CRATE_VERSION='<unknown>'
fi

if [ -n "$RESOLVED" ]; then
    # ONE invocation, then every read off the SAME captured text: the semver and
    # the identity have to describe the same binary and the same run.
    got_all=$("$RESOLVED" --version 2>&1)
    got_first=$(awk 'NR==1{print; exit}' <<< "$got_all")
    got_semver=$(awk 'NR==1{print $2; exit}' <<< "$got_all")

    # SEMVER, POSITIONALLY -- not whole-line equality against "pv $CRATE_VERSION".
    # This block used to do exactly that, and it was correct only while the
    # version line WAS the bare name and a semver. #2559 appended an identity
    # suffix on purpose (four things claim the name `pv`, and the operator
    # settled that this binary keeps it, so the version line IS the
    # disambiguation mechanism), the suffix rode along into the compared string,
    # and this guard went RED against a perfectly fresh HEAD build:
    #     FAIL: $PV reports 'pv 0.63.0 (aprender provable-contracts verifier)'
    #           but the crate is 0.63.0
    # Same class as the last-field extractor #2559 already had to replace in
    # pv_bin.sh (scripts/check_pv_version_parse.sh): a guard's parser pinned to a
    # string that another change deliberately rewrote. Position 2 of line 1 is
    # the shape pinned from the other side by
    # crates/aprender-contracts-cli/tests/version_identity.rs
    # (`semver_stays_the_second_field_of_the_first_line`).
    if [ "$got_semver" = "$CRATE_VERSION" ]; then
        note "\$PV reports semver '$got_semver', matching the crate"
    else
        bad "\$PV reports semver '$got_semver' but the crate is $CRATE_VERSION (first line: $got_first)"
    fi

    # IDENTITY, on the same line -- the property #2559 exists to provide, and
    # therefore the property this guard has to assert. A semver match alone is
    # satisfied by pv(1) the pipe viewer, whose versions are 1.x today but whose
    # numbering shares the whole 0.x/1.x space; asserting only the number would
    # bless any binary that happened to collide.
    case "$got_first" in
        *"$PV_IDENTITY"*)
            note "\$PV identifies itself: '$got_first'" ;;
        *)
            bad "\$PV does not carry the identity marker '$PV_IDENTITY'. First --version line: '$got_first'" ;;
    esac
fi

# 1b. It must be cargo's own output for THIS workspace, not something off PATH.
#     This is the assertion that a PATH fallback would fail: a `pv` on PATH is
#     under neither of the two roots below.
#
#     TWO roots since #2745, and this is measured, not defensive. `cargo
#     metadata` answers for the shell ASKING. The dev shell defines a `cargo`
#     function that moves CARGO_TARGET_DIR; a build launched from a
#     `#!/usr/bin/env bash` script does not get that function and lands in
#     <workspace_root>/target. Sourcing pv_bin.sh from the two shells on this
#     box resolved, in the same worktree at the same commit:
#       zsh  -> /mnt/nvme-raid0/targets/aprender/debug/pv
#       bash -> <worktree>/target/debug/pv
#     Asserting only the metadata answer would fail the second one.
TARGET_DIR=$(cargo metadata --no-deps --format-version 1 2>/dev/null | jq -r '.target_directory')
WS_ROOT=$(cargo metadata --no-deps --format-version 1 2>/dev/null | jq -r '.workspace_root')
if [ -n "$RESOLVED" ]; then
    case "$RESOLVED" in
        "$TARGET_DIR"/*) note "\$PV lives under the target dir cargo metadata reports" ;;
        "$WS_ROOT"/target/*) note "\$PV lives under the target dir of this workspace itself" ;;
        *) bad "\$PV ($RESOLVED) is under neither $TARGET_DIR nor $WS_ROOT/target" ;;
    esac
fi

# 1c. And it must be attributable to THIS tree. 1b only proves the resolved
#     path sits in a directory; every worktree of this repo shares that
#     directory, so a path test cannot answer whose build it is. Cargo's own
#     dep-info can: <bin>.d lists the absolute source paths that went into the
#     binary. A resolver that returned a sibling's build would satisfy every
#     other check in this file, because two worktrees at the same workspace
#     version print byte-identical `pv --version` output.
if [ -n "$RESOLVED" ] && [ -f "$RESOLVED.d" ]; then
    ANCHOR=$(cargo metadata --no-deps --format-version 1 2>/dev/null \
        | jq -r '.packages[].targets[] | select(.name == "pv") | select(.kind | index("bin")) | .src_path')
    if [ -z "$ANCHOR" ]; then
        bad "cargo metadata reported no src_path for the bin target named pv"
    elif grep -qF -- "$ANCHOR" "$RESOLVED.d"; then
        note "\$PV was built from THIS tree, per the dep-info cargo wrote beside it"
    else
        bad "\$PV ($RESOLVED) was NOT built from this tree -- $RESOLVED.d does not name $ANCHOR"
    fi
fi

# ---------------------------------------------------------------------------
# 2. MUTATION: a stale pv must be REFUSED, and the refusal must name the
#    mismatch. `pv 0.0.1` is a three-line script, so this tests the same
#    decision on a machine that has never had a stale pv.
mkdir -p "$TMP/stale"
printf '#!/usr/bin/env bash\nprintf "pv 0.0.1\\n"\n' > "$TMP/stale/pv"
chmod +x "$TMP/stale/pv"

if ( set -euo pipefail; PV_BIN="$TMP/stale/pv" . scripts/pv_bin.sh ) >"$TMP/stale.log" 2>&1; then
    bad "pv_bin.sh ACCEPTED a pv reporting 0.0.1 (expected refusal)"
elif grep -q '0\.0\.1' "$TMP/stale.log"; then
    note "refuses a stale pv, and names the version it saw"
else
    bad "refused the stale pv but did not say what it found"
    sed 's/^/      /' "$TMP/stale.log" >&2
fi

# 2b. CONTROL: the SAME override path with a correct binary must be ACCEPTED.
#     Without this, a pv_bin.sh that refused every PV_BIN would pass check 2
#     while being useless.
if [ -n "$RESOLVED" ]; then
    if ( set -euo pipefail; PV_BIN="$RESOLVED" . scripts/pv_bin.sh ) >"$TMP/good.log" 2>&1; then
        note "accepts a correctly-versioned pv through the same override path"
    else
        bad "pv_bin.sh refused a pv reporting the crate version"
        sed 's/^/      /' "$TMP/good.log" >&2
    fi
fi

# ---------------------------------------------------------------------------
# 2c. MUTATION, IDENTITY: a pv at the RIGHT version that prints the bare
#     `pv <semver>` must be REFUSED.
#
#     This is the case the semver assertion cannot see, and it is not
#     hypothetical: `pv 0.63.0` is byte-for-byte what pv(1) the pipe viewer and
#     the crates.io `pv` crate print, which is exactly why #2559 added an
#     identity. A resolver that proved freshness from the number alone would
#     hand the release gate a pipe viewer the moment the two numbers collided,
#     and every version assertion in this file would stay green.
#
#     The version is taken from cargo, so the ONLY difference between this case
#     and its 2d control is the identity suffix -- the property under test.
mkdir -p "$TMP/bare"
printf '#!/usr/bin/env bash\nprintf "pv %s\\n"\n' "$CRATE_VERSION" > "$TMP/bare/pv"
chmod +x "$TMP/bare/pv"

if ( set -euo pipefail; PV_BIN="$TMP/bare/pv" . scripts/pv_bin.sh ) >"$TMP/bare.log" 2>&1; then
    bad "pv_bin.sh ACCEPTED a bare 'pv $CRATE_VERSION' with no identity (that is what pv(1) prints)"
elif grep -qi 'identif' "$TMP/bare.log"; then
    note "refuses a correctly-versioned pv that does not identify itself, and says so"
else
    bad "refused the unidentified pv, but not on identity grounds -- the message never says what was wrong"
    sed 's/^/      /' "$TMP/bare.log" >&2
fi

# 2d. CONTROL for 2c: the SAME synthesized script, the SAME version, plus the
#     identity suffix, must be ACCEPTED. Without it, 2c is also satisfied by a
#     pv_bin.sh that refuses every synthesized binary for some unrelated reason.
mkdir -p "$TMP/identified"
printf '#!/usr/bin/env bash\nprintf "pv %s %s\\n"\n' "$CRATE_VERSION" "$PV_IDENTITY" \
    > "$TMP/identified/pv"
chmod +x "$TMP/identified/pv"

if ( set -euo pipefail; PV_BIN="$TMP/identified/pv" . scripts/pv_bin.sh ) >"$TMP/identified.log" 2>&1; then
    note "accepts the same version once it carries the identity marker"
else
    bad "pv_bin.sh refused a pv at the crate version carrying '$PV_IDENTITY'"
    sed 's/^/      /' "$TMP/identified.log" >&2
fi

# ---------------------------------------------------------------------------
# 3. The library must be OPTION-NEUTRAL. `set` in a SOURCED file mutates the
#    CALLER's shell; apr_bin.sh shipped that bug and killed the nightly six
#    lines in (CLAUDE.md). check_sourced_libs_option_neutral.sh checks the TEXT;
#    this checks the BEHAVIOUR, which is what actually breaks.
opts_before=$( set -o | LC_ALL=C sort | md5sum )
opts_after=$( . scripts/pv_bin.sh >/dev/null 2>&1 || true; set -o | LC_ALL=C sort | md5sum )
if [ "$opts_before" = "$opts_after" ]; then
    note "sourcing pv_bin.sh leaves the shell options of its caller untouched"
else
    bad "sourcing pv_bin.sh MUTATED the shell options of its caller"
fi

# 3b. CONTROL for 3: prove the fingerprint can detect a change at all, or
#     check 3 is satisfied by a broken measurement rather than a well-behaved
#     library. `pipefail` is a trap here -- this file's own `set -euo pipefail`
#     has already set it, so flipping it changes nothing and the control would
#     be vacuous. `noglob` is off, so flipping it is a real change.
opts_mutated=$( set -o noglob; set -o | LC_ALL=C sort | md5sum )
if [ "$opts_before" = "$opts_mutated" ]; then
    bad "the option fingerprint cannot detect 'set -o noglob' - check 3 is vacuous"
else
    note "the option fingerprint detects a deliberate 'set -o noglob'"
fi

# ---------------------------------------------------------------------------
# 4. WHICH TREE (#2745). Everything above varies ONE axis: which VERSION a
#    binary reports. That axis is not sufficient, and for pv it is weaker than
#    it is for apr.
#
#    `apr --version` carries a git sha. `pv --version` carries a SEMVER and an
#    identity marker, and that semver is the WORKSPACE version -- shared,
#    byte-for-byte, by all 40+ git worktrees of this repo. Two trees at the
#    same release print identical version lines, so no assertion above can tell
#    them apart.
#
#    They need telling apart, because they share a target directory. The dev
#    shell defines a `cargo` function exporting
#    CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/$project keyed on the remote URL,
#    so whichever tree built last owns <target>/debug/pv. Measured live from
#    this worktree, straight out of cargo's own records:
#
#      /mnt/nvme-raid0/targets/aprender/debug/pv.d
#          .../.claude/worktrees/wf_9b8aff2c-325-4/crates/aprender-contracts-cli/src/main.rs
#      /mnt/nvme-raid0/targets/aprender/release/pv.d
#          /home/noah/src/aprender/crates/aprender-contracts-cli/src/main.rs
#      ~/.cargo/.crates2.json
#          aprender-contracts-cli 0.63.0
#          (path+file:///home/noah/src/aprender/crates/aprender-contracts-cli)
#
#    None of those three is this tree, and `release/pv` -- the candidate the
#    resolver fell through to whenever `debug/pv` was missing -- is never
#    written by pv_bin.sh's own build at all.
#
#    So the rows below vary a SECOND axis, which tree, and assert the MESSAGE
#    rather than only the exit code. "STALE" tells the reader to rebuild. That
#    is right for an old build of this tree and useless for someone else's, so
#    a guard that only checked "did it refuse" would pass on the wrong
#    diagnosis.
#
#    HOW THE FIXTURE WORKS, and why it fabricates a `cargo`. pv_bin.sh BUILDS
#    before it resolves -- that is its declared freshness authority -- so a
#    fixture cannot simply plant binaries the way check_apr_bin_resolution.sh
#    does: a real `cargo build` would either fail on a stub workspace or
#    overwrite the very artifacts under test. Each fixture therefore puts a
#    three-case `cargo` shim FIRST on PATH: `build` records that it was called
#    and exits 0, `metadata` prints a fabricated document, anything else fails.
#    Everything the resolver then reasons about -- target_directory,
#    workspace_root, the `pv` bin target's src_path, the manifest dir -- is
#    cargo's own answer shape, and every decision under test runs unmodified.
#    Row 4z asserts the shim's build was actually invoked, so a resolver that
#    quietly stopped building could not pass this table.
#
#    The live end-to-end checks above still run against the REAL cargo in the
#    REAL repo, so the fabrication buys discrimination without replacing proof.
#
# Rows:
#   4a. stale debug + good release        -> resolves RELEASE   (evidence, not profile order)
#   4b. good debug + stale release        -> resolves DEBUG
#   4c. both stale                        -> FAILS, STALE
#   4d. no binaries at all                -> FAILS, "no pv binary"
#   4e. unattributable good debug only    -> resolves DEBUG     (degradation guarantee)
#   4f. foreign good release, no own build-> FAILS, WRONG-TREE  (registered falsification)
#   4g. own stale debug                   -> FAILS, STALE       (discrimination)
#   4h. foreign good release + own good debug  -> resolves DEBUG
#   4i. foreign good release + own stale debug -> FAILS, STALE  (own wins the report)
#   4j. binary only in <ws>/target, target_directory elsewhere -> resolves it
#   4k. cargo install from ANOTHER checkout     -> FAILS, WRONG-TREE
#   4l. cargo install from THIS checkout        -> resolves it  (discrimination)
#   4m. cargo install from the REGISTRY         -> FAILS, WRONG-TREE
#   4n. cargo install from a worktree NESTED under this checkout -> FAILS, WRONG-TREE
#
#   4m and 4n have no counterpart in check_apr_bin_resolution.sh and are the
#   two rows pv needs on its own account. apr can leave a registry install
#   unattributed because a registry build embeds `+no-git` and its sha check
#   refuses it; a crates.io `aprender-contracts-cli` at the declared version
#   prints exactly what this tree prints, and `cargo install
#   aprender-contracts-cli` is the advertised install route. And this repo's
#   agent worktrees live UNDER the main checkout, so the workspace-root PREFIX
#   test that would be the obvious way to attribute an install calls every one
#   of them "ours".

FIXTURE_VERSION='9.9.9'

# A throwaway checkout: a git root, a manifest carrying the declared version, a
# metadata document, and a `cargo` shim that serves both.
#
#   $2/$3   what the debug/release binary REPORTS: good | stale | none
#   $4/$5   which TREE it claims: self | other | none (no dep-info at all,
#           which is what `cargo install` leaves behind and what every
#           pre-#2745 candidate looked like)
#   $6      the target_directory `cargo metadata` will report (default
#           <dir>/target)
make_fixture() {
    fx_dir="$1"
    fx_debug="$2"
    fx_release="$3"
    fx_debug_owner="${4:-none}"
    fx_release_owner="${5:-none}"
    fx_td="${6:-$fx_dir/target}"

    mkdir -p "$fx_dir/crates/aprender-contracts-cli/src" "$fx_dir/shim"
    git -C "$fx_dir" init -q

    # The manifest pv_bin_declared_version reads. Written line by line rather
    # than from a heredoc: bashrs parses an embedded heredoc as shell, so TOML
    # `name = "x"` comes back as SC1007 -- a false positive that would fail the
    # shell-lint ratchet.
    {
        printf '[package]\n'
        printf 'name = "aprender-contracts-cli"\n'
        printf 'version = "%s"\n' "$FIXTURE_VERSION"
    } > "$fx_dir/crates/aprender-contracts-cli/Cargo.toml"

    # `cargo metadata --no-deps --format-version 1`, in the shape pv_bin.sh's
    # four jq queries read: target_directory, workspace_root, the src_path of
    # the bin target named pv, and the manifest_path of the package owning it.
    # Built with `jq -n` so the paths are escaped by the same tool that will
    # read them back.
    jq -n \
        --arg td "$fx_td" \
        --arg ws "$fx_dir" \
        --arg mp "$fx_dir/crates/aprender-contracts-cli/Cargo.toml" \
        --arg sp "$fx_dir/crates/aprender-contracts-cli/src/main.rs" \
        --arg ver "$FIXTURE_VERSION" \
        '{target_directory: $td, workspace_root: $ws, packages: [{name: "aprender-contracts-cli", version: $ver, manifest_path: $mp, targets: [{name: "pv", kind: ["bin"], src_path: $sp}]}]}' \
        > "$fx_dir/metadata.json"

    {
        printf '#!/usr/bin/env bash\n'
        printf 'case "${1:-}" in\n'
        printf '    build) printf "build\\n" >> "%s/build-marker"; exit 0 ;;\n' "$fx_dir"
        printf '    metadata) cat "%s/metadata.json"; exit 0 ;;\n' "$fx_dir"
        printf '    *) exit 1 ;;\n'
        printf 'esac\n'
    } > "$fx_dir/shim/cargo"
    chmod +x "$fx_dir/shim/cargo"

    stage_pv "$fx_td/debug/pv" "$fx_debug" "$fx_debug_owner" "$fx_dir"
    stage_pv "$fx_td/release/pv" "$fx_release" "$fx_release_owner" "$fx_dir"
}

# One fabricated pv, plus the dep-info that attributes it to a tree.
stage_pv() {
    sp_path="$1"
    sp_want="$2"
    sp_owner="$3"
    sp_dir="$4"
    if [ "$sp_want" = "none" ]; then
        return 0
    fi
    mkdir -p "${sp_path%/*}"
    sp_ver="$FIXTURE_VERSION"
    if [ "$sp_want" = "stale" ]; then
        sp_ver='0.0.1'
    fi
    # The identity marker is appended on its own line, away from any `[ ]`:
    # bashrs reads the parens of "(aprender ...)" as an unescaped test
    # expression (SC1028) when a string holding them shares a line with a test.
    printf '#!/usr/bin/env bash\n' > "$sp_path"
    printf 'printf "pv %s %s\\n"\n' "$sp_ver" "$PV_IDENTITY" >> "$sp_path"
    chmod +x "$sp_path"

    # Cargo writes <binary>.d beside every binary it links: a make-style rule
    # whose right-hand side is the absolute path of every source that went into
    # it. `other` uses "$sp_dir-other", a sibling path that does NOT contain
    # "$sp_dir/" as a substring, so a resolver matching on the anchor cannot
    # accept it by accident.
    case "$sp_owner" in
        self)  printf '%s: %s/crates/aprender-contracts-cli/src/main.rs\n' "$sp_path" "$sp_dir" > "$sp_path.d" ;;
        other) printf '%s: %s-other/crates/aprender-contracts-cli/src/main.rs\n' "$sp_path" "$sp_dir" > "$sp_path.d" ;;
        *)     rm -f "$sp_path.d" ;;
    esac
}

# The record `cargo install` leaves behind. This is the only thing that
# attributes an installed binary to a tree: cargo builds it in a temp dir and
# copies over just the finished file, so there is no dep-info beside it. A
# second installed bin is listed so the `bins` MEMBERSHIP test is exercised
# rather than a whole-string compare.
write_crates2() {
    wc_out="$1"
    wc_key="$2"
    jq -n --arg k "$wc_key" \
        '{installs: {($k): {version_req: null, bins: ["pv", "pv-helper"], features: [], all_features: false, no_default_features: false, profile: "release", target: "x86_64-unknown-linux-gnu", rustc: "x"}}}' \
        > "$wc_out"
}

# Resolve inside the fixture. CARGO_HOME points at an empty dir unless a row
# stages one, so the `cargo install` fallback cannot supply a candidate the row
# did not ask for. PATH is extended INSIDE the child rather than in the
# assignment prefix, so the `bash` being launched is still found on the real
# PATH. stderr is captured to a file so the DIAGNOSIS can be asserted, and the
# status is read straight off the assignment -- never through a pipe.
resolve_in() {
    ri_dir="$1"
    ri_err="$2"
    ri_ch="${3:-$ri_dir/empty-cargo-home}"
    mkdir -p "$ri_ch"
    (
        cd "$ri_dir" || exit 1
        SHIM_DIR="$ri_dir/shim" CARGO_HOME="$ri_ch" PV_BIN="" PV_BIN_SH_PATH="$PV_BIN_SH" \
        bash -c 'PATH="$SHIM_DIR:$PATH"; export PATH; . "$PV_BIN_SH_PATH" && printf "%s" "$PV"'
    ) 2>"$ri_err"
}

# A temp dir this script is willing to `rm -rf`. bashrs SEC011 is right that an
# unvalidated one is a loaded gun; this is a real finding, not a false positive.
safe_tmpdir() {
    st_dir=$(mktemp -d) || return 1
    case "$st_dir" in
        /tmp/*|/var/folders/*) printf '%s\n' "$st_dir" ;;
        *) return 1 ;;
    esac
}

# Assert on the resolved path AND on the diagnosis.
#   $2 expect     : debug | release | install | FAIL
#   $3 expect_msg : STALE | WRONG-TREE | any literal | "" (do not care)
check_outcome() {
    co_name="$1"
    co_expect="$2"
    co_expect_msg="$3"
    co_got="$4"
    co_rc="$5"
    co_err="$6"

    if [ "$co_expect" = "FAIL" ]; then
        if [ "$co_rc" -eq 0 ] && [ -n "$co_got" ]; then
            bad "$co_name -> accepted $co_got; the guard was made permissive"
            return 0
        fi
    else
        case "$co_got" in
            */"$co_expect"/pv) : ;;
            *)
                bad "$co_name -> got [$co_got], expected the $co_expect binary"
                sed 's/^/      /' "$co_err" >&2
                return 0
                ;;
        esac
    fi

    # The DIAGNOSIS. grep takes the log as a FILE OPERAND, never a pipe: a
    # `producer | grep -q` returns 141 under pipefail when grep exits early and
    # the producer takes SIGPIPE, though grep MATCHED.
    if [ -n "$co_expect_msg" ]; then
        if ! grep -qF -- "$co_expect_msg" "$co_err"; then
            bad "$co_name -> refused, but never said '$co_expect_msg'"
            sed 's/^/      /' "$co_err" >&2
            return 0
        fi
    fi

    # STALE and WRONG-TREE carry different remedies; emitting both is emitting
    # neither, and emitting either on a SUCCESS is a false alarm.
    co_forbidden=''
    case "$co_expect_msg" in
        STALE) co_forbidden='WRONG-TREE' ;;
        WRONG-TREE) co_forbidden='STALE' ;;
    esac
    if [ -n "$co_forbidden" ] && grep -qF -- "$co_forbidden" "$co_err"; then
        bad "$co_name -> said '$co_expect_msg' AND '$co_forbidden'"
        return 0
    fi
    if [ "$co_expect" != "FAIL" ]; then
        if grep -qE 'STALE|WRONG-TREE' "$co_err"; then
            bad "$co_name -> resolved, but still printed a refusal diagnosis"
            sed 's/^/      /' "$co_err" >&2
            return 0
        fi
    fi

    if [ "$co_expect" = "FAIL" ]; then
        note "$co_name -> refused, and diagnosed ${co_expect_msg:-nothing in particular}"
    else
        note "$co_name -> $co_expect"
    fi
}

# debug_ver debug_owner release_ver release_owner expect expect_msg
row() {
    r_name="$1"
    r_debug="$2"
    r_debug_owner="$3"
    r_release="$4"
    r_release_owner="$5"
    r_expect="$6"
    r_msg="${7:-}"
    r_dir=$(safe_tmpdir) || {
        bad "mktemp -d gave an unusable path, refusing to rm -rf it"
        return 0
    }
    make_fixture "$r_dir" "$r_debug" "$r_release" "$r_debug_owner" "$r_release_owner"
    r_rc=0
    r_got=$(resolve_in "$r_dir" "$r_dir/stderr.log") || r_rc=$?
    check_outcome "$r_name" "$r_expect" "$r_msg" "$r_got" "$r_rc" "$r_dir/stderr.log"
    rm -rf "${r_dir:?refusing to rm an empty path}"
}

printf '\n--- which TREE (#2745) ---\n'

# --- axis 1: which version (the pre-#2745 semantics, still true) ------------
row '4a stale debug + good release' stale none good none release
row '4b good debug + stale release' good none stale none debug
row '4c both stale'                 stale none stale none FAIL 'STALE'
row '4d no binaries at all'         none  none none  none FAIL 'no pv binary'
row '4e unattributable good debug'  good  none none  none debug

# --- axis 2: which tree -----------------------------------------------------
# The registered falsification: with the only build belonging to another tree,
# refuse -- and do not call it staleness.
row '4f foreign good release, no own build' none none good  other FAIL 'WRONG-TREE'
# Discrimination. An honest old build of THIS tree must still read STALE, or
# the fix has simply relabelled every failure.
row '4g own stale debug'                    stale self none none FAIL 'STALE'
# The version alone cannot decide: both candidates report the declared version
# and only one of them is this tree's.
row '4h foreign good release + own good debug'  good self good other debug
# A foreign candidate is skipped even when the only own one is old, and the
# report is about the own one.
row '4i foreign good release + own stale debug' stale self good other FAIL 'STALE'

# --- the tree's own target dir ---------------------------------------------
# `cargo metadata` answers for the shell asking. The dev shell's `cargo`
# function moves the target dir; a build launched from a plain bash script does
# not get that wrapper and lands in <workspace_root>/target. Searching only the
# metadata answer makes the tree's own binary invisible while a sibling's is
# found in its place.
row_local_target_dir() {
    rl_name='4j binary only in <ws>/target, target_directory elsewhere'
    rl_dir=$(safe_tmpdir) || {
        bad "mktemp -d gave an unusable path"
        return 0
    }
    # metadata reports <dir>/elsewhere; the binary is staged in <dir>/target.
    make_fixture "$rl_dir" none none none none "$rl_dir/elsewhere"
    mkdir -p "$rl_dir/elsewhere/debug" "$rl_dir/elsewhere/release"
    stage_pv "$rl_dir/target/debug/pv" good self "$rl_dir"
    rl_rc=0
    rl_got=$(resolve_in "$rl_dir" "$rl_dir/stderr.log") || rl_rc=$?
    check_outcome "$rl_name" debug '' "$rl_got" "$rl_rc" "$rl_dir/stderr.log"
    rm -rf "${rl_dir:?refusing to rm an empty path}"
}
row_local_target_dir

# --- the `cargo install` candidate -----------------------------------------
# It leaves no dep-info, so it is attributed from $CARGO_HOME/.crates2.json,
# which records what it was installed FROM. On this box that record reads
# `path+file:///home/noah/src/aprender/crates/aprender-contracts-cli` -- the
# main checkout, not whichever worktree is asking.
row_cargo_install() {
    rc_name="$1"
    rc_key="$2"
    rc_expect="$3"
    rc_msg="${4:-}"
    rc_dir=$(safe_tmpdir) || {
        bad "mktemp -d gave an unusable path"
        return 0
    }
    make_fixture "$rc_dir" none none none none
    rc_ch="$rc_dir/cargo-home"
    mkdir -p "$rc_ch/bin"
    printf '#!/usr/bin/env bash\n' > "$rc_ch/bin/pv"
    printf 'printf "pv %s %s\\n"\n' "$FIXTURE_VERSION" "$PV_IDENTITY" >> "$rc_ch/bin/pv"
    chmod +x "$rc_ch/bin/pv"
    write_crates2 "$rc_ch/.crates2.json" "$rc_key"
    rc_rc=0
    rc_got=$(resolve_in "$rc_dir" "$rc_dir/stderr.log" "$rc_ch") || rc_rc=$?
    if [ "$rc_expect" = "FAIL" ]; then
        check_outcome "$rc_name" FAIL "$rc_msg" "$rc_got" "$rc_rc" "$rc_dir/stderr.log"
    elif [ "$rc_rc" -eq 0 ] && [ "$rc_got" = "$rc_ch/bin/pv" ]; then
        note "$rc_name -> the installed binary"
    else
        bad "$rc_name -> got [$rc_got] rc=$rc_rc, expected $rc_ch/bin/pv"
        sed 's/^/      /' "$rc_dir/stderr.log" >&2
    fi
    rm -rf "${rc_dir:?refusing to rm an empty path}"
}

row_cargo_install '4k cargo install from ANOTHER checkout' \
    'aprender-contracts-cli 9.9.9 (path+file:///some/other/checkout/crates/aprender-contracts-cli)' \
    FAIL 'WRONG-TREE'

# 4m: pv has no git sha, so a crates.io build at the declared version prints
#     byte-for-byte what this tree prints. apr does not need this row; pv does.
row_cargo_install '4m cargo install from the REGISTRY' \
    'aprender-contracts-cli 9.9.9 (registry+https://github.com/rust-lang/crates.io-index)' \
    FAIL 'WRONG-TREE'

# 4l and 4n are built against the fixture's OWN path, so they cannot reuse a
# literal key.
row_cargo_install_relative() {
    rr_name="$1"
    rr_suffix="$2"
    rr_expect="$3"
    rr_msg="${4:-}"
    rr_dir=$(safe_tmpdir) || {
        bad "mktemp -d gave an unusable path"
        return 0
    }
    make_fixture "$rr_dir" none none none none
    rr_ch="$rr_dir/cargo-home"
    mkdir -p "$rr_ch/bin"
    printf '#!/usr/bin/env bash\n' > "$rr_ch/bin/pv"
    printf 'printf "pv %s %s\\n"\n' "$FIXTURE_VERSION" "$PV_IDENTITY" >> "$rr_ch/bin/pv"
    chmod +x "$rr_ch/bin/pv"
    write_crates2 "$rr_ch/.crates2.json" \
        "aprender-contracts-cli $FIXTURE_VERSION (path+file://$rr_dir$rr_suffix)"
    rr_rc=0
    rr_got=$(resolve_in "$rr_dir" "$rr_dir/stderr.log" "$rr_ch") || rr_rc=$?
    if [ "$rr_expect" = "FAIL" ]; then
        check_outcome "$rr_name" FAIL "$rr_msg" "$rr_got" "$rr_rc" "$rr_dir/stderr.log"
    elif [ "$rr_rc" -eq 0 ] && [ "$rr_got" = "$rr_ch/bin/pv" ]; then
        note "$rr_name -> the installed binary"
    else
        bad "$rr_name -> got [$rr_got] rc=$rr_rc, expected $rr_ch/bin/pv"
        sed 's/^/      /' "$rr_dir/stderr.log" >&2
    fi
    rm -rf "${rr_dir:?refusing to rm an empty path}"
}

row_cargo_install_relative '4l cargo install from THIS checkout' \
    '/crates/aprender-contracts-cli' ok

# 4n: this repo's agent worktrees live UNDER the main checkout
#     (/home/noah/src/aprender/.claude/worktrees/wf_...). A workspace-root
#     PREFIX test -- the obvious way to attribute an install -- calls every one
#     of them "ours". Only whole-path equality against the manifest dir of the
#     package owning the `pv` bin target answers this row correctly.
row_cargo_install_relative '4n cargo install from a worktree NESTED under this checkout' \
    '/.claude/worktrees/wt1/crates/aprender-contracts-cli' FAIL 'WRONG-TREE'

# --- 4z: the resolver must still BUILD ---------------------------------------
# Every row above is served by a `cargo` shim. If pv_bin.sh ever stopped
# calling `cargo build`, the table would go on passing while the file's
# declared freshness authority had been removed. The shim records each build
# invocation; this asserts one happened.
row_build_engaged() {
    rb_dir=$(safe_tmpdir) || {
        bad "mktemp -d gave an unusable path"
        return 0
    }
    make_fixture "$rb_dir" good self none none
    rb_rc=0
    rb_got=$(resolve_in "$rb_dir" "$rb_dir/stderr.log") || rb_rc=$?
    if [ "$rb_rc" -eq 0 ] && [ -s "$rb_dir/build-marker" ]; then
        note "4z pv_bin.sh still runs 'cargo build' before it resolves"
    else
        bad "4z pv_bin.sh resolved [$rb_got] rc=$rb_rc without invoking 'cargo build' -- the fixture proves nothing about a resolver that no longer builds"
    fi
    rm -rf "${rb_dir:?refusing to rm an empty path}"
}
row_build_engaged


printf '\n'
if [ "$fails" -gt 0 ]; then
    printf '%s assertion(s) failed.\n' "$fails" >&2
    exit 1
fi
printf 'PASS: pv_bin.sh resolves the pv THIS TREE built, refuses both a stale one\n'
printf '      and another worktree build, names which of the two it found, and\n'
printf '      leaves the shell options of its caller alone.\n'
