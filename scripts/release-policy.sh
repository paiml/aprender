#!/usr/bin/env bash
# release-policy.sh -- RP-001 (paiml/infra sovereign-release-policy.md §3): CI is the gate, the
# operator is the publisher. Port of paiml/ruchy scripts/release-policy.sh (PMAT-135), for ONT-10
# (#4079): "RP-001 no-publish-in-ci green" is one of that row's release conditions.
#
# Gates. Each prints exactly one line, `PASS <gate>` or `FAIL <gate>: <reason>`; any FAIL exits 1.
# A gate never skips: a missing tool is a FAIL, an unauthenticated gh is a FAIL, a gate over an
# empty set is a FAIL.
#
#   no-publish-in-ci    no non-comment, non --dry-run `cargo publish` anywhere CI can reach
#   no-registry-secret  no CARGO_REGISTRY_TOKEN / CRATES_TOKEN / CARGO_TOKEN / CRATES_IO_TOKEN
#                       secret on the repo or the org (needs an admin gh; operator machine only)
#   receipts-at-tag     the release for --tag carries clean-room, dogfood and fresh-container
#                       receipts naming the tag's 8-char commit SHA (after the publish)
#
# What "CI can reach" means here -- wider than the ruchy original, because aprender's workflows
# call scripts that call scripts, and make targets that call make targets:
#   the workflow files; every `scripts/...` path a reached file names, to a FIXED POINT; every
#   make target a reached file calls (`make [-flags] <t>`, `$(MAKE) <t>`), its prerequisites, and
#   those of the targets they call, to a fixed point. `make publish` stays legal because no
#   workflow reaches it; the day one does, this gate is RED.
#
# Known limits (under-reach; each is absent from this tree, re-check when one appears):
#   a script run through a variable (`"$S"`); a publish split across lines (`cargo \` + newline +
#   `publish`, or a YAML `>` folded scalar); a makefile outside this repo (`make -C ../infra ...`,
#   whose own repo runs its own RP-001); a publisher other than cargo (`cargo release`).
#
# Usage:
#   bash scripts/release-policy.sh [--tag <tag>] [--only <gate>] [--gates-dir <dir>]
#   bash scripts/release-policy.sh --self-test      # every falsifier must turn RED, hermetic
# Exit: 0 all requested gates PASS · 1 a FAIL · 2 usage.
set -uo pipefail

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd)
SELF="${BASH_SOURCE[0]}"

# Whitespace-tolerant: `cargo  publish` (two spaces) defeated a literal match (ruchy #219 quorum).
# `cargo +nightly publish` and `$CARGO publish` / `${CARGO} publish` are the same upload.
PUBLISH_RE='(cargo|\$\{?CARGO\}?)([[:space:]]+\+[A-Za-z0-9._-]+)?[[:space:]]+publish'
PY_PUBLISH_RE="[\"']cargo[\"'][[:space:]]*,[[:space:]]*[\"']publish[\"']"

pass() { printf 'PASS %s\n' "$1"; }
fail() { printf 'FAIL %s: %s\n' "$1" "$2"; }

require_tool() {
    command -v "$2" >/dev/null 2>&1 && return 0
    fail "$1" "$2 not installed"
    return 1
}

# ── no-publish-in-ci ──────────────────────────────────────────────────────────

workflow_files() {  # the real repo's .github/, else every YAML under the root (flat fixtures)
    local root="$1"
    if [ -d "$root/.github" ]; then
        find "$root/.github" -type f \( -name '*.yml' -o -name '*.yaml' \) 2>/dev/null
    else
        find "$root" -maxdepth 3 -type f \( -name '*.yml' -o -name '*.yaml' \) 2>/dev/null
    fi
}

# Every makefile under the root: the git-tracked ones for a repo, all of them for a fixture.
makefiles_of() {
    local root="$1"
    if [ "$(git -C "$root" rev-parse --show-toplevel 2>/dev/null)" = "$root" ]; then
        git -C "$root" ls-files -z | tr '\0' '\n' | grep -E '(^|/)(GNUmakefile|[Mm]akefile|[^/]*\.mk)$' | sed "s|^|$root/|"
    else
        find "$root" -type f \( -name GNUmakefile -o -name Makefile -o -name makefile -o -name '*.mk' \) 2>/dev/null
    fi
}

# awk: is the line a rule header naming target T (`a T b: deps`, not `T := v`)? Sets HDR_REST.
MK_HEADER='function is_header(line, t,   pre, w, n, i) {
    if (line ~ /^[\t#]/ || line !~ /:/ || line ~ /^[^:]*:=/ || line ~ /^[^:=]*=/) return 0
    pre = line; sub(/:.*/, "", pre); n = split(pre, w, /[ \t]+/)
    for (i = 1; i <= n; i++) if (w[i] == t) { HDR_REST = line; sub(/^[^:]*::?/, "", HDR_REST); return 1 }
    return 0 }'

# Recipe lines (file:line:text) of make target $2 in every makefile under $1.
recipe_of() {
    local mk
    makefiles_of "$1" | while IFS= read -r mk; do
        awk -v t="$2" "$MK_HEADER"'
            is_header($0, t) { p = 1; next }
            p && /^\t/ { printf "%s:%d:%s\n", FILENAME, FNR, $0; next }
            p && /^[[:space:]]*$/ { next }
            p { p = 0 }' "$mk"
    done
}

# Prerequisites of make target $2 (the words after the colon on its header lines).
prereqs_of() {
    local mk
    makefiles_of "$1" | while IFS= read -r mk; do
        awk -v t="$2" "$MK_HEADER"'
            is_header($0, t) { r = HDR_REST; sub(/#.*/, "", r); sub(/;.*/, "", r); print r }' "$mk"
    done | tr ' \t|' '\n\n\n' | grep -E '^[A-Za-z0-9_][A-Za-z0-9_.-]*$' || true
}

# Text on stdin without its comment lines: `# ... make publish` in a workflow comment is not a call.
uncommented() { grep -vE '^[[:space:]]*#' || true; }

# scripts/ paths on stdin in an EXECUTION position -- the first word of a command, or the operand
# of an interpreter (bash/sh/zsh/python/source/exec/.) -- so `run_rules "$U" scripts/x.sh` (a guard
# READING a script) is not a run of it. Comment lines are dropped first. LIMIT: a script run
# through a variable (`"$S"` where S=scripts/x.sh) is not followed.
SCRIPT_RE='scripts/[A-Za-z0-9_./-]+'
PREFIX_RE='"?((\$\{?[A-Za-z_]+\}?|\.)/)?'
scripts_run_in() {
    uncommented | grep -oE "((^|[;&|(\`]|run:)[[:space:]]*[@-]*[[:space:]]*|(^|[^A-Za-z0-9_./-])(bash|sh|zsh|python3?|source|exec|\.)[[:space:]]+(-[A-Za-z]+[[:space:]]+)*)$PREFIX_RE$SCRIPT_RE" \
        | grep -oE "$SCRIPT_RE\$" | sort -u || true
}

# make targets named in text on stdin. Per command segment (split on ; & | ) and backticks), after a
# `make` / `$(MAKE)` / `${MAKE}` word: skip flags, the operand of -C -f -I -o -W (and a numeric one
# after -j -l), and VAR=value; every other target-shaped word is a target. `make -C dir t` is `t`,
# not `dir`. `$(...)` is flattened first so `make -C "$(dirname "$X")" t` still yields `t`. A
# segment whose command is echo/printf only TALKS about make. Over-reach is safe (a word that is not
# a target has no recipe); under-reach is the defect.
make_targets_in() {
    awk '
        {
            line = $0
            gsub(/\$\(MAKE\)|\$\{MAKE\}/, "make", line)
            # innermost $(...) first: its INSIDE is scanned too (`X=$(make print-floor)` calls
            # print-floor), then it is flattened so an enclosing `make -C "$(dirname ..)" t` parses
            while (match(line, /\$\([^()]*\)/)) {
                inner = substr(line, RSTART + 2, RLENGTH - 3)
                line = substr(line, 1, RSTART - 1) "SUBST" substr(line, RSTART + RLENGTH)
                line = line ";" inner
            }
            gsub(/["\047]/, " ", line)
            ns = split(line, seg, /[;&|)`]/)
            for (k = 1; k <= ns; k++) {
                n = split(seg[k], w, /[ \t]+/)
                first = ""
                for (i = 1; i <= n; i++) {
                    if (w[i] == "") continue
                    if (first == "" && w[i] !~ /^(-|run:|[@-]+)$/) { first = w[i]; sub(/^[@-]+/, "", first) }
                    if (w[i] != "make" || first ~ /^(echo|printf)$/) continue
                    for (j = i + 1; j <= n; j++) {
                        tk = w[j]
                        if (tk == "") continue
                        if (tk ~ /^-[CfIoW]$/) { j++; continue }
                        if (tk ~ /^-[jl]$/) { if (w[j + 1] ~ /^[0-9.]+$/) j++; continue }
                        if (tk ~ /^-/ || tk ~ /=/ || tk ~ /^[<>0-9]*[<>]/) continue
                        if (tk ~ /^[A-Za-z0-9_][A-Za-z0-9_.-]*$/) print tk
                    }
                    break
                }
            }
        }' || true
}

# Every file and make target CI can reach, to a fixed point. Prints `file <path>` and
# `recipe <file:line:text>` lines.
reach() {
    local root="$1" f t s changed=1 guard=0
    local -A seen_f=() seen_t=()
    local queue_f=() new_t mks
    mks=$(makefiles_of "$root")
    while IFS= read -r f; do [ -n "$f" ] && queue_f+=("$f"); done < <(workflow_files "$root")
    local recipes=""
    while [ "$changed" = 1 ] && [ "$guard" -lt 50 ]; do
        changed=0; guard=$((guard + 1))
        local texts=""
        for f in "${queue_f[@]}"; do
            [ -n "${seen_f[$f]:-}" ] && continue
            seen_f[$f]=1; changed=1
            texts+=$(cat -- "$f" 2>/dev/null)$'\n'
        done
        queue_f=()
        # scripts named by newly read text
        while IFS= read -r s; do
            [ -f "$root/$s" ] && [ -z "${seen_f[$root/$s]:-}" ] && queue_f+=("$root/$s")
        done < <(printf '%s' "$texts" | scripts_run_in)
        # make targets named by newly read text, plus their prerequisites, to a fixed point
        [ -n "$mks" ] || continue
        new_t=$(printf '%s' "$texts" | uncommented | make_targets_in | sort -u)
        while [ -n "$new_t" ]; do
            local next=""
            for t in $new_t; do
                [ -n "${seen_t[$t]:-}" ] && continue
                seen_t[$t]=1; changed=1
                local r; r=$(recipe_of "$root" "$t")
                recipes+="$r"$'\n'
                next+=$(prereqs_of "$root" "$t")$'\n'
                next+=$(printf '%s' "$r" | make_targets_in)$'\n'
                while IFS= read -r s; do
                    [ -f "$root/$s" ] && [ -z "${seen_f[$root/$s]:-}" ] && queue_f+=("$root/$s")
                done < <(printf '%s' "$r" | cut -d: -f3- | scripts_run_in)
            done
            new_t=$(printf '%s' "$next" | grep -v '^$' | sort -u | while IFS= read -r t; do [ -z "${seen_t[$t]:-}" ] && echo "$t"; done)
        done
    done
    for f in "${!seen_f[@]}"; do printf 'file %s\n' "$f"; done
    printf '%s' "$recipes" | grep -v '^$' | sed 's/^/recipe /'
}

# file:line of every publish: matches PUBLISH_RE, not a comment, not --dry-run, not inside an
# echo/printf string (a line that tells a person to publish is not CI publishing).
publish_lines() {
    RE="$PUBLISH_RE" awk 'BEGIN { re = ENVIRON["RE"] }
        {
            rest = $0
            sub(/^[^:]*:[0-9]+:/, "", rest)
            sub(/^[ \t@-]+/, "", rest)
            if (rest ~ /^#/) next
            sub(/[ \t]+#.*$/, "", rest)
            if (rest !~ re) next
            # judge each command, not the line: `cargo publish --dry-run; cargo publish` and
            # `echo x && cargo publish` both publish. Splitting inside a quoted string over-reaches (safe).
            n = split(rest, seg, /;|&&|\|\||\||`|\$\(/)
            hit = 0
            for (k = 1; k <= n; k++) {
                c = seg[k]; sub(/^[ \t({]+/, "", c); sub(/^(then|do|else|if|!)[ \t]+/, "", c)
                if (c !~ re || c ~ /--dry-run/ || c ~ /^(echo|printf)[ \t]/) continue
                hit = 1
            }
            if (!hit) next
            match($0, /^[^:]*:[0-9]+:/)
            print substr($0, 1, RLENGTH - 1)
        }'
}

publish_lines_py() {  # file:line of an argv publish that is not a comment and not --dry-run
    awk '{ rest = $0; sub(/^[^:]*:[0-9]+:/, "", rest); sub(/^[ \t]+/, "", rest)
           if (rest ~ /^#/) next
           # --dry-run excuses the call only inside the same argv list, after "publish"
           if (rest ~ /[Pp][Uu][Bb][Ll][Ii][Ss][Hh]["\047][^]]*["\047]--dry-run["\047]/) next
           match($0, /^[^:]*:[0-9]+:/); print substr($0, 1, RLENGTH - 1) }'
}

publish_hits() {  # $1 = reach output
    local r="$1" files
    files=$(printf '%s\n' "$r" | sed -n 's/^file //p' | grep -v '/release-policy\.sh$' || true)
    {
        [ -n "$files" ] && printf '%s\n' "$files" | grep -v '\.py$' | xargs -r -d '\n' grep -HnE "$PUBLISH_RE" 2>/dev/null | publish_lines
        # python publishes through argv (["cargo", "publish", ...]); its prose about `cargo publish` is not a call
        [ -n "$files" ] && printf '%s\n' "$files" | grep '\.py$' | xargs -r -d '\n' grep -HnE "$PY_PUBLISH_RE" 2>/dev/null | publish_lines_py
        printf '%s\n' "$r" | sed -n 's/^recipe //p' | publish_lines
    } | sort -u
}

gate_no_publish_in_ci() {
    local gate="no-publish-in-ci" root="$1" hits
    [ -d "$root" ] || { fail "$gate" "root $root does not exist"; return 1; }
    [ -n "$(workflow_files "$root")" ] || { fail "$gate" "no workflow files under $root (a gate over nothing is not a pass)"; return 1; }
    local r nf nr
    r=$(reach "$root")
    nf=$(printf '%s\n' "$r" | grep -c '^file ' || true)
    nr=$(printf '%s\n' "$r" | grep -c '^recipe ' || true)
    local unreadable
    unreadable=$(printf '%s\n' "$r" | sed -n 's/^file //p' | while IFS= read -r f; do [ -r "$f" ] || printf '%s ' "$f"; done)
    [ -z "$unreadable" ] || { fail "$gate" "cannot read files CI reaches, so they are unmeasured: $unreadable"; return 1; }
    hits=$(publish_hits "$r")
    if [ -n "$hits" ]; then
        fail "$gate" "CI can reach a crate publish at $(printf '%s' "$hits" | tr '\n' ' ')"
        return 1
    fi
    pass "$gate ($nf files and $nr make recipe lines reachable from CI, none publishes)"
}

# ── no-registry-secret ────────────────────────────────────────────────────────

gate_no_registry_secret() {
    local gate="no-registry-secret" listing found
    require_tool "$gate" gh || return 1
    gh auth status >/dev/null 2>&1 || { fail "$gate" "gh not authenticated"; return 1; }
    listing=$(cd "$ROOT" && gh secret list 2>/dev/null) || { fail "$gate" "gh secret list failed (no admin read on the repository secrets)"; return 1; }
    listing+=$'\n'$(gh secret list --org paiml 2>/dev/null || true)
    found=$(printf '%s\n' "$listing" | grep -Eo 'CARGO_REGISTRY_TOKEN|CRATES_TOKEN|CARGO_TOKEN|CRATES_IO_TOKEN' | sort -u | tr '\n' ' ' || true)
    [ -z "$found" ] || { fail "$gate" "a registry credential is still configured: $found"; return 1; }
    pass "$gate"
}

# ── receipts-at-tag ───────────────────────────────────────────────────────────

missing_receipts() {  # $1 = 8-char SHA; asset names on stdin; prints the missing kinds
    local sha="$1" names missing=""
    names=$(cat)
    printf '%s\n' "$names" | grep -Eq "^clean-room-.*${sha}" || missing+="clean-room "
    printf '%s\n' "$names" | grep -Eq "^(dogfood-.*${sha}|receipt-.*${sha}.*[.]json)" || missing+="dogfood "
    printf '%s\n' "$names" | grep -Eq "^fresh-container-.*${sha}" || missing+="fresh-container "
    printf '%s' "$missing"
}

gate_receipts_at_tag() {
    local gate="receipts-at-tag" tag="$1" sha assets missing
    require_tool "$gate" gh || return 1
    sha=$(git -C "$ROOT" rev-parse "${tag}^{commit}" 2>/dev/null) || { fail "$gate" "tag $tag does not resolve to a commit"; return 1; }
    sha=${sha:0:8}
    assets=$(cd "$ROOT" && gh release view "$tag" --json assets --jq '.assets[].name' 2>/dev/null) || { fail "$gate" "no GitHub release for $tag"; return 1; }
    missing=$(printf '%s\n' "$assets" | missing_receipts "$sha")
    [ -z "$missing" ] || { fail "$gate" "release $tag ($sha) is missing receipts: $missing"; return 1; }
    pass "$gate"
}

# ── self-test: every falsifier turns RED, and the clean control stays GREEN ───

self_test() {
    local d rc=0 got
    d=$(mktemp -d) || return 2
    # shellcheck disable=SC2064
    trap "rm -rf -- '$d'" RETURN
    fx() {  # fx <name> <file> <content>  (content through printf %b)
        mkdir -p "$d/$1/$(dirname -- "$2")"; printf '%b' "$3" > "$d/$1/$2"
    }
    local WF='name: ci\non: push\njobs:\n  b:\n    runs-on: x\n    steps:\n'
    fx clean .github/workflows/ci.yml "$WF      - run: cargo build\n      - run: bash scripts/build.sh\n      - run: make -C . check\n"
    fx clean scripts/build.sh '#!/bin/sh\n# cargo publish is the operator step, never here\ncargo publish --dry-run -p x\necho "then run: cargo publish"\n'
    fx clean Makefile 'check: lint\n\tcargo test\nlint:\n\tcargo clippy\npublish:\n\tcargo publish -p x\n'
    fx direct .github/workflows/r.yml "$WF      - run: cargo publish -p x\n"
    fx twospace .github/workflows/r.yml "$WF      - run: cargo  publish -p x\n"
    fx inscript .github/workflows/r.yml "$WF      - run: bash scripts/rel.sh\n"
    fx inscript scripts/rel.sh '#!/bin/sh\ncargo publish -p x\n'
    fx nested .github/workflows/r.yml "$WF      - run: bash scripts/outer.sh\n"
    fx nested scripts/outer.sh '#!/bin/sh\nbash scripts/inner.sh\n'
    fx nested scripts/inner.sh '#!/bin/sh\ncargo publish --locked -p x\n'
    fx viamake .github/workflows/r.yml "$WF      - run: make release\n"
    fx viamake Makefile 'release:\n\t@cargo publish -p x\n'
    fx makeflags .github/workflows/r.yml "$WF      - run: make -j4 --no-print-directory release\n"
    fx makeflags Makefile 'release:\n\tcargo publish -p x\n'
    fx prereq .github/workflows/r.yml "$WF      - run: make ship\n"
    fx prereq Makefile 'ship: build upload\n\ttrue\nbuild:\n\tcargo build\nupload:\n\tcargo publish -p x\n'
    fx submake .github/workflows/r.yml "$WF      - run: make ship\n"
    fx submake Makefile 'ship:\n\t$(MAKE) upload\nupload:\n\tcargo publish -p x\n'
    fx makescript .github/workflows/r.yml "$WF      - run: make ship\n"
    fx makescript Makefile 'ship:\n\tbash scripts/up.sh\n'
    fx makescript scripts/up.sh '#!/bin/sh\ncargo publish -p x\n'
    # data, not execution: a guard that READS the publish script, a comment naming `make publish`,
    # a python docstring about `cargo publish` -- all reached, none of them publishes
    fx data .github/workflows/ci.yml "$WF      # see #2360 make publish\n      - run: bash scripts/guard.sh\n      - run: python3 scripts/u.py\n"
    fx data scripts/guard.sh '#!/bin/sh\nrun_rules "$U" "${ROOT}/scripts/pub.sh"\ngrep -c TIERS scripts/pub.sh\n'
    fx data scripts/pub.sh '#!/bin/sh\ncargo publish -p x\n'
    fx data scripts/u.py '"""`cargo publish -p x` from the root is impossible."""\nprint(1)\n'
    fx data Makefile 'publish:\n\tcargo publish -p x\n'
    fx pyargv .github/workflows/r.yml "$WF      - run: python3 scripts/p.py\n"
    fx pyargv scripts/p.py 'import subprocess\nsubprocess.run(["cargo", "publish", "-p", "x"])\n'
    fx varpath .github/workflows/r.yml "$WF      - run: timeout 60 bash \"\${GITHUB_WORKSPACE}/scripts/p.sh\"\n"
    fx varpath scripts/p.sh '#!/bin/sh\ncargo publish -p x\n'
    fx cmdpos .github/workflows/r.yml "$WF      - run: |\n          set -e\n          ./scripts/p.sh --all\n"
    fx cmdpos scripts/p.sh '#!/bin/sh\ncargo publish -p x\n'
    fx makeC .github/workflows/r.yml "$WF      - run: make -C \"\$(dirname \"\$MK\")\" -j 4 release\n"
    fx makeC Makefile 'release:\n\tcargo publish -p x\n'
    fx multihdr .github/workflows/r.yml "$WF      - run: make ship\n"
    fx multihdr Makefile 'build ship: ; true\nbuild ship:\n\tcargo publish -p x\n'
    fx toolchain .github/workflows/r.yml "$WF      - run: cargo +nightly publish -p x\n"
    fx cargovar .github/workflows/r.yml "$WF      - run: \\\${CARGO} publish -p x\n"
    fx drycomment .github/workflows/r.yml "$WF      - run: cargo publish -p x  # not a --dry-run\n"
    fx echomake .github/workflows/r.yml "$WF      - run: echo 'release with: make release'\n"
    fx echomake Makefile 'release:\n\tcargo publish -p x\n'
    fx unreadable .github/workflows/r.yml "$WF      - run: bash scripts/p.sh\n"
    fx unreadable scripts/p.sh '#!/bin/sh\ncargo publish -p x\n'
    chmod 000 "$d/unreadable/scripts/p.sh"
    fx cmdsubst .github/workflows/r.yml "$WF      - run: V=\"\$(make -s print-v 2>/dev/null || echo unknown)\"\n"
    fx cmdsubst Makefile 'print-v:\n\tcargo publish -p x\n'
    fx lowermk .github/workflows/r.yml "$WF      - run: make release\n"
    fx lowermk makefile 'release:\n\tcargo publish -p x\n'
    fx drythenreal .github/workflows/r.yml "$WF      - run: cargo publish --dry-run -p x; cargo publish -p x\n"
    fx echothenreal .github/workflows/r.yml "$WF      - run: echo go && cargo publish -p x\n"
    fx pydry .github/workflows/r.yml "$WF      - run: python3 scripts/p.py\n"
    fx pydry scripts/p.py 'import subprocess\nsubprocess.run(["cargo", "publish", "--dry-run"])\nsubprocess.run(["cargo", "publish"]); note = "--dry-run"\n'
    mkdir -p "$d/empty"
    row() {  # row <want PASS|FAIL> <label> <cmd...>
        local want=$1 label=$2; shift 2
        got=$("$@" 2>&1 | head -n 1)
        case "$got" in
            "$want "*) printf '  ok   %-60s %s\n' "$label" "${got%%:*}" ;;
            *) printf '  FAIL %s: want %s, got %s\n' "$label" "$want" "$got"; rc=1 ;;
        esac
    }
    echo "release-policy self-test"
    row PASS 'clean control: comment, --dry-run, echo, unreached make publish' gate_no_publish_in_ci "$d/clean"
    row FAIL 'a workflow step that publishes' gate_no_publish_in_ci "$d/direct"
    row FAIL 'two-space cargo  publish' gate_no_publish_in_ci "$d/twospace"
    row FAIL 'publish inside a script the workflow runs' gate_no_publish_in_ci "$d/inscript"
    row FAIL 'publish two scripts deep' gate_no_publish_in_ci "$d/nested"
    row FAIL 'publish in a make target the workflow calls (@-prefixed)' gate_no_publish_in_ci "$d/viamake"
    row FAIL 'make with flags before the target' gate_no_publish_in_ci "$d/makeflags"
    row FAIL 'publish in a prerequisite of the called target' gate_no_publish_in_ci "$d/prereq"
    row FAIL 'publish in a $(MAKE) sub-target' gate_no_publish_in_ci "$d/submake"
    row FAIL 'publish in a script a make recipe runs' gate_no_publish_in_ci "$d/makescript"
    row PASS 'data refs: a guard reading pub.sh, a comment, a py docstring' gate_no_publish_in_ci "$d/data"
    row FAIL 'python argv ["cargo", "publish"]' gate_no_publish_in_ci "$d/pyargv"
    row FAIL 'bash "${VAR}/scripts/p.sh" under timeout' gate_no_publish_in_ci "$d/varpath"
    row FAIL './scripts/p.sh as a command in a run block' gate_no_publish_in_ci "$d/cmdpos"
    row FAIL 'make -C "$(dirname ..)" -j 4 release: the target, not the dir' gate_no_publish_in_ci "$d/makeC"
    row FAIL 'a target on a multi-target rule header' gate_no_publish_in_ci "$d/multihdr"
    row FAIL 'cargo +nightly publish' gate_no_publish_in_ci "$d/toolchain"
    row FAIL '${CARGO} publish' gate_no_publish_in_ci "$d/cargovar"
    row FAIL '--dry-run only inside a trailing comment' gate_no_publish_in_ci "$d/drycomment"
    row PASS 'echo that names make release is not a call' gate_no_publish_in_ci "$d/echomake"
    if [ "$(id -u)" != 0 ]; then
        row FAIL 'a reached file it cannot read is unmeasured, not clean' gate_no_publish_in_ci "$d/unreadable"
    fi
    row FAIL 'X=$(make -s print-v): a make call inside a command substitution' gate_no_publish_in_ci "$d/cmdsubst"
    row FAIL 'a lowercase makefile' gate_no_publish_in_ci "$d/lowermk"
    row FAIL 'cargo publish --dry-run; cargo publish' gate_no_publish_in_ci "$d/drythenreal"
    row FAIL 'echo go && cargo publish' gate_no_publish_in_ci "$d/echothenreal"
    row FAIL 'python: a real argv publish next to an unrelated "--dry-run" string' gate_no_publish_in_ci "$d/pydry"
    row FAIL 'no workflow files: a gate over nothing' gate_no_publish_in_ci "$d/empty"
    row FAIL 'a missing root' gate_no_publish_in_ci "$d/no-such-dir"
    row FAIL 'no-registry-secret on a missing tool' require_tool no-registry-secret gh-does-not-exist
    local S=deadbeef
    if [ -z "$(printf 'clean-room-%s.log\ndogfood-receipt-%s.json\nfresh-container-%s.log\n' $S $S $S | missing_receipts $S)" ]; then
        echo "  ok   receipts-at-tag: all three receipts at the SHA                PASS"
    else echo "  FAIL receipts-at-tag: a complete asset list reported missing"; rc=1; fi
    if [ -z "$(printf 'clean-room-%s.log\nreceipt-apr-%s.json\nfresh-container-%s.log\n' $S $S $S | missing_receipts $S)" ]; then
        echo "  ok   receipts-at-tag: receipt-*.json stands in for dogfood            PASS"
    else echo "  FAIL receipts-at-tag: receipt-<sha>.json was not accepted as the dogfood receipt"; rc=1; fi
    if [ -n "$(printf 'clean-room-%s.log\ndogfood-receipt-%s.json\n' $S $S | missing_receipts $S)" ]; then
        echo "  ok   receipts-at-tag: fresh-container stripped                       FAIL"
    else echo "  FAIL receipts-at-tag: a stripped asset list passed"; rc=1; fi
    if [ -n "$(printf 'clean-room-0badc0de.log\ndogfood-0badc0de.json\nfresh-container-0badc0de.log\n' | missing_receipts $S)" ]; then
        echo "  ok   receipts-at-tag: receipts naming another SHA                    FAIL"
    else echo "  FAIL receipts-at-tag: receipts for another SHA passed"; rc=1; fi
    # MUTANT: the publish detector made blind must let the direct fixture through.
    local mut="$d/mut.sh"
    sed 's/^PUBLISH_RE=.*/PUBLISH_RE=NEVER_MATCHES_ANYTHING/' "$SELF" > "$mut"
    if cmp -s "$mut" "$SELF"; then echo "  FAIL mutant not built: the PUBLISH_RE anchor moved"; rc=1
    elif bash "$mut" --gates-dir "$d/direct" --only no-publish-in-ci >/dev/null 2>&1; then
        echo "  ok   mutant (blind PUBLISH_RE) lets the publish through: the detector is load-bearing"
    else echo "  FAIL mutant still refuses: something other than PUBLISH_RE is judging"; rc=1; fi
    mutant() {  # mutant <sed expr> <fixture> <what it blinds>
        sed "$1" "$SELF" > "$mut"
        if cmp -s "$mut" "$SELF"; then echo "  FAIL mutant '$1' not built: the anchor moved"; rc=1
        elif bash "$mut" --gates-dir "$d/$2" --only no-publish-in-ci >/dev/null 2>&1; then
            echo "  ok   mutant ($3) lets the $2 publish through: it is load-bearing"
        else echo "  FAIL mutant ($3) still refuses $2: something else is judging"; rc=1; fi
    }
    mutant "s/^SCRIPT_RE=.*/SCRIPT_RE='NO_SUCH_PATH'/" nested 'blind script reach'
    mutant 's/^        new_t=\$(printf .%s. "\$texts" | uncommented | make_targets_in | sort -u)$/        new_t=/' viamake 'blind make reach'
    if [ "$rc" = 0 ]; then pass self-test; else fail self-test "a falsifier did not turn RED"; fi
    return "$rc"
}

# ── driver ────────────────────────────────────────────────────────────────────

main() {
    local tag="" only="" gates_dir="" rc=0
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --tag) tag=${2:-}; shift 2 ;;
            --only) only=${2:-}; shift 2 ;;
            --gates-dir) gates_dir=${2:-}; shift 2 ;;
            --self-test) self_test; return $? ;;
            -h|--help) sed -n '2,27p' "$SELF"; return 0 ;;
            *) printf 'FAIL release-policy: unknown argument %s\n' "$1"; return 2 ;;
        esac
    done
    case "$only" in ""|no-publish-in-ci|no-registry-secret|receipts-at-tag) ;;
        *) printf 'FAIL release-policy: unknown gate %s\n' "$only"; return 2 ;; esac
    if [ -z "$only" ] || [ "$only" = no-publish-in-ci ]; then gate_no_publish_in_ci "${gates_dir:-$ROOT}" || rc=1; fi
    if [ -z "$only" ] || [ "$only" = no-registry-secret ]; then gate_no_registry_secret || rc=1; fi
    if [ "$only" = receipts-at-tag ] && [ -z "$tag" ]; then fail receipts-at-tag "--only receipts-at-tag needs --tag"; rc=1; fi
    if [ -n "$tag" ] && { [ -z "$only" ] || [ "$only" = receipts-at-tag ]; }; then gate_receipts_at_tag "$tag" || rc=1; fi
    return "$rc"
}

main "$@"
