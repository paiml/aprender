#!/usr/bin/env bash
# #5762: contracts_resolve.sh proves the verifier BEFORE it drops the conflicted generated files, so a pv
# that will not resolve, or a stale one, leaves the merge as it was. pv_bin.sh, the gate and readme_sync are stubs.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=$here/../contracts_resolve.sh
W=$(mktemp -d); trap 'rm -rf "${W:?}"' EXIT
fail=0
chk() { [ "$2" = "$3" ] || { fail=1; echo "FAIL: $1: got '$2', want '$3'"; }; }

# mk <name> [src]: a repo whose merge conflicts on contracts/contracts.nt (and on a.txt when "src").
mk() {
    local d=$W/$1; mkdir -p "$d/scripts" "$d/contracts" "$d/crates/aprender-contracts/src" "$d/bin"
    cd "$d"; git init -q -b main; git config user.email t@t; git config user.name t; git config core.hooksPath "$d/.nohooks"
    cp "$src" scripts/contracts_resolve.sh
    printf 'x\n' > crates/aprender-contracts/src/lib.rs; printf "d\n" > crates/aprender-contracts/README.md; printf 'base\n' > contracts/contracts.nt; printf 'base\n' > a.txt; mkdir contracts/witness; printf 's\n' > contracts/shapes.ttl; printf '{}\n' > contracts/witness/w.json; printf 'r\n' > README.md
    printf '#!/bin/sh\n' > bin/pv; chmod +x bin/pv
    printf '#!/bin/sh\necho sat >> %s/calls\n' "$d" > bin/pv-sat; chmod +x bin/pv-sat
    printf '[ -z "${PVFAIL:-}" ] || return 1\nPV=%s/bin/pv; export PV\n' "$d" > scripts/pv_bin.sh
    printf '#!/bin/sh\necho regen >> %s/calls\necho n > contracts/contracts.nt\n' "$d" > scripts/contracts_gate.sh
    printf '#!/bin/sh\necho readme >> %s/calls\n' "$d" > scripts/readme_sync.sh
    printf 'calls\nbin/\n' > .gitignore
    git add -A; git commit -q -m base
    git checkout -q -b other; printf 'theirs\n' > contracts/contracts.nt
    [ "${2:-}" = src ] && printf 'theirs\n' > a.txt
    git commit -qam theirs; git checkout -q main
    printf 'ours\n' > contracts/contracts.nt
    [ "${2:-}" = src ] && printf 'ours\n' > a.txt
    git commit -qam ours; git merge other >/dev/null 2>&1 || true
}
run() { local rc=0; bash scripts/contracts_resolve.sh >/dev/null 2>&1 || rc=$?; echo "$rc"; }

mk fresh; touch bin/pv bin/pv-sat
chk "fresh pv resolves" "$(run)" 0
chk "fresh: nothing left unmerged" "$(git diff --name-only --diff-filter=U | wc -l)" 0
chk "fresh: regen, sat, readme all ran" "$(tr '\n' ' ' < calls)" "regen sat readme "

mk nobuild; export PVFAIL=1
chk "pv_bin failure refuses" "$(run)" 1
chk "pv_bin failure: conflict file intact" "$(grep -c '<<<<<<<' contracts/contracts.nt)" 1
chk "pv_bin failure: still unmerged" "$(git diff --name-only --diff-filter=U)" contracts/contracts.nt
unset PVFAIL

mk stale; touch -d 2000-01-01 bin/pv bin/pv-sat
chk "stale pv refuses" "$(run)" 1
chk "stale: conflict file intact" "$(grep -c '<<<<<<<' contracts/contracts.nt)" 1
chk "stale: nothing regenerated" "$([ -f calls ] && echo ran || echo none)" none

mk docs; touch -d 1999-01-01 crates/aprender-contracts/src/lib.rs; touch -d 2000-01-01 bin/pv bin/pv-sat; touch crates/aprender-contracts/README.md   # a doc newer than pv is not a verifier source
chk "doc-only change newer than pv still resolves" "$(run)" 0

for p in build.rs w3c/x.yaml fixtures/w.json src/lint/plant.yaml; do
    mk "pin-${p//\//_}"; mkdir -p "crates/aprender-contracts/$(dirname "$p")"; printf "d\n" > "crates/aprender-contracts/$p"; git add -A
    touch -d 1999-01-01 crates/aprender-contracts/src/lib.rs "crates/aprender-contracts/$p"; touch -d 2000-01-01 bin/pv bin/pv-sat; touch "crates/aprender-contracts/$p"
    chk "newer $p than pv refuses (include_str!/build input)" "$(run)" 1
done

mk outside src; touch bin/pv bin/pv-sat
chk "conflict outside the generated set refuses" "$(run)" 1
chk "outside: nothing regenerated" "$([ -f calls ] && echo ran || echo none)" none

[ "$fail" = 0 ] && echo "contracts_resolve_test: PASS"
exit "$fail"
