#!/usr/bin/env bash
# stamp_rc_version.sh -- an rc asset's crate version IS its tag: 0.69.3-rc.2, never bare 0.69.3
# (#4110, #4256, #4290)
#
#   bash scripts/release/stamp_rc_version.sh TREE TAG    # rewrite TREE's workspace to TAG's version
#   bash scripts/release/stamp_rc_version.sh --self-test # case table + mutants on a scratch workspace
#
# WHY. The rc is tagged on the commit CI gated, whose Cargo.toml says X.Y.Z. Built as is, every
# v0.69.3-rc.N asset printed `apr 0.69.3 (...)`: rc.1, rc.2 and the final were indistinguishable,
# and every CARGO_PKG_VERSION-keyed record (the F2 forward receipt, #4290) collided across them.
# Operator 2026-09-24: "we need actual version numbers" / "version number needs release canidate
# info in it". binary-release.yml runs this on the checked-out tag tree before `cargo build
# --locked`, so CARGO_PKG_VERSION itself is X.Y.Z-rc.N.
#
# WHAT IS REWRITTEN, and only when the workspace version is exactly TAG's X.Y.Z:
#   - every workspace member manifest (and the root): `version = "X.Y.Z"` on its own line;
#   - every dependency line that names a `path =` and a version requirement "X.Y.Z" / "=X.Y.Z" /
#     "^X.Y.Z" / "~X.Y.Z": a plain "0.69.3" requirement does NOT match 0.69.3-rc.2 (semver: a
#     pre-release only satisfies a requirement naming that same pre-release), so the pins move too;
#   - Cargo.lock `[[package]]` entries at X.Y.Z WITHOUT a `source` (workspace crates only; a
#     registry crate that happens to share the number keeps it).
# A final tag (vX.Y.Z) is a no-op. A tag whose X.Y.Z is not the workspace version is refused:
# stamping it would publish a binary claiming a version its source never was.
# Cargo itself is the oracle: the build that follows runs `--locked`, and the self-test runs
# `cargo metadata --locked --offline` on the stamped scratch workspace.
#
# EXIT 0 stamped (or a final tag: nothing to do) · 1 refused · 2 usage / unreadable tree.
set -uo pipefail
PROG=stamp_rc_version

stamp() {  # stamp TREE TAG
    python3 - "$1" "$2" <<'EOF'
import os, re, sys
tree, tag = sys.argv[1], sys.argv[2]
m = re.fullmatch(r"v(\d+\.\d+\.\d+)(-rc\.(\d+))?", tag)
if not m:
    print(f"refuse: tag {tag!r} is not vX.Y.Z or vX.Y.Z-rc.N"); sys.exit(1)
base, new = m.group(1), tag[1:]
root = os.path.join(tree, "Cargo.toml")
try:
    text = open(root, encoding="utf-8").read()
except OSError as e:
    print(f"usage: {e}"); sys.exit(2)
ws = re.search(r'^\[workspace\.package\]\s*\n(?:(?!\[).*\n)*?version\s*=\s*"([^"]+)"', text, re.M)
if not ws:
    print("usage: no [workspace.package] version in the root Cargo.toml"); sys.exit(2)
have = ws.group(1)
if have == new:
    print(f"ok {new}: already stamped"); sys.exit(0)
if have != base:
    print(f"refuse: tag {tag} names {base}, the workspace is {have}"); sys.exit(1)
if not m.group(2):
    print(f"ok {new}: a final tag, nothing to stamp"); sys.exit(0)
# members: root + every Cargo.toml under crates/ that is not in a nested workspace we exclude
mem = re.search(r'^members\s*=\s*\[(.*?)\]', text, re.M | re.S)
exc = re.search(r'^exclude\s*=\s*\[(.*?)\]', text, re.M | re.S)
quoted = lambda s: re.findall(r'"([^"]+)"', re.sub(r'#.*', '', s)) if s else []
import glob
manifests = [root]
for pat in quoted(mem.group(1) if mem else ""):
    for d in sorted(glob.glob(os.path.join(tree, pat))):
        f = os.path.join(d, "Cargo.toml")
        if os.path.isfile(f):
            manifests.append(f)
excluded = [os.path.normpath(os.path.join(tree, e)) for e in quoted(exc.group(1) if exc else "")]
manifests = [f for f in manifests if not any(os.path.normpath(f).startswith(e + os.sep) for e in excluded)]
b = re.escape(base)
own = re.compile(rf'^(version\s*=\s*"){b}(")', re.M)
# a pin's number may be partial ("0.69" in aprender-core -> apr-format): any X / X.Y / X.Y.Z that
# base satisfies is a pin on this workspace, and none of them matches a pre-release
req = r'([=^~]?)(\d+(?:\.\d+){0,2})'
dep = re.compile(rf'^(.*\bpath\s*=.*\bversion\s*=\s*"){req}(")', re.M)
dep2 = re.compile(rf'^(.*\bversion\s*=\s*"){req}(".*\bpath\s*=)', re.M)
def pin(mo):
    num = mo.group(3).split(".")
    if num != base.split(".")[:len(num)]:
        return mo.group(0)
    return mo.group(1) + mo.group(2) + new + mo.group(4)
n_own = n_dep = 0
for f in manifests:
    s = open(f, encoding="utf-8").read()
    s, a = own.subn(rf'\g<1>{new}\g<2>', s)
    s, c = dep.subn(pin, s)
    s, d = dep2.subn(pin, s)
    n_own, n_dep = n_own + a, n_dep + c + d
    open(f, "w", encoding="utf-8").write(s)
lock = os.path.join(tree, "Cargo.lock")
n_lock = 0
if os.path.isfile(lock):
    blocks = open(lock, encoding="utf-8").read().split("\n[[package]]\n")
    for i, blk in enumerate(blocks):
        if i and "\nsource = " not in blk:
            blk, k = re.subn(rf'^version = "{b}"$', f'version = "{new}"', blk, flags=re.M)
            blocks[i], n_lock = blk, n_lock + k
    open(lock, "w", encoding="utf-8").write("\n[[package]]\n".join(blocks))
print(f"stamped {base} -> {new}: {len(manifests)} manifests, {n_own} package versions, {n_dep} path pins, {n_lock} lock entries")
sys.exit(0 if n_own else 1)
EOF
}

self_test() {
    local d fail=0 got rc row want name
    command -v cargo > /dev/null || { echo "$PROG self-test: needs cargo (it is the oracle)"; return 2; }
    d=$(mktemp -d) || return 2
    # a scratch workspace shaped like the real one: [workspace.package] version, a member that
    # inherits it, one that declares its own, a path dep with a caret pin and one with =, an
    # excluded nested crate, and a lock entry for a REGISTRY crate at the same number
    mkws() {
        local w=$1; rm -rf -- "${w:?}"; mkdir -p "$w/crates/a/src" "$w/crates/b/src" "$w/crates/c/src" "$w/crates/d/src" "$w/crates/x/src"
        printf '[workspace]\nmembers = ["crates/*"]\nexclude = [\n    "crates/x", # nested\n]\nresolver = "2"\n\n[workspace.package]\nversion = "0.69.3"\nedition = "2021"\n\n[workspace.dependencies]\nb = { path = "crates/b", version = "0.69.3" }\n' > "$w/Cargo.toml"
        printf '[package]\nname = "a"\nversion.workspace = true\nedition = "2021"\n\n[dependencies]\nb = { workspace = true }\nd = { path = "../d", version = "0.69" }\nc = { version = "=0.69.3", path = "../c" }\n' > "$w/crates/a/Cargo.toml"
        printf '[package]\nname = "b"\nversion = "0.69.3"\nedition = "2021"\n' > "$w/crates/b/Cargo.toml"
        printf '[package]\nname = "c"\nversion = "0.69.3"\nedition = "2021"\n' > "$w/crates/c/Cargo.toml"; : > "$w/crates/c/src/lib.rs"
        printf '[package]\nname = "d"\nversion = "0.69.3"\nedition = "2021"\n' > "$w/crates/d/Cargo.toml"; : > "$w/crates/d/src/lib.rs"
        printf '[package]\nname = "x"\nversion = "0.69.3"\nedition = "2021"\n[workspace]\n' > "$w/crates/x/Cargo.toml"
        : > "$w/crates/a/src/lib.rs"; : > "$w/crates/b/src/lib.rs"; : > "$w/crates/x/src/lib.rs"
        (cd "$w" && cargo generate-lockfile --offline -q 2> /dev/null) || return 1
    }
    # versions WT -> "a=<v> b=<v>" from cargo, resolving --locked; "LOCKED-FAIL" when cargo refuses
    versions() {
        (cd "$1" && cargo metadata --locked --offline --format-version 1 2> /dev/null) \
        | python3 -c 'import json,sys; print(" ".join(sorted(p["name"]+"="+p["version"] for p in json.load(sys.stdin)["packages"])))' 2> /dev/null \
        || echo LOCKED-FAIL
    }
    expect() {  # expect ROW WANT GOT
        if [ "$3" = "$2" ]; then echo "  ok   $1"; else printf '  FAIL %s\n       want: %s\n       got:  %s\n' "$1" "$2" "$3"; fail=1; fi
    }
    echo "$PROG self-test: case table (cargo metadata --locked is the oracle)"
    mkws "$d/w" || { echo "  FAIL cannot build the scratch workspace"; rm -rf -- "${d:?}"; return 2; }
    stamp "$d/w" v0.69.3-rc.2 > "$d/out"; rc=$?
    expect "rc tag stamps and exits 0" 0 "$rc"
    expect "cargo resolves the stamped workspace --locked, every member at the rc" "a=0.69.3-rc.2 b=0.69.3-rc.2 c=0.69.3-rc.2 d=0.69.3-rc.2" "$(versions "$d/w")"
    expect "the excluded nested crate keeps its version" 'version = "0.69.3"' "$(grep '^version' "$d/w/crates/x/Cargo.toml")"
    mkws "$d/w"; printf '\n[[package]]\nname = "zzz-registry"\nversion = "0.69.3"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\n' >> "$d/w/Cargo.lock"
    stamp "$d/w" v0.69.3-rc.2 > /dev/null
    expect "a registry lock entry at the same number is untouched" 'version = "0.69.3"' "$(grep -A1 'name = "zzz-registry"' "$d/w/Cargo.lock" | tail -1)"
    mkws "$d/w"; stamp "$d/w" v0.69.3-rc.2 > /dev/null
    stamp "$d/w" v0.69.3-rc.2 > /dev/null; expect "re-stamping the same rc is idempotent" 0 "$?"
    mkws "$d/w"; stamp "$d/w" v0.69.3 > /dev/null; rc=$?
    expect "a final tag is a no-op (exit 0, workspace unchanged)" "0 a=0.69.3 b=0.69.3 c=0.69.3 d=0.69.3" "$rc $(versions "$d/w")"
    for row in "1 v0.69.4-rc.1 a tag for another version is refused" "1 v0.69.3-beta.1 a non-rc prerelease is refused" "1 0.69.3-rc.1 a tag without v is refused"; do
        mkws "$d/w"; want=${row%% *}; row=${row#* }; name=${row%% *}
        stamp "$d/w" "$name" > /dev/null; expect "${row#* }" "$want" "$?"
    done
    # MUTANTS, built from THIS file: each drops one rewrite, and cargo must refuse the result.
    # Without them the table could pass on a stamper that only renamed the packages.
    for row in "lock|blocks[i], n_lock = blk, n_lock + k|n_lock = n_lock + k" \
               "path-first pins|s, c = dep.subn|c = 0; _ = dep.subn" \
               "version-first pins|s, d = dep2.subn|d = 0; _ = dep2.subn" \
               "partial (X.Y) pins|(?:\\.\\d+){0,2}|(?:\\.\\d+){2}"; do
        IFS='|' read -r name a1 b1 <<< "$row"
        A1=$a1 B1=$b1 python3 -c 'import os,sys; s=open(sys.argv[1]).read(); a=os.environ["A1"]; assert s.count(a)>=1 and s.index(a)<s.index("self_test()"); open(sys.argv[2],"w").write(s.replace(a,os.environ["B1"],1))' \
            "${BASH_SOURCE[0]}" "$d/mut.sh" 2> /dev/null || { echo "  FAIL mutant $name: anchor moved, re-anchor it"; fail=1; continue; }
        mkws "$d/w"; bash "$d/mut.sh" "$d/w" v0.69.3-rc.2 > /dev/null
        got=$(versions "$d/w")
        case "$got" in *LOCKED-FAIL*) echo "  ok   mutant ($name rewrite dropped): cargo refuses the tree";;
                       *) echo "  FAIL mutant ($name rewrite dropped) survived: $got"; fail=1;; esac
    done
    rm -rf -- "${d:?}"
    if [ "$fail" -eq 0 ]; then echo "$PROG self-test: PASS"; return 0; fi
    echo "$PROG self-test: FAIL"; return 1
}

case "${1:-}" in
    --self-test) self_test ;;
    -h|--help) sed -n '2,28p' "${BASH_SOURCE[0]}" ;;
    *) if [ "$#" -ne 2 ] || [ ! -d "$1" ]; then echo "$PROG: usage: TREE TAG | --self-test" >&2; exit 2; fi
       stamp "$1" "$2" ;;
esac
