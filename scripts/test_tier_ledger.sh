#!/usr/bin/env bash
# test_tier_ledger.sh — the catch ledger for the 80/20 PR tier (PMAT-1105, spec §6): how often the last N days of
# `fix` commits touched each module's tests, keyed EXACTLY as scripts/lib/test_tier.py keys a test's module
# ('<crate>::<module path>' = everything before the last '::' of the nextest test name, so an inline test module
# is '<crate>::<file module>::tests').
#
#   bash scripts/test_tier_ledger.sh [--base origin/main] [--days 60] [--out FILE] [--selftest]
#
# Method (a proxy, stated as such): for every commit whose subject starts with 'fix' (case-insensitive), every
# hunk of every crates/<crate>/src/**/*.rs file: the file's module path is its path relative to src/ with '/'→'::',
# '.rs' dropped and 'mod' dropped; a hunk that adds or removes a line containing '#[test]' or 'fn test_' counts one
# touch for '<crate>::<file module>::tests' when the file declares `mod tests`, else for '<crate>::<file module>'.
# Integration targets (crates/<crate>/tests/*.rs) are not in the --lib junit and are written under 'not_measured'.
set -euo pipefail
BASE="origin/main"; DAYS=60; OUT="evidence/fleet/test-tier-ledger.json"; SELFTEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE="$2"; shift 2 ;;
    --days) DAYS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --selftest) SELFTEST=1; shift ;;
    *) echo "test_tier_ledger: unknown argument $1" >&2; exit 2 ;;
  esac
done

ledger_py() {
python3 - "$@" <<'PY'
import json, re, subprocess, sys
base, days = sys.argv[1], sys.argv[2]
shas = subprocess.run(["git","log",base,f"--since={days}.days","--format=%h","--grep=^fix","-i"],capture_output=True,text=True,check=True).stdout.split()
touches, notm = {}, {}
for sha in shas:
    diff = subprocess.run(["git","show","--format=","-U0",sha,"--","crates/*/src/*.rs","crates/*/tests/*.rs"],capture_output=True,text=True).stdout
    cur=None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path=line[6:]; m=re.match(r"crates/([^/]+)/(src|tests)/(.+)\.rs$", path)
            cur=None
            if m:
                crate, kind, rel = m.groups()
                mod = "::".join(p for p in rel.split("/") if p not in ("mod","lib","main"))
                cur=(crate, kind, mod, path)
        elif cur and (line.startswith("+") or line.startswith("-")) and not line.startswith(("+++","---")):
            if "#[test]" in line:  # one line per test function; `fn test_` would double-count the same function
                crate, kind, mod, path = cur
                if kind=="tests":
                    key=f"{crate}::{mod}"; notm[key]=notm.get(key,0)+1; continue
                has_tests_mod = "mod tests" in (subprocess.run(["git","show",f"{sha}:{path}"],capture_output=True,text=True).stdout)
                key = f"{crate}::{mod}::tests" if has_tests_mod and mod else (f"{crate}::{mod}" if mod else f"{crate}::")
                touches[key]=touches.get(key,0)+1
json.dump({"_meta":{"base":base,"days":int(days),"fix_commits":len(shas),"not_measured":notm}, **touches}, open(sys.argv[3],"w"), indent=0, sort_keys=True)
print(f"test_tier_ledger: base={base} days={days} fix_commits={len(shas)} modules={len(touches)} touches={sum(touches.values())} not_measured_modules={len(notm)}")
PY
}

if [ "$SELFTEST" = 1 ]; then
  T="$(mktemp -d)"; trap '[ -n "$T" ] && [ "$T" != "/" ] && rm -rf "$T"' EXIT
  ( cd "$T" && git init -q . && git config user.email t@t && git config user.name t
    mkdir -p crates/alpha/src/foo crates/alpha/tests
    printf 'pub fn a(){}\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn test_one(){}\n}\n' > crates/alpha/src/foo/bar.rs
    printf '#[test]\nfn test_int(){}\n' > crates/alpha/tests/it.rs
    git add -A && git commit -qm "feat: seed" && git branch -q -M main
    printf 'pub fn a(){}\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn test_one(){}\n    #[test]\n    fn test_two(){}\n}\n' > crates/alpha/src/foo/bar.rs
    printf '#[test]\nfn test_int(){}\n#[test]\nfn test_int2(){}\n' > crates/alpha/tests/it.rs
    git commit -qam "fix(alpha): the second case was never tested"
    printf 'pub fn a(){}\npub fn b(){}\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn test_one(){}\n    #[test]\n    fn test_two(){}\n}\n' > crates/alpha/src/foo/bar.rs
    git commit -qam "feat: b (no test touched)" )
  ( cd "$T" && ledger_py main 3650 "$T/ledger.json" >/dev/null )
  err=0
  chk() { if [ "$2" = "$3" ]; then echo "ok    $1"; else echo "FAIL  $1: got $2 want $3"; err=1; fi; }
  chk "inline test touch keyed crate::module::tests" "$(jq -r '."alpha::foo::bar::tests" // 0' "$T/ledger.json")" 1
  chk "integration test goes to not_measured"       "$(jq -r '._meta.not_measured."alpha::it" // 0' "$T/ledger.json")" 1
  chk "feat commit adds no touch"                    "$(jq '[to_entries[] | select(.key|startswith("_")|not) | .value] | add' "$T/ledger.json")" 1
  chk "fix commit count recorded"                    "$(jq -r '._meta.fix_commits' "$T/ledger.json")" 1
  exit "$err"
fi
mkdir -p "$(dirname "$OUT")"
ledger_py "$BASE" "$DAYS" "$OUT"
