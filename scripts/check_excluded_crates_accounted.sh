#!/usr/bin/env bash
# check_excluded_crates_accounted.sh - every workspace `exclude` entry is either
# exercised by CI or recorded as deliberately unbuilt, with the reason (#3174).
#
# THE CLASS. An excluded directory is invisible to the build tool's metadata,
# to a workspace check and to every workspace-wide CI job. crates/aprender-train-canary,
# the only thing in the tree that measured trueno against Burn, sat there with
# no CI job and a dependency pinned to a Burn pre-release that had shipped four
# months earlier. Nothing noticed, because nothing could see it. Being excluded
# is a decision; being unrunnable is a separate one, and the two got conflated.
#
# THE RULE. The universe is the root manifest's `[workspace] exclude` list
# itself, not a list kept by hand. Each entry needs a row in
# `[workspace.metadata.excluded]` in the same manifest, with exactly one of:
#   ci      = "<script>"  a script CI runs, and whose CODE (comment lines
#                         stripped) names the excluded path as a whole path.
#                         "CI runs" means: a guard in the RUN set of
#                         `scripts/guard_tree.sh --dry-run` (the decision CI's
#                         guard-tree/guard-cargo sections execute, and the one
#                         check_guards_are_wired.sh reads), or a script a
#                         workflow names on a non-comment line
#   unbuilt = "<reason>"  nothing builds it, and this says why
# A row with no matching exclude entry is RED too, so the table cannot go stale.
#
# THE PIN. An excluded crate's dependencies are never re-resolved by the
# workspace, so a pre-release pin there rots silently. Every version
# requirement in every Cargo.toml under an excluded path (nested members
# included; target/ skipped) must be a released version: no `-pre`, `-rc`,
# `-alpha`, `-beta` or any other semver pre-release suffix. Scanned: the
# [dependencies]/[dev-dependencies]/[build-dependencies] tables and their
# target.* forms, [workspace.dependencies], [patch.*] and [replace]. Lockfiles
# are not scanned: a transitive pre-release there is resolution, not a pin.
#
# THE CASE TABLE. `--self-test` runs the checker over fixture trees, one per
# defect, and each must go RED (plus clean fixtures that must stay GREEN). A
# normal run executes the table first, so a checker that stops catching a
# defect fails here rather than passing vacuously.
#
# This guard invokes no build tool: python over the manifests and workflow
# text only. It stays in guard_tree.sh's cargo-free population, so it must
# never write that tool's name followed by a space.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-run}"
case "$MODE" in
    run | --self-test) ;;
    -h | --help)
        sed -n '2,41p' "${BASH_SOURCE[0]}"
        exit 0
        ;;
    *)
        echo "check_excluded_crates_accounted: unknown argument $MODE" >&2
        exit 2
        ;;
esac

exec python3 - "$REPO_ROOT" "$MODE" <<'PY'
import os
import re
import subprocess
import sys
import tempfile
import tomllib

PRE = re.compile(r"\d+\.\d+(?:\.\d+)?-[0-9A-Za-z]")
DEP_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")


def load(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def dep_tables(manifest):
    for t in DEP_TABLES:
        yield t, manifest.get(t, {})
    for target, body in manifest.get("target", {}).items():
        for t in DEP_TABLES:
            yield f"target.{target}.{t}", body.get(t, {})
    yield "workspace.dependencies", manifest.get("workspace", {}).get("dependencies", {})
    for registry, body in manifest.get("patch", {}).items():
        yield f"patch.{registry}", body
    yield "replace", manifest.get("replace", {})


def manifests(root, excluded):
    """Every Cargo.toml under an excluded path, nested members included."""
    top = os.path.join(root, excluded)
    for d, dirs, files in os.walk(top):
        dirs[:] = sorted(x for x in dirs if x not in ("target", ".git"))
        if "Cargo.toml" in files:
            yield os.path.join(d, "Cargo.toml")


def code(text):
    """Text with comment lines (first non-blank char `#`) removed."""
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))


def names(text, token):
    """`token` appears as a whole path: not glued to a longer name (`./x` and `$ROOT/x` count)."""
    return re.search(r"(?<![\w.-])" + re.escape(token) + r"(?![\w-])", text) is not None


def dispatched(root):
    """Scripts guard_tree.sh --dry-run decides to RUN (its `run: <path>` rows)."""
    out = subprocess.run(["bash", "scripts/guard_tree.sh", "--dry-run"], cwd=root,
                         capture_output=True, text=True, check=False).stdout
    run = {m.group(1) for m in re.finditer(r"^run: (\S+)", out, re.M)}
    if not run:
        raise SystemExit("FAIL  guard_tree.sh --dry-run reported no run: rows; cannot tell what CI runs")
    return run


def ci_run(root, script, run_set):
    """A script CI runs: guard_tree.sh dispatches it, or a workflow names it."""
    if script in run_set:
        return True
    wf = os.path.join(root, ".github", "workflows")
    if not os.path.isdir(wf):
        return False
    for name in sorted(os.listdir(wf)):
        if name.endswith((".yml", ".yaml")):
            with open(os.path.join(wf, name), encoding="utf-8") as f:
                if names(code(f.read()), script):
                    return True
    return False


def check(root, run_set):
    errs = []
    ws = load(os.path.join(root, "Cargo.toml")).get("workspace", {})
    excl = [e.rstrip("/") for e in ws.get("exclude", [])]
    ledger = {k.rstrip("/"): v for k, v in ws.get("metadata", {}).get("excluded", {}).items()}
    for e in excl:
        if e not in ledger:
            errs.append(f"{e}: excluded, but has no [workspace.metadata.excluded] row (ci or unbuilt)")
    for k in ledger:
        if k not in excl:
            errs.append(f"{k}: [workspace.metadata.excluded] row for a path that is not excluded")
    for e in excl:
        row = ledger.get(e)
        if row is None:
            continue
        if not isinstance(row, dict):
            errs.append(f"{e}: row must be a table with `ci` or `unbuilt`")
            continue
        has_ci, has_unbuilt = "ci" in row, "unbuilt" in row
        if has_ci == has_unbuilt:
            errs.append(f"{e}: row needs exactly one of `ci` / `unbuilt`, has {'both' if has_ci else 'neither'}")
            continue
        if has_unbuilt:
            if not isinstance(row["unbuilt"], str) or not row["unbuilt"].strip():
                errs.append(f"{e}: `unbuilt` must give the reason")
        else:
            script = row["ci"]
            path = os.path.join(root, script) if isinstance(script, str) else ""
            if not script or not os.path.isfile(path):
                errs.append(f"{e}: ci script {script!r} does not exist")
                continue
            with open(path, encoding="utf-8") as f:
                text = f.read()
            if not ci_run(root, script, run_set):
                errs.append(f"{e}: ci script {script} is neither run by guard_tree.sh --dry-run nor named by a workflow")
            elif not names(code(text), e):
                errs.append(f"{e}: ci script {script} never names {e} outside comments")
    for e in excl:
        for mpath in manifests(root, e):
            rel = os.path.relpath(mpath, root)
            for table, deps in dep_tables(load(mpath)):
                for name, spec in deps.items():
                    req = spec if isinstance(spec, str) else spec.get("version") if isinstance(spec, dict) else None
                    if isinstance(req, str) and PRE.search(req):
                        errs.append(f"{rel} [{table}] {name} = \"{req}\": pinned to a pre-release")
    return errs


def fixture(d, files):
    for rel, text in files.items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)


ROOT_OK = """[workspace]
members = []
exclude = ["crates/canary", "crates/shell/"]
[workspace.metadata.excluded]
"crates/canary" = { ci = "scripts/check_canary.sh" }
"crates/shell" = { unbuilt = "workspace root shell, no package" }
"""
CANARY = '[package]\nname = "canary"\nversion = "0.1.0"\n[dependencies]\nburn = { version = "%s" }\n'
GUARD = "#!/usr/bin/env bash\nbash crates/canary/run.sh\n"


def case(root_toml=ROOT_OK, canary="0.21.0", guard=GUARD, extra=None):
    files = {"Cargo.toml": root_toml, "crates/canary/Cargo.toml": CANARY % canary,
             "scripts/check_canary.sh": guard}
    files.update(extra or {})
    return files


CASES = [
    # (name, files, must_be_red[, guard_tree RUN set]); the default RUN set holds the
    # fixture guard, as guard_tree.sh --dry-run would for a tracked check_*.sh.
    ("clean", case(), False),
    ("exact released pin", case(canary="=0.21.0"), False),
    ("ci via a workflow-named script",
     case(root_toml=ROOT_OK.replace("scripts/check_canary.sh", "scripts/run_canary.sh"),
          extra={"scripts/run_canary.sh": GUARD,
                 ".github/workflows/ci.yml": "steps:\n  - run: bash scripts/run_canary.sh\n"}), False),
    ("exclude entry with no row", case(root_toml=ROOT_OK.replace('"crates/shell" = { unbuilt = "workspace root shell, no package" }\n', "")), True),
    ("stale row", case(root_toml=ROOT_OK + '"crates/gone" = { unbuilt = "x" }\n'), True),
    ("row with both", case(root_toml=ROOT_OK.replace('{ ci = "scripts/check_canary.sh" }', '{ ci = "scripts/check_canary.sh", unbuilt = "x" }')), True),
    ("row with neither", case(root_toml=ROOT_OK.replace('{ ci = "scripts/check_canary.sh" }', '{ issue = 1 }')), True),
    ("empty unbuilt reason", case(root_toml=ROOT_OK.replace('"workspace root shell, no package"', '" "')), True),
    ("ci script missing", case(root_toml=ROOT_OK.replace("check_canary.sh", "check_nope.sh")), True),
    ("ci script never names the path", case(guard="#!/usr/bin/env bash\necho hi\n"), True),
    ("ci script nothing runs",
     case(root_toml=ROOT_OK.replace("scripts/check_canary.sh", "scripts/run_canary.sh"),
          extra={"scripts/run_canary.sh": GUARD}), True),
    ("pre-release string pin", case(canary="0.21.0-pre.2"), True),
    ("rc pin", case(canary="1.0.0-rc.1"), True),
    ("two-part pre-release pin", case(canary="0.22-alpha"), True),
    ("ci script names the path only in a comment",
     case(guard="#!/usr/bin/env bash\n# builds crates/canary\necho hi\n"), True),
    ("ci script names the path under $ROOT/", case(guard="#!/usr/bin/env bash\nbash \"$ROOT/crates/canary/run.sh\"\n"), False),
    ("ci script names only a longer path", case(guard="#!/usr/bin/env bash\nls crates/canary-old\n"), True),
    ("check_ script guard_tree does not run", case(), True, set()),
    ("workflow names the script only in a comment",
     case(root_toml=ROOT_OK.replace("scripts/check_canary.sh", "scripts/run_canary.sh"),
          extra={"scripts/run_canary.sh": GUARD,
                 ".github/workflows/ci.yml": "steps:\n  # - run: bash scripts/run_canary.sh\n"}), True),
    ("nested member pins a pre-release",
     case(extra={"crates/canary/sub/Cargo.toml": CANARY % "1.0.0-beta.1"}), True),
    ("pre-release in target/ is ignored",
     case(extra={"crates/canary/target/x/Cargo.toml": CANARY % "1.0.0-beta.1"}), False),
    ("workspace.dependencies pre-release",
     case(extra={"crates/canary/Cargo.toml": CANARY % "0.21.0" + '[workspace.dependencies]\nx = "3.0.0-rc.2"\n'}), True),
    ("patch pre-release",
     case(extra={"crates/canary/Cargo.toml": CANARY % "0.21.0" + '[patch.crates-io]\nx = { version = "3.0.0-alpha" }\n'}), True),
    ("target-specific pre-release dev-dep",
     case(extra={"crates/canary/Cargo.toml": CANARY % "0.21.0"
                 + '[target.\'cfg(unix)\'.dev-dependencies]\nx = "2.0.0-beta.3"\n'}), True),
]


def self_test():
    bad = 0
    for name, files, must_red, *run_set in CASES:
        run_set = run_set[0] if run_set else {"scripts/check_canary.sh"}
        with tempfile.TemporaryDirectory(prefix="excl-accounted.") as d:
            fixture(d, files)
            errs = check(d, run_set)
        ok = bool(errs) == must_red
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {'RED  ' if must_red else 'GREEN'} {name}" + ("" if ok else f" -> {errs}"))
    print(f"case table: {len(CASES) - bad}/{len(CASES)}")
    return bad == 0


root, mode = sys.argv[1], sys.argv[2]
if not self_test():
    print("FAIL  check_excluded_crates_accounted: the case table does not hold; the checker cannot be trusted")
    sys.exit(1)
if mode == "--self-test":
    sys.exit(0)
errs = check(root, dispatched(root))
for e in errs:
    print(f"FAIL  {e}")
if errs:
    print(f"FAIL  {len(errs)} excluded-crate defect(s) (#3174)")
    sys.exit(1)
n = len(load(os.path.join(root, "Cargo.toml"))["workspace"].get("exclude", []))
print(f"PASS  all {n} workspace exclude entries accounted for; no excluded crate pins a pre-release")
PY
