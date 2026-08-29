#!/usr/bin/env python3
"""Does the required `ci / gate` have anything left to narrow? (aprender#2734)

`ci / gate` is a required status check on main. It is produced by the reusable
workflow `paiml/.github/.github/workflows/sovereign-ci.yml@main`, whose lint and
test steps both end in a fallback chain that narrows SCOPE on failure instead of
reporting it:

    cargo clippy $CLIPPY_ARGS -- -D warnings || \
    cargo clippy -p "$REPO_NAME" -- -D warnings || \
    { echo "::error::Clippy failed ..."; exit 1; }

The second command is not a retry of the same work -- it is a strictly smaller
scope, and the narrowing emits no `::warning::`. So a green `ci / gate` cannot
distinguish "the configured scope passed" from "the configured scope failed and
a smaller one passed". That is an assertion which cannot exclude the outcome it
exists to exclude, and it is upstream: nothing in this repo can disarm it.

What this repo CAN do is hold the difference at zero. If the set of units the
FALLBACK command compiles equals the set the PRIMARY command compiles, then a
silent narrowing has nothing to hide and the required check means what it says.

Both rules below are that one idea, once per chain:

  S1 CLIPPY WINDOW is EMPTY
     units(primary) - units(fallback) must be the empty set, computed from
     cargo's own resolution rather than from reading the manifest. Today it is
     empty because the `aprender` facade declares exactly two targets, one lib
     and one bin, and `--all-targets` adds test/bench/example targets that do
     not exist. That is an accident of the tree, not an invariant -- the window
     opens the day a root-level tests/*.rs, benches/*.rs or examples/*.rs
     appears, and nothing would say so.

     MEASURED at the time of writing (aprender 0.64.0, 79 workspace members):
     workspace_default_members = 1 (the facade alone, because the root manifest
     is both [workspace] and [package] and declares no default-members), and
     that package's targets are ['lib' aprender] and ['bin' apr]. Primary and
     fallback therefore select an identical 2-unit set.

  S2 TEST WINDOW is EMPTY
     the caller must not opt into `test_workspace: true`. For a caller that
     does, TEST_SCOPE becomes `--workspace --lib` and the fallback is
     `--lib -p "$REPO_NAME"` -- a failing test anywhere in the workspace is
     downgraded to a single-package run. aprender does not opt in; its
     workspace testing lives in the repo-owned `workspace-test` job, which has
     no fallback. Opting in would arm the trap for no gain.

WHY A PARSER AND NOT A GREP
    S1 reads `cargo metadata`, so it cannot disagree with what cargo will
    actually build. S2 reads the workflow with PyYAML, so the line
    `# test_workspace: true + GPU-member test_args exclusions first (...)`
    that has sat in ci.yml since the sccache pilot is a comment to this checker
    as well as to Actions. A grep for `test_workspace` matches it; the four
    other guards in this repo whose regexes were wrong were all caught by a
    case table, never by review. scripts/lib/gate_scope_cases/ carries that
    line as an explicit must-not-match row.

DIRECTION OF ERROR
    S1 ignores `required-features`, so a test target gated behind an unset
    feature still counts as window. That over-reports: the failure is a loud
    RED naming a target, never a silent pass. A guard may be conservative; it
    may not be optimistic.

    Likewise an overridden `clippy_args` is refused rather than guessed. This
    checker models the sovereign-ci DEFAULT (`--all-targets`); any override
    changes what "primary" means, and a model that silently kept using the old
    meaning would be the same class of defect one level up.

Usage:
    gate_scope.py <cargo-metadata.json> <workflow.yml> <repo-name>
Exit:
    0 both windows empty   1 a window is open   2 the input could not be read
"""

import json
import sys

import yaml

# The reusable workflow's own default for `clippy_args`. An override is refused,
# not modelled -- see DIRECTION OF ERROR above.
SOVEREIGN_CI_DEFAULT_CLIPPY_ARGS = "--all-targets"
SOVEREIGN_CI_WORKFLOW = "sovereign-ci.yml"

# A target whose `kind` list intersects LIB_KINDS is the package's lib target
# under any of its crate-type spellings; cargo selects it with or without
# `--all-targets`. `custom-build` is in neither set: a build script is compiled
# because something depends on it, never because a target flag selected it.
LIB_KINDS = frozenset({"lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro"})
DEFAULT_KINDS = LIB_KINDS | {"bin"}
ALL_TARGETS_KINDS = DEFAULT_KINDS | {"test", "bench", "example"}


def _kind_of(target):
    """One canonical kind per target, or None for targets no flag selects."""
    kinds = set(target.get("kind") or [])
    if kinds & LIB_KINDS:
        return "lib"
    for k in ("bin", "test", "bench", "example"):
        if k in kinds:
            return k
    return None


def _units(packages, selected_kinds):
    out = set()
    for pkg in packages:
        for target in pkg.get("targets") or []:
            kind = _kind_of(target)
            if kind in selected_kinds:
                out.add((pkg.get("name"), kind, target.get("name")))
    return out


def _sovereign_ci_inputs(workflow):
    """The `with:` block of the job that calls sovereign-ci, or None."""
    for job in (workflow.get("jobs") or {}).values():
        if not isinstance(job, dict):
            continue
        uses = job.get("uses") or ""
        if SOVEREIGN_CI_WORKFLOW in uses:
            return job.get("with") or {}
    return None


def check_clippy_window(metadata, repo_name, inputs):
    """S1 -- list of failure strings; empty means the window is closed."""
    packages = metadata.get("packages") or []
    default_ids = metadata.get("workspace_default_members")
    if not packages or not default_ids:
        return [
            "S1 VACUOUS: cargo metadata carries no packages or no "
            "workspace_default_members. A scope that cannot be measured is not "
            "a scope that is known to be safe."
        ]

    clippy_args = inputs.get("clippy_args")
    if clippy_args is not None and clippy_args != SOVEREIGN_CI_DEFAULT_CLIPPY_ARGS:
        return [
            "S1 UNMODELLED: this caller overrides clippy_args to "
            f"{clippy_args!r}. This checker models the sovereign-ci default "
            f"({SOVEREIGN_CI_DEFAULT_CLIPPY_ARGS!r}) only; an override changes "
            "what the PRIMARY command selects, and the fallback "
            "`cargo clippy -p <repo>` does not change with it. Model the "
            "override here before shipping it."
        ]

    by_id = {p.get("id"): p for p in packages}
    default_pkgs = [by_id[i] for i in default_ids if i in by_id]
    if len(default_pkgs) != len(default_ids):
        return [
            "S1 VACUOUS: workspace_default_members names a package that is "
            "absent from `packages`; the document is not internally consistent."
        ]

    fallback_pkgs = [p for p in packages if p.get("name") == repo_name]
    if not fallback_pkgs:
        return [
            f"S1: the fallback command is `cargo clippy -p {repo_name}`, and no "
            "workspace package carries that name. The fallback cannot run, so "
            "the chain's only remaining branch is its `::error::` exit -- which "
            "is loud, but it means the second command has never been what this "
            "repo thinks it is."
        ]

    primary = _units(default_pkgs, ALL_TARGETS_KINDS)
    fallback = _units(fallback_pkgs, DEFAULT_KINDS)
    window = sorted(primary - fallback)
    if not window:
        return []

    lines = [
        f"S1 CLIPPY WINDOW IS OPEN: {len(window)} unit(s) are compiled by the "
        "PRIMARY command and NOT by the fallback the required `ci / gate` "
        "silently retries with. A clippy error in any of them is masked:"
    ]
    lines += [f"    {pkg} :: {kind} :: {name}" for pkg, kind, name in window]
    lines.append(
        "  Remedy, in order of preference: (1) fix the fallback upstream in "
        "paiml/.github so a narrowed run is never indistinguishable from a "
        "clean one (aprender#2734); (2) move the target into a workspace "
        "member -- every one of this repo's 25,300 tests already lives in one, "
        "and a member's targets are outside BOTH commands' selection, so this "
        "changes nothing about the gate; (3) do not add it."
    )
    return lines


def check_test_window(inputs):
    """S2 -- list of failure strings; empty means the window is closed."""
    if not inputs.get("test_workspace"):
        return []
    return [
        "S2 TEST WINDOW IS OPEN: this caller passes `test_workspace: true`, so "
        "the required check runs `cargo nextest run --workspace --lib` and, on "
        "failure, silently retries `--lib -p <repo>`. A failing test anywhere "
        "in the workspace is downgraded to a single-package run and the gate "
        "goes green (aprender#2734).",
        "  Remedy: keep workspace testing in the repo-owned `workspace-test` "
        "job in ci.yml, which is a required check of its own and has no "
        "fallback -- or fix the fallback upstream in paiml/.github first.",
    ]


def _load_metadata(path):
    """(document, error-string). Reading is separated from deciding so that
    `main` stays under the repo's cognitive-complexity gate — the same split
    #2721 had to make in AprReader::from_bytes for the same reason."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle), None
    except (OSError, ValueError) as exc:
        return None, f"gate_scope: cannot read cargo metadata {path}: {exc}"


def _load_workflow(path):
    """(document, error-string)."""
    try:
        with open(path, encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        return None, f"gate_scope: cannot parse workflow {path}: {exc}"
    if not isinstance(document, dict):
        return None, f"gate_scope: {path} is not a YAML mapping"
    return document, None


def _no_caller_message(wf_path):
    return (
        f"gate_scope: no job in {wf_path} calls {SOVEREIGN_CI_WORKFLOW}. "
        "This guard exists to bound that workflow's fallback chain; with no "
        "caller it would pass over nothing, which is a fail mode."
    )


def main(argv):
    if len(argv) != 4:
        sys.stderr.write(
            "usage: gate_scope.py <cargo-metadata.json> <workflow.yml> <repo-name>\n"
        )
        return 2
    md_path, wf_path, repo_name = argv[1], argv[2], argv[3]

    metadata, error = _load_metadata(md_path)
    if error is not None:
        sys.stderr.write(error + "\n")
        return 2

    workflow, error = _load_workflow(wf_path)
    if error is not None:
        sys.stderr.write(error + "\n")
        return 2

    inputs = _sovereign_ci_inputs(workflow)
    if inputs is None:
        sys.stderr.write(_no_caller_message(wf_path) + "\n")
        return 2

    failures = check_clippy_window(metadata, repo_name, inputs)
    failures += check_test_window(inputs)
    if failures:
        for line in failures:
            print(line)
        return 1

    print("S1 clippy window: EMPTY (primary and fallback select the same units)")
    print("S2 test window:   EMPTY (test_workspace not enabled)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
