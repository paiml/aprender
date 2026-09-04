"""List every include!() target in a crate, resolved against the including file.
Separate file rather than an inline heredoc: bashrs parses an embedded heredoc
as shell, so python assignments read as SC1007 "space after =" -- eight phantom
errors. Same reason assertions_exclude.awk and workflow_path_filters.py live
here.
argv: <crate-dir> [--escapes]
stdout: "<crate-relative target>\t<crate-relative including file>" per line

--escapes (PMAT-958): instead of include!() targets, list every include_str!/
include_bytes! target in NON-TEST, HOST-COMPILED code whose path escapes the
crate directory. Such a file can never be in the package tarball, so `cargo
publish` fails its verification build (aprender-test-lib 0.65.1:
`../../../../scripts/perf-matrix.yaml`, cascade stuck at 67/74). Skipped, and
why: test code (a `#[cfg(test)]` module is not compiled by the verification
build; the crate-local *_tests.rs files legitimately read
`../../../../contracts/*.yaml`) and wasm32-only files (`use wasm_bindgen`),
which the host verification build never compiles either -- those are printed
on stderr as SKIPPED so the residual is visible, never silent.
"""
import os
import re
import sys

PAT_INCLUDE = re.compile(r'include!\s*\(\s*"([^"]+)"\s*\)')
PAT_DATA = re.compile(r'include_(?:str|bytes)!\s*\(\s*"([^"]+)"\s*\)')
LINE_COMMENT = re.compile(r"//[^\n]*")


def is_test_file(rel):
    base = os.path.basename(rel)
    return base.endswith("_tests.rs") or base == "tests.rs" or "/tests/" in rel or "/tests_" in rel


def rust_files(src):
    for root, _dirs, files in os.walk(src):
        for fn in files:
            if fn.endswith(".rs"):
                yield root, os.path.join(root, fn)


def read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def host_body(text):
    """The part of a file the host verification build compiles: no test module, no comments."""
    cut = text.find("#[cfg(test)]")
    body = text if cut < 0 else text[:cut]
    return LINE_COMMENT.sub("", body)


def escapes_in(crate, root, rel_path, text):
    if is_test_file(rel_path):
        return []
    if "use wasm_bindgen" in text:
        print(f"SKIPPED (wasm32-only, not in the host verification build): {rel_path}", file=sys.stderr)
        return []
    found = []
    for m in PAT_DATA.finditer(host_body(text)):
        rel = os.path.relpath(os.path.normpath(os.path.join(root, m.group(1))), crate)
        if rel.startswith(".."):
            found.append((rel, rel_path))
    return found


def includes_in(crate, root, rel_path, text):
    return [
        (os.path.relpath(os.path.normpath(os.path.join(root, m.group(1))), crate), rel_path)
        for m in PAT_INCLUDE.finditer(text)
    ]


def main():
    crate = sys.argv[1]
    escape_mode = "--escapes" in sys.argv[2:]
    judge = escapes_in if escape_mode else includes_in
    for root, path in rust_files(os.path.join(crate, "src")):
        text = read(path)
        if text is None:
            continue
        for target, from_file in judge(crate, root, os.path.relpath(path, crate), text):
            print(f"{target}\t{from_file}")


main()
