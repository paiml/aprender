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
# include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../x")) is the same escape
# spelled through concat!; OUT_DIR-based concat! is a build-script product and fine.
PAT_CONCAT = re.compile(
    r'include_(?:str|bytes)!\s*\(\s*concat!\s*\(\s*env!\s*\(\s*"CARGO_MANIFEST_DIR"\s*\)\s*,\s*"([^"]+)"'
)
LINE_COMMENT = re.compile(r"//[^\n]*")
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
# `#[cfg(test)]` and `#[cfg(all(test, …))]` are test-only; `any(test, …)` is not.
CFG_TEST = re.compile(r"#\[cfg\((?:test|all\(\s*test\b[^)]*\))\)\]")
WASM_USE = re.compile(r"^\s*use\s+wasm_bindgen", re.M)


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


def strip_comments(text):
    return LINE_COMMENT.sub("", BLOCK_COMMENT.sub("", text))


def item_end(text, start):
    """Index just past the item that starts at `start`: a brace-matched body, or the next `;`."""
    brace = text.find("{", start)
    semi = text.find(";", start)
    if brace < 0 or (0 <= semi < brace):
        return len(text) if semi < 0 else semi + 1
    depth, i = 0, brace
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def host_body(text):
    """The part of a file the host verification build compiles: comments gone, every
    `#[cfg(test)]` item removed (a `mod tests { … }` block or a single item), and
    production code AFTER a test module kept — truncating at the first `#[cfg(test)]`
    would hide an escape that follows it."""
    text = strip_comments(text)
    out, pos = [], 0
    for m in CFG_TEST.finditer(text):
        if m.start() < pos:
            continue
        out.append(text[pos : m.start()])
        pos = item_end(text, m.end())
    out.append(text[pos:])
    return "".join(out)


def escapes_in(crate, root, rel_path, text):
    if is_test_file(rel_path):
        return []
    body = host_body(text)
    if WASM_USE.search(body):  # a real `use wasm_bindgen` line, not one in a comment
        print(f"SKIPPED (wasm32-only, not in the host verification build): {rel_path}", file=sys.stderr)
        return []
    found = []
    for m in PAT_DATA.finditer(body):
        rel = os.path.relpath(os.path.normpath(os.path.join(root, m.group(1))), crate)
        if rel.startswith(".."):
            found.append((rel, rel_path))
    for m in PAT_CONCAT.finditer(body):
        rel = os.path.relpath(os.path.normpath(crate + "/" + m.group(1)), crate)
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
