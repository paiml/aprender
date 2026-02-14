#!/usr/bin/env python3
"""Extract inline #[cfg(test)] modules from Rust files into separate files.

For files where removing the test module brings production code under 500 lines,
extracts the test module into a separate file and replaces with a #[path] reference.
"""
import os
import sys
import re
from pathlib import Path

def find_test_module_start(lines):
    """Find the last #[cfg(test)] module declaration."""
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped == '#[cfg(test)]':
            return i
    return None

def find_module_end(lines, start):
    """Find the matching closing brace for the module starting at `start`.
    Handles nested braces correctly."""
    depth = 0
    found_open = False
    for i in range(start, len(lines)):
        for ch in lines[i]:
            if ch == '{':
                depth += 1
                found_open = True
            elif ch == '}':
                depth -= 1
                if found_open and depth == 0:
                    return i
    return len(lines) - 1

def extract_test_module(filepath):
    """Extract test module from a file if it would bring it under 500 lines."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    total = len(lines)
    if total <= 500:
        return None, "already under 500 lines"

    start = find_test_module_start(lines)
    if start is None:
        return None, "no #[cfg(test)] found"

    end = find_module_end(lines, start)
    test_lines = lines[start:end + 1]
    prod_lines = lines[:start]

    # Strip trailing blank lines from production code
    while prod_lines and prod_lines[-1].strip() == '':
        prod_lines.pop()

    prod_count = len(prod_lines)
    if prod_count > 500:
        return None, f"production code still {prod_count} lines"

    return {
        'prod_lines': prod_lines,
        'test_lines': test_lines,
        'prod_count': prod_count,
        'test_count': len(test_lines),
        'start': start,
        'end': end,
    }, None

def compute_test_filename(filepath):
    """Compute the test file path based on naming conventions."""
    p = Path(filepath)
    stem = p.stem
    # If file is mod.rs, use the parent directory name
    if stem == 'mod':
        test_name = f"{p.parent.name}_tests.rs"
        return p.parent / test_name
    else:
        test_name = f"{stem}_tests.rs"
        return p.parent / test_name

def process_file(filepath, dry_run=False):
    """Process a single file: extract tests if beneficial."""
    result, err = extract_test_module(filepath)
    if err:
        return False, err

    test_filepath = compute_test_filename(filepath)

    # Check for naming conflict
    if test_filepath.exists():
        # Try _tests_extracted.rs instead
        p = Path(filepath)
        stem = p.stem if p.stem != 'mod' else p.parent.name
        test_filepath = p.parent / f"{stem}_tests_extracted.rs"
        if test_filepath.exists():
            return False, "test file already exists"

    rel_test = test_filepath.name

    if dry_run:
        print(f"  WOULD: {filepath} ({result['prod_count']+result['test_count']} -> {result['prod_count']} + {result['test_count']} in {rel_test})")
        return True, "dry run"

    # Write the test file
    with open(test_filepath, 'w') as f:
        f.writelines(result['test_lines'])

    # Write the production file with path reference
    prod = result['prod_lines']
    # Add blank line and path reference
    prod.append('\n')
    prod.append(f'#[cfg(test)]\n')
    prod.append(f'#[path = "{rel_test}"]\n')
    prod.append(f'mod tests;\n')

    with open(filepath, 'w') as f:
        f.writelines(prod)

    return True, f"{result['prod_count']+result['test_count']} -> {result['prod_count']} + {result['test_count']} in {rel_test}"

def main():
    dry_run = '--dry-run' in sys.argv
    verbose = '--verbose' in sys.argv or dry_run

    root = Path('.')
    processed = 0
    skipped = 0
    errors = 0

    for filepath in sorted(root.rglob('*.rs')):
        fp = str(filepath)
        if '/target/' in fp or '/book/' in fp:
            continue

        ok, msg = process_file(fp, dry_run=dry_run)
        if ok:
            processed += 1
            if verbose:
                print(f"  OK: {fp}: {msg}")
        else:
            skipped += 1
            if verbose and 'already under' not in msg:
                pass  # Don't spam with already-ok files

    action = "Would process" if dry_run else "Processed"
    print(f"\n{action}: {processed} files, skipped: {skipped}")

if __name__ == '__main__':
    main()
