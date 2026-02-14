#!/usr/bin/env python3
"""Fix extracted test files by removing the outer #[cfg(test)] mod tests { ... } wrapper.

When using #[path = "foo_tests.rs"] mod tests;, the file content IS the module body.
So the extracted file should NOT contain the mod tests { } wrapper.
"""
import sys
from pathlib import Path

def fix_file(filepath):
    """Remove the #[cfg(test)] mod tests { ... } wrapper from an extracted test file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    if not lines:
        return False, "empty file"

    # Find the #[cfg(test)] and mod tests { header
    header_end = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == '#[cfg(test)]':
            continue
        if stripped.startswith('mod tests') and '{' in stripped:
            header_end = i
            break
        if stripped.startswith('mod tests'):
            # Next line might have the {
            if i + 1 < len(lines) and '{' in lines[i + 1]:
                header_end = i + 1
                break

    if header_end is None:
        return False, "no mod tests { header found"

    # Find the matching closing brace (last line of file should be })
    # Remove the last } that closes the mod tests block
    footer_start = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == '}':
            footer_start = i
            break

    if footer_start is None:
        return False, "no closing } found"

    # Extract the body between header and footer
    body = lines[header_end + 1:footer_start]

    # Dedent by one level (4 spaces or 1 tab)
    dedented = []
    for line in body:
        if line.startswith('    '):
            dedented.append(line[4:])
        elif line.startswith('\t'):
            dedented.append(line[1:])
        elif line.strip() == '':
            dedented.append('\n')
        else:
            dedented.append(line)

    with open(filepath, 'w') as f:
        f.writelines(dedented)

    return True, f"unwrapped {len(dedented)} lines"

def main():
    root = Path('.')
    fixed = 0
    skipped = 0

    for filepath in sorted(root.rglob('*_tests.rs')):
        fp = str(filepath)
        if '/target/' in fp or '/book/' in fp:
            continue

        # Only process files that were created by our extraction
        with open(filepath, 'r') as f:
            first_lines = f.readlines()[:5]

        has_wrapper = any('#[cfg(test)]' in l for l in first_lines) or \
                      any('mod tests' in l for l in first_lines)

        if not has_wrapper:
            skipped += 1
            continue

        ok, msg = fix_file(fp)
        if ok:
            fixed += 1
            print(f"  FIXED: {fp}: {msg}")
        else:
            skipped += 1
            if '--verbose' in sys.argv:
                print(f"  SKIP: {fp}: {msg}")

    # Also check *_tests_extracted.rs
    for filepath in sorted(root.rglob('*_tests_extracted.rs')):
        fp = str(filepath)
        if '/target/' in fp or '/book/' in fp:
            continue

        with open(filepath, 'r') as f:
            first_lines = f.readlines()[:5]

        has_wrapper = any('#[cfg(test)]' in l for l in first_lines) or \
                      any('mod tests' in l for l in first_lines)

        if not has_wrapper:
            skipped += 1
            continue

        ok, msg = fix_file(fp)
        if ok:
            fixed += 1
            print(f"  FIXED: {fp}: {msg}")
        else:
            skipped += 1

    print(f"\nFixed: {fixed}, Skipped: {skipped}")

if __name__ == '__main__':
    main()
