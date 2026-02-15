#!/usr/bin/env python3
"""Split Rust files over 500 lines using pmat extract for function boundaries.

Uses pmat extract --list to get AST-aware item boundaries,
then splits at those boundaries into include!() fragments.

Key design decisions:
- include!() is textual inclusion — fragments inherit parent scope
- include!() CANNOT span block boundaries (impl/mod/trait)
- include!() CANNOT be placed inside impl/mod blocks directly
- For large containers, extract methods into separate impl/mod wrapper files
- Doc comments + attributes belong to their item (never orphaned)
- Files in src/bin/ are SKIPPED (part files become separate binaries)
- File-spanning modules (mod tests wrapping entire file) are treated as containers
"""

import json
import os
import subprocess
import sys
from pathlib import Path

MAX_LINES = 450  # Target max lines per chunk (margin below 500)
MIN_SPLIT_LINES = 500  # Only split files over this


def get_files_over_limit(src_dir: str) -> list[tuple[str, int]]:
    """Find all .rs files over MIN_SPLIT_LINES, excluding src/bin/."""
    result = []
    for root, _, files in os.walk(src_dir):
        if "/bin" in root or root.endswith("/bin"):
            continue
        for f in files:
            if f.endswith(".rs"):
                path = os.path.join(root, f)
                with open(path) as fh:
                    lines = sum(1 for _ in fh)
                if lines > MIN_SPLIT_LINES:
                    result.append((path, lines))
    result.sort(key=lambda x: -x[1])
    return result


def pmat_extract(filepath: str) -> list[dict]:
    """Get item boundaries from pmat extract --list."""
    try:
        proc = subprocess.run(
            ["pmat", "extract", "--list", filepath],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            return []
        data = json.loads(proc.stdout)
        if isinstance(data, dict) and "items" in data:
            return data["items"]
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, subprocess.TimeoutExpired):
        return []


def effective_start(lines: list[str], start_line: int) -> int:
    """Walk backwards from start_line to include doc comments and attributes.

    Returns 1-based line number of the effective start.
    """
    idx = start_line - 2  # 0-based, one line before item
    while idx >= 0:
        s = lines[idx].strip()
        if (s.startswith("//")  # Any comment (doc, regular, section separator)
                or s.startswith("#[")  # Attributes
                or s == ""):  # Blank lines between attrs/comments and item
            idx -= 1
        else:
            break
    return idx + 2  # back to 1-based


def is_include_hub(lines: list[str]) -> bool:
    """Check if this file is primarily include!() directives."""
    inc = sum(1 for l in lines if "include!(" in l)
    return inc > 5


def find_existing_parts(filepath: str) -> list[str]:
    base = Path(filepath)
    return sorted(base.parent.glob(f"{base.stem}_part_*.rs"))


def get_next_part_num(filepath: str) -> int:
    parts = find_existing_parts(filepath)
    if not parts:
        return 2
    nums = []
    for p in parts:
        try:
            nums.append(int(p.stem.split("_part_")[1]))
        except (IndexError, ValueError):
            pass
    return max(nums, default=1) + 1


def build_top_level_split(items: list[dict], lines: list[str], total_lines: int):
    """Build split plan using ONLY top-level item boundaries.

    Returns list of (start_0based, end_0based, chunk_lines, None) tuples.

    CRITICAL: File-spanning modules (like `mod tests {}` wrapping the entire file)
    are treated as containers — their children are NOT top-level items.
    """
    # Identify ALL containers (impl blocks, mod blocks, functions that span >1 line)
    # Include file-spanning modules so their children are marked as nested
    # Include functions too — items defined inside functions (local structs/enums)
    # should NOT be used as split points
    containers = []
    for item in items:
        if item["type"] in ("class", "impl", "module", "function"):
            span = item["end_line"] - item["start_line"]
            if span > 1:
                containers.append(item)

    container_ranges = [(c["start_line"], c["end_line"]) for c in containers]

    def is_nested(start, end):
        for cs, ce in container_ranges:
            if start > cs and end <= ce:
                return True
        return False

    # Get effective start lines for top-level items only
    # Skip file-spanning modules as split points (they wrap everything)
    top_starts = []
    for item in items:
        if item["type"] == "module" and item["end_line"] - item["start_line"] > total_lines - 20:
            continue
        if is_nested(item["start_line"], item["end_line"]):
            continue

        eff = effective_start(lines, item["start_line"])
        top_starts.append(eff)

    top_starts = sorted(set(top_starts))
    if not top_starts:
        return []

    # Greedy partition
    splits = []
    chunk_start = 1
    last_valid = None

    for start in top_starts:
        if start - chunk_start <= MAX_LINES:
            last_valid = start
        else:
            if last_valid and last_valid > chunk_start:
                splits.append(last_valid)
                chunk_start = last_valid
                if start - chunk_start <= MAX_LINES:
                    last_valid = start
                else:
                    splits.append(start)
                    chunk_start = start
                    last_valid = None
            else:
                splits.append(start)
                chunk_start = start
                last_valid = None

    remaining = total_lines - chunk_start + 1
    if remaining > MAX_LINES and last_valid and last_valid > chunk_start:
        splits.append(last_valid)

    splits.append(total_lines + 1)

    # Build chunks
    chunks = []
    prev = 0
    for sp in splits:
        sp_idx = sp - 1
        chunk = lines[prev:sp_idx]
        if chunk:
            chunks.append((prev, sp_idx, chunk, None))
        prev = sp_idx

    return chunks


def build_container_split(items: list[dict], lines: list[str], container: dict):
    """Build split plan for items INSIDE a large container (impl/mod block).

    Strategy: partition DIRECT children into groups of <=MAX_LINES, extract all but
    the last group (which stays in the parent file).

    CRITICAL: Only extract direct children, not items nested in sub-containers.
    """
    c_start = container["start_line"]
    c_end = container["end_line"]

    # Find sub-containers within this container
    sub_containers = []
    for item in items:
        if (item["type"] in ("class", "impl", "module", "trait")
                and item["start_line"] > c_start
                and item["end_line"] < c_end):
            sub_containers.append((item["start_line"], item["end_line"]))

    def in_sub_container(start, end):
        """Check if an item is STRICTLY inside a sub-container."""
        for sc_start, sc_end in sub_containers:
            if start > sc_start and end <= sc_end:
                return True
        return False

    # Find DIRECT children of the container
    inner_items = []
    if container["type"] == "module":
        # For modules: include structs, impls, enums, traits, and standalone functions
        # but NOT items nested inside sub-containers
        for item in items:
            if (item["start_line"] > c_start and item["end_line"] < c_end
                    and not in_sub_container(item["start_line"], item["end_line"])):
                eff = effective_start(lines, item["start_line"])
                inner_items.append({
                    "eff_start": eff,
                    "start": item["start_line"],
                    "end": item["end_line"],
                    "name": item["name"],
                    "type": item["type"],
                })
    else:
        # For impl blocks: only direct method children
        for item in items:
            if (item["start_line"] > c_start and item["end_line"] < c_end
                    and item["type"] == "function"
                    and not in_sub_container(item["start_line"], item["end_line"])):
                eff = effective_start(lines, item["start_line"])
                inner_items.append({
                    "eff_start": eff,
                    "start": item["start_line"],
                    "end": item["end_line"],
                    "name": item["name"],
                    "type": item["type"],
                })

    inner_items.sort(key=lambda x: x["eff_start"])
    if len(inner_items) < 2:
        return None

    # Greedy partition into groups of <=MAX_LINES
    groups = []
    current_group = []
    group_start = inner_items[0]["eff_start"]

    for item in inner_items:
        span = item["end"] - group_start + 1
        if span > MAX_LINES and current_group:
            groups.append((group_start, current_group[-1]["end"], current_group))
            current_group = [item]
            group_start = item["eff_start"]
        else:
            current_group.append(item)

    if current_group:
        groups.append((group_start, current_group[-1]["end"], current_group))

    if len(groups) < 2:
        return None

    # Extract all groups EXCEPT the last one (last stays in parent)
    parts = []
    for start_line, end_line, _ in groups[:-1]:
        parts.append((start_line - 1, end_line))  # 0-based start, 1-based end

    return {
        "container": container,
        "parts": parts,
        "first_inner": inner_items[0]["eff_start"],
    }


def get_container_header(lines: list[str], container: dict) -> str:
    """Extract the impl/mod header line."""
    start_idx = container["start_line"] - 1
    header = lines[start_idx].rstrip()
    idx = start_idx
    while "{" not in header and idx < len(lines) - 1:
        idx += 1
        header += " " + lines[idx].strip()
    return header.rstrip()


def split_file(filepath: str, total_lines: int, dry_run: bool = False) -> bool:
    """Split a single file using pmat extract boundaries (top-level only)."""
    rel = os.path.relpath(filepath, start=os.getcwd())
    pfx = "[DRY] " if dry_run else ""
    print(f"\n{pfx}{rel} ({total_lines} lines)")

    lines = list(open(filepath))

    if is_include_hub(lines):
        print(f"  skip: include hub")
        return False

    items = pmat_extract(rel)
    if not items:
        print(f"  skip: no pmat items")
        return False

    chunks = build_top_level_split(items, lines, total_lines)
    if len(chunks) <= 1:
        print(f"  skip: no valid top-level splits")
        return False

    sizes = [c[1] - c[0] for c in chunks]
    print(f"  -> {len(chunks)} chunks: {sizes}")

    if dry_run:
        over = [s for s in sizes if s > 500]
        if over:
            print(f"  WARNING: {len(over)} chunks still >500: {over}")
        return True

    next_part = get_next_part_num(filepath)
    base = Path(filepath)
    stem = base.stem
    parent = base.parent

    part_includes = []
    for i, (start, end, chunk, ctx) in enumerate(chunks[1:], start=next_part):
        part_name = f"{stem}_part_{i:02d}.rs"
        part_path = parent / part_name
        with open(part_path, "w") as f:
            f.write("".join(chunk))
        part_includes.append(f'include!("{part_name}");')

    with open(filepath, "w") as f:
        f.write("".join(chunks[0][2]))
        f.write("\n")
        for inc in part_includes:
            f.write(inc + "\n")

    return True


def split_inside_container(filepath: str, total_lines: int, dry_run: bool = False) -> bool:
    """Split large containers by extracting items into separate files.

    For `impl Foo { ... }`: each part file gets its own `impl Foo { ... }` wrapper,
    include!()'d at top level (multiple impl blocks are legal in Rust).

    For `mod foo { ... }`: part files have RAW items (no wrapper),
    include!()'d INSIDE the module block (mod can only be defined once).
    """
    rel = os.path.relpath(filepath, start=os.getcwd())
    pfx = "[DRY] " if dry_run else ""

    lines = list(open(filepath))
    current_lines = len(lines)

    if current_lines <= MIN_SPLIT_LINES:
        return False

    if is_include_hub(lines):
        return False

    items = pmat_extract(rel)
    if not items:
        return False

    # Find large containers
    large_containers = []
    for item in items:
        if item["type"] in ("class", "impl", "module"):
            span = item["end_line"] - item["start_line"]
            if span > MAX_LINES:
                large_containers.append(item)

    if not large_containers:
        print(f"\n{pfx}{rel} ({current_lines} lines)")
        print(f"  skip: no large containers to split inside")
        return False

    # Process containers (try largest first, fall back to smaller ones)
    large_containers.sort(key=lambda c: c["end_line"] - c["start_line"], reverse=True)
    container = None
    plan = None
    for candidate in large_containers:
        plan = build_container_split(items, lines, candidate)
        if plan:
            container = candidate
            break

    if not plan or not container:
        print(f"\n{pfx}{rel} ({current_lines} lines)")
        print(f"  skip: no containers have valid inner splits")
        return False

    is_module = container["type"] == "module"
    part_sizes = [end - start for start, end in plan["parts"]]
    print(f"\n{pfx}{rel} ({current_lines} lines)")
    print(f"  container: {container['type']} {container['name']} "
          f"(lines {container['start_line']}-{container['end_line']})")
    print(f"  -> {len(plan['parts'])} inner parts: {part_sizes}")

    if dry_run:
        over = [s for s in part_sizes if s > 500]
        if over:
            print(f"  WARNING: {len(over)} inner parts still >500: {over}")
        return True

    # Get container header for wrapping (only used for impl blocks)
    header = get_container_header(lines, container)

    # Check for #[cfg(test)] attribute before mod blocks
    cfg_attr = ""
    if is_module:
        eff = effective_start(lines, container["start_line"])
        for i in range(eff - 1, container["start_line"] - 1):
            line = lines[i].strip()
            if line.startswith("#[cfg(test)]"):
                cfg_attr = "#[cfg(test)]\n"
                break

    next_part = get_next_part_num(filepath)
    base = Path(filepath)
    stem = base.stem
    parent = base.parent

    # Extract parts and build replacement list
    replacements = []
    include_lines = []

    for i, (start_0, end_1based) in enumerate(plan["parts"], start=next_part):
        part_name = f"{stem}_part_{i:02d}.rs"
        part_path = parent / part_name

        chunk = lines[start_0:end_1based]

        with open(part_path, "w") as f:
            if is_module:
                # Modules: raw items (no wrapper) — included INSIDE the mod block
                f.write("".join(chunk))
            else:
                # Impl blocks: own impl wrapper — included at top level
                f.write(cfg_attr)
                f.write(header + "\n")
                f.write("".join(chunk))
                f.write("}\n")

        include_lines.append(f'include!("{part_name}");')
        replacements.append((start_0, end_1based))

    # Remove extracted ranges (work backwards to preserve line numbers)
    for start_0, end_1based in reversed(replacements):
        lines[start_0:end_1based] = []

    content = "".join(lines)
    lines = content.splitlines(True)

    if is_module:
        # For modules: insert include!() INSIDE the module, before its closing }
        # Find the module's closing brace by matching from its opening line
        mod_start_idx = None
        for idx, line in enumerate(lines):
            if container["name"] in line and "mod " in line and "{" in line:
                mod_start_idx = idx
                break

        if mod_start_idx is not None:
            # Brace-count to find matching closing }
            depth = 0
            close_idx = None
            for idx in range(mod_start_idx, len(lines)):
                for ch in lines[idx]:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            close_idx = idx
                            break
                if close_idx is not None:
                    break

            if close_idx is not None:
                # Insert includes before the closing }
                inc_text = "\n".join(include_lines) + "\n"
                lines.insert(close_idx, inc_text)

        with open(filepath, "w") as f:
            f.write("".join(lines))
    else:
        # For impl blocks: append includes at top level after the file
        with open(filepath, "w") as f:
            f.write("".join(lines))
            f.write("\n")
            for inc in include_lines:
                f.write(inc + "\n")

    new_lines = sum(1 for _ in open(filepath))
    print(f"  result: {current_lines} -> {new_lines} lines")
    return True


def verify_build() -> bool:
    print("\n--- cargo check ---")
    proc = subprocess.run(
        ["cargo", "check", "-j", "4"],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode == 0:
        print("OK")
        return True
    errs = [l for l in proc.stderr.split("\n") if "error" in l.lower()][:10]
    print("FAILED:")
    for l in errs:
        print(f"  {l}")
    return False


def main():
    dry_run = "--dry-run" in sys.argv
    inner = "--inner" in sys.argv
    batch_size = 50
    single_file = None

    for arg in sys.argv[1:]:
        if arg.startswith("--batch="):
            batch_size = int(arg.split("=")[1])
        elif not arg.startswith("--"):
            single_file = arg

    if single_file:
        with open(single_file) as f:
            lc = sum(1 for _ in f)
        files = [(single_file, lc)]
    else:
        files = get_files_over_limit("src")

    print(f"Found {len(files)} files over {MIN_SPLIT_LINES} lines")

    split_count = 0
    batch_count = 0

    for filepath, total_lines in files:
        if inner:
            did_split = split_inside_container(filepath, total_lines, dry_run=dry_run)
        else:
            did_split = split_file(filepath, total_lines, dry_run=dry_run)

        if did_split:
            split_count += 1
            batch_count += 1

        if not dry_run and batch_count >= batch_size:
            if not verify_build():
                print(f"\nFAILED after {split_count} files.")
                sys.exit(1)
            batch_count = 0

    if not dry_run and split_count > 0 and batch_count > 0:
        if not verify_build():
            print(f"\nFINAL FAILED after {split_count} files.")
            sys.exit(1)

    print(f"\n{split_count}/{len(files)} files split.")

    if not dry_run:
        remaining = get_files_over_limit("src")
        if remaining:
            print(f"\nStill over limit: {len(remaining)}")
            for fp, lc in remaining[:20]:
                print(f"  {os.path.relpath(fp)}: {lc}")


if __name__ == "__main__":
    main()
