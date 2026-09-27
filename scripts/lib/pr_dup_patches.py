#!/usr/bin/env python3
"""DUP-001 (#4536): find changes a PR shares with another open PR, by patch-id.

Invoked by scripts/check_pr_duplicate_patches.sh; see its header for the rules.
This file is the engine: it turns a PR into a set of `git patch-id --stable`
ids -- one per commit and one per hunk -- and intersects that set with the
ids of every other open PR.

Two levels, because each misses what the other catches:
  * COMMIT ids catch a whole commit carried by two PRs, whatever its message
    (patch-id never reads the message) -- a pure copy or a cherry-pick.
  * HUNK ids catch a copied commit that was then edited (amended with more
    hunks, or squashed with other work): the commit id changes, the untouched
    hunks keep theirs. Hunks come from every commit AND from the PR's
    cumulative diff against its merge-base, so a copy squashed into a
    larger commit on one side still meets its twin.

patch-id --stable hashes the diff without line numbers or whitespace, and
the file path IS part of the hash (measured: the same one-line edit in two
files gives two ids), so the same edit to a different file is not a
duplicate.

Not counted, because two PRs legitimately produce identical bytes there:
  * GENERATED paths (GENERATED_PATHS) -- every regen writes the same output.
  * README.md hunks whose changed lines all sit inside a generated
    `<!-- X_START ... -->`..`<!-- X_END -->` block (readme_sync.sh,
    release_section.rs).
  * whitespace-only hunks, which carry no change patch-id can see.

A PR body line `stacked-on: #N` exempts the pair (this PR, #N) in both
directions: a stack shares its lower PR's commits by construction.
"""
import json
import re
import subprocess
import sys

GENERATED_PATHS = frozenset({
    "Cargo.lock",
    "docs/roadmaps/roadmap.yaml",
    "contracts/census.json",
    "contracts/contracts.nt",
    "contracts/shapes.ttl",
})
README = "README.md"
BLOCK_START = re.compile(r"<!--\s*([A-Z][A-Z0-9_]*)_START\b")
BLOCK_END = re.compile(r"\b([A-Z][A-Z0-9_]*)_END\s*-->")
STACKED_ON = re.compile(r"^\s*stacked-on:\s*#(\d+)\s*$", re.I | re.M)
HUNK_HDR = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
ZERO = "0" * 40
# Detail rows printed per PR pair; the SUMMARY line always carries the full
# count. A fold can share ~1,000 hunks with its twin, which buries every other
# guard's row in the guard-tree log without adding a fact the summary lacks.
DETAIL_PER_PAIR = 20


def git(repo, *args, stdin=None):
    r = subprocess.run(["git", "-C", repo, *args], input=stdin,
                       capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        # Exit 2, never 1: an engine that could not read a PR must not be
        # reported as a PR that carries a duplicate.
        print(f"DUP-001: git {' '.join(args[:3])} failed: {r.stderr.strip()}", file=sys.stderr)
        sys.exit(2)
    return r.stdout


def stacked_on(body):
    return {int(n) for n in STACKED_ON.findall(body or "")}


def block_lines(text):
    """1-based line numbers lying inside a generated README block.

    A line carrying both markers (an inline block) counts; so does every line
    of a multi-line block, markers included."""
    inside, out, depth = set(), None, 0
    for i, line in enumerate(text.splitlines(), 1):
        opens, closes = len(BLOCK_START.findall(line)), len(BLOCK_END.findall(line))
        if opens or closes or depth:
            inside.add(i)
        depth = max(0, depth + opens - closes)
    return inside


def split_files(diff):
    """Yield (path, header_lines, [hunk_lines...]) for each file in a diff."""
    cur = None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            if cur:
                yield cur
            cur = [None, [line], []]
        elif cur is None:
            continue
        elif line.startswith("@@"):
            cur[2].append([line])
        elif cur[2]:
            cur[2][-1].append(line)
        else:
            cur[1].append(line)
            if line.startswith("+++ "):
                p = line[4:]
                cur[0] = p[2:] if p.startswith("b/") else cur[0]
            elif line.startswith("--- ") and cur[0] is None:
                p = line[4:]
                cur[0] = p[2:] if p.startswith("a/") else None
    if cur:
        yield cur


class Generated:
    """Decides whether a README hunk is generated, reading the file at the
    commit (new side) and at its parent (old side)."""

    def __init__(self, repo):
        self.repo, self.cache = repo, {}

    def lines(self, rev):
        if rev not in self.cache:
            r = subprocess.run(["git", "-C", self.repo, "show", f"{rev}:{README}"],
                               capture_output=True, text=True, errors="replace")
            self.cache[rev] = block_lines(r.stdout) if r.returncode == 0 else set()
        return self.cache[rev]

    def hunk(self, old_rev, new_rev, hunk):
        m = HUNK_HDR.match(hunk[0])
        if not m:
            return False
        o, n = int(m.group(1)), int(m.group(3))
        old_in, new_in = self.lines(old_rev), self.lines(new_rev)
        for line in hunk[1:]:
            tag = line[:1]
            if tag == "-":
                if o not in old_in:
                    return False
                o += 1
            elif tag == "+":
                if n not in new_in:
                    return False
                n += 1
            elif tag == " ":
                o, n = o + 1, n + 1
        return True


def blank_only(hunk):
    """True when the hunk changes whitespace only: its removed and added
    lines are equal once all whitespace is dropped."""
    def side(tag):
        return "".join("".join(l[1:].split()) for l in hunk[1:] if l[:1] == tag)
    return side("-") == side("+")


def hunk_text(header, hunk):
    return "\n".join(header + hunk) + "\n"


def patch_ids(repo, patches):
    """[patch text] -> [patch-id, or None for a patch with no change], in order."""
    if not patches:
        return []
    out = [None] * len(patches)
    feed = "".join(f"commit {i:040x}\n{p}" for i, p in enumerate(patches))
    for row in git(repo, "patch-id", "--stable", stdin=feed).splitlines():
        pid, key = row.split()
        out[int(key, 16)] = pid
    return out


def commits_of(log):
    """[(sha, patch text)] from `git log -p --format='commit %H'`."""
    commits, cur = [], None
    for line in log.splitlines(keepends=True):
        if line.startswith("commit ") and len(line.strip()) == 47:
            cur = [line.split()[1], []]
            commits.append(cur)
        elif cur:
            cur[1].append(line)
    return [(sha, "".join(body)) for sha, body in commits]


def landed_ids(repo, mb, base, paths):
    """Hunk patch-ids of every change `base` gained since the PR forked (mb),
    on the PR's own paths. A change already on base is no PR's duplicate,
    whatever sha carries it: two stale PRs that both still carry a fix which
    has since landed are behind main, not copies of each other."""
    ids, chunk = set(), 200  # paths per `git log` call: keep argv bounded
    for i in range(0, len(paths), chunk):
        log = git(repo, "log", "-p", "--no-merges", "--format=commit %H",
                  f"{mb}..{base}", "--", *paths[i:i + chunk])
        texts = [hunk_text(header, h)
                 for _, diff in commits_of(log)
                 for _, header, hunks in split_files(diff) for h in hunks]
        ids.update(pid for pid in patch_ids(repo, texts) if pid)
    return ids


def ids_of(repo, base, head, gen):
    """{patch_id: [where, ...]} for one PR; `where` is (level, sha, path, hunk)."""
    commits = commits_of(git(repo, "log", "-p", "--no-merges", "--right-only",
                             "--format=commit %H", f"{base}...{head}"))
    mb = git(repo, "merge-base", base, head).strip()
    sources = [(sha, f"{sha}^", sha, body) for sha, body in commits]
    sources.append(("cumulative", mb, head,
                    git(repo, "diff", "--no-color", f"{mb}..{head}")))

    items = []  # (label, path, header, hunk) that count
    for label, old_rev, new_rev, diff in sources:
        for path, header, hunks in split_files(diff):
            if path in GENERATED_PATHS:
                continue
            for h in hunks:
                if blank_only(h) or (path == README and gen.hunk(old_rev, new_rev, h)):
                    continue
                items.append((label, path, header, h))

    landed = landed_ids(repo, mb, base, sorted({it[1] for it in items}))
    out, per_commit = {}, {}
    for (label, path, header, h), pid in zip(
            items, patch_ids(repo, [hunk_text(hd, h) for _, _, hd, h in items])):
        if pid is None or pid in landed:
            continue
        out.setdefault(pid, []).append(("hunk", label, path, h[0]))
        if label != "cumulative":
            per_commit.setdefault(label, []).append(hunk_text(header, h))
    # The commit id is taken over the commit's COUNTED hunks only, so a commit
    # that also touched a generated file, or re-carried a landed hunk, still
    # matches its twin.
    labels = list(per_commit)
    for label, pid in zip(labels, patch_ids(repo, ["".join(per_commit[l]) for l in labels])):
        if pid:
            out.setdefault(pid, []).append(("commit", label, "", ""))
    return out


def find_dups(repo, base, subject, others):
    """subject/others: dicts {number, head, body, base?}. Returns report rows."""
    gen = Generated(repo)
    mine = ids_of(repo, base, subject["head"], gen)
    mine_stack = stacked_on(subject.get("body"))
    rows = []
    for o in others:
        if o["number"] == subject["number"]:
            continue
        if o["number"] in mine_stack or subject["number"] in stacked_on(o.get("body")):
            continue
        theirs = ids_of(repo, o.get("base") or base, o["head"], gen)
        seen = set()
        for pid in sorted(set(mine) & set(theirs)):
            a, b = mine[pid][0], theirs[pid][0]
            key = (a[1], a[2], a[3], b[1])
            if key in seen:
                continue
            seen.add(key)
            rows.append({"pr": subject["number"], "other": o["number"], "patch_id": pid,
                         "level": a[0], "sha": a[1], "path": a[2], "hunk": a[3],
                         "other_level": b[0], "other_sha": b[1]})
    return rows


def summarize(rows):
    """{other_pr: {commits, same_sha, hunks, files}} -- one line per PR pair.

    `same_sha` counts commits both PRs carry under the SAME sha: one branch was
    built on the other (an undeclared stack), not a copy. The remedy differs --
    declare `stacked-on:` or rebase off it -- so the summary names it."""
    out = {}
    for r in rows:
        s = out.setdefault(r["other"], {"commits": 0, "same_sha": 0, "hunks": 0, "files": set()})
        if r["level"] == "commit":
            s["commits"] += 1
            if r["sha"] == r["other_sha"]:
                s["same_sha"] += 1
        else:
            s["hunks"] += 1
            s["files"].add(r["path"])
    return out


def render(rows, per_pair=DETAIL_PER_PAIR):
    shown = {}
    for r in rows:
        n = shown[r["other"]] = shown.get(r["other"], 0) + 1
        if n > per_pair:
            continue
        tail = f" {r['path']} {r['hunk']}" if r["level"] == "hunk" else ""
        print(f"DUP-001: PR #{r['pr']} {r['level']} {r['sha'][:10]}{tail} "
              f"duplicates PR #{r['other']} {r['other_level']} {r['other_sha'][:10]} "
              f"(patch-id {r['patch_id'][:12]})")
    for other, s in sorted(summarize(rows).items()):
        hidden = shown[other] - per_pair
        more = f"; {hidden} more row(s) not printed" if hidden > 0 else ""
        stack = (f" -- {s['same_sha']} under the SAME sha: one branch is built on the"
                 f" other; declare 'stacked-on: #{other}' or rebase off it"
                 if s["same_sha"] else "")
        print(f"DUP-001 SUMMARY: PR #{rows[0]['pr']} vs PR #{other}: {s['commits']} commit(s),"
              f" {s['hunks']} hunk(s) in {len(s['files'])} file(s){more}{stack}")


def main(argv):
    if len(argv) != 3 or argv[0] != "check":
        sys.exit("usage: pr_dup_patches.py check <repo> <spec.json>")
    with open(argv[2]) as f:
        spec = json.load(f)
    rows = find_dups(argv[1], spec["base"], spec["subject"], spec["others"])
    render(rows)
    return 1 if rows else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
