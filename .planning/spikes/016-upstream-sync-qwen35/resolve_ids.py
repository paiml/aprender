"""Resolve conflict hunks where upstream == ours + inserted `- id: X` lines (the 1edcb5521 obligation-id sweep)."""
import re, sys, subprocess
files = subprocess.run(["git", "diff", "--name-only", "--diff-filter=U"], capture_output=True, text=True).stdout.split()
H = re.compile(r"<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> upstream/main\n", re.S)
def strip_ids(t):
    t = re.sub(r"^(\s*)- id: [^\n]+\n\s*", r"\1- ", t, flags=re.M)
    return t.replace(" --features setfit", "")
for f in files:
    if not f.endswith((".yaml", ".yml")): continue
    s = open(f).read(); left = 0
    def rep(m):
        global left
        ours, theirs = m.group(1), m.group(2)
        if strip_ids(theirs) == ours: return theirs
        left += 1; return m.group(0)
    s2 = H.sub(rep, s)
    open(f, "w").write(s2)
    print(f"{left:3d} unresolved  {f}")
