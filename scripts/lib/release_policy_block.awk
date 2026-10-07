# release_policy_block.awk -- strict reader of `ladder.release_policy` (see release_policy.sh).
# -v keys="k1 k2 ..." : the keys the block must carry, each exactly once.
# Prints "key<TAB>raw value" per key. On anything it cannot read it prints "ERR<TAB>reason" and
# exits 2. No block at all: no output, exit 0.
BEGIN { n = split(keys, k, " "); for (i = 1; i <= n; i++) ok[k[i]] = 1 }
/^  release_policy:[ ]*$/ {
    if (++blocks > 1) { print "ERR\t" blocks " release_policy blocks: one ladder takes one standing policy"; bad = 1; exit }
    inb = 1; next
}
inb && /^    [^ ]/ {
    line = substr($0, 5)
    if (match(line, /^[a-z_]+: /) == 0) { print "ERR\tunreadable line in release_policy: " $0; bad = 1; exit }
    key = substr(line, 1, RLENGTH - 2); val = substr(line, RLENGTH + 1)
    if (!(key in ok)) { print "ERR\tunknown key in release_policy: " key; bad = 1; exit }
    if (key in seen) { print "ERR\tduplicate key in release_policy: " key; bad = 1; exit }
    if (val == "") { print "ERR\tempty value for release_policy." key; bad = 1; exit }
    seen[key] = 1; print key "\t" val; next
}
inb { inb = 0 }
END {
    if (bad) exit 2
    if (blocks == 1) for (i = 1; i <= n; i++) if (!(k[i] in seen)) { print "ERR\trelease_policy has no " k[i]; exit 2 }
}
