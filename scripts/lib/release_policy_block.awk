# release_policy_block.awk -- strict reader of one indent-2 block of flat keys: `ladder.release_policy`
# (see release_policy.sh) by default, or the block -v block=NAME names (release_entries).
# -v keys="k1 k2 ..." : the keys the block must carry, each exactly once.
# Prints "key<TAB>raw value" per key. On anything it cannot read it prints "ERR<TAB>reason" and
# exits 2. No block at all: no output, exit 0.
BEGIN {
    if (block == "") block = "release_policy"
    if (block !~ /^[a-z_]+$/) { print "ERR\tblock name '" block "' is not [a-z_]+"; bad = 1; exit }
    n = split(keys, k, " "); for (i = 1; i <= n; i++) ok[k[i]] = 1
}
# A key line naming the block at any other indent, or with anything after the colon, is not
# "no block": it is a block this reader cannot read.
$0 ~ ("^[ \t]*" block "[ \t]*:") && $0 !~ ("^  " block ":[ ]*$") { print "ERR\tunreadable " block " header: " $0; bad = 1; exit }
$0 ~ ("^  " block ":[ ]*$") {
    if (++blocks > 1) { print "ERR\t" blocks " " block " blocks: one file takes one " block; bad = 1; exit }
    inb = 1; next
}
inb && /^    [^ ]/ {
    line = substr($0, 5)
    if (match(line, /^[a-z_]+: /) == 0) { print "ERR\tunreadable line in " block ": " $0; bad = 1; exit }
    key = substr(line, 1, RLENGTH - 2); val = substr(line, RLENGTH + 1)
    if (!(key in ok)) { print "ERR\tunknown key in " block ": " key; bad = 1; exit }
    if (key in seen) { print "ERR\tduplicate key in " block ": " key; bad = 1; exit }
    if (val == "") { print "ERR\tempty value for " block "." key; bad = 1; exit }
    # A YAML parser reads a # after whitespace as the start of a comment; this reader would not.
    # Refuse every such shape, so the two can never disagree on a value: a quoted value must close
    # at the end of its line (nothing, not even a comment, after it), and an unquoted one may not
    # carry a # at its start or after a space or tab.
    sub(/[ \t]+$/, "", val); q = substr(val, 1, 1)
    if (q == "\"" || q == "'") {
        if (length(val) < 2 || substr(val, length(val), 1) != q || index(substr(val, 2, length(val) - 2), q) > 0) {
            print "ERR\tquoted " block "." key " does not close at the end of the line: nothing may follow its closing quote, and the value may not hold its own quote character (no \\\" or '' escapes)"; bad = 1; exit
        }
    } else if (q == "#" || val ~ /[ \t]#/) {
        print "ERR\tunquoted # in " block "." key ", which YAML reads as a comment: quote the value"; bad = 1; exit
    }
    seen[key] = 1; print key "\t" val; next
}
inb && /^[ \t]*$/ { next }  # a blank line does not end the block: a key after it is still read
inb { inb = 0 }
END {
    if (bad) exit 2
    if (blocks == 1) for (i = 1; i <= n; i++) if (!(k[i] in seen)) { print "ERR\t" block " has no " k[i]; exit 2 }
}
