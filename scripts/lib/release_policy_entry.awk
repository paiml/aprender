# release_policy_entry.awk -- copy the ladder, inserting one `emergency_scopes` entry built from the
# policy right after the top-level `emergency_scopes:` line (see release_policy.sh). Values come
# from ENVIRON, not -v: awk -v rewrites backslash escapes inside the quote. Exit 1 when the ladder
# has no such line.
BEGIN {
    name = ENVIRON["RP_N"]; v = ENVIRON["RP_V"]; date = ENVIRON["RP_D"]; quote = ENVIRON["RP_Q"]
    hosts = ENVIRON["RP_H"]; thinking = ENVIRON["RP_T"]
}
{ print }
/^  emergency_scopes:[ ]*$/ && !done {
    print "    - name: " name
    print "      release: \"" v "\""
    print "      date: " date
    print "      quote: " quote
    print "      hosts: " hosts
    print "      thinking: " thinking
    done = 1
}
END { exit done ? 0 : 1 }
