import re
H=re.compile(r"<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> upstream/main\n", re.S)
def resolve(f, rules):
    s=open(f).read(); i=[0]
    def rep(m):
        r=rules[i[0]] if isinstance(rules,list) else rules; i[0]+=1
        o,t=m.group(1),m.group(2)
        return r(o,t)
    s=H.sub(rep,s); assert '<<<<<<<' not in s, f; open(f,'w').write(s); print('resolved',f,i[0])
OURS=lambda o,t:o; THEIRS=lambda o,t:t; BOTH=lambda o,t:o+t; BOTH_T=lambda o,t:t+o
def phony(o,t):
    words=t.split(); extra=[w for w in o.split() if w not in words]
    return " ".join(words+extra)+"\n"
v=lambda o,t:o.replace('0.63.0','0.69.0')
resolve('.github/workflows/ci.yml',BOTH)          # our feature-gated SetFit step + their GPU-crate step
resolve('.gitignore',BOTH)
resolve('CLAUDE.md',OURS)                         # spike: keep our rewritten file; upstream edits listed in README
resolve('Cargo.toml',[THEIRS,v])                  # their comment; our entries (incl. aprender-rand) at 0.69.0
resolve('Makefile',[phony,BOTH_T,THEIRS,THEIRS,THEIRS])
resolve('README.md',THEIRS)
resolve('crates/aprender-compute/src/blis/parallel.rs',THEIRS)  # upstream's shared_b_path_available already refuses non-x86_64
resolve('crates/aprender-core/Cargo.toml',[THEIRS,BOTH])
resolve('crates/aprender-core/src/generated_contracts.rs',BOTH)
resolve('crates/aprender-mcp/src/tools/mod.rs',lambda o,t:''.join(sorted(set(o.splitlines(True)+t.splitlines(True)))))
resolve('crates/aprender-serve/src/api/router.rs',lambda o,t:o.replace('server_uptime_sec()','server_uptime_sec(state)'))
resolve('crates/aprender-train/Cargo.toml',[v,lambda o,t:t+o.split('\n',2)[2].replace('0.63.0','0.69.0')])
