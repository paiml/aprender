with open("crates/apr-cli/src/commands/dry_sampling_lint.rs", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "params.map(|o| format!(\"{o:?}\"))" in line and '"params":' in line:
        lines[i] = line.replace('params.map(|o| format!("{o:?}"))', 'params')
    if "identity.map(|o| format!(\"{o:?}\"))" in line and '"identity":' in line:
        lines[i] = line.replace('identity.map(|o| format!("{o:?}"))', 'identity')
    if "match_len.map(|o| format!(\"{o:?}\"))" in line and '"match_len":' in line:
        lines[i] = line.replace('match_len.map(|o| format!("{o:?}"))', 'match_len')
    if "penalty.map(|o| format!(\"{o:?}\"))" in line and '"penalty":' in line:
        lines[i] = line.replace('penalty.map(|o| format!("{o:?}"))', 'penalty')
    if "monotone.map(|o| format!(\"{o:?}\"))" in line and '"monotone":' in line:
        lines[i] = line.replace('monotone.map(|o| format!("{o:?}"))', 'monotone')

with open("crates/apr-cli/src/commands/dry_sampling_lint.rs", "w") as f:
    f.writelines(lines)
