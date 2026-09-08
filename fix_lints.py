import re
import os

replacements = {
    r'shape\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'shape',
    r'schema\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'schema',
    r'passthrough\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'passthrough',
    r'bound\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'bound',
    r'grammar\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'grammar',
    r'format!\("\{:\?\}", schema\)': 'schema',
    r'format!\("\{:\?\}", allowlist\)': 'allowlist',
    r'format!\("\{:\?\}", outcome\)': 'outcome',
    r'greedy\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'greedy',
    r'timeout_outcome\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'timeout_outcome',
    r'success_outcome\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'success_outcome',
    r'exit_outcome\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'exit_outcome',
    r'required\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'required',
    r'ct\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'ct',
    r'rows\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'rows',
    r'mask\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'mask',
    r'html\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'html',
    r'doc_link\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'doc_link',
    r'exit\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'exit',
    r'params\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'params',
    r'identity\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'identity',
    r'match_len\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'match_len',
    r'penalty\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'penalty',
    r'monotone\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'monotone',
    r'err\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'err',
    r'cov\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'cov',
    r'parity\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'parity',
    r'provenance\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'provenance',
    r'head_dim\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'head_dim',
    r'row_count\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'row_count',
    r'determinism\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'determinism',
    r'span\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'span',
    r'attrs\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'attrs',
    r'trace\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'trace',
    r'format!\("\{:\?\}", invariants\)': 'invariants',
    r'format!\("\{:\?\}", size\)': 'size',
    r'json_out\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'json_out',
    r'err_diag\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'err_diag',
    r'masking\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'masking',
    r'range\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'range',
    r'mass\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'mass',
    r'sort\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'sort',
    r'renorm\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)': 'renorm',
}

directory = 'crates/apr-cli/src/commands'
for filename in os.listdir(directory):
    if not filename.endswith('_lint.rs'):
        continue
    filepath = os.path.join(directory, filename)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    in_json = False
    changed = False
    for i, line in enumerate(lines):
        if 'json!({' in line.replace(' ', ''):
            in_json = True
        
        if in_json:
            for pattern, repl in replacements.items():
                if re.search(pattern, line):
                    lines[i] = re.sub(pattern, repl, line)
                    changed = True
                    break
                    
        if in_json and '})' in line.replace(' ', ''):
            in_json = False
            
    if changed:
        with open(filepath, 'w') as f:
            f.writelines(lines)
        print(f"Fixed {filepath}")
