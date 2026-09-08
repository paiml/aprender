import re
import os

enums_to_fix = [
    "DryParamOutcome", "IdentityOutcome", "MatchLenOutcome", "PenaltyOutcome", "MonotonicityOutcome",
    "TypicalPRangeOutcome", "MassCoverageOutcome", "SortOrderOutcome", "RenormOutcome"
]

directories = ['crates/apr-cli/src/commands']
for directory in directories:
    for filename in os.listdir(directory):
        if not filename.endswith('.rs'):
            continue
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            content = f.read()
        
        changed = False
        
        for enum_name in enums_to_fix:
            pattern = r'( *(?:pub(?: \([^)]+\))? )?enum ' + enum_name + r'\b.*?\{)'
            
            matches = list(re.finditer(pattern, content, re.MULTILINE))
            
            # reverse order so replacement indices don't shift
            for match in reversed(matches):
                enum_decl_start = match.start(1)
                
                body_start = content.find('{', enum_decl_start)
                if body_start == -1:
                    continue
                    
                brace_count = 1
                body_end = body_start
                for i, char in enumerate(content[body_start+1:]):
                    if char == '{': brace_count += 1
                    elif char == '}': brace_count -= 1
                    if brace_count == 0:
                        body_end = body_start + 1 + i
                        break
                        
                enum_body = content[body_start:body_end]
                has_float = 'f32' in enum_body or 'f64' in enum_body
                
                derive_str = '#[derive(Debug, Clone, PartialEq, serde::Serialize)]\n#[serde(tag = "status", rename_all = "snake_case")]\n'
                if not has_float:
                    derive_str = '#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]\n#[serde(tag = "status", rename_all = "snake_case")]\n'
                
                lines_before = content[:enum_decl_start].split('\n')
                attrs = []
                idx = len(lines_before) - 1
                if lines_before[idx] == '':
                    idx -= 1
                while idx >= 0:
                    line = lines_before[idx].strip()
                    if line.startswith('#['):
                        attrs.append(idx)
                    elif line.startswith('///'):
                        pass
                    elif line == '':
                        pass
                    else:
                        break
                    idx -= 1
                    
                already_has = False
                for i in attrs:
                    if 'Serialize' in lines_before[i]:
                        already_has = True
                        break
                
                if already_has:
                    print(f"{enum_name} already has Serialize in {filename}")
                    continue
                    
                for i in attrs:
                    if 'derive' in lines_before[i]:
                        lines_before[i] = '' 
                
                new_before = '\n'.join(lines_before)
                if not new_before.endswith('\n'):
                    new_before += '\n'
                new_before += derive_str
                
                content = new_before + content[enum_decl_start:]
                changed = True
                print(f"Fixed {enum_name} in {filename}")
                
        if changed:
            with open(filepath, 'w') as f:
                f.write(content)
