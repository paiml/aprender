import xml.etree.ElementTree as ET
import json
import re

JUNIT_PATH = "/tmp/claude-1000/-home-noah-src-aprender/870bdc8d-ce59-4fcf-ab45-5e938deb4c39/scratchpad/tier/junit.xml"

# Load top 80 modules
with open('/tmp/claude-1000/-home-noah-src-aprender/870bdc8d-ce59-4fcf-ab45-5e938deb4c39/scratchpad/tier/lane/top80.json') as f:
    modules = json.load(f)

# Build a lookup structure
# crate -> list of module prefixes (for non-root) and a flag for root
crate_modules = {}
for m in modules:
    crate = m['crate']
    mod = m['module']
    if crate not in crate_modules:
        crate_modules[crate] = {'root': False, 'prefixes': []}
    
    if mod == "":
        crate_modules[crate]['root'] = True
    else:
        crate_modules[crate]['prefixes'].append(mod + "::")
        # Also exact match if it has no further `::` (nextest filterset: test(/^{mod}(::|$)/))
        crate_modules[crate]['prefixes'].append(mod)

tree = ET.parse(JUNIT_PATH)
root = tree.getroot()

total_tests = 0
total_seconds = 0.0
selected_tests = 0
selected_seconds = 0.0

for ts in root.findall('testsuite'):
    for tc in ts.findall('testcase'):
        classname = tc.get('classname')
        if '/' in classname:
            crate, binary = classname.split('/', 1)
        else:
            crate, binary = classname, classname
            
        name = tc.get('name')
        seconds = float(tc.get('time', 0.0))
        
        total_tests += 1
        total_seconds += seconds
        
        selected = False
        
        if "falsif" in name:
            selected = True
        else:
            if crate in crate_modules:
                c_mods = crate_modules[crate]
                if c_mods['root']:
                    # test(/^(tests::|cov_tests::)?[^:]+$/)
                    parts = name.split('::')
                    if len(parts) == 1:
                        selected = True
                    elif len(parts) == 2 and parts[0] in ('tests', 'cov_tests'):
                        selected = True
                
                if not selected:
                    for prefix in c_mods['prefixes']:
                        if prefix.endswith('::'):
                            if name.startswith(prefix):
                                selected = True
                                break
                        else:
                            if name == prefix:
                                selected = True
                                break
                                
        if selected:
            selected_tests += 1
            selected_seconds += seconds

print(f"Total: {total_tests} tests, {total_seconds:.3f} seconds")
print(f"Selected: {selected_tests} tests, {selected_seconds:.3f} seconds")
print(f"Reduction: {selected_tests/total_tests*100:.2f}% tests, {selected_seconds/total_seconds*100:.2f}% seconds")
