with open("crates/apr-cli/src/commands/dry_sampling_lint.rs", "r") as f:
    content = f.read()

import re
content = re.sub(r'params\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)', 'params', content)
content = re.sub(r'identity\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)', 'identity', content)
content = re.sub(r'match_len\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)', 'match_len', content)
content = re.sub(r'penalty\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)', 'penalty', content)
content = re.sub(r'monotone\.map\(\|\w+\| format!\("\{.*?\?\}"\)\)', 'monotone', content)

with open("crates/apr-cli/src/commands/dry_sampling_lint.rs", "w") as f:
    f.write(content)
