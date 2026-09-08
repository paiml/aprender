import re
import os

enums_to_fix = [
    "ToolCallsShapeOutcome", "SchemaValidationOutcome", "NoToolsPassthroughOutcome",
    "IterationBoundOutcome", "ScratchpadGrammarOutcome",
    "OllamaSchemaOutcome", "OllamaAllowlistOutcome", "OllamaNdjsonOutcome",
    "GreedyPicksArgmaxOutcome",
    "TimeoutDumpOutcome", "EmptyOnSuccessOutcome", "ExitCodeOutcome", "ExitOutcome", # It might be ExitOutcome or ExitCodeOutcome
    "RequiredMetricsOutcome", "ContentTypeOutcome",
    "RowSoftmaxOutcome", "CausalMaskOutcome", "HtmlHeatmapsOutcome",
    "DocLinkOutcome",
    "DryParamsOutcome", "DryIdentityOutcome", "DryMatchLenOutcome", "DryPenaltyOutcome", "DryMonotoneOutcome",
    "ErrorJsonOutcome", "LayerCoverageOutcome",
    "AttnParityNumericsOutcome", "AttnProvenanceOutcome", "AttnHeadDimErrorOutcome",
    "RowCountOutcome", "DeterminismOutcome",
    "SpanPresentOutcome", "GenaiAttributesOutcome", "TracePropagationOutcome",
    "OomSchemaOutcome", "OomInvariantsOutcome", "OomSizeOutcome",
    "GbnfJsonOutcome", "GbnfDiagnosticOutcome", "GbnfMaskingOutcome",
    "TypicalRangeOutcome", "TypicalIdentityOutcome", "TypicalMassOutcome", "TypicalSortOutcome", "TypicalRenormOutcome"
]

directory = 'crates/apr-cli/src/commands'
for filename in os.listdir(directory):
    if not filename.endswith('.rs'):
        continue
    filepath = os.path.join(directory, filename)
    with open(filepath, 'r') as f:
        content = f.read()
    
    changed = False
    
    for enum_name in enums_to_fix:
        # Find the enum declaration
        match = re.search(r'^(?:pub(?:\([^)]+\))?\s+)?enum\s+' + enum_name + r'\b.*?\{', content, re.MULTILINE | re.DOTALL)
        if not match:
            continue
            
        print(f"Found {enum_name} in {filename}")
        
        # Check if already has Serialize
        # We look at the attributes preceding the enum
        # Find the start of the enum attributes
        enum_start_idx = match.start()
        
        # Look backwards for derives
        preceding = content[:enum_start_idx]
        last_attrs_match = list(re.finditer(r'#\[.*?\]\s*', preceding))
        attr_str = ""
        attrs_start = enum_start_idx
        
        # Collect continuous attributes right before enum
        idx = enum_start_idx
        while True:
            # check if there's an attribute ending right before idx (ignoring whitespace/comments)
            # Actually, regex parsing backwards is hard, let's just use regex on the whole block
            pass
            
        # Instead, let's find the enum block and its preceding lines
        # Pattern: optional docs, optional derives, pub enum Name
        block_pattern = r'((?:^ *#\[.*?\]\n)*)(^ *pub(?: \([^)]+\))? enum ' + enum_name + r'\b)'
        block_pattern2 = r'((?:^ *#\[.*?\]\n)*)(^ *enum ' + enum_name + r'\b)'
        
        match_block = re.search(block_pattern, content, re.MULTILINE)
        if not match_block:
            match_block = re.search(block_pattern2, content, re.MULTILINE)
            
        if not match_block:
            print(f"Failed to match block for {enum_name}")
            continue
            
        attrs = match_block.group(1)
        if 'Serialize' in attrs:
            print(f"{enum_name} already has Serialize, skipping")
            continue
            
        # Check for f32/f64 inside the enum to decide on Eq
        # Extract the body
        body_start = match.end()
        # Find matching closing brace
        brace_count = 1
        body_end = body_start
        for i, char in enumerate(content[body_start:]):
            if char == '{': brace_count += 1
            elif char == '}': brace_count -= 1
            if brace_count == 0:
                body_end = body_start + i
                break
        
        enum_body = content[body_start:body_end]
        has_float = 'f32' in enum_body or 'f64' in enum_body
        
        derive_str = '#[derive(Debug, Clone, PartialEq, serde::Serialize)]\n#[serde(tag = "status", rename_all = "snake_case")]\n'
        if not has_float:
            derive_str = '#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]\n#[serde(tag = "status", rename_all = "snake_case")]\n'
            
        new_block = derive_str + match_block.group(2)
        
        # Replace the old attributes + enum decl with the new one
        start_idx = match_block.start(1)
        end_idx = match_block.end(2)
        
        content = content[:start_idx] + new_block + content[end_idx:]
        changed = True
        
    if changed:
        with open(filepath, 'w') as f:
            f.write(content)
