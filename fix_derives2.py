import re
import os

enums_to_fix = [
    "ToolCallsShapeOutcome", "SchemaValidationOutcome", "NoToolsPassthroughOutcome",
    "ReactBoundOutcome", "ReactGrammarOutcome",
    "ToolCallSchemaOutcome", "ToolNameAllowlistOutcome", "StreamingToolCallOutcome",
    "ExplainGreedyOutcome",
    "HangTimeoutOutcome", "HangEmptyOnSuccessOutcome", "HangExitOutcome",
    "PromRequiredOutcome", "PromContentTypeOutcome",
    "AttnRowsOutcome", "AttnCausalMaskOutcome", "AttnHtmlOutcome",
    "NcclDocLinkOutcome", "NcclExitOutcome",
    "DryParamOutcome", "IdentityOutcome", "MatchLenOutcome", "PenaltyOutcome", "MonotonicityOutcome",
    "CheckFiniteErrorOutcome", "CheckFiniteCoverageOutcome",
    "AttnParityNumericsOutcome", "AttnProvenanceOutcome", "AttnHeadDimErrorOutcome",
    "EmbedRowCountOutcome", "EmbedDeterminismOutcome",
    "OtlpSpanPresentOutcome", "OtlpAttributesOutcome", "OtlpTracePropagationOutcome",
    "OomSchemaOutcome", "OomInvariantsOutcome", "OomSizeOutcome",
    "JsonGrammarOutputOutcome", "GrammarErrorDiagnosticOutcome", "IllegalTokenMaskingOutcome",
    "TypicalPRangeOutcome", "MassCoverageOutcome", "SortOrderOutcome", "RenormOutcome"
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
        # Find the enum declaration block
        # Group 1: Preceding attributes/comments
        # Group 2: The enum declaration line
        pattern = r'((?:^ *#\[.*?\]\n)*(?:^ *///.*?\n)*)(^ *(?:pub(?: \([^)]+\))? )?enum ' + enum_name + r'\b)'
        
        # We need to find all matches because there might be multiple (though unlikely for same name in same file)
        matches = list(re.finditer(pattern, content, re.MULTILINE))
        
        for match_block in matches:
            attrs = match_block.group(1)
            
            if 'Serialize' in attrs:
                print(f"{enum_name} already has Serialize in {filename}, skipping")
                continue
                
            # Find the body to check for f32/f64
            enum_start_idx = match_block.end(2)
            # Find the first {
            body_start = content.find('{', enum_start_idx)
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
                
            # Wait, we need to replace the EXISTING #[derive(...)] with the new one.
            # So if attrs contains #[derive(...)], we remove it.
            cleaned_attrs = re.sub(r'^ *#\[derive\(.*?\)\]\n', '', attrs, flags=re.MULTILINE)
            # Also remove #[serde(...)] just in case it's there (shouldn't be, but who knows)
            
            new_block = cleaned_attrs + derive_str + match_block.group(2)
            
            start_idx = match_block.start(1)
            end_idx = match_block.end(2)
            
            content = content[:start_idx] + new_block + content[end_idx:]
            changed = True
            print(f"Fixed {enum_name} in {filename}")
            
            # Since we modified the string, we should break and re-parse or just rely on the fact that enums are unique per file.
            # (IdentityOutcome is unique in its file).
            
    if changed:
        with open(filepath, 'w') as f:
            f.write(content)
