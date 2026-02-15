
impl Default for PatternStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute cosine similarity between two vectors using trueno SIMD.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let va = Vector::from_slice(a);
    let vb = Vector::from_slice(b);

    // Use trueno's SIMD-accelerated operations
    let dot = va.dot(&vb).unwrap_or(0.0);
    let norm_a = va.norm_l2().unwrap_or(0.0);
    let norm_b = vb.norm_l2().unwrap_or(0.0);

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

// ==================== Binary Serialization Helpers ====================

/// Write a pattern and its embedding to a writer.
fn write_pattern<W: IoWrite>(
    writer: &mut W,
    pattern: &ErrorFixPattern,
    embedding: &[f32],
) -> CITLResult<()> {
    // Error code
    write_string(writer, &pattern.error_code.code)?;
    writer.write_all(&[pattern.error_code.category as u8])?;
    writer.write_all(&[pattern.error_code.difficulty as u8])?;

    // Context hash
    writer.write_all(&pattern.context_hash.to_le_bytes())?;

    // Success/failure counts
    writer.write_all(&pattern.success_count.to_le_bytes())?;
    writer.write_all(&pattern.failure_count.to_le_bytes())?;

    // Fix template
    write_fix_template(writer, &pattern.fix_template)?;

    // Embedding
    let dim = embedding.len() as u32;
    writer.write_all(&dim.to_le_bytes())?;
    for val in embedding {
        writer.write_all(&val.to_le_bytes())?;
    }

    Ok(())
}

/// Convert a byte to an `ErrorCategory`.
fn parse_error_category(byte: u8) -> ErrorCategory {
    match byte {
        1 => ErrorCategory::TraitBound,
        2 => ErrorCategory::Unresolved,
        3 => ErrorCategory::Ownership,
        4 => ErrorCategory::Borrowing,
        5 => ErrorCategory::Lifetime,
        6 => ErrorCategory::Async,
        7 => ErrorCategory::TypeInference,
        8 => ErrorCategory::MethodNotFound,
        9 => ErrorCategory::Import,
        _ => ErrorCategory::TypeMismatch, // 0 and unknown default to TypeMismatch
    }
}

/// Convert a byte to a Difficulty.
fn parse_difficulty(byte: u8) -> Difficulty {
    match byte {
        0 => Difficulty::Easy,
        2 => Difficulty::Hard,
        3 => Difficulty::Expert,
        _ => Difficulty::Medium, // 1 and unknown default to Medium
    }
}

/// Convert a byte to a `PlaceholderConstraint`.
fn parse_placeholder_constraint(byte: u8) -> PlaceholderConstraint {
    match byte {
        0 => PlaceholderConstraint::Expression,
        1 => PlaceholderConstraint::Type,
        2 => PlaceholderConstraint::Identifier,
        3 => PlaceholderConstraint::Literal,
        _ => PlaceholderConstraint::Any,
    }
}

/// Read an error code from a reader.
fn read_error_code<R: IoRead>(reader: &mut R) -> CITLResult<ErrorCode> {
    let code_str = read_string(reader)?;
    let mut category_byte = [0u8; 1];
    reader.read_exact(&mut category_byte)?;
    let category = parse_error_category(category_byte[0]);
    let mut difficulty_byte = [0u8; 1];
    reader.read_exact(&mut difficulty_byte)?;
    let difficulty = parse_difficulty(difficulty_byte[0]);
    Ok(ErrorCode::new(&code_str, category, difficulty))
}

/// Read counts (success/failure) from a reader.
fn read_counts<R: IoRead>(reader: &mut R) -> CITLResult<(u64, u64)> {
    let mut success_bytes = [0u8; 8];
    reader.read_exact(&mut success_bytes)?;
    let success_count = u64::from_le_bytes(success_bytes);

    let mut failure_bytes = [0u8; 8];
    reader.read_exact(&mut failure_bytes)?;
    let failure_count = u64::from_le_bytes(failure_bytes);

    Ok((success_count, failure_count))
}

/// Read an embedding vector from a reader.
fn read_embedding<R: IoRead>(reader: &mut R) -> CITLResult<Vec<f32>> {
    let mut dim_bytes = [0u8; 4];
    reader.read_exact(&mut dim_bytes)?;
    let dim = u32::from_le_bytes(dim_bytes) as usize;

    let mut embedding = Vec::with_capacity(dim);
    for _ in 0..dim {
        let mut val_bytes = [0u8; 4];
        reader.read_exact(&mut val_bytes)?;
        embedding.push(f32::from_le_bytes(val_bytes));
    }
    Ok(embedding)
}

/// Read a single placeholder from a reader.
fn read_placeholder<R: IoRead>(reader: &mut R) -> CITLResult<Placeholder> {
    let name = read_string(reader)?;
    let desc = read_string(reader)?;
    let mut constraint_byte = [0u8; 1];
    reader.read_exact(&mut constraint_byte)?;
    let constraint = parse_placeholder_constraint(constraint_byte[0]);
    Ok(Placeholder::new(&name, &desc, constraint))
}

/// Read a vector of placeholders from a reader.
fn read_placeholders<R: IoRead>(reader: &mut R) -> CITLResult<Vec<Placeholder>> {
    let mut ph_count_bytes = [0u8; 2];
    reader.read_exact(&mut ph_count_bytes)?;
    let ph_count = u16::from_le_bytes(ph_count_bytes) as usize;

    let mut placeholders = Vec::with_capacity(ph_count);
    for _ in 0..ph_count {
        placeholders.push(read_placeholder(reader)?);
    }
    Ok(placeholders)
}

/// Read a vector of strings from a reader.
fn read_string_vec<R: IoRead>(reader: &mut R) -> CITLResult<Vec<String>> {
    let mut count_bytes = [0u8; 2];
    reader.read_exact(&mut count_bytes)?;
    let count = u16::from_le_bytes(count_bytes) as usize;

    let mut strings = Vec::with_capacity(count);
    for _ in 0..count {
        strings.push(read_string(reader)?);
    }
    Ok(strings)
}

/// Read a pattern and its embedding from a reader.
fn read_pattern<R: IoRead>(reader: &mut R) -> CITLResult<(ErrorFixPattern, Vec<f32>)> {
    let error_code = read_error_code(reader)?;

    // Context hash
    let mut hash_bytes = [0u8; 8];
    reader.read_exact(&mut hash_bytes)?;
    let context_hash = u64::from_le_bytes(hash_bytes);

    let (success_count, failure_count) = read_counts(reader)?;
    let fix_template = read_fix_template(reader)?;
    let embedding = read_embedding(reader)?;

    let pattern = ErrorFixPattern {
        error_code,
        context_hash,
        fix_template,
        success_count,
        failure_count,
    };

    Ok((pattern, embedding))
}

/// Write a fix template to a writer.
fn write_fix_template<W: IoWrite>(writer: &mut W, template: &FixTemplate) -> CITLResult<()> {
    write_string(writer, &template.pattern)?;
    write_string(writer, &template.description)?;
    writer.write_all(&template.confidence.to_le_bytes())?;

    // Placeholders
    let placeholder_count = template.placeholders.len() as u16;
    writer.write_all(&placeholder_count.to_le_bytes())?;
    for ph in &template.placeholders {
        write_string(writer, &ph.name)?;
        write_string(writer, &ph.description)?;
        writer.write_all(&[ph.constraint as u8])?;
    }

    // Applicable codes
    let codes_count = template.applicable_codes.len() as u16;
    writer.write_all(&codes_count.to_le_bytes())?;
    for code in &template.applicable_codes {
        write_string(writer, code)?;
    }

    Ok(())
}

/// Read a fix template from a reader.
fn read_fix_template<R: IoRead>(reader: &mut R) -> CITLResult<FixTemplate> {
    let pattern = read_string(reader)?;
    let description = read_string(reader)?;

    let mut confidence_bytes = [0u8; 4];
    reader.read_exact(&mut confidence_bytes)?;
    let confidence = f32::from_le_bytes(confidence_bytes);

    let placeholders = read_placeholders(reader)?;
    let applicable_codes = read_string_vec(reader)?;

    Ok(FixTemplate {
        pattern,
        placeholders,
        applicable_codes,
        confidence,
        description,
    })
}

/// Write a length-prefixed string.
fn write_string<W: IoWrite>(writer: &mut W, s: &str) -> CITLResult<()> {
    let bytes = s.as_bytes();
    let len = bytes.len() as u16;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(bytes)?;
    Ok(())
}

/// Read a length-prefixed string.
fn read_string<R: IoRead>(reader: &mut R) -> CITLResult<String> {
    let mut len_bytes = [0u8; 2];
    reader.read_exact(&mut len_bytes)?;
    let len = u16::from_le_bytes(len_bytes) as usize;

    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;

    String::from_utf8(buf).map_err(|_| CITLError::PatternLibraryError {
        message: "Invalid UTF-8 string in pattern file".to_string(),
    })
}
