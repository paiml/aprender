//! Canonical benchmark payloads for tokenizer benchmarks.
//!
//! Each payload represents a real-world tokenizer input pattern:
//! - `SHORT_NL`: Brief user prompt (inference hot path)
//! - `MEDIUM_CODE`: Function signature (inference)
//! - `LONG_CODE_MERGE_SORT`: GH-378 canonical payload (636 chars)
//! - `VERY_LONG_HUMANEVAL`: Full HumanEval problem + solution (~2000 chars)
//! - `CHATML_CONVERSATION`: 3-turn ChatML with special tokens (~400 chars)

/// Short natural-language prompt (~30 chars). Represents a typical user query.
pub const SHORT_NL: &str = "What is the capital of France?";

/// Medium code payload (~200 chars). A typed function signature with docstring.
pub const MEDIUM_CODE: &str = r#"def binary_search(arr: list[int], target: int) -> int:
    """Find target in sorted array using binary search. Returns index or -1."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
"#;

/// Long code payload (636 chars). GH-378 canonical merge-sort benchmark.
pub const LONG_CODE_MERGE_SORT: &str = r#"
def merge_sort(arr: list[int]) -> list[int]:
    """Recursively sort a list using merge sort algorithm."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)

def _merge(left: list[int], right: list[int]) -> list[int]:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
"#;

/// Very long code payload (~2000 chars). Simulates a HumanEval problem + solution.
pub const VERY_LONG_HUMANEVAL: &str = r#"
from typing import List, Tuple, Optional

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers closer to each
    other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False


def separate_paren_groups(paren_string: str) -> List[str]:
    """Input to this function is a string containing multiple groups of nested
    parentheses. Your goal is to separate those groups into separate strings
    and return the list of those. Separate groups are balanced (each open brace
    is properly closed) and not nested within each other.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    result = []
    current_string = []
    current_depth = 0

    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)
            if current_depth == 0:
                result.append(''.join(current_string))
                current_string.clear()

    return result


def truncate_number(number: float) -> float:
    """Given a positive floating point number, it can be decomposed into
    an integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).
    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """
    return number % 1.0


def below_zero(operations: List[int]) -> bool:
    """You're given a list of deposit and withdrawal operations on a bank
    account that starts with zero balance. Your task is to detect if at any
    point the balance of account falls below zero, and at that point function
    should return True. Otherwise it should return False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """
    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False
"#;

/// 3-turn ChatML conversation with special tokens (~400 chars).
/// Exercises `split_on_special_tokens` (PMAT-114).
pub const CHATML_CONVERSATION: &str = "\
<|im_start|>system\n\
You are a helpful coding assistant.<|im_end|>\n\
<|im_start|>user\n\
Write a Python function that computes the nth Fibonacci number using dynamic programming.<|im_end|>\n\
<|im_start|>assistant\n\
```python\n\
def fibonacci(n: int) -> int:\n\
    if n <= 1:\n\
        return n\n\
    dp = [0] * (n + 1)\n\
    dp[1] = 1\n\
    for i in range(2, n + 1):\n\
        dp[i] = dp[i-1] + dp[i-2]\n\
    return dp[n]\n\
```<|im_end|>\n";

/// Generate synthetic text of a given character length for scaling benchmarks.
///
/// Uses a mix of ASCII letters and common code tokens to exercise BPE merges.
pub fn synthetic_text(len: usize) -> String {
    let base = "def merge_sort(arr: list[int]) -> result.append(left[i]) ";
    base.chars().cycle().take(len).collect()
}

/// Generate a batch of prompts with varying lengths for training pre-tokenization benchmarks.
///
/// Returns `count` prompts with lengths uniformly distributed between 50 and 500 chars.
pub fn generate_batch(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| {
            let len = 50 + (i * 450 / count.max(1));
            synthetic_text(len.min(500))
        })
        .collect()
}
