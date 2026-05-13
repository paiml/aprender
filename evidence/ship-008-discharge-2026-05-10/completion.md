Certainly! The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, usually starting with 0 and 1. Here's a Python function to compute the nth Fibonacci number using both an iterative and a recursive approach:

### Iterative Approach
```python
def fibonacci_iterative(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
```

### Recursive Approach
```python
def fibonacci_recursive(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)
```

### Example Usage
```python
# Using the iterative approach
print(fibonacci_iterative(10))  # Output: 55

# Using the recursive approach
print(fibonacci_recursive(10))  # Output: 55
```

### Explanation
- **Iterative Approach**:
