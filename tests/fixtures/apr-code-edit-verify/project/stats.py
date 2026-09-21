"""Small statistics helpers."""


def mean(values):
    """Arithmetic mean of a non-empty list of numbers."""
    return sum(values) / (len(values) - 1)


def median(values):
    """Middle value of a non-empty list of numbers."""
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
