from __future__ import annotations

import copy
import keyword
import re
from itertools import combinations
from typing import Any, Iterable, Sequence

import numpy as np

import exam_prep as ep


COURSE_CONTENTS: list[dict[str, str]] = [
    {"section": "1", "title": "Basic"},
    {"section": "2", "title": "Flow Control"},
    {"section": "3", "title": "Function, Class"},
    {"section": "4", "title": "Package Intro (numpy, scipy, pandas, seaborn, matplotlib)"},
    {"section": "5", "title": "Python feat. VBA (pywin32)"},
    {"section": "6", "title": "Python feat. SQL (pyodbc)"},
    {"section": "7", "title": "API (fastapi)"},
    {"section": "8", "title": "Linear Program (pulp)"},
    {"section": "9", "title": "Machine Learning (scikit learn)"},
    {"section": "10", "title": "Linear Model (statsmodels)"},
    {"section": "11", "title": "Deep Learning (keras, tensorflow)"},
]

DATA_ANALYSIS_WORKFLOW: list[dict[str, str]] = [
    {"stage": "collect", "summary": "Gather data from files, APIs, or databases."},
    {"stage": "clean", "summary": "Standardize, validate, and repair missing or malformed values."},
    {"stage": "explore", "summary": "Summarize distributions, relationships, and anomalies."},
    {"stage": "model", "summary": "Fit numerical, statistical, or machine-learning procedures."},
    {"stage": "communicate", "summary": "Turn results into tables, plots, and decisions."},
]

PYTHON_CORE_FEATURES: list[dict[str, str]] = [
    {"title": "Readable syntax", "summary": "Indentation and simple syntax emphasize logic over boilerplate."},
    {"title": "Dynamic typing", "summary": "Values carry types at runtime and can change across assignments."},
    {"title": "Batteries included", "summary": "The standard library covers text, files, math, and iteration well."},
    {"title": "Multi-paradigm", "summary": "Python supports procedural, object-oriented, and functional styles."},
]

OOP_PRINCIPLES: list[dict[str, str]] = [
    {"title": "Encapsulation", "summary": "Bundle data and behavior behind a clean object interface."},
    {"title": "Abstraction", "summary": "Expose the essential operations while hiding implementation detail."},
    {"title": "Inheritance", "summary": "Reuse and extend behavior through class hierarchies."},
    {"title": "Polymorphism", "summary": "Let different objects answer the same method call in their own way."},
]

UV_NOTES: dict[str, Any] = {
    "title": "uv: The Next-Gen Python Cargo",
    "summary": "A fast Python project and package manager that unifies environment and dependency workflows.",
    "why": [
        "Speed for environment creation and dependency resolution",
        "Consistency via lockfiles and reproducible environments",
        "Single CLI for project, tool, and interpreter management",
    ],
    "keywords": ["uv.lock", "virtual environment", "dependency resolution", "reproducibility"],
}

ALGORITHM_COMPLEXITY_NOTES: list[dict[str, str]] = [
    {"title": "Time complexity", "summary": "Tracks how running time grows as input size increases."},
    {"title": "Space complexity", "summary": "Tracks how much extra memory an algorithm needs."},
    {"title": "Brute force", "summary": "Simple exhaustive search is often easy to write but scales poorly."},
]


def to_snake_case(text: str) -> str:
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|\d+", text.replace("-", " "))
    return "_".join(word.lower() for word in words if word)

# Example:
# to_snake_case("DataScience101")


def to_pascal_case(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text)
    return "".join(word[:1].upper() + word[1:].lower() for word in words)

# Example:
# to_pascal_case("data science 101")


def to_camel_case(text: str) -> str:
    pascal = to_pascal_case(text)
    return pascal[:1].lower() + pascal[1:] if pascal else ""

# Example:
# to_camel_case("data science 101")


def is_python_keyword(name: str) -> bool:
    return keyword.iskeyword(name)

# Example:
# is_python_keyword("for")


def indentation_example(levels: int = 3, indent: int = 4) -> str:
    if levels <= 0 or indent <= 0:
        raise ValueError("levels and indent must be positive.")
    lines = ["if outer_condition:"]
    for level in range(1, levels + 1):
        prefix = " " * (indent * level)
        lines.append(f"{prefix}step_{level} = {level}")
    return "\n".join(lines)

# Example:
# indentation_example(3, 4)


def comment_examples() -> dict[str, str]:
    return {
        "single_line": "# explain one line of logic",
        "multi_line": '"""use a docstring-style block when a longer note helps"""',
    }

# Example:
# comment_examples()


def assignment_demo(value: Any) -> dict[str, Any]:
    assigned = value
    alias = assigned
    return {"value": assigned, "alias_is_same_object": alias is assigned, "type": type(assigned).__name__}

# Example:
# assignment_demo([1, 2, 3])


def basic_type_examples() -> dict[str, Any]:
    return {
        "bool": True,
        "int": 7,
        "float": 3.14,
        "str": "python",
        "list": [1, 2, 3],
        "tuple": (1, 2, 3),
        "dict": {"course": "python"},
        "set": {1, 2, 3},
    }

# Example:
# basic_type_examples()


def boolean_operations(a: bool, b: bool) -> dict[str, bool]:
    return {"and": a and b, "or": a or b, "not_a": not a, "xor": (a and not b) or (not a and b)}

# Example:
# boolean_operations(True, False)


def arithmetic_operations(a: float, b: float) -> dict[str, float]:
    if b == 0:
        raise ZeroDivisionError("b must be non-zero.")
    return {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b, "power": a**b}

# Example:
# arithmetic_operations(8, 2)


def shallow_vs_deep_copy(nested: list[list[int]]) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    original = copy.deepcopy(nested)
    shallow = copy.copy(original)
    deep = copy.deepcopy(original)
    original[0].append(99)
    return original, shallow, deep

# Example:
# shallow_vs_deep_copy([[1, 2], [3, 4]])


def tuple_mutability_demo() -> dict[str, Any]:
    sample = (1, [2, 3])
    sample[1].append(4)
    return {"tuple": sample, "tuple_is_immutable": True, "nested_mutable_changed": sample[1] == [2, 3, 4]}

# Example:
# tuple_mutability_demo()


def list_tuple_tradeoff() -> dict[str, str]:
    return {
        "list": "Mutable, convenient for append/update workloads.",
        "tuple": "Immutable, stable for fixed records and hashable contexts.",
    }

# Example:
# list_tuple_tradeoff()


def dictionary_demo(pairs: Sequence[tuple[str, int]]) -> dict[str, int]:
    return dict(pairs)

# Example:
# dictionary_demo([("Amy", 90), ("Bob", 85)])


def make_frozenset(values: Iterable[Any]) -> frozenset[Any]:
    return frozenset(values)

# Example:
# make_frozenset([1, 2, 2, 3])


def set_demo(a: set[Any], b: set[Any]) -> dict[str, set[Any]]:
    return ep.set_operations(a, b)

# Example:
# set_demo({1, 2, 3}, {2, 3, 4})


def generator_squares(limit: int) -> Iterable[int]:
    if limit < 0:
        raise ValueError("limit must be non-negative.")
    return (k * k for k in range(limit))

# Example:
# list(generator_squares(5))


def consume_generator(gen: Iterable[int]) -> list[int]:
    return list(gen)

# Example:
# consume_generator(generator_squares(5))


def brute_force_shortest_route(cities: Sequence[str], distances: dict[tuple[str, str], float]) -> tuple[list[str], float]:
    if len(cities) < 2:
        raise ValueError("At least two cities are required.")
    best_path: list[str] | None = None
    best_cost: float | None = None
    for perm in combinations(cities[1:], len(cities) - 1):
        path = [cities[0], *perm]
        cost = 0.0
        for left, right in zip(path, path[1:]):
            key = (left, right) if (left, right) in distances else (right, left)
            if key not in distances:
                raise KeyError(f"Missing distance for {left!r}, {right!r}.")
            cost += distances[key]
        if best_cost is None or cost < best_cost:
            best_path = path
            best_cost = cost
    return best_path or list(cities), float(best_cost or 0.0)

# Example:
# brute_force_shortest_route(
#     ["A", "B", "C"],
#     {("A", "B"): 4, ("A", "C"): 2, ("B", "C"): 1},
# )


def growth_rate_examples(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) < 2:
        raise ValueError("values must be a 1-D sequence with at least two items.")
    return {
        "input": arr.tolist(),
        "relative_change": ep.increasing_rate(arr).tolist(),
        "linear": [1, 2, 3, 4, 5],
        "quadratic": [n * n for n in range(1, 6)],
        "exponential": [2**n for n in range(5)],
    }

# Example:
# growth_rate_examples([100, 110, 121])


__all__ = [
    "ALGORITHM_COMPLEXITY_NOTES",
    "COURSE_CONTENTS",
    "DATA_ANALYSIS_WORKFLOW",
    "OOP_PRINCIPLES",
    "PYTHON_CORE_FEATURES",
    "UV_NOTES",
    "arithmetic_operations",
    "assignment_demo",
    "basic_type_examples",
    "boolean_operations",
    "brute_force_shortest_route",
    "comment_examples",
    "consume_generator",
    "dictionary_demo",
    "generator_squares",
    "growth_rate_examples",
    "indentation_example",
    "is_python_keyword",
    "list_tuple_tradeoff",
    "make_frozenset",
    "set_demo",
    "shallow_vs_deep_copy",
    "to_camel_case",
    "to_pascal_case",
    "to_snake_case",
    "tuple_mutability_demo",
]
