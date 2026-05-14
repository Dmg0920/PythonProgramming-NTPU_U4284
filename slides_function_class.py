from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from pydantic import BaseModel, Field, PositiveFloat, field_validator, validate_call

import exam_prep as ep


TYPE_CHECKING_NOTES: list[dict[str, str]] = [
    {"title": "Dynamic typing", "summary": "A variable can reference values of different types over time."},
    {"title": "Type hints", "summary": "Hints improve readability and static analysis without changing runtime semantics."},
    {"title": "Static checking", "summary": "Tools such as mypy or pyright catch type mismatches before execution."},
]

PYDANTIC_FIELD_NOTES: list[dict[str, str]] = [
    {"title": "Default values", "summary": "Field can declare defaults directly on model attributes."},
    {"title": "Validation constraints", "summary": "Field can enforce ranges, lengths, and positivity."},
    {"title": "Metadata", "summary": "Field can carry titles, descriptions, and examples for documentation."},
]

function_definition_notes: dict[str, str] = {
    "define": "Use def to bind a reusable block of statements to a function name.",
    "call": "Call the function with arguments and optionally receive a return value.",
    "docstring": "A docstring documents intent and interface near the implementation.",
}

running_product = ep.running_product
sentence = ep.sentence
make_counter = ep.make_counter
fibonacci_until = ep.fibonacci_until
make_recurrence = ep.make_recurrence
pascal_I = ep.pascal_I
pascal_II = ep.pascal_II
pascal_dp = ep.pascal_dp
Singleton = ep.Singleton
ShapeFactory = ep.ShapeFactory
Circle = ep.Circle
Car = ep.Car
Vehicle = ep.Vehicle
ElectricCar = ep.ElectricCar
Stats = ep.Stats


def add_with_docstring(x: float, y: float) -> float:
    """Return the sum of two numeric values."""
    return x + y

# Example:
# add_with_docstring(3.5, 4.5)


def argument_styles(a: int, b: int = 10, *extras: int, scale: int = 1) -> dict[str, Any]:
    total = a + b + sum(extras)
    return {"positional": a, "default": b, "extras": extras, "keyword_only_scale": scale, "scaled_total": total * scale}

# Example:
# argument_styles(3, 4, 5, 6, scale=2)


def lambda_examples() -> dict[str, Callable[..., Any]]:
    return {
        "square": lambda x: x * x,
        "is_even": lambda x: x % 2 == 0,
        "sort_key": lambda text: (len(text), text.lower()),
    }

# Example:
# lambda_examples()["square"](5)


def dynamic_typing_demo() -> list[str]:
    value: Any = 3
    states = [type(value).__name__]
    value = 3.14
    states.append(type(value).__name__)
    value = "python"
    states.append(type(value).__name__)
    return states

# Example:
# dynamic_typing_demo()


@validate_call
def typed_inner_product(left: list[float], right: list[float]) -> float:
    return ep.iprod(left, right)

# Example:
# typed_inner_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])


class ProductSpec(BaseModel):
    name: str = Field(..., min_length=1, description="Human-readable product name.")
    price: PositiveFloat = Field(..., description="Positive unit price.")
    category: str = Field(default="general", description="Product category label.")

# Example:
# ProductSpec(name="Pen", price=12.5, category="stationery")


class StudentProfile(BaseModel):
    name: str = Field(..., min_length=1, description="Student full name.")
    age: int = Field(..., ge=0, le=120, description="Student age.")
    gpa: float = Field(..., ge=0.0, le=4.3, description="Grade point average.")

    @field_validator("name")
    @classmethod
    def title_case_name(cls, value: str) -> str:
        return value.title()

# Example:
# StudentProfile(name="amy chen", age=20, gpa=3.8)


def hanoi_moves(n: int, source: str = "A", auxiliary: str = "B", target: str = "C") -> list[tuple[str, str]]:
    if n < 0:
        raise ValueError("n must be non-negative.")
    if n == 0:
        return []
    moves = hanoi_moves(n - 1, source, target, auxiliary)
    moves.append((source, target))
    moves.extend(hanoi_moves(n - 1, auxiliary, source, target))
    return moves

# Example:
# hanoi_moves(3)


def hanoi_min_moves(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative.")
    return 2**n - 1

# Example:
# hanoi_min_moves(3)


def factorial_recursive(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative.")
    if n in (0, 1):
        return 1
    return n * factorial_recursive(n - 1)

# Example:
# factorial_recursive(5)


def triangular_recursive(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative.")
    if n == 0:
        return 0
    return n + triangular_recursive(n - 1)

# Example:
# triangular_recursive(5)


def fibonacci_family(n: int) -> dict[str, int]:
    return {
        "recursive": ep.fib_recursive(n),
        "memo": ep.fib_memo(n),
        "iterative": ep.fib_iterative(n),
        "closed_form": ep.fib_binet(n),
    }

# Example:
# fibonacci_family(10)


@dataclass
class ClassCounter:
    name: str
    counter: int = 0

    total_created = 0

    def __post_init__(self) -> None:
        type(self).total_created += 1

    @classmethod
    def created(cls) -> int:
        return cls.total_created

    @staticmethod
    def describe_counter() -> str:
        return "Class variables are shared across instances."

    def bump(self) -> int:
        self.counter += 1
        return self.counter

# Example:
# counter = ClassCounter("Amy"); counter.bump()


class SierpinskiTriangle:
    base = np.array([[1, 1], [1, 0]], dtype=int)

    @staticmethod
    def matrix_method1(k: int) -> np.ndarray:
        if k < 1:
            raise ValueError("k must be >= 1.")
        size = 2**k
        rows = np.arange(size, dtype=int)[:, None]
        cols = np.arange(size, dtype=int)[None, :]
        return (rows & cols == 0).astype(int)

    @classmethod
    def matrix_method2(cls, k: int) -> np.ndarray:
        if k < 1:
            raise ValueError("k must be >= 1.")
        if k == 1:
            return cls.base.copy()
        prev = cls.matrix_method2(k - 1)
        return np.block([[prev, prev], [prev, np.zeros_like(prev)]])

    @classmethod
    def plot_methods(
        cls,
        *,
        method1_ks: Sequence[int],
        method2_ks: Sequence[int],
    ) -> tuple[plt.Figure, np.ndarray]:
        if not method1_ks or not method2_ks:
            raise ValueError("method1_ks and method2_ks must be non-empty.")

        ncols = max(len(method1_ks), len(method2_ks))
        fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8), squeeze=False)

        for col, k in enumerate(method1_ks):
            ax = axes[0, col]
            ax.imshow(cls.matrix_method1(k), cmap="binary", interpolation="nearest")
            ax.set_title(f"Method 1, K={k}")
            ax.set_axis_off()

        for col in range(len(method1_ks), ncols):
            axes[0, col].set_axis_off()

        for col, k in enumerate(method2_ks):
            ax = axes[1, col]
            ax.imshow(cls.matrix_method2(k), cmap="binary", interpolation="nearest")
            ax.set_title(f"Method 2, K={k}")
            ax.set_axis_off()

        for col in range(len(method2_ks), ncols):
            axes[1, col].set_axis_off()

        fig.tight_layout()
        return fig, axes

# Example:
# SierpinskiTriangle.matrix_method1(3)


__all__ = [
    "Car",
    "Circle",
    "ClassCounter",
    "ElectricCar",
    "PYDANTIC_FIELD_NOTES",
    "ProductSpec",
    "SierpinskiTriangle",
    "ShapeFactory",
    "Singleton",
    "Stats",
    "StudentProfile",
    "TYPE_CHECKING_NOTES",
    "Vehicle",
    "add_with_docstring",
    "argument_styles",
    "dynamic_typing_demo",
    "factorial_recursive",
    "fibonacci_family",
    "fibonacci_until",
    "function_definition_notes",
    "hanoi_min_moves",
    "hanoi_moves",
    "lambda_examples",
    "make_counter",
    "make_recurrence",
    "pascal_I",
    "pascal_II",
    "pascal_dp",
    "running_product",
    "sentence",
    "triangular_recursive",
    "typed_inner_product",
]
