from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

import exam_prep as ep


SIMULATION_NOTES: list[dict[str, str]] = [
    {"title": "Pseudo-random numbers", "summary": "Deterministic generators can mimic random-looking behavior reproducibly."},
    {"title": "Law of Large Numbers", "summary": "A sample average stabilizes near the population mean as sample size grows."},
    {"title": "Inverse transform", "summary": "Map uniform draws into a target distribution through its inverse CDF."},
]

LCG_PARAMETER_NOTES: list[dict[str, str]] = [
    {"title": "modulus", "summary": "m controls the cycle length and output range."},
    {"title": "multiplier", "summary": "a shapes the recurrence dynamics."},
    {"title": "increment", "summary": "c shifts the sequence and can help avoid short cycles."},
    {"title": "seed", "summary": "X0 sets the initial state for reproducible runs."},
]

conditional_expression = ep.conditional_expression
try_except_else_finally_demo = ep.try_except_else_finally_demo
loop_else_find = ep.loop_else_find
inverse_transform_discrete = ep.inverse_transform_discrete
cauchy_inverse_transform = ep.cauchy_inverse_transform


def if_elif_else_demo(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    return "D"

# Example:
# if_elif_else_demo(85)


def nested_if_demo(x: int, y: int) -> str:
    if x >= 0:
        if y >= 0:
            return "quadrant I boundary"
        return "quadrant IV boundary"
    if y >= 0:
        return "quadrant II boundary"
    return "quadrant III boundary"

# Example:
# nested_if_demo(-3, 5)


def conditional_expression_demo(score: int, *, pass_mark: int = 60) -> str:
    return "pass" if score >= pass_mark else "fail"

# Example:
# conditional_expression_demo(72, pass_mark=60)


def while_countdown(start: int) -> list[int]:
    if start < 0:
        raise ValueError("start must be non-negative.")
    result: list[int] = []
    while start > 0:
        result.append(start)
        start -= 1
    return result

# Example:
# while_countdown(5)


def while_else_countdown(start: int) -> tuple[list[int], str]:
    if start < 0:
        raise ValueError("start must be non-negative.")
    result: list[int] = []
    while start > 0:
        result.append(start)
        start -= 1
    else:
        status = "completed"
    return result, status

# Example:
# while_else_countdown(5)


def for_loop_sum(values: Iterable[float]) -> float:
    total = 0.0
    for value in values:
        total += value
    return total

# Example:
# for_loop_sum([1.5, 2.5, 3.0])


def lcg_generate(seed: int, a: int, c: int, m: int, size: int) -> list[int]:
    return ep.lcg_sequence(seed, a, c, m, size)

# Example:
# lcg_generate(42, 1664525, 1013904223, 2**32, 5)


def lcg_float_stream(a: int, c: int, m: int, seed: int, size: int) -> list[float]:
    rng = ep.LCG(a, c, m, seed)
    return rng.generate(size)

# Example:
# lcg_float_stream(1664525, 1013904223, 2**32, 42, 5)


def law_of_large_numbers_demo(draws: int, *, seed: int = 0, die_sides: int = 6) -> np.ndarray:
    if draws <= 0:
        raise ValueError("draws must be positive.")
    if die_sides <= 0:
        raise ValueError("die_sides must be positive.")
    rng = np.random.default_rng(seed)
    samples = rng.integers(1, die_sides + 1, size=draws)
    return np.cumsum(samples) / np.arange(1, draws + 1)

# Example:
# law_of_large_numbers_demo(10, seed=7, die_sides=6)


def inverse_transform_continuous(F_inv: Callable[[np.ndarray], np.ndarray], n: int, *, seed: int | None = None) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive.")
    if seed is not None:
        rng = np.random.default_rng(seed)
        u = rng.uniform(0, 1, n)
        return np.asarray(F_inv(u), dtype=float)
    return ep.inverse_transform(F_inv, n)

# Example:
# inverse_transform_continuous(lambda u: -np.log(1 - u), 5, seed=7)


def simulation_summary(samples: np.ndarray) -> dict[str, float]:
    arr = np.asarray(samples, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "min": float(arr.min()), "max": float(arr.max())}

# Example:
# simulation_summary(np.array([1, 2, 3, 4, 5]))


__all__ = [
    "LCG_PARAMETER_NOTES",
    "SIMULATION_NOTES",
    "conditional_expression",
    "conditional_expression_demo",
    "try_except_else_finally_demo",
    "loop_else_find",
    "inverse_transform_discrete",
    "cauchy_inverse_transform",
    "for_loop_sum",
    "if_elif_else_demo",
    "inverse_transform_continuous",
    "law_of_large_numbers_demo",
    "lcg_float_stream",
    "lcg_generate",
    "nested_if_demo",
    "simulation_summary",
    "while_else_countdown",
    "while_countdown",
]
