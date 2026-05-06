from __future__ import annotations

import functools
import math
import operator
import random
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from functools import reduce
from itertools import accumulate, combinations, product
from math import exp, factorial, isclose, log, pow, prod, sqrt
from typing import Annotated, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
import scipy.stats as scs
import seaborn as sns
from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    ValidationError,
    field_validator,
    validate_call,
)
from scipy import linalg
from scipy.integrate import quad
from scipy.linalg import LinAlgError, inv, lstsq, solve
from scipy.stats import gmean, hmean


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


def Gauss_Siedel(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    tol: float = 1e-3,
    max_iter: int = 1000,
) -> np.ndarray:
    """Solve Ax=b by the matrix-form Gauss-Seidel iteration."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.asarray(x, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if b.shape != (A.shape[0], 1) or x.shape != (A.shape[0], 1):
        raise ValueError("b and x must be column vectors with shape (n, 1).")
    if tol <= 0:
        raise ValueError("tol must be positive.")

    L = np.tril(A)
    U = A - L
    if np.any(np.diag(L) == 0):
        raise LinAlgError("Lower triangular matrix L is singular.")

    x_c = x.copy()
    for counter in range(1, max_iter + 1):
        x_c = np.linalg.solve(L, b - U @ x)
        if np.linalg.norm(x_c - x) < tol:
            print(f"Converged after {counter} iterations")
            break
        print(f"iteration {counter}: {x_c.ravel()}")
        x = x_c
    else:
        raise RuntimeError("Gauss-Seidel did not converge within max_iter.")

    return x_c


def DD_check(M: np.ndarray) -> None:
    """Print whether M is strictly diagonally dominant."""
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix.")

    diag = np.abs(np.diag(M))
    off_diag = np.sum(np.abs(M), axis=1) - diag
    if np.all(diag > off_diag):
        print("Matrix is diagonally dominant")
    else:
        print("Not diagonally dominant")


def midpoint_rule(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    """Approximate an integral by the midpoint rule."""
    _validate_positive_n(n)
    grid = np.linspace(a, b, n + 1)
    h = (b - a) / n
    midpoints = (grid[:-1] + grid[1:]) / 2
    return float(h * np.sum(f(midpoints)))


def trapezoid_rule(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    """Approximate an integral by the composite trapezoid rule."""
    _validate_positive_n(n)
    grid = np.linspace(a, b, n + 1)
    h = (b - a) / n
    values = np.asarray(f(grid), dtype=float)
    weights = np.ones(n + 1)
    weights[1:-1] = 2
    return float((h / 2) * np.dot(weights, values))


def trapezoid_min_n(K: float, a: float, b: float, epsilon: float) -> int:
    """Return the minimum n from the trapezoid error bound."""
    if K < 0 or epsilon <= 0:
        raise ValueError("K must be non-negative and epsilon must be positive.")
    return math.ceil(math.sqrt(K * (b - a) ** 3 / (12 * epsilon)))


def simpsons_rule(
    f_or_values: Callable[[np.ndarray], np.ndarray] | ArrayLike,
    h_or_a: float,
    b: float | None = None,
    n: int | None = None,
) -> float:
    """Approximate an integral by Simpson's rule."""
    if callable(f_or_values):
        if b is None or n is None:
            raise ValueError("b and n are required when the first argument is callable.")
        _validate_positive_n(n)
        if n % 2 != 0:
            raise ValueError("n must be even for Simpson's rule.")
        grid = np.linspace(h_or_a, b, n + 1)
        h = (b - h_or_a) / n
        values = np.asarray(f_or_values(grid), dtype=float)
    else:
        values = np.asarray(f_or_values, dtype=float)
        h = h_or_a
        n = len(values) - 1
        if n <= 0 or n % 2 != 0:
            raise ValueError("f_values must have odd length so n is positive and even.")

    coeffs = np.ones(n + 1)
    coeffs[1:-1:2] = 4
    coeffs[2:-1:2] = 2
    return float((h / 3) * np.dot(coeffs, values))


def ARE(est: float, tval: float) -> float:
    """Compute absolute relative error as a percentage."""
    if est == 0:
        raise ZeroDivisionError("est must be non-zero.")
    return abs(tval - est) / abs(est) * 100


def MCMC(f: Callable[[np.ndarray], np.ndarray], area: list[float], runs: int) -> float:
    """Estimate an integral by Monte Carlo sampling on [a, b]."""
    if len(area) != 2:
        raise ValueError("area must be [a, b].")
    _validate_positive_n(runs)
    a, b = area
    samples = np.random.uniform(a, b, runs)
    return float((b - a) * np.mean(f(samples)))


class Integration:
    """Notebook-style integration helper."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> None:
        _validate_positive_n(n)
        self.fn = fn
        self.grid = np.linspace(a, b, n + 1)
        self.height = (b - a) / n

    def Midpt(self) -> float:
        midpoints = (self.grid[:-1] + self.grid[1:]) / 2
        return float((self.height * np.sum(self.fn(midpoints))).item())

    def Trapezoid(self) -> float:
        values = np.asarray(self.fn(self.grid), dtype=float)
        weights = np.ones(len(self.grid))
        weights[1:-1] = 2
        return float(((self.height / 2) * np.dot(weights, values)).item())

    @staticmethod
    def ARE(est: float, tval: float) -> float:
        return ARE(est, tval)

    @staticmethod
    def MCMC(f: Callable[[np.ndarray], np.ndarray], area: list[float], runs: int) -> float:
        return MCMC(f, area, runs)


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> float:
    """Find a bracketed root by bisection."""
    fa = f(a)
    fb = f(b)
    if fa * fb >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")

    c = (a + b) / 2
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        if abs(fc) < tol or abs(b - a) < tol:
            return c
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return c


def newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Find a root by Newton's method."""
    x = x0
    for _ in range(max_iter):
        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError("Derivative is zero at the current iterate.")
        x_new = x - f(x) / dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise RuntimeError("Newton's method did not converge.")


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Find a root by the secant method."""
    x_prev = x0
    x_curr = x1
    for _ in range(max_iter):
        f_prev = f(x_prev)
        f_curr = f(x_curr)
        denom = f_curr - f_prev
        if denom == 0:
            raise ZeroDivisionError("Secant denominator is zero.")
        x_new = x_curr - f_curr * (x_curr - x_prev) / denom
        if abs(x_new - x_curr) < tol:
            return x_new
        x_prev, x_curr = x_curr, x_new
    raise RuntimeError("Secant method did not converge.")


def fixed_point(
    g: Callable[[float], float],
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Find a fixed point x=g(x)."""
    x = x0
    for _ in range(max_iter):
        x_new = g(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise RuntimeError("Fixed-point iteration did not converge.")


class GradientDescent:
    """Gradient descent examples for the MidExam objective."""

    def __init__(self, pt: list[float], maxN: int) -> None:
        if len(pt) != 2:
            raise ValueError("pt must contain exactly two values.")
        if maxN < 0:
            raise ValueError("maxN must be non-negative.")
        self.pt = pt
        self.maxN = maxN

    @staticmethod
    def grad_f(dx: float, dy: float) -> np.ndarray:
        return np.array([(9 / 8) * (dx - 2) + dy / 4, 2 * (dy - 2) + dx / 4])

    def GradD(self, alpha: float) -> np.ndarray:
        vec = np.array(self.pt, dtype=float)
        result = np.zeros((self.maxN + 1, 2), dtype=float)
        result[0] = vec
        for k in range(1, self.maxN + 1):
            vec = vec - alpha * self.grad_f(vec[0], vec[1])
            result[k] = vec
        return result

    def Momentum(self, alpha: float) -> np.ndarray:
        vec = np.array(self.pt, dtype=float)
        wgt = np.zeros(2, dtype=float)
        result = np.zeros((self.maxN + 1, 2), dtype=float)
        result[0] = vec
        for k in range(1, self.maxN + 1):
            wgt = 0.9 * wgt + self.grad_f(vec[0], vec[1])
            vec = vec - alpha * wgt
            result[k] = vec
        return result


def _shift_from_key(shift: int | str) -> int:
    if isinstance(shift, str):
        if len(shift) != 1 or not shift.isalpha():
            raise ValueError("String shift must be a single alphabetic character.")
        return ord(shift.upper()) - ord("A")
    return shift % 26


def encode_each(alpbet: str, key: str) -> str:
    """Encode one alphabetic character by one Vigenere key character."""
    if not alpbet.isalpha() or len(alpbet) != 1:
        raise ValueError("alpbet must be a single alphabetic character.")
    if not key.isalpha() or len(key) != 1:
        raise ValueError("key must be a single alphabetic character.")
    base = ord("A") if alpbet.isupper() else ord("a")
    key_idx = ord(key.upper()) - ord("A")
    return chr(base + ((ord(alpbet) - base + key_idx) % 26))


def Caeser_Cipher(plain_text: str, shift: int | str) -> str:
    """Encrypt text with a Caesar shift while preserving case."""
    k = _shift_from_key(shift)
    result: list[str] = []
    for char in plain_text:
        if char.isalpha():
            base = ord("A") if char.isupper() else ord("a")
            result.append(chr(base + ((ord(char) - base + k) % 26)))
        else:
            result.append(char)
    return "".join(result)


def Vigenere_Cipher(plain_text: str, keyword: str) -> str:
    """Encrypt text with the Vigenere cipher."""
    if not keyword or not keyword.isalpha():
        raise ValueError("keyword must contain at least one alphabetic character.")
    result: list[str] = []
    count = 0
    for char in plain_text:
        if char.isalpha():
            key = keyword[count % len(keyword)]
            result.append(encode_each(char, key))
            count += 1
        else:
            result.append(char)
    return "".join(result)


def Crack_Text(plain_txt: str) -> None:
    """Print all Caesar decryptions for shifts 0 through 25."""
    for k in range(26):
        print(k, Caeser_Cipher(plain_txt, -k))


class CipherTools:
    """Notebook-style namespace for cipher static methods."""

    encode_each = staticmethod(encode_each)
    Caeser_Cipher = staticmethod(Caeser_Cipher)
    Vigenere_Cipher = staticmethod(Vigenere_Cipher)
    Crack_Text = staticmethod(Crack_Text)


def _mod_inverse(a: int, mod: int) -> int:
    a %= mod
    t, new_t = 0, 1
    r, new_r = mod, a
    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
    if r != 1:
        raise ValueError(f"{a} has no inverse modulo {mod}.")
    return t % mod


class AffineCipher:
    """Affine substitution cipher over A-Z."""

    def __init__(self, a: int, b: int) -> None:
        if math.gcd(a, 26) != 1:
            raise ValueError("a must be coprime to 26.")
        self.a = a % 26
        self.b = b % 26
        self.a_inv = _mod_inverse(self.a, 26)

    def encrypt(self, plain_text: str) -> str:
        return self._transform(plain_text, decrypt=False)

    def decrypt(self, cipher_text: str) -> str:
        return self._transform(cipher_text, decrypt=True)

    def _transform(self, text: str, decrypt: bool) -> str:
        result: list[str] = []
        for char in text:
            if not char.isalpha():
                result.append(char)
                continue
            base = ord("A") if char.isupper() else ord("a")
            idx = ord(char) - base
            new_idx = self.a_inv * (idx - self.b) % 26 if decrypt else (self.a * idx + self.b) % 26
            result.append(chr(base + new_idx))
        return "".join(result)


class SubstitutionCipher:
    """Monoalphabetic substitution cipher."""

    def __init__(self, key: str) -> None:
        cleaned = key.upper()
        if len(cleaned) != 26 or set(cleaned) != set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            raise ValueError("key must be a 26-letter permutation of A-Z.")
        plain = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.forward = dict(zip(plain, cleaned))
        self.inverse = {value: key for key, value in self.forward.items()}

    def encrypt(self, plain_text: str) -> str:
        return self._apply(plain_text, self.forward)

    def decrypt(self, cipher_text: str) -> str:
        return self._apply(cipher_text, self.inverse)

    @staticmethod
    def _apply(text: str, mapping: dict[str, str]) -> str:
        result: list[str] = []
        for char in text:
            if char.isalpha():
                mapped = mapping[char.upper()]
                result.append(mapped if char.isupper() else mapped.lower())
            else:
                result.append(char)
        return "".join(result)


def _mod_matrix_inverse(matrix: np.ndarray, mod: int = 26) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=int) % mod
    n = matrix.shape[0]
    aug = np.concatenate([matrix, np.eye(n, dtype=int)], axis=1) % mod

    for col in range(n):
        pivot = None
        for row in range(col, n):
            if math.gcd(int(aug[row, col]), mod) == 1:
                pivot = row
                break
        if pivot is None:
            raise ValueError("Matrix is not invertible modulo 26.")
        if pivot != col:
            aug[[col, pivot]] = aug[[pivot, col]]

        inv_pivot = _mod_inverse(int(aug[col, col]), mod)
        aug[col] = (aug[col] * inv_pivot) % mod
        for row in range(n):
            if row == col:
                continue
            factor = aug[row, col]
            aug[row] = (aug[row] - factor * aug[col]) % mod

    return aug[:, n:] % mod


class HillCipher:
    """Hill cipher with integer matrix arithmetic modulo 26."""

    def __init__(self, key_matrix: np.ndarray) -> None:
        key = np.asarray(key_matrix, dtype=int)
        if key.ndim != 2 or key.shape[0] != key.shape[1]:
            raise ValueError("key_matrix must be square.")
        det_mod = int(round(np.linalg.det(key))) % 26
        if math.gcd(det_mod, 26) != 1:
            raise ValueError("det(key_matrix) must be coprime to 26 modulo 26.")
        self.key_matrix = key % 26
        self.block_size = key.shape[0]
        self.inverse_key = _mod_matrix_inverse(self.key_matrix, 26)

    def encrypt(self, plain_text: str) -> str:
        text = self._clean_and_pad(plain_text)
        return self._transform(text, self.key_matrix)

    def decrypt(self, cipher_text: str) -> str:
        text = "".join(char for char in cipher_text.upper() if char.isalpha())
        if len(text) % self.block_size != 0:
            raise ValueError("cipher_text length must be a multiple of the block size.")
        return self._transform(text, self.inverse_key)

    def _clean_and_pad(self, text: str) -> str:
        cleaned = "".join(char for char in text.upper() if char.isalpha())
        pad_len = (-len(cleaned)) % self.block_size
        return cleaned + ("X" * pad_len)

    def _transform(self, text: str, matrix: np.ndarray) -> str:
        output: list[str] = []
        for start in range(0, len(text), self.block_size):
            block = text[start : start + self.block_size]
            vec = np.array([ord(char) - ord("A") for char in block], dtype=int)
            encoded = matrix @ vec % 26
            output.extend(chr(ord("A") + int(num)) for num in encoded)
        return "".join(output)


class RankCorrelation:
    """Spearman rho and Kendall tau for paired numeric data."""

    def __init__(self, X: ArrayLike, Y: ArrayLike) -> None:
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        if self.X.ndim != 1 or self.Y.ndim != 1 or len(self.X) != len(self.Y):
            raise ValueError("X and Y must be 1-D arrays of equal length.")
        if len(self.X) < 3:
            raise ValueError("At least three observations are required.")

    def rho(self) -> tuple[float, float]:
        n = len(self.X)
        r_x = self.X.argsort().argsort()
        r_y = self.Y.argsort().argsort()
        d = r_x - r_y
        coef = 1 - (6 * np.sum(d**2)) / (n * (n**2 - 1))
        if abs(coef) == 1:
            pval = 0.0
        else:
            t_stat = coef * math.sqrt((n - 2) / (1 - coef**2))
            pval = 2 * (1 - scs.t.cdf(abs(t_stat), n - 2))
        return float(coef), float(pval)

    def tau(self) -> tuple[float, float]:
        n = len(self.X)
        c_sum = 0.0
        for i, j in combinations(range(n), 2):
            c_sum += np.sign(self.X[i] - self.X[j]) * np.sign(self.Y[i] - self.Y[j])
        coef = 2 * c_sum / (n * (n - 1))
        z_stat = coef * math.sqrt(9 * n * (n - 1) / (2 * (2 * n + 5)))
        pval = 2 * (1 - scs.norm.cdf(abs(z_stat)))
        return float(coef), float(pval)


def _target_density_hw2(x: np.ndarray) -> np.ndarray:
    return (np.pi / 2) * np.sin(np.pi * x)


def _triangular_density(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.5, 4 * x, 4 * (1 - x))


def sample_candidate1(n_samples: int) -> tuple[np.ndarray, float]:
    """Rejection sampling with uniform proposal g(x)=1."""
    _validate_positive_n(n_samples)
    accepted: list[float] = []
    total_trials = 0
    M = np.pi / 2
    while len(accepted) < n_samples:
        x = np.random.uniform(0, 1)
        u = np.random.uniform(0, 1)
        total_trials += 1
        if u <= _target_density_hw2(np.array([x]))[0] / M:
            accepted.append(x)
    return np.array(accepted), n_samples / total_trials


def sample_candidate2(n_samples: int) -> tuple[np.ndarray, float]:
    """Rejection sampling with triangular proposal from averaged uniforms."""
    _validate_positive_n(n_samples)
    accepted: list[float] = []
    total_trials = 0
    M = np.pi**2 / 8
    while len(accepted) < n_samples:
        x = (np.random.uniform(0, 1) + np.random.uniform(0, 1)) / 2
        u = np.random.uniform(0, 1)
        total_trials += 1
        ratio = _target_density_hw2(np.array([x]))[0] / (M * _triangular_density(np.array([x]))[0])
        if u <= ratio:
            accepted.append(x)
    return np.array(accepted), n_samples / total_trials


def inverse_transform(F_inv: Callable[[np.ndarray], np.ndarray], n: int) -> np.ndarray:
    """Generate samples by inverse transform sampling."""
    _validate_positive_n(n)
    u = np.random.uniform(0, 1, n)
    return np.asarray(F_inv(u), dtype=float)


def box_muller(n: int) -> np.ndarray:
    """Generate 2n standard-normal samples by the Box-Muller transform."""
    _validate_positive_n(n)
    u1 = np.random.uniform(np.nextafter(0, 1), 1, n)
    u2 = np.random.uniform(0, 1, n)
    radius = np.sqrt(-2 * np.log(u1))
    z1 = radius * np.cos(2 * np.pi * u2)
    z2 = radius * np.sin(2 * np.pi * u2)
    return np.concatenate([z1, z2])


class LCG:
    """Linear congruential generator."""

    def __init__(self, a: int, c: int, m: int, seed: int) -> None:
        if m <= 0:
            raise ValueError("m must be positive.")
        self.a = a
        self.c = c
        self.m = m
        self.state = seed % m

    def next(self) -> float:
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

    def generate(self, n: int) -> list[float]:
        _validate_positive_n(n)
        return [self.next() for _ in range(n)]


def monte_carlo_pi(n: int) -> float:
    """Estimate pi by sampling points in the unit square."""
    _validate_positive_n(n)
    xy = np.random.uniform(0, 1, (n, 2))
    hits = np.sum(np.sum(xy**2, axis=1) <= 1)
    return float(4 * hits / n)


class BankAccount:
    """Bank account with a name-mangled private balance."""

    def __init__(self, owner: str, balance: float, password: int | str) -> None:
        self.owner = owner
        self.__balance = balance
        self.__password = password

    def get_balance(self) -> float:
        return self.__balance


class Temperature:
    """Temperature with validation and a derived Fahrenheit property."""

    def __init__(self, celsius: float) -> None:
        self.celsius = celsius

    @property
    def celsius(self) -> float:
        return self._celsius

    @celsius.setter
    def celsius(self, value: float) -> None:
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero.")
        self._celsius = value

    @property
    def fahrenheit(self) -> float:
        return self._celsius * 9 / 5 + 32


class car:
    """Notebook-style property example using the original lowercase class name."""

    def __init__(self, name: str) -> None:
        self.brand = name

    @property
    def brand(self) -> str:
        return self._brand

    @brand.setter
    def brand(self, value: str) -> None:
        if not value:
            raise ValueError("brand cannot be empty.")
        self._brand = value


class Pet:
    def make_sound(self) -> str:
        return "Hello"


class Dog(Pet):
    def make_sound(self) -> str:
        return "Woof!"


class Cat(Pet):
    def make_sound(self) -> str:
        return "Meow!"


class Snake(Pet):
    def make_sound(self) -> str:
        return "Hiss!"


class ExamA:
    def who(self) -> str:
        return "A"


class ExamB(ExamA):
    def who(self) -> str:
        return "B"


class ExamC(ExamA):
    def who(self) -> str:
        return "C"


class ExamD(ExamB, ExamC):
    pass


class Animal:
    def __init__(self, name: str) -> None:
        self.name = name


class ExamWorkingDog(Animal):
    def __init__(self, name: str, job: str) -> None:
        super().__init__(name)
        self.job = job


class ExamBell:
    def make_sound(self) -> str:
        return "ding"


class ExamPhone:
    def make_sound(self) -> str:
        return "ring"


def running_product(*args: float) -> list[float]:
    """Return cumulative products as floats."""
    result: list[float] = []
    current = 1.0
    for value in args:
        current *= value
        result.append(float(current))
    return result


def sentence(*, subject: str, object: str, verb: str) -> str:
    """Build a simple subject-verb-object sentence."""
    return f"{subject} {verb} {object}"


def wrong_bucket(value: int, bucket: list[int] = []) -> list[int]:
    """Intentional mutable-default example."""
    bucket.append(value)
    return bucket


def correct_bucket(value: int, bucket: list[int] | None = None) -> list[int]:
    """Correct mutable-default pattern."""
    if bucket is None:
        bucket = []
    bucket.append(value)
    return bucket


def vec_op(fn: Callable[..., float], *args: float) -> float:
    """Apply a callable and round the result to two digits."""
    return round(fn(*args), ndigits=2)


def fib_recursive(n: int) -> int:
    """Naive recursive Fibonacci."""
    _validate_nonnegative_int(n)
    if n < 2:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)


def fib_binet(n: int) -> int:
    """Fibonacci by Binet's formula."""
    _validate_nonnegative_int(n)
    phi = (1 + math.sqrt(5)) / 2
    psi = (1 - math.sqrt(5)) / 2
    return int(round((phi**n - psi**n) / math.sqrt(5)))


@functools.lru_cache(maxsize=None)
def fib_memo(n: int) -> int:
    """Memoized recursive Fibonacci."""
    _validate_nonnegative_int(n)
    if n < 2:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)


def fib_iterative(n: int) -> int:
    """Iterative Fibonacci."""
    _validate_nonnegative_int(n)
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def iprod(x: list[int], y: list[int]) -> float:
    """Return the inner product of two equal-length integer lists."""
    if len(x) != len(y):
        raise ValueError("x and y must have equal length.")
    return float(sum(a * b for a, b in zip(x, y)))


def _positive_values(args: tuple[float, ...]) -> np.ndarray:
    if not args:
        raise ValueError("At least one value is required.")
    values = np.asarray(args, dtype=float)
    if np.any(values <= 0):
        raise ValueError("All values must be positive.")
    return values


def QM(*args: float) -> float:
    """Quadratic mean."""
    values = _positive_values(args)
    return round(float(np.sqrt(np.mean(values**2))), 2)


def AM(*args: float) -> float:
    """Arithmatic mean."""
    values = _positive_values(args)
    return round(float(np.mean(values)), 2)


def GM(*args: float) -> float:
    """Geometric mean."""
    values = _positive_values(args)
    return round(float(gmean(values)), 2)


def HM(*args: float) -> float:
    """Harmonic mean."""
    values = _positive_values(args)
    return round(float(hmean(values)), 2)


class Mean_Type:
    QM = staticmethod(QM)
    AM = staticmethod(AM)
    GM = staticmethod(GM)
    HM = staticmethod(HM)


def Final_Value(P: float, r: float, t: float, n: int | str) -> float:
    """Return future value under discrete or continuous compounding."""
    if n == "inf":
        return P * math.exp(r * t)
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer or 'inf'.")
    return P * (1 + r / n) ** (n * t)


def logit_grow(M: float, r: float, t: float) -> float:
    """Logistic growth curve M / (1 + exp(-rt))."""
    return M / (1 + math.exp(-r * t))


def triangular_parts(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return lower, upper, strict upper, and strict lower triangular parts."""
    A = np.asarray(A)
    lower = np.tril(A)
    upper = np.triu(A)
    return lower, upper, A - lower, A - upper


def array_ranks(arr: ArrayLike) -> np.ndarray:
    """Return 0-indexed ranks by double argsort."""
    return np.asarray(arr).argsort().argsort()


def subtract_row_means(data: np.ndarray) -> np.ndarray:
    """Broadcast row means across columns and subtract them."""
    data = np.asarray(data, dtype=float)
    return data - data.mean(axis=1, keepdims=True)


def outer_product(v: ArrayLike) -> np.ndarray:
    """Compute an outer product by broadcasting."""
    v = np.asarray(v, dtype=float)
    return v[:, np.newaxis] * v[np.newaxis, :]


def pairwise_differences(points: np.ndarray) -> np.ndarray:
    """Return pairwise vector differences by broadcasting."""
    points = np.asarray(points, dtype=float)
    return points[:, np.newaxis, :] - points[np.newaxis, :, :]


def view_copy_demo(v2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a view slice and an independent copy slice."""
    return v2[:2, :2], v2[:2, :2].copy()


def meshgrid_values(
    x: ArrayLike,
    y: ArrayLike,
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    indexing: str = "xy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate f over a 2-D meshgrid."""
    X, Y = np.meshgrid(np.asarray(x), np.asarray(y), indexing=indexing)
    Z = f(X, Y)
    return X, Y, Z


def _validate_positive_n(n: int) -> None:
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")


def _validate_nonnegative_int(n: int) -> None:
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer.")


__all__ = [
    "Gauss_Siedel",
    "DD_check",
    "midpoint_rule",
    "trapezoid_rule",
    "trapezoid_min_n",
    "simpsons_rule",
    "ARE",
    "MCMC",
    "Integration",
    "bisection",
    "newton",
    "secant",
    "fixed_point",
    "GradientDescent",
    "Caeser_Cipher",
    "Vigenere_Cipher",
    "encode_each",
    "Crack_Text",
    "CipherTools",
    "AffineCipher",
    "SubstitutionCipher",
    "HillCipher",
    "RankCorrelation",
    "sample_candidate1",
    "sample_candidate2",
    "inverse_transform",
    "box_muller",
    "LCG",
    "monte_carlo_pi",
    "BankAccount",
    "Temperature",
    "car",
    "Pet",
    "Dog",
    "Cat",
    "Snake",
    "ExamA",
    "ExamB",
    "ExamC",
    "ExamD",
    "Animal",
    "ExamWorkingDog",
    "ExamBell",
    "ExamPhone",
    "running_product",
    "sentence",
    "wrong_bucket",
    "correct_bucket",
    "vec_op",
    "fib_recursive",
    "fib_binet",
    "fib_memo",
    "fib_iterative",
    "iprod",
    "QM",
    "AM",
    "GM",
    "HM",
    "Mean_Type",
    "Final_Value",
    "logit_grow",
    "triangular_parts",
    "array_ranks",
    "subtract_row_means",
    "outer_product",
    "pairwise_differences",
    "view_copy_demo",
    "meshgrid_values",
]
