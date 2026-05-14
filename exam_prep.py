# =============================================================================
# Exam Prep Code 目錄
# =============================================================================
# 定位方式：
# - 直接搜尋函式或類別名稱，例如 `Gauss_Siedel`、`HillCipher`、`group_summary`。
# - 區塊順序大致是：spec.md topics -> Sec 1 -> Sec 2 -> Sec 3 -> Sec 4.1 -> Sec 4.2 -> Sec 4.3。
#
# 0. Imports / aliases
# - NumPy/SciPy/pandas/Matplotlib/Seaborn/Pydantic aliases 依照 spec.md Section 0。
#
# 1. Numerical methods / integration / error
# - `Gauss_Siedel`, `DD_check`
# - `midpoint_rule`, `trapezoid_rule`, `trapezoid_min_n`, `simpsons_rule`
# - `ARE`, `MCMC`, `Integration`
# - `bisection`, `newton`, `secant`, `fixed_point`, `steffensen`
# - Slide additions: `solve_linear_system`
#
# 2. Optimization
# - `GradientDescent`
# - `GradientDescent.grad_f`, `GradientDescent.GradD`, `GradientDescent.Momentum`
#
# 3. Ciphers / cryptography
# - `Caeser_Cipher`, `Vigenere_Cipher`, `encode_each`, `Crack_Text`, `CipherTools`
# - `AffineCipher`, `SubstitutionCipher`, `HillCipher`
#
# 4. Statistics / simulation / random generation
# - `RankCorrelation.rho`, `RankCorrelation.tau`
# - `sample_candidate1`, `sample_candidate2`
# - `inverse_transform`, `inverse_transform_discrete`, `cauchy_inverse_transform`
# - `box_muller`, `LCG`, `lcg_sequence`, `monte_carlo_pi`
# - `dice_running_mean`, `random_walk_2d`
#
# 5. OOP patterns from Sec 1 / Sec 3
# - Encapsulation/property: `BankAccount`, `Temperature`, `car`
# - Inheritance/polymorphism/MRO: `Pet`, `Dog`, `Cat`, `Snake`, `ExamA`-`ExamD`
# - Multilevel/super examples: `Animal`, `ExamWorkingDog`, `Vehicle`, `Electric`, `ElectricCar`
# - Dunder/classmethod/factory: `Car`, `Stats`, `iCounter`, `Singleton`, `ShapeFactory`, `Circle`
#
# 6. Function patterns / recursion / type hints from Sec 3
# - Arguments/defaults: `running_product`, `sentence`, `wrong_bucket`, `correct_bucket`
# - Type/function helpers: `iprod`, `parser`, `cosine_sim`, `debug_call`, `merge_dicts_new`
# - Pydantic: `Product`, `User`, `Material`, `Guest`, `NightClub_check`, `Address`, `Employee`
# - Recursion: `fib_recursive`, `fib_binet`, `fib_memo`, `fib_iterative`
# - Notebook aliases: `fibonacci_GS`, `fibonacci_memo`, `fibonacci_iter`, `fibonacci_until`
# - Recurrence/combinatorics: `make_recurrence`, `tri_closed`, `tri_inverse`, `tri_seq`
# - Pascal: `pascal_I`, `pascal_II`, `pascal_dp`
# - Closure: `make_counter`
#
# 7. Means / finance / growth models
# - `QM`, `AM`, `GM`, `HM`, `Mean_Type`, `power_mean`
# - `Final_Value`, `present_value`
# - `logit_grow`, `gompertz`
#
# 8. Sec 1 basics / strings / collections / regex
# - Flow-control slides: `conditional_expression`, `try_except_else_finally_demo`, `loop_else_find`
# - Boolean logic: `exactly_one_choice`, `cafeteria_action`, `truth_table_exactly_one`
# - Number basics: `quotient_remainder`, `exact_decimal_check`, `float_isclose_sum`
# - Regex: `regex_find`, `regex_search_one`, `regex_split`, `regex_replace`
# - Generators: `seq_div`, `exam_countdown`, `generator_consumption_demo`
# - Strings/copy/collections: `join_words`, `shallow_deep_copy_demo`
# - Discrete distribution and growth: `zero_truncated_poisson_pmf`, `increasing_rate`
# - Dict/set helpers: `dict_filter_by_value`, `set_operations`
#
# 9. NumPy from Sec 4.1
# - Matrix/array patterns: `triangular_parts`, `array_ranks`, `meshgrid_values`
# - Broadcasting: `subtract_row_means`, `outer_product`, `pairwise_differences`
# - View/copy/slicing: `view_copy_demo`, `sliding_window`
# - Vectorization/piecewise: `cal_recip`, `heaviside_1`, `heaviside_3`, `ramp`
# - Slide additions: `array_attributes`, `int_array_float_truncation`, `concatenate_stack`, `split_array`
# - Ufunc/mask/sort: `ufunc_out_square`, `ufunc_reduce_sum`, `nan_safe_aggregates`, `mask_between`, `mask_count`
# - Sorting algorithms: `selection_sort`, `bubble_sort`, `insertion_sort`, `merge_sort`, `quick_sort`
# - Linear algebra: `block_matrix`, `projection_matrix`, `project_vector`
#
# 10. pandas from Sec 4.2
# - DataFrame construction/combination: `make_df`, `concat_frames`, `merge_frames`
# - Slides basics: `series_from_mapping`, `dataframe_profile`, `select_loc_iloc`, `index_aligned_add`
# - Missing data: `missing_summary`, `fill_missing`, `interpolate_missing`, `nullable_int_series`, `add_missing_indicator`
# - Relational algebra: `relational_project`, `relational_select`, `relational_rename`
# - Relational set ops: `relational_union`, `relational_set_difference`, `relational_cross_product`, `join_category`
# - Groupby/statistics: `iqr`, `zscore`, `group_zscore`, `group_summary`
# - Pivot: `pivot_counts`
#
# 11. Visualization from Sec 4.3
# - Basic plots: `plot_line`, `plot_scatter`, `plot_errorbar`
# - Error band/contour/hist: `plot_confidence_band`, `contour_grid`, `plot_histogram`
# - Colormap overview: `plot_color_gradients`
# - Slide additions: `ax_set_labels`, `choose_colormap`, `choose_plot_type`
#
# 12. Internal helpers
# - `_validate_positive_n`, `_validate_nonnegative_int`, private cipher/density helpers
# =============================================================================

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
from typing import Annotated, Any, Callable, Optional

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


# 這一區是數值方法的核心：把線性方程、數值積分、誤差與 root-finding
# 都包成可直接呼叫的函式。考試時通常是「給公式，寫成 code」。
# Gauss-Siedal 迭代法
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

    # 防呆
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")  # A 必須為方陣
    if b.shape != (A.shape[0], 1) or x.shape != (A.shape[0], 1):
        raise ValueError("b and x must be column vectors with shape (n, 1).")
    if tol <= 0:
        raise ValueError("tol must be positive.")

    # 將 A 拆成 L + U：L 保留含對角線的下三角，U 是嚴格上三角。
    # Gauss-Seidel 每次用最新的 x 透過 L x_new = b - U x_old 更新解。
    L = np.tril(A)
    U = A - L
    if np.any(np.diag(L) == 0):
        raise LinAlgError("Lower triangular matrix L is singular.")

    x_c = x.copy()
    for counter in range(1, max_iter + 1):
        x_c = np.linalg.solve(L, b - U @ x)
        if np.linalg.norm(x_c - x) < tol:  # 與原值差異，若 < tol 則 converge
            print(f"Converged after {counter} iterations")  # counter 次後收斂
            break
        print(f"iteration {counter}: {x_c.ravel()}")  # 若未收斂，則繼續迭代
        x = x_c
    else:
        raise RuntimeError("Gauss-Seidel did not converge within max_iter.")  # 如果在 max_iter 次後還未收斂，則報錯

    return x_c

# Example:
# Gauss_Siedel(np.array([[4, 1], [2, 3]]), np.array([[1], [2]]), np.zeros((2, 1)))

# Diagonally dominant checker
# 定義：對角線上的元素絕對值，都嚴格大於該列其他所有元素的絕對值總和
# Ax = b 中，如果 A 為 DD ，則當使用迭代法時，數學上保證一定會收斂到唯一解
def DD_check(M: np.ndarray) -> None:
    """Print whether M is strictly diagonally dominant."""
    M = np.asarray(M, dtype=float)  # asarray會會將傳入的 Python 內建列表轉為 NumPy 陣列，並轉為 float
    # Validation
    if M.ndim != 2 or M.shape[0] != M.shape[1]:  # M 必須是二維且必須為方陣
        raise ValueError("M must be a square matrix.")

    diag = np.abs(np.diag(M))  # 提取對角元素，轉為一維陣列，並轉為絕對值
    off_diag = np.sum(np.abs(M), axis=1) - diag  # 算整列的絕對值總和，然後扣掉對角線的值
    if np.all(diag > off_diag):
        print("Matrix is diagonally dominant")
    else:
        print("Not diagonally dominant")

# Example:
# DD_check(np.array([[4, 1], [2, 3]]))

# 另一種寫法
def is_strictly_diagonally_dominant(M: np.ndarray | list) -> bool:
    """
    Check if a matrix is strictly diagonally dominant.
    
    Args:
        M: A square 2D array-like object.
        
    Returns:
        bool: True if strictly diagonally dominant, False otherwise.
        
    Raises:
        ValueError: If M is not a 2D square matrix or is empty.
    """
    M = np.asarray(M, dtype=float)
    
    if M.size == 0:
        raise ValueError("Matrix cannot be empty.")
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Expected a 2D square matrix, got shape {M.shape}")

    diag = np.abs(np.diag(M))
    off_diag = np.sum(np.abs(M), axis=1) - diag
    
    # 浮點數比較有時會有微小精度誤差，若需要極端精準，可考慮 np.isclose
    return bool(np.all(diag > off_diag))

# Example:
# is_strictly_diagonally_dominant([[4, 1], [2, 3]])

# 中點法求數值積分
def midpoint_rule(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:  # 型別提示
    """Approximate an integral by the midpoint rule."""
    _validate_positive_n(n)  # 私有屬性，確保 n 必須大於零
    grid = np.linspace(a, b, n + 1)  # a 到 b 之間產生 n + 1 個均勻點
    h = (b - a) / n  # step size
    midpoints = (grid[:-1] + grid[1:]) / 2  # grid[:-1] 取出了所有區間的左邊界，grid[1:] 取出了所有區間的右邊界，相加並除以 2，算出所有的中點位置
    return float(h * np.sum(f(midpoints)))  # 計算總面積

# Example:
# midpoint_rule(lambda x: x**2, 0, 1, 4)

# 梯形法求積分
def trapezoid_rule(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:  # 型別提示
    """Approximate an integral by the composite trapezoid rule."""
    _validate_positive_n(n)  # 防呆，n 必須大於零
    grid = np.linspace(a, b, n + 1)
    h = (b - a) / n
    values = np.asarray(f(grid), dtype=float)
    weights = np.ones(n + 1)
    weights[1:-1] = 2
    return float((h / 2) * np.dot(weights, values))

# Example:
# trapezoid_rule(lambda x: x**2, 0, 1, 4)

# 梯形法最小樣本數
def trapezoid_min_n(K: float, a: float, b: float, epsilon: float) -> int:
    """Return the minimum n from the trapezoid error bound."""
    if K < 0 or epsilon <= 0:
        raise ValueError("K must be non-negative and epsilon must be positive.")
    return math.ceil(math.sqrt(K * (b - a) ** 3 / (12 * epsilon)))

# Example:
# trapezoid_min_n(2.0, 0, 1, 1e-3)

# Simpson 法
def simpsons_rule(
    f_or_values: Callable[[np.ndarray], np.ndarray] | ArrayLike,
    h_or_a: float,
    b: float | None = None,
    n: int | None = None,
) -> float:  # Type hint
    """Approximate an integral by Simpson's rule."""
    if callable(f_or_values):  # 如果 f_or_values 是一個函數，就會進入這個 if 區塊
        if b is None or n is None:
            raise ValueError("b and n are required when the first argument is callable.")
        _validate_positive_n(n)
        if n % 2 != 0:  # n 必須是偶數
            raise ValueError("n must be even for Simpson's rule.")
        grid = np.linspace(h_or_a, b, n + 1)
        h = (b - h_or_a) / n
        values = np.asarray(f_or_values(grid), dtype=float)
    else:
        values = np.asarray(f_or_values, dtype=float)  # 如果 f_or_values 不是函數，那它就是一組已經算好的 y 值。這行將它轉換成 NumPy 浮點數陣列。
        h = h_or_a
        n = len(values) - 1
        if n <= 0 or n % 2 != 0:
            raise ValueError("f_values must have odd length so n is positive and even.")  # 防呆

    # Simpson's rule 的重點是係數模式：端點 1，奇數 interior 點 4，偶數 interior 點 2。
    coeffs = np.ones(n + 1)
    coeffs[1:-1:2] = 4  # [start:stop:step]，從 index 1 （第二個元素）開始，每次跳兩步，直到 index -1 （最後一個），對應「奇數索引」，不含 stop，改成 4
    coeffs[2:-1:2] = 2  # [start:stop:step]，從 index 2 （第三個元素）開始，每次跳兩步，直到 index -1 （最後一個），對應「偶數索引」，不含 stop，改成 2
    return float((h / 3) * np.dot(coeffs, values))

# Example:
# simpsons_rule(lambda x: x**2, 0, 1, 4)


def ARE(est: float, tval: float) -> float:
    """Compute absolute relative error as a percentage."""
    if est == 0:  # 檢查 est 是否等於零
        raise ZeroDivisionError("est must be non-zero.")
    return abs(tval - est) / abs(est) * 100  # ARE 計算

# Example:
# ARE(2.0082484, 2.0)

# Monte Carlo Integration
def MCMC(f: Callable[[np.ndarray], np.ndarray], area: list[float], runs: int) -> float:
    """Estimate an integral by Monte Carlo sampling on [a, b]."""
    if len(area) != 2:
        raise ValueError("area must be [a, b].")
    _validate_positive_n(runs)  # 抽樣次數 > 0
    a, b = area
    samples = np.random.uniform(a, b, runs)  # 從最小值 a 到最大值 b 之間生成一個連續均勻分配 U(a, b) 的隨機樣本
    return float((b - a) * np.mean(f(samples)))

# Example:
# MCMC(lambda x: x**2, [0, 1], 5000)

# 把上述積分法寫成一個 class
class Integration:
    """Notebook-style integration helper."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> None:
        _validate_positive_n(n)
        self.fn = fn
        self.grid = np.linspace(a, b, n + 1)
        self.height = (b - a) / n

    def Midpt(self) -> float:
        midpoints = (self.grid[:-1] + self.grid[1:]) / 2  # 算出所有區間的中點
        return float((self.height * np.sum(self.fn(midpoints))).item())

    def Trapezoid(self) -> float:
        values = np.asarray(self.fn(self.grid), dtype=float)
        weights = np.ones(len(self.grid))
        weights[1:-1] = 2
        return float(((self.height / 2) * np.dot(weights, values)).item())

    @staticmethod  # Decorator，靜態方法，只是一個普通函數，被收納在這個類別的命名空間下
    def ARE(est: float, tval: float) -> float:
        return ARE(est, tval)  # 前面定義的 ARE 函數

    @staticmethod
    def MCMC(f: Callable[[np.ndarray], np.ndarray], area: list[float], runs: int) -> float:
        return MCMC(f, area, runs)  # 前面定義的 MCMC 函數

# Example:
# Integration(lambda x: x**2, 0, 1, 4).Midpt()


# 這一區是 root-finding：不同方法的差別在於如何產生下一個 x。
# bisection 保證收斂但慢；Newton 快但需要 derivative；secant 用兩點斜率近似 derivative。
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
    if fa * fb >= 0:  # f(a) 和 f(b) 相乘必須小於零（異號）
        raise ValueError("f(a) and f(b) must have opposite signs.")

    c = (a + b) / 2
    for _ in range(max_iter):
        c = (a + b) / 2  # 每次迭代中，計算當前區間 [a, b] 的中點 c
        fc = f(c)  # 計算中點值的函數值存入 fc
        if abs(fc) < tol or abs(b - a) < tol:  # |f(c)| 非常接近 0，即爲所求之根
            return c
        if fa * fc < 0:  # 如果 f(a) 與 f(c) 異號，代表根落在 [a, c]
            b = c  # 把原本右邊界 b 往左縮，更新為現在中點 c
            fb = fc  # 更新右邊界的函數值為 f(c)
        else:
            a = c
            fa = fc
    return c

# Example:
# bisection(lambda x: x**2 - 2, 1, 2)

# 牛頓法
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
        x_new = x - f(x) / dfx  # 計算迭代後新的 x
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise RuntimeError("Newton's method did not converge.")

# Example:
# newton(lambda x: x**2 - 2, lambda x: 2 * x, 1.5)

# 割線法
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

# Example:
# secant(lambda x: x**2 - 2, 1, 2)

# 固點遞迴法
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

# Example:
# fixed_point(lambda x: np.cos(x), 0.5)

# 梯度下降
class GradientDescent:
    """Gradient descent examples for the MidExam objective."""

    def __init__(self, pt: list[float], maxN: int) -> None:
        # 防呆
        if len(pt) != 2:  # 檢查起始點 pt 是否不等於 2
            raise ValueError("pt must contain exactly two values.")
        if maxN < 0:  # 檢查最大迭代次數是否小於零
            raise ValueError("maxN must be non-negative.")
        self.pt = pt
        self.maxN = maxN

    @staticmethod
    def grad_f(dx: float, dy: float) -> np.ndarray:
        return np.array([(9 / 8) * (dx - 2) + dy / 4, 2 * (dy - 2) + dx / 4])  # f 對 x 以及對 y 的偏微分

    def GradD(self, alpha: float) -> np.ndarray:
        # result 保留完整路徑，不只回傳最後一點，方便畫收斂軌跡或檢查每一步。
        vec = np.array(self.pt, dtype=float)  # X0
        result = np.zeros((self.maxN + 1, 2), dtype=float)
        result[0] = vec
        for k in range(1, self.maxN + 1):
            vec = vec - alpha * self.grad_f(vec[0], vec[1])  # 梯度下降
            result[k] = vec
        return result

    def Momentum(self, alpha: float) -> np.ndarray:
        # wgt 是累積方向；momentum 讓更新同時看現在的 gradient 與過去方向。
        vec = np.array(self.pt, dtype=float)
        wgt = np.zeros(2, dtype=float)
        result = np.zeros((self.maxN + 1, 2), dtype=float)
        result[0] = vec
        for k in range(1, self.maxN + 1):
            wgt = 0.9 * wgt + self.grad_f(vec[0], vec[1])  # 更新動量向量，保留 90% 上一步的移動方向，
            vec = vec - alpha * wgt  # 更新位置，減去 alpha x wgt
            result[k] = vec
        return result

# Example:
# GradientDescent([1.0, -1.0], 10).GradD(0.1)


# 這一區是 cipher 題型：核心都是把字母轉成 0-25，做 mod 26 運算後再轉回字母。
# 非字母字元通常保留，大小寫則依輸入維持。
def _shift_from_key(shift: int | str) -> int:
    if isinstance(shift, str):  # 檢查 shift 是不是 string
        if len(shift) != 1 or not shift.isalpha():  # 如果輸入字串長度不為 1 或不是英文字母便報錯
            raise ValueError("String shift must be a single alphabetic character.")
        return ord(shift.upper()) - ord("A")  # ord() 把子母轉為 ASCII 數值
    return shift % 26  # 取餘數

# Example:
# _shift_from_key("D")

# 用一個 key 來加密
def encode_each(alpbet: str, key: str) -> str:
    """Encode one alphabetic character by one Vigenere key character."""
    if not alpbet.isalpha() or len(alpbet) != 1:
        raise ValueError("alpbet must be a single alphabetic character.")
    if not key.isalpha() or len(key) != 1:
        raise ValueError("key must be a single alphabetic character.")
    base = ord("A") if alpbet.isupper() else ord("a")
    key_idx = ord(key.upper()) - ord("A")
    return chr(base + ((ord(alpbet) - base + key_idx) % 26))  # 把明文字母轉換成 0 ~ 25 的數字，並加上金鑰的位移量，並加回原本的基準值

# Example:
# encode_each("A", "LEMON")

# 用一個固定位移量
def Caeser_Cipher(plain_text: str, shift: int | str) -> str:
    """Encrypt text with a Caesar shift while preserving case."""
    k = _shift_from_key(shift)
    result: list[str] = []  # 字串是 immutable，所以需要先建立一個空的清單，把加密字母丟進去再黏起來
    for char in plain_text:
        if char.isalpha():
            base = ord("A") if char.isupper() else ord("a")
            result.append(chr(base + ((ord(char) - base + k) % 26)))  # 字母轉數字，加上位移量 k，除以 26 取餘數，再轉回字母
        else:
            result.append(char)
    return "".join(result)

# Example:
# Caeser_Cipher("HELLO", 3)

# 用一串字母當作位移量
def Vigenere_Cipher(plain_text: str, keyword: str) -> str:
    """Encrypt text with the Vigenere cipher."""
    if not keyword or not keyword.isalpha():
        raise ValueError("keyword must contain at least one alphabetic character.")
    result: list[str] = []
    count = 0
    for char in plain_text:
        if char.isalpha():
            key = keyword[count % len(keyword)]  # 取得金鑰長度，並把現在處理的字母數量 count 除以長度取餘數
            result.append(encode_each(char, key))
            count += 1
        else:
            result.append(char)
    return "".join(result)

# Example:
# Vigenere_Cipher("ATTACK AT DAWN", "LEMON")


# 嘗試所有可能破解 Caeser Cipher
def Crack_Text(plain_txt: str) -> None:
    """Print all Caesar decryptions for shifts 0 through 25."""
    for k in range(26):
        print(k, Caeser_Cipher(plain_txt, -k))

# Example:
# Crack_Text("KHOOR")


class CipherTools:
    """Notebook-style namespace for cipher static methods."""

    encode_each = staticmethod(encode_each)
    Caeser_Cipher = staticmethod(Caeser_Cipher)
    Vigenere_Cipher = staticmethod(Vigenere_Cipher)
    Crack_Text = staticmethod(Crack_Text)

# Example:
# CipherTools.Caeser_Cipher("HELLO", 3)


def _mod_inverse(a: int, mod: int) -> int:
    a %= mod
    t, new_t = 0, 1
    r, new_r = mod, a
    while new_r != 0:
        quotient = r // new_r  # 只保留商數
        t, new_t = new_t, t - quotient * new_t  # 新的餘數 = 舊的被除數 - 商數 x 舊的除數
        r, new_r = new_r, r - quotient * new_r  # 新的係數 = 舊的係數 x 商數 - 舊的係數
    if r != 1:
        raise ValueError(f"{a} has no inverse modulo {mod}.")
    return t % mod

# Example:
# _mod_inverse(5, 26)


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

# Example:
# AffineCipher(5, 8).encrypt("AFFINE")


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

# Example:
# SubstitutionCipher("QWERTYUIOPASDFGHJKLZXCVBNM").encrypt("HELLO")


def _mod_matrix_inverse(matrix: np.ndarray, mod: int = 26) -> np.ndarray:
    # Hill cipher 解密不能用浮點 np.linalg.inv；這裡用 Gauss-Jordan 在 mod 26 下求整數反矩陣。
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

# Example:
# _mod_matrix_inverse(np.array([[3, 3], [2, 5]]))


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
        # 每個 block 變成 column vector 後做 matrix @ vector，再對 26 取餘數回到 A-Z。
        output: list[str] = []
        for start in range(0, len(text), self.block_size):
            block = text[start : start + self.block_size]
            vec = np.array([ord(char) - ord("A") for char in block], dtype=int)
            encoded = matrix @ vec % 26
            output.extend(chr(ord("A") + int(num)) for num in encoded)
        return "".join(output)

# Example:
# HillCipher(np.array([[3, 3], [2, 5]])).encrypt("HELP")


# 這一區是 rank correlation 與抽樣模擬。Spearman/Kendall 比的是排名或 pair 順序，
# rejection sampling 則是用 proposal distribution 產生候選值，再用接受率篩掉不合 target 的點。
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

# Example:
# RankCorrelation([1, 2, 3, 4], [1, 3, 2, 4]).rho()


def _target_density_hw2(x: np.ndarray) -> np.ndarray:
    return (np.pi / 2) * np.sin(np.pi * x)

# Example:
# _target_density_hw2(np.array([0.25, 0.5, 0.75]))


def _triangular_density(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.5, 4 * x, 4 * (1 - x))

# Example:
# _triangular_density(np.array([0.25, 0.5, 0.75]))


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
        # Uniform proposal 的 g(x)=1，所以接受機率就是 f(x) / M。
        if u <= _target_density_hw2(np.array([x]))[0] / M:
            accepted.append(x)
    return np.array(accepted), n_samples / total_trials

# Example:
# samples, acc_rate = sample_candidate1(5)


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
        # triangular proposal 更貼近 target，理論接受率比 uniform proposal 高。
        ratio = _target_density_hw2(np.array([x]))[0] / (M * _triangular_density(np.array([x]))[0])
        if u <= ratio:
            accepted.append(x)
    return np.array(accepted), n_samples / total_trials

# Example:
# samples, acc_rate = sample_candidate2(5)


def inverse_transform(F_inv: Callable[[np.ndarray], np.ndarray], n: int) -> np.ndarray:
    """Generate samples by inverse transform sampling."""
    _validate_positive_n(n)
    u = np.random.uniform(0, 1, n)
    return np.asarray(F_inv(u), dtype=float)

# Example:
# inverse_transform(lambda u: -np.log(1 - u), 5)


def box_muller(n: int) -> np.ndarray:
    """Generate 2n standard-normal samples by the Box-Muller transform."""
    _validate_positive_n(n)
    u1 = np.random.uniform(np.nextafter(0, 1), 1, n)
    u2 = np.random.uniform(0, 1, n)
    radius = np.sqrt(-2 * np.log(u1))
    z1 = radius * np.cos(2 * np.pi * u2)
    z2 = radius * np.sin(2 * np.pi * u2)
    return np.concatenate([z1, z2])

# Example:
# box_muller(3)


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

# Example:
# LCG(1664525, 1013904223, 2**32, 42).generate(3)


def monte_carlo_pi(n: int) -> float:
    """Estimate pi by sampling points in the unit square."""
    _validate_positive_n(n)
    xy = np.random.uniform(0, 1, (n, 2))
    hits = np.sum(np.sum(xy**2, axis=1) <= 1)
    return float(4 * hits / n)

# Example:
# monte_carlo_pi(10000)


# 這一區是 OOP 基礎：封裝、property、繼承、MRO、多型。
# 這些類別不是為了做大型系統，而是保留考試常問的 Python class 行為。
class BankAccount:
    """Bank account with a name-mangled private balance."""

    def __init__(self, owner: str, balance: float, password: int | str) -> None:
        self.owner = owner
        self.__balance = balance
        self.__password = password

    def get_balance(self) -> float:
        return self.__balance

# Example:
# BankAccount("Amy", 1000, "1234").get_balance()


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

# Example:
# Temperature(25).fahrenheit


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

# Example:
# car("Toyota").brand


class Pet:
    def make_sound(self) -> str:
        return "Hello"

# Example:
# Pet().make_sound()


class Dog(Pet):
    def make_sound(self) -> str:
        return "Woof!"

# Example:
# Dog().make_sound()


class Cat(Pet):
    def make_sound(self) -> str:
        return "Meow!"

# Example:
# Cat().make_sound()


class Snake(Pet):
    def make_sound(self) -> str:
        return "Hiss!"

# Example:
# Snake().make_sound()


class ExamA:
    def who(self) -> str:
        return "A"

# Example:
# ExamA().who()


class ExamB(ExamA):
    def who(self) -> str:
        return "B"

# Example:
# ExamB().who()


class ExamC(ExamA):
    def who(self) -> str:
        return "C"

# Example:
# ExamC().who()


class ExamD(ExamB, ExamC):
    # Diamond inheritance 測 MRO：ExamD -> ExamB -> ExamC -> ExamA -> object。
    pass

# Example:
# [cls.__name__ for cls in ExamD.__mro__]


class Animal:
    def __init__(self, name: str) -> None:
        self.name = name

# Example:
# Animal("Lucky").name


class ExamWorkingDog(Animal):
    def __init__(self, name: str, job: str) -> None:
        super().__init__(name)
        self.job = job

# Example:
# ExamWorkingDog("Lucky", "guide").job


class ExamBell:
    def make_sound(self) -> str:
        return "ding"

# Example:
# ExamBell().make_sound()


class ExamPhone:
    def make_sound(self) -> str:
        return "ring"

# Example:
# ExamPhone().make_sound()


# 這一區是 function pattern：*args、**kwargs、mutable default、遞迴、type hints。
# 重點是知道資料怎麼被收集、傳遞、快取，以及哪些寫法會造成 hidden state。
def running_product(*args: float) -> list[float]:
    """Return cumulative products as floats."""
    result: list[float] = []
    current = 1.0
    for value in args:
        current *= value
        result.append(float(current))
    return result

# Example:
# running_product(2, 3, 4)


def sentence(*, subject: str, object: str, verb: str) -> str:
    """Build a simple subject-verb-object sentence."""
    return f"{subject} {verb} {object}"

# Example:
# sentence(subject="Amy", verb="likes", object="Python")


def wrong_bucket(value: int, bucket: list[int] = []) -> list[int]:
    """Intentional mutable-default example."""
    # 這是故意保留的錯誤範例：bucket 只在 def 當下建立一次，後續呼叫會共用同一個 list。
    bucket.append(value)
    return bucket

# Example:
# wrong_bucket(1); wrong_bucket(2)


def correct_bucket(value: int, bucket: list[int] | None = None) -> list[int]:
    """Correct mutable-default pattern."""
    if bucket is None:
        bucket = []
    bucket.append(value)
    return bucket

# Example:
# correct_bucket(1); correct_bucket(2)


def vec_op(fn: Callable[..., float], *args: float) -> float:
    """Apply a callable and round the result to two digits."""
    return round(fn(*args), ndigits=2)

# Example:
# vec_op(lambda x, y: x + y, 1.25, 2.75)


def fib_recursive(n: int) -> int:
    """Naive recursive Fibonacci."""
    _validate_nonnegative_int(n)
    if n < 2:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)

# Example:
# fib_recursive(10)


def fib_binet(n: int) -> int:
    """Fibonacci by Binet's formula."""
    _validate_nonnegative_int(n)
    phi = (1 + math.sqrt(5)) / 2
    psi = (1 - math.sqrt(5)) / 2
    return int(round((phi**n - psi**n) / math.sqrt(5)))

# Example:
# fib_binet(10)


@functools.lru_cache(maxsize=None)
def fib_memo(n: int) -> int:
    """Memoized recursive Fibonacci."""
    _validate_nonnegative_int(n)
    if n < 2:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)

# Example:
# fib_memo(10)


def fib_iterative(n: int) -> int:
    """Iterative Fibonacci."""
    _validate_nonnegative_int(n)
    a, b = 0, 1
    # simultaneous assignment 讓 a,b 同步往下一項推，不需要額外暫存變數。
    for _ in range(n):
        a, b = b, a + b
    return a

# Example:
# fib_iterative(10)


def iprod(x: list[int], y: list[int]) -> float:
    """Return the inner product of two equal-length integer lists."""
    if len(x) != len(y):
        raise ValueError("x and y must have equal length.")
    return float(sum(a * b for a, b in zip(x, y)))

# Example:
# iprod([1, 2, 3], [4, 5, 6])


def _positive_values(args: tuple[float, ...]) -> np.ndarray:
    if not args:
        raise ValueError("At least one value is required.")
    values = np.asarray(args, dtype=float)
    if np.any(values <= 0):
        raise ValueError("All values must be positive.")
    return values

# Example:
# _positive_values((1.0, 2.0, 4.0))


# 這一區是 Sec 3 的數學函式：各種平均、複利、折現、growth model。
# 這些函式通常直接對應公式，參數名稱盡量保留講義符號。
def QM(*args: float) -> float:
    """Quadratic mean."""
    values = _positive_values(args)
    return round(float(np.sqrt(np.mean(values**2))), 2)

# Example:
# QM(1, 2, 4)


def AM(*args: float) -> float:
    """Arithmatic mean."""
    values = _positive_values(args)
    return round(float(np.mean(values)), 2)

# Example:
# AM(1, 2, 4)


def GM(*args: float) -> float:
    """Geometric mean."""
    values = _positive_values(args)
    return round(float(gmean(values)), 2)

# Example:
# GM(1, 2, 4)


def HM(*args: float) -> float:
    """Harmonic mean."""
    values = _positive_values(args)
    return round(float(hmean(values)), 2)

# Example:
# HM(1, 2, 4)


class Mean_Type:
    QM = staticmethod(QM)
    AM = staticmethod(AM)
    GM = staticmethod(GM)
    HM = staticmethod(HM)

# Example:
# Mean_Type.GM(1, 2, 4)


def Final_Value(P: float, r: float, t: float, n: int | str) -> float:
    """Return future value under discrete or continuous compounding."""
    if n == "inf":
        return P * math.exp(r * t)
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer or 'inf'.")
    return P * (1 + r / n) ** (n * t)

# Example:
# Final_Value(1000, 0.05, 2, 12)


def logit_grow(M: float, r: float, t: float) -> float:
    """Logistic growth curve M / (1 + exp(-rt))."""
    return M / (1 + math.exp(-r * t))

# Example:
# logit_grow(100, 0.3, 2)


def triangular_parts(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return lower, upper, strict upper, and strict lower triangular parts."""
    A = np.asarray(A)
    lower = np.tril(A)
    upper = np.triu(A)
    return lower, upper, A - lower, A - upper

# Example:
# triangular_parts(np.array([[1, 2], [3, 4]]))


# 這一區是 NumPy 基礎：矩陣切塊、排名、broadcasting、view/copy、meshgrid。
# 目的不是重寫 NumPy，而是把講義裡常見的一行技巧整理成可直接複習的函式。
def array_ranks(arr: ArrayLike) -> np.ndarray:
    """Return 0-indexed ranks by double argsort."""
    return np.asarray(arr).argsort().argsort()

# Example:
# array_ranks([30, 10, 20, 40])


def subtract_row_means(data: np.ndarray) -> np.ndarray:
    """Broadcast row means across columns and subtract them."""
    data = np.asarray(data, dtype=float)
    return data - data.mean(axis=1, keepdims=True)

# Example:
# subtract_row_means(np.array([[1, 2], [3, 5]], dtype=float))


def outer_product(v: ArrayLike) -> np.ndarray:
    """Compute an outer product by broadcasting."""
    v = np.asarray(v, dtype=float)
    # v[:, np.newaxis] 變成 column vector，v[np.newaxis, :] 變成 row vector，兩者相乘得到 outer product。
    return v[:, np.newaxis] * v[np.newaxis, :]

# Example:
# outer_product([1, 2, 3])


def pairwise_differences(points: np.ndarray) -> np.ndarray:
    """Return pairwise vector differences by broadcasting."""
    points = np.asarray(points, dtype=float)
    return points[:, np.newaxis, :] - points[np.newaxis, :, :]

# Example:
# pairwise_differences(np.array([[0, 0], [1, 2], [3, 4]]))


def view_copy_demo(v2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a view slice and an independent copy slice."""
    return v2[:2, :2], v2[:2, :2].copy()

# Example:
# view_copy_demo(np.array([10, 20, 30, 40]))


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

# Example:
# meshgrid_values((0, 1), (0, 1), 3)


def exactly_one_choice(**choices: bool) -> bool:
    """Return True only when exactly one boolean choice is selected."""
    return sum(bool(value) for value in choices.values()) == 1

# Example:
# exactly_one_choice(coffee=True, tea=False, milk=False)


# 這一區是 Sec 1 基礎題型：Boolean logic、整除餘數、浮點精度、regex、generator、
# copy、dict/set comprehension。這些函式主要用來快速回想語法在題目裡的用途。
def cafeteria_action(coffee: bool, tea: bool, milk: bool) -> str:
    """Coffee/tea/milk logic-gate example from Sec 1."""
    return "deliver the choice" if exactly_one_choice(coffee=coffee, tea=tea, milk=milk) else "light the error light"

# Example:
# cafeteria_action(True, False, False)


def truth_table_exactly_one(names: list[str]) -> list[tuple[tuple[int, ...], bool]]:
    """Enumerate every truth-table row and mark rows with exactly one True."""
    rows: list[tuple[tuple[int, ...], bool]] = []
    for state in product([False, True], repeat=len(names)):
        rows.append((tuple(int(value) for value in state), exactly_one_choice(**dict(zip(names, state)))))
    return rows

# Example:
# truth_table_exactly_one(["coffee", "tea", "milk"])


def quotient_remainder(dividend: int, divisor: int) -> tuple[int, int]:
    """Return integer quotient and remainder."""
    if divisor == 0:
        raise ZeroDivisionError("divisor cannot be zero.")
    return divmod(dividend, divisor)

# Example:
# quotient_remainder(17, 5)


def exact_decimal_check() -> tuple[bool, bool]:
    """Show exact 0.1 arithmetic with Decimal and Fraction."""
    return Decimal("0.1") * 3 == Decimal("0.3"), Fraction(1, 10) * 3 == Fraction(3, 10)

# Example:
# exact_decimal_check()


def float_isclose_sum(abs_tol: float = 0.0) -> bool:
    """Check the standard 0.1 + 0.1 + 0.1 floating-point example."""
    return isclose(0.1 + 0.1 + 0.1, 0.3, abs_tol=abs_tol)

# Example:
# float_isclose_sum()


def regex_find(pattern: str, text: str) -> list[str]:
    """Return every regex match as strings."""
    return re.findall(pattern, text)

# Example:
# regex_find(r"User\d", "User9, UserN, User8")


def regex_search_one(pattern: str, text: str) -> str | None:
    """Return the first regex match, or None when no match exists."""
    match = re.search(pattern, text)
    return match.group() if match else None

# Example:
# regex_search_one(r"[A-Z]\d+", "ID: A12, B34")


def regex_split(pattern: str, text: str, maxsplit: int = 0) -> list[str]:
    """Split text by a regex pattern."""
    return re.split(pattern, text, maxsplit=maxsplit)

# Example:
# regex_split(r"\s", "The rain in Spain", maxsplit=1)


def regex_replace(pattern: str, repl: str, text: str, count: int = 0) -> str:
    """Replace regex matches in text."""
    return re.sub(pattern, repl, text, count=count)

# Example:
# regex_replace("yellow", "nice", "yellow car yellow house", count=1)


def seq_div(n: int, dv_set: list[int]) -> Any:
    """Yield numbers below n divisible by every value in dv_set."""
    if any(k == 0 for k in dv_set):
        raise ZeroDivisionError("dv_set cannot contain zero.")
    for i in range(n):
        if all(i % k == 0 for k in dv_set):
            yield i

# Example:
# seq_div(12, [2, 3, 5])


def exam_countdown(n: int) -> Any:
    """Yield n, n-1, ..., 1."""
    while n > 0:
        yield n
        n -= 1

# Example:
# list(exam_countdown(5))


def generator_consumption_demo(n: int) -> tuple[list[int], list[int]]:
    """Return first and second materialisation of the same generator."""
    gen = (k * k for k in range(n))
    # generator 只能往前消耗；第一次 list(gen) 已經取完，第二次就會是空 list。
    return list(gen), list(gen)

# Example:
# generator_consumption_demo(5)


def join_words(words: list[str], sep: str = " ") -> str:
    """Join strings with sep."""
    return sep.join(words)

# Example:
# join_words(["data", "science"], sep="-")


def shallow_deep_copy_demo(nested: list[list[int]]) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Mutate nested after copying to show shallow vs deep copy behavior."""
    import copy

    original = [row.copy() for row in nested]
    shallow = original.copy()
    deep = copy.deepcopy(original)
    original[0].append(99)
    return original, shallow, deep

# Example:
# shallow_deep_copy_demo([[1, 2], [3, 4]])


def zero_truncated_poisson_pmf(k: int, lam: float) -> float:
    """PMF of a zero-truncated Poisson random variable."""
    if k < 1:
        return 0.0
    if lam <= 0:
        raise ValueError("lam must be positive.")
    return math.exp(-lam) * lam**k / (math.factorial(k) * (1 - math.exp(-lam)))

# Example:
# zero_truncated_poisson_pmf(3, 2.5)


def dict_filter_by_value(data: dict[str, float], threshold: float) -> dict[str, float]:
    """Dictionary-comprehension example: keep values above threshold."""
    return {key: value for key, value in data.items() if value > threshold}

# Example:
# dict_filter_by_value({"a": 1.2, "b": 3.4}, 2.0)


def set_operations(a: set[Any], b: set[Any]) -> dict[str, set[Any]]:
    """Return common set operations."""
    return {
        "union": a | b,
        "intersection": a & b,
        "difference": a - b,
        "symmetric_difference": a ^ b,
    }

# Example:
# set_operations({1, 2, 3}, {2, 3, 4})


def increasing_rate(values: ArrayLike) -> np.ndarray:
    """Return period-over-period growth rates."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) < 2:
        raise ValueError("values must be a 1-D sequence with at least two entries.")
    if np.any(arr[:-1] == 0):
        raise ZeroDivisionError("previous value cannot be zero.")
    return arr[1:] / arr[:-1] - 1

# Example:
# increasing_rate([100, 110, 121])


def lcg_sequence(X0: int, a: int, c: int, m: int, size: int) -> list[int]:
    """Notebook-style LCG that returns integer states including the seed."""
    _validate_positive_n(size)
    if m <= 0:
        raise ValueError("m must be positive.")
    xs = [X0]
    for i in range(1, size):
        xs.append((a * xs[i - 1] + c) % m)
    return xs

# Example:
# lcg_sequence(42, 1664525, 1013904223, 2**32, 5)


# 這一區是 Sec 2 / Slides 補充：隨機模擬、Steffensen、exception flow、for...else、
# discrete inverse transform。重點是控制流程和模擬資料如何一步一步累積。
def steffensen(
    g: Callable[[float], float],
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Accelerated fixed-point iteration by Steffensen's method."""
    x = x0
    for _ in range(max_iter):
        gx = g(x)
        ggx = g(gx)
        denom = ggx - 2 * gx + x
        if denom == 0:
            raise ZeroDivisionError("Steffensen denominator is zero.")
        x_new = x - (gx - x) ** 2 / denom
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise RuntimeError("Steffensen's method did not converge.")

# Example:
# steffensen(lambda x: np.cos(x), 0.5)


def dice_running_mean(runs: int, sides: int = 6) -> np.ndarray:
    """Simulate cumulative dice means for the law of large numbers."""
    _validate_positive_n(runs)
    rolls = np.random.randint(1, sides + 1, runs)
    return np.cumsum(rolls) / np.arange(1, runs + 1)

# Example:
# dice_running_mean(10)


def random_walk_2d(bound: int = 25, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one symmetric 2-D random walk until it leaves a square boundary."""
    if bound <= 0:
        raise ValueError("bound must be positive.")
    rng = random.Random(seed)
    x = [0]
    y = [0]
    # 每次只走上下左右其中一步，直到路徑離開 [-bound, bound]^2。
    while abs(x[-1]) <= bound and abs(y[-1]) <= bound:
        step = rng.randint(1, 4)
        dx, dy = {1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}[step]
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
    return np.array(x), np.array(y)

# Example:
# random_walk_2d(bound=5, seed=0)


def present_value(F: float, r: float, t: float, n: int | str = "inf") -> float:
    """Discount future value F back to present value."""
    if n == "inf":
        return F * math.exp(-r * t)
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer or 'inf'.")
    return F / ((1 + r / n) ** (n * t))

# Example:
# present_value(1100, 0.05, 2, 12)


# 這一區補 Sec 3 的函式寫法：default argument、weighted average、parser、
# cosine similarity、Pydantic validation、closure、factory/classmethod。
def grade_safe(mid: float, final: float, base: list[float] | None = None) -> float:
    """Weighted two-component grade using a safe default weight list."""
    weights = base if base is not None else [0.5, 0.5]
    if len(weights) != 2 or not math.isclose(sum(weights), 1.0):
        raise ValueError("weights must contain two values and sum to 1.")
    return mid * weights[0] + final * weights[1]

# Example:
# grade_safe(80, 90)


def weighted_avg(scores: list[float], weights: list[float]) -> float:
    """Weighted average with automatic normalisation."""
    if len(scores) != len(weights):
        raise ValueError("scores and weights must have equal length.")
    total_w = sum(weights)
    if total_w == 0:
        raise ZeroDivisionError("sum(weights) cannot be zero.")
    return sum(score * weight for score, weight in zip(scores, weights)) / total_w

# Example:
# weighted_avg([80, 90, 70], [0.3, 0.4, 0.3])


def parser(address: str) -> str | None:
    """Return the name part before @, or None for non-email-like input."""
    if "@" not in address:
        return None
    name, _area = address.split("@", maxsplit=1)
    return name

# Example:
# parser("Taipei 100")


def cosine_sim(v1: list[float], v2: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if len(v1) != len(v2):
        raise ValueError("v1 and v2 must have equal length.")
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a**2 for a in v1))
    norm2 = math.sqrt(sum(b**2 for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

# Example:
# cosine_sim([1, 0, 1], [1, 1, 0])


class Product(BaseModel):
    """Pydantic coercion example."""

    price: int

# Example:
# Product(price=25)


class User(BaseModel):
    """Pydantic strict-field example."""

    age: int = Field(strict=True)

# Example:
# User(age=20)


class Material(BaseModel):
    """PositiveFloat validation example."""

    name: str
    density: PositiveFloat

# Example:
# Material(name="Iron", density=7.8)


class Guest(BaseModel):
    """Guest model with age constraint."""

    name: str
    age: Annotated[int, Field(ge=18)]

# Example:
# Guest(name="Amy", age=20)


@validate_call
def NightClub_check(guest: Guest, room_No: int) -> str:
    """Validate a guest and return the admission message."""
    return f"success : {guest.name} enter room {room_No}"

# Example:
# NightClub_check(Guest(name="Amy", age=20), 3)


class Address(BaseModel):
    city: str
    zip_code: str = Field(pattern=r"^\d{5}$")

# Example:
# Address(city="Taipei", zip_code="10001")


class Employee(BaseModel):
    name: str
    salary: float = Field(gt=0)
    address: Address

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, value: str) -> str:
        # validator 在 model 建立時清理資料；這裡把空白名字擋掉，並統一 title case。
        if not value.strip():
            raise ValueError("name cannot be blank")
        return value.title()

# Example:
# Employee(name="amy", salary=50000, address=Address(city="Taipei", zip_code="10001"))


def debug_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call fn with args and kwargs; useful for debugging argument flow."""
    return fn(*args, **kwargs)

# Example:
# debug_call(pow, 2, 5)


def merge_dicts_new(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge dictionaries with Python 3.9+ | without mutating either input."""
    return left | right

# Example:
# merge_dicts_new({"a": 1}, {"b": 2})


def gompertz(M: float, b: float, c: float, t: float) -> float:
    """Gompertz growth model."""
    return M * math.exp(-b * math.exp(-c * t))

# Example:
# gompertz(100, 2, 0.4, 3)


def power_mean(p: int, *args: float) -> float:
    """Generalised p-th power mean."""
    values = _positive_values(args)
    if p == 0:
        raise ValueError("Use GM for the p -> 0 limit.")
    return float((np.mean(values**p)) ** (1 / p))

# Example:
# power_mean(3, 1, 2, 4)


def fibonacci_GS(n: int) -> int:
    """Notebook alias for Binet Fibonacci."""
    return fib_binet(n)

# Example:
# fibonacci_GS(10)


def fibonacci_memo(n: int) -> int:
    """Notebook alias for memoized Fibonacci."""
    return fib_memo(n)

# Example:
# fibonacci_memo(10)


def fibonacci_iter(n: int) -> int:
    """Notebook alias for iterative Fibonacci."""
    return fib_iterative(n)

# Example:
# fibonacci_iter(10)


def fibonacci_until(target: int, max_iter: int = 1000) -> tuple[int, int]:
    """Return the first Fibonacci number >= target and its index."""
    if target <= 0:
        return 0, 0
    a, b, idx = 0, 1, 1
    while b < target:
        if idx >= max_iter:
            raise RuntimeError(f"Did not converge within {max_iter} iterations.")
        a, b, idx = b, a + b, idx + 1
    return b, idx

# Example:
# fibonacci_until(100)


def make_recurrence(coeffs: list[float], init: list[float]) -> Callable[[int], float]:
    """Return a memoized linear recurrence function."""
    if len(coeffs) != len(init):
        raise ValueError("coeffs and init must have the same length.")
    # cache 保存已算過的 T(n)，避免遞迴重複展開同一批子問題。
    cache: dict[int, float] = {k: value for k, value in enumerate(init)}

    def T(n: int) -> float:
        if n < 0:
            raise ValueError("n must be non-negative.")
        if n in cache:
            return cache[n]
        value = sum(coef * T(n - 1 - i) for i, coef in enumerate(coeffs))
        cache[n] = value
        return value

    return T

# Example:
# T = make_recurrence([4, 3, -18], [1, 1, 2]); T(4)


def tri_closed(n: int) -> int:
    """Closed-form triangular number."""
    _validate_nonnegative_int(n)
    return n * (n + 1) // 2

# Example:
# tri_closed(10)


def tri_inverse(target: int) -> int:
    """Smallest n such that n(n+1)/2 >= target."""
    if target <= 0:
        return 0
    return math.ceil((-1 + math.sqrt(1 + 8 * target)) / 2)

# Example:
# tri_inverse(56)


def tri_seq(n: int) -> int:
    """Recursive triangular number."""
    if n < 1:
        raise ValueError("n must be >= 1.")
    if n == 1:
        return 1
    return tri_seq(n - 1) + n

# Example:
# tri_seq(10)


def pascal_I(n: int, k: int) -> int:
    """Recursive Pascal coefficient."""
    if k < 0 or k > n:
        return 0
    if k == 0 or n == k:
        return 1
    return pascal_I(n - 1, k - 1) + pascal_I(n - 1, k)

# Example:
# pascal_I(5, 2)


def pascal_II(n: int, k: int) -> int:
    """Multiplicative Pascal coefficient."""
    if k < 0 or k > n:
        return 0
    if k == 0 or n == k:
        return 1
    return int(pascal_II(n, k - 1) * ((n - k + 1) / k))

# Example:
# pascal_II(5, 2)


def pascal_dp(max_n: int) -> list[list[int]]:
    """Build Pascal's triangle up to row max_n."""
    _validate_nonnegative_int(max_n)
    table = [[1] * (i + 1) for i in range(max_n + 1)]
    for i in range(2, max_n + 1):
        for j in range(1, i):
            table[i][j] = table[i - 1][j - 1] + table[i - 1][j]
    return table

# Example:
# pascal_dp(5)


def make_counter(start: int = 0) -> tuple[Callable[[int], int], Callable[[], None]]:
    """Return increment/reset closures that share nonlocal state."""
    count = start

    def increment(step: int = 1) -> int:
        # nonlocal 讓內層函式修改 make_counter 作用域裡的 count。
        nonlocal count
        count += step
        return count

    def reset() -> None:
        nonlocal count
        count = start

    return increment, reset

# Example:
# inc, reset = make_counter(10); inc(2)


class Car:
    """Class example with validation and dunder methods."""

    def __init__(self, brand: str, year: int) -> None:
        if year < 1886:
            raise ValueError(f"Invalid year: {year}")
        self.brand = brand
        self.year = year

    def __repr__(self) -> str:
        return f"Car(brand={self.brand!r}, year={self.year})"

    def __str__(self) -> str:
        return f"{self.year} {self.brand}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Car):
            return NotImplemented
        return self.brand == other.brand and self.year == other.year

# Example:
# Car("Toyota", 2020) == Car("Toyota", 2020)


class Vehicle:
    def __init__(self, brand: str) -> None:
        self.brand = brand

    def info(self) -> str:
        return f"Vehicle: {self.brand}"

# Example:
# Vehicle("Toyota").info()


class Electric:
    def __init__(self, battery_kwh: float) -> None:
        self.battery_kwh = battery_kwh

    def range_km(self) -> float:
        return self.battery_kwh * 6

# Example:
# Electric(60).range_km()


class ElectricCar(Vehicle, Electric):
    def __init__(self, brand: str, battery_kwh: float) -> None:
        Vehicle.__init__(self, brand)
        Electric.__init__(self, battery_kwh)

    def info(self) -> str:
        return f"{super().info()} | Battery: {self.battery_kwh} kWh"

# Example:
# ElectricCar("Tesla", 75).info()


class Stats:
    """Descriptive statistics container."""

    def __init__(self, data: list[float]) -> None:
        self._data = data

    @classmethod
    def from_csv_line(cls, line: str) -> "Stats":
        values = [float(value.strip()) for value in line.split(",")]
        return cls(values)

    @staticmethod
    def _validate(data: list[float]) -> None:
        if not data:
            raise ValueError("Empty data.")
        if any(value <= 0 for value in data):
            raise ValueError("All values must be positive.")

    def summary(self) -> dict[str, float]:
        self._validate(self._data)
        n = len(self._data)
        return {
            "AM": sum(self._data) / n,
            "GM": float(np.prod(self._data) ** (1 / n)),
            "HM": n / sum(1 / value for value in self._data),
            "QM": (sum(value**2 for value in self._data) / n) ** 0.5,
        }

# Example:
# Stats.from_csv_line("1, 2, 4").summary()


class iCounter:
    """Class variable counter example."""

    count = 0

    def __init__(self) -> None:
        iCounter.count += 1

    @classmethod
    def kids(cls) -> str:
        return f"iCounter create {cls.count} little objects."

# Example:
# iCounter(); iCounter(); iCounter.kids()


class Singleton:
    """Simple singleton via __new__."""

    _instance: "Singleton | None" = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "Singleton":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, value: int) -> None:
        self.value = value

# Example:
# Singleton(1) is Singleton(2)


class ShapeFactory:
    """Registry pattern using classmethod decorators."""

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type], type]:
        # register 回傳 decorator，讓 class 定義時自動被放進 registry。
        def decorator(shape_cls: type) -> type:
            cls._registry[name] = shape_cls
            return shape_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Any:
        if name not in cls._registry:
            raise KeyError(f"Unknown shape: {name}")
        return cls._registry[name](**kwargs)

# Example:
# ShapeFactory.create("circle", radius=3).area()


@ShapeFactory.register("circle")
class Circle:
    def __init__(self, radius: float) -> None:
        self.radius = radius

    def area(self) -> float:
        return math.pi * self.radius**2

# Example:
# Circle(3).area()


def sliding_window(arr: ArrayLike, w: int) -> np.ndarray:
    """Return all length-w sliding windows."""
    if w <= 0:
        raise ValueError("w must be positive.")
    arr_np = np.asarray(arr)
    if w > len(arr_np):
        raise ValueError("w cannot exceed len(arr).")
    return np.array([arr_np[i : i + w] for i in range(len(arr_np) - w + 1)])

# Example:
# sliding_window([1, 2, 3, 4], 2)


# 這一區是 NumPy 進階補充：vectorization、piecewise、projection、ufunc out/reduce、
# mask 與排序算法。註解重點放在「為什麼用 NumPy 寫法」和「資料形狀怎麼變」。
def cal_recip(seq: ArrayLike) -> np.ndarray:
    """Loop-based reciprocal calculation from the vectorization lesson."""
    seq_np = np.asarray(seq, dtype=float)
    if np.any(seq_np == 0):
        raise ZeroDivisionError("seq cannot contain zero.")
    result = np.empty(len(seq_np))
    for k in range(len(result)):
        result[k] = 1 / seq_np[k]
    return result

# Example:
# cal_recip([2, 4, 8])


def heaviside_1(x: float) -> int:
    """Scalar Heaviside function."""
    return 1 if x > 0 else 0

# Example:
# heaviside_1(-3)


def heaviside_3(x: ArrayLike) -> np.ndarray:
    """Vectorized Heaviside using boolean masks."""
    return 1 * (np.asarray(x) > 0)

# Example:
# heaviside_3([-1, 0, 2])


def ramp(x: ArrayLike) -> np.ndarray:
    """Ramp function: 0 for x < 0, x for x >= 0."""
    x_np = np.asarray(x, dtype=float)
    return np.piecewise(x_np, [x_np < 0, x_np >= 0], [0, lambda value: value])

# Example:
# ramp([-2, -1, 0, 3])


def block_matrix(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Build [[A, B], [C, D]] with np.block."""
    return np.block([[A, B], [C, D]])

# Example:
# block_matrix(np.eye(2), np.ones((2, 1)), np.zeros((1, 2)), np.array([[9]]))


def projection_matrix(A: np.ndarray) -> np.ndarray:
    """Projection matrix onto the column space of A."""
    A = np.asarray(A, dtype=float)
    # P = A(A^T A)^(-1)A^T，將任意向量投影到 A 的 column space。
    return A @ np.linalg.inv(A.T @ A) @ A.T

# Example:
# projection_matrix(np.array([[1.0], [1.0]]))


def project_vector(A: np.ndarray, b: ArrayLike) -> np.ndarray:
    """Project b onto the column space of A."""
    return projection_matrix(A) @ np.asarray(b, dtype=float)

# Example:
# project_vector(np.array([[1.0], [1.0]]), [2.0, 0.0])


def make_df(cols: str, ind: list[Any]) -> pd.DataFrame:
    """Quickly make a toy DataFrame with columns named by cols."""
    data = {col: [f"{col}{i}" for i in ind] for col in cols}
    return pd.DataFrame(data, index=ind)

# Example:
# make_df("AB", [0, 1, 2])


# 這一區是 pandas：missing data、concat/merge、groupby/transform/apply、pivot。
# 大多數函式會先 copy，避免為了示範而不小心修改原始 DataFrame。
def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and rates by column."""
    count = df.isna().sum()
    return pd.DataFrame({"missing": count, "rate": count / len(df)})

# Example:
# missing_summary(pd.DataFrame({"x": [1.0, np.nan], "y": [np.nan, 2.0]}))


def fill_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Fill numeric missing values by mean, median, or zero."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    if strategy == "mean":
        out[numeric_cols] = out[numeric_cols].fillna(out[numeric_cols].mean())
    elif strategy == "median":
        out[numeric_cols] = out[numeric_cols].fillna(out[numeric_cols].median())
    elif strategy == "zero":
        out[numeric_cols] = out[numeric_cols].fillna(0)
    else:
        raise ValueError("strategy must be 'mean', 'median', or 'zero'.")
    return out

# Example:
# fill_missing(pd.DataFrame({"x": [1.0, np.nan], "y": [np.nan, 2.0]}), "mean")


def concat_frames(frames: list[pd.DataFrame], axis: int = 0, ignore_index: bool = False) -> pd.DataFrame:
    """Wrapper around pd.concat for course examples."""
    return pd.concat(frames, axis=axis, ignore_index=ignore_index)

# Example:
# concat_frames([pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]})], ignore_index=True)


def merge_frames(left: pd.DataFrame, right: pd.DataFrame, on: str, how: str = "inner") -> pd.DataFrame:
    """Wrapper around pd.merge for key-based joins."""
    return pd.merge(left, right, on=on, how=how)

# Example:
# merge_frames(pd.DataFrame({"id": [1, 2]}), pd.DataFrame({"id": [2], "x": [9]}), on="id")


def iqr(series: pd.Series) -> float:
    """Interquartile range Q3 - Q1."""
    return float(series.quantile(0.75) - series.quantile(0.25))

# Example:
# iqr(pd.Series([1, 2, 3, 4]))


def zscore(x: pd.Series) -> pd.Series:
    """Series z-score used with groupby.transform."""
    std = x.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x.mean()) / std

# Example:
# zscore(pd.Series([1.0, 3.0, 5.0]))


def group_zscore(df: pd.DataFrame, group_col: str, value_col: str, out_col: str | None = None) -> pd.DataFrame:
    """Add a within-group z-score column."""
    out = df.copy()
    target = out_col or f"{value_col}_zscore"
    # transform 會回傳和原資料同長度的 Series，所以能直接接回原 DataFrame 當新欄位。
    out[target] = out.groupby(group_col)[value_col].transform(zscore)
    return out

# Example:
# group_zscore(pd.DataFrame({"g": ["a", "a", "b", "b"], "x": [1, 3, 10, 14]}), "g", "x")


def group_summary(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Return n, mean, std, cv, and iqr per group."""
    grouped = df.groupby(group_col)[value_col]
    result = grouped.agg(n="size", mean="mean", std="std", iqr=iqr)
    result["cv"] = result["std"] / result["mean"]
    return result

# Example:
# group_summary(pd.DataFrame({"g": ["a", "a", "b", "b"], "x": [1, 3, 10, 14]}), "g", "x")


def pivot_counts(df: pd.DataFrame, index: str, columns: str) -> pd.DataFrame:
    """Count rows in a pivot table."""
    return pd.pivot_table(df, index=index, columns=columns, aggfunc="size", fill_value=0)

# Example:
# pivot_counts(pd.DataFrame({"sex": ["F", "F", "M"], "dept": ["A", "B", "A"]}), "sex", "dept")


def plot_line(x: ArrayLike, y: ArrayLike, *, title: str = "", ax: Any | None = None) -> Any:
    """Create a basic Matplotlib line plot and return the axis."""
    axis = ax or plt.subplots()[1]
    axis.plot(x, y)
    axis.set_title(title)
    return axis

# Example:
# plot_line([1, 2, 3], [1, 4, 9], title="line")


# 這一區是 visualization helper：統一使用 Matplotlib OO-style，回傳 axis/figure，
# 這樣外部可以繼續疊圖、改 labels 或存檔。
def plot_scatter(x: ArrayLike, y: ArrayLike, *, title: str = "", ax: Any | None = None) -> Any:
    """Create a scatter plot and return the axis."""
    axis = ax or plt.subplots()[1]
    axis.scatter(x, y)
    axis.set_title(title)
    return axis

# Example:
# plot_scatter([1, 2, 3], [1, 4, 9], title="scatter")


def plot_errorbar(
    x: ArrayLike,
    y: ArrayLike,
    yerr: ArrayLike,
    *,
    title: str = "",
    ax: Any | None = None,
) -> Any:
    """Create an errorbar plot and return the axis."""
    axis = ax or plt.subplots()[1]
    axis.errorbar(x, y, yerr=yerr, fmt="o", capsize=3)
    axis.set_title(title)
    return axis

# Example:
# plot_errorbar([1, 2, 3], [2, 4, 8], [0.2, 0.3, 0.4], title="error")


def plot_confidence_band(
    x: ArrayLike,
    y: ArrayLike,
    err: ArrayLike,
    *,
    ax: Any | None = None,
    alpha: float = 0.25,
) -> Any:
    """Plot a line with a continuous confidence band."""
    axis = ax or plt.subplots()[1]
    x_np = np.asarray(x, dtype=float)
    y_np = np.asarray(y, dtype=float)
    err_np = np.asarray(err, dtype=float)
    axis.plot(x_np, y_np)
    axis.fill_between(x_np, y_np - err_np, y_np + err_np, alpha=alpha)
    return axis

# Example:
# plot_confidence_band([1, 2, 3], [2, 4, 8], [0.2, 0.3, 0.4])


def contour_grid(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    n: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, Y, Z arrays for contour or surface plots."""
    _validate_positive_n(n)
    x = np.linspace(xlim[0], xlim[1], n)
    y = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(x, y)
    return X, Y, f(X, Y)

# Example:
# contour_grid(lambda X, Y: X**2 + Y**2, (-1, 1), (-1, 1), 5)


def plot_histogram(data: ArrayLike, bins: int = 30, *, density: bool = True, ax: Any | None = None) -> Any:
    """Create a histogram and return the axis."""
    axis = ax or plt.subplots()[1]
    axis.hist(data, bins=bins, density=density)
    return axis

# Example:
# plot_histogram([1, 1, 2, 3, 5, 8], bins=4)


def plot_color_gradients(cmap_category: str, cmap_list: list[str]) -> tuple[Any, Any]:
    """Plot Matplotlib colormap gradients."""
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows, figsize=(6.4, figh))
    axs_arr = np.atleast_1d(axs)
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh, left=0.2, right=0.99)
    axs_arr[0].set_title(f"{cmap_category} colormaps", fontsize=14)
    for ax, cmap_name in zip(axs_arr, cmap_list):
        ax.imshow(gradient, aspect="auto", cmap=cmap_name)
        ax.text(-0.01, 0.5, cmap_name, va="center", ha="right", fontsize=10, transform=ax.transAxes)
        ax.set_axis_off()
    return fig, axs

# Example:
# plot_color_gradients("sequential", ["viridis", "magma"])


def conditional_expression(condition: bool, true_value: Any, false_value: Any) -> Any:
    """Return true_value if condition else false_value."""
    return true_value if condition else false_value

# Example:
# conditional_expression(3 > 2, 10, 20)


def try_except_else_finally_demo(value: str) -> tuple[str, list[str]]:
    """Show try/except/else/finally control flow with int parsing."""
    events: list[str] = []
    try:
        parsed = int(value)
    except ValueError:
        events.append("except")
        result = "invalid"
    else:
        events.append("else")
        result = f"parsed:{parsed}"
    finally:
        events.append("finally")
    return result, events

# Example:
# try_except_else_finally_demo("12")


def loop_else_find(iterable: list[Any], target: Any) -> tuple[bool, int | None]:
    """Use for...else: else runs only when the loop does not break."""
    for idx, value in enumerate(iterable):
        if value == target:
            return True, idx
    else:
        return False, None

# Example:
# loop_else_find([10, 20, 30], 20)


def inverse_transform_discrete(values: ArrayLike, probs: ArrayLike, n: int) -> np.ndarray:
    """Generate samples from a finite distribution by discrete inverse transform."""
    _validate_positive_n(n)
    values_np = np.asarray(values)
    probs_np = np.asarray(probs, dtype=float)
    if len(values_np) != len(probs_np):
        raise ValueError("values and probs must have equal length.")
    if np.any(probs_np < 0) or not np.isclose(probs_np.sum(), 1.0):
        raise ValueError("probs must be non-negative and sum to 1.")
    # 把機率加總成 CDF，再用 U 落在哪個區間決定抽到哪個離散值。
    cdf = np.cumsum(probs_np)
    u = np.random.uniform(0, 1, n)
    idx = np.searchsorted(cdf, u, side="right")
    return values_np[idx]

# Example:
# inverse_transform_discrete([1, 2, 3], [0.2, 0.3, 0.5], 5)


def cauchy_inverse_transform(n: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Generate Cauchy(mu, sigma) samples by inverse transform."""
    _validate_positive_n(n)
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    u = np.random.uniform(0, 1, n)
    return mu + sigma * np.tan(np.pi * (u - 0.5))

# Example:
# cauchy_inverse_transform(5, mu=0.0, sigma=1.0)


def array_attributes(arr: np.ndarray) -> dict[str, Any]:
    """Return common NumPy array attributes from the slides."""
    arr_np = np.asarray(arr)
    return {
        "ndim": arr_np.ndim,
        "shape": arr_np.shape,
        "size": arr_np.size,
        "dtype": arr_np.dtype,
        "itemsize": arr_np.itemsize,
        "nbytes": arr_np.nbytes,
    }

# Example:
# array_attributes(np.array([[1, 2], [3, 4]]))


def int_array_float_truncation(values: list[float]) -> np.ndarray:
    """Assign floats into an int array to show silent truncation."""
    arr = np.zeros(len(values), dtype=int)
    arr[:] = values
    return arr

# Example:
# int_array_float_truncation([1.9, -3.1])


def concatenate_stack(a: ArrayLike, b: ArrayLike) -> dict[str, np.ndarray]:
    """Show concatenate/vstack/hstack outputs for two arrays."""
    a_np = np.asarray(a)
    b_np = np.asarray(b)
    return {
        "concatenate": np.concatenate([a_np, b_np]),
        "vstack": np.vstack([a_np, b_np]),
        "hstack": np.hstack([a_np, b_np]),
    }

# Example:
# concatenate_stack([1, 2], [3, 4])


def split_array(arr: ArrayLike, sections: int, axis: int = 0) -> list[np.ndarray]:
    """Split an array into equal sections along an axis."""
    return list(np.split(np.asarray(arr), sections, axis=axis))

# Example:
# split_array(np.arange(8), 4)


def ufunc_out_square(values: ArrayLike) -> np.ndarray:
    """Use a ufunc out argument to store results in a preallocated array."""
    arr = np.asarray(values, dtype=float)
    out = np.empty_like(arr)
    # out 讓 ufunc 把結果寫進既有陣列，避免多建立一份中間結果。
    np.multiply(arr, arr, out=out)
    return out

# Example:
# ufunc_out_square([2, 3, 4])


def ufunc_reduce_sum(values: ArrayLike) -> float:
    """Aggregate by np.add.reduce."""
    return float(np.add.reduce(np.asarray(values, dtype=float)))

# Example:
# ufunc_reduce_sum([1, 2, 3, 4])


def nan_safe_aggregates(values: ArrayLike) -> dict[str, float]:
    """Return NaN-safe aggregate values."""
    arr = np.asarray(values, dtype=float)
    return {
        "nanmin": float(np.nanmin(arr)),
        "nanmax": float(np.nanmax(arr)),
        "nanmean": float(np.nanmean(arr)),
        "nansum": float(np.nansum(arr)),
    }

# Example:
# nan_safe_aggregates([1.0, np.nan, 3.0])


def mask_between(arr: ArrayLike, low: float, high: float, inclusive: bool = True) -> np.ndarray:
    """Return values between low and high by NumPy masking."""
    arr_np = np.asarray(arr)
    if inclusive:
        mask = (arr_np >= low) & (arr_np <= high)
    else:
        mask = (arr_np > low) & (arr_np < high)
    return arr_np[mask]

# Example:
# mask_between([1, 2, 3, 4, 5], 2, 4)


def mask_count(arr: ArrayLike, predicate: Callable[[np.ndarray], np.ndarray]) -> int:
    """Count values satisfying a vectorized predicate."""
    return int(np.count_nonzero(predicate(np.asarray(arr))))

# Example:
# mask_count([1, 2, 3, 4, 5], lambda x: x % 2 == 0)


def selection_sort(values: list[float]) -> list[float]:
    """Selection sort, O(n^2), included for sorting-algorithm slides."""
    arr = list(values)
    for i in range(len(arr) - 1):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# Example:
# selection_sort([5, 2, 4, 1])


def bubble_sort(values: list[float]) -> list[float]:
    """Bubble sort, O(n^2), included for sorting-algorithm slides."""
    arr = list(values)
    for end in range(len(arr) - 1, 0, -1):
        swapped = False
        for i in range(end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        if not swapped:
            break
    return arr

# Example:
# bubble_sort([5, 2, 4, 1])


def insertion_sort(values: list[float]) -> list[float]:
    """Insertion sort: best case O(n), worst case O(n^2)."""
    arr = list(values)
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# Example:
# insertion_sort([5, 2, 4, 1])


def merge_sort(values: list[float]) -> list[float]:
    """Merge sort, O(n log n)."""
    arr = list(values)
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    merged: list[float] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

# Example:
# merge_sort([5, 2, 4, 1])


def quick_sort(values: list[float]) -> list[float]:
    """Simple quick sort: average O(n log n), worst O(n^2)."""
    arr = list(values)
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + mid + quick_sort(right)

# Example:
# quick_sort([5, 2, 4, 1])


def solve_linear_system(A: np.ndarray, b: ArrayLike) -> np.ndarray:
    """Solve Ax=b with np.linalg.solve."""
    return np.linalg.solve(np.asarray(A, dtype=float), np.asarray(b, dtype=float))

# Example:
# solve_linear_system(np.array([[2, 1], [1, 3]]), [1, 2])


def series_from_mapping(data: dict[Any, Any]) -> pd.Series:
    """Create a Pandas Series from a dictionary-like mapping."""
    return pd.Series(data)

# Example:
# series_from_mapping({"a": 1, "b": 2})


def dataframe_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact first-check profile similar to head/info/dtypes."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "head": df.head(),
        "tail": df.tail(),
    }

# Example:
# dataframe_profile(pd.DataFrame({"x": [1, 2], "y": [3, 4]}))


def select_loc_iloc(df: pd.DataFrame, loc_key: Any, iloc_key: Any) -> tuple[Any, Any]:
    """Return matching loc and iloc selections to compare explicit vs positional indexing."""
    return df.loc[loc_key], df.iloc[iloc_key]

# Example:
# select_loc_iloc(pd.DataFrame({"x": [1, 2]}, index=["a", "b"]), "a", 1)


def index_aligned_add(left: pd.Series, right: pd.Series, fill_value: float | None = None) -> pd.Series:
    """Add Series with Pandas index alignment."""
    if fill_value is None:
        return left + right
    return left.add(right, fill_value=fill_value)

# Example:
# index_aligned_add(pd.Series([1, 2], index=["a", "b"]), pd.Series([10, 20], index=["b", "c"]), fill_value=0)


def interpolate_missing(series: pd.Series, method: str = "linear") -> pd.Series:
    """Interpolate missing values in a Series."""
    return series.interpolate(method=method)

# Example:
# interpolate_missing(pd.Series([1.0, np.nan, 3.0]))


def nullable_int_series(values: list[Any]) -> pd.Series:
    """Create a nullable integer Series using Pandas Int64 dtype."""
    return pd.Series(values, dtype="Int64")

# Example:
# nullable_int_series([1, None, 3])


def add_missing_indicator(df: pd.DataFrame, column: str, suffix: str = "_missing") -> pd.DataFrame:
    """Add a dummy variable that indicates whether a column is missing."""
    out = df.copy()
    out[f"{column}{suffix}"] = out[column].isna().astype(int)
    return out

# Example:
# add_missing_indicator(pd.DataFrame({"x": [1, np.nan, 3]}), "x")


def relational_project(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Relational algebra projection: choose columns."""
    return df[columns].copy()

# Example:
# relational_project(pd.DataFrame({"id": [1, 2], "x": [3, 4]}), ["id"])


def relational_select(df: pd.DataFrame, predicate: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    """Relational algebra selection: filter rows by predicate."""
    return df[predicate(df)].copy()

# Example:
# relational_select(pd.DataFrame({"x": [1, 3, 5]}), lambda d: d["x"] > 2)


def relational_rename(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Relational algebra rename."""
    return df.rename(columns=mapping)

# Example:
# relational_rename(pd.DataFrame({"old": [1, 2]}), {"old": "new"})


def relational_union(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Relational algebra union with duplicate rows removed."""
    return pd.concat([left, right], ignore_index=True).drop_duplicates(ignore_index=True)

# Example:
# relational_union(pd.DataFrame({"x": [1, 2]}), pd.DataFrame({"x": [2, 3]}))


def relational_set_difference(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Rows in left that are not also in right."""
    # indicator=True 會標出每列來自 left_only / right_only / both，再保留 left_only。
    merged = left.merge(right.drop_duplicates(), how="left", indicator=True)
    return merged.loc[merged["_merge"] == "left_only", left.columns].reset_index(drop=True)

# Example:
# relational_set_difference(pd.DataFrame({"x": [1, 2, 3]}), pd.DataFrame({"x": [2]}))


def relational_cross_product(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Relational algebra cross product."""
    return left.merge(right, how="cross")

# Example:
# relational_cross_product(pd.DataFrame({"x": [1, 2]}), pd.DataFrame({"y": [10, 20]}))


def join_category(left: pd.DataFrame, right: pd.DataFrame, key: str) -> str:
    """Classify join cardinality as one-to-one, one-to-many, many-to-one, or many-to-many."""
    left_many = left[key].duplicated().any()
    right_many = right[key].duplicated().any()
    if left_many and right_many:
        return "many-to-many"
    if left_many:
        return "many-to-one"
    if right_many:
        return "one-to-many"
    return "one-to-one"

# Example:
# join_category(pd.DataFrame({"id": [1, 1, 2]}), pd.DataFrame({"id": [1, 2]}), "id")


def ax_set_labels(
    ax: Any,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> Any:
    """Use the Matplotlib OO-style ax.set shortcut from the slides."""
    kwargs: dict[str, Any] = {}
    if title is not None:
        kwargs["title"] = title
    if xlabel is not None:
        kwargs["xlabel"] = xlabel
    if ylabel is not None:
        kwargs["ylabel"] = ylabel
    if xlim is not None:
        kwargs["xlim"] = xlim
    if ylim is not None:
        kwargs["ylim"] = ylim
    ax.set(**kwargs)
    return ax

# Example:
# ax_set_labels(plt.subplots()[1], title="Demo", xlabel="x", ylabel="y")


def choose_colormap(data_kind: str) -> str:
    """Recommend a Matplotlib colormap category by data meaning."""
    normalized = data_kind.strip().lower()
    if normalized in {"ordered", "sequential", "magnitude", "density"}:
        return "viridis"
    if normalized in {"deviation", "diverging", "residual", "centered"}:
        return "coolwarm"
    if normalized in {"category", "categorical", "class", "qualitative"}:
        return "tab10"
    if normalized in {"cyclic", "phase", "angle"}:
        return "twilight"
    raise ValueError("Unknown data_kind; use sequential, diverging, categorical, or cyclic.")

# Example:
# choose_colormap("sequential")


def choose_plot_type(data_kind: str, goal: str) -> str:
    """Small decision helper based on the visualization-principle slides."""
    kind = data_kind.strip().lower()
    goal_norm = goal.strip().lower()
    if kind == "categorical" and goal_norm in {"compare", "comparison"}:
        return "bar"
    if kind == "numeric" and goal_norm in {"distribution", "hist"}:
        return "histogram"
    if kind == "numeric-pair" and goal_norm in {"relationship", "correlation"}:
        return "scatter"
    if kind == "time-series":
        return "line"
    if kind == "surface":
        return "contour"
    raise ValueError("No simple recommendation for the given data_kind and goal.")

# Example:
# choose_plot_type("numeric", "distribution")


def _validate_positive_n(n: int) -> None:
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")

# Example:
# _validate_positive_n(5)


def _validate_nonnegative_int(n: int) -> None:
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer.")

# Example:
# _validate_nonnegative_int(0)


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
    "exactly_one_choice",
    "cafeteria_action",
    "truth_table_exactly_one",
    "quotient_remainder",
    "exact_decimal_check",
    "float_isclose_sum",
    "regex_find",
    "regex_search_one",
    "regex_split",
    "regex_replace",
    "seq_div",
    "exam_countdown",
    "generator_consumption_demo",
    "join_words",
    "shallow_deep_copy_demo",
    "zero_truncated_poisson_pmf",
    "dict_filter_by_value",
    "set_operations",
    "increasing_rate",
    "lcg_sequence",
    "steffensen",
    "dice_running_mean",
    "random_walk_2d",
    "present_value",
    "grade_safe",
    "weighted_avg",
    "parser",
    "cosine_sim",
    "Product",
    "User",
    "Material",
    "Guest",
    "NightClub_check",
    "Address",
    "Employee",
    "debug_call",
    "merge_dicts_new",
    "gompertz",
    "power_mean",
    "fibonacci_GS",
    "fibonacci_memo",
    "fibonacci_iter",
    "fibonacci_until",
    "make_recurrence",
    "tri_closed",
    "tri_inverse",
    "tri_seq",
    "pascal_I",
    "pascal_II",
    "pascal_dp",
    "make_counter",
    "Car",
    "Vehicle",
    "Electric",
    "ElectricCar",
    "Stats",
    "iCounter",
    "Singleton",
    "ShapeFactory",
    "Circle",
    "sliding_window",
    "cal_recip",
    "heaviside_1",
    "heaviside_3",
    "ramp",
    "block_matrix",
    "projection_matrix",
    "project_vector",
    "make_df",
    "missing_summary",
    "fill_missing",
    "concat_frames",
    "merge_frames",
    "iqr",
    "zscore",
    "group_zscore",
    "group_summary",
    "pivot_counts",
    "plot_line",
    "plot_scatter",
    "plot_errorbar",
    "plot_confidence_band",
    "contour_grid",
    "plot_histogram",
    "plot_color_gradients",
    "conditional_expression",
    "try_except_else_finally_demo",
    "loop_else_find",
    "inverse_transform_discrete",
    "cauchy_inverse_transform",
    "array_attributes",
    "int_array_float_truncation",
    "concatenate_stack",
    "split_array",
    "ufunc_out_square",
    "ufunc_reduce_sum",
    "nan_safe_aggregates",
    "mask_between",
    "mask_count",
    "selection_sort",
    "bubble_sort",
    "insertion_sort",
    "merge_sort",
    "quick_sort",
    "solve_linear_system",
    "series_from_mapping",
    "dataframe_profile",
    "select_loc_iloc",
    "index_aligned_add",
    "interpolate_missing",
    "nullable_int_series",
    "add_missing_indicator",
    "relational_project",
    "relational_select",
    "relational_rename",
    "relational_union",
    "relational_set_difference",
    "relational_cross_product",
    "join_category",
    "ax_set_labels",
    "choose_colormap",
    "choose_plot_type",
]
