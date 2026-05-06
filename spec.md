# Exam Preparation Specification

---

## 0. Global Setup

### Standard Imports (all aliases must match exactly)

| Package | Import statement | Alias |
|---------|-----------------|-------|
| NumPy | `import numpy as np` | `np` |
| SciPy (top-level) | `import scipy as sc` | `sc` |
| SciPy stats | `import scipy.stats as scs` | `scs` |
| SciPy linalg | `from scipy import linalg` | `linalg` |
| SciPy linalg functions | `from scipy.linalg import inv, solve, lstsq, LinAlgError` | direct |
| SciPy integrate | `from scipy.integrate import quad` | direct |
| SciPy stats functions | `from scipy.stats import gmean, hmean` | direct |
| pandas | `import pandas as pd` | `pd` |
| Matplotlib | `import matplotlib.pyplot as plt` | `plt` |
| Seaborn | `import seaborn as sns` | `sns` |
| math | `import math` | direct |
| math functions | `from math import exp, log, sqrt, factorial, isclose, prod, pow` | direct |
| random | `import random` | direct |
| functools | `import functools` | direct |
| functools reduce | `from functools import reduce` | direct |
| itertools | `from itertools import product, combinations, accumulate` | direct |
| operator | `import operator` | direct |
| re | `import re` | direct |
| collections | `from collections import Counter` | direct |
| dataclasses | `from dataclasses import dataclass` | direct |
| typing | `from typing import Optional, Annotated` | direct |
| pydantic | `from pydantic import BaseModel, Field, field_validator, validate_call, ValidationError, PositiveFloat` | direct |
| decimal | `from decimal import Decimal` | direct |
| fractions | `from fractions import Fraction` | direct |

### Display / Style Settings

- `plt.style.use('ggplot')` — used in `Sec 4.1`
- `plt.style.use('seaborn-v0_8-bright')` — used in `Sec 4.2`
- `plt.style.use('seaborn-v0_8-whitegrid')` — used in `Sec 4.3`
- No `pd.set_option()` calls found; standard display assumed.

---

## 1. Topic Specifications

---

### Topic 1.1 — Gauss-Seidel Linear Solver (Matrix Form)

**Source:** `Sec 4.1 - Package Intro - numpy.ipynb`, `MidExam.ipynb`

**Mathematical definition:**

Decompose $A = L + U$ where $L = \texttt{np.tril}(A)$ (lower triangle including diagonal) and $U = A - L$ (strict upper triangle):

$$x^{(k+1)} = L^{-1}\bigl(b - U\,x^{(k)}\bigr)$$

Convergence criterion: $\|x^{(k+1)} - x^{(k)}\|_2 < \texttt{tol}$, i.e.,

$$\sqrt{(x^{(k+1)} - x^{(k)})^\top (x^{(k+1)} - x^{(k)})} < \texttt{tol}$$

**Interface:**

Function `Gauss_Siedel(A, b, x, tol=1e-3)`:
- Parameters: `A` — square numpy array $(n \times n)$; `b` — column vector $(n \times 1)$; `x` — initial guess column vector $(n \times 1)$; `tol` — float convergence tolerance (default `1e-3`).
- Returns: converged solution vector `x` of shape $(n \times 1)$.
- Side effects: prints iteration count and convergence message.

**Algorithm:**
1. Compute lower-triangular matrix $L = \texttt{np.tril}(A)$.
2. Compute strict upper-triangular matrix $U = A - L$.
3. Initialise iteration counter.
4. Loop:
   a. Compute $x_c = L^{-1}(b - U x)$.
   b. Increment counter.
   c. Compute $\|x_c - x\|_2$.
   d. If $\|x_c - x\|_2 < \texttt{tol}$: print convergence message, break.
   e. Print current iterate values.
   f. Set $x \leftarrow x_c$.
5. Return $x_c$.

**Constraints and edge cases:**
- $A$ must be strictly diagonally dominant to guarantee convergence (see Topic 1.2).
- $L$ must be invertible (diagonal elements $\neq 0$).
- Flag for Codex: the notebooks use `np.linalg.inv(L)` inside the loop (slow); `np.linalg.solve(L, ...)` is preferred but either is accepted.

**Test case:**

Input:
$$A = \begin{bmatrix} 8 & 3 & -3 \\ -2 & -8 & 5 \\ 3 & 5 & 10 \end{bmatrix}, \quad b = \begin{bmatrix} 14 \\ 5 \\ -8 \end{bmatrix}, \quad x_0 = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}, \quad \texttt{tol} = 10^{-3}$$

Expected output: $(x_1, x_2, x_3) \approx (2.0893, -1.5531, -0.6502)$ after 9 iterations.

---

### Topic 1.2 — Diagonal Dominance Checker

**Source:** `MidExam.ipynb`

**Mathematical definition:**

Matrix $A$ is strictly diagonally dominant if and only if

$$|a_{ii}| > \sum_{j \neq i} |a_{ij}| \quad \text{for all } i$$

**Interface:**

Function `DD_check(M)`:
- Parameter: `M` — square numpy array.
- Returns: nothing; prints whether matrix is diagonally dominant.

**Algorithm:**
1. Compute `diag = np.abs(np.diag(M))` — absolute diagonal entries.
2. Compute `off_diag = np.sum(np.abs(M), axis=1) - diag` — row sums of off-diagonal absolute values.
3. If `np.all(diag > off_diag)`: print "Matrix is diagonally dominant".
4. Else: print "Not diagonally dominant".

**Constraints and edge cases:**
- Strict inequality required; equality does not satisfy the condition.

**Test case:**

Input: The matrix $A$ from Topic 1.1. Expected output: "Matrix is diagonally dominant".

---

### Topic 1.3 — Midpoint Rule Integration

**Source:** `MidExam.ipynb`

**Mathematical definition:**

$$\int_a^b f(x)\,dx \approx h \sum_{i=0}^{n-1} f\!\left(\frac{x_i + x_{i+1}}{2}\right), \quad h = \frac{b-a}{n}$$

where $x_i = a + i\,h$ for $i = 0, 1, \ldots, n$.

**Interface:**

Method `Midpt(self)` inside an integration class (see note on class pattern):
- Uses `self.fn` (callable), `self.grid` (1-D array of $n+1$ equally-spaced points), `self.height` (scalar $h$).
- Returns: float scalar (`.item()`).

Standalone-function alternative: `midpoint_rule(f, a, b, n)` → float.

**Algorithm:**
1. Build grid $x_0, x_1, \ldots, x_n$ with $n+1$ points.
2. Compute midpoints $m_i = (x_i + x_{i+1})/2$ for $i = 0, \ldots, n-1$.
3. Evaluate $f(m_i)$ for all $i$.
4. Return $h \cdot \sum_i f(m_i)$.

**Constraints and edge cases:**
- $n$ can be any positive integer (odd or even).

**Test case:**

$f(x) = \sin(x)$, $[a, b] = [0, \pi]$, $n = 11$ → result $\approx 2.00825$, ARE $\approx 0.41\%$.

---

### Topic 1.4 — Trapezoid Rule Integration

**Source:** `MidExam.ipynb`, `homework/HW2.ipynb`

**Mathematical definition:**

$$\int_a^b f(x)\,dx \approx \frac{h}{2} \sum_{i=0}^{n-1} \bigl[f(x_i) + f(x_{i+1})\bigr], \quad h = \frac{b-a}{n}$$

Equivalently: $\displaystyle \frac{h}{2}\bigl[f(x_0) + 2f(x_1) + \cdots + 2f(x_{n-1}) + f(x_n)\bigr]$.

Error bound: $\displaystyle\left|\text{Error}\right| \leq \frac{K(b-a)^3}{12n^2}$ where $K = \max_{x \in [a,b]} |f''(x)|$.

Minimum $n$ to achieve error $< \varepsilon$: $n \geq \sqrt{\dfrac{K(b-a)^3}{12\varepsilon}}$.

**Interface:**

Method `Trapezoid(self)` or standalone `trapezoid_rule(f, a, b, n)` → float.

**Algorithm:**
1. Build grid of $n+1$ points.
2. Evaluate $f$ at all grid points.
3. Apply weights $[1, 2, 2, \ldots, 2, 1]$ multiplied by $h/2$.
4. Return sum.

**Constraints and edge cases:**
- $n$ can be any positive integer.
- For the error-bound variant: estimate $K$ via central finite differences $f''(x) \approx \frac{f(x+h)-2f(x)+f(x-h)}{h^2}$, then solve for minimum $n$.

**Test case:**

$f(x) = \sin(x)$, $[0, \pi]$, $n = 11$ → result $\approx 1.98352$, ARE $\approx 0.83\%$.

HW2 variant: $f(x) = \sin(x)/x$ (with $f(0)=1$), $[0, 1]$, required error $< 0.001$ → $n_{\min} = 6$, result $\approx 0.94539$, exact $\approx 0.94608$, error $\approx 0.00070 < 0.001$.

---

### Topic 1.5 — Simpson's Rule Integration

**Source:** `homework/HW2.ipynb`

**Mathematical definition:**

$$\int_a^b f(x)\,dx \approx \frac{h}{3}\bigl[f(x_0) + 4f(x_1) + 2f(x_2) + 4f(x_3) + \cdots + 4f(x_{n-1}) + f(x_n)\bigr]$$

where $h = (b-a)/n$ and $n$ must be **even**. Coefficient pattern: $1, 4, 2, 4, 2, \ldots, 4, 1$.

**Interface:**

Function `simpsons_rule(f_values, h)` or `simpsons_rule(f, a, b, n)` → float.

Internal: build coefficient array `coeffs` of length $n+1$ with `coeffs[0] = coeffs[n] = 1`, odd-index entries $= 4$, even interior entries $= 2$.

**Algorithm:**
1. Verify $n$ is even; raise `ValueError` if not.
2. Build coefficient array: all ones, then set odd indices to 4, even interior indices to 2.
3. Evaluate $f$ at $n+1$ grid points.
4. Return $(h/3) \cdot \texttt{np.dot}(\texttt{coeffs},\, f\_values)$.

**Constraints and edge cases:**
- $n$ must be even — this is a hard requirement; raise an error if violated.

**Test case (HW2):**

Swimming pool cross-section: diameters (ft) $= [0, 10, 12, 10, 8, 6, 8, 10, 0]$ at positions $x = 0, 2, 4, \ldots, 16$ ft. Cross-sectional area $A(x) = \pi d^2 / 8$ (half-circle). $h = 2$, $n = 8$ (even).

Expected: $V \approx 494.28$ ft³.

---

### Topic 1.6 — Absolute Relative Error (ARE)

**Source:** `MidExam.ipynb`

**Mathematical definition:**

$$\text{ARE} = \left|\frac{x_{\text{true}} - x_{\text{estimate}}}{x_{\text{estimate}}}\right| \times 100\%$$

where `tval` is the true/reference value and `est` is the estimate.

**Interface:**

Static method `ARE(est, tval)` → float (percentage).

**Algorithm:**
1. Compute $|\texttt{tval} - \texttt{est}| / |\texttt{est}| \times 100$.
2. Return result.

**Constraints and edge cases:**
- `est` must be non-zero.
- Some notebooks compare estimate to true analytic value; others compare successive iterates.

**Test case:**

Midpoint rule for $\int_0^\pi \sin x\,dx$ with $n=11$: `est = 2.0082484`, `tval = 2.0` → ARE $\approx 0.4107\%$.

---

### Topic 1.7 — Monte Carlo Integration

**Source:** `MidExam.ipynb`

**Mathematical definition:**

$$\int_a^b f(x)\,dx \approx (b - a) \cdot \frac{1}{N} \sum_{i=1}^{N} f(U_i), \quad U_i \sim \text{Uniform}(a, b)$$

The notebook computes $\int_0^\pi \sin(x)\,dx$ and multiplies by $\pi$ because $b - a = \pi$.

**Interface:**

Static method `MCMC(f, area, runs)`:
- `f` — callable, the integrand.
- `area` — list `[a, b]`.
- `runs` — int, number of random samples $N$.
- Returns: float scalar.

**Algorithm:**
1. Draw $N$ uniform samples $U_i \in [a, b]$ via `np.random.uniform(a, b, runs)`.
2. Evaluate $f$ at all samples.
3. Return $(b - a) \cdot \text{mean}(f(U_i))$.

**Note for Codex:** The notebook multiplies by `np.pi` rather than `(b-a)` due to the specific integral used. Implement the general formula $(b-a) \cdot \text{mean}$ which recovers the notebook result since $b - a = \pi$.

**Constraints and edge cases:**
- Result is stochastic; verify with many runs (e.g., $N = 100\,000$).

**Test cases:**

$f(x) = \sin(x)$, $[0, \pi]$: $N=100 \to \approx 2.03$; $N=10000 \to \approx 2.00$; $N=100000 \to \approx 2.00$.

---

### Topic 1.8 — Bisection Method

**Source:** NOT FOUND in scanned notebooks. Implement from standard definition.

**Mathematical definition:**

Given $f$ continuous on $[a, b]$ with $f(a)\,f(b) < 0$, the bisection method produces:

$$c_k = \frac{a_k + b_k}{2}$$

Update: if $f(a_k)\,f(c_k) < 0$ set $b_{k+1} = c_k$, else set $a_{k+1} = c_k$.

Stop when $|b_k - a_k| < \texttt{tol}$ or ARE between successive midpoints $< \texttt{tol}$.

**Interface:**

Function `bisection(f, a, b, tol=1e-6, max_iter=1000)` → float (root approximation).

**Algorithm:**
1. Verify $f(a) \cdot f(b) < 0$; raise `ValueError` if not.
2. Repeat:
   a. $c = (a + b) / 2$.
   b. If $|f(c)| < \texttt{tol}$ or interval $< \texttt{tol}$: return $c$.
   c. If $f(a) \cdot f(c) < 0$: $b \leftarrow c$, else $a \leftarrow c$.
3. Return $c$.

**Constraints and edge cases:**
- Requires bracket: $f(a)$ and $f(b)$ must have opposite signs.
- Always converges (unlike Newton/secant).

**Test case:** $f(x) = x - \cos(x)$, $[0, \pi]$ → root $\approx 0.7391$.

---

### Topic 1.9 — Newton's Method

**Source:** NOT FOUND in scanned notebooks. Implement from standard definition.

**Mathematical definition:**

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

**Interface:**

Function `newton(f, df, x0, tol=1e-6, max_iter=100)` → float.

- `f` — callable, the function.
- `df` — callable, the derivative.

**Algorithm:**
1. Start with $x \leftarrow x_0$.
2. Repeat up to `max_iter` times:
   a. Compute $x_{\text{new}} = x - f(x)/f'(x)$.
   b. If $|x_{\text{new}} - x| < \texttt{tol}$: return $x_{\text{new}}$.
   c. $x \leftarrow x_{\text{new}}$.
3. Raise `RuntimeError` if not converged.

**Constraints and edge cases:**
- $f'(x) \neq 0$ at each iterate; raise `ZeroDivisionError` if violated.
- May diverge if $x_0$ is far from the root.

**Test case:** $f(x) = x - \cos(x)$, $f'(x) = 1 + \sin(x)$, $x_0 = 0.5$ → root $\approx 0.7391$.

---

### Topic 1.10 — Secant Method

**Source:** NOT FOUND in scanned notebooks. Implement from standard definition.

**Mathematical definition:**

$$x_{n+1} = x_n - f(x_n) \cdot \frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}$$

**Interface:**

Function `secant(f, x0, x1, tol=1e-6, max_iter=100)` → float.

**Algorithm:**
1. Start with $x_{\text{prev}} = x_0$, $x_{\text{curr}} = x_1$.
2. Repeat:
   a. $x_{\text{new}} = x_{\text{curr}} - f(x_{\text{curr}}) \cdot (x_{\text{curr}} - x_{\text{prev}}) / (f(x_{\text{curr}}) - f(x_{\text{prev}}))$.
   b. If $|x_{\text{new}} - x_{\text{curr}}| < \texttt{tol}$: return $x_{\text{new}}$.
   c. $x_{\text{prev}} \leftarrow x_{\text{curr}}$, $x_{\text{curr}} \leftarrow x_{\text{new}}$.
3. Raise `RuntimeError` if not converged.

**Constraints and edge cases:**
- Denominator $f(x_n) - f(x_{n-1})$ must be non-zero.

**Test case:** $f(x) = x - \cos(x)$, $x_0 = 0.5$, $x_1 = 1$ → root $\approx 0.7391$.

---

### Topic 1.11 — Fixed-Point Iteration

**Source:** NOT FOUND in scanned notebooks. Implement from standard definition.

**Mathematical definition:**

Given $g$ such that $x^* = g(x^*)$:

$$x_{n+1} = g(x_n)$$

Convergence guaranteed when $|g'(x)| < 1$ near $x^*$.

**Interface:**

Function `fixed_point(g, x0, tol=1e-6, max_iter=100)` → float.

**Algorithm:**
1. $x \leftarrow x_0$.
2. Repeat:
   a. $x_{\text{new}} = g(x)$.
   b. If $|x_{\text{new}} - x| < \texttt{tol}$: return $x_{\text{new}}$.
   c. $x \leftarrow x_{\text{new}}$.
3. Raise `RuntimeError` if not converged.

**Test case:** $g(x) = \cos(x)$, $x_0 = 0.5$ → fixed point $\approx 0.7391$.

---

### Topic 1.12 — Gradient Descent

**Source:** `MidExam.ipynb`

**Mathematical definition:**

Given differentiable $g: \mathbb{R}^n \to \mathbb{R}$:

$$\theta^{(k+1)} = \theta^{(k)} - \alpha\, \nabla g\!\left(\theta^{(k)}\right)$$

The notebook's specific objective is

$$g(\theta_1, \theta_2) = \frac{9}{16}(\theta_1 - 2)^2 + (\theta_2 - 2)^2 + \frac{\theta_1 \theta_2}{4}$$

with gradient:

$$\frac{\partial g}{\partial \theta_1} = \frac{9}{8}(\theta_1 - 2) + \frac{\theta_2}{4}, \qquad \frac{\partial g}{\partial \theta_2} = 2(\theta_2 - 2) + \frac{\theta_1}{4}$$

**Interface:**

Class with:
- `__init__(self, pt, maxN)` — stores initial point `pt` (list of 2 floats) and max iterations `maxN`.
- `GradD(self, alpha)` — runs gradient descent, returns array of shape `(maxN+1, 2)` containing all iterates.
- Static method `grad_f(dx, dy)` → numpy array of shape `(2,)`.

**Algorithm:**
1. Initialise `vec = np.array(self.pt)`.
2. Store `vec` in row 0 of results array.
3. For $k = 1, \ldots, \texttt{maxN}$:
   a. $\texttt{vec} \leftarrow \texttt{vec} - \alpha \cdot \nabla g(\texttt{vec})$.
   b. Store updated `vec` in row $k$.
4. Return full results array.

**Test case:** $\theta^{(0)} = (5, 4)$, $\alpha = 0.1$, 10 iterations:

- Iter 1: $(4.5625, 3.4750)$
- Iter 10: $(2.5352, 1.8582)$

---

### Topic 1.13 — Gradient Descent with Momentum

**Source:** `MidExam.ipynb`

**Mathematical definition:**

$$v^{(k+1)} = 0.9\,v^{(k)} + \nabla g\!\left(\theta^{(k)}\right)$$
$$\theta^{(k+1)} = \theta^{(k)} - \alpha\, v^{(k+1)}$$

Momentum coefficient is fixed at $\beta = 0.9$ in the notebooks.

**Interface:**

Method `Momentum(self, alpha)` on the same class as `GradD`:
- Same return type: array of shape `(maxN+1, 2)`.

**Algorithm:**
1. Initialise `vec = np.array(self.pt)`, `wgt = np.zeros(2)`.
2. Store `vec` in row 0.
3. For $k = 1, \ldots, \texttt{maxN}$:
   a. $\texttt{wgt} \leftarrow 0.9 \cdot \texttt{wgt} + \nabla g(\texttt{vec})$.
   b. $\texttt{vec} \leftarrow \texttt{vec} - \alpha \cdot \texttt{wgt}$.
   c. Store `vec` in row $k$.
4. Return results array.

**Test case:** $\theta^{(0)} = (5, 4)$, $\alpha = 0.5$, 10 iterations:

- Iter 1: $(2.8125, 1.3750)$
- Iter 10: $(1.6779, 2.7331)$

---

### Topic 1.14 — Caesar / Shift Cipher

**Source:** `MidExam.ipynb`

**Mathematical definition:**

Encryption: $E(x) = (x + k) \bmod 26$, where $x$ is the letter index (A=0).

Decryption: $D(x) = (x - k) \bmod 26$.

**Interface:**

Static method `Caeser_Cipher(plain_text, shift)` → str.

- `plain_text` — str (may contain spaces and mixed case).
- `shift` — int or single letter str (if str, take its 0-based position as the shift).
- Non-alphabetic characters are passed through unchanged.
- The notebook implements Caesar as a special case of Vigenère with a single-character keyword.

**Algorithm:**
1. If `shift` is a string, convert to uppercase and use its ordinal position $(0\text{–}25)$ as $k$.
2. For each character in `plain_text`:
   - If alphabetic: compute shifted index, preserve case.
   - Else: copy unchanged.
3. Return encrypted string.

**Test case:**

Input: `'HELLO EVERYONE'`, shift $= 6$ → `'NKRRU KBKXEUTK'`.

---

### Topic 1.15 — Vigenère Cipher

**Source:** `MidExam.ipynb`

**Mathematical definition:**

For the $i$-th alphabetic character (0-indexed among letters only):

$$E_i(x) = \bigl(x + k_{i \bmod m}\bigr) \bmod 26$$

where $k_j$ is the $j$-th letter's index of the keyword and $m$ is the keyword length.

**Interface:**

Static method `Vigenere_Cipher(plain_text, keyword)` → str.

Helper static method `encode_each(alpbet, key)` → single char.

- Non-alphabetic characters skipped (not counted toward keyword position).
- Case of ciphertext matches case of plaintext character.

**Algorithm:**
1. Initialise counter = 0, result = "".
2. For each character `T` in `plain_text`:
   - If alphabetic:
     - $K = \texttt{keyword}[\texttt{count} \bmod \texttt{len(keyword)}]$.
     - Compute $T_{\text{idx}} = \text{ord}(T) - \text{base}$, $K_{\text{idx}} = \text{ord}(K) - \text{ord}('A')$ (or `'a'`).
     - $E = (T_{\text{idx}} + K_{\text{idx}}) \bmod 26$.
     - Append $\texttt{chr}(\text{base} + E)$; increment counter.
   - Else: append character unchanged.
3. Return result.

**Test case:**

Input: `'attack at dawn'`, keyword: `'LEMON'` → `'lxfopv ef rnhr'`.

---

### Topic 1.16 — Affine Cipher

**Source:** NOT FOUND in scanned notebooks. Implement from standard definition.

**Mathematical definition:**

Encryption: $E(x) = (ax + b) \bmod 26$

Decryption: $D(y) = a^{-1}(y - b) \bmod 26$

where $a^{-1}$ is the modular inverse of $a$ modulo 26.

**Interface:**

Class `AffineCipher`:
- `__init__(self, a, b)` — validates $\gcd(a, 26) = 1$; raises `ValueError` otherwise.
- `encrypt(self, plain_text)` → str.
- `decrypt(self, cipher_text)` → str.

**Algorithm:**
1. Validate: $\gcd(a, 26) = 1$; valid values of $a$: $\{1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25\}$.
2. Compute $a^{-1} \bmod 26$ via extended Euclidean algorithm or `pow(a, -1, 26)`.
3. Apply formula character-by-character; skip non-alphabetic.

**Constraints and edge cases:**
- $\gcd(a, 26) \neq 1$ is invalid — decryption is impossible.
- Preserve case; map uppercase to base 65, lowercase to base 97.

**Test case:**

$a = 5$, $b = 8$: encrypt `'AFFINE'` → `'IHHWVC'`; decrypt `'IHHWVC'` → `'AFFINE'`.

---

### Topic 1.17 — Substitution Cipher

**Source:** NOT FOUND in scanned notebooks. Implement from standard definition.

**Mathematical definition:**

A bijection $\sigma: \{A, \ldots, Z\} \to \{A, \ldots, Z\}$ (an arbitrary permutation of the 26 letters).

Encryption: replace each letter $x$ with $\sigma(x)$.
Decryption: replace each letter $y$ with $\sigma^{-1}(y)$.

**Interface:**

Class `SubstitutionCipher`:
- `__init__(self, key)` — `key` is a 26-character string giving the ciphertext alphabet; validates it is a permutation of A–Z.
- `encrypt(self, plain_text)` → str.
- `decrypt(self, cipher_text)` → str.

**Algorithm:**
1. Build forward map: `plain_alphabet[i] → key[i]` for $i = 0, \ldots, 25$.
2. Build inverse map for decryption.
3. Apply map character-by-character; preserve case; pass non-alphabetic unchanged.

**Constraints and edge cases:**
- `key` must be exactly 26 distinct letters (case-insensitive).

**Test case:**

Key: `'QWERTYUIOPASDFGHJKLZXCVBNM'` — encrypt `'HELLO'` → `'ITSSG'`.

---

### Topic 1.18 — Hill Cipher

**Source:** NOT FOUND in scanned notebooks. Implement from standard definition.

**Mathematical definition:**

Group plaintext into blocks of length $m$. Represent each block as column vector $\mathbf{p} \in \mathbb{Z}_{26}^m$.

Encryption: $\mathbf{c} = K\,\mathbf{p} \bmod 26$

Decryption: $\mathbf{p} = K^{-1}\,\mathbf{c} \bmod 26$

where $K$ is an $m \times m$ integer matrix with $\det(K)$ coprime to 26.

**Interface:**

Class `HillCipher`:
- `__init__(self, key_matrix)` — `key_matrix` is a square numpy integer array; validates $\gcd(\det(K) \bmod 26,\, 26) = 1$.
- `encrypt(self, plain_text)` → str (uppercase, padded with 'X' if needed).
- `decrypt(self, cipher_text)` → str.

**Algorithm:**
1. Convert plaintext to uppercase, strip non-alphabetic, pad to multiple of $m$ with 'X'.
2. Map letters to integers 0–25.
3. For each block: compute $K\,\mathbf{p} \bmod 26$; convert back to letters.
4. For decryption: compute $K^{-1} \bmod 26$ (modular matrix inverse) first.

**Constraints and edge cases:**
- $\det(K) \bmod 26$ must be coprime to 26.
- Modular matrix inverse is not `np.linalg.inv(K)` — must use integer arithmetic mod 26.

**Test case ($m=2$):**

$K = \begin{bmatrix}3 & 3 \\ 2 & 5\end{bmatrix}$: encrypt `'HELP'` → `'HIAT'`.

---

### Topic 1.19 — Brute-Force Caesar Cracker

**Source:** `MidExam.ipynb`

**Mathematical definition:**

Try all 26 possible shift keys $k \in \{0, 1, \ldots, 25\}$; for each, decrypt using $D(x) = (x - k) \bmod 26$ and print the candidate plaintext.

**Interface:**

Static method `Crack_Text(plain_txt)` → None (prints all 26 candidates).

**Algorithm:**
1. For $k = 0, 1, \ldots, 25$:
   a. Apply Caesar decryption with shift $k$ (equivalently, Caesar encryption with shift $26 - k$).
   b. Print $k$ and the candidate plaintext.

**Constraints and edge cases:**
- Human must inspect output to identify the correct plaintext.

**Test case:**

Input: `'exxego ex srgi'` — output at $k=4$: `'attack at once'`.

---

### Topic 1.20 — Spearman's Rho

**Source:** `MidExam.ipynb`

**Mathematical definition:**

$$\rho = 1 - \frac{6 \sum_{i=1}^n d_i^2}{n(n^2 - 1)}$$

where $d_i = r_{X_i} - r_{Y_i}$ is the difference in ranks.

Test statistic (t-distributed with $n-2$ degrees of freedom):

$$t = \rho \sqrt{\frac{n-2}{1 - \rho^2}}$$

Two-tailed p-value: $p = 2\bigl(1 - F_{t,\,n-2}(|t|)\bigr)$ where $F_{t,n-2}$ is the CDF of $t_{n-2}$.

**Interface:**

Method `rho(self)` → tuple `(coef: float, pval: float)`.

- Uses `self.X` and `self.Y` (1-D numpy arrays of equal length $n$).
- Ranks computed via double `argsort`: `rank = arr.argsort().argsort()` (0-indexed).

**Algorithm:**
1. Compute rank arrays $r_X$ and $r_Y$ using double argsort.
2. Compute $d_i = r_{X_i} - r_{Y_i}$.
3. Compute $\rho = 1 - 6\sum d_i^2 / [n(n^2-1)]$.
4. Compute test statistic $t = \rho\sqrt{(n-2)/(1-\rho^2)}$.
5. Compute p-value $= 2(1 - \texttt{scs.t.cdf}(|t|,\, n-2))$.
6. Return $(\rho,\, p)$.

**Test case:**

$X = [106, 100, 86, 101, 99, 103, 97, 113, 112, 110]$,
$Y = [7, 27, 2, 50, 28, 29, 20, 12, 6, 17]$, $n = 10$:

Expected: $\rho \approx -0.1758$, p-value $\approx 0.6272$.

---

### Topic 1.21 — Kendall's Tau

**Source:** `MidExam.ipynb`

**Mathematical definition:**

For all pairs $(i, j)$ with $i < j$:

$$\tau = \frac{2\displaystyle\sum_{i<j} \text{sgn}(\Delta X_{ij})\,\text{sgn}(\Delta Y_{ij})}{n(n-1)}$$

where $\Delta X_{ij} = X_i - X_j$.

Variance under $H_0$: $\text{Var}(\tau) = \dfrac{2(2n+5)}{9n(n-1)}$

Test statistic: $z = \dfrac{\tau}{\sqrt{\text{Var}(\tau)}} = \tau\sqrt{\dfrac{9n(n-1)}{2(2n+5)}}$

Two-tailed p-value: $p = 2(1 - \Phi(|z|))$.

**Interface:**

Method `tau(self)` → tuple `(coef: float, pval: float)`.

**Algorithm:**
1. For all pairs $(i, j)$ with $i < j$ (use `itertools.combinations(range(n), 2)`):
   a. Compute $s_X = \text{sign}(X_i - X_j)$, $s_Y = \text{sign}(Y_i - Y_j)$.
   b. Accumulate $\text{cSum} \mathrel{+}= s_X \cdot s_Y$.
2. $\tau = 2\,\text{cSum} / [n(n-1)]$.
3. $z = \tau\sqrt{9n(n-1)/[2(2n+5)]}$.
4. $p = 2(1 - \texttt{scs.norm.cdf}(|z|))$.
5. Return $(\tau, p)$.

**Test case:**

Same data as Spearman (Topic 1.20): $\tau \approx -0.1111$, p-value $\approx 0.6547$.

---

### Topic 1.22 — Rejection Sampling

**Source:** `homework/HW2.ipynb`

**Mathematical definition:**

To sample from target density $f(x)$: choose envelope $g(x)$ with constant $M$ such that $M\,g(x) \geq f(x)$ for all $x$.

**Algorithm:**
1. Draw $x$ from $g$.
2. Draw $u \sim \text{Uniform}(0, 1)$.
3. Accept $x$ if $u \leq f(x) / [M\,g(x)]$; otherwise reject.

Theoretical acceptance rate: $1/M$.

**HW2 specification — two candidates for $f(x) = \frac{\pi}{2}\sin(\pi x)$, $x \in [0,1]$:**

**Candidate 1:** $g(x) = 1$ (Uniform), $M_1 = \pi/2 \approx 1.5708$, acceptance rate $= 2/\pi \approx 63.7\%$.

**Candidate 2:** Triangular density

$$g_2(x) = \begin{cases} 4x & x \leq 0.5 \\ 4(1-x) & x > 0.5 \end{cases}$$

$M_2 = \pi^2/8 \approx 1.2337$, acceptance rate $= 8/\pi^2 \approx 81.1\%$.

Sampling from $g_2$: average of two $\text{Uniform}(0,1)$ draws.

**Interface:**

Functions `sample_candidate1(n_samples)` and `sample_candidate2(n_samples)` → tuple `(accepted_array, empirical_acceptance_rate)`.

**Constraints and edge cases:**
- Loop until `n_samples` accepted values are collected.
- Track `total_trials` to compute empirical acceptance rate.

**Test case:** $N = 1000$: empirical acceptance rate near $63.7\%$ (C1) and $81.1\%$ (C2).

---

### Topic 1.23 — Inverse Transform Method

**Source:** NOT FOUND in scanned notebooks. Implement from standard definition.

**Mathematical definition:**

If $F$ is the CDF of the target distribution, then $X = F^{-1}(U)$ where $U \sim \text{Uniform}(0,1)$ has distribution $F$.

**Interface:**

Function `inverse_transform(F_inv, n)` → numpy array of $n$ samples.

- `F_inv` — callable, the quantile function (inverse CDF).

**Algorithm:**
1. Draw $n$ samples $U_i \sim \text{Uniform}(0, 1)$.
2. Return $F^{-1}(U_i)$ for all $i$.

**Test case:** Exponential with $\lambda = 1$: $F^{-1}(u) = -\ln(1-u)$. Sample mean $\approx 1.0$ for large $n$.

---

### Topic 1.24 — Box-Muller Transform

**Source:** NOT FOUND in scanned notebooks. Implement from standard definition.

**Mathematical definition:**

Given $U_1, U_2 \sim \text{Uniform}(0,1)$ independent:

$$Z_1 = \sqrt{-2\ln U_1}\,\cos(2\pi U_2), \qquad Z_2 = \sqrt{-2\ln U_1}\,\sin(2\pi U_2)$$

$Z_1, Z_2 \sim \mathcal{N}(0, 1)$ independent.

**Interface:**

Function `box_muller(n)` → numpy array of $2n$ standard-normal samples.

**Algorithm:**
1. Draw $n$ pairs $(U_1, U_2)$.
2. Compute $Z_1, Z_2$ per the formulas above.
3. Return concatenated array of length $2n$.

**Test case:** $n = 10000$: sample mean $\approx 0$, sample std $\approx 1$.

---

### Topic 1.25 — Linear Congruential Generator (LCG)

**Source:** NOT FOUND in scanned notebooks. Implement from standard definition.

**Mathematical definition:**

$$X_{n+1} = (a\,X_n + c) \bmod m$$

Parameters: multiplier $a$, increment $c$, modulus $m$, seed $X_0$.

Uniform pseudo-random numbers: $U_n = X_n / m$.

**Interface:**

Class `LCG`:
- `__init__(self, a, c, m, seed)`.
- `next(self)` → float $\in [0, 1)$.
- `generate(self, n)` → list of $n$ floats.

**Algorithm:**
1. `next()`: compute $X \leftarrow (a \cdot X + c) \bmod m$; return $X/m$.
2. `generate(n)`: call `next()` $n$ times.

**Constraints and edge cases:**
- Full period requires Hull-Dobell conditions: $\gcd(c, m) = 1$; $a \equiv 1 \pmod{p}$ for each prime $p | m$; if $4 | m$ then $4 | (a-1)$.

**Test case:** $a = 1664525$, $c = 1013904223$, $m = 2^{32}$, seed $= 42$: first value $\approx 0.2500$ (verify bit pattern matches known LCG sequences).

---

### Topic 1.26 — Monte Carlo Estimation of $\pi$

**Source:** NOT FOUND in scanned notebooks. Implement from standard definition.

**Mathematical definition:**

Sample $(x, y) \sim \text{Uniform}([0,1]^2)$. Count hits inside unit quarter-circle: $x^2 + y^2 \leq 1$. Then:

$$\pi \approx 4 \cdot \frac{\text{hits}}{N}$$

**Interface:**

Function `monte_carlo_pi(n)` → float.

**Algorithm:**
1. Draw $n$ pairs $(x_i, y_i) \sim \text{Uniform}(0, 1)$.
2. Count $h = \#\{i : x_i^2 + y_i^2 \leq 1\}$.
3. Return $4h/n$.

**Test case:** $n = 10^6$ → result within $0.01$ of $\pi \approx 3.14159$.

---

### Topic 1.27 — Encapsulation with Private Attributes and Property Decorators

**Source:** `Sec 3 - Function, Class.ipynb`

**Mathematical definition:** N/A — software pattern.

**Interface:**

Class pattern:
- Single-underscore `_attr` — convention for "protected" (not enforced by Python).
- Double-underscore `__attr` — name-mangled to `_ClassName__attr` (harder to access externally).
- `@property` — defines getter (read-only access via attribute syntax).
- `@attr.setter` — defines setter (validates before assignment).

**Key example classes:**
1. `BankAccount(owner, balance, password)`: `__balance` is double-underscore private; exposed via `get_balance()` method.
2. `Temperature(celsius)`: `celsius` is a property with setter that raises `ValueError` if below $-273.15$; `fahrenheit` is a read-only derived property.
3. `car(name)`: `brand` property with getter/setter; internal storage in `_brand`.

**Algorithm:**
- `__init__` receives raw value and assigns through the property setter (so validation runs at construction).
- Getter returns `self._attr`.
- Setter validates input and stores in `self._attr`.
- Derived property computes from stored attribute without a setter.

**Test cases:**
- `Temperature(100).fahrenheit` → `212.0`.
- `Temperature(-300)` → raises `ValueError`.
- `BankAccount("A", 1000, 0).__balance` → `AttributeError`.

---

### Topic 1.28 — Single Inheritance with Method Override

**Source:** `Sec 3 - Function, Class.ipynb`

**Interface:**

- Parent class defines a method (e.g., `make_sound`) returning a default string.
- Child class overrides the same method to return a different string.
- `isinstance(child_instance, ParentClass)` returns `True`.

**Example hierarchy:**

`Pet` → `Dog`, `Cat`, `Snake` — all override `make_sound()`.

**Test case:**

`Pet().make_sound()` → `"Hello"`. `Dog().make_sound()` → `"Woof!"`. `isinstance(Dog(), Pet)` → `True`.

---

### Topic 1.29 — Multiple Inheritance and MRO

**Source:** `Sec 3 - Function, Class.ipynb`

**Interface:**

Class `D(B, C)` where `B(A)` and `C(A)` — diamond pattern.

Python uses C3 linearisation for MRO.

**MRO for diamond `D(B, C)`, `B(A)`, `C(A)`:**

`[D, B, C, A, object]`

Method `who()` on `D()` returns the result from the first class in MRO that defines it — `B.who()`.

**Test case:**

`ExamD().who()` → `"B"`. `[cls.__name__ for cls in ExamD.__mro__]` → `['ExamD', 'ExamB', 'ExamC', 'ExamA', 'object']`.

---

### Topic 1.30 — Multilevel Inheritance with `super().__init__`

**Source:** `Sec 3 - Function, Class.ipynb`

**Interface:**

Chain: `Animal` → `Dog(Animal)` → `Coonhound(Dog)`.

Each level adds methods; `super().__init__` passes initialisation up the chain.

**Example:**

`WorkingDog(Animal)` with `__init__(self, name, job)`: calls `super().__init__(name)` to initialise `self.name`, then sets `self.job = job`.

**Test case:**

`ExamWorkingDog("Lucky", "guide").name` → `"Lucky"`. `ExamWorkingDog("Lucky", "guide").job` → `"guide"`.

---

### Topic 1.31 — Polymorphism

**Source:** `Sec 3 - Function, Class.ipynb`

**Interface:**

Duck typing: multiple classes implement the same method name; calling code iterates over heterogeneous objects and calls the method.

**Test case:**

```
for obj in [ExamBell(), ExamPhone()]:
    print(obj.make_sound())
```
Output: `"ding"`, then `"ring"`.

---

### Topic 1.32 — `*args` and `**kwargs`

**Source:** `Sec 3 - Function, Class.ipynb`

**Interface:**

- `*args` — collects extra positional arguments into a tuple.
- `**kwargs` — collects extra keyword arguments into a dict.
- Order in signature: `(fixed_pos, *args, **kwargs)`.

**Key constraints:**
- General positional parameters must come before `*args`.
- Inside the function, `args` is a `tuple`; `kwargs` is a `dict`.
- Calling with `**some_dict` unpacks the dict as keyword arguments.
- Python 3.9+: `dict1 | dict2` merges dicts into a new dict (does not mutate either).

**Test cases:**

`running_product(2, 3, 4, 5)` → `[2.0, 6.0, 24.0, 120.0]`.
`sentence(subject='You', object='me', verb='beat')` → `'You beat me'`.

---

### Topic 1.33 — Mutable Default Argument (Correct Pattern)

**Source:** `Sec 3 - Function, Class.ipynb`

**Interface:**

**Wrong:** `def f(value, bucket=[])` — list created once at `def` time; shared across all calls.

**Correct:** `def f(value, bucket=None)` — test `if bucket is None: bucket = []` inside body.

**Test cases:**

Wrong version: `f(1)` → `[1]`; `f(2)` → `[1, 2]` (contaminated).
Correct version: `f(1)` → `[1]`; `f(2)` → `[2]` (fresh each time).

---

### Topic 1.34 — Lambda and Higher-Order Functions

**Source:** `Sec 3 - Function, Class.ipynb`

**Interface:**

- `lambda` creates anonymous callable: `lambda x: expression`.
- `sorted(iterable, key=lambda x: ...)` — sort by derived key.
- `map(fn, iterable)` — lazy apply.
- `filter(fn, iterable)` — lazy filter.
- `functools.reduce(fn, iterable)` — left fold.
- Higher-order functions accept callables as parameters.

**Test cases:**

`sorted(["pear", "apple", "fig"], key=len)` → `['fig', 'pear', 'apple']`.
`vec_op(lambda *args: sum(args), 12.87, 3.12)` → `15.99`.

---

### Topic 1.35 — Recursion and Memoization (Fibonacci)

**Source:** `Sec 3 - Function, Class.ipynb`

**Mathematical definition:**

$$F(n) = F(n-1) + F(n-2), \quad F(0) = 0,\; F(1) = 1$$

Binet's (closed-form) formula:

$$F(n) = \frac{1}{\sqrt{5}}\left[\left(\frac{1+\sqrt{5}}{2}\right)^n - \left(\frac{1-\sqrt{5}}{2}\right)^n\right]$$

**Four implementations required (all verified against same test):**

1. **Naïve recursive** — exponential time; no memoisation.
2. **Binet's formula** — $O(1)$ but loses precision for large $n$ (floats); use `int(...)`.
3. **Memoised with `@functools.lru_cache(maxsize=None)`** — $O(n)$ time.
4. **Iterative** — $O(n)$ time, $O(1)$ space; simultaneous assignment `a, b = b, a + b`.

**Test case:** `[F(k) for k in range(15)]` → `[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]`.

---

### Topic 1.36 — Type Hints and Docstrings

**Source:** `Sec 3 - Function, Class.ipynb`

**Interface:**

- Parameters annotated as `name: type`.
- Return annotated as `-> type`.
- Union: `int | str` (Python 3.10+) or `Union[int, str]`.
- Optional: `str | None` or `Optional[str]`.
- Docstring: one-line summary in triple quotes immediately after `def` line.
- Type hints are **not enforced at runtime** — use `pydantic` or `@validate_call` for enforcement.

**Test case:**

`iprod([1, 3, 4], [1, 2, -3])` → `-5` (inner product function with type hints `list[int] -> float`).

---

### Topic 1.37 — QM / AM / GM / HM Means

**Source:** `Sec 3 - Function, Class.ipynb`

**Mathematical definition:**

For $n$ positive reals $x_1, \ldots, x_n$:

$$\text{HM} \leq \text{GM} \leq \text{AM} \leq \text{QM}$$

$$\text{AM} = \frac{\sum_{i=1}^n x_i}{n}, \quad
\text{GM} = \left(\prod_{i=1}^n x_i\right)^{1/n}, \quad
\text{HM} = \frac{n}{\sum_{i=1}^n 1/x_i}, \quad
\text{QM} = \sqrt{\frac{\sum_{i=1}^n x_i^2}{n}}$$

**Interface:**

Four standalone functions accepting `*args` (variadic positional):
- `QM(*args)` → float — "Quadratic mean"
- `AM(*args)` → float — "Arithmatic mean" (note: notebook spells it this way)
- `GM(*args)` → float — "Geometric mean"
- `HM(*args)` → float — "Harmonic mean"

Also as static methods of class `Mean_Type`.

Helper: `vec_op(fn, *args)` → `round(fn(*args), ndigits=2)`.

**Constraints and edge cases:**
- All arguments must be positive (HM and GM undefined for zero/negative).

**Test cases (inputs: 12.87, 3.12):**

`QM(12.87, 3.12)` → `9.36`, `AM(...)` → `7.99`, `GM(...)` → `6.26`, `HM(...)` → `5.02`.

---

### Topic 1.38 — Compound Interest (Discrete and Continuous)

**Source:** `Sec 3 - Function, Class.ipynb`

**Mathematical definition:**

Discrete: $F = P\!\left(1 + \dfrac{r}{n}\right)^{nt}$

Continuous (limit as $n \to \infty$): $F = P\,e^{rt}$

Variables: $P$ = present value, $r$ = annual rate, $n$ = compounding periods per year, $t$ = years.

**Interface:**

Function `Final_Value(P, r, t, n)` → float:
- `n` is an integer (periods per year) **or** the string `'inf'` (continuous compounding).
- If `n == 'inf'`: return $P\,e^{rt}$.
- Else: return $P\,(1 + r/n)^{nt}$.

**Test cases ($P = 12000$, $r = 0.075$, $t = 3$):**

| $n$ | Result |
|-----|--------|
| 4 | 14950.549 |
| 12 | 15007.619 |
| 365 | 15028.539 |
| `'inf'` | 15028.809 |

---

### Topic 1.39 — Logistic Growth Model

**Source:** `Sec 3 - Function, Class.ipynb`

**Mathematical definition:**

$$f(t) = \frac{M}{1 + e^{-rt}}$$

Properties: $f(0) = M/2$; $\lim_{t\to+\infty} f(t) = M$; $r$ controls steepness.

**Interface:**

Function `logit_grow(M, r, t)` → float:
- `M` — maximum capacity.
- `r` — growth rate.
- `t` — time.

**Test case:** `logit_grow(100, 1, 0)` → `50.0`.

---

### Topic 1.40 — NumPy: Triangular Matrix Extraction

**Source:** `Sec 4.1 - Package Intro - numpy.ipynb`

**Interface:**

- `np.tril(A)` — returns lower triangle including main diagonal; upper set to 0.
- `np.triu(A)` — returns upper triangle including main diagonal; lower set to 0.
- `A - np.tril(A)` — strict upper triangle (no diagonal).
- `A - np.triu(A)` — strict lower triangle (no diagonal).
- Use case in Gauss-Seidel: `L = np.tril(A)`, `U = A - L`.

**Test case:**

For $A = \begin{bmatrix}1&2&3\\4&5&6\\7&8&9\end{bmatrix}$:
$\texttt{np.tril}(A)_{ij} = A_{ij}$ if $i \geq j$ else 0.

---

### Topic 1.41 — Array Ranking via Double `argsort`

**Source:** `MidExam.ipynb` (used in Spearman's Rho)

**Mathematical definition:**

For array $a$, `a.argsort()` gives indices that sort $a$ ascending. A second `argsort()` converts those indices into ranks (0-indexed).

$$\texttt{rank}[i] = \text{position of } a[i] \text{ in the sorted order}$$

**Interface:**

Pattern: `rank = arr.argsort().argsort()` → integer array of same shape as `arr`.

**Test case:**

`arr = [3, 1, 4, 1, 5]` → ranks `[2, 0, 3, 1, 4]` (note: ties broken by first occurrence order).

---

### Topic 1.42 — NumPy: Broadcasting

**Source:** `Sec 4.1 - Package Intro - numpy.ipynb`

**Mathematical definition:**

NumPy broadcasts arrays with compatible shapes (trailing dimensions match or are 1). No data is copied; virtual expansion is performed.

**Key patterns:**
- Subtract row means: `data - data.mean(axis=1, keepdims=True)` — shape `(m, n) - (m, 1)` → `(m, n)`.
- Outer product: `v[:, np.newaxis] * v[np.newaxis, :]` — shape `(n, 1) * (1, n)` → `(n, n)`.
- Pairwise distances: `points[:, np.newaxis, :] - points[np.newaxis, :, :]` — shape `(n, 1, d) - (1, n, d)` → `(n, n, d)`.

**Constraints and edge cases:**
- Logical operators must be `&`, `|`, `~` (not `and`, `or`, `not`).
- Broadcasting does not copy memory.

---

### Topic 1.43 — NumPy: View vs Copy Semantics

**Source:** `Sec 4.1 - Package Intro - numpy.ipynb`

**Rules:**
- **Slice** (`a[0:2, 0:2]`) → **view** (shares memory; modifying view modifies original).
- **Fancy indexing** (`a[[0,1], :]`) → **copy** (independent).
- `.copy()` — always produces independent copy.
- `np.may_share_memory(a, b)` — checks if two arrays share memory.
- `reshape` returns view when possible (contiguous array); returns copy if array is non-contiguous.

**Test case:**

`v2_sub = v2[:2, :2]` — modifying `v2_sub[0,0]` changes `v2[0,0]`.
`v2_sub_copy = v2[:2, :2].copy()` — modifying copy leaves `v2` unchanged.

---

### Topic 1.44 — NumPy: Meshgrid for 2D Function Evaluation

**Source:** `Sec 4.1 - Package Intro - numpy.ipynb`

**Mathematical definition:**

Given 1-D arrays $x$ (length $m$) and $y$ (length $n$), `np.meshgrid(x, y)` returns:
- $X$ of shape $(n, m)$ with $X_{ij} = x_j$
- $Y$ of shape $(n, m)$ with $Y_{ij} = y_i$

So $Z_{ij} = f(X_{ij},\, Y_{ij})$ evaluates $f$ over the full grid.

**Interface:**

`X, Y = np.meshgrid(x, y)` — default `indexing='xy'` (Cartesian).

**Gotcha:** default `indexing='xy'` means $X$ varies along columns and $Y$ along rows; use `indexing='ij'` for matrix-style (row = $x$ axis).

**Test case:**

`x = y = np.linspace(-1, 1, 5)`, $f(x,y) = e^{-(x^2+y^2)}$: `Z = np.exp(-(X**2 + Y**2))` — shape `(5, 5)`, maximum at centre.

---

## 2. Implementation Pitfall Register

---

### Pitfall 2.1 — Mutable Default Argument

**Wrong assumption:** Using a list or dict as a default argument value initialises a fresh container on each call.

**Correct behavior:** The default is created **once** at `def` time and reused across all calls. Mutations persist.

**Correct pattern:** Use `None` as default; create the container inside the function body.

**Applies to:** Topic 1.33, any function with `list`, `dict`, or `set` default parameters.

---

### Pitfall 2.2 — NumPy Slice Returns View, Not Copy

**Wrong assumption:** `sub = arr[0:2, 0:2]` creates an independent copy; modifying `sub` is safe.

**Correct behavior:** The slice is a **view** — modifying `sub` modifies `arr` in-place.

**Correct pattern:** Use `.copy()` to obtain an independent copy.

**Applies to:** Topic 1.43, any NumPy slicing operation.

---

### Pitfall 2.3 — Float Assignment into Integer Array Silently Truncates

**Wrong assumption:** Assigning a float (e.g., `-np.pi`) to an `int`-dtype array rounds to the nearest integer.

**Correct behavior:** The decimal part is **truncated toward zero** without warning. `-3.1416` becomes `-3`, not `-3`.

**Applies to:** any `np.int32`/`np.int64` array assigned float values.

---

### Pitfall 2.4 — `*` is Element-wise, Not Matrix Multiplication

**Wrong assumption:** `A * B` computes matrix product.

**Correct behavior:** `A * B` is element-wise (Hadamard product). Use `A @ B` or `np.dot(A, B)` for matrix multiplication.

**Applies to:** Topics 1.1, 1.18, any matrix computation.

---

### Pitfall 2.5 — NumPy Logical Operators Must Be Bitwise Symbols

**Wrong assumption:** `arr > 0 and arr < 10` filters array elementwise.

**Correct behavior:** Python `and`/`or`/`not` operate on single booleans and raise `ValueError` on arrays. Use `&`, `|`, `~` with parentheses around each condition: `(arr > 0) & (arr < 10)`.

**Applies to:** any boolean masking in NumPy.

---

### Pitfall 2.6 — `np.linalg.inv(L)` Inside Loop (Efficiency)

**Wrong assumption:** Calling `inv(L)` inside the iteration loop is equivalent to `solve(L, rhs)`.

**Correct behavior:** Both are mathematically equivalent, but `np.linalg.inv` recomputes the full inverse every iteration — $O(n^3)$ overhead. `np.linalg.solve(L, rhs)` performs one triangular solve — $O(n^2)$ for triangular $L$.

**Applies to:** Topic 1.1 (Gauss-Seidel).

---

### Pitfall 2.7 — Simpson's Rule Requires Even `n`

**Wrong assumption:** Simpson's rule works for any positive integer $n$.

**Correct behavior:** $n$ must be **even**; the coefficient pattern $1, 4, 2, \ldots, 4, 1$ is undefined for odd $n$. Raise `ValueError` if `n % 2 != 0`.

**Applies to:** Topic 1.5.

---

### Pitfall 2.8 — `meshgrid` Default Indexing

**Wrong assumption:** `np.meshgrid(x, y)` returns $X$ varying along rows (matrix convention).

**Correct behavior:** Default `indexing='xy'` makes $X$ vary along **columns** and $Y$ along **rows** (Cartesian convention). For matrix-style indexing, pass `indexing='ij'`.

**Applies to:** Topic 1.44.

---

### Pitfall 2.9 — `np.empty` Returns Uninitialized Memory

**Wrong assumption:** `np.empty(shape)` creates an array of zeros.

**Correct behavior:** `np.empty` allocates memory but does **not** initialise values — contents are whatever was in memory. Use `np.zeros` unless you are certain every element will be overwritten.

**Applies to:** any array allocation.

---

### Pitfall 2.10 — Global Variable Shadowed by Local Assignment

**Wrong assumption:** Assigning to a variable inside a function that has the same name as a global reads the global first, then assigns locally.

**Correct behavior:** If Python sees **any** assignment to a name inside a function, that name is treated as **local throughout the entire function** — even lines before the assignment will raise `UnboundLocalError`.

Use `global var` declaration to write to a global from inside a function.

**Applies to:** any function that both reads and writes a name also defined at module scope.

---

### Pitfall 2.11 — Double-Underscore Attribute Access

**Wrong assumption:** `obj.__attr` accesses a double-underscore attribute normally.

**Correct behavior:** Python name-mangles `__attr` in class body to `_ClassName__attr`. External access via `obj.__attr` raises `AttributeError`; the mangled name `obj._ClassName__attr` works but is discouraged.

**Applies to:** Topic 1.27.

---

### Pitfall 2.12 — Affine Cipher: Invalid Multiplier

**Wrong assumption:** Any integer $a$ is a valid Affine cipher key.

**Correct behavior:** $a$ must satisfy $\gcd(a, 26) = 1$; otherwise the encryption function is not injective and decryption is impossible.

**Applies to:** Topic 1.16.

---

### Pitfall 2.13 — Hill Cipher: Modular Inverse ≠ Floating-Point Inverse

**Wrong assumption:** `np.linalg.inv(K)` gives the decryption matrix for Hill cipher.

**Correct behavior:** `np.linalg.inv(K)` computes the real-valued inverse, which gives wrong results in $\mathbb{Z}_{26}$. Must use the **modular matrix inverse**: $K^{-1} \bmod 26$ computed with integer arithmetic.

**Applies to:** Topic 1.18.

---

### Pitfall 2.14 — Double `argsort` for Ranks

**Wrong assumption:** `arr.argsort()` gives ranks.

**Correct behavior:** `arr.argsort()` gives **indices that would sort the array** — a different thing. A second `argsort()` converts those into ranks. Pattern: `ranks = arr.argsort().argsort()`.

**Applies to:** Topics 1.20, 1.21, 1.41.

---

### Pitfall 2.15 — `linspace` vs `arange` Endpoint Behaviour

**Wrong assumption:** `np.arange(a, b, step)` always includes $b$.

**Correct behavior:** `arange` **excludes** $b$ (like Python `range`). With float steps, floating-point rounding can produce 10 or 11 elements unpredictably. Use `np.linspace(a, b, n)` when exact point count matters; it **includes** $b$ by default.

**Applies to:** any grid construction for integration, plotting.

---

## 3. Instructions for Codex

Read the full spec before writing any code.
Implement every topic in Section 1 as described.
Check Section 2 before finalizing any implementation — common mistakes are listed there.
After implementing each topic, run the specified test case and confirm the output matches.
Use only the packages listed in Section 0.
