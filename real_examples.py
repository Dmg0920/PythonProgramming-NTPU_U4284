from __future__ import annotations

import contextlib
import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

import exam_prep as ep


def print_section(title: str) -> None:
    # 統一每個案例段落的輸出格式，執行整份檔案時比較好掃描。
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

# Example:
# print_section("Demo")


def _format_array(values: np.ndarray) -> str:
    # 把 numpy 向量格式化成固定精度，避免不同環境印出太多小數位。
    return np.array2string(np.asarray(values, dtype=float), precision=4, suppress_small=True)

# Example:
# _format_array(np.array([1.23456, 7.89012]))


def example_linear_systems() -> None:
    # MidExam 第一題：先檢查矩陣是否具有對角優勢，再示範 Gauss-Seidel 的收斂結果。
    print_section("1. Solution to Systems of Linear Equations")
    print("案例背景：用 MidExam 的矩陣，先檢查對角優勢，再用 Gauss-Seidel 找近似解。")

    A = np.array([[8, 3, -3], [-2, -8, 5], [3, 5, 10]], dtype=float)
    b = np.array([[14], [5], [-8]], dtype=float)
    x0 = np.zeros((3, 1), dtype=float)

    print("輸入摘要：")
    print(f"A =\n{A}")
    print(f"b =\n{b}")
    print(f"x0 =\n{x0}")

    # 原始函式會直接 print 過程，這裡先攔下來，只保留最後需要看的摘要。
    dd_buffer = io.StringIO()
    gs_buffer = io.StringIO()
    with contextlib.redirect_stdout(dd_buffer):
        ep.DD_check(A)
    with contextlib.redirect_stdout(gs_buffer):
        solution = ep.Gauss_Siedel(A, b, x0, tol=1e-3)

    dd_summary = dd_buffer.getvalue().strip() or "DD check produced no output."
    gs_lines = [line.strip() for line in gs_buffer.getvalue().splitlines() if line.strip()]
    convergence = next((line for line in reversed(gs_lines) if "Converged" in line), "No convergence summary found.")

    print("結果：")
    print(dd_summary)
    print(convergence)
    print(f"final solution = {_format_array(solution.ravel())}")

# Example:
# example_linear_systems()


def example_numerical_integration() -> None:
    # MidExam 第二題：同一個積分題同時比較 deterministic 與 Monte Carlo 近似法。
    print_section("2. Numerical Integration")
    print("案例背景：用 MidExam 的 sin(x) 積分題，比較 Midpoint、Trapezoid 與 Monte Carlo。")

    f = np.sin
    area = [0.0, math.pi]
    seq_n = [11, 101, 1001, 10001]
    true_value = 2.0

    print("輸入摘要：")
    print(f"f(x) = sin(x), area = {area}, n = {seq_n}")

    # MidExam 的 n 寫法是 grid points；這裡用 n-1 對應 exam_prep 的 subinterval 定義。
    rows: list[list[float]] = []
    for n in seq_n:
        num_int = ep.Integration(f, area[0], area[1], n - 1)
        midpoint = num_int.Midpt()
        trapezoid = num_int.Trapezoid()
        rows.append(
            [
                midpoint,
                ep.ARE(midpoint, true_value),
                trapezoid,
                ep.ARE(trapezoid, true_value),
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=["Midpoint", "ARE-Midpoint(%)", "Trapezoid", "ARE-Trapezoid(%)"],
        index=[f"n = {n}" for n in seq_n],
    )

    np.random.seed(7)
    mc_runs = [100, 1000, 10000, 100000]
    mc_rows = []
    for runs in mc_runs:
        # 固定 random seed，讓這份示範檔每次跑出來的數字穩定可比。
        estimate = ep.MCMC(f, area, runs)
        mc_rows.append([estimate, ep.ARE(estimate, true_value)])
    mc_df = pd.DataFrame(mc_rows, columns=["MonteCarlo", "ARE-MonteCarlo(%)"], index=[f"runs = {n}" for n in mc_runs])

    print("結果：")
    print(df.to_string(float_format=lambda x: f"{x:.6f}"))
    print()
    print(mc_df.to_string(float_format=lambda x: f"{x:.6f}"))

# Example:
# example_numerical_integration()


def example_gradient_descent() -> None:
    # MidExam 第三題：把不同 learning rate 的軌跡排成表格，比較收斂速度與穩定性。
    print_section("3. Gradient Descend Method")
    print("案例背景：沿用 MidExam 的目標函式與起點，比較不同 learning rate 的前 10 步路徑。")

    alpha_set = [0.01, 0.1, 0.2, 0.3, 0.5, 0.75]
    pt = [5, 4]
    objective = "(9/16) * (x - 2)^2 + (y - 2)^2 + x*y/4"
    grad_kit = ep.GradientDescent(pt, 10)

    print("輸入摘要：")
    print(f"objective = {objective}")
    print(f"initial point = {pt}")
    print(f"learning rates = {alpha_set}")

    # 每個 alpha 產生一個兩欄 DataFrame，最後橫向拼接成和 MidExam 類似的閱讀方式。
    saver = []
    row_index = ["init pt"] + [f"iter {k}" for k in range(1, 11)]
    for alpha in alpha_set:
        path = grad_kit.GradD(alpha)
        saver.append(
            pd.DataFrame(
                path,
                index=row_index,
                columns=[f"({alpha},theta1)", f"({alpha},theta2)"],
            )
        )
    gd_df = pd.concat(saver, axis=1)

    momentum_df = pd.DataFrame(
        grad_kit.Momentum(0.5),
        index=row_index,
        columns=["(0.5,theta1)", "(0.5,theta2)"],
    )

    print("結果：")
    print(gd_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print()
    print("Momentum(0.5)")
    print(momentum_df.to_string(float_format=lambda x: f"{x:.4f}"))

# Example:
# example_gradient_descent()


def example_ciphers() -> None:
    # MidExam 第四題：保留三種典型情境，包含加密與 brute-force 解密。
    print_section("4. Encryptin Alphabetic Text")
    print("案例背景：直接重跑 MidExam 的 Caesar、Vigenere 與 brute-force cracking。")

    plain_text1 = "HELLO EVERYONE"
    shift = 6
    plain_text2 = "attack at dawn"
    keyword = "LEMON"
    plain_text3 = "exxego ex srgi"

    print("輸入摘要：")
    print(f"plaintext 1 = {plain_text1}, Caesar key = {shift}")
    print(f"plaintext 2 = {plain_text2}, Vigenere key = {keyword}")
    print(f"ciphertext 3 = {plain_text3}")

    print("結果：")
    print(f"Caesar ciphertext = {ep.Caeser_Cipher(plain_text1, shift)}")
    print(f"Vigenere ciphertext = {ep.Vigenere_Cipher(plain_text2, keyword)}")
    # 這一題的重點就是列出所有 shift 候選，所以保留完整 26 行輸出。
    print("Brute-force crack:")
    ep.Crack_Text(plain_text3)

# Example:
# example_ciphers()


def example_association_measures() -> None:
    # MidExam 第五題：同時顯示自寫版本與 scipy 結果，方便檢查公式與實作是否一致。
    print_section("5. Association Measure")
    print("案例背景：比較自寫的 Spearman rho / Kendall tau 與 scipy 的結果。")

    X = [106, 100, 86, 101, 99, 103, 97, 113, 112, 110]
    Y = [7, 27, 2, 50, 28, 29, 20, 12, 6, 17]
    assm = ep.RankCorrelation(X, Y)

    print("輸入摘要：")
    print(f"X = {X}")
    print(f"Y = {Y}")

    rho_coef, rho_pval = assm.rho()
    tau_coef, tau_pval = assm.tau()
    sp_coef, sp_pval = spearmanr(X, Y)
    kd_coef, kd_pval = kendalltau(X, Y, method="asymptotic")

    result = pd.DataFrame(
        [
            ["RankCorrelation.rho()", rho_coef, rho_pval],
            ["scipy.stats.spearmanr", float(sp_coef), float(sp_pval)],
            ["RankCorrelation.tau()", tau_coef, tau_pval],
            ["scipy.stats.kendalltau", float(kd_coef), float(kd_pval)],
        ],
        columns=["Method", "coefficient", "p-value"],
    )

    print("結果：")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Example:
# example_association_measures()


def example_extra_sec_cases() -> None:
    # 這一段不是 MidExam 原題，而是補幾個常見的 Sec notebook 技巧做快速複習。
    print_section("Extra Sec Examples")
    print("案例背景：補 3 個非 MidExam 但考前很好用的 NumPy、pandas、visualization 範例。")

    print("1) pandas missing + groupby")
    sales = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "B"],
            "score": [90.0, np.nan, 75.0, 88.0, np.nan],
            "name": ["Amy", "Alan", "Ben", "Bella", "Bruce"],
        }
    )
    filled = ep.fill_missing(sales, "mean")
    grouped = ep.group_zscore(filled, "group", "score")
    summary = ep.group_summary(grouped, "group", "score")
    # 這裡串起 missing -> fill -> transform -> summary，是 pandas 很常見的流程。
    print("missing summary:")
    print(ep.missing_summary(sales).to_string(float_format=lambda x: f"{x:.4f}"))
    print("group summary:")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print()

    print("2) NumPy broadcasting / projection")
    matrix = np.array([[1.0], [1.0]])
    vector = np.array([2.0, 0.0])
    centered = ep.subtract_row_means(np.array([[1.0, 3.0], [4.0, 10.0]]))
    projection = ep.project_vector(matrix, vector)
    # 一個例子示範 broadcasting，另一個例子示範線性代數中的 projection。
    print(f"subtract_row_means =\n{centered}")
    print(f"projection of {vector.tolist()} onto col(A) = {_format_array(projection)}")
    print()

    print("3) visualization helper")
    x = np.linspace(0, 2 * math.pi, 50)
    y = np.sin(x)
    err = np.full_like(x, 0.15)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # 只建立 figure 驗證 helper 可正常組圖，不在這份案例檔內主動 show 視窗。
    ep.plot_line(x, y, title="sin(x)", ax=axes[0])
    ep.plot_confidence_band(x, y, err, ax=axes[1])
    axes[1].set_title("sin(x) with confidence band")
    print(f"created figure with {len(fig.axes)} axes")
    plt.close(fig)

# Example:
# example_extra_sec_cases()


def main() -> None:
    # 依照 MidExam 題目順序執行，最後再補課堂上的延伸案例。
    example_linear_systems()
    example_numerical_integration()
    example_gradient_descent()
    example_ciphers()
    example_association_measures()
    example_extra_sec_cases()

# Example:
# main()


if __name__ == "__main__":
    main()
