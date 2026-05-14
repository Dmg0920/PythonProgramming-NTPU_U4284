from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

import slides_function_class as sfc


def test_sierpinski_methods_match() -> None:
    helper = sfc.SierpinskiTriangle()
    direct = helper.matrix_method1(3)
    recursive = helper.matrix_method2(3)
    assert direct.shape == (8, 8)
    assert np.array_equal(direct, recursive)


def test_sierpinski_base_pattern() -> None:
    helper = sfc.SierpinskiTriangle()
    expected = np.array([[1, 1], [1, 0]], dtype=int)
    assert np.array_equal(helper.matrix_method1(1), expected)
    assert np.array_equal(helper.matrix_method2(1), expected)


def test_sierpinski_plot_grid() -> None:
    helper = sfc.SierpinskiTriangle()
    fig, axes = helper.plot_methods(method1_ks=[1, 2], method2_ks=[1, 3])
    assert len(fig.axes) == 4
    assert axes.shape == (2, 2)


def main() -> None:
    test_sierpinski_methods_match()
    test_sierpinski_base_pattern()
    test_sierpinski_plot_grid()
    print("sierpinski helper tests passed")


if __name__ == "__main__":
    main()
