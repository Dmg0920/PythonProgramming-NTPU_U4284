from __future__ import annotations

import contextlib
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import slides_basic as sb
import slides_flow_control as sfc
import slides_function_class as sfun
import slides_packages as sp


def test_import_smoke_and_concept_shapes() -> None:
    assert isinstance(sb.COURSE_CONTENTS, list) and sb.COURSE_CONTENTS
    assert isinstance(sb.DATA_ANALYSIS_WORKFLOW, list) and sb.DATA_ANALYSIS_WORKFLOW
    assert isinstance(sb.PYTHON_CORE_FEATURES, list) and sb.PYTHON_CORE_FEATURES
    assert isinstance(sb.OOP_PRINCIPLES, list) and sb.OOP_PRINCIPLES
    assert isinstance(sb.UV_NOTES, dict) and "summary" in sb.UV_NOTES

    assert isinstance(sfun.TYPE_CHECKING_NOTES, list) and sfun.TYPE_CHECKING_NOTES
    assert isinstance(sfun.PYDANTIC_FIELD_NOTES, list) and sfun.PYDANTIC_FIELD_NOTES

    assert isinstance(sp.BROADCASTING_RULES, list) and sp.BROADCASTING_RULES
    assert isinstance(sp.PANDAS_MISSINGNESS_NOTES, list) and sp.PANDAS_MISSINGNESS_NOTES
    assert isinstance(sp.VISUALIZATION_PRINCIPLES, list) and sp.VISUALIZATION_PRINCIPLES

    for group in [
        sb.PYTHON_CORE_FEATURES,
        sb.OOP_PRINCIPLES,
        sfun.TYPE_CHECKING_NOTES,
        sfun.PYDANTIC_FIELD_NOTES,
        sp.BROADCASTING_RULES,
        sp.PANDAS_MISSINGNESS_NOTES,
        sp.VISUALIZATION_PRINCIPLES,
    ]:
        for entry in group:
            assert "title" in entry
            assert "summary" in entry


def test_slides_basic_execution() -> None:
    assert sb.to_snake_case("HelloWorld Example") == "hello_world_example"
    assert sb.to_pascal_case("hello world") == "HelloWorld"
    assert sb.to_camel_case("hello world") == "helloWorld"
    assert sb.is_python_keyword("for")
    assert "step_3" in sb.indentation_example(3)
    assert sb.boolean_operations(True, False)["xor"] is True
    assert sb.arithmetic_operations(8, 2)["divide"] == 4
    original, shallow, deep = sb.shallow_vs_deep_copy([[1], [2]])
    assert original == [[1, 99], [2]]
    assert shallow == [[1, 99], [2]]
    assert deep == [[1], [2]]
    assert sb.make_frozenset([1, 2, 2]) == frozenset({1, 2})
    assert list(sb.generator_squares(4)) == [0, 1, 4, 9]
    growth = sb.growth_rate_examples([100, 110, 121])
    assert np.allclose(growth["relative_change"], [0.1, 0.1])


def test_flow_control_execution_parity() -> None:
    assert sfc.if_elif_else_demo(85) == "B"
    assert sfc.nested_if_demo(-1, 2) == "quadrant II boundary"
    assert sfc.while_countdown(3) == [3, 2, 1]
    assert sfc.for_loop_sum([1, 2, 3]) == 6
    assert sfc.lcg_generate(1, 2, 0, 9, 8) == [1, 2, 4, 8, 7, 5, 1, 2]
    floats = sfc.lcg_float_stream(1664525, 1013904223, 2**32, 42, 2)
    assert len(floats) == 2
    means = sfc.law_of_large_numbers_demo(20, seed=7)
    assert means.shape == (20,)
    inv = sfc.inverse_transform_continuous(lambda u: -np.log(1 - u), 500, seed=7)
    assert 0.8 < float(inv.mean()) < 1.2
    assert sfc.simulation_summary(np.array([1, 2, 3]))["mean"] == 2.0


def test_function_class_execution() -> None:
    assert sfun.add_with_docstring(2, 3) == 5
    styles = sfun.argument_styles(1, 2, 3, 4, scale=2)
    assert styles["scaled_total"] == 20
    lambdas = sfun.lambda_examples()
    assert lambdas["square"](5) == 25
    assert sfun.dynamic_typing_demo() == ["int", "float", "str"]
    assert sfun.typed_inner_product([1, 3, 4], [1, 2, -3]) == -5
    product = sfun.ProductSpec(name="pen", price=20, category="tool")
    assert product.price == 20
    student = sfun.StudentProfile(name="alice chen", age=20, gpa=3.8)
    assert student.name == "Alice Chen"
    assert sfun.hanoi_min_moves(4) == 15
    assert len(sfun.hanoi_moves(3)) == 7
    assert sfun.factorial_recursive(5) == 120
    assert sfun.triangular_recursive(5) == 15
    fib = sfun.fibonacci_family(10)
    assert fib["recursive"] == fib["memo"] == fib["iterative"] == fib["closed_form"] == 55
    counter_a = sfun.ClassCounter("A")
    counter_b = sfun.ClassCounter("B")
    assert counter_a.bump() == 1
    assert counter_b.bump() == 1
    assert sfun.ClassCounter.created() >= 2


def test_packages_parity_and_smoke() -> None:
    attrs = sp.array_attributes(np.zeros((2, 3), dtype=np.int32))
    assert attrs["shape"] == (2, 3)
    assert np.array_equal(sp.int_array_float_truncation([1.9, -3.1]), [1, -3])
    stacks = sp.concatenate_stack([1, 2], [3, 4])
    assert np.array_equal(stacks["concatenate"], [1, 2, 3, 4])
    compat = sp.broadcasting_compatibility((3, 2), (2,))
    assert compat["compatible"] is True
    assert compat["result_shape"] == (3, 2)
    view_demo = sp.slice_is_view_demo()
    assert view_demo["shares_memory"] is True
    assert np.array_equal(sp.mask_between(np.arange(6), 2, 4), [2, 3, 4])
    assert sp.selection_sort([3, 1, 2]) == [1, 2, 3]
    assert sp.bubble_sort([3, 1, 2]) == [1, 2, 3]
    assert sp.insertion_sort([3, 1, 2]) == [1, 2, 3]
    assert sp.merge_sort([3, 1, 2]) == [1, 2, 3]
    assert sp.quick_sort([3, 1, 2]) == [1, 2, 3]
    assert np.allclose(sp.solve_linear_system(np.eye(2), [5, 6]), [5, 6])

    A = np.array([[8, 3, -3], [-2, -8, 5], [3, 5, 10]], dtype=float)
    b = np.array([[14], [5], [-8]], dtype=float)
    x0 = np.zeros((3, 1))
    with contextlib.redirect_stdout(io.StringIO()):
        x = sp.Gauss_Siedel(A, b, x0, tol=1e-3)
    assert np.allclose(x.ravel(), [2.0893, -1.5531, -0.6502], atol=1e-3)

    proj = sp.project_vector(np.array([[1.0], [1.0]]), [2, 0])
    assert np.allclose(proj, [1, 1])

    df = pd.DataFrame({"x": [1, 2, np.nan], "g": ["a", "a", "b"]}, index=["r1", "r2", "r3"])
    loc_row, iloc_row = sp.select_loc_iloc(df, "r1", 0)
    assert loc_row["x"] == iloc_row["x"] == 1
    aligned = sp.index_aligned_add(pd.Series([1, 2], index=["a", "b"]), pd.Series([10, 20], index=["b", "c"]), fill_value=0)
    assert aligned.loc["c"] == 20
    assert sp.interpolate_missing(pd.Series([1.0, np.nan, 3.0])).iloc[1] == 2
    nullable = sp.nullable_int_series([1, None, 3])
    assert str(nullable.dtype) == "Int64"
    indicator = sp.add_missing_indicator(df, "x")
    assert indicator.loc["r3", "x_missing"] == 1
    assert sp.join_category(pd.DataFrame({"id": [1, 2]}), pd.DataFrame({"id": [1, 1]}), "id") == "one-to-many"

    unary = sp.index_preserving_unary(pd.Series([1, 4], index=["a", "b"]), lambda s: np.sqrt(s))
    assert list(unary.index) == ["a", "b"]

    fig1, ax1 = plt.subplots()
    sp.plot_line([0, 1], [0, 1], ax=ax1)
    assert ax1.lines
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    box_ax = sp.plot_box([[1, 2, 3], [2, 3, 4]], labels=["A", "B"], ax=ax2)
    assert box_ax is ax2
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    violin = sp.plot_violin([[1, 2, 3], [2, 3, 4]], labels=["A", "B"], ax=ax3)
    assert violin["ax"] is ax3
    plt.close(fig3)

    fig4, ax4 = plt.subplots()
    heat = sp.plot_heatmap(np.arange(9).reshape(3, 3), ax=ax4)
    assert heat["ax"] is ax4
    plt.close(fig4)

    X, Y, Z = sp.plot_3d_ready_grid(lambda X, Y: X + Y, (-1, 1), (-1, 1), n=5)
    assert X.shape == Y.shape == Z.shape == (5, 5)


def main() -> None:
    test_import_smoke_and_concept_shapes()
    test_slides_basic_execution()
    test_flow_control_execution_parity()
    test_function_class_execution()
    test_packages_parity_and_smoke()
    print("slides modules tests passed")


if __name__ == "__main__":
    main()
