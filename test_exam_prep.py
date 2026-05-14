from __future__ import annotations

import contextlib
import io
import math

import numpy as np

import exam_prep as ep


def close(value: float, expected: float, tol: float) -> None:
    assert abs(value - expected) <= tol, (value, expected)


def test_linear_solver_and_integration() -> None:
    A = np.array([[8, 3, -3], [-2, -8, 5], [3, 5, 10]], dtype=float)
    b = np.array([[14], [5], [-8]], dtype=float)
    x0 = np.zeros((3, 1))
    with contextlib.redirect_stdout(io.StringIO()):
        x = ep.Gauss_Siedel(A, b, x0, tol=1e-3)
        ep.DD_check(A)
    assert np.allclose(x.ravel(), [2.0893, -1.5531, -0.6502], atol=1e-3)

    # The spec lists these expected values with n=11, but the values correspond
    # to 10 subintervals / 11 grid points under the standard formulas.
    close(ep.midpoint_rule(np.sin, 0, np.pi, 10), 2.00825, 1e-5)
    close(ep.trapezoid_rule(np.sin, 0, np.pi, 10), 1.98352, 1e-5)
    close(ep.simpsons_rule(np.sin, 0, np.pi, 10), 2.00011, 1e-4)
    close(ep.ARE(2.0082484, 2.0), 0.4107, 1e-3)

    diameters = np.array([0, 10, 12, 10, 8, 6, 8, 10, 0], dtype=float)
    areas = np.pi * diameters**2 / 8
    close(ep.simpsons_rule(areas, 2), 494.28, 0.01)


def test_roots_gradient_and_ciphers() -> None:
    f = lambda x: x - math.cos(x)
    df = lambda x: 1 + math.sin(x)
    close(ep.bisection(f, 0, np.pi), 0.7391, 1e-4)
    close(ep.newton(f, df, 0.5), 0.7391, 1e-4)
    close(ep.secant(f, 0.5, 1), 0.7391, 1e-4)
    close(ep.fixed_point(math.cos, 0.5), 0.7391, 1e-4)

    gd = ep.GradientDescent([5, 4], 10)
    assert np.allclose(gd.GradD(0.1)[1], [4.5625, 3.4750], atol=1e-4)
    assert np.allclose(gd.GradD(0.1)[10], [2.5352, 1.8582], atol=1e-4)
    assert np.allclose(gd.Momentum(0.5)[1], [2.8125, 1.3750], atol=1e-4)
    assert np.allclose(gd.Momentum(0.5)[10], [1.6779, 2.7331], atol=1e-4)

    assert ep.Caeser_Cipher("HELLO EVERYONE", 6) == "NKRRU KBKXEUTK"
    assert ep.Vigenere_Cipher("attack at dawn", "LEMON") == "lxfopv ef rnhr"
    assert ep.AffineCipher(5, 8).encrypt("AFFINE") == "IHHWVC"
    assert ep.AffineCipher(5, 8).decrypt("IHHWVC") == "AFFINE"
    sub = ep.SubstitutionCipher("QWERTYUIOPASDFGHJKLZXCVBNM")
    assert sub.encrypt("HELLO") == "ITSSG"
    hill = ep.HillCipher(np.array([[3, 3], [2, 5]]))
    assert hill.encrypt("HELP") == "HIAT"


def test_statistics_sampling_and_random_generators() -> None:
    X = [106, 100, 86, 101, 99, 103, 97, 113, 112, 110]
    Y = [7, 27, 2, 50, 28, 29, 20, 12, 6, 17]
    corr = ep.RankCorrelation(X, Y)
    rho, rho_p = corr.rho()
    tau, tau_p = corr.tau()
    close(rho, -0.1758, 1e-4)
    close(rho_p, 0.6272, 1e-4)
    close(tau, -0.1111, 1e-4)
    close(tau_p, 0.6547, 1e-4)

    np.random.seed(7)
    s1, r1 = ep.sample_candidate1(300)
    s2, r2 = ep.sample_candidate2(300)
    assert len(s1) == len(s2) == 300
    assert 0.45 < r1 < 0.80
    assert 0.65 < r2 < 0.95

    np.random.seed(7)
    samples = ep.inverse_transform(lambda u: -np.log(1 - u), 5000)
    close(float(samples.mean()), 1.0, 0.05)
    normals = ep.box_muller(5000)
    close(float(normals.mean()), 0.0, 0.05)
    close(float(normals.std()), 1.0, 0.05)

    lcg = ep.LCG(1664525, 1013904223, 2**32, 42)
    close(lcg.next(), 0.2523451748, 1e-10)
    np.random.seed(7)
    close(ep.monte_carlo_pi(100000), math.pi, 0.02)


def test_oop_functions_and_numpy_patterns() -> None:
    assert ep.Temperature(100).fahrenheit == 212.0
    try:
        ep.Temperature(-300)
    except ValueError:
        pass
    else:
        raise AssertionError("Temperature below absolute zero should fail.")

    account = ep.BankAccount("A", 1000, 0)
    assert account.get_balance() == 1000
    assert not hasattr(account, "__balance")
    assert ep.Pet().make_sound() == "Hello"
    assert ep.Dog().make_sound() == "Woof!"
    assert isinstance(ep.Dog(), ep.Pet)
    assert ep.ExamD().who() == "B"
    assert [cls.__name__ for cls in ep.ExamD.__mro__] == [
        "ExamD",
        "ExamB",
        "ExamC",
        "ExamA",
        "object",
    ]
    wd = ep.ExamWorkingDog("Lucky", "guide")
    assert wd.name == "Lucky"
    assert wd.job == "guide"
    assert [obj.make_sound() for obj in [ep.ExamBell(), ep.ExamPhone()]] == ["ding", "ring"]

    assert ep.running_product(2, 3, 4, 5) == [2.0, 6.0, 24.0, 120.0]
    assert ep.sentence(subject="You", object="me", verb="beat") == "You beat me"
    assert ep.correct_bucket(1) == [1]
    assert ep.correct_bucket(2) == [2]
    expected_fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    assert [ep.fib_iterative(k) for k in range(15)] == expected_fib
    assert [ep.fib_memo(k) for k in range(15)] == expected_fib
    assert [ep.fib_binet(k) for k in range(15)] == expected_fib
    assert ep.iprod([1, 3, 4], [1, 2, -3]) == -5
    assert (ep.QM(12.87, 3.12), ep.AM(12.87, 3.12), ep.GM(12.87, 3.12), ep.HM(12.87, 3.12)) == (
        9.36,
        7.99,
        6.34,
        5.02,
    )
    # These assert the formula in spec.md. The table values in Topic 1.38 are
    # not consistent with P=12000, r=0.075, t=3.
    close(ep.Final_Value(12000, 0.075, 3, 4), 14996.5965, 1e-3)
    close(ep.Final_Value(12000, 0.075, 3, "inf"), 15027.8726, 1e-3)
    assert ep.logit_grow(100, 1, 0) == 50.0

    arr = np.array([3, 1, 4, 1, 5])
    assert ep.array_ranks(arr).tolist() == [2, 0, 3, 1, 4]
    A = np.arange(1, 10).reshape(3, 3)
    lower, upper, strict_upper, strict_lower = ep.triangular_parts(A)
    assert np.array_equal(lower, np.tril(A))
    assert np.array_equal(upper, np.triu(A))
    assert np.array_equal(strict_upper, A - np.tril(A))
    assert np.array_equal(strict_lower, A - np.triu(A))
    data = np.array([[1, 2], [3, 4]])
    assert np.allclose(ep.subtract_row_means(data), [[-0.5, 0.5], [-0.5, 0.5]])
    assert np.array_equal(ep.outer_product([1, 2]), [[1, 2], [2, 4]])
    Xg, Yg, Zg = ep.meshgrid_values(
        np.linspace(-1, 1, 5),
        np.linspace(-1, 1, 5),
        lambda X, Y: np.exp(-(X**2 + Y**2)),
    )
    assert Xg.shape == Yg.shape == Zg.shape == (5, 5)
    assert Zg[2, 2] == 1.0


def test_sec_notebook_expansions() -> None:
    assert ep.cafeteria_action(True, False, False) == "deliver the choice"
    assert ep.cafeteria_action(True, True, False) == "light the error light"
    assert ep.quotient_remainder(7, 2) == (3, 1)
    assert ep.exact_decimal_check() == (True, True)
    assert ep.float_isclose_sum()
    assert ep.regex_search_one(r"[A-Z]\d+", "ID: A12, B34") == "A12"
    assert ep.regex_find(r"User\d", "User9, UserN, User8") == ["User9", "User8"]
    assert ep.regex_split(r"\s", "The rain in Spain", maxsplit=1) == ["The", "rain in Spain"]
    assert ep.regex_replace("yellow", "nice", "yellow car yellow house", count=1) == "nice car yellow house"
    assert list(ep.seq_div(20, [2, 3])) == [0, 6, 12, 18]
    assert list(ep.exam_countdown(3)) == [3, 2, 1]
    assert ep.generator_consumption_demo(3) == ([0, 1, 4], [])
    assert ep.join_words(["Python", "is", "clear"]) == "Python is clear"
    original, shallow, deep = ep.shallow_deep_copy_demo([[1], [2]])
    assert original == [[1, 99], [2]]
    assert shallow == [[1, 99], [2]]
    assert deep == [[1], [2]]
    close(ep.zero_truncated_poisson_pmf(1, 2), 0.3130, 1e-3)
    assert ep.dict_filter_by_value({"a": 1, "b": 3}, 2) == {"b": 3}
    assert ep.set_operations({1, 2}, {2, 3})["symmetric_difference"] == {1, 3}
    assert np.allclose(ep.increasing_rate([100, 110, 121]), [0.1, 0.1])

    assert ep.lcg_sequence(1, 2, 0, 9, 8) == [1, 2, 4, 8, 7, 5, 1, 2]
    close(ep.steffensen(math.cos, 0.5), 0.7391, 1e-4)
    np.random.seed(11)
    assert len(ep.dice_running_mean(20)) == 20
    walk_x, walk_y = ep.random_walk_2d(bound=3, seed=1)
    assert abs(walk_x[-1]) > 3 or abs(walk_y[-1]) > 3

    fv = ep.Final_Value(12000, 0.075, 3, "inf")
    close(ep.present_value(fv, 0.075, 3), 12000, 1e-8)
    assert ep.grade_safe(90, 30) == 60
    assert ep.weighted_avg([90, 80, 70], [0.3, 0.3, 0.4]) == 79
    assert ep.parser("cy.sc10@nycu.edu.tw") == "cy.sc10"
    assert ep.parser("not-email") is None
    assert ep.cosine_sim([1, 0], [0, 1]) == 0
    assert ep.Product(price="99").price == 99
    try:
        ep.User(age="20")
    except Exception:
        pass
    else:
        raise AssertionError("strict pydantic field should reject string age")
    assert ep.NightClub_check(ep.Guest(name="Diego", age=28), room_No=9457) == "success : Diego enter room 9457"
    emp = ep.Employee(name="alice chen", salary=75000, address=ep.Address(city="Taipei", zip_code="10617"))
    assert emp.name == "Alice Chen"
    assert ep.debug_call(lambda x, y=0: x + y, 2, y=3) == 5
    assert ep.merge_dicts_new({"verb": "beats"}, {"subject": "Python"}) == {"verb": "beats", "subject": "Python"}
    close(ep.gompertz(1, 5, 1, 0), math.exp(-5), 1e-12)
    close(ep.power_mean(2, 12.87, 3.12), 9.36, 0.01)
    assert ep.fibonacci_until(100) == (144, 12)
    recurrence = ep.make_recurrence([4, 3, -18], [1, 1, 2])
    assert [int(recurrence(k)) for k in range(5)] == [1, 1, 2, -7, -40]
    assert ep.tri_closed(5) == 15
    assert ep.tri_inverse(11) == 5
    assert ep.tri_seq(5) == 15
    assert ep.pascal_I(5, 2) == 10
    assert ep.pascal_II(5, 2) == 10
    assert ep.pascal_dp(4)[4] == [1, 4, 6, 4, 1]
    inc, reset = ep.make_counter(10)
    assert inc() == 11
    assert inc(5) == 16
    reset()
    assert inc() == 11
    assert repr(ep.Car("Tesla", 2023)) == "Car(brand='Tesla', year=2023)"
    ev = ep.ElectricCar("Tesla Model 3", 75)
    assert ev.info() == "Vehicle: Tesla Model 3 | Battery: 75 kWh"
    assert ev.range_km() == 450
    summary = ep.Stats.from_csv_line("12.87, 3.12").summary()
    close(summary["AM"], 7.995, 1e-12)
    assert ep.Circle(5).area() == ep.ShapeFactory.create("circle", radius=5).area()

    assert np.array_equal(ep.sliding_window(np.arange(5), 3), [[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    assert np.allclose(ep.cal_recip([1, 2, 4]), [1, 0.5, 0.25])
    assert ep.heaviside_1(-1) == 0
    assert np.array_equal(ep.heaviside_3([-1, 0, 2]), [0, 0, 1])
    assert np.array_equal(ep.ramp([-1, 0, 2]), [0, 0, 2])
    A = np.array([[1.0], [1.0]])
    assert np.allclose(ep.projection_matrix(A), [[0.5, 0.5], [0.5, 0.5]])
    assert np.allclose(ep.project_vector(A, [2, 0]), [1, 1])

    df1 = ep.make_df("AB", [1, 2])
    df2 = ep.make_df("AB", [3])
    assert ep.concat_frames([df1, df2]).shape == (3, 2)
    missing = ep.missing_summary(pd_df := ep.pd.DataFrame({"x": [1.0, np.nan], "g": ["a", "a"]}))
    assert missing.loc["x", "missing"] == 1
    assert ep.fill_missing(pd_df, "mean")["x"].isna().sum() == 0
    left = ep.pd.DataFrame({"id": [1], "x": [2]})
    right = ep.pd.DataFrame({"id": [1], "y": [3]})
    assert ep.merge_frames(left, right, on="id").iloc[0]["y"] == 3
    assert ep.iqr(ep.pd.Series([1, 2, 3, 4])) == 1.5
    grouped = ep.group_zscore(ep.pd.DataFrame({"g": ["a", "a", "b", "b"], "x": [1, 3, 10, 14]}), "g", "x")
    assert "x_zscore" in grouped.columns
    assert "cv" in ep.group_summary(grouped, "g", "x").columns
    pivot = ep.pivot_counts(ep.pd.DataFrame({"a": ["x", "x", "y"], "b": ["m", "n", "m"]}), "a", "b")
    assert pivot.loc["x", "m"] == 1
    Xc, Yc, Zc = ep.contour_grid(lambda X, Y: X + Y, (-1, 1), (-1, 1), n=5)
    assert Xc.shape == Yc.shape == Zc.shape == (5, 5)


def test_slide_expansions() -> None:
    assert ep.conditional_expression(True, "yes", "no") == "yes"
    assert ep.try_except_else_finally_demo("12") == ("parsed:12", ["else", "finally"])
    assert ep.try_except_else_finally_demo("x") == ("invalid", ["except", "finally"])
    assert ep.loop_else_find(["a", "b"], "b") == (True, 1)
    assert ep.loop_else_find(["a", "b"], "z") == (False, None)

    np.random.seed(21)
    samples = ep.inverse_transform_discrete(["a", "b"], [0.25, 0.75], 20)
    assert set(samples).issubset({"a", "b"})
    cauchy = ep.cauchy_inverse_transform(10, mu=1, sigma=2)
    assert len(cauchy) == 10

    attrs = ep.array_attributes(np.zeros((2, 3), dtype=np.int32))
    assert attrs["ndim"] == 2
    assert attrs["shape"] == (2, 3)
    assert attrs["dtype"] == np.dtype("int32")
    assert np.array_equal(ep.int_array_float_truncation([1.9, -3.1]), [1, -3])
    stacks = ep.concatenate_stack([1, 2], [3, 4])
    assert np.array_equal(stacks["concatenate"], [1, 2, 3, 4])
    assert np.array_equal(stacks["vstack"], [[1, 2], [3, 4]])
    assert len(ep.split_array(np.arange(6), 3)) == 3
    assert np.array_equal(ep.ufunc_out_square([2, 3]), [4, 9])
    assert ep.ufunc_reduce_sum([1, 2, 3]) == 6
    nan_stats = ep.nan_safe_aggregates([1, np.nan, 3])
    assert nan_stats["nanmean"] == 2
    assert np.array_equal(ep.mask_between(np.arange(6), 2, 4), [2, 3, 4])
    assert ep.mask_count(np.arange(6), lambda x: x % 2 == 0) == 3
    unsorted = [3, 1, 2, 1]
    assert ep.selection_sort(unsorted) == [1, 1, 2, 3]
    assert ep.bubble_sort(unsorted) == [1, 1, 2, 3]
    assert ep.insertion_sort(unsorted) == [1, 1, 2, 3]
    assert ep.merge_sort(unsorted) == [1, 1, 2, 3]
    assert ep.quick_sort(unsorted) == [1, 1, 2, 3]
    assert np.allclose(ep.solve_linear_system(np.eye(2), [5, 6]), [5, 6])

    s = ep.series_from_mapping({"a": 1, "b": 2})
    assert s.loc["a"] == 1
    df = ep.pd.DataFrame({"x": [1, 2, np.nan], "g": ["a", "a", "b"]}, index=["r1", "r2", "r3"])
    profile = ep.dataframe_profile(df)
    assert profile["shape"] == (3, 2)
    loc_row, iloc_row = ep.select_loc_iloc(df, "r1", 0)
    assert loc_row["x"] == iloc_row["x"] == 1
    left = ep.pd.Series([1, 2], index=["a", "b"])
    right = ep.pd.Series([10, 20], index=["b", "c"])
    aligned = ep.index_aligned_add(left, right)
    assert np.isnan(aligned.loc["a"])
    assert ep.index_aligned_add(left, right, fill_value=0).loc["c"] == 20
    assert ep.interpolate_missing(ep.pd.Series([1.0, np.nan, 3.0])).iloc[1] == 2
    nullable = ep.nullable_int_series([1, None, 3])
    assert str(nullable.dtype) == "Int64"
    indicator = ep.add_missing_indicator(df, "x")
    assert indicator.loc["r3", "x_missing"] == 1
    rel = ep.pd.DataFrame({"A": [1, 2, 2], "B": [3, 4, 4]})
    assert list(ep.relational_project(rel, ["A"]).columns) == ["A"]
    assert ep.relational_select(rel, lambda d: d["A"] > 1).shape[0] == 2
    assert "C" in ep.relational_rename(rel, {"A": "C"}).columns
    assert ep.relational_union(rel.iloc[:2], rel.iloc[1:]).shape[0] == 2
    assert ep.relational_set_difference(rel.iloc[:2], rel.iloc[1:]).iloc[0]["A"] == 1
    cross = ep.relational_cross_product(ep.pd.DataFrame({"A": [1, 2]}), ep.pd.DataFrame({"B": [3, 4]}))
    assert cross.shape == (4, 2)
    assert ep.join_category(ep.pd.DataFrame({"id": [1, 2]}), ep.pd.DataFrame({"id": [1, 1]}), "id") == "one-to-many"

    fig, ax = ep.plt.subplots()
    ep.ax_set_labels(ax, title="T", xlabel="X", ylabel="Y", xlim=(0, 1), ylim=(0, 1))
    assert ax.get_title() == "T"
    assert ep.choose_colormap("diverging") == "coolwarm"
    assert ep.choose_plot_type("time-series", "trend") == "line"
    ep.plt.close(fig)


def main() -> None:
    test_linear_solver_and_integration()
    test_roots_gradient_and_ciphers()
    test_statistics_sampling_and_random_generators()
    test_oop_functions_and_numpy_patterns()
    test_sec_notebook_expansions()
    test_slide_expansions()
    print("all exam_prep tests passed")


if __name__ == "__main__":
    main()
