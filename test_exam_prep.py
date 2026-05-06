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


def main() -> None:
    test_linear_solver_and_integration()
    test_roots_gradient_and_ciphers()
    test_statistics_sampling_and_random_generators()
    test_oop_functions_and_numpy_patterns()
    print("all exam_prep tests passed")


if __name__ == "__main__":
    main()
