from __future__ import annotations

import contextlib
import io

import matplotlib

matplotlib.use("Agg")

import real_examples


def test_real_examples_main_smoke() -> None:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        real_examples.main()

    output = buffer.getvalue()
    for title in [
        "Solution to Systems of Linear Equations",
        "Numerical Integration",
        "Gradient Descend Method",
        "Encryptin Alphabetic Text",
        "Association Measure",
    ]:
        assert title in output

    for marker in ["HELLO EVERYONE", "attack at dawn", "exxego ex srgi"]:
        assert marker in output


def main() -> None:
    test_real_examples_main_smoke()
    print("real_examples smoke test passed")


if __name__ == "__main__":
    main()
