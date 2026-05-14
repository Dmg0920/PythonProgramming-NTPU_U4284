from __future__ import annotations

import slides_flow_control as sfc


def test_conditional_expression_demo() -> None:
    assert sfc.conditional_expression_demo(92) == "pass"
    assert sfc.conditional_expression_demo(59) == "fail"


def test_while_else_countdown() -> None:
    values, status = sfc.while_else_countdown(3)
    assert values == [3, 2, 1]
    assert status == "completed"


def main() -> None:
    test_conditional_expression_demo()
    test_while_else_countdown()
    print("flow control coverage tests passed")


if __name__ == "__main__":
    main()
