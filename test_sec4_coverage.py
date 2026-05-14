from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import slides_packages as sp


def test_hat_matrix_via_qr() -> None:
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    result = sp.hat_matrix_via_qr(X)
    assert result["Q"].shape == (3, 2)
    assert result["R"].shape == (2, 2)
    assert result["hat"].shape == (3, 3)
    assert np.allclose(result["hat"], result["Q"] @ result["Q"].T)


def test_split_rgb_channels() -> None:
    image = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    channels = sp.split_rgb_channels(image)
    assert np.array_equal(channels["red"], image[:, :, 0])
    assert np.array_equal(channels["green"], image[:, :, 1])
    assert np.array_equal(channels["blue"], image[:, :, 2])


def test_annotate_peak() -> None:
    fig, ax = plt.subplots()
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 3.0, 2.0])
    note = sp.annotate_peak(ax, x, y, text="peak")
    assert note["index"] == 1
    assert note["xy"] == (1.0, 3.0)
    plt.close(fig)


def test_save_figure() -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "fig.png"
        saved = sp.save_figure(fig, out)
        assert saved == out
        assert out.exists()
    plt.close(fig)


def test_plot_line3d_and_surface3d() -> None:
    z = np.linspace(0, 1, 5)
    line = sp.plot_line3d(np.sin(z), np.cos(z), z)
    assert line["ax"].name == "3d"
    plt.close(line["fig"])

    X, Y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    Z = X**2 + Y**2
    surf = sp.plot_surface3d(X, Y, Z)
    assert surf["ax"].name == "3d"
    plt.close(surf["fig"])


def test_plot_kde() -> None:
    df = pd.DataFrame({"x": np.linspace(-1, 1, 20), "y": np.linspace(1, -1, 20)})
    kde = sp.plot_kde(df, x="x", y="y")
    assert kde["ax"] is not None
    plt.close(kde["ax"].figure)


def main() -> None:
    test_hat_matrix_via_qr()
    test_split_rgb_channels()
    test_annotate_peak()
    test_save_figure()
    test_plot_line3d_and_surface3d()
    test_plot_kde()
    print("sec4 coverage tests passed")


if __name__ == "__main__":
    main()
