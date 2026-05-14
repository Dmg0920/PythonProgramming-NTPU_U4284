from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import exam_prep as ep


BROADCASTING_RULES: list[dict[str, str]] = [
    {"title": "Rule 1", "summary": "Pad the smaller-rank shape on the left with ones."},
    {"title": "Rule 2", "summary": "Dimensions match when equal or when one of them is 1."},
    {"title": "Rule 3", "summary": "If a dimension mismatch cannot be resolved by a 1, broadcasting fails."},
]

SORTING_COMPLEXITY_NOTES: list[dict[str, str]] = [
    {"title": "Selection sort", "summary": "Quadratic-time iterative algorithm."},
    {"title": "Bubble sort", "summary": "Quadratic-time iterative algorithm with repeated adjacent swaps."},
    {"title": "Insertion sort", "summary": "Quadratic worst-case but efficient on nearly sorted input."},
    {"title": "Merge sort", "summary": "O(n log n) divide-and-conquer stable algorithm."},
    {"title": "Quick sort", "summary": "O(n log n) average-case divide-and-conquer algorithm."},
]

PANDAS_MISSINGNESS_NOTES: list[dict[str, str]] = [
    {"title": "MCAR", "summary": "Missing completely at random."},
    {"title": "MAR", "summary": "Missing at random conditional on observed variables."},
    {"title": "MNAR", "summary": "Missing not at random even after conditioning on observed variables."},
]

PANDAS_MISSING_STRATEGIES: list[dict[str, str]] = [
    {"title": "Listwise deletion", "summary": "Drop records with any missing value."},
    {"title": "Indicator variable", "summary": "Add a binary column that marks missingness."},
    {"title": "Imputation", "summary": "Replace missing values with estimated or fixed values."},
    {"title": "Interpolation", "summary": "Estimate missing values from nearby observations."},
]

VISUALIZATION_PRINCIPLES: list[dict[str, str]] = [
    {"title": "Know your goal", "summary": "Choose visuals based on the audience and action you want."},
    {"title": "Use the right data", "summary": "Avoid statistical misuse and overloaded messaging."},
    {"title": "Select suitable visualizations", "summary": "Match encodings to data and decision tasks."},
    {"title": "Design for aesthetics", "summary": "Reduce clutter and improve legibility."},
    {"title": "Choose an effective medium", "summary": "Pick a delivery channel that fits the audience context."},
    {"title": "Check results", "summary": "Verify that the figure communicates what you intended."},
]

MARKER_REFERENCE: list[dict[str, str]] = [
    {"title": "o", "summary": "Circle marker."},
    {"title": "s", "summary": "Square marker."},
    {"title": "^", "summary": "Triangle-up marker."},
    {"title": "x", "summary": "X marker."},
]

LINE_STYLE_REFERENCE: list[dict[str, str]] = [
    {"title": "-", "summary": "Solid line."},
    {"title": "--", "summary": "Dashed line."},
    {"title": "-.", "summary": "Dash-dot line."},
    {"title": ":", "summary": "Dotted line."},
]

array_attributes = ep.array_attributes
int_array_float_truncation = ep.int_array_float_truncation
concatenate_stack = ep.concatenate_stack
split_array = ep.split_array
ufunc_out_square = ep.ufunc_out_square
ufunc_reduce_sum = ep.ufunc_reduce_sum
nan_safe_aggregates = ep.nan_safe_aggregates
mask_between = ep.mask_between
mask_count = ep.mask_count
selection_sort = ep.selection_sort
bubble_sort = ep.bubble_sort
insertion_sort = ep.insertion_sort
merge_sort = ep.merge_sort
quick_sort = ep.quick_sort
solve_linear_system = ep.solve_linear_system
Gauss_Siedel = ep.Gauss_Siedel
projection_matrix = ep.projection_matrix
project_vector = ep.project_vector
series_from_mapping = ep.series_from_mapping
dataframe_profile = ep.dataframe_profile
select_loc_iloc = ep.select_loc_iloc
index_aligned_add = ep.index_aligned_add
interpolate_missing = ep.interpolate_missing
nullable_int_series = ep.nullable_int_series
add_missing_indicator = ep.add_missing_indicator
concat_frames = ep.concat_frames
merge_frames = ep.merge_frames
join_category = ep.join_category
group_zscore = ep.group_zscore
group_summary = ep.group_summary
pivot_counts = ep.pivot_counts
plot_line = ep.plot_line
plot_scatter = ep.plot_scatter
plot_errorbar = ep.plot_errorbar
plot_confidence_band = ep.plot_confidence_band
contour_grid = ep.contour_grid
plot_histogram = ep.plot_histogram
plot_color_gradients = ep.plot_color_gradients
ax_set_labels = ep.ax_set_labels
choose_colormap = ep.choose_colormap
choose_plot_type = ep.choose_plot_type


def slice_is_view_demo() -> dict[str, Any]:
    arr = np.arange(6)
    sub = arr[1:4]
    sub[0] = -99
    return {"source": arr, "slice": sub, "shares_memory": bool(np.shares_memory(arr, sub))}

# Example:
# slice_is_view_demo()


def broadcasting_compatibility(left_shape: tuple[int, ...], right_shape: tuple[int, ...]) -> dict[str, Any]:
    max_dim = max(len(left_shape), len(right_shape))
    left = (1,) * (max_dim - len(left_shape)) + left_shape
    right = (1,) * (max_dim - len(right_shape)) + right_shape
    result: list[int] = []
    compatible = True
    for a, b in zip(left, right):
        if a == b:
            result.append(a)
        elif a == 1:
            result.append(b)
        elif b == 1:
            result.append(a)
        else:
            compatible = False
            break
    return {"left": left, "right": right, "compatible": compatible, "result_shape": tuple(result) if compatible else None}

# Example:
# broadcasting_compatibility((3, 1), (1, 4))


def dataframe_column_access(df: pd.DataFrame, column: str) -> dict[str, Any]:
    value = df[column]
    attribute_access = hasattr(df, column) and not callable(getattr(df, column))
    return {"dict_style": value, "attribute_style_available": attribute_access}

# Example:
# dataframe_column_access(pd.DataFrame({"score": [90, 80]}), "score")


def numpy_indexing_examples(arr: np.ndarray) -> dict[str, Any]:
    arr_np = np.asarray(arr)
    return {
        "first": arr_np[0],
        "last": arr_np[-1],
        "slice": arr_np[1:4],
        "every_other": arr_np[::2],
    }

# Example:
# numpy_indexing_examples(np.array([10, 20, 30, 40, 50]))


def dataframe_as_array(df: pd.DataFrame) -> np.ndarray:
    return df.to_numpy()

# Example:
# dataframe_as_array(pd.DataFrame({"x": [1, 2], "y": [3, 4]}))


def index_preserving_unary(series: pd.Series, func: Callable[[pd.Series], pd.Series | np.ndarray]) -> pd.Series:
    result = func(series)
    if isinstance(result, pd.Series):
        return result
    return pd.Series(result, index=series.index)

# Example:
# index_preserving_unary(pd.Series([1, 2, 3], index=["a", "b", "c"]), lambda s: s * 10)


def missing_data_taxonomy() -> list[dict[str, str]]:
    return PANDAS_MISSINGNESS_NOTES

# Example:
# missing_data_taxonomy()


def pyplot_vs_oo_notes() -> dict[str, str]:
    return {
        "pyplot": "Stateful interface that mimics MATLAB-style plotting.",
        "oo": "Object-oriented interface that configures plots through Figure and Axes objects.",
    }

# Example:
# pyplot_vs_oo_notes()


def plot_vs_scatter_note() -> dict[str, str]:
    return {
        "plot": "Efficient for large datasets when the same marker style is enough.",
        "scatter": "Flexible for per-point size and color at higher rendering cost.",
    }

# Example:
# plot_vs_scatter_note()


def plot_box(data: list[np.ndarray] | list[list[float]], *, labels: list[str] | None = None, ax: Any | None = None) -> Any:
    axis = ax or plt.subplots()[1]
    axis.boxplot(data, labels=labels)
    return axis

# Example:
# plot_box([[1, 2, 3], [2, 3, 4]], labels=["A", "B"])


def plot_violin(data: list[np.ndarray] | list[list[float]], *, labels: list[str] | None = None, ax: Any | None = None) -> Any:
    axis = ax or plt.subplots()[1]
    parts = axis.violinplot(data, showmeans=True, showextrema=True)
    if labels is not None:
        axis.set_xticks(np.arange(1, len(labels) + 1))
        axis.set_xticklabels(labels)
    return {"ax": axis, "parts": parts}

# Example:
# plot_violin([[1, 2, 3], [2, 3, 4]], labels=["A", "B"])


def plot_heatmap(matrix: np.ndarray, *, cmap: str = "viridis", ax: Any | None = None) -> Any:
    axis = ax or plt.subplots()[1]
    image = axis.imshow(np.asarray(matrix, dtype=float), cmap=cmap, aspect="auto")
    return {"ax": axis, "image": image}

# Example:
# plot_heatmap(np.array([[1, 2], [3, 4]]), cmap="magma")


def plot_3d_ready_grid(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    n: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return contour_grid(f, xlim, ylim, n)

# Example:
# plot_3d_ready_grid(lambda X, Y: X**2 + Y**2, (-1, 1), (-1, 1), 5)


def hat_matrix_via_qr(X: np.ndarray) -> dict[str, np.ndarray]:
    X_np = np.asarray(X, dtype=float)
    if X_np.ndim != 2:
        raise ValueError("X must be a 2-D array.")
    Q, R = np.linalg.qr(X_np)
    return {"Q": Q, "R": R, "hat": Q @ Q.T}

# Example:
# hat_matrix_via_qr(np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]))


def split_rgb_channels(image: np.ndarray) -> dict[str, np.ndarray]:
    image_np = np.asarray(image)
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3).")
    return {
        "red": image_np[:, :, 0].copy(),
        "green": image_np[:, :, 1].copy(),
        "blue": image_np[:, :, 2].copy(),
    }

# Example:
# split_rgb_channels(np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8))


def annotate_peak(ax: Any, x: np.ndarray, y: np.ndarray, *, text: str = "peak", xytext: tuple[float, float] = (12, 12)) -> dict[str, Any]:
    x_np = np.asarray(x, dtype=float)
    y_np = np.asarray(y, dtype=float)
    if x_np.ndim != 1 or y_np.ndim != 1 or len(x_np) != len(y_np) or len(x_np) == 0:
        raise ValueError("x and y must be non-empty 1-D arrays of equal length.")
    idx = int(np.argmax(y_np))
    xy = (float(x_np[idx]), float(y_np[idx]))
    annotation = ax.annotate(text, xy=xy, xytext=xytext, textcoords="offset points", arrowprops={"arrowstyle": "->"})
    return {"ax": ax, "annotation": annotation, "index": idx, "xy": xy}

# Example:
# fig, ax = plt.subplots(); annotate_peak(ax, np.array([1, 2, 3]), np.array([2, 5, 4]))


def save_figure(fig: plt.Figure, path: str | Path, *, dpi: int = 150, bbox_inches: str = "tight") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches=bbox_inches)
    return out

# Example:
# save_figure(plt.figure(), Path("/tmp/demo_plot.png"))


def plot_line3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, *, title: str = "") -> dict[str, Any]:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    line = ax.plot(np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float))
    if title:
        ax.set_title(title)
    return {"fig": fig, "ax": ax, "line": line}

# Example:
# plot_line3d(np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), title="diag")


def plot_surface3d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, *, cmap: str = "viridis", title: str = "") -> dict[str, Any]:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(np.asarray(X, dtype=float), np.asarray(Y, dtype=float), np.asarray(Z, dtype=float), cmap=cmap, antialiased=True)
    if title:
        ax.set_title(title)
    return {"fig": fig, "ax": ax, "surface": surface}

# Example:
# X, Y = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3)); plot_surface3d(X, Y, X**2 + Y**2)


def plot_kde(data: pd.DataFrame | np.ndarray, *, x: str | None = None, y: str | None = None, fill: bool = True, ax: Any | None = None) -> dict[str, Any]:
    axis = ax or plt.subplots()[1]
    if isinstance(data, pd.DataFrame):
        plot = sns.kdeplot(data=data, x=x, y=y, fill=fill, ax=axis)
    else:
        values = np.asarray(data, dtype=float)
        if values.ndim != 1:
            raise ValueError("array input must be 1-D for KDE.")
        plot = sns.kdeplot(x=values, fill=fill, ax=axis)
    return {"ax": axis, "artist": plot}

# Example:
# plot_kde(np.array([1.0, 1.5, 2.0, 2.5, 3.0]))


__all__ = [
    "BROADCASTING_RULES",
    "Gauss_Siedel",
    "LINE_STYLE_REFERENCE",
    "MARKER_REFERENCE",
    "PANDAS_MISSINGNESS_NOTES",
    "PANDAS_MISSING_STRATEGIES",
    "SORTING_COMPLEXITY_NOTES",
    "VISUALIZATION_PRINCIPLES",
    "add_missing_indicator",
    "annotate_peak",
    "array_attributes",
    "ax_set_labels",
    "broadcasting_compatibility",
    "bubble_sort",
    "choose_colormap",
    "choose_plot_type",
    "concatenate_stack",
    "concat_frames",
    "contour_grid",
    "dataframe_as_array",
    "dataframe_column_access",
    "dataframe_profile",
    "group_summary",
    "group_zscore",
    "hat_matrix_via_qr",
    "index_aligned_add",
    "index_preserving_unary",
    "insertion_sort",
    "int_array_float_truncation",
    "interpolate_missing",
    "join_category",
    "mask_between",
    "mask_count",
    "merge_frames",
    "merge_sort",
    "missing_data_taxonomy",
    "nan_safe_aggregates",
    "nullable_int_series",
    "numpy_indexing_examples",
    "pivot_counts",
    "plot_3d_ready_grid",
    "plot_box",
    "plot_color_gradients",
    "plot_confidence_band",
    "plot_errorbar",
    "plot_heatmap",
    "plot_histogram",
    "plot_kde",
    "plot_line",
    "plot_line3d",
    "plot_scatter",
    "plot_surface3d",
    "plot_violin",
    "plot_vs_scatter_note",
    "project_vector",
    "projection_matrix",
    "pyplot_vs_oo_notes",
    "quick_sort",
    "save_figure",
    "select_loc_iloc",
    "selection_sort",
    "series_from_mapping",
    "slice_is_view_demo",
    "solve_linear_system",
    "split_rgb_channels",
    "split_array",
    "ufunc_out_square",
    "ufunc_reduce_sum",
]
