"""Geometry helpers and history-window construction.

Behavior follows zhancx-edu/NeuralKernelFunction ``utils.py``. Windows are
returned as float32, which is the dtype TensorFlow uses once the arrays are
fed to the model.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import List, Tuple, Union


def quadrilateral_area(coords: List[Tuple[float, float]]) -> float:
    """Area of a quadrilateral given in order around the boundary."""
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    x4, y4 = coords[3]
    area = abs(
        x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1
        - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1)
    ) / 2
    return float(area)


def azimuthal_equidistant_projection(
    latitude: Union[float, np.ndarray],
    longitude: Union[float, np.ndarray],
    center_latitude: float,
    center_longitude: float,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Geographic coordinates to a local azimuthal equidistant plane, in km."""
    radius_km = 6371.0
    phi1 = np.radians(center_latitude)
    phi2 = np.radians(latitude)
    delta_lambda = np.radians(longitude - center_longitude)

    delta_sigma = np.arccos(
        np.sin(phi1) * np.sin(phi2)
        + np.cos(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    )
    azimuth = np.arctan2(
        np.sin(delta_lambda),
        np.cos(phi1) * np.tan(phi2) - np.sin(phi1) * np.cos(delta_lambda),
    )
    x = radius_km * delta_sigma * np.cos(azimuth)
    y = radius_km * delta_sigma * np.sin(azimuth)
    return x, y


def _windows(values: np.ndarray, time_step: int) -> np.ndarray:
    """History windows ``values[i:i+time_step]`` for ``i`` in ``range(n-time_step)``."""
    view = sliding_window_view(np.asarray(values), time_step)[:-1]
    return np.ascontiguousarray(view).reshape(-1, time_step, 1)


def process_data(
    time_step: int,
    times: np.ndarray,
    mags: np.ndarray,
    locs_x: np.ndarray,
    locs_y: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """Build the history / current-event arrays used by ``KernelPointProcess``.

    For each target event, the history is the preceding ``time_step`` events.
    ``dis_t1`` is the age of each history event at the target time.
    ``dis_t2`` is the age of each history event except the most recent one,
    measured at the previous target. The zero-lag term ``F(0)`` is omitted
    because both the Omori CDF and the bias-free temporal network satisfy
    ``F(0) = 0``.

    Returns float32 arrays:
    ``hist_t, hist_m, hist_x, hist_y, cur_t, cur_x, cur_y, dis_xy, dis_t1,
    dis_t2, elapsed_time``.
    """
    times = np.asarray(times, dtype=np.float64)
    mags = np.asarray(mags, dtype=np.float64)
    locs_x = np.asarray(locs_x, dtype=np.float64)
    locs_y = np.asarray(locs_y, dtype=np.float64)
    n = int(mags.shape[0])
    if time_step < 1 or n <= time_step:
        raise ValueError(
            f"Need more events than time_step (n={n}, time_step={time_step})."
        )

    hist_t = _windows(times, time_step)
    hist_m = _windows(mags, time_step)
    hist_x = _windows(locs_x, time_step)
    hist_y = _windows(locs_y, time_step)

    n_out = n - time_step
    cur_t = times[time_step:].reshape(n_out, 1, 1)
    cur_x = locs_x[time_step:].reshape(n_out, 1, 1)
    cur_y = locs_y[time_step:].reshape(n_out, 1, 1)

    dis_xy = ((cur_x - hist_x) ** 2 + (cur_y - hist_y) ** 2) ** 0.5
    dis_t1 = cur_t - hist_t
    dis_t2 = hist_t[:, -1:, :] - hist_t[:, :-1, :]
    elapsed_time = np.diff(times[time_step - 1 :]).reshape(n_out, 1, 1)

    def f32(array: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(array, dtype=np.float32)

    return (
        f32(hist_t),
        f32(hist_m),
        f32(hist_x),
        f32(hist_y),
        f32(cur_t),
        f32(cur_x),
        f32(cur_y),
        f32(dis_xy),
        f32(dis_t1),
        f32(dis_t2),
        f32(elapsed_time),
    )
