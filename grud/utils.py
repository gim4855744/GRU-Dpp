from typing import Literal

import pandas as pd
import numpy as np


def ffill(
    x: np.ndarray
) -> np.ndarray:

    """
    Forward fill time series.

    Parameters
    ----------
    x(np.ndarray):
        Time series to be filled.
        Shape(n_times, n_features).

    Returns
    -------
    Filled time series.
    """
    
    n_times, n_features = x.shape
    mask = ~np.isnan(x)

    idx = np.arange(n_times)
    idx = idx.reshape(-1, 1)
    idx = np.where(mask, idx, 0)
    idx = np.maximum.accumulate(idx, axis=0)

    np.linalg.eig()

    return x[idx, np.arange(n_features)]


def get_interval(
    x: np.ndarray,
    time_stamps: np.ndarray = None
) -> np.ndarray:

    """
    Get intervals of time stamps.

    Parameters
    ----------
    x(np.ndarray):
        Time series to get intervals.
        Shape(n_times, n_features).
    time_stamps(np.ndarray, optional):
        Time stamps of each time steps.
        Shape(n_times, n_features).

    Returns
    -------
    Intervals between time steps.
    """

    n_times, n_features = x.shape
    mask = ~np.isnan(x)
    interval0 = np.zeros(n_features)
    interval = [interval0]

    for t in range(n_times - 1):
        prev_interval = interval[t]
        if time_stamps is None:
            next_interval = np.where(mask[t], 1, prev_interval + 1)
        else:
            gap = time_stamps[t + 1] - time_stamps[t]
            next_interval = np.where(mask[t], gap, prev_interval + gap)
        interval.append(next_interval)

    return np.array(interval)


def prepare_data(
    data: pd.DataFrame,
    label_col: str,
    task: Literal['many2one', 'many2many'],
    time_stamps: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Prepare data for GRU-D++.

    Parameters
    ----------
    data(pd.DataFrame):
        Total dataset.
        Index must be set and each index indicates same subject.
    label_col: Name of label column.
    task:
        Indicator of output task.
        Must be one of many2one or many2many.
    time_stamps(np.ndarray, optional):
        Time stamps of each time steps.
        Shape(n_times, n_features).

    Returns
    -------
    Tuple of input features, mask of features, last observations of features, mask of last observations, intervals, and label.
    """

    x, x_mask, last_obs, last_obs_mask, interval, y = [], [], [], [], [], []

    for idx, v in data.groupby(data.index.name):

        x_i = v
        y_i = v.pop(label_col)

        x_i = x_i.values
        if task == 'many2one':
            y_i = y_i.tail(1).values.reshape(-1, 1)
        else:
            y_i = y_i.values.reshape(-1, 1)

        x_mask_i = ~np.isnan(x_i)
        last_obs_i = ffill(x_i)
        last_obs_mask_i = ~np.isnan(last_obs_i)
        interval_i = get_interval(x_i, time_stamps)

        x.append(x_i)
        x_mask.append(x_mask_i)
        last_obs.append(last_obs_i)
        last_obs_mask.append(last_obs_mask_i)
        interval.append(interval_i)

    return x, x_mask, last_obs, last_obs_mask, interval, y
