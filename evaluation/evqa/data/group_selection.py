"""Explicit, validated packed-group selection; never silently skip samples."""
import numpy as np


def select_group_indices(n, indices_path=None, sampling_rate=1.0):
    if not 0 < sampling_rate <= 1:
        raise ValueError('sampling_rate must be in (0, 1]')
    if indices_path:
        selected = np.load(indices_path, allow_pickle=False)
        if selected.dtype != np.int64 or selected.ndim != 1 or not len(selected):
            raise ValueError('Allowed group IDs must be a nonempty int64 vector')
        if selected[0] < 0 or selected[-1] >= n or np.any(np.diff(selected) <= 0):
            raise ValueError('Allowed group IDs must be sorted, unique and in range')
    else:
        selected = np.arange(n, dtype=np.int64)
    if sampling_rate < 1:
        selected = np.random.choice(selected, int(len(selected) * sampling_rate), replace=False)
    return selected
