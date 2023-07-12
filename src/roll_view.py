import numpy as np

def roll_view(arr, shift, axis=None):
    if axis is None:
        arr = arr.flatten()
        axis = 0

    axis = np.core.multiarray.normalize_axis_index(axis, arr.ndim)

    # Calculate the effective shift value within the range of the array size
    size = arr.shape[axis]
    shift %= size

    # Create a view with rolled elements
    rolled_view = np.take(arr, np.arange(size - shift, size), axis=axis)
    rolled_view = np.concatenate((rolled_view, np.take(arr, np.arange(size - shift), axis=axis)), axis=axis)

    return rolled_view