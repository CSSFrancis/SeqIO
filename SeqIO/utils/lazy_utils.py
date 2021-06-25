import numpy as np


def process_function_blockwise(data,
                               function,
                               nav_indexes=None,
                               output_signal_size=None,
                               block_info=None,
                               **kwargs):
    """
    Function for processing the function blockwise...
    This gets passed to map_blocks so that the function
    only gets applied to the signal axes.

    Parameters:
    ------------
    data: np.array
        The data for one chunk
    *args: tuple
        Any signal the is iterated alongside the data in. In the
        form ((key1, value1),(key2,value2)...)
    function: function
        The function to applied to the signal axis
    nav_indexes: tuple
        The indexes of the navigation axes for the dataset.
    output_signal_shape: tuple
        The shape of the output signal.  For a ragged signal
        this is equal to 1.
    block_info: dict
        The block info as described by the `dask.array.map_blocks` function
    arg_keys: tuple
        The list of keys for the passed arguments (args).  Together this makes
        a set of key:value pairs to be passed to the function.
    **kwargs: dict
        Any additional key value pairs to be used by the function
        (Note that these are the constants that are applied.)

    """
    # Both of these values need to be passed in
    dtype = block_info[None]["dtype"]
    chunk_nav_shape = tuple([data.shape[i] for i in sorted(nav_indexes)])
    output_shape = chunk_nav_shape + tuple(output_signal_size)
    # Pre-allocating the output array
    output_array = np.empty(output_shape, dtype=dtype)
    # There aren't any BaseSignals for iterating
    for nav_index in np.ndindex(chunk_nav_shape):
        islice = np.s_[nav_index]
        output_array[islice] = function(data[islice],
                                        **kwargs)
    return output_array