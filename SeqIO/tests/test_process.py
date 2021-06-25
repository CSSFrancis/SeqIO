from SeqIO.utils.process_utils import process
import pytest
import numpy as np


@pytest.mark.parametrize("integrate", [True,False])
@pytest.mark.parametrize("counting", [True, False])
@pytest.mark.parametrize("nav_shape", [None, (20, 10), (20, 11)])
@pytest.mark.parametrize("chunk_shape", [None, (10, 1), (1, 10)])
def test_process(integrate, counting, nav_shape, chunk_shape):
    if nav_shape is None:
        chunk_shape = None
    data = process(directory='testUpgrade/',
                   integrate=integrate,
                   counting=counting,
                   nav_shape=nav_shape,
                   chunk_shape=chunk_shape,
                   verbose=True)
    if chunk_shape is not None:
        assert ((data.data.chunks[0][0], data.data.chunks[1][0]) ==
                chunk_shape)
    if integrate or not counting:
        assert data.data.dtype is np.dtype(np.float32)
    else:
        assert data.data.dtype is np.dtype(bool)
    if nav_shape:
        assert tuple(reversed(data.axes_manager.navigation_shape)) == nav_shape
