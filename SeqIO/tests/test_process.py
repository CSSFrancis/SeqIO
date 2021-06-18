from SeqIO.utils import process
import pytest

@pytest.mark.parametrize("integrate", [True,False])
@pytest.mark.parametrize("counting", [True, False])
@pytest.mark.parametrize("convolve", [True, False])
@pytest.mark.parametrize("nav_shape", [None, (20, 10), (20, 11)])
@pytest.mark.parametrize("fast_axis", [0, 1])
def test_process(integrate, counting, convolve, nav_shape, fast_axis):
    if nav_shape is None:
        fast_axis = 0
    data = process(directory='testUpgrade/',
                   integrate=integrate,
                   counting=counting,
                   convolve=convolve,
                   nav_shape=nav_shape,
                   fast_axis=fast_axis)
