import SeqIO
import matplotlib.pyplot as plt
import pytest
import numpy

class Test4D:
    @pytest.fixture
    def ans(self):
        data = SeqIO.load_folder("testUpgrade/").sum()
        return data.data

    def test_load(self):
        data = SeqIO.load("seq4dSTEM/")
        print(data.axes_manager)

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("shape", [None, (20, 10), (21, 10)])
    def test_load_upgrade(self, ans, lazy, shape):
        data = SeqIO.load_folder("testUpgrade/",
                                 nav_shape=shape,
                                 lazy=lazy)
        numpy.testing.assert_array_almost_equal(data.sum().data,
                                                ans)
