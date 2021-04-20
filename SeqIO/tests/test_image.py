import SeqIO
from hyperspy._lazy_signals import LazySignal2D
import matplotlib.pyplot as plt

class Test4D:
    def test_load(self):
        data = SeqIO.load("seqImage/12-55-58.276.seq", nav_shape=[4,5])
        print(data.axes_manager)

    def test_lazy_load(self):
        data = SeqIO.load("seqImage/12-55-58.276.seq", lazy=True)
        print(data.data)
        assert isinstance(data, LazySignal2D)

    def test_lazy_load_chunks(self):
        data = SeqIO.load("seqImage/12-55-58.276.seq", lazy=True, chunks=4)
        print(data.data)
        assert isinstance(data, LazySignal2D)

    def test_lazy_load_chunks_nav(self):
        data = SeqIO.load("seqImage/12-55-58.276.seq", lazy=True, chunks=4, nav_shape=(4, 5))
        print(data)
        assert isinstance(data, LazySignal2D)
    def test_celeritas(self):
        import numpy as np
        data = SeqIO.load_celeritas(top='/media/hdd/home/PtNW_100fps_Top_16-32-11.473.seq',
                                    bottom='/media/hdd/home/PtNW_100fps_Bottom_16-32-11.508.seq')
        print(np.shape(data.data))
        data.plot()
        plt.show()
    def test_celeritas_lazy(self):
        import numpy as np
        data = SeqIO.load_celeritas(top='/media/hdd/home/PtNW_100fps_Top_16-32-11.473.seq',
                                    bottom='/media/hdd/home/PtNW_100fps_Bottom_16-32-11.508.seq',
                                    lazy=True,
                                    chunks=10)
        print(data)
        print(data.data)
        data.compute()
