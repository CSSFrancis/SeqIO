import SeqIO
from hyperspy._lazy_signals import LazySignal2D
import matplotlib.pyplot as plt
from SeqIO.CeleritasSeqReader import SeqReader

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
        print("the data shape", np.shape(data.data))
        data.sum().plot()
        plt.show()

    def test_celeritas_lazy(self):
        data = SeqIO.load_celeritas(top='/media/hdd/home/1000FPS SS7 200x200/top.seq',
                                    bottom='/media/hdd/home/1000FPS SS7 200x200/bottom.seq',
                                    xml_file='/media/hdd/home/1000FPS SS7 200x200/metadata.xml',
                                    metadata='/media/hdd/home/1000FPS SS7 200x200/metadata.metadata',
                                    lazy=True,
                                    chunks=100)
        print(data)
        print(data.metadata)
        print(data.axes_manager)
        data.compute()
        data.plot()

    def test_read_xml(self):
        test1 = SeqReader(xml_file='/media/hdd/home/1000FPS SS7 200x200/1000FPS_SS7_200x200.seq.Config.Metadata.xml.Config.Metadata.xml.Config.Metadata.xml.Config.Metadata.xml')
        test1._get_xml_file()