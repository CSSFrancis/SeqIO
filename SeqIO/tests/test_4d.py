import SeqIO
import matplotlib.pyplot as plt

class Test4D:
    def test_load(self):
        data = SeqIO.load("seq4dSTEM/")
        print(data.axes_manager)

    def test_load_upgrade(self):
        data = SeqIO.load_folder("testUpgrade/")
        print(data.axes_manager)
        data.axes_manager[0].scale=1
        data.plot()
        data.sum().plot()
        plt.show()