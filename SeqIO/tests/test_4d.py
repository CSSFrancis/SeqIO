import SeqIO

class Test4D:
    def test_load(self):
        data = SeqIO.load("seq4dSTEM/")
        print(data.axes_manager)