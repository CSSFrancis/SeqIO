import SeqIO
import pytest
import numpy as np
from hyperspy._lazy_signals import LazySignal2D


class Test4D:
    def test_load(self):
        data = SeqIO.load("seqImage/12-55-58.276.seq")
        print(data.axes_manager)

    def test_lazy_load(self):
        data = SeqIO.load("seqImage/12-55-58.276.seq", lazy=True)
        print(data.data)
        assert isinstance(data, LazySignal2D)