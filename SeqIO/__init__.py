from SeqIO.SeqReader import file_reader
from hyperspy._signals.signal2d import Signal2D
from hyperspy.io import dict2signal


def load(filename, lazy=False, chunks=None, parameters=None):
    """Loads a .seq file into hyperspy.  Metadata taken from
    the .metadata file as well as from a paramters.txt file that
    can be passed as well.  The parameters file is used calibrate using
    the 4-D STEM parameters for some signal.

    Parameters
    -----------
    filename: str
        The name of the file to be loaded (.seq file)

    """

    sig = dict2signal(file_reader(filename=filename, lazy=lazy, chunks=chunks),
                      lazy=lazy)
    return sig
