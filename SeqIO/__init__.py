from SeqIO.SeqReader import file_reader
from SeqIO.CeleritasSeqReader import file_reader as c_reader
from hyperspy._signals.signal2d import Signal2D
from hyperspy.io import dict2signal


def load(filename=None,lazy=False, chunks=None, nav_shape=None, parameters=None):
    """Loads a .seq file into hyperspy.  Metadata taken from
    the .metadata file as well as from a paramters.txt file that
    can be passed as well.  The parameters file is used calibrate using
    the 4-D STEM parameters for some signal.

    Parameters
    -----------
    filename: str
        The name of the file to be loaded (.seq file)

    """
    sig = dict2signal(file_reader(filename=filename,
                                  lazy=lazy,
                                  chunks=chunks,
                                  nav_shape=nav_shape),
                      lazy=lazy)
    return sig

def load_celeritas(top,
                   bottom,
                   dark=None,
                   gain=None,
                   metadata=None,
                   xml_file=None,
                   lazy=False,
                   chunks=None,
                   nav_shape=None,
                   parameters=None):
    """Loads a .seq file into hyperspy.  Metadata taken from
    the .metadata file as well as from a paramters.txt file that
    can be passed as well.  The parameters file is used calibrate using
    the 4-D STEM parameters for some signal.

    Parameters
    -----------
    filename: str
        The name of the file to be loaded (.seq file)

    """
    sig = dict2signal(c_reader(top=top,
                               bottom=bottom,
                               gain=gain,
                               dark=dark,
                               metadata=metadata,
                               xml_file=xml_file,
                               lazy=lazy,
                               chunks=chunks,
                               nav_shape=nav_shape),
                      lazy=lazy)
    return sig