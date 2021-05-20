from SeqIO.SeqReader import file_reader
from SeqIO.CeleritasSeqReader import file_reader as c_reader
from hyperspy._signals.signal2d import Signal2D
from hyperspy.io import dict2signal
import glob




def load_folder(folder,
                lazy=False,
                chunks=None,
                nav_shape=None,
                parameters=None):
    top = glob.glob(folder+"*Top*.seq")
    bottom = glob.glob(folder+"*Bottom*.seq")
    seq =  glob.glob(folder+"*.seq")
    gain = glob.glob(folder+"*gain*.mrc")
    dark = glob.glob(folder+"*dark*.mrc")
    xml = glob.glob(folder+"*.xml")
    meta = glob.glob(folder+"*.metadata")
    if len(top) == 0 and len(bottom) == 0:
        s = load(seq[0], lazy=lazy)
    else:
        s = load_celeritas(top=top[0],
                           bottom=bottom[0],
                           dark=dark[0],
                           gain=gain[0],
                           xml_file=xml[0],
                           metadata=meta[0],
                           lazy=lazy,
                           chunks=chunks,
                           nav_shape=nav_shape)
    return s

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
    lazy : bool, default False
        Load the signal lazily.
    top : str
        The filename for the top part of the detector
    bottom:
        The filename for the bottom part of the detector
    dark: str
        The filename for the dark reference to be applied to the data
    gain: str
        The filename for the gain reference to be applied to the data
    metadata: str
        The filename for the metadata file
    xml_file: str
        The filename for the xml file to be applied.
    nav_shape:
        The navigation shape for the dataset to be divided into to
    chunks:
        If lazy=True this divides the dataset into this many chunks
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