from SeqIO.SeqReader import file_reader
from SeqIO.CeleritasSeqReader import file_reader as c_reader
from hyperspy._signals.signal2d import Signal2D
from hyperspy.io import dict2signal
import glob
from SeqIO.version import __version__


def load_folder(folder,
                lazy=False,
                chunk_shape=None,
                nav_shape=None,
                ):
    """Loads a folder of .seq type files (i.e. metadata dark/gain references

    Parameters
    ------------
    lazy: bool
        If the dataset should be loaded into RAM or loaded lazily
    chunk_size: int or tuple
        If lazy is True --> The size of each chunk (-1 spans the full dimension).  This is only for
        the navigation axes.  (Eventually maybe change this to allow for chunking in signal axes if
        there is enough demand)
    nav_shape: tuple
        The shape of the navigation axes.  This reshapes the navigation axes
    """
    params = dict()
    params["top"] = glob.glob(folder+"*Top*.seq")
    params["bottom"] = glob.glob(folder+"*Bottom*.seq")
    params["seq"] = glob.glob(folder+"*.seq")
    params["gain"] = glob.glob(folder+"*gain*.mrc")
    params["dark"] = glob.glob(folder+"*dark*.mrc")
    params["xml_file"] = glob.glob(folder+"*.xml")
    params["metadata"] = glob.glob(folder+"*.metadata")
    if len(params["top"]) == 0 and len(params["bottom"]) == 0:
        params.pop("top")
        params.pop("bottom")
        for key in params:
            if len(params[key]) == 0:
                params[key] = None
            else:
                params[key] = params[key][0]
        s = load(**params,
                 lazy=lazy,
                 chunk_shape=chunk_shape,
                 nav_shape=nav_shape)
        # Rewrite regular .Seq loading...
    else:
        params.pop("seq")
        for key in params:
            if len(params[key]) == 0:
                params[key] = None
            else:
                params[key] = params[key][0]
        s = load_celeritas(**params,
                           lazy=lazy,
                           chunk_shape=chunk_shape,
                           nav_shape=nav_shape)
    return s


def load(seq=None,
         gain = None,
         dark=None,
         metadata=None,
         xml_file=None,
         lazy=False,
         chunk_shape=None,
         nav_shape=None,
         ):
    """Loads a .seq file into hyperspy.  Metadata taken from
    the .metadata file as well as from a paramters.txt file that
    can be passed as well.  The parameters file is used calibrate using
    the 4-D STEM parameters for some signal.

    Parameters
    -----------
    filename: str
        The name of the file to be loaded (.seq file)

    """
    sig = dict2signal(file_reader(filename=seq,
                                  dark=dark,
                                  gain=gain,
                                  metadata=metadata,
                                  xml_file=xml_file,
                                  lazy=lazy,
                                  chunk_shape=chunk_shape,
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
                   chunk_shape=None,
                   nav_shape=None,
                   ):
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
                               chunk_shape=chunk_shape,
                               nav_shape=nav_shape),
                      lazy=lazy)
    return sig