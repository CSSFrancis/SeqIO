import glob
import logging
import struct

import numpy as np

_logger =logging.getLogger()


def get_files(folder):
    file_dict = {"top": glob.glob(folder + "/*Top*.seq"),
                 "bottom": glob.glob(folder + "/*Bottom*.seq"),
                 "seq": glob.glob(folder + "/*.seq"),
                 "gain": glob.glob(folder + "/*gain*.mrc"),
                 "dark": glob.glob(folder + "/*dark*.mrc"),
                 "xml_file": glob.glob(folder + "/*.xml"),
                 "metadata": glob.glob(folder + "/*.metadata")}
    return file_dict


def read_ref(file_name,
             height,
             width,
             ):
    """Reads a reference image from the file using the file name as well as the width and height of the image. """
    if file_name is None:
        return
    try:
        with open(file_name, mode='rb') as file:
            file.seek(0)
            read_bytes = file.read(8)
            frame_width = struct.unpack('<i', read_bytes[0:4])[0]
            frame_height = struct.unpack('<i', read_bytes[4:8])[0]
            file.seek(256 * 4)
            bytes = file.read(frame_width * frame_height * 4)
            dark_ref = np.reshape(np.frombuffer(bytes, dtype=np.float32), (height, width))
        return dark_ref
    except FileNotFoundError:
        _logger.warning("No Dark Reference image found.  The Dark reference should be in the same directory "
                        "as the image and have the form xxx.seq.dark.mrc")
        return
