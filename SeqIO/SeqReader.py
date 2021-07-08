# Plugin characteristics
# ----------------------
format_name = 'seq sequential file'
description = """The file format used by StreamPix
and an output for Direct Electron detectors"""
full_support = False
# Recognised file extension
file_extensions = ('seq')
default_extension = 0
# Reading capabilities
reads_images = True
reads_spectrum = False
reads_spectrum_image = True
# Writing capabilities
writes = False

import os
import logging
import numpy as np
from hyperspy.docstrings.signal import OPTIMIZE_ARG
import struct
from SeqIO.utils.file_utils import read_ref
import dask.array as da


_logger = logging.getLogger(__name__)

data_types = {8: np.uint8, 16: np.uint16, 32: np.uint32}  # Stream Pix data types


class SeqReader(object):
    """ Class to read .seq files. File format from StreamPix and Output for Direct Electron Cameras
    """
    def __init__(self,
                 seq=None,
                 dark=None,
                 gain=None,
                 metadata=None,
                 xml_file=None):
        self.dark_file = dark
        self.gain_file = gain
        self.xml_file = xml_file
        self.metadata_file = metadata
        self.seq=seq
        self.metadata_dict = {}
        self.axes= []
        self.image_dict = {}
        self.dark_ref = None
        self.gain_ref = None
        self.image_dtype_list = None
        self.dtype_full_list=None

    def parse_header(self):
        with open(self.seq, mode='rb') as file:  # b is important -> binary
            file.seek(548)
            image_info_dtype = [(("ImageWidth"),("<u4")),
                                (("ImageHeight"), ("<u4")),
                                (("ImageBitDepth"), ("<u4")),
                                (("ImageBitDepthReal"), ("<u4"))]
            image_info = np.fromfile(file, image_info_dtype, count=1)[0]
            self.image_dict['ImageWidth'] = image_info[0]
            self.image_dict['ImageHeight'] = image_info[1]
            self.image_dict['ImageBitDepth'] = data_types[image_info[2]]  # image bit depth
            self.image_dict["ImageBitDepthReal"] = image_info[3]  # actual recorded bit depth
            self.image_dict["FrameLength"] = image_info[0] * image_info[1]
            _logger.info('Each frame is %i x %i pixels', (image_info[0], image_info[1]))
            file.seek(572)

            read_bytes = file.read(4)
            self.image_dict["NumFrames"] = struct.unpack('<i', read_bytes)[0]
            _logger.info('%i number of frames found', self.image_dict["NumFrames"])
            _logger.info(self.image_dict["NumFrames"])

            file.seek(580)
            read_bytes = file.read(4)
            self.image_dict["ImgBytes"] = struct.unpack('<L', read_bytes[0:4])[0]

            file.seek(584)
            read_bytes = file.read(8)
            self.image_dict["FPS"] = struct.unpack('<d', read_bytes)[0]
            self.dtype_full_list = [(("Array"),
                                     self.image_dict["ImageBitDepth"],
                                     (self.image_dict["ImageWidth"],
                                      self.image_dict["ImageHeight"]))]
        return

    def parse_metadata_file(self):
        """ This reads the metadata from the .metadata file """
        try:
            with open(self.seq + ".metadata", 'rb') as meta:
                meta.seek(320)
                image_info_dtype = [(("SensorGain"), (np.float64)),
                                    (("Magnification"), (np.float64)),
                                    (("PixelSize"), (np.float64)),
                                    (("CameraLength"), (np.float64)),
                                    (("DiffPixelSize"), (np.float64))]
                m = np.fromfile(meta, image_info_dtype, count=1)[0]
                self.metadata_dict["SensorGain"] = m[0]
                self.metadata_dict["Magnification"] = m[1]
                self.metadata_dict["PixelSize"] = m[2]
                self.metadata_dict["CameraLength"] = m[3]
                self.metadata_dict["DiffPixelSize"] = m[4]

        except FileNotFoundError:
            _logger.info("No metadata file.  The metadata should be in the same directory "
                         "as the image and have the form xxx.seq.metadata")
        return

    def create_axes(self, nav_shape=None, nav_names=["x", "y", "time"]):
        axes = []
        if nav_shape is None:
            axes.append({'name': 'time', 'offset': 0, 'scale': 1, 'size': self.image_dict["NumFrames"],
                        'navigate': True, 'index_in_array': 0})
            axes[0]['scale'] = 1 / self.image_dict["FPS"]
        else:
            for i, s in enumerate(nav_shape):
                axes.append({'name': nav_names[i], 'offset': 0, 'scale': 1, 'size': s,
                             'navigate': True, 'index_in_array': 0})
        axes.append({'name': 'ky', 'offset': 0, 'scale': 1, 'size': self.image_dict["ImageHeight"],
                     'navigate': False, 'index_in_array': 1})
        axes.append({'name': 'kx', 'offset': 0, 'scale': 1, 'size': self.image_dict["ImageWidth"],
                        'navigate': False, 'index_in_array': 2})

        if self.metadata_dict is not {} and self.metadata_dict["PixelSize"] != 0:
            # need to still determine a way to properly set units and scale
            axes[-2]['scale'] = self.metadata_dict["PixelSize"]
            axes[-1]['scale'] = self.metadata_dict["PixelSize"]
        return axes

    def create_metadata(self):
        metadata = {'General': {'original_filename': os.path.split(self.seq)[1]},
                    'Signal': {'signal_type': 'Signal2D'}}
        if self.metadata_dict is not {}:
            metadata['Acquisition_instrument'] = {'TEM':
                                                      {'camera_length': self.metadata_dict["CameraLength"],
                                                       'magnification': self.metadata_dict["Magnification"]}}
        return metadata

    def get_image_data(self):
        with open(self.seq, mode='rb') as file:
            dtype_list = [(("Array"), self.image_dict["ImageBitDepth"],
                           (self.image_dict["ImageWidth"], self.image_dict["ImageHeight"]))]
            # (("t_value"),("<u4")), (("Milliseconds"), ("<u2")), (("Microseconds"), ("<u2"))]
            data = np.empty(self.image_dict["NumFrames"], dtype=dtype_list)  # creating an empty array
            max_pix = 2**self.image_dict["ImageBitDepthReal"]
            for i in range(self.image_dict["NumFrames"]):
                file.seek(8192 + i * self.image_dict["ImgBytes"])
                d = np.fromfile(file, dtype_list, count=1)
                if self.dark_ref is not None:
                    d["Array"] = (d["Array"] - self.dark_ref)
                    d["Array"][d["Array"] > max_pix] = 0
                if self.gain_ref is not None:
                    d["Array"] = d["Array"] * self.gain_ref
                data[i] = d
            self.dtype_list = [(("Array"), self.image_dict["ImageBitDepth"],
                               (self.image_dict["ImageWidth"], self.image_dict["ImageHeight"]))]
        return data["Array"]

    def get_image_chunk(self,
                        chunk_indexes):
        with open(self.seq, mode='rb') as file:
            # (("t_value"),("<u4")), (("Milliseconds"), ("<u2")), (("Microseconds"), ("<u2"))]
            chunk_shape = np.shape(chunk_indexes)
            data = np.empty(chunk_shape,
                            dtype=self.dtype_full_list)  # creating an empty array
            max_pix = 2**12
            for chunk_ind, ind in np.ndenumerate(chunk_indexes):
                file.seek(8192 + ind * self.image_dict["ImgBytes"])
                d = np.fromfile(file,
                                self.dtype_full_list,
                                count=1)
                new_d = np.empty(1, dtype=self.dtype_full_list)
                try:
                    dtemp = d["Array"][0]
                    if self.dark_ref is not None:
                        dtemp = (dtemp - self.dark_ref)
                        dtemp[dtemp < 0] = 0
                        dtemp[dtemp > max_pix] = 0
                    if self.gain_ref is not None:
                        dtemp = dtemp * self.gain_ref  # Numpy doesn't check for overflow.
                        # There might be a better way to do this. OpenCV has a method for subtracting
                    new_d["Array"] = dtemp
                except IndexError:
                    _logger.info(msg="Adding a Frame")
                data[chunk_ind] = new_d
        return data["Array"]

    def get_chunk_index(self,
                        chunk_shape,
                        nav_shape):
        if np.prod(nav_shape) != self.image_dict["NumFrames"] and nav_shape is not None:
            num_frames = np.prod(nav_shape)
        else:
            num_frames = self.image_dict["NumFrames"]
        indexes = da.arange(num_frames)
        if nav_shape is not None:
            indexes = da.reshape(indexes, nav_shape)
        indexes = da.rechunk(indexes, chunks=chunk_shape)
        return indexes

    def read_data(self,
                  lazy=False,
                  chunk_shape=None,
                  nav_shape=None,
                  ):
        """Reads the data from the file provided.

        Parameters:
        lazy: bool
            If the data should be loaded lazily
        chunk_shape: "str" or int
            The number (or chunk size to read in) if None then the chunk size is ~100mb
        """
        if lazy:
            if chunk_shape is None:
                img_bytes = (self.image_dict["ImageWidth"] *
                             self.image_dict["ImageHeight"]) * 2
                chunk_shape = int(np.ceil(100000000/img_bytes))
            indexes = self.get_chunk_index(chunk_shape=chunk_shape,
                                           nav_shape=nav_shape,
                                           )
            chunks = indexes.chunks + (self.image_dict["ImageHeight"],
                                       self.image_dict["ImageWidth"])
            if nav_shape is not None:
                new_axis = (len(nav_shape), len(nav_shape)+1)
            else:
                new_axis = (1, 2)
            data = indexes.map_blocks(self.get_image_chunk,
                                      chunks=chunks,
                                      new_axis=new_axis,
                                      dtype=np.float32)
        else:
            data = self.get_image_data()
        return data


def file_reader(filename,
                dark=None,
                gain=None,
                metadata=None,
                xml_file=None,
                lazy=False,
                nav_shape=None,
                chunk_shape=None):
    """Reads a .seq file.
    Parameters
    ----------
    filename: str
        The filename to be loaded
    lazy : bool, default False
        Load the signal lazily.
    nav_shape: tuple
        The shape of the navigation axis to be loaded
    chunk_shape: tuple
        The shape for each chunk to be loaded. This has some performance implications when dealing with
        the data later...
    """
    seq = SeqReader(filename,
                    dark,
                    gain,
                    metadata,
                    xml_file)
    seq.parse_header()
    seq.dark_ref = read_ref(dark,
                            height=seq.image_dict["ImageHeight"],
                            width=seq.image_dict["ImageWidth"])
    seq.gain_ref = read_ref(gain,
                            height=seq.image_dict["ImageHeight"],
                            width=seq.image_dict["ImageWidth"])
    seq.parse_metadata_file()
    axes = seq.create_axes(nav_shape)
    metadata = seq.create_metadata()
    data = seq.read_data(lazy=lazy,
                         chunk_shape=chunk_shape,
                         nav_shape=nav_shape)
    dictionary = {
        'data': data,
        'metadata': metadata,
        'axes': axes,
        'original_metadata': metadata,
    }

    return dictionary