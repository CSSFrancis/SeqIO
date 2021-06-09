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


_logger = logging.getLogger(__name__)

data_types = {8: np.uint8, 16: np.uint16, 32: np.uint32}  # Stream Pix data types


class SeqReader(object):
    """ Class to read .seq files. File format from StreamPix and Output for Direct Electron Cameras
    """
    def __init__(self, file):
        self.file = file
        self.metadata_dict = {}
        self.axes= []
        self.image_dict = {}
        self.dark_ref = None
        self.gain_ref = None
        self.image_dtype_list = None

    def _get_dark_ref(self):
        # The dark and gain references are saved as 32 bit floats. I can either change the image to 32 bit float
        # or change the dark and gain references to 16 bit int images.
        try:
            with open (self.file + ".dark.mrc", mode='rb') as file:
                file.seek(0)
                read_bytes = file.read(8)
                frame_width = struct.unpack('<i', read_bytes[0:4])[0]
                frame_height = struct.unpack('<i', read_bytes[4:8])[0]
                file.seek(256 * 4)
                bytes = file.read(frame_width * frame_height * 4)
                self.dark_ref = np.array(
                    np.round(
                        np.reshape(
                            np.frombuffer(bytes, dtype=np.float32),(self.image_dict["ImageWidth"],
                                                                    self.image_dict["ImageHeight"]))),
                    dtype=self.image_dict["ImageBitDepth"])
        except FileNotFoundError:
            print("No Dark Reference image found.  The Dark reference should be in the same directory "
                  "as the image and have the form xxx.seq.dark.mrc")

    def _get_gain_ref(self):
        try:
            with open (self.file+".gain.mrc", mode='rb') as file:
                file.seek(0)
                read_bytes = file.read(8)
                frame_width = struct.unpack('<i', read_bytes[0:4])[0]
                frame_height = struct.unpack('<i', read_bytes[4:8])[0]
                file.seek(256 * 4)
                bytes = file.read(frame_width * frame_height * 4)
                self.gain_ref = np.array(
                    np.round(
                        np.reshape(
                            np.frombuffer(bytes, dtype=np.float32), (self.image_dict["ImageWidth"],
                                                                     self.image_dict["ImageHeight"]))),
                    dtype=self.image_dict["ImageBitDepth"])  # Casting to 16 bit ints
        except FileNotFoundError:
            print("No gain reference image found.  The Gain reference should be in the same directory "
                  "as the image and have the form xxx.seq.gain.mrc")

    def parse_header(self):
        with open(self.file, mode='rb') as file:  # b is important -> binary
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
            print(self.image_dict["NumFrames"])

            file.seek(580)
            read_bytes = file.read(4)
            self.image_dict["ImgBytes"] = struct.unpack('<L', read_bytes[0:4])[0]

            file.seek(584)
            read_bytes = file.read(8)
            self.image_dict["FPS"] = struct.unpack('<d', read_bytes)[0]
        return

    def parse_metadata_file(self):
        """ This reads the metadata from the .metadata file """
        try:
            with open(self.file + ".metadata", 'rb') as meta:
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
            print("No metadata file.  The metadata should be in the same directory "
                  "as the image and have the form xxx.seq.metadata")
        return

    def create_axes(self, nav_shape=None, nav_names=["x","y","time"]):
        axes = []
        if nav_shape is None:
            axes.append({'name':'time', 'offset': 0, 'scale': 1, 'size': self.image_dict["NumFrames"],
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
        metadata = {'General': {'original_filename': os.path.split(self.file)[1]},
                    'Signal': {'signal_type': 'Signal2D'}}
        if self.metadata_dict is not {}:
            metadata['Acquisition_instrument'] = {'TEM':
                                                      {'camera_length': self.metadata_dict["CameraLength"],
                                                       'magnification': self.metadata_dict["Magnification"]}}
        return metadata

    def get_image_data(self):
        with open(self.file, mode='rb') as file:
            dtype_list = [(("Array"), self.image_dict["ImageBitDepth"],
                           (self.image_dict["ImageWidth"], self.image_dict["ImageHeight"]))]
            # (("t_value"),("<u4")), (("Milliseconds"), ("<u2")), (("Microseconds"), ("<u2"))]
            data = np.empty(self.image_dict["NumFrames"], dtype=dtype_list)  # creating an empty array
            max_pix = 2**self.image_dict["ImageBitDepthReal"]
            for i in range(self.image_dict["NumFrames"]):
                file.seek(8192 + i * self.image_dict["ImgBytes"])
                if self.dark_ref is not None and self.gain_ref is not None:
                    d = np.fromfile(file, dtype_list, count=1)
                    d["Array"] = (d["Array"] - self.dark_ref)
                    d["Array"][d["Array"] > max_pix] = 0
                    d["Array"] = d["Array"] * self.gain_ref  # Numpy doesn't check for overflow.
                    # There might be a better way to do this. OpenCV has a method for subtracting
                    data[i] = d
                else:
                    data[i] = np.fromfile(file, dtype_list, count=1)
        return data["Array"]

    def get_image_chunk(self, im_start, chunk_size):
        with open(self.file, mode='rb') as file:
            dtype_list = [(("Array"), self.image_dict["ImageBitDepth"],
                           (self.image_dict["ImageWidth"], self.image_dict["ImageHeight"]))]
            # (("t_value"),("<u4")), (("Milliseconds"), ("<u2")), (("Microseconds"), ("<u2"))]
            data = np.empty(chunk_size, dtype=dtype_list)  # creating an empty array
            max_pix = 2**self.image_dict["ImageBitDepthReal"]
            for i in range(chunk_size):
                start =im_start*self.image_dict["ImgBytes"]
                file.seek(8192+start + i * self.image_dict["ImgBytes"])
                if self.dark_ref is not None and self.gain_ref is not None:
                    d = np.fromfile(file, dtype_list, count=1)
                    d["Array"] = (d["Array"] - self.dark_ref)
                    d["Array"][d["Array"] > max_pix] = 0
                    d["Array"] = d["Array"] * self.gain_ref  # Numpy doesn't check for overflow.
                    # There might be a better way to do this. OpenCV has a method for subtracting
                    data[i] = d
                else:
                    data[i] = np.fromfile(file, dtype_list, count=1)
        return data["Array"]

    def read_data(self, lazy=False, chunks=None, chunksize=None, nav_shape=None):
        if lazy:
            if chunks is None and chunksize is not None:
                chunks = int(np.ceil(chunksize/(self.image_dict["ImageWidth"] *
                                self.image_dict["ImageHeight"]/2)))
            elif chunks is None and chunksize is None:
                chunks = 10
            from dask import delayed
            from dask.array import from_delayed
            from dask.array import concatenate
            from dask.array import reshape
            per_chunk = np.floor_divide(self.image_dict["NumFrames"], (chunks-1))
            extra = np.remainder(self.image_dict["NumFrames"], (chunks-1))
            chunk = [per_chunk]*(chunks-1) + [extra]

            val = [delayed(self.get_image_chunk, pure=True)
                   (per_chunk*i, chunk_size) for
                   i, chunk_size in enumerate(chunk)]
            data = [from_delayed(v,
                                 shape=(chunk_size,
                                        self.image_dict["ImageWidth"],
                                        self.image_dict["ImageHeight"]),
                                 dtype=self.image_dict["ImageBitDepth"])
                    for chunk_size, v in zip(chunk, val)]
            data = concatenate(data, axis=0)
        else:
            data = self.get_image_data()
        if nav_shape is not None:
            shape = list(nav_shape) + [self.image_dict["ImageWidth"], self.image_dict["ImageHeight"]]
            data = reshape(data, shape, )
        return data


def file_reader(filename, lazy=False, nav_shape=None, chunks=10):
    """Reads a .seq file.
    Parameters
    ----------
    filename: str
        The filename to be loaded
    lazy : bool, default False
        Load the signal lazily.
    """
    seq = SeqReader(filename)
    seq.parse_header()
    seq._get_dark_ref()
    seq._get_gain_ref()
    seq.parse_metadata_file()
    axes = seq.create_axes(nav_shape)
    metadata = seq.create_metadata()
    data = seq.read_data(lazy=lazy, chunks=chunks, nav_shape=nav_shape)
    dictionary = {
        'data': data,
        'metadata': metadata,
        'axes': axes,
        'original_metadata': metadata,}

    return dictionary