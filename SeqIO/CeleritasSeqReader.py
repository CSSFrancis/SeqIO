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

import xmltodict


_logger = logging.getLogger(__name__)

data_types = {8: np.uint8, 16: np.uint16, 32: np.uint32}  # Stream Pix data types


class SeqReader(object):
    """ Class to read .seq files. File format from StreamPix and Output for Direct Electron Cameras
    """
    def __init__(self,
                 top=None,
                 bottom=None,
                 dark=None,
                 gain=None,
                 metadata=None,
                 xml_file=None):
        self.top = top
        self.bottom = bottom
        self.metadata_dict = {}
        self.axes = []
        self.image_dict = {}
        self.dark_file = dark
        self.gain_file = gain
        self.xml_file = xml_file
        self.metadata_file = metadata
        self.dark_ref = None
        self.gain_ref = None
        self.xml_metadata={}
        self.segment_prebuffer=None
        self.image_dtype_full_list = None
        self.image_dtype_split_list = None

    def _get_dark_ref(self):
        # The dark and gain references are saved as 32 bit floats. I can either change the image to 32 bit float
        # or change the dark and gain references to 16 bit int images.
        if self.dark_file is None:
            return
        try:
            with open(self.dark_file, mode='rb') as file:
                file.seek(0)
                read_bytes = file.read(8)
                frame_width = struct.unpack('<i', read_bytes[0:4])[0]
                frame_height = struct.unpack('<i', read_bytes[4:8])[0]
                file.seek(256 * 4)
                bytes = file.read(frame_width * frame_height * 4)
                self.dark_ref = np.array(
                    np.round(
                        np.reshape(
                            np.frombuffer(bytes, dtype=np.float32), (self.image_dict["ImageWidth"],
                                                                     self.image_dict["ImageHeight"]*2))),
                    dtype=self.image_dict["ImageBitDepth"])
        except FileNotFoundError:
            print("No Dark Reference image found.  The Dark reference should be in the same directory "
                  "as the image and have the form xxx.seq.dark.mrc")

    def _get_gain_ref(self):
        if self.gain_file is None:
            return
        try:
            with open (self.gain_file, mode='rb') as file:
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
                                                                     self.image_dict["ImageHeight"]*2))),
                    dtype=self.image_dict["ImageBitDepth"])  # Casting to 16 bit ints
        except FileNotFoundError:
            print("No gain reference image found.  The Gain reference should be in the same directory "
                  "as the image and have the form xxx.seq.gain.mrc")

    def _get_xml_file(self):
        if self.xml_file is None:
            print("No xml file is given. While the program might still run it is important to have "
                  "and xml file.  At faster FPS the XML file will contain information necessary to"
                  "properly load the images")
            return
        with open(self.xml_file, "r") as file:
            dict = xmltodict.parse(file.read())
            file_info = dict["Configuration"]["FileInfo"]
            for key in file_info:
                file_info[key] = file_info[key]["@Value"]
            self.segment_prebuffer = int(file_info["SegmentPreBuffer"])  # divide frames by this
            self.image_dict["NumFrames"] = int(file_info["TotalFrames"])
            self.xml_metadata = file_info
            return


    def parse_header(self):
        with open(self.top, mode='rb') as file:  # b is important -> binary
            file.seek(548)
            image_info_dtype = [(("ImageWidth"), ("<u4")),
                                (("ImageHeight"), ("<u4")),
                                (("ImageBitDepth"), ("<u4")),
                                (("ImageBitDepthReal"), ("<u4"))]
            image_info = np.fromfile(file, image_info_dtype, count=1)[0]
            self.image_dict['ImageWidth'] = int(image_info[0])
            self.image_dict['ImageBitDepth'] = data_types[image_info[2]]  # image bit depth
            self.image_dict["ImageBitDepthReal"] = image_info[3]  # actual recorded bit depth
            self.image_dict["FrameLength"] = image_info[0] * image_info[0]
            _logger.info('Each frame is %i x %i pixels', (image_info[0], image_info[0]))
            file.seek(572)
            file.seek(580)
            read_bytes = file.read(4)
            if self.segment_prebuffer is None:
                print("Trying to guess the segment pre-factor... Please load .xml File to help")
                # try to guess it?
                if self.image_dict["ImageWidth"] == 512:
                    self.segment_prebuffer = 16
                elif self.image_dict["ImageWidth"] == 256:
                    self.segment_prebuffer = 64
                else:
                    self.segment_prebuffer = 4
            self.image_dict['ImageHeight'] = int(image_info[1]/self.segment_prebuffer)
            self.image_dict["ImgBytes"] = int(struct.unpack('<L', read_bytes[0:4])[0]/self.segment_prebuffer-128)
            self.image_dict["GroupingBytes"] = int(struct.unpack('<L', read_bytes[0:4])[0])
            if "NumFrames" not in self.image_dict:
                print("Guessing the number of frames")
                self.image_dict["NumFrames"] = int((os.path.getsize(self.top)-8192) /
                                                   (self.image_dict["GroupingBytes"]/self.segment_prebuffer))
            _logger.info('%i number of frames found', self.image_dict["NumFrames"])

            file.seek(584)
            read_bytes = file.read(8)
            self.image_dict["FPS"] = struct.unpack('<d', read_bytes)[0]
            self.dtype_full_list = [(("Array"),
                                    self.image_dict["ImageBitDepth"],
                                    (int(self.image_dict["ImageHeight"]*2),
                                     self.image_dict["ImageWidth"]))]
            self.dtype_split_list = [(("Array"),
                                     self.image_dict["ImageBitDepth"],
                                     (self.image_dict["ImageHeight"],
                                     self.image_dict["ImageWidth"]))]
        return

    def parse_metadata_file(self):
        """ This reads the metadata from the .metadata file """
        if self.metadata_file is None:
            return
        try:
            with open(self.metadata_file, 'rb') as meta:
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
        axes.append({'name': 'ky', 'offset': 0, 'scale': 1, 'size': self.image_dict["ImageHeight"]*2,
                     'navigate': False, 'index_in_array': 1})
        axes.append({'name': 'kx', 'offset': 0, 'scale': 1, 'size': self.image_dict["ImageWidth"],
                        'navigate': False, 'index_in_array': 2})
        if self.metadata_dict != {} and self.metadata_dict["PixelSize"] > 1e-30:
            # need to still determine a way to properly set units and scale
            axes[-2]['scale'] = self.metadata_dict["PixelSize"]
            axes[-1]['scale'] = self.metadata_dict["PixelSize"]
        return axes

    def create_metadata(self):
        metadata = {'General': {'original_top_filename': os.path.split(self.top)[1],
                                'original_bottom_filename': os.path.split(self.bottom)[1]},
                    'Signal': {'signal_type': 'Signal2D'}}
        if self.metadata_dict != {}:
            metadata['Acquisition_instrument'] = {'TEM':
                                                      {'camera_length': self.metadata_dict["CameraLength"],
                                                       'magnification': self.metadata_dict["Magnification"]}}
        if self.xml_metadata != {}:
            metadata['General']["ImageData"] = self.xml_metadata
        return metadata

    def get_image_data(self):
        with open(self.top, mode='rb') as top, open(self.bottom, mode='rb') as bottom:
            # (("t_value"),("<u4")), (("Milliseconds"), ("<u2")), (("Microseconds"), ("<u2"))]
            data = np.empty(self.image_dict["NumFrames"], dtype=self.dtype_full_list)  # creating an empty array
            max_pix = 2 ** 12
            for i in range(self.image_dict["NumFrames"]):
                group = int(np.true_divide(i, self.segment_prebuffer))
                top.seek(8192 + group*self.image_dict["GroupingBytes"] +
                         int(i-group*self.segment_prebuffer) * self.image_dict["ImgBytes"])
                bottom.seek(8192 + group*self.image_dict["GroupingBytes"] +
                         int(i-group*self.segment_prebuffer) * self.image_dict["ImgBytes"])
                t = np.fromfile(top,
                                self.dtype_split_list,
                                count=1)
                b = np.fromfile(bottom,
                                self.dtype_split_list,
                                count=1)
                d = np.concatenate((np.flip(t["Array"][0],
                                            axis=0),
                                    b["Array"][0]),
                                   axis=0)
                if self.dark_ref is not None:
                    d = (d - self.dark_ref)
                    d[d > max_pix] = 0
                if self.gain_ref is not None:
                    d = d * self.gain_ref  # Numpy doesn't check for overflow.
                    # There might be a better way to do this. OpenCV has a method for subtracting
                new_d = np.empty(1, dtype=self.dtype_full_list)
                new_d["Array"] = d
                data[i] = new_d
        return data["Array"]


    def get_single_image_data(self, im_start, chunk_size):
        with open(self.top, mode='rb') as top, open(self.bottom, mode='rb') as bottom:
            # (("t_value"),("<u4")), (("Milliseconds"), ("<u2")), (("Microseconds"), ("<u2"))]
            data = np.empty(chunk_size,
                            dtype=self.dtype_full_list)  # creating an empty array
            max_pix = 2**12
            for i in range(chunk_size):
                temp_i = im_start+i
                group, rem = np.divmod(temp_i, self.segment_prebuffer, dtype=int)
                top.seek(8192 +
                         group*self.image_dict["GroupingBytes"] +
                         rem * self.image_dict["ImgBytes"])
                bottom.seek(8192 +
                         group * self.image_dict["GroupingBytes"] +
                         rem * self.image_dict["ImgBytes"])
                t = np.fromfile(top,
                                self.dtype_split_list,
                                count=1)
                b = np.fromfile(bottom,
                                self.dtype_split_list,
                                count=1)
                d = np.concatenate((np.flip(t["Array"][0], axis=0), b["Array"][0]), axis=0)
                if self.dark_ref is not None:
                    d = (d - self.dark_ref)
                    d[d > max_pix] = 0
                if self.gain_ref is not None:
                    d = d * self.gain_ref  # Numpy doesn't check for overflow.
                    # There might be a better way to do this. OpenCV has a method for subtracting
                new_d = np.empty(1, dtype=self.dtype_full_list)
                new_d["Array"] = d
                data[i] = new_d
        return data["Array"]

    def read_data(self, lazy=False, chunks=None, nav_shape=None):
        if lazy:
            if chunks is None:
                chunks =10
            from dask import delayed
            from dask.array import from_delayed
            from dask.array import concatenate
            per_chunk = np.floor_divide(self.image_dict["NumFrames"], (chunks-1))
            extra = np.remainder(self.image_dict["NumFrames"], (chunks-1))
            chunk = [per_chunk]*(chunks-1) + [extra]
            val = [delayed(self.get_single_image_data, pure=True)(per_chunk*i, chunk_size) for i, chunk_size in enumerate(chunk)]
            data = [from_delayed(v,
                                 shape=(chunk_size,
                                        self.image_dict["ImageWidth"],
                                        self.image_dict["ImageHeight"]*2),
                                 dtype=self.image_dict["ImageBitDepth"])
                    for chunk_size, v in zip(chunk, val)]
            data = concatenate(data, axis=0)
        else:
            data = self.get_image_data()
        if nav_shape is not None:
            shape = list(nav_shape) + [self.image_dict["ImageWidth"], self.image_dict["ImageHeight"]*2]
            data = np.reshape(data, shape)
        return data


def file_reader(top=None,
                bottom=None,
                dark=None,
                gain=None,
                metadata=None,
                xml_file=None,
                lazy=False,
                nav_shape=None, chunks=10):
    """Reads a .seq file.

    Parameters
    ----------
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
    seq = SeqReader(top,
                    bottom,
                    dark,
                    gain,
                    metadata,
                    xml_file)

    seq._get_xml_file()
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
        'original_metadata': metadata,
    }

    return dictionary