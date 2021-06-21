import time

from scipy.ndimage import label
from scipy.ndimage import center_of_mass, maximum_position
from scipy.ndimage import sum as sum_labels
from scipy.signal import fftconvolve
import argparse
import os
import glob

import numpy as np
import time
from SeqIO.SeqReader import SeqReader
from SeqIO.CeleritasSeqReader import SeqReader as CeleritasSeqReader
import hyperspy.api as hs
from hyperspy.io import dict2signal
from dask.array import reshape

try:
    import cupy
    from cupyx.scipy.ndimage import label as clabel
    from cupyx.scipy.ndimage import sum as csum_labels
    from cupyx.scipy.ndimage import center_of_mass as ccenter_of_mass
    from cupyx.scipy.ndimage import maximum_position as cmaximum_position
    from cupyx.scipy.signal import fftconvolve as cfftconvolve
except ImportError:
    print("Cupy is not installed.  No GPU operations are possible")


def _counting_filter_cpu(image,
                         threshold=5,
                         integrate=False,
                         hdr_mask=None,
                         method="maximum",
                         mean_electron_val=104,
                         convolve=True
                         ):
    """This counting filter is GPU designed so that you can apply an hdr mask
    for regions of the data that are higher than some predetermined threshold.

    It also allows for you to integrate the electron events rather than counting
    them.
    """
    tick = time.time()
    try:
        if hdr_mask is not None and integrate is False:
            hdr_img = image * hdr_mask
            hdr_img = hdr_img / mean_electron_val
            if len(image.shape) == 3:
                image[:, hdr_mask] = 0
            else:
                image[hdr_mask] = 0
        thresh = image > threshold

        if len(image.shape) == 3:
            kern = np.zeros((2, 4, 4))
            kern[0, :, :] = 1
        else:
            kern = np.ones((4, 4))
        if convolve:
            conv = fftconvolve(thresh,
                                kern,
                                mode="same")
            conv = conv > 0.5
        else:
            conv = thresh
        if len(image.shape) == 3:
            struct = [[[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]],
                      [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]],
                      [[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]]
                      ]
        else:
            struct = [[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]]
        all_labels, num = label(conv, structure=struct)  # get blobs
        print(num)
        if method is "center_of_mass":
            ind = center_of_mass(image, all_labels, range(1, num))
        elif method is "maximum":
            ind = maximum_position(image, all_labels, range(1, num))
        ind = np.rint(ind).astype(int)
        x = np.zeros(shape=image.shape)
        if integrate:
            try:
                image[~threshold] = 0
                sum_lab = sum_labels(image, all_labels, range(1, num))
                if len(image.shape) == 3:
                    x[ind[:, 0], ind[:, 1], ind[:, 2]] = sum_lab
                else:
                    x[ind[:, 0], ind[:, 1]] = sum_lab
            except:
                pass
        else:
            try:
                if len(image.shape) == 3:
                    x[ind[:, 0], ind[:, 1], ind[:, 2]] = 1
                else:
                    x[ind[:, 0], ind[:, 1]] = 1
            except:
                pass
        if hdr_mask is not None and integrate is False:
            if len(image.shape) == 3:
                x[:, hdr_mask] = hdr_img[:, hdr_mask]
            else:
                image[hdr_mask] = hdr_img[hdr_mask]
        if integrate is False and hdr_mask is None:
            x = x.astype(bool) # converting to boolean...
        tock = time.time()
        print("Time elapsed for one Chunk", tock-tick, "seconds")
        return x
    except MemoryError:
        print("Failed....  Memory Error")


def _counting_filter_gpu(image,
                         threshold=5,
                         integrate=False,
                         hdr_mask=None,
                         method="maximum",
                         mean_electron_val=104,
                         convolve=False,
                         ):
    """This counting filter is GPU designed so that you can apply an hdr mask
    for regions of the data that are higher than some predetermined threshold.

    It also allows for you to integrate the electron events rather than counting
    them.
    """

    try:
        if hdr_mask is not None and integrate is False:
            hdr_img = image * hdr_mask
            hdr_img = hdr_img / mean_electron_val
            if len(image.shape) == 3:
                image[:, hdr_mask] = 0
            else:
                image[hdr_mask] = 0
        thresh = image > threshold

        if len(image.shape) == 3:
            kern = cupy.zeros((2, 4, 4))
            kern[0, :, :] = 1
        else:
            kern = cupy.ones((4, 4))
        if convolve:
            conv = cfftconvolve(thresh,
                               kern,
                               mode="same")
            conv = conv > 0.5
        else:
            conv = thresh
        del thresh  # Cleaning up GPU Memory

        if len(image.shape) == 3:
            struct = cupy.asarray([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                   [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                   [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                                   ])
        else:
            struct = cupy.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

        all_labels, num = clabel(conv, structure=struct)  # get blobs
        print("Number of Electrons per chunk:", num)
        del conv  # Cleaning up GPU Memory
        del kern  # Cleaning up GPU Memory

        if method is "center_of_mass":
            ind = cupy.asarray(ccenter_of_mass(image, all_labels, cupy.arange(1, num)))
        elif method is "maximum":
            ind = cupy.asarray(cmaximum_position(image, all_labels, cupy.arange(1, num)))
        ind = cupy.rint(ind).astype(int)
        if hdr_mask is not None and integrate is False:
            x = cupy.zeros(shape=image.shape, dtype=float)
        else:
            x = cupy.zeros(shape=image.shape, dtype=bool)
        if integrate:
            try:
                image[~threshold] = 0
                sum_lab = csum_labels(image, all_labels, cupy.arange(1, num))
                if len(image.shape) == 3:
                    x[ind[:, 0], ind[:, 1], ind[:, 2]] = sum_lab
                else:
                    x[ind[:, 0], ind[:, 1]] = sum_lab
            except:
                pass
        else:
            try:
                if len(image.shape) == 3:
                    x[ind[:, 0], ind[:, 1], ind[:, 2]] = 1
                else:
                    x[ind[:, 0], ind[:, 1]] = 1
            except:
                pass
        if hdr_mask is not None and integrate is False:
            if len(image.shape) == 3:
                x[:, hdr_mask] = hdr_img[:, hdr_mask]
            else:
                image[hdr_mask] = hdr_img[hdr_mask]
        del image
        del all_labels
        del ind
        x = x.get()
        return x
    except MemoryError:
        print("Failed....  Memory Error")


def _load_folder(folder):
    top = glob.glob(folder+"*Top*.seq")[0]
    bottom = glob.glob(folder+"*Bottom*.seq")[0]
    gain = glob.glob(folder+"*gain*.mrc")[0]
    dark = glob.glob(folder+"*dark*.mrc")[0]
    xml = glob.glob(folder+"*.xml")[0]
    meta = glob.glob(folder+"*.metadata")[0]
    s = SeqIO.load_celeritas(top=top,
                             bottom=bottom,
                             dark=dark,
                             gain=gain,
                             xml_file=xml,
                             metadata=meta)
    return s

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--directory",
                        type=str,
                        default=os.getcwd(),
                        help="Input directory which contains dark/gain/metadata/xml file")
    parser.add_argument("-t",
                        "--threshold",
                        type=int,
                        default=7,
                        help="The threshold for the counting filter")
    parser.add_argument("-i",
                        "--integrate",
                        action="store_true",
                        help="If the data should be integrated instead of counted. For testing...")
    parser.add_argument("-c",
                        "--counting",
                        action="store_true",
                        help="If the dataset should be counted or just converted")
    parser.add_argument("-hd",
                        "--hdr",
                        type=str,
                        default=None,
                        help=" .hspy signal to apply as a HDR Mask")
    parser.add_argument("-m",
                        "--mean_e",
                        type=int,
                        default=104,
                        help="The mean electron value")
    parser.add_argument("-g",
                        "--gpu",
                        action="store_true",
                        help="Use GPU for Counting")
    parser.add_argument("-n",
                        "--nav_shape",
                        nargs="+",
                        type=int,
                        default=None,
                        help="The navigation shape for some n dimensional dataset")
    parser.add_argument("-con",
                        "--convolve",
                        action="store_true",
                        help="Convolve when counting, default is False")
    parser.add_argument("-f",
                        "--fast_axis",
                        type=int,
                        default=0,
                        help="The navigation shape for some n dimensional dataset"
                        )
    args = parser.parse_args()
    return args


def get_files(folder):
    file_dict = {"top": glob.glob(folder + "/*Top*.seq"),
                 "bottom": glob.glob(folder + "/*Bottom*.seq"),
                 "seq": glob.glob(folder + "/*.seq"),
                 "gain": glob.glob(folder + "/*gain*.mrc"),
                 "dark": glob.glob(folder + "/*dark*.mrc"),
                 "xml_file": glob.glob(folder + "/*.xml"),
                 "metadata": glob.glob(folder + "/*.metadata")}
    return file_dict


def process(directory,
            threshold=6,
            integrate=False,
            counting=False,
            hdr=None,
            mean_e=256,
            gpu=False,
            nav_shape=None,
            fast_axis=0,
            convolve=False):
    tick = time.time()
    file_dict = get_files(folder=directory)
    if len(file_dict["top"]) == 0 and len(file_dict["bottom"]) == 0:
        try:
            reader = SeqReader(file=file_dict["seq"][0])
        except IndexError:
            print("The folder : ", directory, " Doesn't have a .seq file in it")
    else:
        file_dict.pop("seq")
        for key in file_dict:
            if len(file_dict[key])==0:
                file_dict[key]=None
            else:
                file_dict[key] = file_dict[key][0]
        reader = CeleritasSeqReader(**file_dict)
        reader._get_xml_file()

    reader.parse_header()
    reader.parse_metadata_file()
    reader._get_dark_ref()
    reader._get_gain_ref()

    if gpu:
        #Trying to import cupy
        try:
            import cupy
            CUPY_INSTALLED = True
            gpu_mem = cupy.cuda.runtime.getDeviceProperties(0)["totalGlobalMem"]
            chunksize = gpu_mem/200
            print("The available Memory for each GPU is: ", gpu_mem/1000000000, "Gb")
            print("Each chunk is: ", chunksize / 1000000000, "Gb")
        except ImportError:
            CUPY_INSTALLED = False
            gpu = False
            print("Cupy Must be installed to use GPU Processing.... ")
            print("Using CPU Processing instead")
    if not gpu:
        chunksize = 100000000
        print("Each chunk is: ", chunksize / 1000000000, "Gb")

    if hdr is not None:
        hdr = hs.load(hdr).data
    else:
        hdr = None
    if nav_shape is not None:
        fast = nav_shape[-1]
    else:
        fast=None
    data = reader.read_data(lazy=True,
                            chunks=None,
                            fast_shape=fast)
    print(chunksize)
    print(data)
    if hdr is None and integrate is False:
        dtype = bool
    else:
        dtype = np.float32

    if counting:
        if gpu:
            data = data.map_blocks(cupy.asarray)
            counted = data.map_blocks(_counting_filter_gpu,
                                      threshold=threshold,
                                      integrate=integrate,
                                      hdr_mask=cupy.asarray(hdr),
                                      method="maximum",
                                      mean_electron_val=mean_e,
                                      convolve=convolve,
                                      dtype=dtype)
        else:
            counted = data.map_blocks(_counting_filter_cpu,
                                      threshold=threshold,
                                      integrate=integrate,
                                      hdr_mask=hdr,
                                      method="maximum",
                                      mean_electron_val=mean_e,
                                      convolve=convolve,
                                      dtype=dtype)
    else:
        counted = data
    print()
    if nav_shape is not None:
        new_shape = list(nav_shape) + [reader.image_dict["ImageHeight"]*2, reader.image_dict["ImageWidth"]]
        print("The output data Shape: ", new_shape)
        if counted.shape[0] != np.prod(nav_shape):
            print("The data cannot but reshaped into the nav shape.  Probably this is because the "
                  "pre-segment buffer is not a factor of the nav shape and thus frames are dropped...")
            frames_added = np.prod(nav_shape) - counted.shape[0]
            print("Adding :", frames_added, "frames")
            print("Data", counted)
            from dask.array import concatenate
            from dask.array import zeros
            counted = concatenate([counted, zeros((frames_added,
                                                   reader.image_dict["ImageHeight"] * 2,
                                                   reader.image_dict["ImageWidth"]))],
                                  axis=0)
        print("Data after adding frames", counted)
        counted = np.reshape(counted,
                             new_shape)
        print("Data after reshape", counted)
        test_size = 1
        for i in new_shape:
            test_size = test_size*i
        axes = reader.create_axes(nav_shape=list(nav_shape))
    else:
        axes = reader.create_axes()
    metadata = reader.create_metadata()
    #counted = counted.rechunk({fast_axis: -1, -1: -1, -2: -1})
    dictionary = {
        'data': counted,
        'metadata': metadata,
        'axes': axes,
        'original_metadata': metadata,
    }
    print(counted)
    sig = dict2signal(dictionary, lazy=True)
    print("Data... :", sig.data)
    print("Dtype:", sig.data.dtype)
    print("Saving... ")
    sig.save(directory + ".hspy", compression=False, overwrite=True)
    tock = time.time()
    print("Total time elapsed : ", tock-tick, " sec")
    print("Time per frame: ",  (tock-tick)/reader.image_dict["NumFrames"], "sec")








