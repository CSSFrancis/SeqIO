import time

from scipy.ndimage import label
from scipy.ndimage import center_of_mass, maximum_position
from scipy.ndimage import sum_labels
from scipy.signal import fftconvolve
import numpy as np
import glob

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