import numpy as np
import logging

import time
from scipy.ndimage import label
from scipy.ndimage import center_of_mass, maximum_position
from scipy.ndimage import sum as sum_labels

_logger = logging.getLogger(__name__)

def _counting_filter_cpu(image,
                         threshold=5,
                         integrate=False,
                         hdr_mask=None,
                         method="maximum",
                         mean_electron_val=256,
                         ):
    """This counting filter is GPU designed so that you can apply an hdr mask
    for regions of the data that are higher than some predetermined threshold.

    It also allows for you to integrate the electron events rather than counting
    them.
    """
    tick = time.time()
    nav_dims = len(image.shape) - 2
    try:
        if hdr_mask is not None:
            hdr_img = image * hdr_mask
            hdr_img = hdr_img / mean_electron_val
            nav_slice = slice(None) * nav_dims
            if nav_slice == ():
                image[hdr_mask] = 0
            else:
                image[nav_slice, hdr_img] = 0
        thresh = image > threshold
        struct = np.zeros((3,)*(nav_dims+2))
        if nav_dims == ():
            struct = 1
        else:
            struct[(1,)*nav_dims] = 1
        all_labels, num = label(thresh,
                                structure=struct)  # get blobs
        print("Number of electrons Found! : ", num, flush=True)
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
                inds = (i for i in ind)
                x[inds] = sum_lab
            except:
                pass
        else:
            try:
                inds = (i for i in ind)
                x[inds] = 1
            except:
                pass
        if hdr_mask is not None:
            if nav_slice == ():
                image[hdr_mask] = hdr_img[hdr_mask]
            else:
                image[nav_slice, hdr_mask] = hdr_img[nav_slice, hdr_mask]
        if integrate is False and hdr_mask is None:
            x = x.astype(bool)  # converting to boolean...
        tock = time.time()
        _logger.info(msg="Time elapsed for one Chunk" + str(tock-tick) + " seconds")
        return x
    except MemoryError:
        _logger.error("Failed....  Memory Error")
