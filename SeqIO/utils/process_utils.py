import argparse
import os
import sys
import logging
import time

import numpy as np
import hyperspy.api as hs
from hyperspy.io import dict2signal
import dask.array as da

from SeqIO.SeqReader import file_reader as file_reader
from SeqIO.CeleritasSeqReader import file_reader as cel_file_reader
from SeqIO.version import __version__
from SeqIO.utils.file_utils import get_files
from SeqIO.utils.counting import _counting_filter_cpu

_logger = logging.getLogger(__name__)


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
    parser.add_argument("-n",
                        "--nav_shape",
                        nargs="+",
                        type=int,
                        default=None,
                        help="The navigation shape for some n dimensional dataset")
    parser.add_argument("-cs",
                        "--chunk_shape",
                        nargs="+",
                        type=int,
                        default=None,
                        help="The navigation shape for some n dimensional dataset"
                        )
    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="Increase verbosity of output"
                        )
    args = parser.parse_args()
    return args


def process(directory,
            threshold=6,
            integrate=False,
            counting=False,
            hdr=None,
            mean_e=256,
            nav_shape=None,
            chunk_shape=None,
            verbose=False):
    if verbose:
        _logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s \n')
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
    
    _logger.info(msg="\n\n .SEQ Processor Application (and Counting)...\n"
                     "Created by: Carter Francis (csfrancis@wisc.edu)\n"
                     "Updated 2021-06-18\n"
                     "------------------\n")
    _logger.info(msg="Version:" + __version__)
    tick = time.time()
    file_dict = get_files(folder=directory)
    for key in file_dict:
        if len(file_dict[key]) == 0:
            file_dict[key].pop()
        else:
            file_dict[key] = file_dict[key][0]
    if "top" in file_dict and "bottom" in file_dict:
        file_dict.pop("seq")
        data_dict = cel_file_reader(**file_dict,
                                    nav_shape=nav_shape,
                                    chunk_shape=chunk_shape,
                                    lazy=True)
    elif "seq" in file_dict:
        data_dict = file_reader(**file_dict,
                                nav_shape=nav_shape,
                                chunk_shape=chunk_shape,
                                lazy=True)
    if hdr is not None:
        hdr = hs.load(hdr).data
    else:
        hdr = None
    if hdr is None and integrate is False:
        dtype = bool
    else:
        dtype = np.float32

    if counting:
        data_dict["data"] = data_dict["data"].map_blocks(_counting_filter_cpu,
                                               threshold=threshold,
                                               integrate=integrate,
                                               hdr_mask=hdr,
                                               method="maximum",
                                               mean_electron_val=mean_e,
                                               dtype=dtype)

    _logger.info(data_dict)
    sig = dict2signal(data_dict, lazy=True)

    _logger.info("Data... :" + str(sig.data))
    _logger.info("Dtype:" + str(sig.data.dtype))
    _logger.info("Saving... ")

    da.to_zarr(sig.data, directory+"_zarr", overwrite=True)

    #sig.save(directory + ".hspy",
    #         compression=False,
    #         overwrite=True)

    tock = time.time()
    _logger.info("Total time elapsed : " + str(tock-tick) + " sec")
    return sig

