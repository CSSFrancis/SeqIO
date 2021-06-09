#!/usr/bin/python
import os
import glob
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
import time
from SeqIO.SeqReader import SeqReader
from SeqIO.CeleritasSeqReader import SeqReader as CeleritasSeqReader
import hyperspy.api as hs
from hyperspy.io import dict2signal
from SeqIO.utils import _counting_filter_cpu, _counting_filter_gpu
from dask.array import reshape

def get_files(folder):
    file_dict = {"top": glob.glob(folder + "/*Top*.seq"),
                 "bottom": glob.glob(folder + "/*Bottom*.seq"),
                 "seq": glob.glob(folder + "/*.seq"),
                 "gain": glob.glob(folder + "/*gain*.mrc"),
                 "dark": glob.glob(folder + "/*dark*.mrc"),
                 "xml_file": glob.glob(folder + "/*.xml"),
                 "metadata": glob.glob(folder + "/*.metadata")}
    return file_dict

if __name__ == '__main__':
    print()
    print()
    tick = time.time()
    print(".SEQ Processor Application (and Counting)...")
    print("Created by: Carter Francis (csfrancis@wisc.edu)")
    print("Updated 2021-05-20")
    print("------------------")
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
                        type=bool,
                        default=False,
                        help="If the data should be integrated instead of counted. For testing...")
    parser.add_argument("-c",
                        "--counting",
                        type=bool,
                        default=True,
                        help="If the dataset should be counted or just converted")
    parser.add_argument("-hd",
                        "--hdr",
                        type=bool,
                        default=True,
                        help="If a high dynamic range mask should be calculated and applied")
    parser.add_argument("-n",
                        "--nonlinear",
                        type=bool,
                        default=True,
                        help="If a nonlinear range mask should be calculated and applied")
    parser.add_argument("-m",
                        "--mean_e",
                        type=int,
                        default=104,
                        help="The mean electron value")
    parser.add_argument("-mp",
                        "--mean_p",
                        type=int,
                        default=80,
                        help="The pixel value for an electron NOT TOTAL EVENT VALUE")
    parser.add_argument("-nc",
                        "--nonlinear_cutoff",
                        type=int,
                        default=3,
                        help="The cut off in number of electrons to be considered non linear")
    parser.add_argument("-g",
                        "--gpu",
                        type=bool,
                        default=False,
                        help="Use GPU for Counting")
    parser.add_argument("-ns",
                        "--nav_shape",
                        nargs="+",
                        type=int,
                        default=None,
                        help="The navigation shape for some n dimensional dataset")
    args = parser.parse_args()
    #getting the relevant files
    file_dict = get_files(folder=args.directory)
    if len(file_dict["top"])==0 and len(file_dict["bottom"])==0:
        try:
            reader = SeqReader(file=file_dict["seq"][0])
        except IndexError:
            print("The folder : ", args.folder, " Doesn't have a .seq file in it")
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

    if args.gpu:
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
            args.gpu = False
            print("Cupy Must be installed to use GPU Processing.... ")
            print("Using CPU Processing instead")
    if not args.gpu:
        chunksize = 100000000
        print("Each chunk is: ", chunksize / 1000000000, "Gb")

    if args.hdr:
        #If HDR should be calculated and output as a mask

        if reader.image_dict["NumFrames"] < 1000:
            print("High dynamic Range only works (well)  with more than 1000 Frames ")
        else:
            temp = reader.get_image_chunk(20, 1000)
            temp[temp > args.threshold] = 0
            temp_sum = (np.sum(temp, axis=0)/980)/args.mean_p
            hdr = (temp_sum > 3)  # Average 3 electrons
            nonlinear = (temp_sum < 3) > 0.05  # 95% Sparse data only
            print()
            hs.signals.Signal2D(hdr).save(args.directory+"hdr_mask.hspy",
                                          compression=False, overwrite=True)
            print()
            hs.signals.Signal2D(nonlinear).save(args.directory + "nonlinear_mask.hspy",
                                                compression=False, overwrite=True)
    else:
        hdr = None

    data = reader.read_data(lazy=True,
                            chunksize=chunksize)
    if hdr is None and args.integrate is False:
        dtype = bool
    else:
        dtype = np.float32


    print("Input Dataset:", data)
    if args.gpu:
        data = data.map_blocks(cupy.asarray)

        counted = data.map_blocks(_counting_filter_gpu,
                                  threshold=args.threshold,
                                  integrate=args.integrate,
                                  hdr_mask=cupy.asarray(hdr),
                                  method="maximum",
                                  mean_electron_val=args.mean_e,
                                  convolve=True,
                                  dtype=dtype)
    else:
        counted = data.map_blocks(_counting_filter_cpu,
                                  threshold=args.threshold,
                                  integrate=args.integrate,
                                  hdr_mask=hdr,
                                  method="maximum",
                                  mean_electron_val=args.mean_e,
                                  convolve=True,
                                  dtype=dtype)

    counted = counted.astype(dtype=bool)

    if args.nav_shape is not None:
        new_shape = list(args.nav_shape) + [reader.image_dict["ImageWidth"], reader.image_dict["ImageHeight"]*2]
        print("The output data Shape: ", new_shape)
        counted = reshape(counted, new_shape)
        test_size = 1
        for i in new_shape:
            test_size = test_size*i
        axes = reader.create_axes(nav_shape=list(args.nav_shape))
    else:
        axes = reader.create_axes()
    metadata = reader.create_metadata()
    counted = counted.rechunk({-1 : -1, -2 : -1})
    dictionary = {
        'data': counted,
        'metadata': metadata,
        'axes': axes,
        'original_metadata': metadata,
    }
    sig = dict2signal(dictionary, lazy=True)
    print("Data... :", sig.data)
    print("Dtype:", sig.data.dtype)
    print("Saving... ")
    sig.save(args.directory + ".hspy", compression=False, overwrite=True)
    tock = time.time()
    print("Total time elapsed : ", tock-tick, " sec")
    print("Time per frame: ",  (tock-tick)/reader.image_dict["NumFrames"], "sec")











