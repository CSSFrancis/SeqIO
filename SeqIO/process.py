#!/usr/bin/python

from SeqIO.utils.process_utils import build_parser, process
import logging
import dask
from SeqIO.version import __version__

_logger = logging.getLogger(__name__)

if __name__ == '__main__':
    with dask.config.set(scheduler="single-threaded"):
        args = build_parser()
        process(**vars(args))


