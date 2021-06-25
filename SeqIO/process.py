#!/usr/bin/python

from SeqIO.utils.process_utils import build_parser, process
import logging
from SeqIO.version import __version__

_logger = logging.getLogger(__name__)

if __name__ == '__main__':
    args = build_parser()
    process(**vars(args))


