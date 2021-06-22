#!/usr/bin/python

from SeqIO.utils import build_parser, process
import logging

_logger = logging.getLogger(__name__)

if __name__ == '__main__':
    _logger.info(msg="\n\n .SEQ Processor Application (and Counting)...\n"
                     "Created by: Carter Francis (csfrancis@wisc.edu)\n"
                     "Updated 2021-06-18\n"
                     "------------------\n")
    args = build_parser()
    process(**vars(args))


