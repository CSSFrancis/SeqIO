#!/usr/bin/python

from SeqIO.utils import build_parser, process


if __name__ == '__main__':
    print("\n\n .SEQ Processor Application (and Counting)...\n"
          "Created by: Carter Francis (csfrancis@wisc.edu)\n"
          "Updated 2021-06-18\n"
          "------------------\n")
    args = build_parser()
    process(**vars(args))


