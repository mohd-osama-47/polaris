#!/usr/bin/env python3

import argparse
from tqdm import tqdm, trange
import os
from time import sleep
from pathlib import Path
from polaris import functions

def dosomething(buf):
    """Do something with the content of a file"""
    sleep(0.0001)
    pass

def walkdir(folder):
    """Walk through each files in a directory"""
    for dirpath, dirs, files in os.walk(folder):
        for filename in files:
            yield os.path.abspath(os.path.join(dirpath, filename))


def process_content(inputpath, blocksize=1024):
    # Preprocess the total files sizes
    sizecounter = 0
    for filepath in tqdm(walkdir(inputpath), unit="files"):
        sizecounter += os.stat(filepath).st_size

    # Load tqdm with size counter instead of file counter
    with tqdm(total=sizecounter,
              unit='B', unit_scale=True, unit_divisor=1024) as pbar:
        for filepath in walkdir(inputpath):
            with open(filepath, 'rb') as fh:
                buf = 1
                while (buf):
                    buf = fh.read(blocksize)
                    dosomething(buf)
                    if buf:
                        pbar.set_postfix(file=filepath[-10:], refresh=False)
                        pbar.update(len(buf))

def _main(parser=argparse.ArgumentParser()):

    # construct the argument parser and parse the arguments
    parser.add_argument("-i", "--input_path", help="Path to the input image directory", type=str)
    
    parser.add_argument("-o", "--output_path", help="Path to the output image directory", type=str)

    args = parser.parse_args()

    if args.input_path:
        process_content(args.input_path)
    if args.output_path:
        process_content(args.output_path)

    

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    _main()