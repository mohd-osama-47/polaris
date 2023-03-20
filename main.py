#!/usr/bin/env python3

import os
import argparse
from tqdm import tqdm, trange
from polaris import functions as polfuncs


VALID_IMAGES = ["jpg","png"]

class InputDirectoryException(Exception):
    pass

class OutputDirectoryException(Exception):
    pass


def get_images(folder):
    """Get all images that can be manipulated in the passed folder ONLY"""
    for item in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, item)):
            try:
                if item.split(".")[1].lower() not in VALID_IMAGES:
                    continue
            except IndexError:
                continue
            yield os.path.abspath(os.path.join(folder, item))

def check_if_folder(folder_in, folder_out):
    """Checks if the passed directory is a valid folder to be operated on"""
    if not os.path.isdir(folder_in):
        raise InputDirectoryException
    if not os.path.isdir(folder_out):
        raise OutputDirectoryException


def process_content(inputpath, outputpath):
    # Preprocess the total files count
    img_list = []
    for filepath in tqdm(get_images(inputpath), unit=" images"):
        img_list.append(filepath)

    print(f"[INFO] FOUND {len(img_list)} VALID IMAGES IN {inputpath}, RUNNING MODEL ON ALL IMAGES NOW....")
    img_list_res = []
    for image in tqdm(img_list, total=len(img_list), unit="images"):
        cur_res = polfuncs.get_preds(image, outputpath)
        # print(cur_res)
            
    
    print(f"[INFO] IMAGES WITH PREDICTIONS SAVED AT DIRECTORY [{outputpath}]!")


def _main(parser=argparse.ArgumentParser()):

    # construct the argument parser and parse the arguments
    subparser = parser.add_subparsers(dest='command')

    predict = subparser.add_parser("predict", help="Run model on a directory of images and same the results on a passed folder")
    
    predict.add_argument("-i", "--input_path", help="Path to the input image directory", type=str, required=True)
    
    predict.add_argument("-o", "--output_path", help="Path to the output image directory", type=str, required=True)
    

    args = parser.parse_args()
    if args.command == 'predict':
        try:
            check_if_folder(args.input_path, args.output_path)
            process_content(args.input_path, args.output_path)
        except InputDirectoryException:
            print(f"[WARN] INPUT FOLDER PASSED IS NOT A VALID DIRECTORY!")
        except OutputDirectoryException:
            print(f"[WARN] OUTPUT FOLDER PASSED IS NOT A VALID DIRECTORY!")

    

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    _main()