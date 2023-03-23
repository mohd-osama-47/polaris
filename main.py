#!/usr/bin/env python3

import os
import argparse
from tqdm import tqdm, trange
import polaris
from polaris import functions as polfuncs


VALID_IMAGES = ["jpg","png"]

class InputDirectoryException(Exception):
    pass

class OutputDirectoryException(Exception):
    pass

class DatasetDirectoryException(Exception):
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

def check_dataset_folder(folder_in):
    """Checks if the passed directory is a valid folder to be operated on for dataset manipulation"""
    if not os.path.isdir(folder_in):
        raise InputDirectoryException
    dataset_structure = {}
    for dirpath, dirs, files in tqdm(os.walk(folder_in), unit="folders"):
        # print(f"AT:{dirpath}, DIRECTORIES ARE:{dirs}, AND FILES ARE:")
        dataset_structure[dirpath] = dirs
    
    print(f"[INFO] Found {len(dataset_structure)} folders within the passed directory [{folder_in}]!")
    
    # return dataset_structure

def preprocess_dataset_directory(folder_in):
    import zipfile, shutil

    # Go to the Daytime directory and extract the zip file containing the IR images
    print(f"[INFO] EXTRACTING DAYTIME IMAGES LOCATED IN [{folder_in}/Daytime/IR.zip].....")
    with zipfile.ZipFile(os.path.join(folder_in, 'Daytime/IR.zip'), 'r') as zip_ref:
        # zip_ref.extractall(os.path.join(folder_in, 'Daytime/images'))

        # Loop over each file
        for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist()), unit="images"):
            # Extract each file to another directory
            # If you want to extract to current working directory, don't specify path
            zip_ref.extract(member=file, path=os.path.join(folder_in, 'Daytime/'))

    # Move the images from the IR directory to a newly created images directory
    os.mkdir(os.path.join(folder_in, 'Daytime/images'))
    for file in os.listdir(os.path.join(folder_in, 'Daytime/IR')):
        shutil.move(os.path.join(f'{folder_in}/Daytime/IR', file), os.path.join(folder_in, 'Daytime/images'))
    
    # Remove the extracted IR directory since its empty now
    os.removedirs(os.path.join(folder_in, 'Daytime/IR'))

    # Prepare an annotation directory and put the json file there
    os.mkdir(os.path.join(folder_in, 'Daytime/annotations'))
    shutil.copyfile(os.path.join(folder_in, 'Daytime/daytime.json'), os.path.join(folder_in, 'Daytime/annotations/coco_annotation.json'))
    
    print(f"[INFO] SUCCESSFULLY EXTRACTED DAYTIME IMAGES!\nIMAGES SAVED TO [{folder_in}/Daytime/images] AND ANNOTATIONS TO [{folder_in}/Daytime/annotations/coco_annotataion.json].")


    # Repeat the process again for the nighttime subset...
    print(f"[INFO] EXTRACTING NIGHTTIME IMAGES LOCATED IN [{folder_in}/Nighttime/IR.zip].....")
    with zipfile.ZipFile(os.path.join(folder_in, 'Nighttime/IR.zip'), 'r') as zip_ref:
        # zip_ref.extractall(os.path.join(folder_in, 'Nighttime/images'))

        # Loop over each file
        for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist()), unit="folders"):
            zip_ref.extract(member=file, path=os.path.join(folder_in, 'Nighttime/'))

    # os.rename(os.path.join(folder_in, 'Nighttime/IR'), "images")
    os.mkdir(os.path.join(folder_in, 'Nighttime/images'))
    for file in os.listdir(os.path.join(folder_in, 'Nighttime/IR')):
        shutil.move(os.path.join(f'{folder_in}/Nighttime/IR', file), os.path.join(folder_in, 'Nighttime/images'))
    os.removedirs(os.path.join(folder_in, 'Nighttime/IR'))
    os.mkdir(os.path.join(folder_in, 'Nighttime/annotations'))
    shutil.copyfile(os.path.join(folder_in, 'Nighttime/nighttime.json'), os.path.join(folder_in, 'Nighttime/annotations/coco_annotation.json'))

    print(f"[INFO] SUCCESSFULLY EXTRACTED NIGHTTIME IMAGES!\nIMAGES SAVED TO [{folder_in}/Nighttime/images] AND ANNOTATIONS TO [{folder_in}/Nighttime/annotations/coco_annotataion.json].")

    # Done :)

def datum_pipeline(folder_in):
    #! KEEP IN MIND THAT NIGHT DATASET HAS A MISTAKE
    #! ID 11 HAS AN EMPTY SUPER CATEGORY, FILL IT 
    #! IN AS "person"

    #! ALSO IN THE VENV, COMMENT OUT LINE 165 in
    #! "venv/lib/python3.8/site-packages/datumaro/components/annotation.py"
    from datumaro.components.dataset import Dataset
    import datumaro.plugins.transforms as transforms

    # Import and export a dataset
    day_dataset = Dataset.import_from(os.path.join(folder_in,'Daytime'), 'coco_instances')
    night_dataset = Dataset.import_from(os.path.join(folder_in,'Nighttime'), 'coco_instances')
    print("[INFO] ACCESSING DAYTIME DATASET...")

    day_dataset = transforms.RemapLabels(day_dataset,
        mapping = {
        'None': 'None',
        'Car1':  'Car',
        'Car2':  'Car',
        'Car3':  'Car',
        'Car4':  'Car',
        'Car5':  'Car',
        'Car6':  'Car',
        'Car7':  'Car',
        'Car8':  'Car',
        'Car9':  'Car',
        'Car10': 'Car',
        'Car11': 'Car',
        'Bus':      'Bus',
        'SmallBus': 'Bus',
        'Person1': 'Person',
        'Person2': 'Person',
        'Person3': 'Person',
        'Person4': 'Person',
        'Person5': 'Person',
        'Person6': 'Person',
        'Person7': 'Person',
        'Person8': 'Person',
        'Person9': 'Person',
        'ATV driver': 'driver',
        'Truck':        'Truck',
        'UtilityTruck': 'Truck',
        'Buggy': 'offroad_vehicle',
        'ATV':   'offroad_vehicle',
        'Motorcyclist': 'Motorcyclist'
        }, default='keep')
    
    day_dataset = Dataset.from_extractors(day_dataset)


    print("[INFO] MAPPED ALL SUB-CATEGORIES BASED ON SUPER-CATEGORY")
    print("[INFO] RE-INDEXING DAYTIME DATASET TO START FROM 1...")
    day_dataset.transform('reindex',start=1)
    day_dataset.export(f'{folder_in}/test1', 'coco_instances')
    
    print("[INFO] ACCESSING NIGHTTIME DATASET...")
    night_dataset = transforms.RemapLabels(night_dataset,
        mapping = {
        'None': 'None',
        'Car1':   'Car',
        'Car12':  'Car',
        'Car13':  'Car',
        'Car14':  'Car',
        'Car15':  'Car',
        'Car16':  'Car',
        'Car17':  'Car',
        'Car18':  'Car',
        'Car19':  'Car',
        'Person1':  'Person',
        'Person4':  'Person',
        'Person10': 'Person',
        'Person2':  'Person',
        'Person3':  'Person',
        'Person6':  'Person',
        'ATV driver': 'driver',
        'Buggy': 'offroad_vehicle',
        'ATV':   'offroad_vehicle',
        'Motorcyclist': 'Motorcyclist',
        }, default='keep')
    
    night_dataset = Dataset.from_extractors(night_dataset)
    print("[INFO] MAPPED ALL SUB-CATEGORIES BASED ON SUPER-CATEGORY")
    print("[INFO] RE-INDEXING NIGHTTIME DATASET TO START FROM 5624...")
    night_dataset.transform('reindex',start=5624)
    day_dataset.export(f'{folder_in}/nighttime', 'coco_instances')

def process_content(inputpath, outputpath):
    # Preprocess the total files count
    img_list = []
    for filepath in tqdm(get_images(inputpath), unit="image"):
        img_list.append(filepath)

    print(f"[INFO] FOUND {len(img_list)} VALID IMAGES IN [{os.path.abspath(inputpath)}], RUNNING MODEL ON ALL IMAGES NOW....")
    img_list_res = []
    for image in tqdm(img_list, total=len(img_list), unit="images"):
        cur_res = polfuncs.get_preds(image, outputpath)
            
    
    print(f"[INFO] IMAGES WITH PREDICTIONS SAVED AT DIRECTORY [{os.path.abspath(outputpath)}]!")


def _main(parser=argparse.ArgumentParser()):

    # construct the argument parser and parse the arguments
    subparser = parser.add_subparsers(dest='command')

    predict = subparser.add_parser("predict", help="Run model on a directory of images and same the results on a passed folder")

    prepare_set = subparser.add_parser("set-prep", help="Given the supplied dataset, prepare the dataset to follow yolo convention and merge all categories to their respective super-category")
    
    predict.add_argument("-i", "--input_path", help="Path to the input image directory", type=str, required=True)
    
    predict.add_argument("-o", "--output_path", help="Path to the output image directory", type=str, required=True)
    
    prepare_set.add_argument("-i", "--input_path", help="Path to the input dataset directory", type=str, required=True)

    args = parser.parse_args()

    if args.command == 'predict':
        try:
            check_if_folder(args.input_path, args.output_path)
            process_content(args.input_path, args.output_path)
        except InputDirectoryException:
            print(f"[WARN] INPUT FOLDER PASSED IS NOT A VALID DIRECTORY!")
        except OutputDirectoryException:
            print(f"[WARN] OUTPUT FOLDER PASSED IS NOT A VALID DIRECTORY!")
        
    if args.command == 'set-prep':
        try:
            # check_dataset_folder(args.input_path)
            # preprocess_dataset_directory(args.input_path)
            datum_pipeline(args.input_path)
        except InputDirectoryException:
            print(f"[WARN] INPUT FOLDER PASSED IS NOT A VALID DIRECTORY!")
        except DatasetDirectoryException:
            print(f"[WARN] DATASET DIRECTORY PASSED NOT MATCHING EXPECTED LAYOUT!")
        except FileNotFoundError as e:
            print(f"[WARN] SOME OF THE DATASET DIRECTORY CONTENTS ARE MISSING!, CHECK ERROR BELOW:")
            print(e)

    

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    _main()