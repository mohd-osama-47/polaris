#!/usr/bin/env python3

import os
import argparse
from tqdm import tqdm, trange
import polaris
import shutil
from polaris import functions as polfuncs
# from polaris import _track as trackfuncs
import re
import time
from pathlib import Path

VALID_IMAGES = ["jpg","png"]

class InputDirectoryException(Exception):
    pass

class OutputDirectoryException(Exception):
    pass

class DatasetDirectoryException(Exception):
    pass

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def get_images(folder:str):
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
    for dirpath, dirs, files in tqdm(os.walk(folder_in), unit="folders", colour='green'):
        # print(f"AT:{dirpath}, DIRECTORIES ARE:{dirs}, AND FILES ARE:")
        dataset_structure[dirpath] = dirs
    
    print(f"[INFO] Found {len(dataset_structure)} folders within the passed directory [{folder_in}]!")
    
    # return dataset_structure

def preprocess_dataset_directory(folder_in):
    import zipfile

    # Go to the Daytime directory and extract the zip file containing the IR images
    print(f"[INFO] EXTRACTING DAYTIME IMAGES LOCATED IN [{folder_in}/Daytime/IR.zip].....")
    with zipfile.ZipFile(os.path.join(folder_in, 'Daytime/IR.zip'), 'r') as zip_ref:
        # zip_ref.extractall(os.path.join(folder_in, 'Daytime/images'))

        # Loop over each file
        for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist()), unit="images", colour='green'):
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
        for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist()), unit="folders", colour='green'):
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

    # Merge all categories based on their supercategory
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
    # just export the json file for now...
    day_dataset.export(f'{folder_in}/daytime', 'coco_instances')
    
    # Repeat the process for the night dataset...
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

    # Since some labels done exist here when compared to the day dataset,
    # simple add them so that merging can occur.
    night_dataset = transforms.ProjectLabels(night_dataset,
        dst_labels = ["Person", "offroad_vehicle", "Motorcyclist", "driver", "None", "Car", "Bus", "Truck"])
    night_dataset = Dataset.from_extractors(night_dataset)

    print("[INFO] MAPPED ALL SUB-CATEGORIES BASED ON SUPER-CATEGORY")
    print("[INFO] RE-INDEXING NIGHTTIME DATASET TO START FROM 5624...")
    night_dataset.transform('reindex',start=5624)
    
    # again, just save the json file for now
    night_dataset.export(f'{folder_in}/nighttime', 'coco_instances')

    # start generating the fused dataset with all the images in one directory
    final_dataset = Dataset.from_extractors(day_dataset, night_dataset)
    print(f"[INFO] SAVING MERGED DATASET TO [{folder_in}final].....")
    final_dataset.export(f'{folder_in}/final', 'coco_instances', save_media=True)
    
    # this here moves the files to more conveniently labeled directories
    for file in tqdm(os.listdir(os.path.join(folder_in,'final/images/default')), unit="image", colour='green'):
        file_name = os.path.join(os.path.join(folder_in,'final/images/default'), file)
        shutil.move(file_name, os.path.join(folder_in,'final/images'))
    os.removedirs(os.path.join(folder_in,'final/images/default'))
    
    # Done! :)

def export_to_yolo(input_folder):
    from datumaro.components.dataset import Dataset
    print("[INFO] EXPORTING AS YOLO DATASET...")
    final_dataset = Dataset.import_from(os.path.join(input_folder,'final'), 'coco_instances')
    final_dataset.export(f'{input_folder}/finalYOLO', 'yolo', save_media=True)
    print(f"[INFO] DONE EXPORTING DATASET AS YOLO TO [{input_folder}/finalYOLO]!")

def cleanup_files(input_folder):
    '''
    Removes all extracted and generated folders to clean up the directory
    '''
    os.remove(os.path.join(input_folder, "final/annotations/instances_default.json"))
    shutil.rmtree(os.path.join(input_folder, "daytime"), ignore_errors=True)
    shutil.rmtree(os.path.join(input_folder, "nighttime"), ignore_errors=True)
    shutil.rmtree(os.path.join(input_folder, "Daytime/images"), ignore_errors=True)
    shutil.rmtree(os.path.join(input_folder, "Daytime/annotations"), ignore_errors=True)
    shutil.rmtree(os.path.join(input_folder, "Nighttime/images"), ignore_errors=True)
    shutil.rmtree(os.path.join(input_folder, "Nighttime/annotations"), ignore_errors=True)

def process_content(inputpath:str, outputpath:str, save_images:bool):
    # Preprocess the total files count
    img_list = []
    for filepath in tqdm(get_images(inputpath), unit="image", colour='green'):
        img_list.append(filepath)
    # polfuncs.load_model()
    print(f"[INFO] FOUND {len(img_list)} VALID IMAGES IN [{os.path.abspath(inputpath)}], RUNNING MODEL ON ALL IMAGES NOW....")
    img_list = natural_sort(img_list)
    # for i, image in tqdm(enumerate(img_list, start=1), total=len(img_list), unit="images"):
    #     polfuncs.get_preds(image, outputpath, image_num=int(image.split("/")[-1].split(".")[0]), image_name = image.split("/")[-1], save_files=save_images)
    polfuncs.get_preds(img_list, outputpath, save_images)
            
    
    print(f"[INFO] IMAGES WITH PREDICTIONS SAVED AT DIRECTORY [{os.path.abspath(outputpath)}]!")

def process_tracking(inputpath:str, outputpath:str, save_images:bool):
    # Preprocess the total files count
    img_list = []
    for filepath in tqdm(get_images(inputpath), unit="image", colour='green'):
        img_list.append(filepath)

    print(f"[INFO] FOUND {len(img_list)} VALID IMAGES IN [{os.path.abspath(inputpath)}], RUNNING TRACKING ON ALL IMAGES NOW....")

    polfuncs.run_tracker(img_list, outputpath)

def _main(parser=argparse.ArgumentParser()):

    # construct the argument parser and parse the arguments
    subparser = parser.add_subparsers(dest='command')

    predict = subparser.add_parser("predict", help="Run model on a directory of images and same the results on a passed folder")
    track = subparser.add_parser("track", help="Run tracking on a directory of images and save the results to output directory")
    prepare_set = subparser.add_parser("set-prep", help="Given the supplied dataset, prepare the dataset to follow yolo convention and merge all categories to their respective super-category")
    
    
    predict.add_argument("-i", "--input_path", help="Path to the input image directory", type=str, required=True)
    predict.add_argument("-o", "--output_path", help="Path to the output image directory", type=str, required=True)
    predict.add_argument("--save-images", help="Save annotated images in the passed directory", action='store_true')
    parser.set_defaults(save_images=False)
    
    # Track mode parameters
    track.add_argument("-i", "--input_path", help="Path to the input image directory", type=str, required=True)
    track.add_argument("-o", "--output_path", help="Path to the output image directory", type=str, required=True)
    track.add_argument("--save-json", help="Save tracker results as a JSON file in the output directory.", action='store_true')
    track.add_argument("--save-vid", help="Save tracking results as a video in the output directory.", action='store_true')
    track.add_argument("--show-vid", help="Show tracking results in real-time as a video feed.", action='store_true')
    track.add_argument("--verbose", help="Show results in the terminal, useful for debugging", action='store_true')
    track.add_argument("--show-traj", help="Show trajectory of movement of tracked object", action='store_true')
    track.add_argument("--is-video", help="Specifies if the passed input is a video file or not", action='store_true')
    parser.set_defaults(save_json=False)
    parser.set_defaults(save_vid=False)
    parser.set_defaults(show_vid=False)
    parser.set_defaults(verbose=False)
    parser.set_defaults(show_traj=False)
    parser.set_defaults(is_video=False)
    
    # Dataset preparation mode parameters
    prepare_set.add_argument("-i", "--input_path", help="Path to the input dataset directory", type=str, required=True)

    args = parser.parse_args()
    
    if args.command == 'predict':
        try:
            check_if_folder(args.input_path, args.output_path)
            process_content(args.input_path, args.output_path, args.save_images)
        except InputDirectoryException:
            print(f"[WARN] INPUT FOLDER PASSED IS NOT A VALID DIRECTORY!")
        except OutputDirectoryException:
            print(f"[WARN] OUTPUT FOLDER PASSED IS NOT A VALID DIRECTORY!")

    if args.command == 'track':
        try:
            if not args.is_video:
                check_if_folder(args.input_path, args.output_path)
            # process_tracking(args.input_path, args.output_path, args.save_images)
            polfuncs.run_tracker(args.input_path, outfolder=Path(args.output_path),save_JSON=args.save_json, save_vid=args.save_vid, show_vid=args.show_vid, is_verbose=args.verbose, save_trajectories=args.show_traj)
        except InputDirectoryException:
            print(f"[WARN] INPUT FOLDER PASSED IS NOT A VALID DIRECTORY! DID YOU MEAN TO PASS A VIDEO FILE? CHECK --help")
        except OutputDirectoryException:
            print(f"[WARN] OUTPUT FOLDER PASSED IS NOT A VALID DIRECTORY!")
        
    if args.command == 'set-prep':
        try:
            check_dataset_folder(args.input_path)
            preprocess_dataset_directory(args.input_path)
            datum_pipeline(args.input_path)
            # next, just merge the two generated json files and add it to the final directory with all the images
            polfuncs.combine(f'{args.input_path}/daytime/annotations/instances_default.json', f'{args.input_path}/nighttime/annotations/instances_default.json', os.path.join(args.input_path, 'final/annotations/coco_annotation.json'))
            # remove all extracted and generated files for clean-up :)
            cleanup_files(args.input_path)
            print("[INFO] SUCCESS! DATASET IS NOW READY FOR TRAINNING!")
            # export_to_yolo(args.input_path)
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