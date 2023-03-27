'''
Install FiftyOne with pip as follows:
pip install fiftyone

and then run this python file and you should get a web page that visualizes the dataset for you!
'''

import fiftyone as fo

# A name for the dataset
name = "sample_subset"

# The directory containing the dataset to import

IMAGES_DIR = "images"          # INPUT DIRECTORY OF IMAGES
LABELS_DIR = "out/output.json" # JSON DIRECTORY FOR PREDICTIONS

coco_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=IMAGES_DIR,
    labels_path=LABELS_DIR,
    include_id=True,
    name=name,
)

session = fo.launch_app(coco_dataset)
session.wait()