'''
Install FiftyOne with pip as follows:
pip install fiftyone

and then run this python file and you should get a web page that visualizes the dataset for you!
'''

import fiftyone as fo

# A name for the dataset
name = "custom_set"

# The directory containing the dataset to import
IMAGES_DIR = "../Sample_dataset/final/images"
LABELS_DIR = "../Sample_dataset/final/annotations/coco_annotation.json"

# The splits to load
splits = ["train", "val"]


# Load the dataset, using tags to mark the samples in each split
# dataset = fo.Dataset(name)
# for split in splits:
#     dataset.add_dir(
#         dataset_dir=dataset_dir,
#         dataset_type=fo.types.YOLOv5Dataset,
#         split=split,
#         tags=split,
# )

coco_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=IMAGES_DIR,
    labels_path=LABELS_DIR,
    include_id=True,
)

session = fo.launch_app(coco_dataset)
session.wait()