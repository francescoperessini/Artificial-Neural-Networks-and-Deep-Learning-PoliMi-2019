import os
import json
from collections import defaultdict

dst_dataset_dir = "../homework_1/data/New_Classification_Dataset/"

# Sub-directories
dst_training_path = os.path.join(dst_dataset_dir, "training")
dst_validation_path = os.path.join(dst_dataset_dir, "validation")

training_folders = sorted(os.listdir(dst_training_path))
validation_folders = sorted(os.listdir(dst_validation_path))

train_dict = defaultdict(list)

for sub_dir in training_folders:
    sub_files = sorted(os.listdir(os.path.join(dst_training_path, sub_dir)))
    for train_image in sub_files:
        key = sub_dir
        val = train_image
        train_dict[key].append(val)

validation_dict = defaultdict(list)

for sub_dir in validation_folders:
    sub_files = os.listdir(os.path.join(dst_validation_path, sub_dir))
    for validation_image in sub_files:
        key = sub_dir
        val = validation_image
        validation_dict[key].append(val)

master_dict = {'training': train_dict, 'validation': validation_dict}

with open('dataset_split.json', 'w') as json_file:
    json.dump(master_dict, json_file)
