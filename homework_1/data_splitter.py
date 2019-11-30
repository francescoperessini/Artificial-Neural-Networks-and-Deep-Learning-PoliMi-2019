import os
import shutil
import numpy as np
classes = ['airplanes', 'bear', 'calculator', 'computer-monitor', 'fireworks', 'galaxy', 'grand-piano', 'kangaroo',
           'laptop', 'lightbulb', 'lightning', 'mountain-bike', 'owl', 'school-bus', 'sheet-music', 'skyscraper',
           'sword', 't-shirt', 'waterfall', 'wine-bottle']


dataset_dir = "../homework_1/data/Classification_Dataset/"
# Sub-directories
test_path = os.path.join(dataset_dir, "test")
training_path = os.path.join(dataset_dir, "training")


dst_dataset_dir = "../homework_1/data/New_Classification_Dataset/"

# Sub-directories
dst_test_path = os.path.join(dst_dataset_dir, "test")
dst_validation_path = os.path.join(dst_dataset_dir, "validation")
dst_training_path = os.path.join(dst_dataset_dir, "training")


def remove():
    try:
        shutil.rmtree(dst_dataset_dir)
    except OSError:
        print("Nothing to delete...")


ratio = [0.2, 0.8]

# create root folder:
#   -training       ---> split
#   -validation     ---> split
#   -test           ---> the given one

# Try to delete the old destination folder
remove()

# Try to create the new destination folder
try:
    os.mkdir(dst_dataset_dir)
    shutil.copytree(test_path, dst_test_path)

    f_name = os.listdir(training_path)

    # Create the new folder in the dest folder
    path_dst_training = os.path.join(dst_dataset_dir, "training")
    os.mkdir(path_dst_training)

    path_dst_validation = os.path.join(dst_dataset_dir, "validation")
    os.mkdir(path_dst_validation)

    for sub_dir in f_name:
        sub_files = os.listdir(os.path.join(training_path, sub_dir))
        size = len(sub_files)
        mask = [True] * (int(len(sub_files)*ratio[0]))
        mask += [False] * (int(len(sub_files) * ratio[1]))

        np.random.shuffle(mask)

        res_validation = [x for x, y in zip(sub_files, mask) if y]
        res_training = [x for x, y in zip(sub_files, mask) if not y]

        temp_dest_validation = os.path.join(path_dst_validation, sub_dir)
        os.mkdir(temp_dest_validation)

        temp_dest_training = os.path.join(path_dst_training, sub_dir)
        os.mkdir(temp_dest_training)

        # Iterate on the "selected" image and move them to the new
        for r in res_validation:
            temp_src = os.path.join(training_path, os.path.join(sub_dir, r))
            shutil.copyfile(temp_src, os.path.join(temp_dest_validation, r))

        for r in res_training:
            temp_src = os.path.join(training_path, os.path.join(sub_dir, r))
            shutil.copyfile(temp_src, os.path.join(temp_dest_training, r))

except OSError:
    print("Failed")
else:
    print("Folder created...")
