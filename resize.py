import os
import shutil
import cv2
from tqdm import tqdm

directory = "./training/dataset_512"
subdirs = ["source", "target"]


target_size = 512

counter = 0

for image_dir in subdirs:
    path = os.path.join(directory, image_dir)
    out_directory = os.path.join(directory + "_resized_" + str(target_size))
    output_path = os.path.join(out_directory, image_dir)
    os.makedirs(output_path, exist_ok=True)
    if image_dir == "target":
        interpolation_method = cv2.INTER_AREA
    else:  
        interpolation_method = cv2.INTER_NEAREST

    for filename in tqdm(os.listdir(path)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(path, filename))
            img = cv2.resize(img, (target_size, target_size), interpolation=interpolation_method)
            cv2.imwrite(os.path.join(output_path, filename), img)
            counter += 1

prompt_path = os.path.join(directory, "prompt.json")
shutil.copy(prompt_path, os.path.join(out_directory, "prompt.json"))

print("Resized ", counter, " images.")