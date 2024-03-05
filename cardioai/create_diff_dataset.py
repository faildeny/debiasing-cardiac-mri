import torch
import pandas as pd
import time
import os
from torchvision.utils import save_image
import yaml
import torchio as tio
from tqdm import tqdm
from pathlib import Path

import cardioai.dataset as dataset
from cardioai.transforms import (
    get_dataset_torchio,
    get_transforms_torchio,
    prepare_batch,
)

BATCH_SIZE = 20
config_path = "config.yaml"
dataset_type = "stacked_EDES"
dataset_dir = "cache/"
dataset_suffix = "fold_0_prev_0_01"
dataset_name = dataset_type + "_" + dataset_suffix + "/"
num_workers = 16

def generate_prompt(metadata, phase):
    bmi = metadata["BMI.2.0"].values[0]
    age = metadata["Age"].values[0]
    sex = metadata["Sex.0.0"].values[0]
    label = metadata["label"].values[0]

    if bmi < 18.5:
        bmi = "underweight"
    elif bmi < 25:
        bmi = "normal"
    elif bmi < 30:
        bmi = "overweight"
    else:
        bmi = "obese"
    
    age = str(age)[0] + "0s"

    # if label == 0:
    #     label = "healthy"
    # else:
    #     label = "heart failure"

    full_label = ""
    if "healthy" in label:
        full_label += ", healthy"
    if "HF" in label:
        full_label += ", heart failure"
    if "MI" in label:
        full_label += ", myocardial infarction"
    if "IHD" in label:
        full_label += ", ischemic heart disease"
    if "AF" in label:
        full_label += ", atrial fibrillation"

    prompt = f"{sex}, age in {age}, {bmi} BMI{full_label}"

    return prompt


with open(config_path) as config_file:
    config = yaml.safe_load(config_file)


dataset_path = os.path.join(dataset_dir, dataset_name)
os.makedirs(dataset_path+"source", exist_ok=True)
os.makedirs(dataset_path+"target", exist_ok=True)

prompt_file = os.path.join(dataset_path, "prompt.json")
# open prompt file
prompt_file = open(prompt_file, "w")

traind_list, vald_list, testd_list = dataset.get_data_lists(config)
datad_list = traind_list
# datad_list = dataset.get_data_UKBB(
#             code=config["dataset"]["icd_code"],
#             use_masks=config["params"]["mask_images"],
#             debug=config["debug"],
#             config=config,
#         )
for sample in datad_list: sample["subset"] = "train"
training_transforms = get_transforms_torchio(config, test=True)
ds = get_dataset_torchio(datad_list, training_transforms)
dataloader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers
    )
# ids_labeled = {sample["id"]: int(sample["label"][1].item()) for sample in datad_list}
ids_labeled = {sample["id"]: sample["text_label"] for sample in datad_list}
ids_labeled = pd.DataFrame([[id, ids_labeled[id]] for id in ids_labeled], columns=["f.eid", "label"])
metadata = dataset.get_metadata_for_UKBB(ids_labeled, config["dataset"]["UKBB"]["dataset_path"])

if dataset_type == "stacked_EDES" or dataset_type == "stacked_EDES_masked":
    phases = ["EDES"]
else:
    phases = ["ED", "ES"]
    
print(f"Generating dataset of {len(ds)} samples")

for batch in tqdm(dataloader):
    image_names = batch["image_path"]
    batch = prepare_batch(batch, config)
    inputs, targets, masks, id = batch["image"], batch["label"], batch["mask"], batch["id"]

    for image, label, mask, id, image_path in zip(inputs, targets, masks, id, image_names):
        sample_metadata = metadata[metadata["f.eid"] == int(id)]

        image_name = Path(image_path).stem
        for index, phase in enumerate(phases):
            if len(phases) == 1:
                subimage = image
                submask = mask
            else:
                subimage = image[index]
                submask = mask[index]
            image_path = "target/" + image_name + "_" + phase + ".png"
            mask_path = "source/" + image_name + "_" + phase + "_mask.png"
            prompt = generate_prompt(sample_metadata, phase)
            prompt_line = "{\"source\": \"" + mask_path + "\", \"target\": \"" + image_path + "\", \"prompt\": \"" + prompt + "\"}\n"
            prompt_file.write(prompt_line)
            save_image(subimage, dataset_path + image_path)
            submask = submask.float()/255.0
            save_image(submask, dataset_path + mask_path)
            
prompt_file.close()

print(f"Generated {len(ds)} samples")
