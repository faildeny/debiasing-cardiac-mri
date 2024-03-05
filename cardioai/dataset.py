import os
import pandas as pd
import torch
import hashlib

def set_sample_weights(datad_list, balance_by_sensitive_features=False):
        '''
        Set sample weight for each sample in datad_list based on the number of samples in each metagroup
        if balance_by_sensitive_features is True
        Otherwise, set sample weight for each sample in datad_list based on the number of samples in each diagnosis label
        '''
        
        if balance_by_sensitive_features:
            # get number of samples for each metagroup in datad_list
            metagroups_counts = {}
            for sample in datad_list:
                metagroup = sample["meta_label"]
                if metagroup in metagroups_counts:
                    metagroups_counts[metagroup] += 1
                else:
                    metagroups_counts[metagroup] = 1
            # set weights for each metagroup
            for sample in datad_list:
                sample["sample_weight"] = 1 / metagroups_counts[sample["meta_label"]]    
        else:
            # get number of samples for each diagnosis label in datad_list
            labels_counts = {}
            for sample in datad_list:
                label = sample["label"][0].item()
                if sample["subset"] == "synthetic":
                    increment = 1
                else:
                    increment = 9
                if label in labels_counts:
                    labels_counts[label] += increment
                else:
                    labels_counts[label] = increment

            # set weights for each diagnosis label
            for sample in datad_list:
                sample["sample_weight"] = 1 / labels_counts[sample["label"][0].item()]
        
        return datad_list

def get_saved_split(config, only_test=False):
    """
    Load train, val and test ids from file
    """
    train_ids = open("data/ids_train.csv", "r").read().splitlines()
    val_ids = open("data/ids_val.csv", "r").read().splitlines()
    test_ids = open("data/ids_test.csv", "r").read().splitlines()

    dataset_file_path = config["dataset"]["UKBB"]["dataset_file"]
    config["dataset"]["icd_code"].insert(0, "healthy")
    codes = config["dataset"]["icd_code"]
    # Load dataset csv from file
    metadata = pd.read_csv(dataset_file_path)

    # Create list of samples with metadata
    datad_list = []
    for index, row in metadata.iterrows():
        id = str(row["f.eid"])
        # if 'healthy not in label, set label to 1
        if codes[1] in row["label"]:
            image_label = 1
        else:
            image_label = 0
        image_label = torch.nn.functional.one_hot(
            torch.as_tensor(image_label), num_classes=len(codes)
        ).float()

        meta_label = row["metagroup"]
        all_codes = [" healthy", " C09_HF", " C01_IHD", " C02_MI", " C04_AF"]
        # Remove code from all codes
        for code in codes:
            all_codes.remove(" "+code)
        # Remove all codes from label
        for code in all_codes:
            meta_label = meta_label.replace(code, "")
        # if no code in meta_label, set to healthy
        if codes[0] not in meta_label and codes[1] not in meta_label:
            meta_label = "healthy" + meta_label

        datad_list.append(
            {
                # "image": image_path,
                # "segmentation": seg_path,
                # "segmentation_es": seg_es_path,
                "label": image_label,
                "sample_weight": 0.5,
                "id": id,
                "meta_label": meta_label,
                "es_index": -1,
            }
        )
    datad_list = sorted(datad_list, key=lambda k: k["id"])

    traind_list = [d for d in datad_list if d["id"] in train_ids]
    vald_list = [d for d in datad_list if d["id"] in val_ids]
    testd_list = [d for d in datad_list if d["id"] in test_ids]

    train_hash = hashlib.sha256(str([d["id"] for d in traind_list]).encode('utf-8')).hexdigest()


    traind_list = set_prevalence(traind_list, config["dataset"]["train_prevalence"])
    vald_list = set_prevalence(vald_list, config["dataset"]["val_prevalence"])
    testd_list = set_prevalence(testd_list, config["dataset"]["test_prevalence"])

    # Add info about subset
    for sample in traind_list: sample["subset"] = "train"
    for sample in vald_list: sample["subset"] = "val"
    for sample in testd_list: sample["subset"] = "test"

    print(f"Loaded saved split with {len(traind_list)} train ids with hash: {train_hash}")

    if config["debug"]:
        traind_list = traind_list[:20]
        vald_list = vald_list[:20]
        testd_list = testd_list[:20]

    if only_test:
        return testd_list
    else:
        return traind_list, vald_list, testd_list

def load_synthetic_dataset(config):
    synthetic_path = config['dataset']['synthetic_path']
    n_samples = config['dataset']['synthetic_samples_to_load']
    files_list = os.listdir(synthetic_path)
    codes = config["dataset"]["icd_code"]
    datad_list = []
    num_positives = 0
    counter = 0
    for sample in files_list:
        if "mask" in sample:
            image_name = sample.replace("_mask", "")
            image_path = os.path.join(synthetic_path, image_name)
            seg_path = os.path.join(synthetic_path, sample)
            seg_es_path = seg_path
            image_label = 1 if "healthy" not in image_name else 0
            if image_label == 1:
                num_positives += 1
            image_label = torch.nn.functional.one_hot(torch.as_tensor(image_label), num_classes=2).float()
            es_frame = -1
            id = "synthetic_" + str(image_name.split("_")[0])

            metalabel = ""
            if "healthy" in sample:
                metalabel = "healthy"
            if "heart_failure" in sample:
                metalabel += " C09_HF"
            if "ischemic_heart_disease" in sample:
                metalabel += " C01_IHD"
            if "myocardial_infarction" in sample:
                metalabel += " C02_MI"
            if "atrial_fibrillation" in sample:
                metalabel += " C04_AF"
            if "Female" in sample:
                metalabel += "Female"
            if "Male" in sample:
                metalabel += "Male"
            if "40s" in sample:
                metalabel += "56.0"
            if "50s" in sample:
                metalabel += "56.0"
            if "60s" in sample:
                metalabel += "66.0"
            if "70s" in sample:
                metalabel += "73.0"
            if "80s" in sample:
                metalabel += "73.0"
            if "90s" in sample:
                metalabel += "73.0"
            if "underweight" in sample:
                metalabel += "1 Normal"
            if "normal" in sample:
                metalabel += "1 Normal"
            if "overweight" in sample:
                metalabel += "2 Overweight"
            if "obese" in sample:
                metalabel += "3 Obese"
            
            all_codes = [" healthy", " C09_HF", " C01_IHD", " C02_MI", " C04_AF"]
            # Remove code from all codes
            for code in codes:
                all_codes.remove(" "+code)
            # Remove all codes from label
            for code in all_codes:
                metalabel = metalabel.replace(code, "")

            datad_list.append(
                {
                    "image": image_path,
                    "segmentation": seg_path,
                    "segmentation_es": seg_es_path,
                    "label": image_label,
                    "sample_weight": 0.5,
                    "id": id,
                    "subset": "synthetic",
                    "meta_label": metalabel,
                    "es_index": es_frame,
                }
            )
            counter += 1
        if counter == n_samples:
            break
    
    print(
        f"Number of synthetic positives: {num_positives} and negatives: {len(datad_list)-num_positives}"
    )

    return datad_list

def get_data_lists(config, only_test=False):
    if config["dataset"]["name"] == "UKBB":
        if config["params"]["saved_split"]:
            return get_saved_split(config, only_test=only_test)
    raise ValueError("Dataset not found")

def set_prevalence(datad_list, prevalence):
    healthy_labels = [sample["label"][0].item() for sample in datad_list]
    num_prevalence_labels = sum([sample["label"][1].item() for sample in datad_list])
    num_other_unhealthy_labels = len(datad_list) - sum(healthy_labels) - num_prevalence_labels

    # Set prevalence for each subset
    labels, datad_list = (
        list(t)
        for t in zip(*sorted(zip(healthy_labels, datad_list), key=lambda x: x[0], reverse=False))
    )

    datad_list = datad_list[: int(num_other_unhealthy_labels + num_prevalence_labels // prevalence)]

    print("Subset size: ", len(datad_list))
    for label in range(len(datad_list[0]["label"])):
        print(f"Number of {label} class labels: {sum([sample['label'][label].item() for sample in datad_list])}")

    return datad_list
