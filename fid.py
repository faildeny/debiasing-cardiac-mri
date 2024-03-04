import argparse
import os
import random
import shutil
import subprocess
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculates the Frechet Inception Distance between two distributions using RadImageNet model."
    )
    parser.add_argument(
        "dataset_path_1",
        type=str,
        help="Path to images from first dataset",
    )
    parser.add_argument(
        "dataset_path_2",
        type=str,
        help="Path to images from second dataset",
    )
    parser.add_argument(
        "--keyword_1",
        type=str,
        nargs="?",
        default=None,
        help="Keyword to search for in the first dataset",
    )
    parser.add_argument(
        "--keyword_2",
        type=str,
        nargs="?",
        default=None,
        help="Keyword to search for in the second dataset",
    )
    parser.add_argument(
        "--array_eval",
        action="store_true",
        help="If set, the multiple comparisons between sensitive feature subgroups are performed",
    )
    
    args = parser.parse_args()

    return args

def create_temporary_directory(directory, n_samples, lower_bound=False, only_unique_ids=True, keyword=None):
    # get dir 
    # generate random hash  
    temp_dir = os.path.join("tmp", directory + str(random.getrandbits(128)))
    os.makedirs(temp_dir, exist_ok=False)
    files_list = os.listdir(directory)
    # exclude files with "mask" in name
    files_list = [file for file in files_list if "mask" not in file]
    # Only keep files with keyword in name
    if keyword:
        files_list = [file for file in files_list if keyword in file]
    if len(files_list) < n_samples:
        print("Number of files with keyword", keyword, "in", directory, ":", len(files_list))
        print("Not enough samples. Duplicates will be used as an alternative.")
        # Add duplicates to the list
        files_list = files_list * (2*n_samples // len(files_list) + 1)
        print("Number of files after adding duplicates:", len(files_list))

    if lower_bound:
        # Extract ids from file names
        ids = [file.split("_")[0] for file in files_list]
        ids = list(set(ids))
        set1 = set(ids[:len(ids)//2])
        set2 = set(ids[len(ids)//2:])
        
        files_list1 = [file for file in files_list if file.split("_")[0] in set1]
        files_list2 = [file for file in files_list if file.split("_")[0] in set2]
        files_list = files_list1
        # print("Number of unique ids:", len(ids)) 
        # print("Number of unique ids in set1:", len(set1))
        # print("Number of unique ids in set2:", len(set2))
        # print("Number of files in set1:", len(files_list))
        # print("Number of files in set2:", len(files_list2))
        
    # Only keep full file names with unique id in name
    # if only_unique_ids:
    #     files_unique = []
    #     unique_ids = set()
    #     for file in files_list:
    #         # id = file.split("_")[0]
    #         id = file
    #         if id not in unique_ids:
    #             unique_ids.add(id)
    #             files_unique.append(file)
    #     files_list = files_unique

    # Get n random files
    print("Getting", n_samples, "random samples from", len(files_list), "samples")
    random_files_list = random.sample(files_list, n_samples)

    for file in random_files_list:
        shutil.copyfile(
            os.path.join(directory, file),
            os.path.join(temp_dir, file),
        )
        files_list.remove(file)

    if lower_bound:
        # copy the same images to another directory
        temp_dir2 = os.path.join("tmp", directory + str(random.getrandbits(128)))
        os.makedirs(temp_dir2, exist_ok=False)
        print("Getting", n_samples, "random samples from", len(files_list2), "samples")
        random_files_list = random.sample(files_list2, n_samples)

        for file in random_files_list:
            shutil.copyfile(
                os.path.join(directory, file),
                os.path.join(temp_dir2, file),
            )
        return temp_dir, temp_dir2
    

    return temp_dir


def calculate_fid(directory_1, directory_2, keyword_1=None, keyword_2=None, n_samples=500):

    if directory_1 == directory_2 and keyword_1 == keyword_2:
        lower_bound = True
        print("Calculating lower bound of FID between unique ids in the same dataset")
    else:
        lower_bound = False

    if lower_bound:
        temp_dir1, temp_dir2 = create_temporary_directory(directory_1, n_samples, lower_bound=True, keyword=keyword_1)
    else:
        temp_dir1 = create_temporary_directory(directory_1, n_samples, keyword=keyword_1)
        temp_dir2 = create_temporary_directory(directory_2, n_samples, keyword=keyword_2)

    # print("Calculated between", temp_dir1, "and", temp_dir2, "with", n_samples, "samples")

    # execute command to calculate FID
    command = f"python -m pytorch_fid {temp_dir1} {temp_dir2}"
    training_process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    
    # remove temporary directories
    shutil.rmtree(temp_dir1)
    shutil.rmtree(temp_dir2)

    for line in training_process.stdout.decode("utf-8").split("\n"):
        if "FID" in line:
            return float(line.split(" ")[-1])


if __name__ == "__main__":
    args = parse_args()

    directory_1 = args.dataset_path_1
    directory_2 = args.dataset_path_2
    keyword_1 = args.keyword_1
    keyword_2 = args.keyword_2
    full_evaluation = args.array_eval

    if full_evaluation:
        sex = ["Male", "Female"]
        bmi = ["normal", "obese"]
        age = ["in_50s", "in_70s"]
        age2 = ["in_50s", "in_80s"]
        disease = ["healthy", "heart_failure"]
        features = {"sex": sex, "bmi": bmi, "age": age, "disease": disease}
        # features = {"sex": sex, "bmi": bmi}
        repetitions = 4
        if directory_1 != directory_2:
            directories = [directory_1, directory_2]
        else:
            directories = [directory_1]
        # Calculate FID for all combinations in subgroup
        results = {}
        for directory_2 in directories:
            fids_per_feature = {}
            fids = []
            for run in range(repetitions):
                fid = calculate_fid(directory_1, directory_2)
                fids.append(fid)
            # Assign mean and standard deviation
            fids_per_feature["Base"] = [("FID", f'{np.mean(fids):.2f} ({np.std(fids):.2f})')]
            # fids_per_feature["Base"] = [("FID", f'{fid:.2f}')]

            for feature, keywords in features.items():
                for keyword_1 in keywords:
                    for keyword_2 in keywords:
                        print("Calculating FID between", keyword_1, "and", keyword_2, "in", feature)
                        fids = []
                        for run in range(repetitions):
                            fid = calculate_fid(directory_1, directory_2, keyword_1, keyword_2)
                            fids.append(fid)

                        if fids_per_feature.get(feature):
                            fids_per_feature[feature].append(((keyword_1, keyword_2), f'{np.mean(fids):.2f} ({np.std(fids):.2f})'))
                            # fids_per_feature[feature].append(((keyword_1, keyword_2), f'{fid:.2f}'))
                        else:
                            fids_per_feature[feature] = [((keyword_1, keyword_2), f'{np.mean(fids):.2f} ({np.std(fids):.2f})')]
                            # fids_per_feature[feature] = [((keyword_1, keyword_2), f'{fid:.2f}')]

            results[directory_2] = fids_per_feature
            
        for directory in directories:
            if directory_1 == directory:
                print("\nReal vs Real")
            else:
                print("\nReal vs Synthetic")
            fids_per_feature = results[directory]
            for feature, fids in fids_per_feature.items():
                print("\n", feature)
                for fid in fids:
                    print(fid[0], ":", fid[1])
        
    else:
        fid = calculate_fid(directory_1, directory_2, keyword_1, keyword_2)
        print("FID:", fid)


