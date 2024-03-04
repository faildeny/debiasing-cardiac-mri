from share import *
import config
import os
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm

from tqdm import tqdm
import cv2
import einops
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, WeightedRandomSampler
from tutorial_dataset import MyDataset

from pytorch_lightning import seed_everything
from torchvision.utils import make_grid
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from ldm.util import log_txt_as_img

random.seed(10)

time_schedule = False
start_sleep_hour = 8
end_sleep_hour = 21

# Set which gpu to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

output_size = 120

# model_checkpoint = './lightning_logs/version_21/checkpoints/epoch=0-step=30296.ckpt' # less than 1 epoch
# model_checkpoint = 'lightning_logs/version_23/checkpoints/epoch=3-step=121187.ckpt' # 512
# model_checkpoint = 'lightning_logs/version_43/checkpoints/epoch=18-step=71971.ckpt' # 128 
# model_checkpoint = "logs/Jan16_16-59-59_model_SD_2.1_512_lr_1e-05_sd_locked_True_control_locked_False_40k/lightning_logs/version_0/checkpoints/epoch=0-step=88586.ckpt"
# model_checkpoint = "logs/Jan26_16-28-11_model_SD_1.5_512_lr_2e-06_sd_locked_Falsesd_first_half_True_control_locked_True/lightning_logs/version_0/checkpoints/epoch=2-step=265760.ckpt"
# model_checkpoint = "logs/Feb16_17-13-55_model_SD_2.1_512_lr_1e-05_sd_lck_1_sd_f_hlf_1_c_lck_0continue/lightning_logs/version_0/checkpoints/epoch=1-step=177173.ckpt"
# model_checkpoint = "logs/Feb20_23-20-18_model_SD_2.1_512_lr_1e-05_sd_lck_1_sd_f_hlf_1_c_lck_0continue/lightning_logs/version_0/checkpoints/epoch=1-step=177173.ckpt"
model_checkpoint = "logs/Feb25_11-03-17_model_SD_2.1_512_lr_1e-05_sd_lck_1_sd_f_hlf_1_c_lck_0continue/lightning_logs/version_0/checkpoints/epoch=2-step=265760.ckpt"
# model_checkpoint = "models/_128_SD_epoch=2-step=44294.ckpt"
# model_checkpoint = "models/_128_SD_epoch=5-step=88589.ckpt"


if '_512_' in model_checkpoint:
    source_dataset_path = "/data/stacked_EDES_fold_0_prev_0_01_resized_512/"
elif '_128_' in model_checkpoint:
    source_dataset_path = "/data/stacked_EDES_fold_0_prev_0_01_resized_128/"
else:
    raise ValueError("Unrecognized image resoultion")
    
# Features for random prompt generation
sex = ['Male', 'Female']
age = ['age in 50s', 'age in 60s', 'age in 70s', 'age in 80s', 'age in 90s']
bmi = ['normal BMI', 'overweight BMI', 'obese BMI']
# diagnosis = ['heart failure']
diagnosis = ['healthy', 'heart failure']
# diagnosis = ['healthy', 'atrial fibrillation', 'ischemic heart disease', 'myocardial infarction', 'heart failure']

features = [sex, age, bmi, diagnosis]

# Initialize model
model = create_model('./models/cldm_v21.yaml').cuda()
# model = create_model('./models/cldm_v15.yaml').cuda()
model.load_state_dict(load_state_dict(model_checkpoint, location='cpu'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

# model = None
dataset = MyDataset(source_dataset_path)


# Load paths to masks with diagnosed heart failure
def get_masks_list(dataset, feature_to_find, exclude=False):
    """
    Get list of masks with given feature in the prompt
    """
    samples_list = dataset.get_samples_list()
    masks_list = []
    for sample in samples_list:
        prompt = sample['prompt']
        mask_filename = os.path.join(dataset.dataset_path, sample['source'])
        if feature_to_find in prompt and not exclude:
            masks_list.append(mask_filename)
        if feature_to_find not in prompt and exclude:
            masks_list.append(mask_filename)

    print("Found {} masks with feature {}".format(len(masks_list), feature_to_find))

    return masks_list

diagnosed_masks = get_masks_list(dataset, feature_to_find="heart failure")
healthy_masks = get_masks_list(dataset, feature_to_find="heart failure", exclude=True)

def generate_prompt(features):
    """
    Create random prompt from given features
    """

    prompt = ""
    for feature in features:
        chosen_value = random.choice(feature)
        prompt += ", " + chosen_value
    prompt = prompt[2:]

    return prompt

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps=20, guess_mode=False, strength=1, scale=4, seed=-1, eta=0):
    """
    Generate synthetic image from input mask and prompt
    """
    if time_schedule:
        current_time = datetime.now().strftime("%H")
        # get day of the week
        day = datetime.now().weekday()
        current_time = int(current_time)
        if day < 5:
            if current_time >= start_sleep_hour and current_time < end_sleep_hour:
                print("Day and hour is: " + str(day) + " " + str(current_time))
                # Sleep until end of office hours
                time_to_sleep = end_sleep_hour - current_time
                print("Going to sleep for " + str(time_to_sleep) + " hours")
                time.sleep(time_to_sleep * 60 * 60)

    with torch.no_grad():
        
        img = input_image
        H, W, C = img.shape

        control = img.cuda()

        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
        
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        
    return x_samples

def generate_synthetic_copy(dataset_path, output_path=None, per_sample_multiplier=1, n_samples=None):
    """
    Generate synthetic version of dataset with real prompts and corresponding masks
    """

    out_dir = 'synthetic_dataset'

    if output_path is None:
        print(dataset_path)
        print(os.path.basename(dataset_path))
        output_name = os.path.basename(dataset_path) + "_synthetic"
        output_dir = os.path.join(out_dir, output_name)
    else:
        output_dir = output_path

    os.makedirs(output_dir, exist_ok=True)

    dataset = MyDataset(dataset_path, sample_weight_clipping=5)
    n_samples = len(dataset) if n_samples is None else n_samples
    sampler = WeightedRandomSampler(dataset.sample_weights, len(dataset))
    dataloader = DataLoader(dataset, num_workers=20, batch_size=1, sampler=sampler)
    counter = 0
    for item in tqdm(dataloader):
        jpg = item['jpg'][0]
        txt = item['txt'][0]
        hint = item['hint'][0]
        if "failure" not in txt:
            if "healthy" not in txt:
                continue
        filename = item['filename'][0]
        samples = process(hint, txt, "", "", per_sample_multiplier, hint.shape[1])
        for i in range(samples.shape[0]):
            sample = samples[i]
            # sample = sample.transpose(1, 2, 0)
            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
            hint = cv2.cvtColor(hint.numpy() * 255.0, cv2.COLOR_RGB2BGR)
            sample = cv2.resize(sample, (output_size, output_size))
            hint = cv2.resize(hint, (output_size, output_size), interpolation=cv2.INTER_NEAREST)

            filename = Path(filename).stem
            unique_id = str(random.getrandbits(128))[:4]
            output_path = os.path.join(output_dir, f'{filename}_synthetic_{unique_id}.png')
            # print(output_path)
            cv2.imwrite(output_path, sample)
            cv2.imwrite(output_path.replace(".png", "_mask.png"), hint)

            counter += 1
        if counter >= n_samples:
            break

def generate_images_with_parameter_array(output_dir, steps_min, steps_max, cfg_min, cfg_max):
    """
    Generate images with different parameters
    """
    prompt = generate_prompt(features)
    samples_list = dataset.get_samples_list()
    masks_list = []
    sample = random.choice(samples_list)
    prompt = sample['prompt']
    image = os.path.join(dataset.dataset_path, sample['target'])
    mask = os.path.join(dataset.dataset_path, sample['source'])

    mask = cv2.imread(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = mask.astype(np.float32) / 255.0
    mask = torch.from_numpy(mask)

    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = torch.from_numpy(image)
    image = image.transpose(0, 2).transpose(1, 2)

    seed = 1234

    os.makedirs(output_dir, exist_ok=True)
    # steps_list = list(range(steps_min, steps_max+1, 2))
    steps_list = [1, 2, 4, 5, 7, 11, 20, 50, 200]
    
    # cfg_list = list(range(cfg_min, cfg_max+1, 2))
    # cfg_list.insert(1, 1)
    # cfg_list.append(12)
    # cfg_list.append(20)
    cfg_list = [0, 1, 2, 4, 6, 8, 12, 20]
    row_length = len(cfg_list) + 1
    text_panel_size = (512, 512)

    steps_array = []
    columns = []
    # column_name = log_txt_as_img(text_panel_size, [f'CFG\nvs\nDDIM steps'], size=55)[0]
    # columns.append(column_name)
    columns.append(image)
    for cfg in cfg_list:
        column_name = log_txt_as_img(text_panel_size, [f'CFG: {cfg}'], size=55)[0]
        columns.append(column_name)
    steps_array.append(columns)
    for steps in steps_list:
        cfg_array = []
        cfg_array.append(log_txt_as_img(text_panel_size, [f'DDIM Steps: {steps}'], size=55)[0])
        for cfg in cfg_list:
            samples = process(mask, prompt, "", "", 1, mask.shape[1], ddim_steps=steps, guess_mode=False, strength=1, scale=cfg, seed=seed, eta=0)
            sample = samples[0]
            sample = torch.tensor(sample.transpose(2, 0, 1))
            cfg_array.append(sample)
        steps_array.append(cfg_array)

    grid_rows = []
    for row in steps_array:
        grid = make_grid(row, nrow=row_length)
        grid_rows.append(grid)


    grid = make_grid(grid_rows, nrow=1)
    # grid = (grid + 1.0) / 2.0
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = (grid).astype(np.uint8)

    filename = f'cfg_steps_array.png'
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, grid)
    print(f"Saved {output_path}")
    
def generate_random_prompt_dataset(output_dir, diagnosis_feature, n_samples = 10, use_negative_prompts=False):
    """
    Generate dataset with random prompts and real masks from patients diagnosed heart failure
    """

    # Generate all random inputs before, as random sampling over long time intervals
    # skews the distribution significantly

    random_inputs = []
    for i in range(n_samples):
        prompt = generate_prompt(features)
        if diagnosis_feature not in prompt:
            mask = random.choice(healthy_masks)
        else:
            mask = random.choice(diagnosed_masks)

        random_inputs.append((prompt, mask))

    os.makedirs(output_dir, exist_ok=False)

    for i, random_input in tqdm(enumerate(random_inputs)):
        prompt, mask = random_input
        if use_negative_prompts:
            if diagnosis_feature in prompt:
                negative_prompt = "healthy"
            else:
                negative_prompt = diagnosis_feature
        else:
            negative_prompt = ""

        # Extract id from mask filename
        mask_id = mask.split("/")[-1].split("_")[0]

        mask = cv2.imread(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = mask.astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)

        sample = process(mask, prompt, "", negative_prompt, 1, mask.shape[1], scale=9, strength=1)[0]

        filename = prompt.replace(", ", "_").replace(" ", "_")
        filename = f'{i}_{filename}_{mask_id}.png'
        output_path = os.path.join(output_dir, filename)
        sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
        mask = cv2.cvtColor(mask.numpy() * 255.0, cv2.COLOR_RGB2BGR)
        sample = cv2.resize(sample, (output_size, output_size))
        mask = cv2.resize(mask, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(output_path, sample)
        cv2.imwrite(output_path.replace(".png", "_mask.png"), mask)
    
    print("Generated random prompt dataset with {} samples".format(n_samples))


# Select one of the methods for synthetic dataset generation

start_time = time.time()
# generate_random_prompt_dataset("synthetic_dataset/random_dataset_40k_2_1_128_4_epoch2, 'heart failure', 5000, use_negative_prompts=False)
# generate_random_prompt_dataset("synthetic_dataset/random_dataset_40k_2_1_128_4_epoch_seed", 'heart failure', 5000, use_negative_prompts=False)
# generate_random_prompt_dataset("synthetic_dataset/128_SD_5", 'heart failure', 5000, use_negative_prompts=False)
generate_synthetic_copy(source_dataset_path, "synthetic_dataset/synth_copy_512_Control_1_6_more_balance", per_sample_multiplier=1, n_samples=10000)
# generate_images_with_parameter_array("grid_search/parameter_array", 1, 15, 0, 8)
# generate_synthetic_copy(source_dataset_path)

#Print execution time in hours
print("Execution time: ", (time.time() - start_time) / 3600)
