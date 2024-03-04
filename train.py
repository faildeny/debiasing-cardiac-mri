from share import *
from datetime import datetime
import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.profiler import PyTorchProfiler
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Get gpu id from command line
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
gpus = [args.gpu]


model = '2.1'
stage = 'control'
size = 512
balanced_sampling = True

model_definition = './models/cldm_v21.yaml'
resume_path = './models/control_sd21_ini.ckpt'

batch_size = 1
if size == 128:
    dataset_path = "./training/dataset_128"
elif size == 512:
    dataset_path = "./training/dataset_512"
else:
    raise Exception("Invalid size")

logger_freq = 300
only_mid_control = False

if stage == 'SD':
    sd_locked = False
    sd_locked_first_half = True
    control_locked = True
    learning_rate = 2e-6
elif stage == 'control':
    sd_locked = True
    sd_locked_first_half = True
    control_locked = False
    learning_rate = 1e-5

appendix = "_model_SD_" + model + "_" + str(size) + "_lr_" + str(learning_rate) + "_sd_lck_" + str(int(sd_locked)) + "_sd_f_hlf_" + str(int(sd_locked_first_half)) + "_c_lck_" + str(int(control_locked))
current_time = datetime.now().strftime("%b%d_%H-%M-%S")
log_dir = os.path.join("logs", current_time + appendix)
os.makedirs(log_dir, exist_ok=True)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(model_definition).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.sd_locked_first_half = sd_locked_first_half
model.control_locked = control_locked
model.only_mid_control = only_mid_control

logger_params = dict(sample = True, plot_denoise_rows= False, plot_diffusion_rows= False, unconditional_guidance_scale=6.0)

# Misc
dataset = MyDataset(dataset_path, sample_weight_clipping=20)
if balanced_sampling:
    sampler = WeightedRandomSampler(dataset.sample_weights, len(dataset))
    dataloader = DataLoader(dataset, num_workers=20, batch_size=batch_size, sampler=sampler)
else:
    dataloader = DataLoader(dataset, num_workers=20, batch_size=batch_size, shuffle=True)

logger = ImageLogger(batch_frequency=logger_freq, log_images_kwargs=logger_params)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], default_root_dir=log_dir, accumulate_grad_batches=2)


# Train!
trainer.fit(model, dataloader)
