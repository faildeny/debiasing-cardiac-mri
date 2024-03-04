from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

# model_checkpoint = './lightning_logs/version_21/checkpoints/epoch=0-step=30296.ckpt' # less than 1 epoch
# model_checkpoint = 'lightning_logs/version_23/checkpoints/epoch=3-step=121187.ckpt'
model_checkpoint = 'logs/Nov29_13-05-17_model_SD_1.5_128_lr_1e-05_sd_locked_True_control_locked_False/lightning_logs/version_0/checkpoints/epoch=24-step=94699.ckpt' # 128
model_checkpoint = 'logs/Nov30_12-47-55_model_SD_2.1_512_lr_1e-05_sd_locked_True_control_locked_False/lightning_logs/version_0/checkpoints/epoch=4-step=227229.ckpt' # 512
model_checkpoint = 'logs/Jan09_17-44-35_model_SD_2.1_512_lr_1e-05_sd_locked_True_control_locked_False_long_trained_25k/lightning_logs/version_0/checkpoints/epoch=8-step=409013.ckpt'
model_checkpoint = 'logs/Jan16_16-59-59_model_SD_2.1_512_lr_1e-05_sd_locked_True_control_locked_False_40k/lightning_logs/version_0/checkpoints/epoch=0-step=88586.ckpt'
model_checkpoint = 'models/_128_epoch=3-step=59059.ckpt'
model_checkpoint = 'logs/Feb25_11-03-17_model_SD_2.1_512_lr_1e-05_sd_lck_1_sd_f_hlf_1_c_lck_0continue/lightning_logs/version_0/checkpoints/epoch=2-step=265760.ckpt'
# model_checkpoint = 'logs/Feb08_17-48-07_model_SD_2.1_512_lr_1e-05_sd_lck_1_sd_f_hlf_1_c_lck_0continue/lightning_logs/version_0/checkpoints/epoch=2-step=265760.ckpt'
# model_checkpoint = 'logs/Jan26_16-28-11_model_SD_1.5_512_lr_2e-06_sd_locked_Falsesd_first_half_True_control_locked_True/lightning_logs/version_0/checkpoints/epoch=2-step=265760.ckpt'
# model_checkpoint = 'lightning_logs/version_43/checkpoints/epoch=18-step=71971.ckpt' # 128
# 
res = 512


model = create_model('./models/cldm_v21.yaml').cuda()
# model = create_model('./models/cldm_v15.yaml').cuda()
# model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(model_checkpoint, location='cpu'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = img
        # detected_map = np.zeros_like(img, dtype=np.uint8)
        # detected_map[np.min(img, axis=2) < 127] = 255

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        # control = torch.from_numpy(detected_map.copy()).float().cpu() / 255.0
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
        # x_samples = (einops.rearrange(x_samples, 'b c h w -> c h w b') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        # x_samples = np.transpose(x_samples, (0, 3, 1, 2))
        unpacked_samples = []
        unpacked_masks = []
        # Resize image and mask to requested resolution
        resized_samples = np.zeros((x_samples.shape[0], res, res, 3), dtype=np.uint8)

        for i in range(x_samples.shape[0]):
            resized_samples[i] = cv2.resize(x_samples[i], (res, res), interpolation=cv2.INTER_LANCZOS4)
        x_samples = resized_samples

        detected_map = cv2.resize(detected_map, (res, res), interpolation=cv2.INTER_LANCZOS4)

        unpacked_masks.append(detected_map)
        for channel in range(2):
            mask_1ch = detected_map[:, :, channel]
            mask_3ch = np.stack([mask_1ch, mask_1ch, mask_1ch], axis=2)
            unpacked_masks.append(mask_3ch)

        for i in range(x_samples.shape[0]):
            unpacked_samples.append(x_samples[i])
            for channel in range(2):
                sample_1ch = x_samples[i, :, :, channel]
                sample_3ch = np.stack([sample_1ch, sample_1ch, sample_1ch], axis=2)
                unpacked_samples.append(sample_3ch)
                
            unpacked_samples.extend(unpacked_masks)


        # print('x_samples.dtype', x_samples.dtype)
        # samples_unpacked = np.transpose(x_samples, (3, 1, 2, 0))
        # x_samples = samples_unpacked


        # results = [x_samples[i] for i in range(num_samples)]
        results = unpacked_samples
    # return [255 - detected_map] + results
    return results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Cardiac Segmentation Mask")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="Female, age in 70s, overweight BMI, healthy", placeholder="Female, age in 70s, overweight BMI, healthy")
            input_image = gr.Image(sources='upload', type="numpy", height=256)
            run_button = gr.Button(value="Generate")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=64, maximum=768, value=res, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", columns=3)
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0')
