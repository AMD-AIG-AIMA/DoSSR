from typing import List, Tuple, Optional
import os
import math
from argparse import ArgumentParser, Namespace
import time
import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from ldm.xformers_state import disable_xformers
from model.shift_sampler import SpacedSampler

from model.cldm import ControlLDM
from model.cond_fn import MSEGuidance
import esrgan.RRDBNet_arch as arch
from utils.image import auto_resize, pad
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts


@torch.no_grad()
def process(
    model: ControlLDM,
    control_imgs: List[np.ndarray],
    start_imgs: List[np.ndarray],
    steps: int,
    strength: float,
    color_fix_type: str,
    tiled: bool,
    tile_size: int,
    tile_stride: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Apply DiffBIR model on a list of low-quality images.
    
    Args:
        model (ControlLDM): Model.
        control_imgs (List[np.ndarray]): A list of low-quality images (HWC, RGB, range in [0, 255]).
        steps (int): Sampling steps.
        strength (float): Control strength. Set to 1.0 during training.
        color_fix_type (str): Type of color correction for samples.
        disable_preprocess_model (bool): If specified, preprocess model (SwinIR) will not be used.
        cond_fn (Guidance | None): Guidance function that returns gradient to guide the predicted x_0.
        tiled (bool): If specified, a patch-based sampling strategy will be used for sampling.
        tile_size (int): Size of patch.
        tile_stride (int): Stride of sliding patch.
    
    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
        stage1_preds (List[np.ndarray]): Outputs of preprocess model (HWC, RGB, range in [0, 255]). 
            If `disable_preprocess_model` is specified, then preprocess model's outputs is the same 
            as low-quality inputs.
    """
    n_samples = len(control_imgs)
    sampler = SpacedSampler(model, var_type="fixed_small") 
    control = torch.tensor(np.stack(control_imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()
    start_imgs = torch.tensor(np.stack(start_imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    start_imgs = einops.rearrange(start_imgs, "n h w c -> n c h w").contiguous()
    # pre-process
    with torch.no_grad():
        start_imgs = model.preprocess_model(start_imgs).float().clamp_(0, 1)

    control = start_imgs
    model.control_scales = [strength] * 13
    height, width = control.size(-2), control.size(-1)
    shape = (n_samples, 4, height // 8, width // 8)
    # default sampler order 3, step 5
    # recommend order 1 for more steps, such as step=10+
    samples = sampler.shift_sample_3_order(
            steps=steps, shape=shape, cond_img=control, x_T=start_imgs,
            cfg_scale=1.0, color_fix_type=color_fix_type
        )
    
    x_samples = samples.clamp(0, 1)
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    control = (einops.rearrange(control, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    preds = [x_samples[i] for i in range(n_samples)]
    return preds


def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    # TODO: add help info for these options
    parser.add_argument("--ckpt", required=True, type=str, help="full checkpoint path")
    parser.add_argument("--config", required=True, type=str, help="model config path")
    
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--sr_scale", type=float, default=1)
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--disable_preprocess_model", action="store_true")
    
    # patch-based sampling
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_stride", type=int, default=256)
    
    # latent image guidance
    parser.add_argument("--use_guidance", action="store_true")
    parser.add_argument("--g_scale", type=float, default=0.0)
    parser.add_argument("--g_t_start", type=int, default=1001)
    parser.add_argument("--g_t_stop", type=int, default=-1)
    parser.add_argument("--g_space", type=str, default="latent")
    parser.add_argument("--g_repeat", type=int, default=5)
    
    parser.add_argument("--color_fix_type", type=str, default="wavelet", choices=["wavelet", "adain", "none"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--show_lq", action="store_true")
    parser.add_argument("--skip_if_exist", action="store_true")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    
    return parser.parse_args()

def check_device(device):
    if device == "cuda":
        # check if CUDA is available
        if not torch.cuda.is_available():
            print("CUDA not available because the current PyTorch install was not "
                "built with CUDA enabled.")
            device = "cpu"
    else:
        # xformers only support CUDA. Disable xformers when using cpu or mps.
        disable_xformers()
        if device == "mps":
            # check if MPS is available
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print("MPS not available because the current PyTorch install was not "
                        "built with MPS enabled.")
                    device = "cpu"
                else:
                    print("MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine.")
                    device = "cpu"
    print(f'using device {device}')
    return device

def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed) 
    args.device = check_device(args.device)

    model: ControlLDM = instantiate_from_config(OmegaConf.load(args.config))
    load_state_dict(model, torch.load(args.ckpt, map_location="cpu"), strict=True)
    model.freeze()
    model.to(args.device)

    assert os.path.isdir(args.input)
    
    for file_path in list_image_files(args.input, follow_links=True):
        lq = Image.open(file_path).convert("RGB")
        orignal_lq = np.array(lq)

        if args.sr_scale != 1:
            lq = lq.resize(
                tuple(math.ceil(x * args.sr_scale) for x in lq.size),
                Image.BICUBIC
            )
        if not args.tiled:
            lq_resized = auto_resize(lq, 512)
        else:
            lq_resized = auto_resize(lq, args.tile_size)
        x = pad(np.array(lq_resized), scale=64)
 
        for i in range(args.repeat_times):
            start_time = time.time()

            save_path = os.path.join(args.output, os.path.relpath(file_path, args.input))
            parent_path, stem, _ = get_file_name_parts(save_path)
            save_path = os.path.join(parent_path, f"{stem}_{i}.png")
            if os.path.exists(save_path):
                if args.skip_if_exist:
                    print(f"skip {save_path}")
                    continue
                else:
                    raise RuntimeError(f"{save_path} already exist")
            os.makedirs(parent_path, exist_ok=True)
                   
            preds = process(
                model, [x], [orignal_lq], steps=args.steps,
                strength=1,
                color_fix_type=args.color_fix_type,
                tiled=args.tiled, tile_size=args.tile_size, tile_stride=args.tile_stride
            )
            pred = preds[0]
            
            # remove padding
            pred = pred[:lq_resized.height, :lq_resized.width, :]
            Image.fromarray(pred).resize(lq.size, Image.LANCZOS).save(save_path)
            
            print(f"save to {save_path}")
            end_time = time.time()      
            execution_time = end_time - start_time
            print("Execution time: {:.2f} seconds".format(execution_time))


if __name__ == "__main__":
    main()
