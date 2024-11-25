from typing import Optional, Tuple, Dict, List, Callable
import torch
import numpy as np
from tqdm import tqdm
from ldm.modules.diffusionmodules.util import make_beta_schedule
from utils.image import (
    wavelet_reconstruction, adaptive_instance_normalization
)
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/respace.py
def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    try:
        # float64 as default. float64 is not supported by mps device.
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    except:
        # to be compatible with mps
        res = torch.from_numpy(arr.astype(np.float32)).to(device=timesteps.device)[timesteps].float()
        
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def customized_func(x, func_type = 1):
    if func_type == 1:
        return x**2
    elif func_type == 2:
        return x * (np.exp(x**0.3) + 10)
        
class SpacedSampler:
    """
    Implementation for spaced sampling schedule proposed in IDDPM. This class is designed
    for sampling ControlLDM.
    
    https://arxiv.org/pdf/2102.09672.pdf
    """
    
    def __init__(
        self,
        model: "ControlLDM",
        schedule: str="linear",
        var_type: str="fixed_small"
    ) -> "SpacedSampler":
        self.model = model
        self.original_num_steps = model.num_timesteps
        self.schedule = schedule
        self.var_type = var_type

    def make_schedule(self, num_steps: int) -> None:
        """
        Initialize sampling parameters according to `num_steps`.
        
        Args:
            num_steps (int): Sampling steps.

        Returns:
            None
        """
        # NOTE: this schedule, which generates betas linearly in log space, is a little different
        # from guided diffusion.
        original_betas = make_beta_schedule(
            self.schedule, self.original_num_steps, linear_start=self.model.linear_start,
            linear_end=self.model.linear_end
        )
        original_alphas = 1.0 - original_betas
        original_alphas_cumprod = np.cumprod(original_alphas, axis=0)

        # New add, 2024.01.20
        original_etas = (torch.linspace(0, 1, self.original_num_steps, dtype=torch.float64)).numpy()

        # calcualte betas for spaced sampling
        # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/respace.py
        used_timesteps = space_timesteps(self.original_num_steps, str(num_steps))
        print(f"timesteps used in spaced sampler: \n\t{sorted(list(used_timesteps))}")
        
        etas = [0.0]
        betas = []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(original_alphas_cumprod):
            if i in used_timesteps:
                # marginal distribution is the same as q(x_{S_t}|x_0)
                betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                etas.append(original_etas[i])

        assert len(betas) == num_steps
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        
        # New add, 2024.01.20
        etas = np.array(etas, dtype=np.float64)[::-1]
        print(len(etas), etas)
        self.etas = etas


        self.timesteps = np.array(sorted(list(used_timesteps)), dtype=np.int32) # e.g. [0, 10, 20, ...]
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (num_steps, )
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for er_sde_solver
        vp_alphas_cumprod = np.concatenate((np.array([1.0]), self.alphas_cumprod), axis=0)
        self.vp_alphas = np.sqrt(vp_alphas_cumprod)[::-1]
        self.vp_sigmas = np.sqrt(1.0 - vp_alphas_cumprod)[::-1]
        self.lambdas = np.sqrt(1.0 / vp_alphas_cumprod - 1.0)[::-1]

    def _predict_xstart_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor
    ) -> torch.Tensor:
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def predict_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        cfg_scale: float,
        uncond: Optional[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        if uncond is None or cfg_scale == 1.:
            model_output = self.model.apply_model(x, t, cond)
        else:
            # apply classifier-free guidance
            model_cond = self.model.apply_model(x, t, cond)
            model_uncond = self.model.apply_model(x, t, uncond)
            model_output = model_uncond + cfg_scale * (model_cond - model_uncond)
        
        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output
        return e_t

    @torch.no_grad()
    def DDIM_sample(
        self,
        steps: int,
        shape: Tuple[int],
        cond_img: torch.Tensor,
        positive_prompt: str,
        negative_prompt: str,
        x_T: Optional[torch.Tensor]=None,
        cfg_scale: float=1.,
        color_fix_type: str="none"
    ) -> torch.Tensor:

        self.make_schedule(num_steps=steps)
        device = next(self.model.parameters()).device
        b = shape[0]
    
        time_range = np.flip(self.timesteps) # [1000, 950, 900, ...]
        total_steps = len(self.timesteps)
        iterator = tqdm(time_range, desc="Spaced Sampler", total=total_steps)

        # positive_prompt = 'Super-resolution,4k,real-world'

        cond = {
            "c_latent": [self.model.apply_condition_encoder(cond_img)],
            "c_crossattn": [self.model.get_learned_conditioning([positive_prompt] * b)]
        }
        uncond = {
            "c_latent": [self.model.apply_condition_encoder(cond_img)],
            "c_crossattn": [self.model.get_learned_conditioning([negative_prompt] * b)]
        }
        
        encoder_posterior = self.model.encode_first_stage(x_T * 2 - 1)
        x_start = self.model.get_first_stage_encoding(encoder_posterior).detach()

        start_step = 0
        img = self.vp_alphas[start_step] * x_start + self.vp_sigmas[start_step] * torch.randn_like(x_start)

        # img = torch.randn(shape, dtype=torch.float32, device=device)

        for i, step in enumerate(iterator):
            if i >= start_step:
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                index = torch.full_like(ts, fill_value=total_steps - i - 1)

                # mean of posterior distribution q(x_{t-1}|x_t, x_0)
                e_t = self.predict_noise(
                    img, ts, cond, cfg_scale, uncond
                )
                pred_x0 = self._predict_xstart_from_eps(x_t=img, t=index, eps=e_t)
                # 
                # DDIM
                r_fn = (self.lambdas[i + 1] / self.lambdas[i])**2.0
                r_alphas = self.vp_alphas[i + 1] / self.vp_alphas[i]                
                noise = torch.randn_like(img) * np.sqrt(self.lambdas[i + 1]**2 - self.lambdas[i]**2 * r_fn**2) * self.vp_alphas[i + 1]
                img = r_alphas * r_fn * img + self.vp_alphas[i + 1] * (1 - r_fn) * pred_x0 + noise

        img_pixel = (self.model.decode_first_stage(img) + 1) / 2
        # apply color correction (borrowed from StableSR)
        if color_fix_type == "adain":
            img_pixel = adaptive_instance_normalization(img_pixel, cond_img)
        elif color_fix_type == "wavelet":
            img_pixel = wavelet_reconstruction(img_pixel, cond_img)
        else:
            assert color_fix_type == "none", f"unexpected color fix type: {color_fix_type}"
        return img_pixel

    @torch.no_grad()
    def DDIM_shift_sample(
        self,
        steps: int,
        shape: Tuple[int],
        cond_img: torch.Tensor,
        positive_prompt: str,
        negative_prompt: str,
        x_T: Optional[torch.Tensor]=None,
        cfg_scale: float=1.,
        color_fix_type: str="none"
    ) -> torch.Tensor:

        self.make_schedule(num_steps=steps)
        device = next(self.model.parameters()).device
        b = shape[0]
    
        time_range = np.flip(self.timesteps) # [1000, 950, 900, ...]
        total_steps = len(self.timesteps)
        iterator = tqdm(time_range, desc="Spaced Sampler", total=total_steps)

        cond = {
            "c_latent": [self.model.apply_condition_encoder(cond_img)],
            "c_crossattn": [self.model.get_learned_conditioning([positive_prompt] * b)]
        }
        uncond = {
            "c_latent": [self.model.apply_condition_encoder(cond_img)],
            "c_crossattn": [self.model.get_learned_conditioning([negative_prompt] * b)]
        }
        
        encoder_posterior = self.model.encode_first_stage(x_T * 2 - 1)
        x_start = self.model.get_first_stage_encoding(encoder_posterior).detach()
        start_step = 0
        img = self.vp_alphas[start_step] * x_start + self.vp_sigmas[start_step] * torch.randn_like(x_start)

        for i, step in enumerate(iterator):
            if i >= start_step:
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                index = torch.full_like(ts, fill_value=total_steps - i - 1)

                # mean of posterior distribution q(x_{t-1}|x_t, x_0)
                e_t = self.predict_noise(
                    img, ts, cond, cfg_scale, uncond
                )
                pred_x0 = self._predict_xstart_from_eps(x_t=img, t=index, eps=e_t)

                # shift transform
                pred_x0 = self.etas[i + 1] * x_start + (1 - self.etas[i + 1]) * pred_x0
                # DDIM
                r_fn = (self.lambdas[i + 1] / self.lambdas[i])**2.0
                r_alphas = self.vp_alphas[i + 1] / self.vp_alphas[i]                
                # noise = torch.randn_like(img) * np.sqrt(self.lambdas[i + 1]**2 - self.lambdas[i]**2 * r_fn**2) * self.vp_alphas[i + 1]
                img = r_alphas * r_fn * img + self.vp_alphas[i + 1] * (1 - r_fn) * pred_x0

        img_pixel = (self.model.decode_first_stage(img) + 1) / 2
        # apply color correction (borrowed from StableSR)
        if color_fix_type == "adain":
            img_pixel = adaptive_instance_normalization(img_pixel, cond_img)
        elif color_fix_type == "wavelet":
            img_pixel = wavelet_reconstruction(img_pixel, cond_img)
        else:
            assert color_fix_type == "none", f"unexpected color fix type: {color_fix_type}"
        return img_pixel


    # Rec    
    @torch.no_grad()
    def Shift_sample(
        self,
        steps: int,
        shape: Tuple[int],
        cond_img: torch.Tensor,
        positive_prompt: str,
        negative_prompt: str,
        x_T: Optional[torch.Tensor]=None,
        cfg_scale: float=1.,
        color_fix_type: str="none"
    ) -> torch.Tensor:

        self.make_schedule(num_steps=steps)
        device = next(self.model.parameters()).device
        b = shape[0]
    
        time_range = np.flip(self.timesteps) # [1000, 950, 900, ...]
        total_steps = len(self.timesteps)
        iterator = tqdm(time_range, desc="Spaced Sampler", total=total_steps)

        cond = {
            "c_latent": [self.model.apply_condition_encoder(cond_img)],
            "c_crossattn": [self.model.get_learned_conditioning([positive_prompt] * b)]
        }
        uncond = {
            "c_latent": [self.model.apply_condition_encoder(cond_img)],
            "c_crossattn": [self.model.get_learned_conditioning([negative_prompt] * b)]
        }
        
        encoder_posterior = self.model.encode_first_stage(x_T * 2 - 1)
        x_start = self.model.get_first_stage_encoding(encoder_posterior).detach()
        img = self.vp_alphas[0] * x_start + self.vp_sigmas[0] * torch.randn_like(x_start)

        for i, step in enumerate(iterator):
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            index = torch.full_like(ts, fill_value=total_steps - i - 1)

            # mean of posterior distribution q(x_{t-1}|x_t, x_0)
            e_t = self.predict_noise(
                img, ts, cond, cfg_scale, uncond
            )
            pred_x0 = self._predict_xstart_from_eps(x_t=img, t=index, eps=e_t)

            # shift transform
            pred_x0 = self.etas[i + 1] * x_start + (1 - self.etas[i + 1]) * pred_x0
            img = self.vp_alphas[i + 1] * pred_x0 + self.vp_sigmas[i + 1] * torch.randn_like(img)

        img_pixel = (self.model.decode_first_stage(img) + 1) / 2
        # apply color correction (borrowed from StableSR)
        if color_fix_type == "adain":
            img_pixel = adaptive_instance_normalization(img_pixel, cond_img)
        elif color_fix_type == "wavelet":
            img_pixel = wavelet_reconstruction(img_pixel, cond_img)
        else:
            assert color_fix_type == "none", f"unexpected color fix type: {color_fix_type}"
        return img_pixel

    @torch.no_grad()
    def ER_SDE_sample(
        self,
        steps: int,
        shape: Tuple[int],
        cond_img: torch.Tensor,
        positive_prompt: str,
        negative_prompt: str,
        x_T: Optional[torch.Tensor]=None,
        cfg_scale: float=1.,
        color_fix_type: str="none",
        fn_lambda = customized_func
    ) -> torch.Tensor:
        self.make_schedule(num_steps=steps)

        device = next(self.model.parameters()).device
        b = shape[0]
 
        time_range = np.flip(self.timesteps) # [1000, 950, 900, ...]
        total_steps = len(self.timesteps)
        iterator = tqdm(time_range, desc="Spaced Sampler", total=total_steps)

        nums_intergrate = 100.0
        nums_indices = np.arange(nums_intergrate, dtype=np.float64)

        cond = {
            "c_latent": [self.model.apply_condition_encoder(cond_img)],
            "c_crossattn": [self.model.get_learned_conditioning([positive_prompt] * b)]
        }
        uncond = {
            "c_latent": [self.model.apply_condition_encoder(cond_img)],
            "c_crossattn": [self.model.get_learned_conditioning([negative_prompt] * b)]
        }
        encoder_posterior = self.model.encode_first_stage(x_T * 2 - 1)
        x_start = self.model.get_first_stage_encoding(encoder_posterior).detach()
        img = self.vp_alphas[0] * x_start + self.vp_sigmas[0] * torch.randn_like(x_start)
        
        old_x0 = None
        for i, step in enumerate(iterator):
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            index = torch.full_like(ts, fill_value=total_steps - i - 1)

            # mean of posterior distribution q(x_{t-1}|x_t, x_0)
            e_t = self.predict_noise(
                img, ts, cond, cfg_scale, uncond
            )
            pred_x0 = self._predict_xstart_from_eps(x_t=img, t=index, eps=e_t) 
            # shift transform
            pred_x0 = self.etas[i + 1] * x_start + (1 - self.etas[i + 1]) * pred_x0            
            # ER-SDE-Sample
            r_fn = fn_lambda(self.lambdas[i + 1]) / fn_lambda(self.lambdas[i])
            r_alphas = self.vp_alphas[i + 1] / self.vp_alphas[i]
            noise = torch.randn_like(img) * np.sqrt(self.lambdas[i + 1]**2 - self.lambdas[i]**2 * r_fn**2) * self.vp_alphas[i + 1]
            if old_x0 == None or self.vp_sigmas[i + 1]==0:
                img = r_alphas * r_fn * img + self.vp_alphas[i + 1] * (1 - r_fn) * pred_x0 + noise
            else:
                lambda_indices = self.lambdas[i + 1] + nums_indices / nums_intergrate * (self.lambdas[i] - self.lambdas[i + 1])
                s_int = np.sum(1.0 / fn_lambda(lambda_indices) * (self.lambdas[i] - self.lambdas[i + 1]) / nums_intergrate)
                d_x0 = (pred_x0 - old_x0)/(self.lambdas[i] - self.lambdas[i - 1])
                img = r_alphas * r_fn * img + self.vp_alphas[i + 1] * (1 - r_fn) * pred_x0 + self.vp_alphas[i + 1] * (self.lambdas[i + 1] - self.lambdas[i] + s_int * fn_lambda(self.lambdas[i + 1])) * d_x0 + noise
            old_x0 = pred_x0

        img_pixel = (self.model.decode_first_stage(img) + 1) / 2
        # apply color correction (borrowed from StableSR)
        if color_fix_type == "adain":
            img_pixel = adaptive_instance_normalization(img_pixel, cond_img)
        elif color_fix_type == "wavelet":
            img_pixel = wavelet_reconstruction(img_pixel, cond_img)
        else:
            assert color_fix_type == "none", f"unexpected color fix type: {color_fix_type}"
        return img_pixel

    # 3-order
    @torch.no_grad()
    def ER_SDE_3_order_sample(
        self,
        steps: int,
        shape: Tuple[int],
        cond_img: torch.Tensor,
        positive_prompt: str,
        negative_prompt: str,
        x_T: Optional[torch.Tensor]=None,
        cfg_scale: float=1.,
        color_fix_type: str="none",
        fn_lambda = customized_func
    ) -> torch.Tensor:
        self.make_schedule(num_steps=steps)

        device = next(self.model.parameters()).device
        b = shape[0]
 
        time_range = np.flip(self.timesteps) # [1000, 950, 900, ...]
        total_steps = len(self.timesteps)
        iterator = tqdm(time_range, desc="Spaced Sampler", total=total_steps)

        nums_intergrate = 100.0
        nums_indices = np.arange(nums_intergrate, dtype=np.float64)

        cond = {
            "c_latent": [self.model.apply_condition_encoder(cond_img)],
            "c_crossattn": [self.model.get_learned_conditioning([positive_prompt] * b)]
        }
        uncond = {
            "c_latent": [self.model.apply_condition_encoder(cond_img)],
            "c_crossattn": [self.model.get_learned_conditioning([negative_prompt] * b)]
        }
        encoder_posterior = self.model.encode_first_stage(x_T * 2 - 1)
        x_start = self.model.get_first_stage_encoding(encoder_posterior).detach()
        img = self.vp_alphas[0] * x_start + self.vp_sigmas[0] * torch.randn_like(x_start)
        
        old_x0 = None
        old_d_x0 = None
        for i, step in enumerate(iterator):
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            index = torch.full_like(ts, fill_value=total_steps - i - 1)

            # mean of posterior distribution q(x_{t-1}|x_t, x_0)
            e_t = self.predict_noise(
                img, ts, cond, cfg_scale, uncond
            )
            pred_x0 = self._predict_xstart_from_eps(x_t=img, t=index, eps=e_t) 
            # shift transform
            pred_x0 = self.etas[i + 1] * x_start + (1 - self.etas[i + 1]) * pred_x0    

            # ER-SDE-Sample
            r_fn = fn_lambda(self.lambdas[i + 1]) / fn_lambda(self.lambdas[i])
            r_alphas = self.vp_alphas[i + 1] / self.vp_alphas[i]
            noise = torch.randn_like(img) * np.sqrt(self.lambdas[i + 1]**2 - self.lambdas[i]**2 * r_fn**2) * self.vp_alphas[i + 1]
            if old_x0 == None or self.vp_sigmas[i + 1]==0:
                img = r_alphas * r_fn * img + self.vp_alphas[i + 1] * (1 - r_fn) * pred_x0 + noise
                old_x0 = pred_x0
            elif (old_x0 !=None) and (old_d_x0 == None):
                lambda_indices = self.lambdas[i + 1] + nums_indices / nums_intergrate * (self.lambdas[i] - self.lambdas[i + 1])
                s_int = np.sum(1.0 / fn_lambda(lambda_indices) * (self.lambdas[i] - self.lambdas[i + 1]) / nums_intergrate)
                d_x0 = (pred_x0 - old_x0)/(self.lambdas[i] - self.lambdas[i - 1])
                img = r_alphas * r_fn * img + self.vp_alphas[i + 1] * (1 - r_fn) * pred_x0 + self.vp_alphas[i + 1] * (self.lambdas[i + 1] - self.lambdas[i] + s_int * fn_lambda(self.lambdas[i + 1])) * d_x0 + noise
                old_x0 = pred_x0
                old_d_x0 = d_x0
            else:
                lambda_indices = self.lambdas[i + 1] + nums_indices / nums_intergrate * (self.lambdas[i] - self.lambdas[i + 1])
                s_int = np.sum(1.0 / fn_lambda(lambda_indices) * (self.lambdas[i] - self.lambdas[i + 1]) / nums_intergrate)
                s_d_int = np.sum((lambda_indices - self.lambdas[i]) / fn_lambda(lambda_indices) * (self.lambdas[i] - self.lambdas[i + 1]) / nums_intergrate)
                d_x0 = (pred_x0 - old_x0)/(self.lambdas[i] - self.lambdas[i - 1])
                dd_x0 = 2 * (d_x0 - old_d_x0)/(self.lambdas[i] - self.lambdas[i - 2])
                img = r_alphas * r_fn * img + self.vp_alphas[i + 1] * (1 - r_fn) * pred_x0 \
                    + self.vp_alphas[i + 1] * (self.lambdas[i + 1] - self.lambdas[i] + s_int * fn_lambda(self.lambdas[i +1])) * d_x0 \
                    + self.vp_alphas[i + 1] * ((self.lambdas[i + 1] - self.lambdas[i])**2/2 + s_d_int * fn_lambda(self.lambdas[i + 1])) * dd_x0 + noise
                old_x0 = pred_x0
                old_d_x0 = d_x0
                
        img_pixel = (self.model.decode_first_stage(img) + 1) / 2
        # apply color correction (borrowed from StableSR)
        if color_fix_type == "adain":
            img_pixel = adaptive_instance_normalization(img_pixel, cond_img)
        elif color_fix_type == "wavelet":
            img_pixel = wavelet_reconstruction(img_pixel, cond_img)
        else:
            assert color_fix_type == "none", f"unexpected color fix type: {color_fix_type}"
        return img_pixel


def customized_func(x, func_type = 1):
    if func_type == 1:
        return x**2.0
    elif func_type == 2:
        return x * (np.exp(x**0.3) + 10)

