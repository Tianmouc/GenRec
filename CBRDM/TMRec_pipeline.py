# This file is based on Marigold training script (https://github.com/prs-eth/marigold)
# with modifications by Tianmouc, 2025.
# These modifications are part of the work "Diffusion-Based Extreme High-speed Scenes Reconstruction
# with the Complementary Vision Sensor" published in ICCV 2025.
# Project repository: https://github.com/Tianmouc/GenRec
#
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------


import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DModel
)
from diffusers.utils import BaseOutput
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from CBRDM.unet2d_btchw import UNet2DModelBTCHW

class TMRecOutput(BaseOutput):

    rec_np: np.ndarray
    reclr_np: np.ndarray


class TianmoucSingleStageReconstructionPipeline(DiffusionPipeline):
    def __init__(
        self,
        unet: UNet2DModel,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        time_embedding = False, 
        without_td = False,
        without_sd = False,
                        
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):

        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
        )
        self.register_to_config(
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.time_embedding = time_embedding
        self.without_td = without_td
        self.without_sd = without_sd

    
    @torch.no_grad()
    def __call__(
        self,
        rgb_back: torch.Tensor, 
        td_back_accum: torch.Tensor, 
        rgb_front: torch.Tensor, 
        td_front_accum: torch.Tensor, 
        sd: torch.Tensor,
        time_step = None,
        lr = None,
        denoising_steps: Optional[int] = None,
        generator: Union[torch.Generator, None] = None,
        show_progress_bar: bool = True,
    ) -> TMRecOutput:
        
        # Model-specific optimal default values leading to fast and reasonable results.
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        rec_pred = self.single_infer(
                rgb_back = rgb_back, 
                td_back_accum = td_back_accum, 
                rgb_front = rgb_front, 
                td_front_accum = td_front_accum, 
                sd = sd,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
                lr = lr,
                time_step = time_step,
            )


        # Convert to numpy
        rec_pred = rec_pred.squeeze()
        rec_pred = rec_pred.cpu().numpy()

        # Clip output range
        rec_pred = rec_pred.clip(0, 1)

        return TMRecOutput(
            rec_np=rec_pred,
        )

    @torch.no_grad()
    def single_infer(
        self,
        rgb_back: torch.Tensor, 
        td_back_accum: torch.Tensor, 
        rgb_front: torch.Tensor, 
        td_front_accum: torch.Tensor, 
        sd: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
        lr = None,
        time_step = None,
    ) -> torch.Tensor:

        device = self.device
        rgb_back = rgb_back.to(device)
        td_back_accum = td_back_accum.to(device)
        rgb_front = rgb_front.to(device)
        td_front_accum = td_front_accum.to(device)
        sd = sd.to(device)
        if time_step is not None:
            time_step = time_step.to(device)
        if lr is not None:
            lr = lr.to(device)
            
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Initial depth map (noise) 
        latent_size = rgb_back.shape

        noisy_sample = torch.randn(
            latent_size,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )  # [B, 4, h, w]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:

            if self.without_td and self.without_sd:
                unet_input = torch.cat(
                    [rgb_back, rgb_front, noisy_sample], dim=-3
                    )
            elif self.without_sd:
                unet_input = torch.cat(
                    [rgb_back, td_back_accum, rgb_front, td_front_accum, noisy_sample], dim=-3
                )
            elif self.without_td:
                unet_input = torch.cat(
                    [rgb_back, rgb_front, sd, noisy_sample], dim=-3
                )
            else:
                unet_input = torch.cat(
                    [rgb_back, td_back_accum, rgb_front, td_front_accum, sd, noisy_sample], dim=-3
                )  # this order is important BTCHW
            
            if lr is not None:
                unet_input = torch.cat([lr, unet_input], dim=-3)

            # predict the noise residual
            if self.time_embedding:
                noise_pred = self.unet(
                    unet_input, t, class_labels=time_step*1000
                ).sample  # [B, 4, h, w]
            else:
                noise_pred = self.unet(
                    unet_input, t
                ).sample

            # compute the previous noisy sample x_t -> x_t-1
            noisy_sample = self.scheduler.step(
                noise_pred, t, noisy_sample, generator=generator
            ).prev_sample

        sample = noisy_sample

        # clip prediction and shift to [0, 1]
        sample = torch.clip(sample, -1.0, 1.0)
        sample = (sample + 1.0) / 2.0

        return sample
    
    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")



class TianmoucCascadedReconstructionPipeline(DiffusionPipeline):

    def __init__(
        self,
        unet: UNet2DModelBTCHW,
        sr_unet: UNet2DModel,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        BTCHW_mode = True,
        first_stage_time_embedding = False,
        sr_stage_time_embedding = True, 
        default_denoising_steps: Optional[int] = 200,
        default_sr_denoising_steps: Optional[int] = 50,
        default_processing_resolution: Optional[int] = None,
        max_eval_sr_bs = 8,
        midsize = (48,96),
        
    ):

        super().__init__()
        
        self.register_modules(
            unet=unet,
            sr_unet=sr_unet,
            scheduler=scheduler,
        )
        self.register_to_config(
            default_denoising_steps=default_denoising_steps,
            default_sr_denoising_steps=default_sr_denoising_steps,
            default_processing_resolution=default_processing_resolution,
            first_stage_time_embedding=first_stage_time_embedding,
            sr_stage_time_embedding=sr_stage_time_embedding,
            BTCHW_mode=BTCHW_mode,
            max_eval_sr_bs=max_eval_sr_bs
        )

        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution
        self.first_stage_time_embedding = first_stage_time_embedding
        self.sr_stage_time_embedding=sr_stage_time_embedding
        self.BTCHW_mode=BTCHW_mode
        self.default_sr_denoising_steps=default_sr_denoising_steps
        self.max_eval_sr_bs = max_eval_sr_bs
        self.midsize = midsize


    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    
    @torch.no_grad()
    def __call__(
        self,
        rgb_back: torch.Tensor, 
        td_back_accum: torch.Tensor, 
        rgb_front: torch.Tensor, 
        td_front_accum: torch.Tensor, 
        sd: torch.Tensor,
        time_step = None,  # B*T
        denoising_steps: Optional[int] = None,
        sr_denoising_steps: Optional[int] = None,
        generator: Union[torch.Generator, None] = None,
        show_progress_bar: bool = True,
        first_stage_repeat = 1,
    ) -> TMRecOutput:
        
        if self.first_stage_time_embedding or self.sr_stage_time_embedding:
            assert time_step is not None, "time_step should be provided when time embedding is used."
        
        # Model-specific optimal default values leading to fast and reasonable results.
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if sr_denoising_steps is None:
            sr_denoising_steps = self.default_sr_denoising_steps

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)
        self._check_inference_step(sr_denoising_steps)

        # STEP 1: first stage -- get LR result

        def resize_output(output_size, *tensors):
            resized_tensors = []
            for tensor in tensors:
                if tensor.ndim == 5: 
                    b, t, c, h, w = tensor.shape
                    tensor = tensor.view(-1, c, h, w)  # [B*T, C, H, W]
                    resized_tensor = TF.resize(tensor, output_size)
                    resized_tensor = resized_tensor.view(b, t, c, *output_size)
                elif tensor.ndim == 4: 
                    resized_tensor = TF.resize(tensor, output_size)
                else:
                    raise ValueError(f"Unsupported tensor dimensions: {tensor.ndim}")
                resized_tensors.append(resized_tensor)
            return tuple(resized_tensors)
        rgb_back_lr, td_back_accum_lr, rgb_front_lr, td_front_accum_lr, sd_lr = resize_output(self.midsize, rgb_back, td_back_accum, rgb_front, td_front_accum, sd)
        B, T, C, H_LR, W_LR = rgb_back_lr.shape

        if not self.BTCHW_mode:
            rgb_back_lr = rgb_back_lr.flatten(0, 1)
            td_back_accum_lr = td_back_accum_lr.flatten(0, 1)
            rgb_front_lr = rgb_front_lr.flatten(0, 1)
            td_front_accum_lr = td_front_accum_lr.flatten(0, 1)
            sd_lr = sd_lr.flatten(0, 1)
            if time_step is not None:
                time_step = time_step.flatten(0, 1)
        
        lr_preds = []

        for repeat_idx in range(first_stage_repeat):
            if self.first_stage_time_embedding:
                lr_pred = self.single_infer(
                    rgb_back = rgb_back_lr, 
                    td_back_accum = td_back_accum_lr, 
                    rgb_front = rgb_front_lr, 
                    td_front_accum = td_front_accum_lr, 
                    sd = sd_lr,
                    num_inference_steps=denoising_steps,
                    show_pbar=show_progress_bar,
                    generator=generator,
                    time_step = time_step,
                )
            else:
                lr_pred = self.single_infer(
                    rgb_back = rgb_back_lr, 
                    td_back_accum = td_back_accum_lr, 
                    rgb_front = rgb_front_lr, 
                    td_front_accum = td_front_accum_lr, 
                    sd = sd_lr,
                    num_inference_steps=denoising_steps,
                    show_pbar=show_progress_bar,
                    generator=generator,
                )

            lr_pred = lr_pred.clip(0, 1)

            lr_preds.append(lr_pred)

        lr_pred = torch.stack(lr_preds, dim=0).mean(dim=0)

        lr_pred = resize_output((320,640), lr_pred)[0]  # BTCHW
        lr_pred_norm = lr_pred * 2.0 - 1.0

        # STEP 2: SR stage -- get HR result

        if not self.BTCHW_mode:
            _, _, H_HR, W_HR = lr_pred.shape
        else:
            _, _, _, H_HR, W_HR = lr_pred.shape
        max_batch_size = self.max_eval_sr_bs  # Define the maximum batch size

        # Flatten tensors to B*T C H W
        rgb_back_flat = rgb_back.flatten(0, 1)
        td_back_accum_flat = td_back_accum.flatten(0, 1)
        rgb_front_flat = rgb_front.flatten(0, 1)
        td_front_accum_flat = td_front_accum.flatten(0, 1)
        sd_flat = sd.flatten(0, 1)
        if self.BTCHW_mode:
            lr_pred_norm_flat = lr_pred_norm.flatten(0, 1)
            if time_step is not None:
                time_step_flat = time_step.flatten(0, 1)
        else:
            lr_pred_norm_flat = lr_pred_norm
            if time_step is not None:
                time_step_flat = time_step

        rec_pred_parts = []

        total_frames = B * T
        for start_idx in range(0, total_frames, max_batch_size):
            end_idx = min(start_idx + max_batch_size, total_frames)
            if self.sr_stage_time_embedding:
                rec_pred_part = self.single_infer(
                    rgb_back=rgb_back_flat[start_idx:end_idx],
                    td_back_accum=td_back_accum_flat[start_idx:end_idx],
                    rgb_front=rgb_front_flat[start_idx:end_idx],
                    td_front_accum=td_front_accum_flat[start_idx:end_idx],
                    sd=sd_flat[start_idx:end_idx],
                    num_inference_steps=sr_denoising_steps,  # denoising_steps
                    show_pbar=show_progress_bar,
                    generator=generator,
                    lr=lr_pred_norm_flat[start_idx:end_idx],
                    time_step=time_step_flat[start_idx:end_idx],
                )
            else:
                rec_pred_part = self.single_infer(
                    rgb_back=rgb_back_flat[start_idx:end_idx],
                    td_back_accum=td_back_accum_flat[start_idx:end_idx],
                    rgb_front=rgb_front_flat[start_idx:end_idx],
                    td_front_accum=td_front_accum_flat[start_idx:end_idx],
                    sd=sd_flat[start_idx:end_idx],
                    num_inference_steps=sr_denoising_steps,  # denoising_steps
                    show_pbar=show_progress_bar,
                    generator=generator,
                    lr=lr_pred_norm_flat[start_idx:end_idx],
                )
            rec_pred_parts.append(rec_pred_part)

        # Concatenate results and reshape to the original shape
        rec_pred = torch.cat(rec_pred_parts, dim=0)
        rec_pred = rec_pred.view(B, T, C, H_HR, W_HR)


        # Convert to numpy
        # rec_pred = rec_pred.squeeze()
        rec_pred = rec_pred.cpu().numpy()

        # Clip output range
        rec_pred = rec_pred.clip(0, 1)

        return TMRecOutput(
            rec_np=rec_pred,
            reclr_np=lr_pred.cpu().numpy().clip(0, 1)
        )

    @torch.no_grad()
    def single_infer(
        self,
        rgb_back: torch.Tensor, 
        td_back_accum: torch.Tensor, 
        rgb_front: torch.Tensor, 
        td_front_accum: torch.Tensor, 
        sd: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
        lr = None,
        time_step = None,
    ) -> torch.Tensor:

        device = self.device
        rgb_back = rgb_back.to(device)
        td_back_accum = td_back_accum.to(device)
        rgb_front = rgb_front.to(device)
        td_front_accum = td_front_accum.to(device)
        sd = sd.to(device)
        if time_step is not None:
            time_step = time_step.to(device)
        if lr is not None:
            lr = lr.to(device)
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Initial depth map (noise)
        noisy_sample = torch.randn(
            rgb_back.shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )  # [B, c, h, w]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_back, td_back_accum, rgb_front, td_front_accum, sd, noisy_sample], dim=-3
            )  # this order is important BTCHW or BCHW, note that C in dim -3
            if lr is not None:
                unet_input = torch.cat([lr, unet_input], dim=-3)

            # predict the noise residual
            if time_step is not None:
                if lr is not None:
                    noise_pred = self.sr_unet(
                        unet_input, t, class_labels=time_step*1000
                    ).sample
                else:
                    noise_pred = self.unet(
                        unet_input, t, class_labels=time_step*1000
                    ).sample
            else:
                if lr is not None:
                    noise_pred = self.sr_unet(
                        unet_input, t
                    ).sample
                else:
                    noise_pred = self.unet(
                        unet_input, t
                    ).sample

            # compute the previous noisy sample x_t -> x_t-1
            noisy_sample = self.scheduler.step(
                noise_pred, t, noisy_sample, generator=generator
            ).prev_sample

        sample = noisy_sample

        # clip prediction
        sample = torch.clip(sample, -1.0, 1.0)
        # shift to [0, 1]
        sample = (sample + 1.0) / 2.0

        return sample
