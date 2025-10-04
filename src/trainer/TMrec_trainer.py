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
import os
import shutil
from datetime import datetime
from typing import List, Union
import torch.nn.functional as F
import cv2
import numpy as np
import torch
from diffusers import DDPMScheduler, DDIMScheduler
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from omegaconf import ListConfig
from src.util import metric
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger, eval_dic_to_text
from src.util.loss import get_loss
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.multi_res_noise import multi_res_noise_like
from src.util.seeding import generate_seed_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

class TMRec_BaseTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model,
        train_dataloader: DataLoader,
        device,
        base_ckpt_dir,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accumulation_steps: int,
        val_dataloaders: List[DataLoader] = None,
        vis_dataloaders: List[DataLoader] = None,
        real_unet_in_channels = 12,
    ):
        encoding_type = model.encoding_type if hasattr(model, "encoding_type") else None
        assert encoding_type == "empty_text" or encoding_type == "img_rgb" or encoding_type is None
        self.cfg: OmegaConf = cfg
        self.model = model
        self.device = device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders
        self.vis_loaders: List[DataLoader] = vis_dataloaders
        self.accumulation_steps: int = accumulation_steps
        self.encoding_type = encoding_type

        # Adapt input layers
        self.real_unet_in_channels = real_unet_in_channels
        
        self._init_before_train()

        self.model.unet.enable_xformers_memory_efficient_attention()

        # set optimizer
        self._set_optimizer()

        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs)

        # Training noise scheduler
        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            os.path.join(
                base_ckpt_dir,
                cfg.trainer.training_noise_scheduler.pretrained_path,
                "scheduler",
            )
        )
        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        assert (
            self.prediction_type == self.model.scheduler.config.prediction_type
        ), "Different prediction types"
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )

        # Eval metrics
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]
        self.train_metrics = MetricTracker(*["loss"])
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        # main metric for best checkpoint saving
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal
        assert (
            self.main_val_metric in cfg.eval.eval_metrics
        ), f"Main eval metric `{self.main_val_metric}` not found in evaluation metrics."
        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period

        # Multi-resolution noise
        self.apply_multi_res_noise = hasattr(self.cfg, "multi_res_noise")and self.cfg.multi_res_noise is not None
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.cfg.multi_res_noise.strength
            self.annealed_mr_noise = self.cfg.multi_res_noise.annealed
            self.mr_noise_downscale_strategy = (
                self.cfg.multi_res_noise.downscale_strategy
            )

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming
    
    def _init_before_train(self):
        # if self.real_unet_in_channels != self.model.unet.config["in_channels"]:
        self._replace_unet_conv_in()

        if self.encoding_type == "empty_text":
            # Encode empty text prompt
            self._encode_empty_text()
    
    def _encode_empty_text(self):
        # Encode empty text prompt
        self.model.encode_empty_text()
        self.empty_text_embed = self.model.text_embed_empty_or_clip_img.detach().clone().to(self.device)
    
    def _set_optimizer(self):

        # Trainability
        if hasattr(self.model, "vae"):
            self.model.vae.requires_grad_(False)
        if hasattr(self.model, "text_encoder"):
            self.model.text_encoder.requires_grad_(False)
        if hasattr(self.model, "image_encoder"):
            self.model.image_encoder.requires_grad_(False)
        lr = self.cfg.lr

        params_train = []
        if hasattr(self.cfg, 'finetune_parameters'):

            if self.cfg.finetune_parameters == "temporal":
                trained_params = []
                non_trained_params = []
                for name, param in self.model.unet.named_parameters():
                    if "spatial" in name or "conv_in" in name or "conv_out" in name or "conv_norm_out" in name:
                        param.requires_grad = False
                        non_trained_params.append(name)
                    else:
                        param.requires_grad = True
                        trained_params.append(name)
                logging.info(f"Trained parameters: {str(trained_params)}")
                logging.info(f"Fixed parameters: {str(non_trained_params)}")
                params_train = [param for param in self.model.unet.parameters() if param.requires_grad]
            else:
                logging.info(f"Trained parameters: {str(self.cfg.finetune_parameters)}")
                trained_params = []
                non_trained_params = []

                for model_part in self.cfg.finetune_parameters.keys():

                    if isinstance(getattr(self.cfg.finetune_parameters, model_part), ListConfig):
                        getattr(self.model, model_part).requires_grad_(False)

                        for model_part_sub in getattr(self.cfg.finetune_parameters, model_part):
                            if model_part_sub == "attentions":
                                
                                for name, param in self.model.unet.named_parameters():
                                    if "attentions" in name:
                                        param.requires_grad = True
                                        trained_params.append(name)
                                
                                params_train += [param for param in self.model.unet.parameters() if param.requires_grad]
                            else:
                                trained_params.append(model_part_sub)
                                getattr(getattr(self.model, model_part), model_part_sub).requires_grad_(True)
                                params_train += list(getattr(getattr(self.model, model_part), model_part_sub).parameters())
                    else:
                        print(type(getattr(self.cfg.finetune_parameters, model_part)))
                        raise NotImplementedError
                logging.info(f"Trained parameters: {str(trained_params)}")

        else:
            logging.info(f"Trained parameters: All UNet")
            params_train += list(self.model.unet.parameters())
            self.model.unet.requires_grad_(True)
        
        self.optimizer = Adam(params_train, lr=lr)

        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

    def _replace_unet_conv_in(self):

        if self.model.unet.config["in_channels"] != self.real_unet_in_channels:
            assert self.real_unet_in_channels % self.model.unet.config["in_channels"] == 0 , "self.real_unet_in_channels mod 4 must be 0, or you can't use repeat below."

            # replace the first layer to accept `self.real_unet_in_channels` in_channels
            _weight = self.model.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
            _bias = self.model.unet.conv_in.bias.clone()  # [320]
            _weight = _weight.repeat((1, self.real_unet_in_channels // self.model.unet.config["in_channels"], 1, 1))  # Keep selected channel(s)
            # half the activation magnitude
            _weight *= 1 / (self.real_unet_in_channels // self.model.unet.config["in_channels"])
            # new conv_in channel
            _n_convin_out_channel = self.model.unet.conv_in.out_channels
            _new_conv_in = Conv2d(
                self.real_unet_in_channels, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            )
            _new_conv_in.weight = Parameter(_weight)
            _new_conv_in.bias = Parameter(_bias)
            self.model.unet.conv_in = _new_conv_in
            logging.info("Unet conv_in layer is replaced")
            # replace config
            self.model.unet.config["in_channels"] = self.real_unet_in_channels
            logging.info("Unet config is updated")
        return

    def _sample_noise(self, timesteps, gt_latent, rand_num_generator, device):
        # Sample noise
        if self.apply_multi_res_noise:
            strength = self.mr_noise_strength
            if self.annealed_mr_noise:
                # calculate strength depending on t
                strength = strength * (timesteps / self.scheduler_timesteps)
            noise = multi_res_noise_like(
                gt_latent,
                strength=strength,
                downscale_strategy=self.mr_noise_downscale_strategy,
                generator=rand_num_generator,
                device=device,
            )
        else:
            noise = torch.randn(
                gt_latent.shape,
                device=device,
                generator=rand_num_generator,
            )  # [B, 4, h, w]
        return noise
    
    def _log_tensorboard(self, epoch):
        # Log to tensorboard
        accumulated_loss = self.train_metrics.result()["loss"]
        tb_logger.log_dic(
            {
                f"train/{k}": v
                for k, v in self.train_metrics.result().items()
            },
            global_step=self.effective_iter,
        )
        tb_logger.writer.add_scalar(
            "lr",
            self.lr_scheduler.get_last_lr()[0],
            global_step=self.effective_iter,
        )
        tb_logger.writer.add_scalar(
            "n_batch_in_epoch",
            self.n_batch_in_epoch,
            global_step=self.effective_iter,
        )
        logging.info(
            f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
        )
        self.train_metrics.reset()

    def train(self, t_end=None):
        raise NotImplementedError

    def _train_step_callback(self):
        """Executed after every iteration"""
        # Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

        _is_latest_saved = False

        # Visualization
        if self.vis_period > 0 and 0 == self.effective_iter % self.vis_period:
            self.visualize()
        
        # Validation
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            self.in_evaluation = True  # flag to do evaluation in resume run if validation is not finished
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            _is_latest_saved = True
            self.validate()
            self.in_evaluation = False
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Save training checkpoint (can be resumed)
        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and not _is_latest_saved
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

    def validate(self):
        if self.val_loaders is not None:
            for i, val_loader in enumerate(self.val_loaders):
                val_dataset_name = val_loader.dataset.disp_name
                val_out_dir = os.path.join(
                    self.out_dir_vis, self._get_backup_ckpt_name(), val_dataset_name
                )
                os.makedirs(val_out_dir, exist_ok=True)
                val_metric_dic = self.validate_single_dataset(
                    data_loader=val_loader, metric_tracker=self.val_metrics, save_to_dir=val_out_dir
                )

                logging.info(
                    f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {val_metric_dic}"
                )
                tb_logger.log_dic(
                    {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dic.items()},
                    global_step=self.effective_iter,
                )
                # save to file
                eval_text = eval_dic_to_text(
                    val_metrics=val_metric_dic,
                    dataset_name=val_dataset_name,
                    sample_list_path=val_loader.dataset.filename_ls_path,
                )
                _save_to = os.path.join(
                    self.out_dir_eval,
                    f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt",
                )
                with open(_save_to, "w+") as f:
                    f.write(eval_text)

                # Update main eval metric
                if 0 == i:
                    main_eval_metric = val_metric_dic[self.main_val_metric]
                    if (
                        "minimize" == self.main_val_metric_goal
                        and main_eval_metric < self.best_metric
                        or "maximize" == self.main_val_metric_goal
                        and main_eval_metric > self.best_metric
                    ):
                        self.best_metric = main_eval_metric
                        logging.info(
                            f"Best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
                        )
                        # Save a checkpoint
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                        )

    def visualize(self):
        if self.vis_loaders is not None:
            for vis_loader in self.vis_loaders:
                vis_dataset_name = vis_loader.dataset.disp_name
                vis_out_dir = os.path.join(
                    self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
                )
                os.makedirs(vis_out_dir, exist_ok=True)
                _ = self.validate_single_dataset(
                    data_loader=vis_loader,
                    metric_tracker=self.val_metrics,
                    save_to_dir=vis_out_dir,
                )

    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        save_to_dir: str = None,
    ):
        raise NotImplementedError

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # Save UNet
        unet_path = os.path.join(ckpt_dir, "unet")
        current_model = self.model.unet.module if hasattr(self, "world_size") and self.world_size > 1 else  self.model.unet
        current_model.save_pretrained(unet_path, safe_serialization=False)
        logging.info(f"UNet is saved to: {unet_path}")

        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

            logging.info(f"Trainer state is saved to: {train_state_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True, effective_iter=None
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")
        # Load UNet
        _model_path = os.path.join(ckpt_path, "unet", "diffusion_pytorch_model.bin")
        current_model = self.model.unet.module if hasattr(self, "world_size") and self.world_size > 1 else  self.model.unet

        try:
            current_model.load_state_dict(
                torch.load(_model_path, map_location=self.device)  # self.device
            )
        except:
            missing_keys, unexpected_keys = current_model.load_state_dict(torch.load(_model_path, map_location=self.device), strict=False)
            if missing_keys:
                logging.info(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logging.info(f"Unexpected keys: {unexpected_keys}")


        logging.info(f"UNet parameters are loaded from {_model_path}")
        
        # Load training states
        if load_trainer_state:
            checkpoint = torch.load(os.path.join(ckpt_path, "trainer.ckpt"), map_location=self.device)
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.best_metric = checkpoint["best_metric"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"optimizer state is loaded from {ckpt_path}")

            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            logging.info(f"Optimizer state is loaded and moved to {self.device} from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_path}")
        
        if effective_iter is not None:
            self.effective_iter = effective_iter

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"


class TMPixelRec_Trainer(TMRec_BaseTrainer):
    def __init__(
        self,
        cfg: OmegaConf,
        model,
        train_dataloader: DataLoader,
        device,
        base_ckpt_dir,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accumulation_steps: int,
        val_dataloaders: List[DataLoader] = None,
        vis_dataloaders: List[DataLoader] = None,
        real_unet_in_channels = 13,
        rank = 0,
        world_size = 1,
        BTCHW = False,
        SR = False,
    ):

        super().__init__(
            cfg,
            model,
            train_dataloader,
            device,
            base_ckpt_dir,
            out_dir_ckpt,
            out_dir_eval,
            out_dir_vis,
            accumulation_steps,
            val_dataloaders,
            vis_dataloaders,
            real_unet_in_channels
        )
        self.rank = rank
        self.world_size = world_size
        self.time_embedding = model.time_embedding if hasattr(model, "time_embedding") else False
        self.without_td = model.without_td if hasattr(model, "without_td") else False
        self.without_sd = model.without_sd if hasattr(model, "without_sd") else False

        if self.time_embedding:
            logging.debug("enable time embedding")
        self.BTCHW = BTCHW
        if self.BTCHW:
            logging.debug("BTCHW mode")
        self.SR = SR
        if self.SR:
            logging.debug("SR mode")
        if self.without_td:
            logging.debug("no TD mode")
        if self.without_sd:
            logging.debug("no SD mode")
        self.upsample_noise = model.upsample_noise if hasattr(model, "upsample_noise") else False
        if self.upsample_noise:
            logging.debug("upsample_noise by nearest x4")

    def makeDDPmodel(self):
        self.model.to(self.device)
        self.model.unet = DDP(self.model.unet, find_unused_parameters=False)
        pass
    
    def _replace_unet_conv_in(self):

        if self.real_unet_in_channels is not None and self.model.unet.config["in_channels"] != self.real_unet_in_channels:

            # new conv_in channel
            _n_convin_out_channel = self.model.unet.conv_in.out_channels
            _new_conv_in = Conv2d(
                self.real_unet_in_channels, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            )
            self.model.unet.conv_in = _new_conv_in
            logging.info("Unet conv_in layer is replaced")
            # replace config
            self.model.unet.config["in_channels"] = self.real_unet_in_channels
            logging.info(f"Unet config is updated to {self.real_unet_in_channels}")
        return
    

    def train(self, t_end=None):
        logging.info("Start training")

        device = self.device
        if self.world_size == 1:
            self.model.to(device)

        if self.rank == 0 and self.in_evaluation:
            logging.info(
                "Last evaluation was not finished, will do evaluation before continue training."
            )
            self.validate()

        self.train_metrics.reset()
        accumulated_step = 0

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            if self.rank == 0:
                logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                self.model.unet.train()

                # globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # >>> With gradient accumulation >>>

                # Get data
                rgb_back = batch["rgb_norm"].to(device)  # 3
                td_back_accum = batch["td_back_norm"].to(device)  # 1
                rgb_front = batch["rgb_front_norm"].to(device)  # 3
                td_front_accum = batch["td_front_norm"].to(device)  # 1
                sd = batch["sd_norm"].to(device)  # 2
                gt = batch["gt_norm"].to(device)
                if self.time_embedding:
                    time_ebd = batch["div_idx_float"].to(device)
                if self.SR:
                    gt_lr_norm = batch["gt_lr_norm"].to(device)
                gt_latent = gt
                rgb_back_latent = rgb_back
                rgb_front_latent = rgb_front
                td_back_accum_latent = td_back_accum
                td_front_accum_latent = td_front_accum
                sd_latent = sd

                batch_size = rgb_back.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    self.scheduler_timesteps,
                    (batch_size,),
                    device=device,
                    generator=rand_num_generator,
                ).long()  # [B]

                # Sample Noise
                noise = self._sample_noise(timesteps, gt_latent, rand_num_generator, device)

                # Add noise to the latents (diffusion forward process)
                noisy_latents = self.training_noise_scheduler.add_noise(
                    gt_latent, noise, timesteps
                )  # [B, 4, h, w]

                # set input
                if self.upsample_noise:
                    noisy_latents = F.interpolate(noisy_latents, scale_factor=(1, 4, 4), mode='nearest')
                if self.without_td and self.without_sd:
                    unet_input = torch.cat([rgb_back_latent, rgb_front_latent, noisy_latents], dim=-3)
                elif self.without_td:
                    unet_input = torch.cat([rgb_back_latent, rgb_front_latent, sd_latent, noisy_latents], dim=-3)
                elif self.without_sd:
                    unet_input = torch.cat([rgb_back_latent, td_back_accum_latent, rgb_front_latent, td_front_accum_latent, noisy_latents], dim=-3)
                else:
                    unet_input = torch.cat([rgb_back_latent, td_back_accum_latent, rgb_front_latent, td_front_accum_latent, sd_latent, noisy_latents], dim=-3)

                if self.SR:
                    unet_input = torch.cat([gt_lr_norm, unet_input], dim=-3)

                if self.time_embedding:
                    model_pred = self.model.unet(
                        unet_input, timesteps, class_labels = time_ebd * 1000.0
                    ).sample
                else:
                    model_pred = self.model.unet(
                        unet_input, timesteps,
                    ).sample

                if torch.isnan(model_pred).any():
                    logging.warning("model_pred contains NaN.")

                # Get the target for loss depending on the prediction type
                if "sample" == self.prediction_type:
                    target = gt_latent
                elif "epsilon" == self.prediction_type:
                    target = noise
                elif "v_prediction" == self.prediction_type:
                    target = self.training_noise_scheduler.get_velocity(
                        gt_latent, noise, timesteps
                    )  # [B, 4, h, w]
                else:
                    raise ValueError(f"Unknown prediction type {self.prediction_type}")

                latent_loss = self.loss(model_pred.float(), target.float())

                loss = latent_loss.mean()

                self.train_metrics.update("loss", loss.item())

                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                accumulated_step += 1

                self.n_batch_in_epoch += 1
                # Practical batch end

                # Perform optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    accumulated_step = 0

                    self.effective_iter += 1

                    if self.rank == 0:
                        # Log to tensorboard
                        self._log_tensorboard(epoch)
                        # Per-step callback
                        self._train_step_callback()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=False,
                        )
                        logging.info("Training ended.")
                        return
                    # Time's up
                    elif t_end is not None and datetime.now() >= t_end:
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        logging.info("Time is up, training paused.")
                        return

                    # <<< Effective batch end <<<

            # Epoch end
            self.n_batch_in_epoch = 0

    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        save_to_dir: str = None,
        save_single_img_root_path: str = None
    ):

        metric_tracker.reset()

        # Generate seed sequence for consistent evaluation
        val_init_seed = self.cfg.validation.init_seed
        val_seed_ls = generate_seed_sequence(val_init_seed, len(data_loader))
        logging.info(f"saving visualizing results at {save_to_dir}")
        for i, batch in enumerate(
            tqdm(data_loader, desc=f"evaluating on {data_loader.dataset.disp_name}"),
            start=1,
        ):
            assert 1 == data_loader.batch_size
            
            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)
            
            # get data
            rgb_back = batch["rgb_norm"]
            td_back_accum = batch["td_back_norm"]
            rgb_front = batch["rgb_front_norm"]
            td_front_accum = batch["td_front_norm"]
            sd = batch["sd_norm"]
            if self.time_embedding:
                time_ebd = batch["div_idx_float"]
            if self.SR:
                lr = batch["gt_lr_norm"]
            gt = batch["gt"].squeeze()
            gt = gt.to(self.device)
            
            model_args = {
                "rgb_back": rgb_back,
                "td_back_accum": td_back_accum, 
                "rgb_front": rgb_front, 
                "td_front_accum": td_front_accum, 
                "sd": sd,
                "denoising_steps": self.cfg.validation.denoising_steps,
                "generator": generator,
                "show_progress_bar": True,
            } 

            if self.time_embedding:
                model_args["time_step"] = time_ebd
            if self.SR:
                model_args["lr"] = lr

            pipe_out = self.model(**model_args)

            gt_pred: np.ndarray = pipe_out.rec_np

            # Evaluate
            sample_metric = []
            gt_pred_ts = torch.from_numpy(gt_pred).to(self.device)

            if self.BTCHW:
                num_T = gt.shape[0]
                mid_div = (num_T-1)//2
                gt_mid = gt[[mid_div],...]
                gt_pred_ts_mid = gt_pred_ts[[mid_div],...]
            else:
                gt_mid = gt
                gt_pred_ts_mid = gt_pred_ts


            for met_func in self.metric_funcs:
                _metric_name = met_func.__name__
                _metric = met_func(gt_pred_ts_mid, gt_mid, valid_mask=None).item()
                sample_metric.append(_metric.__str__())
                metric_tracker.update(_metric_name, _metric)

            if save_single_img_root_path is not None:
                save_dir = os.path.join(save_single_img_root_path, batch["folder_name"][0])
                os.makedirs(save_dir, exist_ok=True)
                png_save_path = os.path.join(save_dir, f'{int(batch["pack_idx"][0]):03d}_{int(batch["div_idx"][0]):02d}_lr.png')
                lr_to_save = (pipe_out.rec_np * 255.0).astype(np.uint8).transpose(1, 2, 0)

                cv2.imwrite(png_save_path, cv2.cvtColor(lr_to_save, cv2.COLOR_RGB2BGR))


            # Save as 8-bit uint jpg
            if save_to_dir is not None:
                if self.BTCHW:
                    def save_combined_images(gt, gt_pred_ts, rgb_back, rgb_front, td_back_accum, td_front_accum, sd, save_path):

                        T, C, H, W = gt.shape
                        
                        def resize_to_gt(tensor):
                            return F.interpolate(tensor, size=(H, W), mode='bilinear', align_corners=False)
                        
                        rgb_back = resize_to_gt(rgb_back)
                        rgb_front = resize_to_gt(rgb_front)
                        td_back_accum = resize_to_gt(td_back_accum)
                        td_front_accum = resize_to_gt(td_front_accum)
                        sd = resize_to_gt(sd)

                        gt_np = gt.permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, C]
                        gt_pred_np = gt_pred_ts.permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, C]
                        rgb_back_np = rgb_back.permute(0, 2, 3, 1).cpu().numpy()
                        rgb_front_np = rgb_front.permute(0, 2, 3, 1).cpu().numpy()
                        td_back_accum_np = td_back_accum.permute(0, 2, 3, 1).cpu().numpy()
                        td_front_accum_np = td_front_accum.permute(0, 2, 3, 1).cpu().numpy()
                        sd_np = sd.permute(0, 2, 3, 1).cpu().numpy()

                        step = 1  # T // 5
                        combined_images = np.vstack([
                            np.hstack([rgb_back_np[i] for i in range(0, T, step)]),
                            np.hstack([rgb_front_np[i] for i in range(0, T, step)]),
                            np.hstack([td_back_accum_np[i] for i in range(0, T, step)]),
                            np.hstack([td_front_accum_np[i] for i in range(0, T, step)]),
                            np.hstack([sd_np[i] for i in range(0, T, step)]),
                            np.hstack([gt_pred_np[i] for i in range(0, T, step)]),
                            np.hstack([gt_np[i] for i in range(0, T, step)]),
                        ])

                        combined_images = (combined_images * 255).clip(0, 255).astype(np.uint8)

                        cv2.imwrite(save_path, cv2.cvtColor(combined_images, cv2.COLOR_RGB2BGR))
                    
                    img_name = batch["rgb_relative_path"][0].replace("/", "_")+"_"+batch["pack_idx"][0]+"_"+batch["div_idx"][0]

                    jpg_save_path = os.path.join(save_to_dir, f"{img_name}_pred.jpg")
                    save_combined_images(gt.squeeze(), gt_pred_ts.squeeze(), 
                                        (rgb_back.squeeze()+1.0)/2.0, (rgb_front.squeeze()+1.0)/2.0, 
                                        (torch.clip(td_back_accum, -1.0, 1.0).repeat(1,1,3,1,1)+1.0).squeeze()/2.0, (torch.clip(td_front_accum, -1.0, 1.0).repeat(1,1,3,1,1)+1.0).squeeze()/2.0,
                                        (torch.cat([torch.zeros_like(td_back_accum, device=sd.device), torch.clip(sd*16, -1.0, 1.0)], dim=-3).squeeze()+1.0)/2.0,                                          
                                        jpg_save_path)

                else:
                    img_name = batch["rgb_relative_path"][0].replace("/", "_")+"_"+batch["pack_idx"][0]+"_"+batch["div_idx"][0]
                    
                    jpg_save_path = os.path.join(save_to_dir, f"{img_name}.jpg")
                    print(pipe_out.rec_np.shape, gt.shape, rgb_back.shape, rgb_front.shape, td_back_accum.shape, td_front_accum.shape)
                    result_to_save = (pipe_out.rec_np * 255.0).astype(np.uint8).transpose(1, 2, 0)
                    gt_to_save = (gt.squeeze().cpu().numpy() * 255.0).astype(np.uint8).transpose(1, 2, 0)
                    rgbback_to_save = ((rgb_back.squeeze().cpu().numpy() + 1.0) / 2.0 * 255.0).astype(np.uint8).transpose(1, 2, 0)
                    rgbfront_to_save = ((rgb_front.squeeze().cpu().numpy() + 1.0) / 2.0 * 255.0).astype(np.uint8).transpose(1, 2, 0)
                    tdback_to_save = (np.clip((td_back_accum.squeeze().cpu().numpy() + 1.0) / 2.0, 0.0, 1.0) * 255.0).astype(np.uint8)
                    tdfront_to_save = (np.clip((td_front_accum.squeeze().cpu().numpy() + 1.0) / 2.0, 0.0, 1.0) * 255.0).astype(np.uint8)
                    if self.SR:
                        lr_to_save = ((lr.squeeze().cpu().numpy() + 1.0) / 2.0 * 255.0).astype(np.uint8).transpose(1, 2, 0)
                    else:
                        lr_to_save = None

                    tdback_to_save = np.stack([tdback_to_save] * 3, axis=-1)
                    tdfront_to_save = np.stack([tdfront_to_save] * 3, axis=-1)

                    if self.SR:
                        combined_image = np.concatenate(
                            [rgbback_to_save, tdback_to_save, lr_to_save, result_to_save, gt_to_save, tdfront_to_save, rgbfront_to_save], axis=1
                        )
                    else:
                        combined_image = np.concatenate(
                            [rgbback_to_save, tdback_to_save, result_to_save, gt_to_save, tdfront_to_save, rgbfront_to_save], axis=1
                        )

                    Image.fromarray(combined_image).save(jpg_save_path, format="JPEG")

        return metric_tracker.result()


