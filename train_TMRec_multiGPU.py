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


import argparse
import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import List

import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from diffusers import DDPMScheduler, DDIMScheduler
from CBRDM import get_pipeline_cls
from CBRDM.unet2d_btchw import UNet2DModelBTCHW  # used in stage1
from diffusers import UNet2DModel, DiffusionPipeline  # used in stage2
from src.trainer import get_trainer_cls
from src.dataset import get_rec_dataset
from src.dataset.mixed_sampler import MixedBatchSampler
from src.util.config_util import (
    find_value_in_omegaconf,
    recursive_load_config,
)
from src.util.logging_util import (
    config_logging,
    init_wandb,
    load_wandb_job_id,
    log_slurm_job_id,
    save_wandb_job_id,
    tb_logger,
)
from src.util.slurm_util import get_local_scratch_dir, is_on_slurm
import requests
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """Setup distributed process group."""
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

def cleanup():
    """Cleanup distributed process group."""
    dist.destroy_process_group()

def main_worker(rank, world_size, args):

    if world_size > 1:
        setup(rank, world_size)

    resume_run = args.resume_run
    output_dir = args.output_dir
    base_ckpt_dir = (
        args.base_ckpt_dir
        if args.base_ckpt_dir is not None
        else os.environ["BASE_CKPT_DIR"]
    )

    # -------------------- Initialization --------------------
    # Resume previous run
    if resume_run is not None:
        print(f"Resume run: {resume_run}")
        out_dir_run = os.path.dirname(os.path.dirname(resume_run))
        job_name = os.path.basename(out_dir_run)
        # Resume config file
        cfg = OmegaConf.load(os.path.join(out_dir_run, "config.yaml"))
    else:
        # Run from start
        cfg = recursive_load_config(args.config)
        # Full job name
        pure_job_name = os.path.basename(args.config).split(".")[0]
        # Add time prefix
        if args.add_datetime_prefix:
            job_name = f"{t_start.strftime('%y_%m_%d-%H_%M_%S')}-{pure_job_name}"
        else:
            job_name = pure_job_name

        # Output dir
        if output_dir is not None:
            out_dir_run = os.path.join(output_dir, job_name)
        else:
            out_dir_run = os.path.join("./output", job_name)
        # if rank == 0:
        os.makedirs(out_dir_run, exist_ok=True)

    cfg_data = cfg.dataset

    # Other directories
    out_dir_ckpt = os.path.join(out_dir_run, "checkpoint")
    out_dir_tb = os.path.join(out_dir_run, "tensorboard")
    out_dir_eval = os.path.join(out_dir_run, "evaluation")
    out_dir_vis = os.path.join(out_dir_run, "visualization")

    # if rank == 0:
    if not os.path.exists(out_dir_ckpt):
        os.makedirs(out_dir_ckpt, exist_ok=True)
    if not os.path.exists(out_dir_tb):
        os.makedirs(out_dir_tb, exist_ok=True)
    if not os.path.exists(out_dir_eval):
        os.makedirs(out_dir_eval, exist_ok=True)
    if not os.path.exists(out_dir_vis):
        os.makedirs(out_dir_vis, exist_ok=True)
    
    # -------------------- Logging settings --------------------
    config_logging(cfg.logging, out_dir=out_dir_run)
    logging.debug(f"config: {cfg}")

    # Initialize wandb
    if not args.no_wandb:
        if resume_run is not None:
            wandb_id = load_wandb_job_id(out_dir_run)
            wandb_cfg_dic = {
                "id": wandb_id,
                "resume": "must",
                **cfg.wandb,
            }
        else:
            wandb_cfg_dic = {
                "config": dict(cfg),
                "name": job_name,
                "mode": "online",
                **cfg.wandb,
            }
        wandb_cfg_dic.update({"dir": out_dir_run})
        wandb_run = init_wandb(enable=True, **wandb_cfg_dic)
        save_wandb_job_id(wandb_run, out_dir_run)
    else:
        init_wandb(enable=False)

    # Tensorboard (should be initialized after wandb)
    tb_logger.set_dir(out_dir_tb)

    log_slurm_job_id(step=0)

    # -------------------- Device --------------------
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    logging.info(f"Rank {rank} is using device {device}")

    # -------------------- Snapshot of code and config --------------------
    if resume_run is None:
        _output_path = os.path.join(out_dir_run, "config.yaml")
        with open(_output_path, "w+") as f:
            OmegaConf.save(config=cfg, f=f)
        logging.info(f"Config saved to {_output_path}")

    # -------------------- Gradient accumulation steps --------------------
    eff_bs = cfg.dataloader.effective_batch_size
    accumulation_steps = eff_bs / cfg.dataloader.max_train_batch_size / world_size
    assert int(accumulation_steps) == accumulation_steps, f"eff_bs must be divided by (max_train_batch_size * world_size) \n but Effective batch size: {eff_bs}, world_size: {world_size}"
    accumulation_steps = int(accumulation_steps)

    logging.info(
        f"Effective batch size: {eff_bs}, accumulation steps: {accumulation_steps}, world_size: {world_size}"
    )

    # -------------------- Data --------------------
    loader_seed = cfg.dataloader.seed
    if loader_seed is None:
        loader_generator = None
    else:
        loader_generator = torch.Generator().manual_seed(loader_seed)

    # Training dataset
    train_dataset = get_rec_dataset(
        cfg_data.train,
    )
    if "mixed" == cfg_data.train.name:
        dataset_ls = train_dataset
        assert len(cfg_data.train.prob_ls) == len(
            dataset_ls
        ), "Lengths don't match: `prob_ls` and `dataset_list`"
        concat_dataset = ConcatDataset(dataset_ls)
        mixed_sampler = MixedBatchSampler(
            src_dataset_ls=dataset_ls,
            batch_size_per_gpu=cfg.dataloader.max_train_batch_size,
            drop_last=True,
            prob=cfg_data.train.prob_ls,
            shuffle=True,
            generator=loader_generator,
        )
        train_loader = DataLoader(
            concat_dataset,
            batch_sampler=mixed_sampler,
            num_workers=cfg.dataloader.num_workers,
        )
    else:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.dataloader.max_train_batch_size,
            sampler=train_sampler,
            num_workers=cfg.dataloader.num_workers,
            shuffle=False,  # DistributedSampler already process shuffle
            generator=loader_generator,
        )
    if rank == 0:
        # Validation dataset
        val_loaders: List[DataLoader] = []
        for _val_dic in cfg_data.val:
            _val_dataset = get_rec_dataset(
            _val_dic,
        )
            
            _val_loader = DataLoader(
                dataset=_val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=cfg.dataloader.num_workers,
            )
            val_loaders.append(_val_loader)

        # Visualization dataset
        vis_loaders: List[DataLoader] = []
        if hasattr(cfg_data, "vis"):
            for _vis_dic in cfg_data.vis:
                _vis_dataset = get_rec_dataset(
                _vis_dic,
                )
                
                _vis_loader = DataLoader(
                    dataset=_vis_dataset,
                    batch_size=8,
                    shuffle=False,
                    num_workers=cfg.dataloader.num_workers,
                )
                vis_loaders.append(_vis_loader)
        else:
            vis_loaders = None
    else:
        val_loaders = None
        vis_loaders = None

    # -------------------- Model --------------------
    _pipeline_kwargs = cfg.pipeline.kwargs if hasattr(cfg.pipeline, "kwargs") and cfg.pipeline.kwargs is not None else {}
    pipeline_cls: DiffusionPipeline = get_pipeline_cls(cfg.pipeline.name)

    if cfg.pipeline.name == "TianmoucSingleStageReconstructionPipeline":
        _pipeline_component = {}
        
        logging.info(f"Initializing {cfg.pipeline.name} from {cfg.model.pretrained_path}")
        
        scheduler = DDIMScheduler.from_pretrained(os.path.join(base_ckpt_dir, cfg.model.pretrained_path, "scheduler"))

        # Select UNet class
        if cfg.model.name == "TianmoucRec_BRDM":  # multi frame bi-directional recurrent reconstruction
            cls = UNet2DModelBTCHW
        elif cfg.model.name == "TianmoucRec_Base" or cfg.model.name == "TianmoucRec_SR":  # single Frame reconstruction
            cls = UNet2DModel
        else:
            raise NotImplementedError(f"Model {cfg.model.name} not implemented")
        # Initialize UNet
        try:
            unet = cls.from_pretrained(os.path.join(base_ckpt_dir, cfg.model.pretrained_path, "unet"))
        except:
            logging.info(f"Initializing {cfg.pipeline.name} from pretrained {cfg.model.pretrained_path} error, only use config and init weights!!!")
            unet = cls.from_config(os.path.join(base_ckpt_dir, cfg.model.pretrained_path, "unet"))
        
        # Maybe has pretrained weights
        if hasattr(cfg.trainer, "unet_pretrained_path"):
            logging.info(f"Load net parameters from {cfg.trainer.unet_pretrained_path}")
            try:
                # Load the pretrained state dict
                state_dict = torch.load(
                    cfg.trainer.unet_pretrained_path,
                    map_location=device
                )
                # Define the keys to remove (if necessary)
                keys_to_remove = {}  # "conv_in.weight" Can add other keys here if needed
                state_dict = {k: v for k, v in state_dict.items() if k not in keys_to_remove}
                # Load the model's current state dict
                model_state_dict = unet.state_dict()
                # Initialize the lists to track mismatched keys
                mismatched_keys = []
                valid_state_dict = {}
                # Iterate through the state_dict and check the size of each tensor
                for k, v in state_dict.items():
                    if k in model_state_dict:
                        if v.shape == model_state_dict[k].shape:
                            valid_state_dict[k] = v
                        else:
                            mismatched_keys.append(k)
                    else:
                        mismatched_keys.append(k)
                # Log any mismatched keys
                if mismatched_keys:
                    logging.info(f"Mismatched keys (size mismatch): {mismatched_keys}")
                # Load the valid state dict into the model
                missing_keys, unexpected_keys = unet.load_state_dict(valid_state_dict, strict=False)
                # Log missing and unexpected keys
                if missing_keys:
                    logging.info(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    logging.info(f"Unexpected keys: {unexpected_keys}")
            except Exception as e:
                logging.info(f"Error loading state_dict: {e}")

        # Initialize pipeline
        model = pipeline_cls(
                unet, scheduler, **_pipeline_kwargs,
            )
    else:
        raise NotImplementedError

    # -------------------- Trainer --------------------
    # Exit time
    if args.exit_after > 0:
        t_end = t_start + timedelta(minutes=args.exit_after)
        logging.info(f"Will exit at {t_end}")
    else:
        t_end = None

    _trainer_kwargs = cfg.trainer.kwargs if hasattr(cfg.trainer, "kwargs") and cfg.trainer.kwargs is not None else {}
    trainer_cls = get_trainer_cls(cfg.trainer.name)
    logging.debug(f"Trainer: {trainer_cls}")
    logging.debug(f"_trainer_kwargs: {_trainer_kwargs}")
    trainer = trainer_cls(
        cfg=cfg,
        model=model,
        train_dataloader=train_loader,
        device=device,
        base_ckpt_dir=base_ckpt_dir,
        out_dir_ckpt=out_dir_ckpt,
        out_dir_eval=out_dir_eval,
        out_dir_vis=out_dir_vis,
        accumulation_steps=accumulation_steps,
        val_dataloaders=val_loaders,
        vis_dataloaders=vis_loaders,

        rank = rank,
        world_size = world_size,

        **_trainer_kwargs
    )

    # make DDP
    if world_size > 1:
        trainer.makeDDPmodel()
    
    # -------------------- Checkpoint --------------------
    if resume_run is not None:
        try:
            pass
            trainer.load_checkpoint(
                resume_run, load_trainer_state=True, resume_lr_scheduler=True
            )
        except:
            logging.info(f"WARNING optimizer state mismatch, try to use `load_trainer_state=False` ")
            trainer.load_checkpoint(
                resume_run, load_trainer_state=False, resume_lr_scheduler=True
            )

    # -------------------- Training & Evaluation Loop --------------------
    try:
        trainer.train(t_end=t_end)
    except BaseException as e:
        logging.exception(e)

if "__main__" == __name__:
    t_start = datetime.now()
    print(f"start at {t_start}")

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(description="Train your cute model!")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file.",
    )
    parser.add_argument(
        "--resume_run",
        action="store",
        default=None,
        help="Path of checkpoint to be resumed. If given, will ignore --config, and checkpoint in the config",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="directory to save checkpoints"
    )
    parser.add_argument("--no_cuda", action="store_true", help="Do not use cuda.")
    parser.add_argument(
        "--exit_after",
        type=int,
        default=-1,
        help="Save checkpoint and exit after X minutes.",
    )
    parser.add_argument("--no_wandb", action="store_true", help="run without wandb")

    parser.add_argument(
        "--base_ckpt_dir",
        type=str,
        default=None,
        help="directory of pretrained checkpoint, if None, use os.environ[`BASE_CKPT_DIR`]",
    )
    parser.add_argument(
        "--add_datetime_prefix",
        action="store_true",
        help="Add datetime to the output folder name, can reuse the same config file and avoid dictionary existed",
    )

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size > 1:
        logging.info(f"Total GPUs available: {world_size}")
    else:
        logging.info(f"Single GPU training")

    if world_size > 1:
        try:
            mp.spawn(main_worker, args=(world_size, args, ), nprocs=world_size)
        except:
            cleanup()
    else:
        main_worker(0, 1, args)
    