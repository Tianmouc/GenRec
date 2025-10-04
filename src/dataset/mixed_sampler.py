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

import torch
from torch.utils.data import (
    BatchSampler,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

class MixedBatchSampler(BatchSampler):

    def __init__(
        self, src_dataset_ls, batch_size_per_gpu, drop_last, shuffle, prob=None, generator=None
    ):

        self.base_sampler = None
        self.batch_size = batch_size_per_gpu
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

        self.src_dataset_ls = src_dataset_ls
        self.n_dataset = len(self.src_dataset_ls)

        self.dataset_length = [len(ds) for ds in self.src_dataset_ls]
        self.cum_dataset_length = [
            sum(self.dataset_length[:i]) for i in range(self.n_dataset)
        ]

        self.is_distributed = dist.is_available() and dist.is_initialized()

        self.src_batch_samplers = []
        for ds in self.src_dataset_ls:
            if self.is_distributed:
                sampler = DistributedSampler(
                    ds, shuffle=self.shuffle
                )
            else:
                sampler = RandomSampler(
                    ds, replacement=False, generator=self.generator
                ) if self.shuffle else SequentialSampler(ds)
            
            batch_sampler = BatchSampler(
                sampler=sampler,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
            )
            self.src_batch_samplers.append(batch_sampler)
    
        self.raw_batches = [
            list(bs) for bs in self.src_batch_samplers
        ]
        self.n_batches = [len(b) for b in self.raw_batches]
        self.n_total_batch = sum(self.n_batches)

        if prob is None:
            self.prob = torch.tensor(self.n_batches) / self.n_total_batch
        else:
            self.prob = torch.as_tensor(prob)

    def __iter__(self):

        for _ in range(self.n_total_batch):
            idx_ds = torch.multinomial(
                self.prob, 1, replacement=True, generator=self.generator
            ).item()
            if 0 == len(self.raw_batches[idx_ds]):
                self.raw_batches[idx_ds] = list(self.src_batch_samplers[idx_ds])
            batch_raw = self.raw_batches[idx_ds].pop()
            shift = self.cum_dataset_length[idx_ds]
            batch = [n + shift for n in batch_raw]

            yield batch

    def __len__(self):
        return self.n_total_batch

