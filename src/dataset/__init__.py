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

import os

from .metadata_dataset import TianmoucMetadataInterpolationDatasetHdf5BTCHW, TianmoucMetadataInterpolationDatasetHdf5SingleSR

dataset_name_class_dict = {
    "TianmoucMetadataInterpolationDatasetHdf5BTCHW": TianmoucMetadataInterpolationDatasetHdf5BTCHW,
    "TianmoucMetadataInterpolationDatasetHdf5SingleSR": TianmoucMetadataInterpolationDatasetHdf5SingleSR,
}

def get_rec_dataset(
    cfg_data_split
):
    if "mixed" == cfg_data_split.name:
        dataset_ls = [
            get_rec_dataset(_cfg)
            for _cfg in cfg_data_split.dataset_list
        ]
        return dataset_ls
    elif cfg_data_split.name in dataset_name_class_dict.keys():
        dataset_class = dataset_name_class_dict[cfg_data_split.name]
        params = {k: v for k, v in cfg_data_split.items() if k not in ["name", ]}
        dataset = dataset_class(
            **params
        )
    else:
        raise NotImplementedError

    return dataset