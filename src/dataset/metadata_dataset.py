# Author: YapengMeng
# Last modified: 2025-09-23


import os
import random
from skimage.transform import resize
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import torch.nn.functional as F
import h5py


class TianmoucMetadataInterpolationDatasetHdf5Single(Dataset):
    def __init__(self, disp_name, dir, resize_to_hw=None, val_pack=None, val_div=None, rgb_key="rgb", mode="train", hdf5_mode="single"):
        """
        Args:
            root_dir (string): Directory with all the subfolders.
        """
        assert rgb_key in ['rgb', 'F', "GT"]
        assert mode in ['train', 'val']
        assert hdf5_mode in ['single', 'multi']
        self.mode = mode
        self.rgb_key = rgb_key
        self.size_multi = 1 if mode == "val" else 25
        self.hdf5_mode = hdf5_mode

        self.disp_name = disp_name
        self.root_dir = dir
        self.filename_ls_path = dir
        self.val_pack = val_pack
        self.val_div = val_div
        self.resize_to_hw = resize_to_hw
        hdf5s_all = [os.path.join(dir, d) for d in sorted(os.listdir(dir)) if d.endswith("hdf5")]

        self.hdf5s = hdf5s_all
        
        
    def __len__(self):
        """ Total length is the number of subfolders. """
        return len(self.hdf5s)*self.size_multi
    
    def resize_output(self, output_size, *tensors):
        resized_tensors = tuple(TF.resize(tensor, output_size) for tensor in tensors)
        return resized_tensors


class TianmoucMetadataInterpolationDatasetHdf5SingleSR(TianmoucMetadataInterpolationDatasetHdf5Single):
    
    def __init__(self, disp_name, dir, resize_to_hw=None, val_pack=None, val_div=None, rgb_key="rgb", mode="train", hdf5_mode="single", midsize=(48, 96)):
        
        super().__init__(disp_name, dir, resize_to_hw, val_pack, val_div, rgb_key, mode, hdf5_mode)

        self.midsize = midsize

    def __getitem__(self, idx):

        idx = idx % len(self.hdf5s)

        if self.hdf5_mode == "single":
            with h5py.File(self.hdf5s[idx], 'r') as h5_file:

                div = random.randint(0, 25) if self.val_div is None else self.val_div

                pack_idx = 0

                # rgb
                if self.rgb_key != "GT":
                    rgb_back_norm = np.transpose(h5_file[f'{self.rgb_key}0'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0
                    rgb_front_norm = np.transpose(h5_file[f'{self.rgb_key}1'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0
                else:
                    rgb_back_norm = np.transpose(h5_file['gt_0'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0
                    rgb_front_norm = np.transpose(h5_file['gt_25'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0

                # sd
                sd = np.transpose(h5_file[f'tsd_{div}'][:][:, :, 1:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0  # Shape: 2 * H * W

                # td_accum
                if div == 0:
                    td_accum_back = np.zeros_like(h5_file[f'tsd_0'][:][:, :, 0][None, ...])
                else:
                    td_accum_back = np.sum([h5_file[f'tsd_{i}'][:][:, :, 0][None, ...].astype(np.float32) / 255.0 * 2.0 - 1.0 for i in range(1, div + 1)], axis=0)
                if div == 25:
                    td_accum_front = np.zeros_like(h5_file[f'tsd_0'][:][:, :, 0][None, ...])
                else:
                    td_accum_front = np.sum([h5_file[f'tsd_{i}'][:][:, :, 0][None, ...].astype(np.float32) / 255.0 * 2.0 - 1.0 for i in range(div + 1, 26)], axis=0)
                
                
                td_h, td_w = td_accum_back.shape[1], td_accum_back.shape[2]  # Get the target height and width from td_accum

                # gt
                gt = np.transpose(h5_file[f'gt_{div}'][:], (2, 0, 1)).astype(np.float32) / 255.0  # Shape: 3 * H * W
                gt_resized = resize(gt, (3, td_h, td_w), preserve_range=True)  # Resize to 3 * H_new * W_new

                img_path = self.hdf5s[idx]
        
        elif self.hdf5_mode=="multi":
            with h5py.File(self.hdf5s[idx], 'r') as h5_file:
                group_count = sum(1 for item in h5_file.values() if isinstance(item, h5py.Group))
                pack_idx = random.randint(0, group_count - 2) if self.val_pack is None else self.val_pack

                div = random.randint(0, 25) if self.val_div is None else self.val_div

                images_back = h5_file[str(pack_idx)]
                images_front = h5_file[str(pack_idx+1)]

                # rgb
                if self.rgb_key == "rgb":
                    rgb_back_norm = np.transpose(images_back['rgb'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0
                    rgb_front_norm = np.transpose(images_front['rgb'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0
                elif self.rgb_key == "F":
                    rgb_back_norm = np.transpose(images_back['F0'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0
                    rgb_front_norm = np.transpose(images_front['F0'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0
                elif self.rgb_key == "GT":
                    rgb_back_norm = np.transpose(images_back['gt_0'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0
                    rgb_front_norm = np.transpose(images_front['gt_0'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0

                # sd
                if div == 25:
                    sd = np.transpose(images_front[f'tsd_0'][:][:, :, 1:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0  # Shape: 2 * H * W
                else:
                    sd = np.transpose(images_back[f'tsd_{div}'][:][:, :, 1:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0  # Shape: 2 * H * W

                # td_accum
                if div == 0:
                    td_accum_back = np.zeros_like(images_back[f'tsd_0'][:][:, :, 0][None, ...])
                    td_accum_front = np.sum([images_back[f'tsd_{i}'][:][:, :, 0][None, ...].astype(np.float32) / 255.0 * 2.0 - 1.0 for i in range(1, 25)] + [images_front[f'tsd_0'][:][:, :, 0][None, ...].astype(np.float32) / 255.0 * 2.0 - 1.0], axis=0)
                elif div == 25:
                    td_accum_back = np.sum([images_back[f'tsd_{i}'][:][:, :, 0][None, ...].astype(np.float32) / 255.0 * 2.0 - 1.0 for i in range(1, 25)] + [images_front[f'tsd_0'][:][:, :, 0][None, ...].astype(np.float32) / 255.0 * 2.0 - 1.0], axis=0)
                    td_accum_front = np.zeros_like(images_back[f'tsd_0'][:][:, :, 0][None, ...])
                else:
                    td_accum_back = np.sum([images_back[f'tsd_{i}'][:][:, :, 0][None, ...].astype(np.float32) / 255.0 * 2.0 - 1.0 for i in range(1, div + 1)], axis=0)
                    td_accum_front = np.sum([images_back[f'tsd_{i}'][:][:, :, 0][None, ...].astype(np.float32) / 255.0 * 2.0 - 1.0 for i in range(div + 1, 25)] + [images_front[f'tsd_0'][:][:, :, 0][None, ...].astype(np.float32) / 255.0 * 2.0 - 1.0], axis=0)
                
                td_h, td_w = td_accum_back.shape[1], td_accum_back.shape[2]  # Get the target height and width from td_accum

                # gt
                if div == 25:
                    gt = np.transpose(images_front[f'gt_0'][:], (2, 0, 1)).astype(np.float32) / 255.0  # Shape: 3 * H * W
                else:
                    gt = np.transpose(images_back[f'gt_{div}'][:], (2, 0, 1)).astype(np.float32) / 255.0  # Shape: 3 * H * W
                
                gt_resized = resize(gt, (3, td_h, td_w), preserve_range=True)  # Resize to 3 * H_new * W_new

                img_path = self.hdf5s[idx]


        rgb_back_norm_tensor = torch.tensor(rgb_back_norm, dtype=torch.float32)
        rgb_front_norm_tensor = torch.tensor(rgb_front_norm, dtype=torch.float32)
        
        td_accum_back_tensor = torch.tensor(td_accum_back, dtype=torch.float32)
        td_accum_front_tensor = torch.tensor(td_accum_front, dtype=torch.float32)

        sd_tensor = torch.tensor(sd, dtype=torch.float32)
        gt_tensor = torch.tensor(gt_resized, dtype=torch.float32)

        if self.resize_to_hw is not None:
            rgb_back_norm_tensor, rgb_front_norm_tensor, td_accum_back_tensor, td_accum_front_tensor, sd_tensor, gt_tensor = self.resize_output(self.resize_to_hw, rgb_back_norm_tensor, rgb_front_norm_tensor, td_accum_back_tensor, td_accum_front_tensor, sd_tensor, gt_tensor)

        gt_lr = self.resize_output(self.resize_to_hw, self.resize_output(self.midsize, gt_tensor)[0])[0]

        result = {
            'rgb_norm': rgb_back_norm_tensor,
            'td_back_norm': td_accum_back_tensor,
            'rgb_front_norm': rgb_front_norm_tensor,
            'td_front_norm': td_accum_front_tensor,
            'sd_norm': sd_tensor,
            'gt_norm': gt_tensor*2.0-1.0,
            'gt': gt_tensor,
            'gt_lr_norm': gt_lr*2.0-1.0,
            'rgb_relative_path': img_path,
            'div_idx_float': torch.tensor(div/25, dtype=torch.float32),  # 0-1
            'pack_idx': str(pack_idx),
            'div_idx': str(div)
        }

        return result


class TianmoucMetadataInterpolationDatasetHdf5BTCHW(TianmoucMetadataInterpolationDatasetHdf5Single):

    def __init__(self, disp_name, dir, resize_to_hw=None, val_pack=None, val_div=None, rgb_key="rgb", mode="train", hdf5_mode="single", T_sample=6, T_step_max=5, select_divs=None):
        
        if mode == "train":
            assert val_pack is None, "For training mode, val_pack should be None"
            assert select_divs is not None or T_sample == 1, "For training mode, either select_divs should be given or T_sample should be 1 (randomly select one frame)"

        super().__init__(disp_name, dir, resize_to_hw, val_pack, val_div, rgb_key, mode, hdf5_mode)

        self.select_divs = select_divs
        self.T_sample = T_sample
        self.T_step_max = T_step_max

        self.size_multi = 1

    def __len__(self):

        return len(self.hdf5s)*self.size_multi
    
    def __getitem__(self, idx):

        hdf_path = self.hdf5s[idx]
        idx = idx % len(self.hdf5s)

        if self.select_divs is None:
            if self.mode == "val":
                divs = list(range(26)) if self.val_div is None else [self.val_div]
            else:
                self.T_step = random.randint(1, self.T_step_max)
                self.T_start_max = 25-(self.T_sample-1)*self.T_step
                self.T_start = random.randint(0, self.T_start_max)
                divs = [self.T_start + i * self.T_step for i in range(self.T_sample)]
        else:
            divs = self.select_divs

        rgb_back_norms = []
        rgb_front_norms = []
        sds = []
        td_accum_backs = []
        td_accum_fronts = []
        gts = []
        div_idx_float = []

        if self.hdf5_mode == "single":

            with h5py.File(hdf_path, 'r') as h5_file:
                img_path = hdf_path
                pack_idx = 0

                for div in divs:
                    # rgb
                    if self.rgb_key != "GT":
                        rgb_back_norm = np.transpose(h5_file[f'{self.rgb_key}0'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0
                        rgb_front_norm = np.transpose(h5_file[f'{self.rgb_key}1'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0
                    else:
                        rgb_back_norm = np.transpose(h5_file['gt_0'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0
                        rgb_front_norm = np.transpose(h5_file['gt_25'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0

                    # sd
                    sd = np.transpose(h5_file[f'tsd_{div}'][:][:, :, 1:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0

                    # td_accum
                    if div == 0:
                        td_accum_back = np.zeros_like(h5_file[f'tsd_0'][:][:, :, 0][None, ...])
                    else:
                        td_accum_back = np.sum([h5_file[f'tsd_{i}'][:][:, :, 0][None, ...].astype(np.float32) / 255.0 * 2.0 - 1.0 for i in range(1, div + 1)], axis=0)
                    if div == 25:
                        td_accum_front = np.zeros_like(h5_file[f'tsd_0'][:][:, :, 0][None, ...])
                    else:
                        td_accum_front = np.sum([h5_file[f'tsd_{i}'][:][:, :, 0][None, ...].astype(np.float32) / 255.0 * 2.0 - 1.0 for i in range(div + 1, 26)], axis=0)
                    
                    
                    td_h, td_w = td_accum_back.shape[1], td_accum_back.shape[2]

                    # gt
                    gt = np.transpose(h5_file[f'gt_{div}'][:], (2, 0, 1)).astype(np.float32) / 255.0
                    gt_resized = resize(gt, (3, td_h, td_w), preserve_range=True)

                    rgb_back_norms.append(rgb_back_norm)
                    rgb_front_norms.append(rgb_front_norm)
                    sds.append(sd)
                    td_accum_backs.append(td_accum_back)
                    td_accum_fronts.append(td_accum_front)
                    gts.append(gt_resized)
                    div_idx_float.append(div/25)
                
        elif self.hdf5_mode=="multi":
            
            with h5py.File(hdf_path, 'r') as h5_file:
                img_path = hdf_path

                group_count = sum(1 for item in h5_file.values() if isinstance(item, h5py.Group))
                pack_idx = random.randint(0, group_count - 2) if self.val_pack is None else self.val_pack
                

                for div in divs:
                
                    images_back = h5_file[str(pack_idx)]
                    images_front = h5_file[str(pack_idx+1)]

                    # rgb
                    if self.rgb_key == "rgb":
                        rgb_back_norm = np.transpose(images_back['rgb'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0
                        rgb_front_norm = np.transpose(images_front['rgb'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0
                    elif self.rgb_key == "F":
                        rgb_back_norm = np.transpose(images_back['F0'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0
                        rgb_front_norm = np.transpose(images_front['F0'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0
                    elif self.rgb_key == "GT":
                        rgb_back_norm = np.transpose(images_back['gt_0'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0
                        rgb_front_norm = np.transpose(images_front['gt_0'][:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0

                    # sd
                    if div == 25:
                        sd = np.transpose(images_front[f'tsd_0'][:][:, :, 1:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0  # Shape: 2 * H * W
                    else:
                        sd = np.transpose(images_back[f'tsd_{div}'][:][:, :, 1:], (2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0  # Shape: 2 * H * W

                    # td_accum
                    if div == 0:
                        td_accum_back = np.zeros_like(images_back[f'tsd_0'][:][:, :, 0][None, ...])
                        td_accum_front = np.sum([images_back[f'tsd_{i}'][:][:, :, 0][None, ...].astype(np.float32) / 255.0 * 2.0 - 1.0 for i in range(1, 25)] + [images_front[f'tsd_0'][:][:, :, 0][None, ...].astype(np.float32) / 255.0 * 2.0 - 1.0], axis=0)
                    elif div == 25:
                        td_accum_back = np.sum([images_back[f'tsd_{i}'][:][:, :, 0][None, ...].astype(np.float32) / 255.0 * 2.0 - 1.0 for i in range(1, 25)] + [images_front[f'tsd_0'][:][:, :, 0][None, ...].astype(np.float32) / 255.0 * 2.0 - 1.0], axis=0)
                        td_accum_front = np.zeros_like(images_back[f'tsd_0'][:][:, :, 0][None, ...])
                    else:
                        td_accum_back = np.sum([images_back[f'tsd_{i}'][:][:, :, 0][None, ...].astype(np.float32) / 255.0 * 2.0 - 1.0 for i in range(1, div + 1)], axis=0)
                        td_accum_front = np.sum([images_back[f'tsd_{i}'][:][:, :, 0][None, ...].astype(np.float32) / 255.0 * 2.0 - 1.0 for i in range(div + 1, 25)] + [images_front[f'tsd_0'][:][:, :, 0][None, ...].astype(np.float32) / 255.0 * 2.0 - 1.0], axis=0)
                    
                    
                    td_h, td_w = td_accum_back.shape[1], td_accum_back.shape[2]  # Get the target height and width from td_accum

                    # gt
                    if div == 25:
                        gt = np.transpose(images_front[f'gt_0'][:], (2, 0, 1)).astype(np.float32) / 255.0  # Shape: 3 * H * W
                    else:
                        gt = np.transpose(images_back[f'gt_{div}'][:], (2, 0, 1)).astype(np.float32) / 255.0  # Shape: 3 * H * W
                    gt_resized = resize(gt, (3, td_h, td_w), preserve_range=True)  # Resize to 3 * H_new * W_new


                    rgb_back_norms.append(rgb_back_norm)
                    rgb_front_norms.append(rgb_front_norm)
                    sds.append(sd)
                    td_accum_backs.append(td_accum_back)
                    td_accum_fronts.append(td_accum_front)
                    gts.append(gt_resized)
                    div_idx_float.append(div/25)
                
        # [CHW, ...] -> TCHW
        rgb_back_norm = np.stack(rgb_back_norms, axis=0)
        rgb_front_norm = np.stack(rgb_front_norms, axis=0)
        td_accum_back = np.stack(td_accum_backs, axis=0)
        td_accum_front = np.stack(td_accum_fronts, axis=0)
        sd = np.stack(sds, axis=0)
        gt_resized = np.stack(gts, axis=0)

        rgb_back_norm_tensor = torch.tensor(rgb_back_norm, dtype=torch.float32)
        rgb_front_norm_tensor = torch.tensor(rgb_front_norm, dtype=torch.float32)

        td_accum_back_tensor = torch.tensor(td_accum_back, dtype=torch.float32)
        td_accum_front_tensor = torch.tensor(td_accum_front, dtype=torch.float32)

        sd_tensor = torch.tensor(sd, dtype=torch.float32)
        gt_tensor = torch.tensor(gt_resized, dtype=torch.float32)


        if self.resize_to_hw is not None:
            rgb_back_norm_tensor, rgb_front_norm_tensor, td_accum_back_tensor, td_accum_front_tensor, sd_tensor, gt_tensor = self.resize_output(self.resize_to_hw, rgb_back_norm_tensor, rgb_front_norm_tensor, td_accum_back_tensor, td_accum_front_tensor, sd_tensor, gt_tensor)
        
        if self.T_sample != 1:
            # return TCHW
            result = {
                'rgb_norm': rgb_back_norm_tensor,
                'td_back_norm': td_accum_back_tensor,
                'rgb_front_norm': rgb_front_norm_tensor,
                'td_front_norm': td_accum_front_tensor,
                'sd_norm': sd_tensor,
                'gt_norm': gt_tensor*2.0-1.0,
                'gt': gt_tensor,
                'rgb_relative_path': img_path,
                'folder_name': os.path.splitext(os.path.basename(img_path))[0],
                'div_idx_float': torch.tensor(div_idx_float, dtype=torch.float32),  # 0-1
                'pack_idx': str(pack_idx),
                'div_idx': str(0)
            }
        else:
            # return CHW
            result = {
                'rgb_norm': rgb_back_norm_tensor[0],
                'td_back_norm': td_accum_back_tensor[0],
                'rgb_front_norm': rgb_front_norm_tensor[0],
                'td_front_norm': td_accum_front_tensor[0],
                'sd_norm': sd_tensor[0],
                'gt_norm': gt_tensor[0]*2.0-1.0,
                'gt': gt_tensor[0],
                'rgb_relative_path': img_path,
                'folder_name': os.path.splitext(os.path.basename(img_path))[0],
                'div_idx_float': torch.tensor(div_idx_float[0], dtype=torch.float32),  # 0-1
                'pack_idx': str(pack_idx),
                'div_idx': str(div)
            }


        return result

 