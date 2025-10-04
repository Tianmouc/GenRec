import os
import cv2
import numpy as np
import torch
from tianmoucv.data import TianmoucDataReader
from CBRDM.TMRec_pipeline import TianmoucCascadedReconstructionPipeline
import argparse
import subprocess

"""
Example:
    python demo.py --sample_name VanGogh --device cuda:0
    python demo.py --sample_name qrcode_rotate --device cuda:1
    python demo.py --sample_name dog_rotate --device cuda:2
    python demo.py --sample_name qrcode_shaking --device cuda:3
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Demo for Tianmouc Cascaded Reconstruction')
    parser.add_argument('--sample_name', type=str, required=True, help='Sample name')
    parser.add_argument('--cop_idx', type=int, nargs='+', default=None, help='Frame index range, e.g. 49 51')
    parser.add_argument('--output_folder', type=str, default='./demo_output', help='Output folder (default: ./demo_output)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (default: cuda:0)')
    parser.add_argument('--max_eval_sr_bs', type=int, default=8, help='Max SR eval batch size (default: 8), if OOM, try smaller value')
    return parser.parse_args()


def vizDiff_BBG(diff, thresh=0, gain=4):
    """
    Visualize spatio-temporal difference map as blue-yellow contrast image.

    Args:
        diff (np.ndarray): Input difference map (H, W), values in [-1, 1].
        thresh (float): Threshold to suppress small values.
        gain (float): Amplification factor.

    Returns:
        np.ndarray: RGB visualization (H, W, 3), float in [0,1], order [B,G,R].
    """
    diff = np.clip(diff, -1, 1)
    h, w = diff.shape
    rgb_diff = np.zeros((3, h, w), dtype=diff.dtype)

    diff[np.abs(diff) < thresh] = 0
    diff = np.clip(diff * gain, -1, 1)

    # negative -> yellow (R+G)
    neg_mask = diff < 0
    rgb_diff[1, neg_mask] = -diff[neg_mask]
    rgb_diff[2, neg_mask] = -diff[neg_mask]

    # positive -> blue
    pos_mask = diff > 0
    rgb_diff[0, pos_mask] = diff[pos_mask]

    return np.transpose(rgb_diff, (1, 2, 0))

def mp4_to_gif_with_palette(mp4_path, gif_path, fps=12, width=640):
    palette_path = os.path.join(os.path.dirname(gif_path), "_palette.png")
    vf_common = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run([
        "ffmpeg", "-y", "-i", mp4_path,
        "-vf", f"{vf_common},palettegen", palette_path
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", mp4_path, "-i", palette_path,
        "-filter_complex", f"{vf_common}[x];[x][1:v]paletteuse",
        gif_path
    ], check=True)
    try:
        os.remove(palette_path)
    except OSError:
        pass

def save_combined_images(gt_pred_ts, lr_pred_ts, rgb_back, rgb_front, td_back_accum, td_front_accum, sd, save_path):
    """
    Save combined visualization as a single image (for qualitative comparison).
    """
    T, C, H, W = gt_pred_ts.shape
    gt_pred_np = gt_pred_ts.permute(0, 2, 3, 1).cpu().numpy()
    lr_pred_np = lr_pred_ts.permute(0, 2, 3, 1).cpu().numpy()
    rgb_back_np = rgb_back.permute(0, 2, 3, 1).cpu().numpy()
    rgb_front_np = rgb_front.permute(0, 2, 3, 1).cpu().numpy()
    td_back_accum_np = td_back_accum.permute(0, 2, 3, 1).cpu().numpy()
    td_front_accum_np = td_front_accum.permute(0, 2, 3, 1).cpu().numpy()
    sd_np = sd.permute(0, 2, 3, 1).cpu().numpy()

    step = 1
    td_back_accum_np_vis = vizDiff_BBG(np.hstack([td_back_accum_np[i] for i in range(0, T, step)])[:, :, 0] * 0.8, gain=1, thresh=0.05)
    td_front_accum_np_vis = (np.clip(np.hstack([td_back_accum_np[i] for i in range(0, T, step)]), -1.0, 1.0) + 1.0) / 2.0
    sd_vis = vizDiff_BBG(np.hstack([sd_np[i] for i in range(0, T, step)])[:, :, 1], gain=16, thresh=0.05)

    combined_images = np.vstack([
        np.hstack([rgb_back_np[i] for i in range(0, T, step)]),
        np.hstack([rgb_front_np[i] for i in range(0, T, step)]),
        td_back_accum_np_vis,
        td_front_accum_np_vis,
        sd_vis,
        np.hstack([lr_pred_np[i] for i in range(0, T, step)]),
        np.hstack([gt_pred_np[i] for i in range(0, T, step)])
    ])

    combined_images = (combined_images * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(combined_images, cv2.COLOR_RGB2BGR))


def save_as_video(gt_pred_ts, lr_pred_ts, rgb_back, rgb_front, td_back_accum, td_front_accum, sd):
    """
    Assemble frames into a video visualization with four quadrants:
      - Top-left: RGB
      - Top-right: Reconstruction
      - Bottom-left: SD
      - Bottom-right: TD Accumulate
    """
    T, C, H, W = gt_pred_ts.shape
    gt_pred_np = gt_pred_ts.permute(0, 2, 3, 1).cpu().numpy()
    lr_pred_np = lr_pred_ts.permute(0, 2, 3, 1).cpu().numpy()
    rgb_back_np = rgb_back.permute(0, 2, 3, 1).cpu().numpy()
    rgb_front_np = rgb_front.permute(0, 2, 3, 1).cpu().numpy()
    td_back_accum_np = td_back_accum.permute(0, 2, 3, 1).cpu().numpy()
    td_front_accum_np = td_front_accum.permute(0, 2, 3, 1).cpu().numpy()
    sd_np = sd.permute(0, 2, 3, 1).cpu().numpy()

    frames = []

    def put_label(img, text):
        """Add label with semi-transparent background."""
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = np.ascontiguousarray(img)

        overlay = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, thickness = 1.0, 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        rect_x1, rect_y1 = 5, 5
        rect_x2, rect_y2 = rect_x1 + text_w + 10, rect_y1 + text_h + 10

        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        cv2.putText(img, text, (rect_x1 + 5, rect_y2 - 5), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
        return img

    for i in range(T):
        top_left = rgb_back_np[i] if i < T // 2 else rgb_front_np[i]
        top_left = put_label(top_left, "RGB")
        top_right = put_label(gt_pred_np[i], "Reconstruction")
        bottom_left = put_label(vizDiff_BBG(sd_np[i][:, :, 1], gain=8, thresh=0.05), "SD")
        bottom_right = put_label(vizDiff_BBG(td_back_accum_np[i][:, :, 0] * 0.8, gain=1, thresh=0.05), "TD Accumulate")

        top_row = np.hstack([top_left, top_right])
        bottom_row = np.hstack([bottom_left, bottom_right])
        frame = np.vstack([top_row, bottom_row])
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame_bgr)

    return frames


if __name__ == "__main__":
    args = parse_args()

    device = torch.device(args.device)
    seed = args.seed
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    RATIO, RGBOFFSET = 25, 3

    save_dir = os.path.join(args.output_folder, args.sample_name)
    os.makedirs(save_dir, exist_ok=True)

    base_ckpt_dir = "./checkpoints"

    if args.sample_name == "VanGogh":
        data_path = "./demo_data/VanGogh"
        cop_idx = list(range(65, 71)) if args.cop_idx is None else list(range(args.cop_idx[0], args.cop_idx[1]))
    elif args.sample_name == "qrcode_rotate":
        data_path = "./demo_data/qrcode_rotate"
        cop_idx = list(range(5, 10)) if args.cop_idx is None else list(range(args.cop_idx[0], args.cop_idx[1]))
    elif args.sample_name == "dog_rotate":
        data_path = "./demo_data/dog_rotate"
        cop_idx = list(range(7, 10)) if args.cop_idx is None else list(range(args.cop_idx[0], args.cop_idx[1]))
    elif args.sample_name == "qrcode_shaking":
        data_path = "./demo_data/qrcode_shaking"
        cop_idx = list(range(1, 9)) if args.cop_idx is None else list(range(args.cop_idx[0], args.cop_idx[1]))

    dataset = TianmoucDataReader(data_path, N=1, camera_idx=0)

    pipe = TianmoucCascadedReconstructionPipeline.from_pretrained(
        os.path.join(base_ckpt_dir, "TianmoucRec_CBRDM"), 
        first_stage_time_embedding=False, sr_stage_time_embedding=True, BTCHW_mode=True, use_safetensors=False,
        max_eval_sr_bs = args.max_eval_sr_bs
    ).to(device)

    video_frames = []

    for p_idx in cop_idx:
        sample = dataset[p_idx]
        sample2 = dataset[p_idx + 1]
        td_offset = sample['tsdiff'][0, ...].numpy() if RGBOFFSET == 0 else np.concatenate(
            [sample['tsdiff'][0, RGBOFFSET:RATIO, ...].numpy(), sample2['tsdiff'][0, :RGBOFFSET+1, ...].numpy()], axis=0
        )

        rgb_back_norms, rgb_front_norms, sds, td_accum_backs, td_accum_fronts, div_idx_float = [], [], [], [], [], []

        for div_idx in range(0, RATIO+1, 4):
            rgb_back_np = sample["F0"].numpy() * 2.0 - 1.0
            rgb_front = sample["F1"].numpy() * 2.0 - 1.0
            sd_np = (sample['tsdiff'][1:, div_idx+RGBOFFSET, ...].permute(1,2,0).numpy()
                     if div_idx+RGBOFFSET < RATIO else
                     sample2['tsdiff'][1:, div_idx+RGBOFFSET-RATIO, ...].permute(1,2,0).numpy())

            if div_idx == 0:
                td_accum_back = np.zeros((320, 640, 1), dtype=np.float32)
                td_accum_front = np.sum(td_offset[div_idx+1:, ...], axis=0)[..., None]
            elif div_idx == RATIO:
                td_accum_back = np.sum(td_offset[1:div_idx+1, ...], axis=0)[..., None]
                td_accum_front = np.zeros((320, 640, 1), dtype=np.float32)
            else:
                td_accum_back = np.sum(td_offset[1:div_idx+1, ...], axis=0)[..., None]
                td_accum_front = np.sum(td_offset[div_idx+1:, ...], axis=0)[..., None]

            rgb_back = torch.from_numpy(rgb_back_np).permute(2,0,1).to(device).unsqueeze(0)
            td_accum_back = torch.from_numpy(td_accum_back).permute(2,0,1).to(device).unsqueeze(0)
            rgb_front = torch.from_numpy(rgb_front).permute(2,0,1).to(device).unsqueeze(0)
            td_accum_front = torch.from_numpy(td_accum_front).permute(2,0,1).to(device).unsqueeze(0)
            sd = torch.from_numpy(sd_np).permute(2,0,1).to(device).unsqueeze(0)
            time_step = torch.tensor([div_idx/RATIO], dtype=torch.float32)

            rgb_back = torch.flip(rgb_back, dims=[-1])
            td_accum_back = torch.flip(td_accum_back, dims=[-1])
            rgb_front = torch.flip(rgb_front, dims=[-1])
            td_accum_front = torch.flip(td_accum_front, dims=[-1])
            sd = torch.flip(sd, dims=[-1])

            rgb_back_norms.append(rgb_back)
            rgb_front_norms.append(rgb_front)
            sds.append(sd)
            td_accum_backs.append(td_accum_back)
            td_accum_fronts.append(td_accum_front)
            div_idx_float.append(time_step)

        rgb_back = torch.stack(rgb_back_norms, dim=1)
        td_accum_back = torch.stack(td_accum_backs, dim=1)
        rgb_front = torch.stack(rgb_front_norms, dim=1)
        td_accum_front = torch.stack(td_accum_fronts, dim=1)
        sd = torch.stack(sds, dim=1)
        time_step = torch.stack(div_idx_float, dim=1)

        pred = pipe(rgb_back, td_accum_back, rgb_front, td_accum_front, sd,
                    time_step=time_step, denoising_steps=200, sr_denoising_steps=50, generator=generator)

        gt_pred, gt_pred_lr = pred.rec_np, pred.reclr_np
        jpg_save_path = os.path.join(save_dir, f"{p_idx}_{seed}.jpg")

        save_combined_images(torch.from_numpy(gt_pred.squeeze()), torch.from_numpy(gt_pred_lr.squeeze()),
                             (rgb_back.squeeze()+1.0)/2.0, (rgb_front.squeeze()+1.0)/2.0,
                             td_accum_back.repeat(1,1,3,1,1).squeeze(),
                             td_accum_front.repeat(1,1,3,1,1).squeeze(),
                             torch.cat([torch.zeros_like(td_accum_front, device=sd.device), sd], dim=-3).squeeze(),
                             jpg_save_path)

        frame_bgrs = save_as_video(torch.from_numpy(gt_pred.squeeze()), torch.from_numpy(gt_pred_lr.squeeze()),
                                   (rgb_back.squeeze()+1.0)/2.0, (rgb_front.squeeze()+1.0)/2.0,
                                   td_accum_back.repeat(1,1,3,1,1).squeeze(),
                                   td_accum_front.repeat(1,1,3,1,1).squeeze(),
                                   torch.cat([torch.zeros_like(td_accum_front, device=sd.device), sd], dim=-3).squeeze())
        video_frames.extend(frame_bgrs[1:])

    print(f"Making video with {len(video_frames)} frames...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    mp4_path = os.path.join(save_dir, f"{args.sample_name}_{cop_idx[0]}_{cop_idx[-1]}.mp4")
    video_out = cv2.VideoWriter(mp4_path, fourcc, fps=15, frameSize=(2*640, 2*320))
    for frame_bgr in video_frames:
        video_out.write(frame_bgr)
    video_out.release()
    print(f"Demo results are saved at {save_dir}")
    gif_path = os.path.join(save_dir, f"{args.sample_name}_{cop_idx[0]}_{cop_idx[-1]}.gif")
    mp4_to_gif_with_palette(mp4_path, gif_path, fps=15, width=640)
    print(f"GIF saved to: {save_dir}")
