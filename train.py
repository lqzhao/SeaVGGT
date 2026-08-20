# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pdb
import piq
import glob
import time
import random
import threading
import argparse
from typing import List, Optional


import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from oct2py import Oct2Py
from matplotlib import colormaps

from uw_model import UnderwaterEnhanceNet, LearnableLinear, TransmissionEstimator
from uw_model_new_new import TokenPrototypeModulator, LightAEstimator, TokenPrototypeModulatorGAT
from uw_loss import (UnderwaterLoss, UWConsistencyLoss, FirstFrameConsistencyLoss, UIQM, ColorLoss, 
                    TransmissionDepthConsistencyLoss, gradient_consistency_loss, ConsistencyLoss,
                    compute_losses_new, compute_inf_mask, compute_reconstruction_loss, compute_mixup_consistency_loss,
                    build_photometric_loss, build_photometric_loss_pairwise)
from visual_util import (save_multiframe_colored_pointcloud, estimate_beta_A_multiframe_rgb_tensor, 
                        compute_J_from_d_v2, get_A, estimate_A, estimate_A_darkchannel, compute_A_gt,
                        scale_align, compute_depth_metrics, compute_si_rmse, mixup_old_new_tokens)
from dataset import RandomSequencePathDataset, SingleImageDataset, SingleImageDepthDataset

from vggt.heads.dpt_head import DPTHead
from vggt.heads.camera_head import CameraHead

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from tqdm import tqdm


try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def save_final_image_as_jpg(final_imgs, save_dir, iter_idx=0, prefix="final"):
    """
    保存最终增强后的图像（Image_final）
    final_imgs: shape [B, 3, H, W]，取值范围 [0, 1]，tensor
    图片命名格式：{prefix}_iter{iter_idx}_j{j}.jpg
    """
    os.makedirs(save_dir, exist_ok=True)

    final_imgs = torch.clamp(final_imgs.detach().cpu(), 0.0, 1.0)
    B, C, H, W = final_imgs.shape
    to_pil = transforms.ToPILImage()

    save_paths = []

    for j in range(B):
        img_tensor = final_imgs[j]
        img = to_pil(img_tensor)
        save_path = os.path.join(save_dir, f"{prefix}_iter{iter_idx}_j{j}.jpg")
        img.save(save_path)
        save_paths.append(save_path)

    # print(f"✅ Saved {B} final images to {save_dir} (prefix={prefix}, iter={iter_idx})")


def save_concatenated_images_with_labels(J, T, D, depth, I_deg, I, A_tensor, A_pred_tensor, iter_idx, save_dir='output_concat', cmap_name='plasma'):
    """
    将 J、T、D、depth、I_deg、I、A_pred、A_tensor 拼成一张横向带文字标注的大图保存
    J, T, I_deg, I: tensor, shape [b, 3, h, w]
    D, depth: tensor, shape [1, n, h, w, 1]
    A_tensor, A_pred_tensor: tensor, shape [1, n, 3]
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm

    os.makedirs(save_dir, exist_ok=True)
    
    b, c, h, w = J.shape
    num_frames = D.shape[1]

    titles = ['J', 'T', 'D', 'Depth_GT', 'I_deg', 'I', 'A_pred', 'A_select']

    for i in range(b):
        fig, axes = plt.subplots(1, 8, figsize=(32, 5))
        plt.subplots_adjust(wspace=0.05)

        # J
        img_J = J[i].detach().cpu().numpy().transpose(1, 2, 0)
        img_J = np.clip(img_J * 255.0, 0, 255).astype(np.uint8)
        axes[0].imshow(img_J)
        axes[0].set_title('J')
        axes[0].axis('off')

        # T
        img_T = T[i].detach().cpu().numpy().transpose(1, 2, 0)
        img_T = np.clip(img_T * 255.0, 0, 255).astype(np.uint8)
        axes[1].imshow(img_T)
        axes[1].set_title('T')
        axes[1].axis('off')

        # D
        if i < num_frames:
            depth_map = D[0, i, :, :, 0].cpu().numpy()
            depth_norm = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-5)
            cmap = cm.get_cmap(cmap_name)
            depth_color_uint8 = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)
        else:
            depth_color_uint8 = np.zeros((h, w, 3), dtype=np.uint8)
        axes[2].imshow(depth_color_uint8)
        axes[2].set_title('D')
        axes[2].axis('off')

        # Depth_GT
        if i < num_frames:
            depth_gt_map = depth[0, i, :, :, 0].cpu().numpy()
            depth_gt_norm = (depth_gt_map - np.min(depth_gt_map)) / (np.max(depth_gt_map) - np.min(depth_gt_map) + 1e-5)
            cmap = cm.get_cmap(cmap_name)
            depth_gt_color_uint8 = (cmap(depth_gt_norm)[:, :, :3] * 255).astype(np.uint8)
        else:
            depth_gt_color_uint8 = np.zeros((h, w, 3), dtype=np.uint8)
        axes[3].imshow(depth_gt_color_uint8)
        axes[3].set_title('Depth_init')
        axes[3].axis('off')

        # I_deg
        img_I_deg = I_deg[i].detach().cpu().numpy().transpose(1, 2, 0)
        img_I_deg = np.clip(img_I_deg * 255.0, 0, 255).astype(np.uint8)
        axes[4].imshow(img_I_deg)
        axes[4].set_title('I_deg')
        axes[4].axis('off')

        # I
        img_I = I[i].detach().cpu().numpy().transpose(1, 2, 0)
        img_I = np.clip(img_I * 255.0, 0, 255).astype(np.uint8)
        axes[5].imshow(img_I)
        axes[5].set_title('I')
        axes[5].axis('off')

        # A_pred
        a_pred_val = A_pred_tensor[0, 0, 0, 0].detach().cpu().numpy()  # shape (3,)
        a_pred_img = np.ones((h, w, 3), dtype=np.float32)
        a_pred_img[:, :, 0] *= a_pred_val[0]
        a_pred_img[:, :, 1] *= a_pred_val[1]
        a_pred_img[:, :, 2] *= a_pred_val[2]
        a_pred_img_uint8 = np.clip(a_pred_img * 255.0, 0, 255).astype(np.uint8)
        axes[6].imshow(a_pred_img_uint8)
        axes[6].set_title('A_pred')
        axes[6].axis('off')

        # A_gt
        a_val = A_tensor[0, 0, 0, 0].detach().cpu().numpy()
        a_img = np.ones((h, w, 3), dtype=np.float32)
        a_img[:, :, 0] *= a_val[0]
        a_img[:, :, 1] *= a_val[1]
        a_img[:, :, 2] *= a_val[2]
        a_img_uint8 = np.clip(a_img * 255.0, 0, 255).astype(np.uint8)
        axes[7].imshow(a_img_uint8)
        axes[7].set_title('A_select')
        axes[7].axis('off')

        # 保存
        save_path = os.path.join(save_dir, f'iter_{iter_idx}_img_{i}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()





parser = argparse.ArgumentParser(description="VGGT demo with viser for 3D visualization")
parser.add_argument(
    "--image_folder", type=str, default="/home/zlq/code/uw/vggt/examples/uw", help="Path to folder containing images"
)
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
parser.add_argument(
    "--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out"
)
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")

def main():
    """
    Main function for the VGGT demo with viser for 3D visualization.

    This function:
    1. Loads the VGGT model
    2. Processes input images from the specified folder
    3. Runs inference to generate 3D points and camera poses
    4. Optionally applies sky segmentation to filter out sky points
    5. Visualizes the results using viser

    Command-line arguments:
    --image_folder: Path to folder containing input images
    --use_point_map: Use point map instead of depth-based points
    --background_mode: Run the viser server in background mode
    --port: Port number for the viser server
    --conf_threshold: Initial percentage of low-confidence points to filter out
    --mask_sky: Apply sky segmentation to filter out sky points
    """
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # uw_T_head = DPTHead(dim_in=2048, output_dim=4, activation="sigmoid", conf_activation="expp1").cuda()
    # uw_T_head.load_state_dict(model.point_head.state_dict())
    uw_J_head = DPTHead(dim_in=2048, output_dim=4, activation="linear", conf_activation="expp1").cuda()
    uw_J_head.load_state_dict(model.point_head.state_dict())
    # uw_D_head = DPTHead(dim_in=2048, output_dim=2, activation="exp", conf_activation="expp1").cuda()
    # uw_D_head.load_state_dict(model.depth_head.state_dict())
    # uw_beta_head = CameraHead(dim_in=2048, fl_act="linear").cuda()
    # linear_layer = TransmissionEstimator(feature_dim=3, max_depth=5.0, min_beta=0.55, max_beta=10.0).cuda()

    # uw_net = UnderwaterDepthResidualNet().cuda()
    modulator = TokenPrototypeModulatorGAT(token_dim=2048, num_prototypes=24).cuda()
    A_net = LightAEstimator().cuda()

    # 加载权重
    model_path = "checkpoints/epoch_01_best_rmse_0.6133_24proto_sota.pth" #  epoch_01_best_result_35.1193
    checkpoint = torch.load(model_path)
    uw_J_head.load_state_dict(checkpoint['uw_J_head_state_dict'])
    # modulator.load_state_dict(checkpoint['modulator_state_dict'], strict=False)
    A_net.load_state_dict(checkpoint['A_net_state_dict'])
    print("Model weights loaded.")


    # optimizer = torch.optim.AdamW(
    #     # list(uw_T_head.parameters()) + 
    #     list(linear_layer.parameters()) +
    #     list(uw_beta_head.parameters()) + 
    #     list(uw_J_head.parameters()) +
    #     list(uw_D_head.parameters()), 
    #     lr=5e-4, weight_decay=1e-4
    # )
    optimizer = torch.optim.AdamW(
        # list(uw_T_head.parameters()) + 
        # list(linear_layer.parameters()) +
        # list(uw_net.parameters())+
        list(uw_J_head.parameters()) +
        list(modulator.parameters()) +       
        list(A_net.parameters()), 
        lr=1e-4, weight_decay=1e-4
    )

    print("Optimizer parameters:")
    for name, param in modulator.named_parameters():
        if "A_values" in name:
            print(f"  - {name}: requires_grad={param.requires_grad}, shape={param.shape}")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = 50
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # thre = 0.2
    mse_loss = torch.nn.MSELoss().cuda()
    color_loss = ColorLoss()
    # beta_d_loss = TransmissionDepthConsistencyLoss()
    # geometry_loss = FirstFrameConsistencyLoss(threshold=thre)
    new_mse_loss = ConsistencyLoss(lambda_l1=1.0, lambda_ssim=1.0, lambda_percep=0.3, lambda_grad=0.3).cuda()


    folder_list1 = ['/data1/data/zlq_datasets/underwater/SeathruNeRF_dataset/IUI3-RedSea/Images_wb', 
                   '/data1/data/zlq_datasets/underwater/SeathruNeRF_dataset/Curasao/images_wb', 
                   '/data1/data/zlq_datasets/underwater/SeathruNeRF_dataset/JapaneseGradens-RedSea/images_wb',
                   '/data1/data/zlq_datasets/underwater/SeathruNeRF_dataset/Panama/images_wb',
                   ]
    # base_path = "/data1/data/wuhao/underwater_dataset/mvk/train_colmap"
    # folder_list2 = glob.glob(os.path.join(base_path, "*", "*", "metric_select03/rgb"))

    folder_list2 = open('valid_folders.txt').read().splitlines()

    # merged_list = folder_list1 + folder_list2
    train_dataset = SingleImageDataset(folder_list2, random=False, is_train=True)

    # train_dataset = RandomSequencePathDataset(folder_list2, max_interval=2)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)

    folder_list1_val = ['/home/zlq/code/uw/USUIR-main/Dataset/UIE/UIEBD/test/image',]
    val_dataset = SingleImageDataset(folder_list1_val)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

    test_image_folder=["/data1/data/wuhao/underwater_dataset/FLsea/canyons/u_canyon/imgs"]
    test_dataset = SingleImageDepthDataset(test_image_folder)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    best_rmse = float('inf')  # 初始最好RMSE
    matlab_func_dir = '/home/zlq/code/uw/vggt/UCIQE'
    image_folder = '/home/zlq/code/uw/vggt/output_J_val/'

    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

        uw_J_head.train()
        modulator.train()
        A_net.train()

        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}")

        for i, (images, paths) in pbar:
            images = images.to(device).squeeze(0)
            num_images = images.shape[0]

            if num_images != 1:
                images = images.squeeze(1)

            # A_pred = estimate_A_darkchannel(images, ratio=0.001, r=7, eps=1e-3)  # [B, 3]
            A_pred = A_net(images)
            aggregated_tokens_list, patch_start_idx = model.aggregator(images.unsqueeze(1))
            with torch.no_grad():
                D_init, D_conf = model.depth_head(
                    aggregated_tokens_list, images=images.unsqueeze(1), patch_start_idx=patch_start_idx
                )
            #################
            aggregated_tokens_list_new, A, beta, scale = modulator(aggregated_tokens_list, A_pred)
            #################
            # depth_head不需要no_grad，保留前向和梯度回传，但冻结参数
            depth, depth_conf = model.depth_head(
                aggregated_tokens_list_new, images=images.unsqueeze(1), patch_start_idx=patch_start_idx
            )

            J_out, J_conf = uw_J_head(aggregated_tokens_list_new, images=images.unsqueeze(1), patch_start_idx=patch_start_idx)
            # J = torch.clamp(images.permute(0, 2, 3, 1).unsqueeze(0) + torch.tanh(J_out), 0.0, 1.0)
            J = torch.sigmoid(J_out)
            # J = 0.99 * torch.sigmoid(J_out) + 0.01 * images.permute(0, 2, 3, 1).unsqueeze(1)

            k = scale[..., 0:1]  ; b = scale[..., 1:2] ; norm_depth = k * depth + b
            T = torch.exp(-beta * norm_depth) # [1, 1, 1, 1, 3]
            # I_deg = J * T + (1 - T) * A.view(num_images, 1, 1, 1, 3)
            I_deg = J * T + (1 - T) * A_pred.view(num_images, 1, 1, 1, 3)

            ################## image mixup ####################
            lam = np.random.beta(1, 1)
            input_mix = lam * images.unsqueeze(1) + (1 - lam) * J.permute(0, 1, 4, 2, 3)
            aggregated_tokens_list_new, patch_start_idx_new = model.aggregator(input_mix)

            ################## token mixup ####################
            # aggregated_tokens_list_mixed, lam = mixup_old_new_tokens(aggregated_tokens_list, aggregated_tokens_list_new, 1)
            # input_mix = (1 - lam) * images.unsqueeze(1) + lam * J.permute(0, 1, 4, 2, 3)
            # patch_start_idx_new = patch_start_idx
            
            aggregated_tokens_list_new, A, beta, scale = modulator(aggregated_tokens_list_new, A_pred)
            J_out_mix, J_conf_mix = uw_J_head(aggregated_tokens_list_new, images=input_mix, patch_start_idx=patch_start_idx_new)
            # J_mix = torch.clamp(images.permute(0, 2, 3, 1).unsqueeze(0) + torch.tanh(J_out_mix), 0.0, 1.0)
            J_mix = torch.sigmoid(J_out_mix)
            # J_mix = 0.99 * torch.sigmoid(J_out_mix) + 0.01 * images.permute(0, 2, 3, 1).unsqueeze(1)

            A_gt = compute_A_gt(images, depth)

            # loss_1 = 100 * mse_loss(I_deg, images.unsqueeze(0).permute(0,1,3,4,2))
            # pdb.set_trace()
            loss_1 = 10 * new_mse_loss(I_deg.permute(0, 1, 4, 2, 3), images.unsqueeze(1))
            loss_2 = 10 * mse_loss(J_mix, J.detach())
            loss_3 = color_loss(J.permute(0, 1, 4, 2, 3).squeeze(0))
            loss_4 = 10 * mse_loss(A_pred, A_gt.detach())
            loss_5 = 10 * mse_loss(A, A_gt.detach()) # 加不加？

            loss = 1 * loss_1 + 1 * loss_2 + 0.01 * loss_3 + 1 * loss_4 + 1 * loss_5

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # tqdm实时更新
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Recon': f"{loss_1.item():.4f}",
                'Mix': f"{loss_2.item():.4f}",
                'Color': f"{loss_3.item():.4f}",
                'A_pred': f"{loss_4.item():.4f}",
                'A': f"{loss_5.item():.4f}"
            })

            if i % 200 == 0:
                # print(f"\n[Epoch {epoch+1} | Step {i}] "
                #       f"loss_recon={loss_1:.4f}, "
                #       f"loss_mixup={loss_2:.4f}, "
                #       f"loss_color={loss_3:.4f}, "
                #       f"total={loss.item():.4f}")
                # pdb.set_trace()

                # 图片保存（新结构变量）
                save_concatenated_images_with_labels(
                    J.permute(0, 1, 4, 2, 3).squeeze(0),
                    T.permute(0, 1, 4, 2, 3).squeeze(0),
                    depth.cpu().detach(),
                    D_init.cpu().detach(),
                    I_deg.permute(0, 1, 4, 2, 3).squeeze(0),
                    images,
                    A.view(1, 1, 1, 1, 3).cpu().detach(),
                    A_pred.view(1, 1, 1, 1, 3).cpu().detach(),
                    iter_idx=f"epoch{epoch+1}_step{i}"
                )
                # if i==2000:
                #     pdb.set_trace()

            
            if (i % 2000 == 0) and (i>0):
                uw_J_head.eval()
                modulator.eval()
                A_net.eval()

                rmse_list = []

                for j, (images, gt_depths, _) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                    with torch.no_grad():
                        images = images.to(device).squeeze(0)
                        gt_depths = gt_depths.to(device)
                        gt_depth = gt_depths[0, 0, 0]  # 假设gt_depth shape: [1, 1, H, W]

                        A_pred = A_net(images)
                        aggregated_tokens_list, patch_start_idx = model.aggregator(images.unsqueeze(0))

                        D_init, _ = model.depth_head(
                            aggregated_tokens_list, images=images.unsqueeze(0), patch_start_idx=patch_start_idx
                        )
                        D_init_pred = D_init[0, 0, :, :, 0]

                        # 调整 token
                        aggregated_tokens_list, A, beta, scale = modulator(aggregated_tokens_list, A_pred)

                        # 再预测
                        depth, _ = model.depth_head(
                            aggregated_tokens_list, images=images.unsqueeze(0), patch_start_idx=patch_start_idx
                        )
                        Depth_pred = depth[0, 0, :, :, 0]

                        # J图像恢复
                        J_out, _ = uw_J_head(aggregated_tokens_list, images=images.unsqueeze(0), patch_start_idx=patch_start_idx)
                        J = torch.sigmoid(J_out)

                        # 模型中的scale和光学衰减T
                        k = scale[..., 0:1]
                        b = scale[..., 1:2]
                        norm_depth = k * depth + b
                        T = torch.exp(-beta * norm_depth)

                        # I_deg = J * T + (1 - T) * A.view(1, 1, 1, 1, 3)
                        I_deg = J * T + (1 - T) * A_pred.view(1, 1, 1, 1, 3)

                        # 保存图像
                        save_concatenated_images_with_labels(
                            J.permute(0, 1, 4, 2, 3).squeeze(0),
                            T.permute(0, 1, 4, 2, 3).squeeze(0),
                            depth.cpu().detach(),
                            D_init.cpu().detach(),
                            I_deg.permute(0, 1, 4, 2, 3).squeeze(0),
                            images,
                            A.view(1, 1, 1, 1, 3).cpu().detach(),
                            A_pred.view(1, 1, 1, 1, 3).cpu().detach(),
                            iter_idx=f"epoch{epoch+1}_val{j}",
                            save_dir='output_concat_val'
                        )

                        gt_depth = F.interpolate(
                            gt_depth.unsqueeze(0).unsqueeze(0),  # [1,1,H,W]
                            size=Depth_pred.shape,
                            mode='nearest'   # 深度图通常用 nearest，防止产生奇怪插值
                        ).squeeze(0).squeeze(0)
                        # 计算深度指标
                        D_init_aligned = scale_align(D_init_pred, gt_depth)
                        Depth_aligned  = scale_align(Depth_pred, gt_depth)

                        metrics = compute_depth_metrics(Depth_aligned, gt_depth)
                        rmse = metrics['RMSE']
                        rmse_list.append(rmse)

                # 平均RMSE
                mean_rmse = np.mean(rmse_list)
                print(f"[Epoch {epoch+1}] Validation RMSE: {mean_rmse:.4f}")

                # 判断是否是最优
                if mean_rmse < best_rmse:
                    best_rmse = mean_rmse
                    best_model_path = os.path.join(save_dir, f'epoch_{epoch+1:02d}_best_rmse_{best_rmse:.4f}.pth')

                    torch.save({
                        'epoch': epoch + 1,
                        'uw_J_head_state_dict': uw_J_head.state_dict(),
                        'modulator_state_dict': modulator.state_dict(),
                        'A_net_state_dict': A_net.state_dict(),
                        # 'optimizer_state_dict': optimizer.state_dict(),
                        # 'scheduler_state_dict': scheduler.state_dict(),
                    }, best_model_path)

                    print(f">>> New best model (RMSE {best_rmse:.4f}) saved to: {best_model_path}")

        scheduler.step()
    # recon_loss = F.l1_loss(I_deg, images)
    
    pdb.set_trace()

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    print("Processing model outputs...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

    # predictions keys (['pose_enc', 'depth', 'depth_conf', 'world_points', 'world_points_conf', 'images'])
    pdb.set_trace()



if __name__ == "__main__":
    main()
