# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pdb
import glob
import time
import threading
import argparse
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import color
import torch.nn.functional as F
from torchvision import models


from uw_model import UnderwaterEnhanceNet
from visual_util import get_A

def depth_aware_projection_loss(I_enhanced, I_input, A, depth, mask=None, k=1.0):
    """
    深度感知背景光投影损失，支持掩码
    Args:
        I_enhanced: 增强图像 [B, C, H, W]
        I_input: 原始图像 [B, C, H, W]
        A: 背景光 [B, C]
        depth: 深度图 [B, H, W]
        mask: 掩码 [B, 1, H, W] 或 None
        k: 衰减系数
    Returns:
        loss: 标量
    """
    B, C, H, W = I_input.shape

    delta_I = I_enhanced - I_input  # [B, C, H, W]
    A_norm = torch.norm(A, dim=1, keepdim=True) + 1e-8  # [B, 1]
    A_expand = A.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]

    proj = (delta_I * A_expand).sum(dim=1, keepdim=True) / A_norm.unsqueeze(-1).unsqueeze(-1)  # [B, 1, H, W]
    proj_abs = torch.abs(proj)

    w = torch.exp(-k * depth).unsqueeze(1)  # [B, 1, H, W]

    if mask is not None:
        mask = mask.float()
        w = w * mask

    loss = (proj_abs * w).sum() / (w.sum() + 1e-8)

    return loss

class UnderwaterLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.2, delta=0.1, proj=0.1, k=1.0):
        super().__init__()
        self.alpha = alpha       # 重建loss权重
        self.beta = beta         # 一致性loss权重
        self.gamma = gamma       # 平滑loss权重
        self.delta = delta       # 其它平滑loss权重（如梯度平滑）
        self.proj = proj         # 投影loss权重
        self.k = k               # 投影loss中的深度衰减系数
        self.l1_loss = nn.L1Loss(reduction='none')
        self.color_loss = ColorLoss()
        
    def forward(self, predictions, enhanced_images, beta_pred, A_pred, I_recon, depth_new, loss_mask):
        """
        Args:
            predictions: 原始预测字典，包含:
                'images': [B, T, C, H, W]
            enhanced_images: 增强后的图像 [B, T, C, H, W]
            I_recon: 重建后的图像 [B, T, C, H, W]
            loss_mask: 损失掩码 [B, T, 1, H, W] 或 [B, T, H, W]
        Returns:
            total_loss: 总损失值
            loss_dict: 各损失分量字典
        """
        B, T, C, H, W = enhanced_images.shape

        # 确保掩码形状正确 [B, T, H, W]
        if loss_mask.dim() == 5 and loss_mask.size(2) == 1:
            loss_mask = loss_mask.squeeze(2)
        loss_mask = loss_mask.float()

        # 1. 重建损失 (带掩码)
        recon_loss = 0
        proj_loss = 0
        for t in range(T):
            I_t = predictions['images'][:, t]     # [B, C, H, W]
            I_recon_t = I_recon[:, t]             # [B, C, H, W]
            mask_t = loss_mask[:, t].unsqueeze(1) # [B, 1, H, W]

            # 逐像素重建损失
            pixel_recon_loss = self.l1_loss(I_recon_t, I_t)  # [B, C, H, W]
            masked_recon_loss = pixel_recon_loss * mask_t

            valid_pixels = mask_t.sum() + 1e-8
            recon_loss += masked_recon_loss.sum() / valid_pixels

            depth_t = depth_new[:, t, :, :, 0]  # [B, H, W]
            proj_loss += depth_aware_projection_loss(
                enhanced_images[:, t], I_t, A_pred[:, t], depth_t, mask=mask_t, k=self.k
            )


        recon_loss /= T
        proj_loss /= T
            
        # 2. 平滑损失 (带掩码)
        smooth_loss = 0
        for t in range(T):
            J_t = enhanced_images[:, t]  # [B, C, H, W]
            d_t = depth_new[:, t, :, :, 0]  # [B, H, W]
            mask_t = loss_mask[:, t]  # [B, H, W]
            
            # 计算梯度
            grad_J_x = torch.abs(J_t[:, :, :, :-1] - J_t[:, :, :, 1:])  # [B, C, H, W-1]
            grad_J_y = torch.abs(J_t[:, :, :-1, :] - J_t[:, :, 1:, :])  # [B, C, H-1, W]
            
            grad_d_x = torch.abs(d_t[:, :, :-1] - d_t[:, :, 1:]).unsqueeze(1)  # [B, 1, H, W-1]
            grad_d_y = torch.abs(d_t[:, :-1, :] - d_t[:, 1:, :]).unsqueeze(1)  # [B, 1, H-1, W]
            
            # 深度感知平滑损失
            loss_x = grad_J_x / (1 + 10 * grad_d_x)
            loss_y = grad_J_y / (1 + 10 * grad_d_y)
            
            # 为梯度损失创建掩码 (缩小尺寸匹配梯度图)
            mask_x = mask_t[:, :, :-1]  # [B, H, W-1]
            mask_y = mask_t[:, :-1, :]  # [B, H-1, W]
            
            # 扩展掩码到通道维度
            mask_x = mask_x.unsqueeze(1)  # [B, 1, H, W-1]
            mask_y = mask_y.unsqueeze(1)  # [B, 1, H-1, W]
            
            # 应用掩码并计算平均损失
            valid_pixels_x = mask_x.sum() + 1e-8
            valid_pixels_y = mask_y.sum() + 1e-8
            
            masked_loss_x = (loss_x * mask_x).sum() / valid_pixels_x
            masked_loss_y = (loss_y * mask_y).sum() / valid_pixels_y
            
            smooth_loss += masked_loss_x + masked_loss_y
            
        smooth_loss /= T
        
        # 3. 多帧一致性损失 (带掩码)
        consistency_loss = 0
        num_pairs = 0
        
        # 对于每个场景中的连续帧对
        for t1 in range(T):
            for t2 in range(t1+1, T):
                # 计算增强图像之间的差异
                diff = torch.abs(enhanced_images[:, t1] - enhanced_images[:, t2])  # [B, C, H, W]
                
                # 使用深度图加权：深度相近区域权重更高
                depth_diff = torch.abs(
                    depth_new[:, t1, :, :, 0] - 
                    depth_new[:, t2, :, :, 0]
                ).unsqueeze(1)  # [B, 1, H, W]
                
                # 深度差异小的区域赋予更高权重
                weights = torch.exp(-5 * depth_diff)  # [B, 1, H, W]
                
                # 获取两帧的掩码并计算交集
                mask_t1 = loss_mask[:, t1].unsqueeze(1)  # [B, 1, H, W]
                mask_t2 = loss_mask[:, t2].unsqueeze(1)  # [B, 1, H, W]
                combined_mask = mask_t1 * mask_t2  # [B, 1, H, W]
                
                # 计算加权损失
                weighted_diff = diff * weights * combined_mask
                
                # 计算有效区域的平均损失
                valid_pixels = combined_mask.sum() + 1e-8
                consistency_loss += weighted_diff.sum() / valid_pixels
                num_pairs += 1
        
        if num_pairs > 0:
            consistency_loss /= num_pairs

        color_loss = self.color_loss(enhanced_images.squeeze(0)).mean()
            
        # 总损失
        total_loss = (
                     self.alpha * (recon_loss + proj_loss) +
                     self.beta * consistency_loss 
                     # 0.01 * color_loss
                     # self.gamma * smooth_loss
                     )
        
        # 返回损失分量用于记录
        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'consistency': consistency_loss.item() if num_pairs > 0 else 0,
            'smooth': smooth_loss.item(),
        }
        
        return total_loss, loss_dict


class UWConsistencyLoss(nn.Module):
    def __init__(self, weights=(1.0, 0.1, 0.1, 0.5, 1.0)):
        """
        水下模型输出一致性损失
        
        Args:
            weights: 各分量损失权重元组 (J_weight, beta_weight, A_weight, I_deg_weight)
        """
        super().__init__()
        self.weights = weights
        self.mse_loss = nn.MSELoss()
        
    def forward(self, output1, output2):
        """
        计算两组水下模型输出之间的一致性损失
        
        Args:
            output1: 第一组模型输出元组 (J1, beta1, A1, I_deg1)
            output2: 第二组模型输出元组 (J2, beta2, A2, I_deg2)
            
        Returns:
            加权后的总损失值
        """
        # 解包输出元组
        J1, beta1, A1, I_deg1, depth1 = output1
        J2, beta2, A2, I_deg2, depth2 = output2
        
        # 计算各分量的MSE损失
        loss_j = self.mse_loss(J1, J2)
        loss_beta = self.mse_loss(beta1, beta2)
        loss_A = self.mse_loss(A1, A2)
        loss_I_deg = self.mse_loss(I_deg1, I_deg2)
        loss_depth = self.mse_loss(depth1, depth2)
        
        # 应用权重
        w_j, w_beta, w_A, w_I_deg, w_depth = self.weights
        total_loss = (
            w_j * loss_j +
            w_beta * loss_beta +
            w_A * loss_A +
            w_I_deg * loss_I_deg + 
            w_depth * loss_depth

        )
        
        return total_loss

def get_valid_track_points(predictions, threshold=0.7):
    """
    根据 predictions 中 vis 和 conf，筛选出符合条件的像素点坐标
    Args:
        predictions: dict，包含 'track', 'vis', 'conf'
        threshold: float
    Returns:
        dict: {frame_idx: tensor of shape (N, 2)}  # N 是该帧符合条件的点数
    """
    track_points = predictions['track'][0]  # (num_frames, num_tracks, 2)
    vis = predictions['vis'][0]             # (num_frames, num_tracks)
    conf = predictions['conf'][0]           # (num_frames, num_tracks)

    num_frames, num_tracks, _ = track_points.shape

    valid_points_dict = {}

    for frame_idx in range(num_frames):
        mask = (vis[frame_idx] > threshold) & (conf[frame_idx] > threshold)  # (num_tracks)
        if mask.sum() == 0:
            valid_points_dict[frame_idx] = torch.empty((0, 2), device=track_points.device)
            continue

        points = track_points[frame_idx][mask]  # (N, 2)
        valid_points_dict[frame_idx] = points

    return valid_points_dict

class FirstFrameConsistencyLoss(nn.Module):
    def __init__(self, threshold=0.7, eps=1e-8):
        """
        基于第一帧的多帧颜色一致性损失
        
        Args:
            threshold: 轨迹点可见性和置信度阈值
            eps: 避免除零的小常数
        """
        super().__init__()
        self.threshold = threshold
        self.eps = eps
        
    def bilinear_sampling(self, image, points):
        """
        双线性采样图像上的点
        
        Args:
            image: 图像张量 [C, H, W]
            points: 点坐标 [N, 2] (x, y)
            
        Returns:
            采样颜色 [N, C]
        """
        # 归一化坐标到 [-1, 1]
        H, W = image.shape[-2:]
        x_norm = 2.0 * points[:, 0] / (W - 1) - 1.0
        y_norm = 2.0 * points[:, 1] / (H - 1) - 1.0
        
        # 创建采样网格 [N, 2] -> [1, 1, N, 2]
        grid = torch.stack([x_norm, y_norm], dim=1).unsqueeze(0).unsqueeze(0)
        
        # 扩展图像维度 [C, H, W] -> [1, C, H, W]
        image = image.unsqueeze(0)
        
        # 双线性采样 [1, C, 1, N] -> [N, C]
        sampled = F.grid_sample(image, grid, align_corners=True, mode='bilinear')
        return sampled.squeeze(0).squeeze(1).permute(1, 0)
    
    def forward(self, predictions, enhanced_images):
        """
        计算基于第一帧的多帧颜色一致性损失
        
        Args:
            predictions: 模型预测字典，包含:
                'track': 轨迹点 [B, T, N, 2]
                'vis': 可见性 [B, T, N]
                'conf': 置信度 [B, T, N]
            enhanced_images: 增强后的图像 [B, T, C, H, W]
            
        Returns:
            颜色一致性损失值
        """
        B, T, C, H, W = enhanced_images.shape
        total_loss = 0.0
        valid_pair_count = 0
        
        # 遍历batch中的每个样本
        for b in range(B):
            # 获取当前样本的预测
            track = predictions['track'][b]  # [T, N, 2]
            vis = predictions['vis'][b]      # [T, N]
            conf = predictions['conf'][b]    # [T, N]
            
            # 获取第一帧的有效点
            first_frame_mask = (vis[0] > self.threshold) & (conf[0] > self.threshold)
            first_frame_indices = torch.where(first_frame_mask)[0]
            
            # 如果没有有效点，跳过此样本
            if len(first_frame_indices) == 0:
                continue
                
            # 采样第一帧的颜色
            first_frame_points = track[0, first_frame_indices]  # [N_valid, 2]
            first_frame_colors = self.bilinear_sampling(
                enhanced_images[b, 0], first_frame_points
            )  # [N_valid, C]
            
            # 遍历后续帧 (t=1 到 T-1)
            for t in range(1, T):
                # 获取当前帧的对应点
                current_points = track[t, first_frame_indices]  # [N_valid, 2]
                
                # 检查可见性
                current_vis = vis[t, first_frame_indices] > self.threshold
                current_conf = conf[t, first_frame_indices] > self.threshold
                valid_mask = current_vis & current_conf
                
                # 如果没有有效点对，跳过此帧
                if valid_mask.sum() == 0:
                    continue
                    
                # 采样当前帧的颜色
                valid_indices = torch.where(valid_mask)[0]
                current_colors = self.bilinear_sampling(
                    enhanced_images[b, t], current_points[valid_indices]
                )  # [N_valid_current, C]
                
                # 获取对应的第一帧颜色
                corresponding_first_colors = first_frame_colors[valid_indices]
                
                # 计算颜色差异 (MSE)
                color_diff = torch.mean(
                    (current_colors - corresponding_first_colors) ** 2, 
                    dim=1
                )  # [N_valid_current]
                
                # 累加损失
                total_loss += torch.sum(color_diff)
                valid_pair_count += len(valid_indices)
        
        # 计算平均损失
        if valid_pair_count > 0:
            return total_loss / valid_pair_count
        else:
            return torch.tensor(0.0, device=enhanced_images.device)

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransmissionDepthConsistencyLoss(nn.Module):
    def __init__(self, eps=1e-8, beta_min=0.1, beta_max=1.0):
        """
        透射率-深度关系一致性损失
        
        Args:
            eps: 避免除零的小常数
            beta_min: 衰减系数的最小合理值
            beta_max: 衰减系数的最大合理值
        """
        super().__init__()
        self.eps = eps
        self.beta_min = beta_min
        self.beta_max = beta_max
        
    def forward(self, T, D):
        """
        计算透射率与深度之间的一致性损失
        
        Args:
            T: 透射率图 [B, 3, H, W] 或 [B, H, W, 3]
            D: 深度图 [B, 1, H, W] 或 [B, H, W, 1]
            
        Returns:
            一致性损失值
        """
        # 确保通道维度正确
        if T.dim() == 4 and T.size(1) == 3:  # [B, C, H, W]
            T = T.permute(0, 2, 3, 1)  # 转换为 [B, H, W, C]
        if D.dim() == 4 and D.size(1) == 1:  # [B, C, H, W]
            D = D.permute(0, 2, 3, 1)  # 转换为 [B, H, W, 1]
        
        B, H, W, C = T.shape
        total_loss = 0.0
        valid_count = 0

        beta_var_list = []
        
        # 对每个批次和每个颜色通道单独处理
        for b in range(B):
            for c in range(3):  # RGB三个通道
                T_c = T[b, :, :, c]  # [H, W]
                D_b = D[b, :, :, 0]  # [H, W]
                
                # 1. 确保物理约束：透射率应在(0,1]范围内
                # 惩罚小于0或大于1的值
                below_zero = torch.relu(-T_c)  # T < 0
                above_one = torch.relu(T_c - 1)  # T > 1
                range_loss = below_zero.mean() + above_one.mean()
                
                # 2. 深度-透射率关系约束
                # 计算负对数透射率：-log(T) ≈ beta * D
                neg_log_T = -torch.log(T_c.clamp(min=1e-5))  # [H, W]
                
                # 计算每个点的理想β范围
                beta_min_tensor = torch.ones_like(D) * self.beta_min
                beta_max_tensor = torch.ones_like(D) * self.beta_max
                
                # 计算每个点的β值
                beta_est = neg_log_T / (D_b + self.eps)
                beta_var_list.append(beta_est.var())


                # 计算每个深度值对应的平均-log(T)
                # 将深度离散化为多个区间
                num_bins = 20
                min_depth = D_b.min().item()
                max_depth = D_b.max().item()
                
                # 避免深度范围太小
                if max_depth - min_depth < self.eps:
                    continue
                
                # 创建深度区间
                bin_edges = torch.linspace(min_depth, max_depth, num_bins+1, device=T.device)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                bin_loss = 0.0
                bin_count = 0
                
                for i in range(num_bins):
                    # 获取当前深度区间内的像素
                    mask = (D_b >= bin_edges[i]) & (D_b < bin_edges[i+1])
                    
                    # 如果区间内没有像素，跳过
                    if mask.sum() == 0:
                        continue
                    
                    # 计算该深度区间内的平均 -log(T)
                    mean_neg_log_T = neg_log_T[mask].mean()
                    
                    # 计算该深度区间内的平均深度
                    mean_depth = D_b[mask].mean()
                    
                    # 计算期望的 -log(T) 范围
                    expected_min = self.beta_min * mean_depth
                    expected_max = self.beta_max * mean_depth
                    
                    # 惩罚超出合理范围的值
                    below_min = torch.relu(expected_min - mean_neg_log_T)
                    above_max = torch.relu(mean_neg_log_T - expected_max)
                    bin_loss += below_min + above_max
                    bin_count += 1
                
                if bin_count > 0:
                    bin_loss /= bin_count
                    total_loss += range_loss + bin_loss
                    valid_count += 1
        
        if valid_count > 0:
            return total_loss / valid_count, torch.stack(beta_var_list).mean()
        else:
            return torch.tensor(0.0, device=T.device), torch.stack(beta_var_list).mean()

class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = torch.pow(mr-0.5, 2)
        Dg = torch.pow(mg-0.5, 2)
        Db = torch.pow(mb-0.5, 2)
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Dg, 2) + torch.pow(Db, 2), 0.5)
        return k.mean()


class UIQM(nn.Module):
    """
    水下图像质量度量 (Underwater Image Quality Measure) 的 PyTorch 实现
    
    参考文献:
    Panetta, Karen, Chen Gao, and Sos Agaian. 
    "Human-visual-system-inspired underwater image quality measures." 
    IEEE Journal of Oceanic Engineering 41.3 (2015): 541-551.
    
    计算公式:
    UIQM = 0.0282 * UICM + 0.2953 * UISM + 3.5753 * UIConM
    """
    def __init__(self):
        super(UIQM, self).__init__()
        # 固定权重参数
        self.register_buffer('weights', torch.tensor([0.0282, 0.2953, 3.5753]))
        
    def forward(self, images):
        """
        计算一批图像的 UIQM 分数
        
        参数:
            images: 输入图像张量 [B, C, H, W]，值范围 [0, 1]，RGB 格式
        
        返回:
            uiqm: UIQM 分数 [B]
        """
        # 确保输入在 [0, 1] 范围内
        images = torch.clamp(images, 0, 1)
        
        # 计算三个分量
        uicm = self._calculate_uicm(images)
        uism = self._calculate_uism(images)
        uiconm = self._calculate_uiconm(images)
        
        # 加权求和
        components = torch.stack([uicm, uism, uiconm], dim=1)  # [B, 3]
        uiqm = torch.sum(components * self.weights, dim=1)  # [B]
        
        return uiqm
    
    def _calculate_uicm(self, images):
        """计算水下图像色彩度量 (Underwater Image Colorfulness Measure)"""
        # 转换到 YCbCr 色彩空间
        ycbcr_images = self._rgb_to_ycbcr(images)
        
        # 提取 Cb 和 Cr 通道
        cb = ycbcr_images[:, 1, :, :]
        cr = ycbcr_images[:, 2, :, :]
        
        # 计算均值
        cb_mean = torch.mean(cb, dim=[1, 2])
        cr_mean = torch.mean(cr, dim=[1, 2])
        
        # 计算标准差
        cb_std = torch.std(cb, dim=[1, 2])
        cr_std = torch.std(cr, dim=[1, 2])
        
        # 计算颜色分布
        cb_dist = torch.sqrt(torch.mean((cb - cb_mean[:, None, None])**2, dim=[1, 2]))
        cr_dist = torch.sqrt(torch.mean((cr - cr_mean[:, None, None])**2, dim=[1, 2]))
        
        # 计算 UICM
        uicm = -0.0268 * torch.sqrt(cb_dist**2 + cr_dist**2) + \
               0.1586 * torch.sqrt(cb_std**2 + cr_std**2)
        
        return uicm
    
    def _calculate_uism(self, images):
        """计算水下图像清晰度度量 (Underwater Image Sharpness Measure)"""
        # Sobel 算子 - 水平和垂直
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=images.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=images.device).view(1, 1, 3, 3)
        
        # 初始化清晰度分数
        uism = torch.zeros(images.size(0), device=images.device)
        
        # 对每个颜色通道分别计算
        for c in range(3):
            channel = images[:, c:c+1, :, :]  # [B, 1, H, W]
            
            # 应用 Sobel 算子
            gx = F.conv2d(channel, sobel_x, padding=1)
            gy = F.conv2d(channel, sobel_y, padding=1)
            
            # 计算梯度幅值
            edge_magnitude = torch.sqrt(gx**2 + gy**2)  # [B, 1, H, W]
            
            # 计算每个图像的清晰度分数
            channel_uism = torch.mean(edge_magnitude, dim=[1, 2, 3])
            
            # 加权求和 (红:0.299, 绿:0.587, 蓝:0.114)
            weight = torch.tensor([0.299, 0.587, 0.114], device=images.device)[c]
            uism += weight * channel_uism
        
        return uism
    
    def _calculate_uiconm(self, images):
        """计算水下图像对比度度量 (Underwater Image Contrast Measure)"""
        # 转换到灰度图
        gray_images = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]  # [B, H, W]
        
        # 计算局部对比度
        window_size = 3
        padding = window_size // 2
        
        # 使用平均池化计算局部均值
        local_mean = F.avg_pool2d(
            gray_images.unsqueeze(1), 
            kernel_size=window_size, 
            stride=1, 
            padding=padding
        ).squeeze(1)  # [B, H, W]
        
        # 计算局部方差
        local_var = F.avg_pool2d(
            (gray_images**2).unsqueeze(1), 
            kernel_size=window_size, 
            stride=1, 
            padding=padding
        ).squeeze(1) - local_mean**2  # [B, H, W]
        
        # 避免负值
        local_var = torch.clamp(local_var, min=0)
        
        # 计算对比度
        contrast = torch.sqrt(local_var) / (local_mean + 1e-6)  # [B, H, W]
        
        # 计算 UIConM
        uiconm = torch.mean(contrast, dim=[1, 2])
        
        return uiconm
    
    def _rgb_to_ycbcr(self, images):
        """将 RGB 图像转换为 YCbCr 色彩空间"""
        # 转换矩阵
        transform = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.1687, -0.3313, 0.5],
            [0.5, -0.4187, -0.0813]
        ], dtype=torch.float32, device=images.device)
        
        # 添加偏移
        offset = torch.tensor([0, 128, 128], dtype=torch.float32, device=images.device)
        
        # 重塑图像以便矩阵乘法
        B, C, H, W = images.shape
        images_flat = images.permute(0, 2, 3, 1).reshape(-1, 3)  # [B*H*W, 3]
        
        # 应用转换
        ycbcr_flat = torch.mm(images_flat, transform.t()) + offset
        
        # 重塑回图像
        ycbcr = ycbcr_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)
        
        # 归一化到 0-255 范围
        ycbcr = ycbcr / 255.0
        
        return ycbcr


def gradient(img):
    """
    计算图像 x 和 y 方向梯度
    img: [B, C, H, W]
    """
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return dx, dy

def gradient_consistency_loss(T, D, mask=None, reduction='mean'):
    """
    T: [B, 3, H, W, 3] RGB图
    D: [B, 3, H, W, 1] 单通道图
    mask: [B, 1, H, W] (可选)
    reduction: 'mean' or 'sum'
    """

    # reshape T 和 D 成 [B, C, H, W]
    B, C, H, W, _ = T.shape
    T = T.permute(0, 4, 1, 2, 3).reshape(B, -1, H, W)  # T: [B, 9, H, W]
    D = D.permute(0, 4, 1, 2, 3).reshape(B, -1, H, W)  # D: [B, 3, H, W]

    # 将 T 转灰度
    T_gray = T.mean(dim=1, keepdim=True)
    D_gray = D.mean(dim=1, keepdim=True)

    # 计算梯度
    T_dx, T_dy = gradient(T_gray)
    D_dx, D_dy = gradient(D_gray)

    # 计算梯度差异
    diff_x = torch.abs(T_dx - D_dx)
    diff_y = torch.abs(T_dy - D_dy)

    # mask区域
    if mask is not None:
        diff_x = diff_x * mask[:, :, :, 1:]
        diff_y = diff_y * mask[:, :, 1:, :]

    if reduction == 'mean':
        loss = diff_x.mean() + diff_y.mean()
    elif reduction == 'sum':
        loss = diff_x.sum() + diff_y.sum()
    else:
        raise ValueError("reduction should be 'mean' or 'sum'")

    return loss

# SSIM 工具函数 (for [B,N,C,H,W])
def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    mu1 = F.avg_pool2d(img1.view(-1, *img1.shape[2:]), 3, 1, 1)
    mu2 = F.avg_pool2d(img2.view(-1, *img2.shape[2:]), 3, 1, 1)
    sigma1 = F.avg_pool2d(img1.view(-1, *img1.shape[2:]) ** 2, 3, 1, 1) - mu1 ** 2
    sigma2 = F.avg_pool2d(img2.view(-1, *img2.shape[2:]) ** 2, 3, 1, 1) - mu2 ** 2
    sigma12 = F.avg_pool2d((img1 * img2).view(-1, *img1.shape[2:]), 3, 1, 1) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    return ssim_map.mean()

# Gradient 损失 (for [B,N,C,H,W])
def gradient(img):
    dx = img[..., 1:] - img[..., :-1]
    dy = img[..., 1:, :] - img[..., :-1, :]
    return dx, dy

def gradient_loss(pred, target):
    dx1, dy1 = gradient(pred)
    dx2, dy2 = gradient(target)
    return (F.l1_loss(dx1, dx2) + F.l1_loss(dy1, dy2))

# Perceptual Loss (for [B,N,C,H,W])
class PerceptualLoss(nn.Module):
    def __init__(self, layer_num=16):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:layer_num].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

    def forward(self, pred, target):
        # reshape to [B*N, C, H, W]
        B, N, C, H, W = pred.shape
        pred_flat = pred.view(B * N, C, H, W)
        target_flat = target.view(B * N, C, H, W)
        return F.l1_loss(self.vgg(pred_flat), self.vgg(target_flat))

# 整合感知一致性损失 (for [B,N,C,H,W])
class ConsistencyLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_ssim=1.0, lambda_percep=0.5, lambda_grad=0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_percep = lambda_percep
        self.lambda_grad = lambda_grad
        self.percep_loss = PerceptualLoss()

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        ssim_loss = 1 - ssim(pred, target)
        percep = self.percep_loss(pred, target)
        grad = gradient_loss(pred, target)

        total_loss = (self.lambda_l1 * l1 +
                      self.lambda_ssim * ssim_loss +
                      self.lambda_percep * percep +
                      self.lambda_grad * grad)
        return total_loss


def compute_losses_new(out, img, depth_init, mask_inf, lambda_consist=1.0, lambda_color=1.0, lambda_smooth=0.1):
    depth_final = out['depth']       # [B,1,H,W]
    delta_d     = out['delta_d']     # [B,1,H,W]
    delta_c     = out['delta_c']     # [B,3,H,W]
    beta        = out['beta']        # [B,1,H,W]

    B, _, H, W = img.shape

    # ΔD 和 ΔC 残差耦合一致性loss (L1)
    delta_d_expand = delta_d.repeat(1,3,1,1)  # 复制到3通道
    L_consist = F.l1_loss(delta_c * mask_inf, delta_d_expand * mask_inf)

    # 颜色残差物理一致性loss (L_color)
    C_bg = torch.mean(img * mask_inf, dim=(2,3), keepdim=True)  # 背景色估计 [B,3,1,1]
    D = depth_final

    beta_resized = F.interpolate(beta, size=D.shape[-2:], mode='bilinear', align_corners=False)
    exp_term = torch.exp(-beta_resized * D)
    C_hat = C_bg * exp_term + img * (1 - exp_term)
    L_color = F.l1_loss(C_hat * mask_inf, img * mask_inf)

    # ΔD 平滑正则化 (L_smooth)
    def gradient_x(img):  return img[:, :, :, :-1] - img[:, :, :, 1:]
    def gradient_y(img):  return img[:, :, :-1, :] - img[:, :, 1:, :]

    dx = gradient_x(delta_d)
    dy = gradient_y(delta_d)
    L_smooth = (dx.abs().mean() + dy.abs().mean())

    # 总loss
    total_loss = lambda_consist * L_consist + lambda_color * L_color + lambda_smooth * L_smooth

    loss_dict = {
        'total_loss': total_loss,
        'L_consist': L_consist.item(),
        'L_color': L_color.item(),
        'L_smooth': L_smooth.item()
    }
    return total_loss, loss_dict


def compute_inf_mask(depth, top_ratio=0.2):
    B, _, H, W = depth.shape
    mask = torch.zeros_like(depth)
    num_pixels = H * W
    num_inf = int(num_pixels * top_ratio)

    for b in range(B):
        d = depth[b,0].flatten()
        threshold = torch.topk(d, num_inf, largest=True)[0][-1]
        mask[b,0] = (depth[b,0] >= threshold).float()
    return mask

def compute_reconstruction_loss(images, out, A_tensor, idx):
    """
    根据物理模型 I = J*T + (1-T)*A 计算重建图像 I_rec，
    并计算 MSE(I_rec, images)
    
    images: [B,3,H,W] 原图
    out: dict，包含 delta_c, beta, depth
    mse_loss_fn: torch.nn.MSELoss()
    
    返回: loss_1 (标量), I_rec (重建图像)
    """
    delta_c = out['delta_c']           # [B,3,H,W]
    beta_map = out['beta']             # [B,1,H,W]
    depth_final = out['depth']         # [B,1,H,W]
    J_map = out['J_map']

    beta_resized = F.interpolate(beta_map, size=images.shape[-2:], mode='bilinear', align_corners=False)
    T_map = torch.exp(-beta_resized * depth_final)    # [B,1,H,W]
    # A_tensor = torch.stack([get_A(img.unsqueeze(0)).cuda() for img in images]).squeeze(1)

    # if idx == 200:
    #     pdb.set_trace()

    I_rec = J_map * T_map + (1.0 - T_map) * A_tensor

    loss_1 = mse_loss_fn(I_rec, images)

    return loss_1, I_rec, T_map

def mse_loss_fn(input, target, reduction='mean'):
    """
    计算 input 和 target 的均方误差 (MSE) 损失

    Args:
        input (Tensor): 预测值 tensor
        target (Tensor): 真实值 tensor
        reduction (str): 'mean' | 'sum' | 'none'

    Returns:
        Tensor: 损失值
    """
    criterion = nn.MSELoss(reduction=reduction)
    return criterion(input, target)

def compute_mixup_consistency_loss(images, depth_init, out, model):
    """
    计算 Mixup consistency loss，适配 model(images, depth_init) 接口

    Args:
        images (Tensor): 原始图像 [B, 3, H, W]
        depth_init (Tensor): 初始深度 [B, 1, H, W]
        j_out (Tensor): 增强后的图像 [B, 3, H, W]
        model (nn.Module): 当前网络
        mse_loss_fn (function): MSE 损失函数

    Returns:
        loss (Tensor): consistency 损失值
    """
    J_map = out['J_map']

    lam = np.random.beta(1, 1)
    input_mix = lam * images + (1 - lam) * J_map  # 混合图像

    # 注意：depth_init 不变，直接用原来的
    out_mix = model(input_mix, depth_init)

    loss_color = mse_loss_fn(out_mix['J_map'], J_map.detach())
    loss_depth = mse_loss_fn(out_mix['depth'], out['depth'].detach())

    loss = loss_color + loss_depth

    return loss

def build_photometric_loss(images, pred_depths, J_maps, intrinsics, extrinsics):
    """
    构建Photometric Loss
    
    参数:
        images: 输入图像序列, 形状 [B, T, 3, H, W]
        pred_depths: 预测的深度图, 形状 [B, T, 1, H, W]
        J_maps: 恢复后的清晰图像, 形状 [B, T, 3, H, W]
        intrinsics: 相机内参, 形状 [B, T, 3, 3]
        extrinsics: 相机外参, 形状 [B, T, 3, 4]
    
    返回:
        loss: Photometric Loss
    """
    B, T, _, H, W = images.shape
    device = images.device
    
    # 选择参考帧 (第0帧)
    ref_idx = 0
    src_idxs = [i for i in range(T) if i != ref_idx]
    
    total_loss = 0.0
    valid_pairs = 0
    
    # 1. 将外参转换为4x4齐次矩阵
    extrinsics_4x4 = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
    extrinsics_4x4[:, :, :3, :] = extrinsics
    
    for src_idx in src_idxs:
        # 2. 计算从参考帧到源帧的变换矩阵: T_ref_to_src = T_src * inv(T_ref)
        T_ref = extrinsics_4x4[:, ref_idx]  # [B, 4, 4]
        T_src = extrinsics_4x4[:, src_idx]  # [B, 4, 4]
        
        # 计算T_ref的逆
        R_ref = T_ref[:, :3, :3]
        t_ref = T_ref[:, :3, 3]
        R_ref_inv = R_ref.transpose(1, 2)  # 旋转矩阵的逆是转置
        t_ref_inv = -torch.bmm(R_ref_inv, t_ref.unsqueeze(2)).squeeze(2)
        
        T_ref_inv = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        T_ref_inv[:, :3, :3] = R_ref_inv
        T_ref_inv[:, :3, 3] = t_ref_inv
        
        # 计算变换矩阵: T_ref_to_src = T_src * T_ref_inv
        T_ref_to_src = torch.bmm(T_src, T_ref_inv)  # [B, 4, 4]
        
        # 3. 获取相机内参
        K_ref = intrinsics[:, ref_idx]  # [B, 3, 3]
        K_src = intrinsics[:, src_idx]  # [B, 3, 3]
        
        # 4. 获取参考帧的深度图和J_map
        depth_ref = pred_depths[:, ref_idx]  # [B, 1, H, W]
        J_ref = J_maps[:, ref_idx]  # [B, 3, H, W]
        
        # 5. 生成参考帧的像素坐标网格 (归一化到[-1, 1])
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        ones = torch.ones_like(x_coords)
        grid_homo = torch.stack([x_coords, y_coords, ones], dim=0)  # [3, H, W]
        grid_homo = grid_homo.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 3, H, W]
        
        # 6. 将像素坐标转换为相机坐标 (参考帧坐标系)
        # 注意: 这里使用逆内参变换
        K_ref_inv = torch.inverse(K_ref)
        cam_coords_ref = torch.bmm(
            K_ref_inv, 
            grid_homo.view(B, 3, -1)
        )  # [B, 3, H*W]
        
        # 乘以深度值
        cam_coords_ref = cam_coords_ref.view(B, 3, H, W)
        cam_coords_ref = cam_coords_ref * depth_ref  # [B, 3, H, W]
        
        # 7. 将3D点变换到源帧坐标系
        # 转换为齐次坐标 [B, 4, H, W]
        cam_coords_ref_homo = torch.cat([
            cam_coords_ref, 
            torch.ones(B, 1, H, W, device=device)
        ], dim=1)
        
        # 应用变换: [B, 4, H*W]
        cam_coords_src_homo = torch.bmm(
            T_ref_to_src, 
            cam_coords_ref_homo.view(B, 4, -1)
        )  # [B, 4, H*W]
        
        # 8. 将3D点投影到源帧图像平面
        xyz_src = cam_coords_src_homo[:, :3]  # [B, 3, H*W]
        z = xyz_src[:, 2:3]  # [B, 1, H*W]
        
        # 避免除以零 (添加小量)
        z = torch.where(z.abs() < 1e-7, torch.ones_like(z) * 1e-7, z)
        
        # 归一化坐标
        xy_normalized = xyz_src[:, :2] / z  # [B, 2, H*W]
        
        # 应用源帧内参
        pixel_coords_src = torch.bmm(
            K_src, 
            torch.cat([xy_normalized, torch.ones(B, 1, H*W, device=device)], dim=1)
        )  # [B, 3, H*W]
        
        # 归一化到[-1, 1] (用于grid_sample)
        u = pixel_coords_src[:, 0] / (W - 1) * 2 - 1  # 归一化到[-1, 1]
        v = pixel_coords_src[:, 1] / (H - 1) * 2 - 1  # 归一化到[-1, 1]
        grid_src = torch.stack([u, v], dim=2).view(B, H, W, 2)  # [B, H, W, 2]
        
        # 9. 采样源帧的J_map
        J_src_actual = J_maps[:, src_idx]  # 真实的源帧图像 [B, 3, H, W]
        J_src_projected = F.grid_sample(
            J_ref, 
            grid_src, 
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # [B, 3, H, W]
        
        # 10. 计算Photometric Loss (L1 + SSIM)
        # L1 Loss
        l1_loss = F.l1_loss(J_src_projected, J_src_actual, reduction='none')
        
        # SSIM Loss (简化版)
        ssim_loss = 1.0 - ssim(J_src_projected, J_src_actual)
        
        # 组合损失
        photometric_loss = 0.85 * ssim_loss + 0.15 * l1_loss.mean([1, 2, 3])
        
        # 11. 添加深度有效性掩码 (忽略深度无效区域)
        valid_mask = (z.view(B, 1, H, W) > 0.1) & (z.view(B, 1, H, W) < 10.0)
        photometric_loss = photometric_loss * valid_mask.float()
        
        # 平均损失
        total_loss += photometric_loss.mean()
        valid_pairs += 1
    
    # 计算平均损失
    if valid_pairs > 0:
        total_loss /= valid_pairs
    else:
        total_loss = torch.tensor(0.0, device=device)
    
    return total_loss


def build_photometric_loss_pairwise(images, pred_depths, J_maps, intrinsics, extrinsics):
    """
    构建Photometric Loss (两两帧之间都计算)
    
    参数:
        images: [B, T, 3, H, W]
        pred_depths: [B, T, 1, H, W]
        J_maps: [B, T, 3, H, W]
        intrinsics: [B, T, 3, 3]
        extrinsics: [B, T, 3, 4]
    
    返回:
        loss: Photometric Loss
    """
    import torch.nn.functional as F
    import torch

    B, T, _, H, W = images.shape
    device = images.device
    
    total_loss = 0.0
    valid_pairs = 0
    
    extrinsics_4x4 = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
    extrinsics_4x4[:, :, :3, :] = extrinsics

    # 像素网格
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    ones = torch.ones_like(x_coords)
    grid_homo = torch.stack([x_coords, y_coords, ones], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 3, H, W]

    for ref_idx in range(T):
        for src_idx in range(T):
            if ref_idx == src_idx:
                continue

            # T_ref 和 T_src
            T_ref = extrinsics_4x4[:, ref_idx]  # [B, 4, 4]
            T_src = extrinsics_4x4[:, src_idx]  # [B, 4, 4]

            # T_ref_inv
            R_ref = T_ref[:, :3, :3]
            t_ref = T_ref[:, :3, 3]
            R_ref_inv = R_ref.transpose(1, 2)
            t_ref_inv = -torch.bmm(R_ref_inv, t_ref.unsqueeze(2)).squeeze(2)
            T_ref_inv = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
            T_ref_inv[:, :3, :3] = R_ref_inv
            T_ref_inv[:, :3, 3] = t_ref_inv

            # T_ref_to_src = T_src * T_ref_inv
            T_ref_to_src = torch.bmm(T_src, T_ref_inv)

            K_ref = intrinsics[:, ref_idx]
            K_src = intrinsics[:, src_idx]

            depth_ref = pred_depths[:, ref_idx]  # [B, 1, H, W]
            J_ref = J_maps[:, ref_idx]
            J_src_actual = J_maps[:, src_idx]

            # 计算相机坐标
            K_ref_inv = torch.inverse(K_ref)
            cam_coords_ref = torch.bmm(
                K_ref_inv, grid_homo.view(B, 3, -1)
            ).view(B, 3, H, W)
            cam_coords_ref = cam_coords_ref * depth_ref  # [B, 3, H, W]

            # 齐次坐标
            cam_coords_ref_homo = torch.cat([
                cam_coords_ref,
                torch.ones(B, 1, H, W, device=device)
            ], dim=1)

            # 变换到源帧
            cam_coords_src_homo = torch.bmm(
                T_ref_to_src, 
                cam_coords_ref_homo.view(B, 4, -1)
            )

            xyz_src = cam_coords_src_homo[:, :3]
            z = xyz_src[:, 2:3]
            z = torch.where(z.abs() < 1e-7, torch.ones_like(z) * 1e-7, z)
            xy_normalized = xyz_src[:, :2] / z

            pixel_coords_src = torch.bmm(
                K_src, 
                torch.cat([xy_normalized, torch.ones(B, 1, H*W, device=device)], dim=1)
            )
            u = pixel_coords_src[:, 0] / (W - 1) * 2 - 1
            v = pixel_coords_src[:, 1] / (H - 1) * 2 - 1
            grid_src = torch.stack([u, v], dim=2).view(B, H, W, 2)

            # 采样源帧J_map
            J_src_projected = F.grid_sample(
                J_ref, grid_src, mode='bilinear', padding_mode='zeros', align_corners=False
            )

            # photometric loss
            l1_loss = F.l1_loss(J_src_projected, J_src_actual, reduction='none')
            ssim_loss_val = 1.0 - ssim(J_src_projected, J_src_actual)

            photometric_loss = 0.85 * ssim_loss_val + 0.15 * l1_loss.mean([1, 2, 3])

            valid_mask = (z.view(B, 1, H, W) > 0.1) & (z.view(B, 1, H, W) < 10.0)
            photometric_loss = photometric_loss * valid_mask.float().mean([1, 2, 3])

            total_loss += photometric_loss.mean()
            valid_pairs += 1

    if valid_pairs > 0:
        total_loss /= valid_pairs
    else:
        total_loss = torch.tensor(0.0, device=device)

    return total_loss

# def ssim(img1, img2, window_size=11, size_average=True):
#     """
#     计算SSIM (结构相似性)
#     简化实现，实际中应使用高斯加权
#     """
#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2
    
#     mu1 = F.avg_pool2d(img1, window_size, 1, 0)
#     mu2 = F.avg_pool2d(img2, window_size, 1, 0)
    
#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2
    
#     sigma1_sq = F.avg_pool2d(img1 * img1, window_size, 1, 0) - mu1_sq
#     sigma2_sq = F.avg_pool2d(img2 * img2, window_size, 1, 0) - mu2_sq
#     sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, 0) - mu1_mu2
    
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
#                ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

#     return ssim_map.mean([1, 2, 3]) if size_average else ssim_map