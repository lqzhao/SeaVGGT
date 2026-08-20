import os
import pdb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F

from vggt.models.vggt import VGGT
from vggt.heads.dpt_head import DPTHead
from uw_model_new_new import TokenPrototypeModulator, LightAEstimator, TokenPrototypeModulatorGAT
from dataset import SingleImageDataset, SingleImageDepthDataset, RandomSequencePathDataset, TwoimageDepthDataset_SQUID1, MultiImageDepthDatasetV2

from visual_util import (save_multiframe_colored_pointcloud, estimate_beta_A_multiframe_rgb_tensor, 
                        compute_J_from_d_v2, get_A, estimate_A, estimate_A_darkchannel, compute_A_gt, 
                        scale_align, compute_depth_metrics, compute_si_rmse)
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

def save_results_to_file(D_init_metrics_list, Depth_metrics_list, filename, img_paths_all, img_identifiers=None):
    if img_identifiers is None or len(img_identifiers) != len(D_init_metrics_list):
        img_identifiers = [f"Index_{i}" for i in range(len(D_init_metrics_list))]
    def summarize(metrics_list):
        return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}

    D_init_summary = summarize(D_init_metrics_list)
    Depth_summary  = summarize(Depth_metrics_list)

    lines = []

    # 表格输出
    lines.append("\n================= 📊 Depth Estimation Results =================")
    header = f"{'Metric':<10} | {'VGGT D_init':>12} | {'Ours':>12}"
    lines.append(header)
    lines.append("-"*len(header))
    for k in D_init_summary.keys():
        lines.append(f"{k:<10} | {D_init_summary[k]:>12.4f} | {Depth_summary[k]:>12.4f}")
    lines.append("="*len(header))

    # 计算 RMSE 差值
    rmse_gains = []
    for i in range(len(D_init_metrics_list)):
        diff = D_init_metrics_list[i]['RMSE'] - Depth_metrics_list[i]['RMSE']
        img_path = img_paths_all[i] if i < len(img_paths_all) else f"Index_{i}"
        rmse_gains.append((i, diff, img_path))

    # 排序
    rmse_gains_sorted = sorted(rmse_gains, key=lambda x: x[1], reverse=True)
    # top10 = rmse_gains_sorted[:10]

    # # 提升最大的10个
    # lines.append("\n🏆 提升最大的10个图片:")
    # for idx, diff in top10:
    #     lines.append(f"图片 {idx}: RMSE 提升 {diff:.4f}")

    # top20 = rmse_gains_sorted[:20]
    # lines.append("\n🏆 提升最大的20个图片:")
    # for idx, diff in top20:
    #     identifier = img_identifiers[idx]
    #     lines.append(f"图片 {idx:03d}: RMSE 提升 {diff:.4f} - {identifier}")
    #     print(f"图片 {idx:03d}: RMSE 提升 {diff:.4f} - {identifier}")  # ✅ 打印出来

    top20 = sorted(rmse_gains, key=lambda x: x[1], reverse=True)[:20]
    print("\n🏆 提升最大的20个图片:")
    for idx, diff, img_path in top20:
        print(f"图片 {idx:03d}: RMSE 提升 {diff:.4f} - 路径: {img_path}")

    # 写入文件
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    print(f"\n📄 结果已写入 {filename}")


    return D_init_summary, Depth_summary

def average_metrics(metrics_list):
    # metrics_list 是若干个 dict，结构相同
    avg_metrics = {}
    num_samples = len(metrics_list)
    if num_samples == 0:
        return avg_metrics  # 防止空

    keys = metrics_list[0].keys()
    for key in keys:
        avg_metrics[key] = sum([m[key] for m in metrics_list]) / num_samples

    return avg_metrics

def save_depth_comparison_with_image(img_tensor, D_init, Depth_pred, gt_depth, save_path, vmin=None, vmax=None, cmap='plasma'):
    """
    将 原图 / D_init / ours / gt 四张图拼接可视化，保存到 save_path
    """
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # [H,W,3]
    img_np = np.clip(img_np, 0, 1)

    D_init_np    = D_init.detach().cpu().numpy()
    Depth_pred_np= Depth_pred.detach().cpu().numpy()
    gt_depth_np  = gt_depth.detach().cpu().numpy()

    if vmin is None:
        # vmin = min(D_init_np.min(), Depth_pred_np.min(), gt_depth_np.min())
        # vmin = gt_depth_np.min()
        vmin = np.mean([D_init_np.min(), Depth_pred_np.min(), gt_depth_np.min()])
    if vmax is None:
        # vmax = max(D_init_np.max(), Depth_pred_np.max(), gt_depth_np.max())
        # vmax = gt_depth_np.max()
        vmax = np.mean([D_init_np.max(), Depth_pred_np.max(), gt_depth_np.max()])

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].imshow(D_init_np, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("VGGT D_init")
    axes[1].axis('off')

    axes[2].imshow(Depth_pred_np, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title("Ours")
    axes[2].axis('off')

    axes[3].imshow(gt_depth_np, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[3].set_title("GT Depth")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def get_save_path_flsea(img_path, method):
    """
    从原始图片路径生成保存路径：
    /home/zlq/code/uw/seavggt_results/FLSea/{place}/{scene}/{img_name}.npy
    """
    # 解析 place / scene / img_name
    parts = os.path.dirname(img_path).split(os.sep)
    place = parts[-3]       # e.g., red_sea
    scene = parts[-2]       # e.g., big_dice_loop
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    # 拼路径
    save_path = f"/home/zlq/code/uw/seavggt_results/FLSea/{method}/{place}/{scene}/{img_name}.npy"
    return save_path



def validate_depth_model(model_ckpt_path, test_set, test_set1=None, num_images=2, save_dir="output_depth_eval"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_identifiers = []  # 新增，用于记录图像路径或索引信息

    if test_set=='squid':
        rgb_dir = '/home/zlq/code/uw/vggt/results_collection/rgb/squid'
        depth_dir = '/home/zlq/code/uw/vggt/results_collection/gt/squid'
        if num_images==2:
            test_dataset = TwoimageDepthDataset_SQUID1(rgb_dir, depth_dir)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
        if num_images==1:
            test_dataset = TwoimageDepthDataset_SQUID1(rgb_dir, depth_dir, num_images=num_images)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)            

    elif test_set=='flsea_1':
        rgb_dir = '/home/zlq/code/uw/vggt/results_collection/rgb/flsea_1'
        depth_dir = '/home/zlq/code/uw/vggt/results_collection/gt/flsea_1'

        test_dataset = MultiImageDepthDatasetV2(rgb_dir, depth_dir, num_images=num_images)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    elif test_set=='flsea_2':
        rgb_dir = '/home/zlq/code/uw/vggt/results_collection/rgb/flsea_2'
        depth_dir = '/home/zlq/code/uw/vggt/results_collection/gt/flsea_2'

        test_dataset = MultiImageDepthDatasetV2(rgb_dir, depth_dir, num_images=num_images)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    else:
        if test_set1 is not None: 
            rgb_dir = '/data1/data/wuhao/underwater_dataset/FLsea/{}/{}/imgs'.format(test_set, test_set1)
            depth_dir = '/data1/data/wuhao/underwater_dataset/FLsea/{}/{}/depth'.format(test_set, test_set1)

        test_dataset = MultiImageDepthDatasetV2(rgb_dir, depth_dir, num_images=num_images, endswith='.tif')
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)


    print("Loading model...")
    model = VGGT().to(device)
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    modulator = TokenPrototypeModulatorGAT(token_dim=2048, num_prototypes=24).to(device)
    A_net = LightAEstimator().to(device)

    checkpoint = torch.load(model_ckpt_path)
    modulator.load_state_dict(checkpoint['modulator_state_dict'], strict=False)
    A_net.load_state_dict(checkpoint['A_net_state_dict'])
    print("Model weights loaded.")

    modulator.eval()
    A_net.eval()

    os.makedirs(save_dir, exist_ok=True)

    D_init_metrics_list, Depth_metrics_list = [], []
    predictions={}

    img_paths_all = []  # 新增，用于收集真实图像路径

    with torch.no_grad():
        for idx, (images, gt_depths, img_paths) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            for img_path in img_paths:
                img_paths_all.append(img_path)

            images = images.to(device).squeeze(0)  # [2, 3, H, W]
            gt_depths = gt_depths.to(device)

            A_pred = A_net(images)  # [2, 3]
            tokens, patch_start_idx = model.aggregator(images.unsqueeze(0))  # tokens: [1, 2, 25681, 2048]

            # pose_enc_list = model.camera_head(tokens)
            # predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
            # extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            # predictions["extrinsic"] = extrinsic
            # predictions["intrinsic"] = intrinsic
            # pdb.set_trace()
            
            # 提取图像识别信息（尽可能用路径名，否则就用 index）
            if isinstance(img_path, list) and len(img_path) > 0:
                identifier = os.path.basename(img_path[0])  # 可改成其他方式
            elif isinstance(img_path, str):
                identifier = os.path.basename(img_path)
            else:
                identifier = f"Index_{idx}"
            img_identifiers.append(identifier)


            D_init, _ = model.depth_head(tokens, images=images.unsqueeze(0), patch_start_idx=patch_start_idx)
            D_init_pred = D_init[0]  # [2, H, W, 1]


            ########################
            # 初始化：images.shape[0]=2，创建 2 个空 list
            mod_tokens_list = [[] for _ in range(images.shape[0])]  

            # 先把 tokens 拆成每帧对应的 list
            split_tokens_per_frame = [[] for _ in range(images.shape[0])]  # [[...], [...]]
            for token in tokens:  # 遍历24个 token
                for i in range(images.shape[0]):  # 2帧
                    token_i = token[:, i:i+1]  # [1, 1, 25681, 2048]
                    split_tokens_per_frame[i].append(token_i)

            # 每帧的 24 个 token list 和 A_pred_i 送 modulator
            for i in range(images.shape[0]):
                A_pred_i = A_pred[i:i+1]  # [1, 3]
                mod_tokens, _, _, _ = modulator(split_tokens_per_frame[i], A_pred_i)  # mod_tokens是list，长度24
                mod_tokens_list[i] = mod_tokens

            # 每帧拼成 list → tensor，最后再拼成 tokens
            mod_tokens_per_frame = [torch.cat(mod_tokens_list[i], dim=0) for i in range(images.shape[0])]
            # 每个 mod_tokens_per_frame[i]: [24, 1, 25681, 2048]

            # 按原token排列方式重新拼成 final_mod_tokens
            final_mod_tokens = []
            for j in range(len(tokens)):
                mod_token_j = torch.cat([mod_tokens_per_frame[i][j:j+1] for i in range(images.shape[0])], dim=1)
                # shape: [1, 2, 25681, 2048]
                final_mod_tokens.append(mod_token_j)
            ########################

            # 再过 depth head
            Depth_pred, _ = model.depth_head(final_mod_tokens, images=images.unsqueeze(0), patch_start_idx=patch_start_idx)
            Depth_pred = Depth_pred[0]  # [2, H, W, 1]


            if test_set=='squid':
                # 单帧指标计算
                for i in range(images.shape[0]):
                    D_init_pred_i = D_init_pred[i, :, :, 0]
                    Depth_pred_i  = Depth_pred[i, :, :, 0]

                    # 取出GT深度图，形状 [H_gt, W_gt]
                    gt_depth_i = gt_depths[0, i, 0]  # [H_gt, W_gt]

                    # 获取目标尺寸
                    target_size = gt_depth_i.shape  # (H_gt, W_gt)

                    # 插值上采样到 GT 分辨率
                    D_init_pred_i = F.interpolate(D_init_pred_i.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze()
                    Depth_pred_i  = F.interpolate(Depth_pred_i.unsqueeze(0).unsqueeze(0),  size=target_size, mode='bilinear', align_corners=False).squeeze()

                    # scale align
                    D_init_aligned = scale_align(D_init_pred_i, gt_depth_i)
                    Ours_aligned   = scale_align(Depth_pred_i, gt_depth_i)

                    # 指标
                    D_init_metrics = compute_depth_metrics(D_init_aligned, gt_depth_i)
                    Depth_metrics  = compute_depth_metrics(Ours_aligned, gt_depth_i)

                    # si-RMSE
                    D_init_metrics['si-RMSE'] = compute_si_rmse(D_init_pred_i, gt_depth_i)
                    Depth_metrics['si-RMSE']  = compute_si_rmse(Depth_pred_i, gt_depth_i)

                    D_init_metrics_list.append(D_init_metrics)
                    Depth_metrics_list.append(Depth_metrics)
            else:
                D_init_pred_i = D_init_pred[0, :, :, 0]
                Depth_pred_i  = Depth_pred[0, :, :, 0]

                # 取出GT深度图，形状 [H_gt, W_gt]
                gt_depth_i = gt_depths[0, 0, 0]  # [H_gt, W_gt]

                # 获取目标尺寸
                target_size = gt_depth_i.shape  # (H_gt, W_gt)

                # 插值上采样到 GT 分辨率
                D_init_pred_i = F.interpolate(D_init_pred_i.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze()
                Depth_pred_i  = F.interpolate(Depth_pred_i.unsqueeze(0).unsqueeze(0),  size=target_size, mode='bilinear', align_corners=False).squeeze()

                # scale align
                D_init_aligned = scale_align(D_init_pred_i, gt_depth_i)
                Ours_aligned   = scale_align(Depth_pred_i, gt_depth_i)

                # 指标
                D_init_metrics = compute_depth_metrics(D_init_aligned, gt_depth_i)
                if D_init_metrics is None:
                    print(f"样本 {i} 无有效像素，跳过")
                    continue  # 或者 return/skip，不累加指标
                Depth_metrics  = compute_depth_metrics(Ours_aligned, gt_depth_i)

                # si-RMSE
                D_init_metrics['si-RMSE'] = compute_si_rmse(D_init_pred_i, gt_depth_i)
                Depth_metrics['si-RMSE']  = compute_si_rmse(Depth_pred_i, gt_depth_i)

                D_init_metrics_list.append(D_init_metrics)
                Depth_metrics_list.append(Depth_metrics)                

            # pdb.set_trace()


            # # 保存可视化
            # save_path = os.path.join(save_dir, f"iter_{idx}.png")
            # save_depth_comparison_with_image(images[0], D_init_aligned, Ours_aligned, gt_depth, save_path)
            
            # outdir = '/home/zlq/code/uw/vggt/results_collection/ours/usod10k'
            # save_path = os.path.join(outdir, os.path.splitext(os.path.basename(paths[0]))[0] + ".npy")
            # np.save(save_path, Depth_pred.cpu().numpy())

            # outdir = '/home/zlq/code/uw/vggt/results_collection/vggt/flsea'
            # save_path = os.path.join(outdir, os.path.splitext(os.path.basename(paths[0]))[0] + ".npy")
            # np.save(save_path, D_init_pred.cpu().numpy())

    if test_set1 is not None:
        save_results_to_file(
            D_init_metrics_list, 
            Depth_metrics_list, 
            f'results_summary_{test_set}_{test_set1}_numimage{num_images}.txt',
            img_paths_all,
            img_identifiers
        )
    else:
        save_results_to_file(
            D_init_metrics_list, 
            Depth_metrics_list, 
            f'results_summary_{test_set}_numimage{num_images}.txt',
            img_paths_all,
            img_identifiers
        )


# ✅ 调用验证
validate_depth_model(
    # model_ckpt_path="checkpoints/epoch_01_best_rmse_0.6050.pth", # epoch_01_best_rmse_0.6133_24proto_sota
    # model_ckpt_path="checkpoints/epoch_01_best_rmse_0.6133_24proto_sota.pth",
    # model_ckpt_path="checkpoints/epoch_01_best_rmse_0.6025.pth", # sota_best
    # model_ckpt_path='checkpoints/epoch_01_best_rmse_0.6156.pth',
    model_ckpt_path = '/home/zlq/code/uw/vggt/checkpoints/epoch_01_best_rmse_0.4983.pth',
    # test_set='squid',
    # test_set='flsea_1', num_images=1, 
    # test_set='canyons', test_set1='tiny_canyon', num_images=1,
    test_set='red_sea', test_set1='cross_pyramid_loop', num_images=1,
    # test_set='flsea_2', num_images=1, save_dir="output_depth_eval_flsea",
    # test_set='squid', num_images=1, save_dir="output_depth_eval_squid"
)
