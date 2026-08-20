# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import trimesh
import gradio as gr
import numpy as np
import matplotlib
from scipy.spatial.transform import Rotation
import copy
import cv2
import os
import pdb
import requests
import torch
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from PIL import Image, ImageFilter
from torchvision.transforms import ToTensor
import torch.nn.functional as F

def predictions_to_glb(
    predictions,
    conf_thres=50.0,
    filter_by_frames="all",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    target_dir=None,
    prediction_mode="Predicted Pointmap",
) -> trimesh.Scene:
    """
    Converts VGGT predictions to a 3D scene represented as a GLB file.

    Args:
        predictions (dict): Dictionary containing model predictions with keys:
            - world_points: 3D point coordinates (S, H, W, 3)
            - world_points_conf: Confidence scores (S, H, W)
            - images: Input images (S, H, W, 3)
            - extrinsic: Camera extrinsic matrices (S, 3, 4)
        conf_thres (float): Percentage of low-confidence points to filter out (default: 50.0)
        filter_by_frames (str): Frame filter specification (default: "all")
        mask_black_bg (bool): Mask out black background pixels (default: False)
        mask_white_bg (bool): Mask out white background pixels (default: False)
        show_cam (bool): Include camera visualization (default: True)
        mask_sky (bool): Apply sky segmentation mask (default: False)
        target_dir (str): Output directory for intermediate files (default: None)
        prediction_mode (str): Prediction mode selector (default: "Predicted Pointmap")

    Returns:
        trimesh.Scene: Processed 3D scene containing point cloud and cameras

    Raises:
        ValueError: If input predictions structure is invalid
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if conf_thres is None:
        conf_thres = 10.0

    print("Building GLB scene")
    selected_frame_idx = None
    if filter_by_frames != "all" and filter_by_frames != "All":
        try:
            # Extract the index part before the colon
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    if "Pointmap" in prediction_mode:
        print("Using Pointmap Branch")
        if "world_points" in predictions:
            pred_world_points = predictions["world_points"]  # No batch dimension to remove
            pred_world_points_conf = predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
        else:
            print("Warning: world_points not found in predictions, falling back to depth-based points")
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
    else:
        print("Using Depthmap and Camera Branch")
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    # Get images from predictions
    images = predictions["images"]
    # Use extrinsic matrices instead of pred_extrinsic_list
    camera_matrices = predictions["extrinsic"]

    if mask_sky:
        if target_dir is not None:
            import onnxruntime

            skyseg_session = None
            target_dir_images = target_dir + "/images"
            image_list = sorted(os.listdir(target_dir_images))
            sky_mask_list = []

            # Get the shape of pred_world_points_conf to match
            S, H, W = (
                pred_world_points_conf.shape
                if hasattr(pred_world_points_conf, "shape")
                else (len(images), images.shape[1], images.shape[2])
            )

            # Download skyseg.onnx if it doesn't exist
            if not os.path.exists("skyseg.onnx"):
                print("Downloading skyseg.onnx...")
                download_file_from_url(
                    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx"
                )

            for i, image_name in enumerate(image_list):
                image_filepath = os.path.join(target_dir_images, image_name)
                mask_filepath = os.path.join(target_dir, "sky_masks", image_name)

                # Check if mask already exists
                if os.path.exists(mask_filepath):
                    # Load existing mask
                    sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
                else:
                    # Generate new mask
                    if skyseg_session is None:
                        skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
                    sky_mask = segment_sky(image_filepath, skyseg_session, mask_filepath)

                # Resize mask to match H×W if needed
                if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
                    sky_mask = cv2.resize(sky_mask, (W, H))

                sky_mask_list.append(sky_mask)

            # Convert list to numpy array with shape S×H×W
            sky_mask_array = np.array(sky_mask_list)

            # Apply sky mask to confidence scores
            sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
            pred_world_points_conf = pred_world_points_conf * sky_mask_binary

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_matrices = camera_matrices[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)
    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    conf = pred_world_points_conf.reshape(-1)
    # Convert percentage threshold to actual confidence value
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = np.percentile(conf, conf_thres)

    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    if mask_black_bg:
        black_bg_mask = colors_rgb.sum(axis=1) >= 16
        conf_mask = conf_mask & black_bg_mask

    if mask_white_bg:
        # Filter out white background pixels (RGB values close to white)
        # Consider pixels white if all RGB values are above 240
        white_bg_mask = ~((colors_rgb[:, 0] > 240) & (colors_rgb[:, 1] > 240) & (colors_rgb[:, 2] > 240))
        conf_mask = conf_mask & white_bg_mask

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        # Calculate the 5th and 95th percentiles along each axis
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)

        # Calculate the diagonal length of the percentile bounding box
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)

    scene_3d.add_geometry(point_cloud_data)

    # Prepare 4x4 matrices for camera extrinsics
    num_cameras = len(camera_matrices)
    extrinsics_matrices = np.zeros((num_cameras, 4, 4))
    extrinsics_matrices[:, :3, :4] = camera_matrices
    extrinsics_matrices[:, 3, 3] = 1

    if show_cam:
        # Add camera models to the scene
        for i in range(num_cameras):
            world_to_camera = extrinsics_matrices[i]
            camera_to_world = np.linalg.inv(world_to_camera)
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])

            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)

    # Align scene to the observation of the first camera
    scene_3d = apply_scene_alignment(scene_3d, extrinsics_matrices)

    print("GLB Scene built")
    return scene_3d


def integrate_camera_into_scene(scene: trimesh.Scene, transform: np.ndarray, face_colors: tuple, scene_scale: float):
    """
    Integrates a fake camera mesh into the 3D scene.

    Args:
        scene (trimesh.Scene): The 3D scene to add the camera model.
        transform (np.ndarray): Transformation matrix for camera positioning.
        face_colors (tuple): Color of the camera face.
        scene_scale (float): Scale of the scene.
    """

    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    # Create cone shape for camera
    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    # Combine transformations
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    # Generate mesh for the camera
    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    vertices_combined = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices_transformed = transform_points(complete_transform, vertices_combined)

    mesh_faces = compute_camera_faces(camera_cone_shape)

    # Add the camera mesh to the scene
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def apply_scene_alignment(scene_3d: trimesh.Scene, extrinsics_matrices: np.ndarray) -> trimesh.Scene:
    """
    Aligns the 3D scene based on the extrinsics of the first camera.

    Args:
        scene_3d (trimesh.Scene): The 3D scene to be aligned.
        extrinsics_matrices (np.ndarray): Camera extrinsic matrices.

    Returns:
        trimesh.Scene: Aligned 3D scene.
    """
    # Set transformations for scene alignment
    opengl_conversion_matrix = get_opengl_conversion_matrix()

    # Rotation matrix for alignment (180 degrees around the y-axis)
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()

    # Apply transformation
    initial_transformation = np.linalg.inv(extrinsics_matrices[0]) @ opengl_conversion_matrix @ align_rotation
    scene_3d.apply_transform(initial_transformation)
    return scene_3d


def get_opengl_conversion_matrix() -> np.ndarray:
    """
    Constructs and returns the OpenGL conversion matrix.

    Returns:
        numpy.ndarray: A 4x4 OpenGL conversion matrix.
    """
    # Create an identity matrix
    matrix = np.identity(4)

    # Flip the y and z axes
    matrix[1, 1] = -1
    matrix[2, 2] = -1

    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int = None) -> np.ndarray:
    """
    Applies a 4x4 transformation to a set of points.

    Args:
        transformation (np.ndarray): Transformation matrix.
        points (np.ndarray): Points to be transformed.
        dim (int, optional): Dimension for reshaping the result.

    Returns:
        np.ndarray: Transformed points.
    """
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]

    # Apply transformation
    transformation = transformation.swapaxes(-1, -2)  # Transpose the transformation matrix
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]

    # Reshape the result
    result = points[..., :dim].reshape(*initial_shape, dim)
    return result


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    """
    Computes the faces for the camera mesh.

    Args:
        cone_shape (trimesh.Trimesh): The shape of the camera cone.

    Returns:
        np.ndarray: Array of faces for the camera mesh.
    """
    # Create pseudo cameras
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)

    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone

        faces_list.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )

    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


def segment_sky(image_path, onnx_session, mask_filename=None):
    """
    Segments sky from an image using an ONNX model.
    Thanks for the great model provided by https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing

    Args:
        image_path: Path to input image
        onnx_session: ONNX runtime session with loaded model
        mask_filename: Path to save the output mask

    Returns:
        np.ndarray: Binary mask where 255 indicates non-sky regions
    """

    assert mask_filename is not None
    image = cv2.imread(image_path)

    result_map = run_skyseg(onnx_session, [320, 320], image)
    # resize the result_map to the original image size
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    # Fix: Invert the mask so that 255 = non-sky, 0 = sky
    # The model outputs low values for sky, high values for non-sky
    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255  # Use threshold of 32

    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    cv2.imwrite(mask_filename, output_mask)
    return output_mask


def run_skyseg(onnx_session, input_size, image):
    """
    Runs sky segmentation inference using ONNX model.

    Args:
        onnx_session: ONNX runtime session
        input_size: Target size for model input (width, height)
        image: Input image in BGR format

    Returns:
        np.ndarray: Segmentation mask
    """

    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result


def download_file_from_url(url, filename):
    """Downloads a file from a Hugging Face model repo, handling redirects."""
    try:
        # Get the redirect URL
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status()  # Raise HTTPError for bad requests (4xx or 5xx)

        if response.status_code == 302:  # Expecting a redirect
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            print(f"Unexpected status code: {response.status_code}")
            return

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")


def save_multiframe_colored_pointcloud(predictions, save_path='output_all_frames.ply'):
    """
    将多帧 world_points 和 images 保存成带颜色的点云 ply 文件
    """
    world_points = predictions['world_points'][0].cpu().numpy()  # [N, H, W, 3]
    images = predictions['images'][0].cpu().numpy()  # [N, 3, H, W]
    num_frames, H, W, _ = world_points.shape

    all_points = []
    all_colors = []

    for frame_idx in range(num_frames):
        # 当前帧的点和颜色
        pts = world_points[frame_idx].reshape(-1, 3)
        img = images[frame_idx].transpose(1, 2, 0)  # [H, W, 3]
        colors = (img * 255).astype(np.uint8).reshape(-1, 3)

        # 去掉无效点（比如 NaN）
        valid_mask = ~np.isnan(pts).any(axis=1)
        pts = pts[valid_mask]
        colors = colors[valid_mask]

        all_points.append(pts)
        all_colors.append(colors)

    # 合并多帧的点和颜色
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # 写入 ply 文件
    with open(save_path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(all_points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(all_points, all_colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

    print(f"Saved combined point cloud of {num_frames} frames to {save_path}")


def estimate_beta_A_multiframe_rgb_tensor(predictions, valid_points_dict):
    num_frames = len(valid_points_dict)
    device = predictions['images'].device

    I_r_list, I_g_list, I_b_list, d_list = [], [], [], []

    for frame_idx in valid_points_dict.keys():
        points = valid_points_dict[frame_idx]  # (N, 2) float

        if points.shape[0] == 0:
            continue

        u = points[:, 0].round().long().clamp(0, 518-1)
        v = points[:, 1].round().long().clamp(0, 350-1)

        images = predictions['images'][0, frame_idx]  # (3, H, W)
        depth = predictions['depth'][0, frame_idx, :, :, 0]  # (H, W)

        I_r = images[0, v, u].cpu().numpy()
        I_g = images[1, v, u].cpu().numpy()
        I_b = images[2, v, u].cpu().numpy()
        d = depth[v, u].cpu().numpy()

        I_r_list.append(I_r)
        I_g_list.append(I_g)
        I_b_list.append(I_b)
        d_list.append(d)

    # 拼接多帧所有匹配点
    I_r_all = np.concatenate(I_r_list)
    I_g_all = np.concatenate(I_g_list)
    I_b_all = np.concatenate(I_b_list)
    d_all   = np.concatenate(d_list)

    def residual(params, I, d):
        beta, A = params
        pred_I = A + (1.0 - A) * np.exp(-beta * d)
        return pred_I - I

    init_params = [0.5, 0.5]

    beta_list, A_list = [], []

    # R/G/B通道分别拟合
    for I_all in [I_r_all, I_g_all, I_b_all]:
        res = least_squares(residual, init_params, args=(I_all, d_all), bounds=([0, 0], [10, 1]))
        beta, A = res.x
        beta_list.append(beta)
        A_list.append(A)

    # 转成 1x3 tensor，复制 num_frames 行
    beta_tensor = torch.tensor(beta_list, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_frames, 1)
    A_tensor    = torch.tensor(A_list, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_frames, 1)

    return beta_tensor, A_tensor


def compute_J_from_d_v2(I, A, beta, d):
    """
    I:     [1, 5, 3, H, W]
    A:     [5, 3]
    beta:  [5, 3]
    d:     [1, 5, H, W, 1]
    
    return:
    J:     [1, 5, 3, H, W]
    """
    B, V, C, H, W = I.shape  # batch, views, channels, H, W

    # 调整A和beta形状用于广播
    A_expand = A.view(1, V, C, 1, 1)       # [1, 5, 3, 1, 1]
    beta_expand = beta.view(1, V, C, 1, 1) # [1, 5, 3, 1, 1]
    
    # 调整d形状用于广播
    d_expand = d.permute(0,1,4,2,3)        # [1, 5, 1, H, W] → [1, 5, 1, H, W]
    
    t = torch.exp(-beta_expand * d_expand) # [1, 5, 3, H, W]
    
    J = (I - A_expand * (1 - t)) / (t + 1e-8)

    # pdb.set_trace()
    J = torch.clamp(J, 0.0, 1.0)

    return J

def get_A(x):
    x_np = np.clip(torch_to_np(x), 0, 1)
    x_pil = np_to_pil(x_np)
    h, w = x_pil.size
    windows = (h + w) / 2
    A = x_pil.filter(ImageFilter.GaussianBlur(windows))
    A = ToTensor()(A)
    return A.unsqueeze(0)

def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]

def my_save_image(name, image_np, output_path=""):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    p = np_to_pil(image_np)
    p.save(output_path + "{}".format(name))


def pil_to_np(img_PIL, with_transpose=True):
    """
    Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]


def estimate_A(images, ratio=0.001):
    """
    images: [B, 3, H, W], float, 0-1
    ratio: top ratio of brightest pixels to consider
    return: [B, 3] 每张图的 A
    """
    B, C, H, W = images.shape
    device = images.device

    # 转为灰度: [B, H, W]
    gray = torch.mean(images, dim=1)

    num_pixels = H * W
    num_search = max(1, int(num_pixels * ratio))

    A_values = []
    for b in range(B):
        # 展平灰度图，取 top 0.1%
        values, indices = torch.topk(gray[b].view(-1), num_search)

        # 取这些像素对应的 RGB 值，再求均值
        selected_rgb = images[b].permute(1, 2, 0).view(-1, 3)[indices]  # [num_search, 3]
        A = torch.mean(selected_rgb, dim=0)  # [3]
        A_values.append(A)

    A_values = torch.stack(A_values, dim=0)  # [B, 3]
    return A_values

def estimate_A_darkchannel(images, ratio=0.001, r=7, eps=1e-3):
    """
    images: [B, 3, H, W], float 0-1
    ratio: 选取暗通道最暗像素的比例
    r: guided filter 半径
    eps: guided filter 正则化项
    return: [B, 3] 每张图的 A
    """
    B, C, H, W = images.shape
    device = images.device

    # dark channel: [B, 1, H, W]
    dark_channel = torch.min(images, dim=1, keepdim=True).values

    # guidance 使用灰度图
    gray = torch.mean(images, dim=1, keepdim=True)

    # guided filtering, 去噪
    refined_dark = guided_filter(gray, dark_channel, r=r, eps=eps)

    num_pixels = H * W
    num_search = max(1, int(num_pixels * ratio))

    A_values = []
    for b in range(B):
        refined_flat = refined_dark[b, 0].view(-1)
        values, indices = torch.topk(refined_flat, num_search, largest=False)

        selected_rgb = images[b].permute(1, 2, 0).view(-1, 3)[indices]
        A = torch.mean(selected_rgb, dim=0)
        A_values.append(A)

    A_values = torch.stack(A_values, dim=0)
    return A_values



def guided_filter(I, p, r, eps=1e-3):
    """
    I: guidance图像, [B, 1, H, W]  (灰度图)
    p: 输入图像, [B, 1, H, W] 或 [B, C, H, W]
    r: 半径
    eps: 正则化系数
    """
    B, C, H, W = p.shape

    ones = torch.ones((B, 1, H, W), device=I.device)

    N = F.avg_pool2d(ones, kernel_size=2*r+1, stride=1, padding=r)

    mean_I = F.avg_pool2d(I, kernel_size=2*r+1, stride=1, padding=r)
    mean_p = F.avg_pool2d(p, kernel_size=2*r+1, stride=1, padding=r)
    mean_Ip = F.avg_pool2d(I * p, kernel_size=2*r+1, stride=1, padding=r)

    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = F.avg_pool2d(I * I, kernel_size=2*r+1, stride=1, padding=r)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = F.avg_pool2d(a, kernel_size=2*r+1, stride=1, padding=r)
    mean_b = F.avg_pool2d(b, kernel_size=2*r+1, stride=1, padding=r)

    q = mean_a * I + mean_b
    return q

def compute_A_gt(images, depth, top_ratio=0.001):
    """
    根据深度图的最大部分像素区域，取对应图像颜色均值作为A_gt。

    Args:
        images: [B, 3, H, W]  归一化0~1
        depth:  [B, 1, H, W, 1]
        top_ratio: 取top比例的最远像素，默认0.1%

    Returns:
        A_gt: [B, 3]
    """
    B, C, H, W = images.shape
    device = images.device

    # reshape depth: [B, H, W]
    depth_map = depth.view(B, H, W)

    num_pixels = H * W
    num_top = max(1, int(num_pixels * top_ratio))

    A_gt_list = []
    for b in range(B):
        # 展平成一维，取 top N 最大深度值的索引
        depth_flat = depth_map[b].view(-1)
        topk_values, topk_indices = torch.topk(depth_flat, num_top, largest=True)

        # 取这些位置对应图像的 RGB 值
        img_flat = images[b].permute(1, 2, 0).view(-1, 3)  # [H*W, 3]
        selected_colors = img_flat[topk_indices]  # [num_top, 3]

        # 求均值作为 A_gt
        A_gt = torch.mean(selected_colors, dim=0)  # [3]
        A_gt_list.append(A_gt)

    A_gt = torch.stack(A_gt_list, dim=0)  # [B, 3]
    return A_gt

def scale_align(pred, gt):
    """
    pred, gt: torch.Tensor, shape [H,W], 必须 mask 好无效值
    """
    mask = (gt > 0)
    pred_valid = pred[mask]
    gt_valid = gt[mask]

    if pred_valid.numel() == 0:
        return pred  # 全是无效值，直接返回原值

    scale = torch.sum(pred_valid * gt_valid) / torch.sum(pred_valid ** 2)
    pred_aligned = pred * scale.item()
    return pred_aligned

def compute_depth_metrics(pred_depth, gt_depth, mask=None, epsilon=1e-6):
    """
    计算常见的深度估计指标：MAE, RMSE, REL, delta1, delta2, delta3

    Args:
        pred_depth (Tensor): shape [H, W] or [1, H, W]
        gt_depth (Tensor):   shape [H, W] or [1, H, W]
        mask (Tensor):       可选，bool类型，shape相同。为True的位置参与计算
        epsilon (float):     防止除0

    Returns:
        dict 或 None: 如果有有效像素，返回各项指标的字典；否则返回 None
    """
    # 保证 shape 一致 & 去掉多余维度
    if pred_depth.ndim == 3:
        pred_depth = pred_depth.squeeze(0)
    if gt_depth.ndim == 3:
        gt_depth = gt_depth.squeeze(0)

    # 有效值mask（默认gt>0）+ 可选外部mask
    valid_mask = (gt_depth > 0) & torch.isfinite(gt_depth) & torch.isfinite(pred_depth)
    if mask is not None:
        valid_mask = valid_mask & mask

    # 如果无有效像素，返回 None，不中断程序
    if valid_mask.sum() == 0:
        return None

    pred = pred_depth[valid_mask]
    gt   = gt_depth[valid_mask]

    abs_diff = torch.abs(pred - gt)
    mae  = abs_diff.mean().item()
    rmse = torch.sqrt((abs_diff ** 2).mean()).item()
    rel  = (abs_diff / (gt + epsilon)).mean().item()

    ratio = torch.max(pred / (gt + epsilon), gt / (pred + epsilon))
    delta1 = (ratio < 1.25).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'REL': rel,
        'delta1': delta1,
        'delta2': delta2,
        'delta3': delta3
    }

    return metrics

def compute_si_rmse(pred, gt):
    """
    Scale-invariant RMSE (si-RMSE)
    pred, gt: [H,W]，无效值需 mask 掉（gt>0）
    """
    mask = (gt > 0)
    pred_valid = pred[mask]
    gt_valid   = gt[mask]

    if pred_valid.numel() == 0:
        return np.nan

    log_diff = torch.log(pred_valid + 1e-8) - torch.log(gt_valid + 1e-8)
    mse = torch.mean(log_diff ** 2)
    si_rmse = torch.sqrt(mse - (torch.mean(log_diff)) ** 2)

    return si_rmse.item()

def mixup_old_new_tokens(tokens_old_list, tokens_new_list, alpha=1.0):
    """
    将第一次modulator前后的token做mixup
    tokens_old_list, tokens_new_list: List of Tensor，shape [B, N, C]
    alpha: Beta分布参数
    返回: mixed_tokens_list, lam
    """
    assert len(tokens_old_list) == len(tokens_new_list), "Token list数量必须一致"

    mixed_tokens_list = []
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    for tokens_old, tokens_new in zip(tokens_old_list, tokens_new_list):
        mixed_tokens = lam * tokens_old + (1 - lam) * tokens_new
        mixed_tokens_list.append(mixed_tokens)

    return mixed_tokens_list, lam