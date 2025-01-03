import cv2
import einops
import numpy as np
import torch
from torchvision import transforms

from data_related.converter import Converter

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=np.uint8,
)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=np.uint8,
)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
    dtype=np.uint8,
)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
    dtype=np.uint8,
)


def img_transform(resolution):
    augmentations = transforms.Compose([
        # TODO: 又 resize 又 centercrop 感觉这两步有些重复了
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def transform(imgs):
        return [augmentations(img.convert("RGB")) for img in imgs]

    return transform


def _depth_normalize(dpt: np.ndarray):
    dpt = dpt.astype(np.float64)
    vmin, vmax = np.percentile(dpt, 2), np.percentile(dpt, 85)
    dpt -= vmin  # 首先将所有的深度值减去 vmin，使得最小值变为零.
    dpt /= vmax - vmin  # 接着将所有的深度值除以 (vmax - vmin)，使得归一化后的深度值范围在 [0, 1] 之间。
    # 这一操作将深度值进行反转，原本较小的深度值变为较大的值，较大的深度值变为较小的值。这样做的目的是为了使得“近距离”在视觉上更加突出（通常在深度图中，深度越小表示越近）。
    dpt = 1.0 - dpt
    # 将归一化后的深度值（范围 [0, 1]）映射到 8-bit 图像的值域 [0, 255]。
    # clip(0, 255) 确保输出的像素值不会超出这个范围，最后将其转换为 np.uint8 类型，适合用于图像显示。
    return (dpt * 255.0).clip(0, 255).astype(np.uint8)


def _HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    match C:
        case 3:
            return x
        case 1:
            return np.concatenate([x, x, x], axis=2)
        case 4:
            color, alpha = x[:, :, 0:3].astype(np.float32), x[:, :, 3:4].astype(np.float32) / 255.0
            y = color * alpha + 255.0 * (1.0 - alpha)
            return y.clip(0, 255).astype(np.uint8)
        case _:
            NotImplementedError("暂未实现对于其他通道数图像的转换")


def _resize_img(img, resolution):
    H, W, C = img.shape
    H, W = float(H), float(W)
    k = float(resolution) / min(H, W)  # 计算缩放因子 `k`
    H, W = int(np.round((H * k) / 64.0)) * 64, int(np.round((W * k) / 64.0)) * 64
    # my adding for kitti process
    if W > 2.5 * H:
        W = int(2.5 * H)
    """
    `interpolation` 插值方法解析:
    如果 k > 1 (即图像被放大), 则使用 cv2.INTER_LANCZOS4（Lanczos插值), 这种方法适用于放大图像时能够保留更多细节.
    如果 k <= 1 (即图像被缩小), 则使用 cv2.INTER_AREA (面积插值), 这种方法在缩小图像时通常效果较好.
    """
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def fill_in_fast(depth_map, max_depth=15.0, custom_kernel=DIAMOND_KERNEL_7, extrapolate=False, blur_type="bilateral"):
    """
    Fast, in-place depth completion.

    Args:
    depth_map: projected depths
    max_depth: max depth value for inversion
    custom_kernel: kernel to apply initial dilation
    extrapolate: whether to extrapolate by extending depths to top of
    the frame, and applying a 31x31 full kernel dilation
    blur_type:
    'bilateral' - preserves local structure (recommended)
    'gaussian' - provides lower RMSE

    Returns:
    depth_map: dense depth map
    """

    # Invert
    valid_pixels = depth_map > 0.1
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = depth_map < 0.1
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0 : top_row_pixels[pixel_col_idx], pixel_col_idx] = top_pixel_values[pixel_col_idx]

        # Large Fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    # Bilateral or Gaussian blur
    if blur_type == "bilateral":
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == "gaussian":
        # Gaussian blur
        valid_pixels = depth_map > 0.1
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = depth_map > 0.1
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map


def pcd_transform(resolution):
    def transform(pc):
        # 投影成深度图
        dpt = Converter.proj_pc2dpt(pc, extrinsic=np.eye(4), intrinsic=np.eye(3), h=resolution, w=resolution)
        dpt = fill_in_fast(dpt.astype(np.float32))  # 对深度图进行致密化, 之前写代码的时候忘记把这一段逻辑抄上了
        # 对深度图的大小进行调整
        dpt = np.array(_resize_img(_HWC3(_depth_normalize(dpt)), resolution))
        # 转换成 Tensor 并进行维度调整
        dpt = torch.from_numpy(dpt.copy()).float() / 255.0
        dpt = einops.rearrange(dpt, "h w c -> c h w").clone()
        return dpt

    return transform
