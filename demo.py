from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch import autocast
from pathlib import Path
import open3d as o3d

# Custom models
from models.stereoanywhere import StereoAnywhere as StereoAnywhere
from models.depth_anything_v2 import get_depth_anything_v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

"""
Example usage:
CUDA_VISIBLE_DEVICES=0 python demo.py \
    --left eth3d/training/delivery_area_1l/im0.png \
    --right eth3d/training/delivery_area_1l/im1.png \
    --loadstereomodel weights/stereoanywhere_sceneflow.pth \
    --loadmonomodel weights/depth_anything_v2_vitl.pth \
    --outdir tmp/ \
    --intrinsics 541.764 0 553.869 0 541.764 232.396 0 0 1 \
    --baseline 0.0599
"""

parser = argparse.ArgumentParser(description='StereoAnywhere Demo')

# Image inputs
parser.add_argument('--left', nargs='+', required=True, help='path to left image(s)')
parser.add_argument('--right', nargs='+', required=True, help='path to right image(s)')

# Model paths
parser.add_argument('--loadstereomodel', required=True, help='load stereo model')
parser.add_argument('--loadmonomodel', default=None, help='load mono model')

# Camera parameters (optional for depth computation)
parser.add_argument('--intrinsics', type=float, nargs=9, default=None,
                    help='camera intrinsics as 9 values (row-major 3x3 matrix)')
parser.add_argument('--baseline', type=float, default=None,
                    help='baseline in meters')

# Output
parser.add_argument('--outdir', required=True, help='output directory')

# Model parameters
parser.add_argument('--vit_encoder', default='vitl', choices=['vitl', 'vitb', 'vits'],
                    help='select vit encoder (Only for DAv2)')
parser.add_argument('--monomodel', default='DAv2', help='select mono model')
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

# Architecture parameters
parser.add_argument('--n_downsample', type=int, default=2)
parser.add_argument('--n_additional_hourglass', type=int, default=0)
parser.add_argument('--volume_channels', type=int, default=8)
parser.add_argument('--vol_downsample', type=float, default=0)
parser.add_argument('--vol_n_masks', type=int, default=8)
parser.add_argument('--use_truncate_vol', action='store_true')
parser.add_argument('--mirror_conf_th', type=float, default=0.98)
parser.add_argument('--mirror_attenuation', type=float, default=0.9)
parser.add_argument('--use_aggregate_stereo_vol', action='store_true')
parser.add_argument('--use_aggregate_mono_vol', action='store_true')
parser.add_argument('--normal_gain', type=int, default=10)
parser.add_argument('--lrc_th', type=float, default=1.0)
parser.add_argument('--iters', type=int, default=32, help='Number of iterations')

# Processing options
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
parser.add_argument('--mixed_precision', action='store_true')
parser.add_argument('--iscale', type=float, default=1.0, help='scale factor for input images')

# Point cloud parameters
parser.add_argument('--min_depth', type=float, default=0.1, help='minimum depth threshold in meters for point cloud')
parser.add_argument('--max_depth', type=float, default=15.0, help='maximum depth threshold in meters for point cloud')
parser.add_argument('--voxel_size', type=float, default=0.002, help='voxel size for point cloud downsampling')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

autocast_device = str(device).split(':')[0]

class StereoAnywhereWrapper(torch.nn.Module):
    """Wrapper for StereoAnywhere to enable torch.compile optimization."""
    def __init__(self, args, stereo_model, mono_model):
        super(StereoAnywhereWrapper, self).__init__()
        self.args = args
        self.stereo_model = stereo_model
        self.mono_model = mono_model

    def forward(self, left_img, right_img, left_mono, right_mono):
        # Pad images to be divisible by 32
        ht, wt = left_img.shape[-2], left_img.shape[-1]
        pad_ht = (((ht // 32) + 1) * 32 - ht) % 32
        pad_wd = (((wt // 32) + 1) * 32 - wt) % 32
        
        _pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        left_img = F.pad(left_img, _pad, mode='replicate')
        right_img = F.pad(right_img, _pad, mode='replicate')
        left_mono = F.pad(left_mono, _pad, mode='replicate')
        right_mono = F.pad(right_mono, _pad, mode='replicate')
        
        # Run stereo inference
        pred_disps, _ = self.stereo_model(left_img, right_img, left_mono, right_mono, 
                                         test_mode=True, iters=self.args.iters)
        
        # Remove padding
        pred_disp = -pred_disps.squeeze(1)
        hd, wd = pred_disp.shape[-2:]
        c = [_pad[2], hd-_pad[3], _pad[0], wd-_pad[1]]
        pred_disp = pred_disp[..., c[0]:c[1], c[2]:c[3]]
        
        return pred_disp

def load_image(image_path, iscale=1.0):
    """Load and preprocess image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    # Handle grayscale images
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Store original shape before scaling
    original_shape = img.shape
    
    # Scale image if needed
    if iscale != 1.0:
        new_width = round(img.shape[1] / iscale)
        new_height = round(img.shape[0] / iscale)
        img = cv2.resize(img, (new_width, new_height))
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    return img, original_shape


def save_disparity(disparity, output_path):
    """Save disparity map as npy, and colorized version."""
    # Save raw disparity as numpy array
    disparity_np = disparity.squeeze().cpu().numpy()
    
    # Save as numpy array
    npy_path = output_path.replace('.png', '.npy')
    np.save(npy_path, disparity_np)
    
    # Save colorized version
    max_val = disparity_np.max()
    if max_val > 0:
        normalized = (disparity_np / max_val * 255).astype(np.uint8)
        colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        cv2.imwrite(output_path, colorized)
    
    print(f"Disparity saved to: {output_path}")
    print(f"Disparity array saved to: {npy_path}")


def save_depth(depth, output_path):
    """Save depth map as npy, and colorized version."""
    # Save raw depth as numpy array
    depth_np = depth.squeeze().cpu().numpy()
    
    # Save as numpy array (in meters)
    npy_path = output_path.replace('.png', '.npy')
    np.save(npy_path, depth_np)
    
    # Save colorized version
    max_val = depth_np.max()
    if max_val > 0:
        normalized = (depth_np / max_val * 255).astype(np.uint8)
        colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        cv2.imwrite(output_path, colorized)
    
    print(f"Depth saved to: {output_path}")
    print(f"Depth array saved to: {npy_path}")

@torch.no_grad()
def run_inference(left_img, right_img, wrapper, mono_model, args, original_shape):
    """Run stereo inference."""
    
    if args.cuda:
        left_img = left_img.to(device)
        right_img = right_img.to(device)
    
    # Get monocular depth priors
    with autocast(autocast_device, enabled=args.mixed_precision):
        if mono_model is not None:
            # Concatenate left and right images
            mono_input = torch.cat([left_img, right_img], 0)
            mono_depths = mono_model.infer_image(mono_input, input_size_width=518, input_size_height=518)
            
            # Normalize depth between 0 and 1
            mono_depths = (mono_depths - mono_depths.min()) / (mono_depths.max() - mono_depths.min())
            left_mono, right_mono = mono_depths[0].unsqueeze(0), mono_depths[1].unsqueeze(0)
        else:
            left_mono = torch.zeros_like(left_img)[:, 0:1]
            right_mono = torch.zeros_like(right_img)[:, 0:1]
    
    # Run inference through wrapper
    with autocast(autocast_device, enabled=args.mixed_precision):
        pred_disp = wrapper(left_img, right_img, left_mono, right_mono)
    
    # Resize to original resolution if iscale was used
    if args.iscale != 1.0:
        pred_disp = pred_disp.squeeze(0).cpu().numpy()
        pred_disp = cv2.resize(pred_disp, (original_shape[1], original_shape[0]))
        pred_disp = torch.from_numpy(pred_disp).unsqueeze(0)
    
    return pred_disp


def disparity_to_depth(disparity, intrinsics, baseline):
    """Convert disparity to depth using camera parameters.
    
    Args:
        disparity: disparity map (H, W) in pixels
        intrinsics: 3x3 camera intrinsic matrix
        baseline: baseline in meters
    
    Returns:
        depth map (H, W) in meters
    """
    # focal length from intrinsics (assuming fx = fy)
    fx = intrinsics[0, 0]
    
    # Depth = (focal_length * baseline) / disparity
    # Avoid division by zero
    depth = torch.where(disparity > 0, 
                       (fx * baseline) / disparity,
                       torch.zeros_like(disparity))
    
    return depth


def save_point_cloud(depth, left_img, intrinsics, output_path, voxel_size=0.002, min_depth=0.1, max_depth=15.0):
    """Save point cloud as PLY file.
    
    Args:
        depth: depth map (H, W) as numpy array
        left_img: left image tensor (1, 3, H, W) normalized to [0, 1]
        intrinsics: 3x3 camera intrinsic matrix
        output_path: path to save PLY file
        voxel_size: voxel size for downsampling
        min_depth: minimum depth threshold in meters
        max_depth: maximum depth threshold in meters
    """
    
    # Get image dimensions
    H, W = depth.shape
    
    # Extract intrinsic parameters
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    
    # Create pixel coordinate grids
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Reproject to 3D using depth map and intrinsics
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = depth
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    
    # Stack coordinates
    points_3d = np.stack([X, Y, Z], axis=-1)
    
    # Create valid mask with min and max depth thresholds
    valid_mask = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    
    # Extract valid points
    pts = points_3d[valid_mask]
    
    # Get colors from left image (convert from tensor to numpy and extract valid pixels)
    left_img_np = left_img.detach().cpu().numpy()[0]  # (3, H, W)
    cols = left_img_np[:, valid_mask].T.astype(np.float64)  # (N, 3)
    
    print(f"Shape of point cloud: {pts.shape}, colors: {cols.shape}")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    
    # Voxel downsampling
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Point cloud after downsampling: {len(pcd.points)} points")
    
    # Save as PLY
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"Point cloud saved to: {output_path}")


def main():
    import tqdm
    
    if len(args.left) != len(args.right):
        raise ValueError(f"Number of left images ({len(args.left)}) must match number of right images ({len(args.right)})")
    
    print("Loading models...")
    
    # Set dtype
    dtype = torch.float32
    
    # Load stereo model
    stereo_model = StereoAnywhere(args)
    stereo_model = nn.DataParallel(stereo_model)
    
    if args.cuda:
        stereo_model.cuda()
    
    pretrain_dict = torch.load(args.loadstereomodel, map_location=device)
    pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
    stereo_model.load_state_dict(pretrain_dict, strict=True)
    stereo_model = stereo_model.module.eval().to(dtype)
    
    # Load mono model
    if args.monomodel == 'DAv2' and args.loadmonomodel is not None:
        mono_model = get_depth_anything_v2(args.loadmonomodel, encoder=args.vit_encoder)
        mono_model = mono_model.to(device).to(dtype)
        mono_model.eval()
    else:
        mono_model = None
    
    # Create wrapper
    wrapper = StereoAnywhereWrapper(args, stereo_model, mono_model)
    wrapper = wrapper.cuda().eval() if args.cuda else wrapper.eval()
        
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Process each image pair
    for left_path, right_path in tqdm.tqdm(zip(args.left, args.right), total=len(args.left), desc="Processing images"):
        print(f"\nProcessing: {os.path.basename(left_path)}")
        
        # Load images
        left_img, left_shape = load_image(left_path, args.iscale)
        right_img, right_shape = load_image(right_path, args.iscale)
        
        print(f"Image size: {left_img.shape[2]}x{left_img.shape[3]}")
        
        # Run inference
        disparity = run_inference(left_img, right_img, wrapper, mono_model, args, left_shape)
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(left_path))[0]
        disparity_path = os.path.join(args.outdir, f'{base_name}_disparity.png')
        
        # Save disparity
        save_disparity(disparity, disparity_path)
        
        # Compute and save depth if camera parameters are provided
        if args.intrinsics is not None and args.baseline is not None:
            intrinsics = np.array(args.intrinsics).reshape(3, 3)
            intrinsics_tensor = torch.from_numpy(intrinsics).float()

            print(f"K: {intrinsics}; Baseline: {args.baseline}m; Min depth: {args.min_depth}m; Max depth: {args.max_depth}m")
            
            depth = disparity_to_depth(disparity, intrinsics_tensor, args.baseline)
            
            depth_path = os.path.join(args.outdir, f'{base_name}_depth.png')
            save_depth(depth, depth_path)
            
            print(f"Depth range: {depth[depth > 0].min():.3f}m to {depth.max():.3f}m")
            
            # Save point cloud as PLY
            ply_path = os.path.join(args.outdir, f'{base_name}.ply')
            save_point_cloud(depth.squeeze().cpu().numpy(), left_img, intrinsics, ply_path,
                           voxel_size=args.voxel_size, min_depth=args.min_depth, max_depth=args.max_depth)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
