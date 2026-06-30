import cv2
import numpy as np
import shutil
import os
from numba import njit
import torch.nn.functional as F
import torch
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from kornia.filters import spatial_gradient

def correlation_score(normals_a, normals_b):
    B, C, H, W = normals_a.shape

    normals_a = normals_a.view(B, C, H, W)
    normals_b = normals_b.view(B, C, H, W)

    corr_score = torch.sum(normals_a * normals_b, dim=1, keepdim=True) # B 1 H W
    
    return corr_score

def estimate_normals(depth, normal_gain):
    xy_gradients = -spatial_gradient(normal_gain*depth, mode='diff', order=1, normalized=False).squeeze(1) # B 2 H W
    normals = torch.cat([xy_gradients, torch.ones_like(xy_gradients[:,0:1])], 1) # B 3 H W
    normals = normals / torch.linalg.norm(normals, dim=1, keepdim=True)
    return normals

def ransac_depth_scale_shift(mde, reference_depths, residual_threshold=1.0, depth_threshold=0.1):
    """
    Perform RANSAC to estimate depth scale and shift (linear regression) after applying a threshold.
    
    Parameters:
    mde: torch.tensor, shape (B, 1, H, W)
        The depth values from the model-derived estimate (MDE), a 2D array.
    reference_depths: torch.tensor, shape (B, 1, H, W)
        The true or reference depth values, a 2D array.
    residual_threshold: float, default=1.0
        The maximum residual for a data point to be classified as an inlier.
    depth_threshold: float, default=0.1
        Threshold below which mde values are excluded from fitting.
        
    Returns:
    scale: torch.tensor, shape (B, 1, 1, 1)
        The estimated scale factor for the depth values.
    shift: torch.tensor, shape (B, 1, 1, 1)
        The estimated shift factor for the depth values.
    """

    B, _, _, _ = mde.shape
    device = mde.device

    mde = mde.detach().cpu().numpy()
    reference_depths = reference_depths.detach().cpu().numpy()

    scale_and_shift = torch.zeros((B, 2), dtype=torch.float32, device=device)

    for b in range(B):
        # Apply mask to filter out mde values below the threshold

        _mde = mde[b, 0]
        _reference_depths = reference_depths[b, 0]

        valid_mask = _reference_depths >= depth_threshold
        mde_valid = _mde[valid_mask]
        reference_depths_valid = _reference_depths[valid_mask]
        
        if mde_valid.size == 0:
            raise ValueError("No valid depth values found after applying threshold.")
        
        # Flatten valid points for RANSAC processing
        mde_valid = mde_valid.reshape(-1, 1)
        reference_depths_valid = reference_depths_valid.flatten()

        # RANSAC with a linear regression model
        ransac = RANSACRegressor(LinearRegression(), residual_threshold=residual_threshold)
        
        # Fit the model to the valid MDE and reference depths
        ransac.fit(mde_valid, reference_depths_valid)
        
        # Get the linear model parameters
        scale = ransac.estimator_.coef_[0]  # slope (scale)
        shift = ransac.estimator_.intercept_  # intercept (shift)

        scale_and_shift[b, 0] = torch.from_numpy(np.array(scale, dtype=np.float32)).to(device)
        scale_and_shift[b, 1] = torch.from_numpy(np.array(shift, dtype=np.float32)).to(device)
    
    return scale_and_shift[:, 0].reshape(B, 1, 1, 1), scale_and_shift[:, 1].reshape(B, 1, 1, 1)

@njit
def _fast_warp_depth(depth_map, pts):
    hh,hw = depth_map.shape[:2]
    for i in range(pts.shape[0]):
        u,v,z = pts[i]
        u,v = round(u), round(v)
        if u < hw and v < hh:
            if depth_map[v,u] == 0 or depth_map[v,u] > z:
                depth_map[v,u] = z

@njit
def _fast_warp_disparity(disparity_map, pts):
    hh,hw = disparity_map.shape[:2]
    for i in range(pts.shape[0]):
        u,v,d = pts[i]
        u,v = round(u), round(v)
        if u < hw and v < hh:
            if disparity_map[v,u] == 0 or disparity_map[v,u] < d:
                disparity_map[v,u] = d  

@njit
def _fast_warp_depth(depth_map, pts):
    hh,hw = depth_map.shape[:2]
    for i in range(pts.shape[0]):
        u,v,z = pts[i]
        u,v = round(u), round(v)
        if u < hw and v < hh:
            if depth_map[v,u] == 0 or depth_map[v,u] > z:
                depth_map[v,u] = z

def pcd_to_depth(pcd, intrins, out_shape):
    h, w = out_shape
    pcd_homo = pcd / pcd[..., -1:]
    pts = intrins @ pcd_homo.transpose([1, 0])
    depth = np.zeros([h, w])
    
    _fast_warp_depth(depth, pts)

    return depth

def resize_3d(dmap, K, factor):
    v,u = np.where(dmap > 0)
    z = dmap[v,u]

    ones = np.ones_like(v)
    uv = np.vstack([u,v,ones])
    z = np.expand_dims(z,-1).T
    z = np.concatenate([z,z,z], axis=0)

    K_inv = np.linalg.inv(K)
    pcd = z * (K_inv @ uv)

    hK = K / factor
    hK[2,2] = 1

    pts = hK @ pcd
    pts[:2,:] /= pts[2,:]
    pts[:2,:] = np.round(pts[:2,:])
    pts = pts.T #Nx3

    hh,hw = dmap.shape[:2]
    hh,hw = hh//factor,hw//factor

    #Ignore boudaries check
    hdmap = np.zeros((hh,hw), dtype=np.float32)

    _fast_warp_depth(hdmap,pts)

    return hdmap

def resize_3d_short(dmap: np.ndarray, factor: float, dmap_type: str = "depth"):
    v,u = np.where(dmap > 0)
    d = dmap[v,u]
    pts = np.vstack([u,v,d])

    pts[:2,:] /= factor
    pts = pts.T #Nx3

    hh,hw = dmap.shape[:2]
    hh,hw = hh//factor,hw//factor

    #Ignore boudaries check
    hdmap = np.zeros((hh,hw), dtype=np.float32)

    if dmap_type == "depth":
        _fast_warp_depth(hdmap, pts)
    elif dmap_type == "disparity":
        _fast_warp_disparity(hdmap, pts)
    else:
        raise Exception(f"Unknown type {dmap_type}. Need depth or disparity")

    scale_factor = 1 if dmap_type == "depth" else factor

    return hdmap / scale_factor

# def pytorch_3d_short(dmap, factor, dtype= "depth"):
#     h,w = dmap.shape[-2], dmap.shape[-1]
#     hh,hw = h//factor,w//factor



def sgm_opencv(imgL, imgR, maxdisp=192, p_factor=7, w_size=16, interpolate=True):
    # disparity range is tuned for 'aloe' image pair
    window_size = p_factor
    min_disp = 0
    num_disp = maxdisp - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=w_size,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=3,
                                   uniquenessRatio=10,
                                   speckleWindowSize=150,
                                   speckleRange=32
                                   )

    # print('computing disparity...')
    padded_left = cv2.copyMakeBorder(imgL, 0,0,num_disp,0,cv2.BORDER_CONSTANT,None,0)
    padded_right = cv2.copyMakeBorder(imgR, 0,0,num_disp,0,cv2.BORDER_CONSTANT,None,0)
    disp = stereo.compute(padded_left, padded_right).astype(np.float32) / 16.0
    disp = disp[:,num_disp:]
    if interpolate:
        _interpolate_background(disp)
    return disp

def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(
        ".", "..", ".git*", "*pycache*", "*build", "*.fuse*", "*_drive_*",
        "*pretrained*", "*log*", "*.vscode*", "*tmp*", "*weights*", "*thirdparty*", "*samples*")

    if os.path.exists(backup_directory):
        raise Exception(f"Backup directory {backup_directory} already exists")

    shutil.copytree('.', backup_directory, ignore=ignore_hidden)
    os.system("chmod -R g+w {}".format(backup_directory))

#https://github.com/simonmeister/motion-rcnn/blob/master/devkit/cpp/io_disp.h
@njit
def _interpolate_background(dmap):
    h,w = dmap.shape[:2]

    for v in range(h):
        count = 0
        for u in range(w):
            if dmap[v,u] > 0:
                if count >= 1:#at least one pixel requires interpolation
                    u1,u2 = u-count,u-1#first and last value for interpolation
                    if u1>0 and u2<w-1:#set pixel to min disparity
                        d_ipol = min(dmap[v,u1-1], dmap[v,u2+1])
                        for u_curr in range(u1,u2+1):
                            dmap[v,u_curr] = d_ipol
                count = 0
            else:
                count +=1
        
        #Border interpolation(left,right): first valid dmap value is used as filler
        for u in range(w):
            if dmap[v,u] > 0:
                for u2 in range(u):
                    dmap[v,u2] = dmap[v,u]
                break

        for u in range(w-1,-1,-1):
            if dmap[v,u] > 0:
                for u2 in range(u+1,w):
                    dmap[v,u2] = dmap[v,u]
                break
        
    #Border interpolation(top,bottom): first valid dmap value is used as filler
    for u in range(w):
        for v in range(h):
            if dmap[v,u] > 0:
                for v2 in range(v):
                    dmap[v2,u] = dmap[v,u]
                break
        
        for v in range(h-1,-1,-1):
            if dmap[v,u] > 0:
                for v2 in range(v+1,h):
                    dmap[v2,u] = dmap[v,u]
                break


_color_map_errors_kitti = np.array([
        [ 0,       0.1875, 149,  54,  49],
        [ 0.1875,  0.375,  180, 117,  69],
        [ 0.375,   0.75,   209, 173, 116],
        [ 0.75,    1.5,    233, 217, 171],
        [ 1.5,     3,      248, 243, 224],
        [ 3,       6,      144, 224, 254],
        [ 6,      12,       97, 174, 253],
        [12,      24,       67, 109, 244],
        [24,      48,       39,  48, 215],
        [48,  np.inf,       38,   0, 165]
]).astype(float)

def color_error_image_kitti(errors, scale=1, mask=None, BGR=True, dilation=1):
    errors_flat = errors.flatten()
    colored_errors_flat = np.zeros((errors_flat.shape[0], 3))
    
    for col in _color_map_errors_kitti:
        col_mask = np.logical_and(errors_flat>=col[0]/scale, errors_flat<=col[1]/scale)
        colored_errors_flat[col_mask] = col[2:]
        
    if mask is not None:
        colored_errors_flat[mask.flatten() == 0] = 0

    if not BGR:
        colored_errors_flat = colored_errors_flat[:, [2, 1, 0]]

    colored_errors = colored_errors_flat.reshape(errors.shape[0], errors.shape[1], 3).astype(np.uint8)

    if dilation>0:
        kernel = np.ones((dilation, dilation))
        colored_errors = cv2.dilate(colored_errors, kernel)
    return colored_errors


def guided_visualize(disp, gt, valid, scale=1, dilation=7):
    H,W = disp.shape[:2]

    error = np.abs(disp-gt)
    error[valid==0] = 0
    
    bad1 = np.zeros((H,W,3), dtype=np.uint8)
    bad1[error > 1.,:] = (49, 54,  149)    
    bad1[error <= 1.,:] = (165,  0, 38)   
    bad1[valid==0,:] = (0,0,0)

    bad2 = np.zeros((H,W,3), dtype=np.uint8)
    bad2[error > 2.,:] = (49, 54,  149)       
    bad2[error <= 2.,:] = (165,  0, 38)   
    bad2[valid==0,:] = (0,0,0)

    bad3 = np.zeros((H,W,3), dtype=np.uint8)
    bad3[error > 3.,:] = (49, 54,  149)       
    bad3[error <= 3.,:] = (165,  0, 38) 
    bad3[valid==0,:] = (0,0,0)

    bad4 = np.zeros((H,W,3), dtype=np.uint8)
    bad4[error > 4.,:] = (49, 54,  149)     
    bad4[error <= 4.,:] = (165,  0, 38)  
    bad4[valid==0,:] = (0,0,0)
 
    if dilation>0:
        kernel = np.ones((dilation, dilation))
        bad1 = cv2.dilate(bad1, kernel)
        bad2 = cv2.dilate(bad2, kernel)
        bad3 = cv2.dilate(bad3, kernel)
        bad4 = cv2.dilate(bad4, kernel)

    avgerr = color_error_image_kitti(error, scale=scale, mask=valid, dilation=dilation)
    
    rms = (disp-gt)**2
    rms = np.sqrt(rms)

    rms = color_error_image_kitti(rms, scale=scale, mask=valid, dilation=dilation)
    
    return {'bad 1.0':bad1, 'bad 2.0':bad2, 'bad 3.0': bad3, 'bad 4.0':bad4, 'avgerr':avgerr, 'rms':rms}



def sample_hints(hints, probability=0.05):
    new_hints = hints.copy()
    valid_hints = hints>0
    valid_hints = (valid_hints & (np.random.rand(*hints.shape)<probability)).astype(np.float32)
    new_hints[valid_hints==0] = 0
    return new_hints

_cache_dict = {}

def disp_warping(disp, img, right_disp=False):
    B, _, H, W = disp.shape

    global _cache_dict

    if f'mycoords_{H}_{W}' not in _cache_dict:
        mycoords_y, mycoords_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        mycoords_x = mycoords_x[None].repeat(B, 1, 1).to(disp.device)
        mycoords_y = mycoords_y[None].repeat(B, 1, 1).to(disp.device)
        _cache_dict[f'mycoords_{H}_{W}'] = (mycoords_x, mycoords_y)
    else:
        mycoords_x, mycoords_y = _cache_dict[f'mycoords_{H}_{W}']

    if right_disp:
        grid = 2 * torch.cat([(mycoords_x+disp.squeeze(1)).unsqueeze(-1) / W, mycoords_y.unsqueeze(-1) / H], -1) - 1
    else:
        grid = 2 * torch.cat([(mycoords_x-disp.squeeze(1)).unsqueeze(-1) / W, mycoords_y.unsqueeze(-1) / H], -1) - 1

    # grid_sample: B,C,H,W & B H W 2 -> B C H W
    warped_img = F.grid_sample(img, grid, align_corners=False)

    return warped_img

def image_to_coordinates_array(grayscale_image):
    """
    Convert an HxW grayscale image tensor into an Nx3 tensor where N = H * W.
    Each row represents the (x, y, grayscale_value) of a pixel.

    Parameters:
    grayscale_image (torch.Tensor): HxW grayscale image.

    Returns:
    torch.Tensor: Nx3 tensor with columns [x, y, grayscale_value].
    """
    B, _, H, W = grayscale_image.shape

    y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    
    # Generate coordinates
    x_coords = x_coords.unsqueeze(0).repeat(B, 1, 1).reshape(B, -1).to(grayscale_image.device)
    y_coords = y_coords.unsqueeze(0).repeat(B, 1, 1).reshape(B, -1).to(grayscale_image.device)
    
    # Flatten grayscale image to get pixel values
    values = grayscale_image[:,0].reshape(B, -1)
    
    # Stack to create BxNx3 tensor
    result = torch.stack((x_coords, y_coords, values), dim=-1)
    
    return result

def coordinates_array_to_image(coordinates_array, H, W, scale=1):
    """
    Convert an BxNx3 tensor with (x, y, grayscale_value) rows back into an Bx1xHxW grayscale image tensor.

    Parameters:
    coordinates_array (torch.Tensor): BxNx3 tensor with columns [x, y, grayscale_value].
    H (int): Height of the output image.
    W (int): Width of the output image.
    scale (int): Scale factor for the coordinates.

    Returns:
    torch.Tensor: Bx1xHxW grayscale image tensor.
    """
    # Initialize an empty HxW image
    B = coordinates_array.shape[0]
    image = torch.zeros((B, 1, H, W), dtype=coordinates_array[:, 2].dtype).to(coordinates_array.device)
    
    # Extract x, y coordinates and grayscale values
    x_coords = (coordinates_array[0, :, 0] * scale).long() 
    y_coords = (coordinates_array[0, :, 1] * scale).long()
    values = coordinates_array[:, :, 2]
    
    # Assign values to the appropriate locations in the image
    image[:, 0, y_coords, x_coords] = values
    
    return image

def normalize(x, eps=1e-4):
    if not isinstance(x, list):
        x = [x]

    prev_min = None
    prev_max = None
        
    for i in range(len(x)):
        _min = -F.max_pool2d(-x[i], (x[i].size(2), x[i].size(3)), stride=1, padding=0).detach()
        _min = _min if prev_min is None else torch.min(_min, prev_min)
        _max = F.max_pool2d(x[i], (x[i].size(2), x[i].size(3)), stride=1, padding=0).detach()
        _max = _max if prev_max is None else torch.max(_max, prev_max)
        prev_min = _min
        prev_max = _max
        
    return [(_x-_min)/(_max-_min+eps) for _x in x]

def estimate_normals(depth, normal_gain):
    xy_gradients = -spatial_gradient(normal_gain*depth, mode='diff', order=1, normalized=False).squeeze(1) # B 2 H W
    normals = torch.cat([xy_gradients, torch.ones_like(xy_gradients[:,0:1])], 1) # B 3 H W
    normals = normals / torch.linalg.norm(normals, dim=1, keepdim=True)
    return normals

def estimate_normals_from_disparity(disparity_map):
    """
    Estimate surface normals from a disparity map.
    
    Args:
        disparity_map (torch.Tensor): Input disparity map of shape (B, 1, H, W).
        
    Returns:
        normals (torch.Tensor): Output surface normals of shape (B, 3, H, W).
    """
    # Compute gradients in the x and y directions
    dx = F.pad(disparity_map[:, :, :, 1:] - disparity_map[:, :, :, :-1], (1, 0, 0, 0), mode='replicate')
    dy = F.pad(disparity_map[:, :, 1:, :] - disparity_map[:, :, :-1, :], (0, 0, 1, 0), mode='replicate')
    
    # Z component of the normal is -1 (assuming view vector along z-axis)
    dz = torch.ones_like(disparity_map) * -1

    # Stack gradients to form the normal vectors
    normals = torch.cat((dx, dy, dz), dim=1)  # Concatenate along the channel dimension

    # Normalize the normals
    norm = torch.norm(normals, dim=1, keepdim=True)
    normals = normals / (norm + 1e-8)  # Small epsilon to avoid division by zero

    return normals


def weighted_lsq(mono, stereo, conf, min_disp=0.5):
    B, _, _, _ = mono.shape
    # Weighted LSQ
    mono, stereo, conf = mono.reshape(B, -1), stereo.reshape(B, -1), conf.reshape(B, -1)

    stereo = F.relu(stereo)

    scale_shift = torch.zeros((B, 2), device=mono.device)

    for b in range(B):
        _mono = mono[b].unsqueeze(0)
        _stereo = stereo[b].unsqueeze(0)
        _conf = conf[b].unsqueeze(0)

        _min_mask = _stereo > min_disp
        _mono = _mono[_min_mask].unsqueeze(0)
        _conf = _conf[_min_mask].unsqueeze(0)
        _stereo = _stereo[_min_mask].unsqueeze(0)
        
        weights = torch.sqrt(_conf+1e-3)
        A_matrix = _mono * weights
        A_matrix = torch.cat([A_matrix.unsqueeze(-1), weights.unsqueeze(-1)], -1)
        B_matrix = (_stereo * weights).unsqueeze(-1)

        _scale_shift = torch.linalg.lstsq(A_matrix, B_matrix)[0].squeeze(2) # 1 x 2 x 1 -> 1 x 2,
        scale_shift[b] = _scale_shift.squeeze(0)

    return scale_shift[:, 0:1].reshape(B,1,1,1), scale_shift[:, 1:2].reshape(B,1,1,1)


def naive_scale_shift(mde, disp, conf, conf_th = 0.5):
    B, _, _, _ = mde.shape

    scale_and_shift_values = torch.zeros((B, 2), device=mde.device)

    for b in range(B):
        _mde = mde[b].unsqueeze(0)
        _disp = disp[b].unsqueeze(0)
        _conf = conf[b].unsqueeze(0)

        _mde = _mde[_conf > conf_th].unsqueeze(0)
        _disp = _disp[_conf > conf_th].unsqueeze(0)

        mde_90_percentile = torch.quantile(_mde, 0.9)
        mde_median = torch.median(_mde)

        disp_90_percentile = torch.quantile(_disp, 0.9)
        disp_median = torch.median(_disp)

        scale = (disp_90_percentile - disp_median) / (mde_90_percentile - mde_median)
        shift = disp_median - scale * mde_median

        scale_and_shift_values[b] = torch.tensor([scale, shift], device=mde.device)
    
    return scale_and_shift_values[:, 0:1].reshape(B,1,1,1), scale_and_shift_values[:, 1:2].reshape(B,1,1,1)
