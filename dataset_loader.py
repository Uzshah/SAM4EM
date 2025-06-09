import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
import tqdm
import torchvision.transforms as transforms
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from PIL import Image, ImageEnhance
import torch.nn.functional as F
import h5py
import re
from torchvision.transforms import ToTensor, Normalize, Resize

# Augmentation Functions
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def random_crop(image, label):
    min_ratio = 0.2
    max_ratio = 0.8
    w, h = image.shape
    ratio = random.random()
    scale = min_ratio + ratio * (max_ratio - min_ratio)
    new_h = int(h * scale)
    new_w = int(w * scale)
    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)
    image = image[x:x+new_w, y:y+new_h]
    label = label[x:x+new_w, y:y+new_h]
    return image, label

def random_scale(image, label, scale_factor=0.6):
    min_ratio = 0.2
    max_ratio = 0.8
    w, h = image.shape
    ratio = random.random()
    scale = min_ratio + ratio * (max_ratio - min_ratio)
    new_h = int(h * scale)
    new_w = int(w * scale)
    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)
    image = image[x:x+new_w, y:y+new_h]
    label = label[x:x+new_w, y:y+new_h]
    return image, label

def random_elastic(image, label, alpha, sigma, alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    shape_size = shape[:2]
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    imageB = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    labelB = cv2.warpAffine(label, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    imageC = map_coordinates(imageB, indices, order=1, mode='constant').reshape(shape)
    labelC = map_coordinates(labelB, indices, order=1, mode='constant').reshape(shape)
    return imageC, labelC

def random_gaussian(image, var=0.1):
    noise = np.random.normal(0, var, image.shape)
    image = image + noise
    return image

def random_gaussian_filter(image, K_size=3, sigma=1.3):
    img = np.asarray(np.uint8(image))
    H, W = img.shape
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(float)
    K = np.zeros((K_size, K_size), dtype=float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma) 
    K /= K.sum()
    tmp = out.copy()
    for y in range(H):
       for x in range(W):
            out[pad + y, pad + x] = np.sum(K * tmp[y: y + K_size, x: x + K_size])
    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    out = out.astype(np.float32)
    return out

def random_affine(image, label, degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10):
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), random.uniform(-degrees, degrees), random.uniform(*scale))
    image = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    label = cv2.warpAffine(label, M, (cols, rows), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return image, label

def random_enhance(image):
    pil_img = Image.fromarray(np.uint8(image))
    if random.random() > 0.5:
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(random.uniform(0.8, 1.5))
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(random.uniform(0.8, 1.5))
    if random.random() > 0.5:
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(random.uniform(0.8, 1.5))
    image = np.array(pil_img).astype(np.float32) 
    return image



def method_I_gray(image, mask):
    # Random horizontal flip
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    
    # Random vertical flip
    if np.random.rand() > 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    
    # Random rotation (0 to 360 degrees)
    angle = np.random.uniform(0, 360)
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(mask, rot_mat, mask.shape[1::-1], flags=cv2.INTER_NEAREST)
    
    # Random scaling
    scale = np.random.uniform(0.8, 1.2)
    image = cv2.resize(image, None, fx=scale, fy=scale)
    mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    
    # Gaussian blur (only applied to image)
    # if np.random.rand() > 0.5:
    #     ksize = np.random.choice([3, 5, 7])
    #     image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    
    # Brightness adjustment (only applied to image)
    if np.random.rand() > 0.5:
        image = cv2.convertScaleAbs(image, alpha=np.random.uniform(0.8, 1.2))
    
    # Mirroring
    if np.random.rand() > 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)
    
    return image, mask

def method_III_gray(image, mask):
    # Resized crop
    crop_size = np.random.uniform(0.6, 1.0)
    h, w = image.shape
    new_h, new_w = int(h * crop_size), int(w * crop_size)
    image = cv2.resize(image, (new_w, new_h))
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Horizontal flip
    image, mask = method_I_gray(image, mask)  # Includes horizontal flip and other augmentations
    
    # Autocontrast
    if np.random.rand() > 0.5:
        image = cv2.equalizeHist(image)
    
    # Invert
    if np.random.rand() > 0.5:
        image = cv2.bitwise_not(image)
    
    # Rotate
    image, mask = method_I_gray(image, mask)  # Already includes rotation
    
    # Posterize (reduce bit depth)
    if np.random.rand() > 0.5:
        image = np.bitwise_and(image, np.random.randint(0, 255))
    
    # Solarize (invert intensities above a threshold)
    if np.random.rand() > 0.5:
        image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)[1]
    
    return image, mask

import numpy as np
import torchvision.transforms as T
from imgaug import augmenters as iaa
import PIL.Image as Image

# Define Elastic Transformation with desired parameters
elastic_transform = iaa.ElasticTransformation(alpha=34, sigma=4)
# Random Generator Class with Augmentations
class RandomGenerator(object):
    def __init__(self, output_size, low_res, mod=True, norm_type='balanced'):
        """
        Args:
            output_size (list): Target output size [H, W]
            low_res (list): Low resolution size [H, W]
            mod (bool): Whether to apply modifications/augmentations
            norm_type (str): 
                - 'balanced': Balanced normalization for EM (recommended)
                - 'simple': Just divide by 255
                - 'minmax': Min-max scaling to [0,1]
                - 'standard': Zero mean, unit variance
        """
        self.output_size = output_size
        self.low_res = low_res
        self.mod = mod
        self.norm_type = norm_type
        
        # Default values that work well for EM images
        if norm_type == 'balanced':
            # These values tend to work well for EM images
            self.mean = [0.485, 0.485, 0.485]  # Single value repeated for consistency
            self.std = [0.229, 0.229, 0.229]   # Single value repeated for consistency
            self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        else:
            self.normalize = None
            
    def normalize_image(self, image):
        """
        Normalize image based on selected method
        """
        image = image.astype(np.float32)
        
        if self.norm_type == 'simple':
            # Simple division by 255
            image = image / 255.0
            
        elif self.norm_type == 'minmax':
            # Min-max scaling to [0,1] range
            image_min = np.min(image)
            image_max = np.max(image)
            if image_max > image_min:
                image = (image - image_min) / (image_max - image_min)
            else:
                image = image / 255.0
                
        elif self.norm_type == 'standard':
            # Zero mean, unit variance
            mean = np.mean(image)
            std = np.std(image)
            if std > 0:
                image = (image - mean) / std
            else:
                image = image / 255.0
                
        elif self.norm_type == 'balanced':
            # Balanced normalization for EM images
            image = image / 255.0  # First scale to [0,1]
            image = torch.from_numpy(image).unsqueeze(0)
            image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
            image = self.normalize(image)
            
        return image

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Apply augmentations if enabled
        if self.mod:
            # if random.random() > 0.55:
            #     image, label = random_rotate(image, label)
            # elif random.random() > 0.65:
            #     image, label = random_scale(image, label)
            if random.random() > 0.35:
                image, label = random_elastic(image, label, image.shape[1] * 2,
                                          image.shape[1] * 0.08,
                                          image.shape[1] * 0.08)
            # if np.random.rand() > 0.6:
            #     image = np.fliplr(image)
            #     label = np.fliplr(label)
            # elif np.random.rand() > 0.7:
            #     image = np.flipud(image)
            #     label = np.flipud(label)
            # if random.random() > 0.55:
            #     image, label = method_I_gray(image, label)
            elif random.random() > 0.55:
                image = random_enhance(image)
            elif random.random() > 0.65:
                image, label = motion_blur(image, label)

        # Ensure consistent size
        image = cv2.resize(image, (self.output_size[0], self.output_size[1]))
        label = cv2.resize(label, (self.output_size[0], self.output_size[1]), 
                          interpolation=cv2.INTER_NEAREST)

        # Apply normalization
        image = self.normalize_image(image)
        
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).unsqueeze(0)
            image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)

        # Create low resolution label
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        
        # Convert labels to tensors
        label = torch.from_numpy(label.astype(np.float32)).long()
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32)).long()
        
        # Prepare final sample
        sample = {
            'image': image,
            'label': label,
            'low_res_label': low_res_label,
            'case_name': sample['case_name']
        }
        
        return sample



def motion_blur(image, mask, kernel_size=11, p=0.5):
    """
    Applies motion blur to a 2D image and returns the motion-blurred image along with the original mask.
    
    Args:
        image (numpy array): Input 2D image (H, W).
        mask (numpy array): Input binary mask (H, W) where 1 represents the foreground.
        kernel_size (int): Kernel size for motion blur. Default is 11.
        p (float): Probability of applying the augmentation. Default is 0.5.
        
    Returns:
        tuple: (blurred_image, original_mask)
    """
    # if random.random() > p:
    #     # Return the original image and mask if augmentation is not applied
    #     return image, mask

    # Generate the motion blur kernel
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    if random.random() > 0.5:  # Horizontal blur
        kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    else:  # Vertical blur
        kernel_motion_blur[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel_motion_blur /= kernel_size

    # Apply the motion blur kernel to the image
    blurred_image = cv2.filter2D(image, -1, kernel_motion_blur)
    
    # Return the blurred image and the original mask
    return blurred_image, mask
    
def sort_labels_left_to_right_top_to_bottom(instance_mask):
    num_labels = instance_mask.max() + 1
    bounding_boxes = []

    for label in range(1, num_labels):
        mask = (instance_mask == label).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(mask)
        bounding_boxes.append((x, y, w, h, label))

    # Sort first by x-coordinate (left-to-right), then by y-coordinate (top-to-bottom)
    bounding_boxes.sort(key=lambda box: (box[0], box[1], box[2], box[3]))
    
    # Create a new mask where labels are sorted by x and y coordinates
    sorted_instance_mask = np.zeros_like(instance_mask)
    for new_label, (_, _, _, _, old_label) in enumerate(bounding_boxes, start=1):
        sorted_instance_mask[instance_mask == old_label] = new_label

    return sorted_instance_mask


class EM_dataset(Dataset):
    def __init__(self, base_dir, img_dir, label_dir, img_size, transform=None, list_folders=['mouse1'], 
                 is_training=True, mask_value=0, calc_stats=False, evals = False):
        """
        Enhanced EM dataset class with statistics calculation capability.
        
        Args:
            base_dir (str): Base directory containing the data
            img_dir (str): Directory containing images
            label_dir (str): Directory containing labels
            img_size (int): Target image size
            transform: Data augmentation transforms
            list_folders (list): List of folder names to include
            is_training (bool): Whether this is a training dataset
            mask_value (int): If >0, converts multiclass mask to binary by setting specified value to 1
            calc_stats (bool): Whether to calculate dataset statistics on initialization
        """
        self.transform = transform
        self.img_size = img_size
        self.mask_value = mask_value
        self.evals = evals
        # Create sample list from provided folders
        if evals:
            sample_list = []
            for mouse in list_folders:
                rgb_path = os.path.join(base_dir, mouse, img_dir)
                # semantic_path = os.path.join(base_dir, mouse, label_dir)
                rgb_files = sorted(os.listdir(rgb_path))
                # print(rgb_files)
                # semantic_files = sorted(os.listdir(semantic_path))
                sample_list.extend([
                    (os.path.join(rgb_path, f_rgb))
                    for f_rgb in rgb_files
                ])
                # print(sample_list)
            # Handle training vs validation sets
            # self.sample_list = sample_list if not is_training else sample_list
            self.sample_list = sorted(sample_list, key=sort_key_mito)
            # self.sample_list = self.sample_list
            # print(self.sample_list)
        else:
            sample_list = []
            for mouse in list_folders:
                rgb_path = os.path.join(base_dir, mouse, img_dir)
                semantic_path = os.path.join(base_dir, mouse, label_dir)
                rgb_files = sorted(os.listdir(rgb_path))
                semantic_files = sorted(os.listdir(semantic_path))
                sample_list.extend([
                    (os.path.join(rgb_path, f_rgb), os.path.join(semantic_path, f_semantic))
                    for f_rgb, f_semantic in zip(rgb_files, semantic_files)
                ])
                
            # Handle training vs validation sets
            self.sample_list = sample_list 
            if is_training:
                self.sample_list = sorted(self.sample_list)#, key=sort_key_mito)
            else:
                self.sample_list = sorted(self.sample_list)#, key=sort_key_mito)
                # print(sample_list[:500])
        

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # Load image and mask
        # image = cv2.imread(self.sample_list[idx], cv2.IMREAD_GRAYSCALE)
        # print(self.sample_list[idx])
        if not self.evals:
            instance_mask = cv2.imread(self.sample_list[idx][1], 0)
            image = cv2.imread(self.sample_list[idx][0], cv2.IMREAD_GRAYSCALE)
            
            # Process the mask based on mask_value
            if self.mask_value > 0:
                # Convert multiclass to binary by selecting specific class
                label = (instance_mask == int(self.mask_value)).astype(np.uint8)
            else:
                # Treat as binary mask where any positive value is foreground
                label = np.zeros_like(instance_mask)
                label[instance_mask > 0] = 1
        else:
            # print(self.sample_list[idx][0])
            image = cv2.imread(self.sample_list[idx], cv2.IMREAD_GRAYSCALE)
        
        # Resize both image and label to target size
        image = cv2.resize(image, (self.img_size, self.img_size))
        if not self.evals:
            label = cv2.resize(label, (self.img_size, self.img_size), 
                          interpolation=cv2.INTER_NEAREST)
        
        # Prepare sample dictionary
        
        if not self.evals:
            case_name = os.path.basename(self.sample_list[idx][1])
            sample = {'image': image, 'label': label, 'case_name': case_name}
        else:
            case_name = os.path.basename(self.sample_list[idx])
            sample = {'image': image, 'label': image, 'case_name': case_name}
        # Apply transforms if specified
        if self.transform:
            sample = self.transform(sample)
            
        return sample

def sort_key_mito(path_tuple):
    """Sort key for paired image and label paths"""
    # Get filename from image path (first element of tuple)
    filename = path_tuple[0].split('/')[-1]
    
    # Extract numbers from filename (e.g., 'im0400_patch_10_0.png')
    slice_num = int(re.search(r'im(\d+)', filename).group(1))
    patch_nums = re.search(r'patch_(\d+)_(\d+)', filename)
    i, j = map(int, patch_nums.groups())
    
    # Sort by patch position first, then slice number
    return (i, j, slice_num)
# Example usage function
def calculate_and_setup_dataset(base_dir, img_dir, label_dir, img_size, list_folders=['mouse1']):
    """
    Helper function to calculate statistics and set up the dataset properly.
    """
    # First, create a dataset instance to calculate statistics
    initial_dataset = EM_dataset(
        base_dir=base_dir,
        img_dir=img_dir,
        label_dir=label_dir,
        img_size=img_size,
        transform=None,  # No transform for statistics calculation
        list_folders=list_folders,
        calc_stats=True  # Calculate statistics
    )
    
    # Get the calculated statistics
    means, stds = initial_dataset.get_stats_for_channels()
    
    # Create transform with the calculated statistics
    transform = RandomGenerator(
        output_size=[img_size, img_size],
        low_res=[img_size//4, img_size//4],
        mean=means,
        std=stds
    )
    
    # Create the final dataset with the properly configured transform
    final_dataset = EM_dataset(
        base_dir=base_dir,
        img_dir=img_dir,
        label_dir=label_dir,
        img_size=img_size,
        transform=transform,
        list_folders=list_folders
    )
    
    return final_dataset, means, stds


def sort_key2(path):
    # Extract the base filename from the full path
    filename = path.split('/')[-1]
    
    # Extract main number and sub number (e.g., from '185_3.png')
    main_num = re.search(r'(\d+)_', filename)  # Matches '185' in '185_3.png'
    sub_num = re.search(r'_(\d+)\.', filename)  # Matches '3' in '185_3.png'
    
    # Convert both numbers to integers
    main_num = int(main_num.group(1)) if main_num else 0
    sub_num = int(sub_num.group(1)) if sub_num else 0
    
    # Return a tuple that will sort first by sub_num, then by main_num
    return (sub_num, main_num)
# Sorting function
def sort_key(pair):
    # Extract the number from the first element of the tuple (image name)
    match = re.search(r'(\d+)', pair[0])
    return int(match.group(1)) if match else 0  # Convert extracted number to int for sorting
    
from skimage import img_as_float

def anisotropic_diffusion(img, niter=10, kappa=50, gamma=0.1, option=2):
    """
    Perform anisotropic diffusion (Perona-Malik filter) on an image.
    
    Parameters:
    - img: Input image (grayscale).
    - niter: Number of iterations.
    - kappa: Conduction coefficient, controls sensitivity to edges.
    - gamma: Time step between iterations.
    - option: 1 or 2, selecting between two diffusion functions.
    
    Returns:
    - diffused: The diffused image.
    """
    img = img_as_float(img)
    diffused = img.copy()
    
    for i in range(niter):
        # Compute gradients
        north = np.zeros_like(diffused)
        south = np.zeros_like(diffused)
        east = np.zeros_like(diffused)
        west = np.zeros_like(diffused)
        
        north[:-1, :] = diffused[1:, :] - diffused[:-1, :]
        south[1:, :] = diffused[:-1, :] - diffused[1:, :]
        east[:, :-1] = diffused[:, 1:] - diffused[:, :-1]
        west[:, 1:] = diffused[:, :-1] - diffused[:, 1:]
        
        # Choose the diffusion function
        if option == 1:
            c_north = np.exp(-(north / kappa) ** 2)
            c_south = np.exp(-(south / kappa) ** 2)
            c_east = np.exp(-(east / kappa) ** 2)
            c_west = np.exp(-(west / kappa) ** 2)
        elif option == 2:
            c_north = 1.0 / (1.0 + (north / kappa) ** 2)
            c_south = 1.0 / (1.0 + (south / kappa) ** 2)
            c_east = 1.0 / (1.0 + (east / kappa) ** 2)
            c_west = 1.0 / (1.0 + (west / kappa) ** 2)
        
        # Update image
        diffused += gamma * (c_north * north + c_south * south + c_east * east + c_west * west)
    
    return diffused
# import numpy as np
# from skimage import img_as_float
# from numba import jit, cuda
# import math

# # CPU-optimized version using Numba
# @jit(nopython=True, parallel=True)
# def anisotropic_diffusion_numba(img, niter=10, kappa=50, gamma=0.1, option=2):
#     """
#     Optimized anisotropic diffusion using Numba.
#     """
#     rows, cols = img.shape
#     diffused = img.copy()
    
#     for _ in range(niter):
#         north = np.zeros_like(diffused)
#         south = np.zeros_like(diffused)
#         east = np.zeros_like(diffused)
#         west = np.zeros_like(diffused)
        
#         # Compute gradients more efficiently
#         north[:-1, :] = diffused[1:, :] - diffused[:-1, :]
#         south[1:, :] = diffused[:-1, :] - diffused[1:, :]
#         east[:, :-1] = diffused[:, 1:] - diffused[:, :-1]
#         west[:, 1:] = diffused[:, :-1] - diffused[:, 1:]
        
#         if option == 1:
#             # Vectorized exponential calculations
#             c_north = np.exp(-(north/kappa)**2)
#             c_south = np.exp(-(south/kappa)**2)
#             c_east = np.exp(-(east/kappa)**2)
#             c_west = np.exp(-(west/kappa)**2)
#         else:
#             # Vectorized rational calculations
#             c_north = 1.0 / (1.0 + (north/kappa)**2)
#             c_south = 1.0 / (1.0 + (south/kappa)**2)
#             c_east = 1.0 / (1.0 + (east/kappa)**2)
#             c_west = 1.0 / (1.0 + (west/kappa)**2)
        
#         # Update image in one operation
#         diffused += gamma * (c_north * north + c_south * south + 
#                            c_east * east + c_west * west)
    
#     return diffused

# # CUDA version for GPU acceleration
# @cuda.jit
# def anisotropic_diffusion_kernel(d_image, d_output, kappa, gamma, option):
#     """
#     CUDA kernel for anisotropic diffusion.
#     """
#     row, col = cuda.grid(2)
#     rows, cols = d_image.shape
    
#     if row < rows and col < cols:
#         # Compute local gradients
#         north = 0.0
#         south = 0.0
#         east = 0.0
#         west = 0.0
        
#         if row > 0:
#             north = d_image[row-1, col] - d_image[row, col]
#         if row < rows-1:
#             south = d_image[row+1, col] - d_image[row, col]
#         if col > 0:
#             west = d_image[row, col-1] - d_image[row, col]
#         if col < cols-1:
#             east = d_image[row, col+1] - d_image[row, col]
        
#         # Compute coefficients
#         if option == 1:
#             c_north = math.exp(-(north/kappa)**2)
#             c_south = math.exp(-(south/kappa)**2)
#             c_east = math.exp(-(east/kappa)**2)
#             c_west = math.exp(-(west/kappa)**2)
#         else:
#             c_north = 1.0 / (1.0 + (north/kappa)**2)
#             c_south = 1.0 / (1.0 + (south/kappa)**2)
#             c_east = 1.0 / (1.0 + (east/kappa)**2)
#             c_west = 1.0 / (1.0 + (west/kappa)**2)
        
#         # Update pixel
#         d_output[row, col] = d_image[row, col] + gamma * (
#             c_north * north + c_south * south + 
#             c_east * east + c_west * west
#         )

# def anisotropic_diffusion_gpu(img, niter=10, kappa=50, gamma=0.1, option=2):
#     """
#     GPU-accelerated anisotropic diffusion using CUDA.
#     """
#     img = img_as_float(img)
#     rows, cols = img.shape
    
#     # Configure CUDA grid
#     threadsperblock = (16, 16)
#     blockspergrid_x = math.ceil(rows / threadsperblock[0])
#     blockspergrid_y = math.ceil(cols / threadsperblock[1])
#     blockspergrid = (blockspergrid_x, blockspergrid_y)
    
#     # Allocate device memory
#     d_image = cuda.to_device(img)
#     d_output = cuda.device_array_like(img)
    
#     # Run iterations
#     for _ in range(niter):
#         anisotropic_diffusion_kernel[blockspergrid, threadsperblock](
#             d_image, d_output, kappa, gamma, option)
#         # Swap input and output
#         d_image, d_output = d_output, d_image
    
#     # Get result back to host
#     result = d_image.copy_to_host()
#     return result

# # Main function that chooses the best implementation
# def anisotropic_diffusion_optimized(img, niter=10, kappa=50, gamma=0.1, option=2, use_gpu=False):
#     """
#     Choose the best implementation based on available hardware and image size.
    
#     Parameters:
#     - img: Input image
#     - niter: Number of iterations
#     - kappa: Conduction coefficient
#     - gamma: Time step
#     - option: 1 or 2, selecting diffusion function
#     - use_gpu: Whether to use GPU acceleration if available
    
#     Returns:
#     - diffused: The diffused image
#     """
#     img = img_as_float(img)
    
#     try:
#         if use_gpu and cuda.is_available():
#             return anisotropic_diffusion_gpu(img, niter, kappa, gamma, option)
#         else:
#             return anisotropic_diffusion_numba(img, niter, kappa, gamma, option)
#     except:
#         # Fallback to CPU version if any issues occur
#         return anisotropic_diffusion_numba(img, niter, kappa, gamma, option)

# # Example usage:
# if __name__ == "__main__":
#     # Test with both CPU and GPU versions
#     img = np.random.rand(512, 512)
    
#     # CPU version
#     result_cpu = anisotropic_diffusion_optimized(img, use_gpu=False)
    
#     # GPU version if available
#     result_gpu = anisotropic_diffusion_optimized(img, use_gpu=True)

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, split, img_size, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join("/home/ushah/SegmentationBaseline/lists/lists_Synapse", self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # if self.split == "train":
        slice_name = self.sample_list[idx].strip('\n')
        data_path = os.path.join(self.data_dir, slice_name+'.npz')
        data = np.load(data_path)
        image, label = data['image'], data['label']
        # else:
        #     vol_name = self.sample_list[idx].strip('\n')
        #     filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
        #     data = h5py.File(filepath)
        #     image, label = data['image'][:], data['label'][:]

        # Input dim should be consistent
        # Since the channel dimension of nature image is 3, that of medical image should also be 3
        image = cv2.resize(image, (self.img_size, self.img_size))
        label = cv2.resize(label, (self.img_size, self.img_size))
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
        
# Example Usage
if __name__ == '__main__':
    transform = RandomGenerator(output_size=(256, 256), low_res=(64, 64), mod=True)
    dataset = EM_dataset(base_dir='/path/to/data', img_dir='images', label_dir='labels', img_size=256, transform=transform, list_folders=['mouse1'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # Example to show how data is loaded and augmented
    for i, data in enumerate(dataloader):
        images, labels, low_res_labels = data['image'], data['label'], data['low_res_label']
        print(images.shape, labels.shape, low_res_labels.shape)
        break
