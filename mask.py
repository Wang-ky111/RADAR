import torch
import math
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw
import numpy as np
import torch.fft as fft
import torch.nn.functional as F
import random
import os

from dataset import *

from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from imageio import imsave


class PatchMaskGenerator:
    def __init__(self, ratio: float = 0.3) -> None:
        self.ratio = ratio

    def transform(self, image: Image.Image) -> Image.Image:
        # Get the height and width of the image
        width, height = image.size

        # Compute the patch size
        patch_size = 16
        while height % patch_size != 0 or width % patch_size != 0:
            patch_size -= 1

        # Compute the number of patches
        num_patches = (height * width) // (patch_size * patch_size)

        # Compute the number of patches to mask
        mask_patches = int(np.ceil(num_patches * self.ratio))

        # Create a mask of ones
        mask = Image.new("L", (width, height), color=255)
        draw = ImageDraw.Draw(mask)

        # Randomly select patches to mask
        mask_patch_indices = random.sample(range(num_patches), mask_patches)

        for index in mask_patch_indices:
            start_y = (index // (width // patch_size)) * patch_size
            start_x = (index % (width // patch_size)) * patch_size
            draw.rectangle([start_x, start_y, start_x + patch_size, start_y + patch_size], fill=0)

        # Convert both image and mask to numpy arrays
        image_np = np.array(image)
        mask_np = np.array(mask) / 255.0  # Normalize to [0, 1]

        # If the image is a 3-channel image, repeat the mask for all channels
        if len(image_np.shape) == 3:
            mask_np = np.expand_dims(mask_np, axis=-1)
            mask_np = np.repeat(mask_np, image_np.shape[-1], axis=-1)

        # Apply the mask
        masked_image_np = image_np * mask_np

        # Convert the numpy array back to a PIL Image
        masked_image = Image.fromarray(np.uint8(masked_image_np))

        return masked_image


class PixelMaskGenerator:
    def __init__(self, ratio: float = 0.6) -> None:
        self.ratio = ratio

    def transform(self, pil_image):
        # Convert PIL image to numpy array
        image = np.array(pil_image)

        # Infer the height and width from the image
        height, width, channels = image.shape

        pixel_count = height * width
        mask_count = int(np.ceil(pixel_count * self.ratio))

        # Generate random mask
        mask_idx = np.random.permutation(pixel_count)[:mask_count]
        mask = np.ones(pixel_count, dtype=np.float32)  # Initialize mask as ones
        mask[mask_idx] = 0  # Set selected indices to zero

        mask = mask.reshape((height, width))

        # Repeat the mask for all channels
        mask = np.repeat(mask[:, :, np.newaxis], channels, axis=2)

        masked_image = image * mask

        # Convert numpy array back to PIL image
        masked_pil_image = Image.fromarray(np.uint8(masked_image))

        return masked_pil_image


class FrequencyMaskGenerator:
    def __init__(self, ratio: float = 0.3, band: str = 'all') -> None:
        self.ratio = ratio
        self.band = band  # 'low', 'mid', 'high', 'all'
        self.low_weight = 0.5
        self.mid_weight = 0.25
        self.high_weight = 0.25

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.transform(image)

    def transform(self, image: Image.Image) -> Image.Image:
        image_array = np.array(image).astype(np.complex64)
        freq_image = np.fft.fftn(image_array, axes=(0, 1))

        height, width, _ = image_array.shape

        mask = self._create_balanced_mask(height, width)
        self.masked_freq_image = freq_image * mask
        masked_image_array = np.fft.ifftn(self.masked_freq_image, axes=(0, 1)).real

        masked_image_array = np.clip(masked_image_array, 0, 255)

        masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
        return masked_image

    def _sample_from_region(self, mask, y_start, y_end, x_start, x_end, num_frequencies):
        region_h = y_end - y_start
        region_w = x_end - x_start
        region_size = region_h * region_w

        if region_size <= 0 or num_frequencies <= 0:
            return mask

        num_frequencies = min(num_frequencies, region_size)

        mask_frequencies_indices = np.random.permutation(region_size)[:num_frequencies]
        y_indices = mask_frequencies_indices // region_w + y_start
        x_indices = mask_frequencies_indices % region_w + x_start

        mask[y_indices, x_indices, :] = 0
        return mask

    def _create_balanced_mask(self, height, width):
        mask = np.ones((height, width, 3), dtype=np.complex64)

        if self.band == 'low':
            y_start, y_end = 0, height // 4
            x_start, x_end = 0, width // 4
            num_frequencies = int(np.ceil((y_end - y_start) * (x_end - x_start) * self.ratio))
            mask = self._sample_from_region(mask, y_start, y_end, x_start, x_end, num_frequencies)

        elif self.band == 'mid':
            y_start, y_end = height // 4, 3 * height // 4
            x_start, x_end = width // 4, 3 * width // 4
            num_frequencies = int(np.ceil((y_end - y_start) * (x_end - x_start) * self.ratio))
            mask = self._sample_from_region(mask, y_start, y_end, x_start, x_end, num_frequencies)

        elif self.band == 'high':
            y_start, y_end = 3 * height // 4, height
            x_start, x_end = 3 * width // 4, width
            num_frequencies = int(np.ceil((y_end - y_start) * (x_end - x_start) * self.ratio))
            mask = self._sample_from_region(mask, y_start, y_end, x_start, x_end, num_frequencies)

        elif self.band == 'all':
            total_frequencies = int(np.ceil(height * width * self.ratio))

            low_num = int(round(total_frequencies * self.low_weight))
            mid_num = int(round(total_frequencies * self.mid_weight))
            high_num = total_frequencies - low_num - mid_num  

            mask = self._sample_from_region(
                mask,
                0, height // 4,
                0, width // 4,
                low_num
            )

            mask = self._sample_from_region(
                mask,
                height // 4, 3 * height // 4,
                width // 4, 3 * width // 4,
                mid_num
            )

            mask = self._sample_from_region(
                mask,
                3 * height // 4, height,
                3 * width // 4, width,
                high_num
            )

        else:
            raise ValueError(f"Invalid band: {self.band}")

        return mask


def test_mask_generator(
    image_path,
    mask_type,
    ratio=0.13,
    sample_size=20
    ):

    # Create a MaskGenerator
    if mask_type == 'spectral':
        mask_generator = FrequencyMaskGenerator(ratio=ratio, band='all')
    elif mask_type == 'pixel':
        mask_generator = PixelMaskGenerator(ratio=ratio)
    elif mask_type == 'patch':
        mask_generator = PatchMaskGenerator(ratio=ratio)
    else:
        mask_generator = None

    transform = transforms.Compose([
        transforms.Lambda(lambda img: mask_generator.transform(img)),
        transforms.ToTensor(),
    ])

    # data = ForenSynths(image_path, transform=transform)
    data = Wang_CVPR20(image_path, transform=transform)
    dataloader = DataLoader(data, batch_size=32, shuffle=False)

    # Access the first image and label directly
    image, label = data[1]
    image_to_save = image

    # Convert the tensor image to NumPy and transpose if necessary
    image_to_save = image_to_save.numpy().transpose(1, 2, 0)

    # Clip the values to the range [0, 1] if the image is in float format
    if image_to_save.dtype == np.float32 or image_to_save.dtype == np.float64:
        image_to_save = np.clip(image_to_save, 0, 1)

    sample_path = f'./samples'
    os.makedirs(sample_path, exist_ok=True)

    # Save the image using imageio's imsave
    imsave(f"{sample_path}/masked_{mask_type}.jpg", (image_to_save * 255).astype(np.uint8))

