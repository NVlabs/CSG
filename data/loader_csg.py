# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.)

import warnings
from PIL import ImageFilter, Image
import math
import random
import torch
from torchvision.transforms import functional as F

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class RandomResizedCrop_two(object):
    # generate two closely located patches
    """Crop the given PIL Image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio, augment=(0.025, 0.075)):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                # return i, j, h, w
                ##### augment #####
                delta = random.randint(int(h*augment[0]), int(h*augment[1])) * random.choice([-1, 1])
                h_a = h + delta; h_a = min(max(1, h_a), height)
                delta = random.randint(int(w*augment[0]), int(w*augment[1])) * random.choice([-1, 1])
                w_a = w + delta; w_a = min(max(1, w_a), width)
                delta = random.randint(int(h*augment[0]), int(h*augment[1])) * random.choice([-1, 1])
                i_a = i + delta; i_a = min(max(0, i_a), height - h_a)
                delta = random.randint(int(w*augment[0]), int(w*augment[1])) * random.choice([-1, 1])
                j_a = j + delta; j_a = min(max(0, j_a), width - w_a)
                ###################
                return i, j, h, w, i_a, j_a, h_a, w_a

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        ##### augment #####
        delta = random.randint(int(h*augment[0]), int(h*augment[1])) * random.choice([-1, 1])
        h_a = h + delta; h_a = min(max(1, h_a), height)
        delta = random.randint(int(w*augment[0]), int(w*augment[1])) * random.choice([-1, 1])
        w_a = w + delta; w_a = min(max(1, w_a), width)
        delta = random.randint(int(h*augment[0]), int(h*augment[1])) * random.choice([-1, 1])
        i_a = i + delta; i_a = min(max(0, i_a), height - h_a)
        delta = random.randint(int(w*augment[0]), int(w*augment[1])) * random.choice([-1, 1])
        j_a = j + delta; j_a = min(max(0, j_a), width - w_a)
        ###################
        return i, j, h, w, i_a, j_a, h_a, w_a

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w, i_a, j_a, h_a, w_a = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), F.resized_crop(img, i_a, j_a, h_a, w_a, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class ImageTransform:
    """return both image and tensor"""

    def __init__(self, transform):
        self.base_transform = transform[0] # resize, centercrop
        self.totensor_norm = transform[1] # totensor, **normalize**

    def __call__(self, x):
        image = self.base_transform(x)
        tensor = self.totensor_norm(image)
        return [tensor, F.to_tensor(image)]


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, q_transform, k_transform):
        self.q_transform = q_transform
        self.k_transform = k_transform

    def __call__(self, x):
        q = self.q_transform(x)
        k = self.k_transform(x)
        return [q, k]


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
