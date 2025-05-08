import random
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms

IGNORE_INDEX = 255

class Normalize(object):
    """Normalize a PIL image (H×W×C) to float32 tensor in CHW order."""
    def __init__(self, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
        self.mean = mean
        self.std  = std

    def __call__(self, sample):
        img, mask = sample['image'], sample['label']
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return {'image': img, 'label': np.array(mask, dtype=np.int64)}


class ToTensor(object):
    """Convert numpy arrays to torch Tensors (C×H×W for image, H×W for mask)."""
    def __call__(self, sample):
        img, mask = sample['image'], sample['label']
        # image: H×W×C -> C×H×W
        img = torch.from_numpy(img.transpose(2,0,1)).float()
        mask = torch.from_numpy(mask).long()
        return {'image': img, 'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img, mask = sample['image'], sample['label']
        if random.random() < 0.5:
            img  = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image': img, 'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img, mask = sample['image'], sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return {'image': img, 'label': mask}


class RandomScaleCrop(object):
    """Randomly scale (0.5×–2.0×) then crop to fixed size, padding mask with IGNORE_INDEX."""
    def __init__(self, base_size, crop_size, fill=IGNORE_INDEX):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill      = fill

    def __call__(self, sample):
        img, mask = sample['image'], sample['label']
        short = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        w, h  = img.size
        if h > w:
            ow = short
            oh = int(short * h / w)
        else:
            oh = short
            ow = int(short * w / h)
        img  = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad if needed
        if short < self.crop_size:
            padw = max(0, self.crop_size - ow)
            padh = max(0, self.crop_size - oh)
            img  = ImageOps.expand(img, border=(0,0,padw,padh), fill=0)
            mask = ImageOps.expand(mask, border=(0,0,padw,padh), fill=self.fill)

        # random crop
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img  = img.crop((x1, y1, x1+self.crop_size, y1+self.crop_size))
        mask = mask.crop((x1, y1, x1+self.crop_size, y1+self.crop_size))

        return {'image': img, 'label': mask}


class FixedResize(object):
    """Resize both image and mask to given size. Mask uses NEAREST, image BILINEAR."""
    def __init__(self, size):
        self.resize_img  = transforms.Resize(size,   Image.BILINEAR)
        self.resize_mask = transforms.Resize(size,   Image.NEAREST)

    def __call__(self, sample):
        img, mask = sample['image'], sample['label']
        img  = self.resize_img(img)
        mask = self.resize_mask(mask)
        return {'image': img, 'label': mask}
