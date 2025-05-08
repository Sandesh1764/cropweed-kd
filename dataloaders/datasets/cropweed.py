import os
import numpy as np
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders.custom_transforms import (
    RandomHorizontalFlip,
    RandomScaleCrop,
    RandomGaussianBlur,
    FixedResize,
    Normalize,
    ToTensor,
    IGNORE_INDEX
)
from glob import glob


class CropWeedSegmentation(data.Dataset):
    """
    Supports both split-based and flat directory layouts:

    Layout A (with splits):
      root/
        images/
          train/*.png or .jpg
          val  /*.png or .jpg
          test /*.png or .jpg
        masks/ or labels/
          train/*.png
          val  /*.png
          test /*.png

    Layout B (flat):
      root/
        images/*.png or .jpg
        labels/*.png

    """
    NUM_CLASSES = 3
    IGNORE_INDEX = IGNORE_INDEX

    def __init__(self, args, root=None, split="train"):
        if root is None:
            root = Path.db_root_dir('cropweed')
        self.root = root
        self.split = split  # e.g., 'train', 'val', 'test'
        self.args = args

        # Determine images directory
        split_img_dir = os.path.join(self.root, 'images', split)
        flat_img_dir = os.path.join(self.root, 'images')
        if os.path.isdir(split_img_dir):
            self.images_dir = split_img_dir
        elif os.path.isdir(flat_img_dir):
            self.images_dir = flat_img_dir
        else:
            raise RuntimeError(f"Images directory not found in {split_img_dir} or {flat_img_dir}")

        # Determine masks directory (try 'labels' then 'masks')
        # priority: split labels -> split masks -> flat labels -> flat masks
        candidates = [
            os.path.join(self.root, 'labels', split),
            os.path.join(self.root, 'masks', split),
            os.path.join(self.root, 'labels'),
            os.path.join(self.root, 'masks')
        ]
        self.masks_dir = None
        for d in candidates:
            if os.path.isdir(d):
                self.masks_dir = d
                break
        if self.masks_dir is None:
            raise RuntimeError(f"Masks directory not found in one of: {candidates}")

        # Gather all image file paths
        patterns = ['*.png', '*.jpg', '*.jpeg']
        image_paths = []
        for pat in patterns:
            image_paths.extend(glob(os.path.join(self.images_dir, pat)))
        image_paths = sorted(image_paths)

        # Pair images with masks by matching basename
        self.paired = []
        for img_path in image_paths:
            base = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(self.masks_dir, base + '.png')
            if os.path.exists(mask_path):
                self.paired.append((img_path, mask_path))

        if not self.paired:
            raise RuntimeError(
                f"No image-mask pairs found in images: {self.images_dir} and masks: {self.masks_dir}"
            )

        print(f"Found {len(self.paired)} '{split}' samples in CropWeedDataset")

    def __len__(self):
        return len(self.paired)

    def __getitem__(self, idx):
        img_path, mask_path = self.paired[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        arr = np.array(mask, dtype=np.uint8)
        valid = (arr == 0) | (arr == 1) | (arr == 2)
        arr[~valid] = self.IGNORE_INDEX
        mask = Image.fromarray(arr)

        sample = {"image": img, "label": mask}
        if self.split == 'train':
            return self.transform_tr(sample)
        else:
            return self.transform_val(sample)

    def transform_tr(self, sample):
        trf = transforms.Compose([
            RandomHorizontalFlip(),
            RandomScaleCrop(
                base_size=self.args.base_size,
                crop_size=self.args.crop_size,
                fill=self.IGNORE_INDEX
            ),
            RandomGaussianBlur(),
            Normalize(),
            ToTensor()
        ])
        return trf(sample)

    def transform_val(self, sample):
        trf = transforms.Compose([
            FixedResize(size=self.args.crop_size),
            Normalize(),
            ToTensor()
        ])
        return trf(sample)
