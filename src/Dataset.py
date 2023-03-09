import numpy as np
import os
from tqdm.notebook import tqdm
from tqdm.contrib import tenumerate

import warnings

import torch
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, WeightedRandomSampler, default_collate
from torchvision import transforms

from utils import get_gpu_mem_usage, get_ram_usage



def _get_dataset_files(root_dir: str):
    img_path = os.path.join(root_dir, 'image')
    mask_path = os.path.join(root_dir, 'mask')

    images = os.listdir(img_path)
    masks = os.listdir(mask_path)

    images.sort(); masks.sort()

    items = []
    for it_im, it_ma in zip(images, masks):
        item = (os.path.join(img_path, it_im), os.path.join(mask_path, it_ma))
        items.append(item)

    return items


def data_augment_(data, n: int, load_on_gpu=False):
    if n <= 0:
        warnings.warn("Need a strictly positive integer for n for data augmentation. Will skip augmentation.")
        return None

    original_dataset_lenght = len(data)
    for idx in tqdm(range(original_dataset_lenght), desc="Expending the dataset {} more times".format(n)):
        img, mask, original_file = data[idx]
        for i_n in range(n):
            new_img = img.detach().clone()
            new_mask = mask.detach().clone()
            if load_on_gpu:
                new_img = new_img.cuda()
                new_mask = new_mask.cuda()

            random_mod = np.random.choice([0, 1], size=10, p=[.8, .2])    # Proba of 0.2 for each event = ~2 transformations

            # non-spacial variations (only on image)
            if random_mod[0]:     # Brightness shift
                new_img = transforms.ColorJitter(brightness=.5)(new_img)
            if random_mod[1]:     # Autocontrast
                new_img = transforms.functional.autocontrast(new_img)
            if random_mod[2]:     # Sharpness
                new_img = transforms.functional.adjust_sharpness(new_img, sharpness_factor=np.random.uniform(0, 2))
            if random_mod[3]:     # Gaussian Blur
                new_img = transforms.functional.gaussian_blur(new_img, kernel_size=5)

            # spacial variations (on image & mask)
            img_size = transforms.functional.get_image_size(new_img)[::-1]
            if random_mod[4]:     # Flip
                new_img = transforms.functional.vflip(new_img)
                new_mask = transforms.functional.vflip(new_mask)
            if random_mod[5]:     # Mirror
                new_img = transforms.functional.hflip(new_img)
                new_mask = transforms.functional.hflip(new_mask)
            if random_mod[6]:     # Rotate
                angle = np.random.uniform(-180, 180)
                new_img = transforms.functional.rotate(new_img, angle=angle)
                new_mask = transforms.functional.rotate(new_mask, angle=angle)
            if random_mod[7]:     # Crop
                crop_size = tuple(int(np.random.uniform(x/2, x)) for x in img_size)             # [240; 320] to [480; 640]
                params = transforms.RandomCrop.get_params(new_img, output_size=crop_size)
                new_img = transforms.functional.crop(new_img, *params)
                new_img = transforms.functional.resize(new_img, size=img_size)
                new_mask = transforms.functional.crop(new_mask, *params)
                new_mask = transforms.functional.resize(new_mask, size=img_size)
            if random_mod[8]:     # Affine movement
                params = transforms.RandomAffine.get_params(img_size=img_size, degrees=(-180, 180), translate=(0.1, 0.3), scale_ranges=(0.75, 1.), shears=(-10, 10))
                new_img = transforms.functional.affine(new_img, *params)
                new_mask = transforms.functional.affine(new_mask, *params)
            if random_mod[9]:     # Z Axe Shift
                scale = np.random.uniform(0, 0.1)
                params = transforms.RandomPerspective.get_params(width=img_size[1], height=img_size[0], distortion_scale=scale)
                new_img = transforms.functional.perspective(new_img, *params)
                new_mask = transforms.functional.perspective(new_mask, *params)

            data[original_dataset_lenght + idx * n + i_n] = (new_img, new_mask, original_file)

    print("\t- Augmentation done, {}".format(get_gpu_mem_usage() if load_on_gpu else get_ram_usage()))
    print("\t- Got {} new images and a total of {} images.".format(len(data) - original_dataset_lenght, len(data)))


class POCDataset(Dataset):
    """docstring for POCDataset"""
    def __init__(self, data, transform=None, target_transform=None, negative_mining=True):
        super(POCDataset, self).__init__()
        self.sampler = None
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        if negative_mining:
            self.sampler = WeightedRandomSampler(torch.ones(self.__len__()), num_samples=self.__len__(), replacement=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, mask, original_file = self.data[idx]
        img = img.float() / 255                         # convert [0;255] int8 to [0;1] float32
        mask = mask.float()                             # convert to float for gaussian filter

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)

        mask = torch.cat((1. - mask, mask), dim=0)      # Convert to [C,H,W]

        return [img, mask, original_file, idx]

    def set_weight(self, idx: torch.Tensor, weight: float):
        if self.sampler is not None:
            self.sampler.weights.index_fill_(0, idx, weight)


class POCDataReader(object):
    """docstring for POCDataLoader"""
    def __init__(self, root_dir, load_on_gpu=True):
        super(POCDataReader, self).__init__()
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.load_on_gpu = torch.cuda.is_available() and load_on_gpu
        self._files = _get_dataset_files(root_dir)

        if load_on_gpu != self.load_on_gpu:
            print("Cannot load Dataset on GPU, cuda is not available.")

        self.data = {}
        for i, (img_path, mask_path) in tenumerate(self._files, desc=f"Loading dataset into {'GPU' if self.load_on_gpu else 'RAM'}", tqdm_class=tqdm):
            img = read_image(img_path, mode=ImageReadMode.GRAY)
            mask = read_image(mask_path, mode=ImageReadMode.GRAY).bool()
            file_name = os.path.basename(img_path)
            if self.load_on_gpu:
                self.data[i] = (img.cuda(), mask.cuda(), file_name)
            else:
                self.data[i] = (img, mask, file_name)

        print("\t- Loading done, {}".format(get_gpu_mem_usage() if self.load_on_gpu else get_ram_usage()))
        print("\t- Got a total of {} images.".format(self.__len__()))

    def __len__(self):
        return len(self.data)

    def split(self, splits):
        assert len(splits) == 3, "Can only cut data into 3 samples"
        splits = np.floor(np.array(splits) * self.__len__()).astype(int)

        cut1 = splits[0]; cut2 = splits[0] + splits[1]

        samples = np.random.permutation(self.__len__())
        train_data = {i: self.data[k] for i, k in enumerate(samples[:cut1])}
        val_data = {i: self.data[k] for i, k in enumerate(samples[cut1:cut2])}
        test_data = {i: self.data[k] for i, k in enumerate(samples[cut2:])}

        return train_data, val_data, test_data
