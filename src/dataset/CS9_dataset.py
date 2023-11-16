import numpy as np
import os
from tqdm.auto import tqdm, trange
from tqdm.contrib import tenumerate

import warnings

import torch
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, WeightedRandomSampler, default_collate
from torchvision import transforms

from my_utils import get_gpu_mem_usage, get_ram_usage



def _get_dataset_files(root_dir: str, mode: str, skip_test_non_crack=True):
    img_path = os.path.join(root_dir, 'Images')
    mask_path = os.path.join(root_dir, 'Masks')

    if mode == "Train":
        with open(os.path.join(root_dir, 'train.txt')) as f:
            file_list = f.read().splitlines()
    elif mode == "Test":
        with open(os.path.join(root_dir, 'test.txt')) as f:
            file_list = f.read().splitlines()
    else:
        file_list = os.listdir(img_path)

    file_list.sort()

    items = []
    for it_file in file_list:
        if skip_test_non_crack and mode == "Test" and it_file.startswith('noncrack_'):
            continue
        item = (os.path.join(img_path, it_file), os.path.join(mask_path, it_file))
        if os.path.isfile(item[0]) and os.path.isfile(item[1]):
            items.append(item)

    return items


class CS9Dataset(Dataset):
    """docstring for CS9Dataset"""
    def __init__(self, data, transform=None, target_transform=None, load_on_gpu: bool = False, verbose: bool = False):
        super(CS9Dataset, self).__init__()
        self.load_on_gpu = torch.cuda.is_available() and load_on_gpu
        self.transform = transform
        self.target_transform = target_transform

        if self.load_on_gpu:
            self.data = {k: (img.cuda(), mask.cuda(), file) for k, (img, mask, file) in data.items()}
        else:
            self.data = {k: (img.clone(), mask.clone(), file) for k, (img, mask, file) in data.items()}
        if verbose:
            print("\t- Loading done, {}".format(get_gpu_mem_usage() if self.load_on_gpu else get_ram_usage()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, mask, original_file = self.data[idx]
        img = img.float() / 255                         # convert [0;255] int8 to [0;1] float32
        mask = mask.float() / 255                       # convert to float for gaussian filter

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)

        mask = torch.cat((1. - mask, mask), dim=0)      # Convert to [C,H,W]

        return [img, mask, original_file, idx]

    def set_weight(self, idx: torch.Tensor, weight: float):
        pass

    def precompute_transform(self, verbose: bool = False):
        item_loop = tqdm(self.data.items(), desc="Applying transform to the Dataset") if verbose else self.data.items()
        for key, (img, mask, file_name) in item_loop:
            img = img.float() / 255         # Convert both to float for opperations
            mask = mask.float() / 255

            if self.transform:
                img = self.transform(img)
            if self.target_transform:
                mask = self.target_transform(mask)

            img = (img * 255).clamp(0, 255).byte()     # Convert both back to uint8 for storage
            mask = (mask * 255).clamp(0, 255).byte()
            self.data[key] = (img, mask, file_name)

        self.transform = None
        self.target_transform = None

        if verbose:
            print("\t- Transformation done, {}".format(get_gpu_mem_usage() if self.load_on_gpu else get_ram_usage()))


class CS9DataReader(object):
    """docstring for POCDataLoader"""
    def __init__(self, root_dir, mode: str = "Test", load_on_gpu: bool = False, limit: int = None, verbose: bool = False):
        super(CS9DataReader, self).__init__()
        """
        Args:
            root_dir (string): Directory with all the images.
            load_on_gpu (bool): Should data tensor be on cuda device.
        """
        assert mode in ["Train", "Test", "All"]
        self.root_dir = root_dir
        self.load_on_gpu = torch.cuda.is_available() and load_on_gpu
        self._files = _get_dataset_files(root_dir, mode)

        if load_on_gpu != self.load_on_gpu:
            print("Cannot load Dataset on GPU, cuda is not available.")

        self._data = {}
        loop_enum = tenumerate(self._files, desc=f"Loading dataset into {'GPU' if self.load_on_gpu else 'RAM'}", tqdm_class=tqdm) if verbose else enumerate(self._files)
        for i, (img_path, mask_path) in loop_enum:
            if limit is not None and i >= limit:
                break

            img = read_image(img_path, mode=ImageReadMode.UNCHANGED)
            mask = read_image(mask_path, mode=ImageReadMode.GRAY)
            file_name = os.path.basename(img_path)
            if self.load_on_gpu:
                self._data[i] = (img.cuda(), mask.cuda(), file_name)
            else:
                self._data[i] = (img, mask, file_name)

        if verbose:
            print("\t- Loading done, {}".format(get_gpu_mem_usage() if self.load_on_gpu else get_ram_usage()))
            print("\t- Got a total of {} images.".format(self.__len__()))

    def __len__(self):
        return len(self._data)

    @property
    def data(self):
        return self._data
