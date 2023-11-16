import numpy as np
import os
from tqdm.auto import tqdm, trange
from tqdm.contrib import tenumerate

import warnings

import torch
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from my_utils import get_gpu_mem_usage, get_ram_usage


def data_augment_(data: dict, n: int, load_on_gpu: bool = False, verbose: bool = False, seed: int = None):
    if n <= 0:
        warnings.warn("Need a strictly positive integer for n for data augmentation. Will skip augmentation.")
        return None

    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    original_dataset_lenght = len(data)
    loop = trange(original_dataset_lenght, desc="Expending the dataset {} more times".format(n)) if verbose else range(original_dataset_lenght)
    for idx in loop:
        img, mask, original_file = data[idx]
        for i_n in range(n):
            new_img = img.detach().clone()
            new_mask = mask.detach().clone()
            if load_on_gpu:
                new_img = new_img.cuda()
                new_mask = new_mask.cuda()

            random_mod_nsp = rng.choice([0, 1], size=4, p=[3/4, 1/4])    # Proba of 0.25 for each event = ~1 transformations
            random_mod_sp = rng.choice([0, 1], size=3, p=[2/3, 1/3])    # Proba of 1/3 for each event = ~1 transformations

            # non-spacial variations (only on image)
            if random_mod_nsp[0]:     # Brightness shift
                new_img = transforms.ColorJitter(brightness=.5)(new_img)
            if random_mod_nsp[1]:     # Autocontrast
                new_img = transforms.functional.autocontrast(new_img)
            if random_mod_nsp[2]:     # Sharpness
                new_img = transforms.functional.adjust_sharpness(new_img, sharpness_factor=rng.uniform(0, 2))
            if random_mod_nsp[3]:     # Gaussian Blur
                new_img = transforms.functional.gaussian_blur(new_img, kernel_size=5)

            # spacial variations (on image & mask)
            img_size = transforms.functional.get_image_size(new_img)[::-1]
            if random_mod_sp[0]:     # Flip
                new_img = transforms.functional.vflip(new_img)
                new_mask = transforms.functional.vflip(new_mask)
            if random_mod_sp[1]:     # Mirror
                new_img = transforms.functional.hflip(new_img)
                new_mask = transforms.functional.hflip(new_mask)
            if random_mod_sp[2]:     # Rotate
                angle = 90 * rng.randint(1, 4)
                new_img = transforms.functional.rotate(new_img, angle=angle)
                new_mask = transforms.functional.rotate(new_mask, angle=angle)

            data[original_dataset_lenght + idx * n + i_n] = (new_img, new_mask, original_file)

    if verbose:
        print("\t- Augmentation done, {}".format(get_gpu_mem_usage() if load_on_gpu else get_ram_usage()))
        print("\t- Got {} new images and a total of {} images.".format(len(data) - original_dataset_lenght, len(data)))

    return data


class POCDataReader(object):
    """docstring for POCDataLoader"""
    def __init__(self, root_dir, load_on_gpu: bool = False, limit: int = None, verbose: bool = False):
        super(POCDataReader, self).__init__()
       """
        Args:
            root_dir (string): Directory with all the images.
            load_on_gpu (bool): Should data tensor be on cuda device.
        """
        self.root_dir = root_dir
        self.load_on_gpu = torch.cuda.is_available() and load_on_gpu
        self._train_files, self._val_files, self._test_files = self._get_dataset_files(root_dir)

        if load_on_gpu != self.load_on_gpu:
            print("Cannot load Dataset on GPU, cuda is not available.")

        self._train_data = {}; self._val_data = {}; self._test_data = {}

        # torch.manual_seed(seed)

        for data_dict, file_list in zip([self._train_data, self._val_data, self._test_data],[self._train_files, self._val_files, self._test_files]):
            loop_enum = tenumerate(file_list, desc=f"Loading dataset into {'GPU' if self.load_on_gpu else 'RAM'}", tqdm_class=tqdm) if verbose else enumerate(file_list)
            for i, (img_path, mask_path) in loop_enum:
                if limit is not None and i >= limit:
                    break

                # resize = transforms.RandomCrop(size=(256, 256))

                img = read_image(img_path, mode=ImageReadMode.UNCHANGED)
                mask = read_image(mask_path, mode=ImageReadMode.GRAY)
                file_name = os.path.basename(img_path)

                # if img.size(1) < 256 or img.size(2) < 256:
                img = transforms.functional.resize(img, size=(256,256))
                mask = transforms.functional.resize(mask, size=(256,256))

                # rng_state = torch.get_rng_state()
                # img = resize(img)
                # torch.set_rng_state(rng_state)
                # mask = resize(mask)

                if self.load_on_gpu:
                    data_dict[i] = (img.cuda(), mask.cuda(), file_name)
                else:
                    data_dict[i] = (img, mask, file_name)

        if verbose:
            print("\t- Loading done, {}".format(get_gpu_mem_usage() if self.load_on_gpu else get_ram_usage()))
            print("\t- Got a total of {} images.".format(self.__len__()))

    def __len__(self):
        return len(self._train_data) + len(self._val_data) + len(self._test_data)

    @staticmethod
    def _get_dataset_files(root_dir: str):

        with open(os.path.join(root_dir, 'training.txt')) as f:
            train_item_name = f.read().splitlines()
        with open(os.path.join(root_dir, 'validation.txt')) as f:
            val_item_name = f.read().splitlines()
        with open(os.path.join(root_dir, 'testing.txt')) as f:
            test_item_name = f.read().splitlines()

        img_path = os.path.join(root_dir, 'image')
        mask_path = os.path.join(root_dir, 'mask')

        train_list = []
        for item_name in train_item_name:
            img_filename = os.path.join(img_path, item_name)
            mask_filename = os.path.join(mask_path, item_name)
            if os.path.isfile(img_filename) and os.path.isfile(mask_filename):
                train_list.append((img_filename, mask_filename))

        val_list = []
        for item_name in val_item_name:
            img_filename = os.path.join(img_path, item_name)
            mask_filename = os.path.join(mask_path, item_name)
            if os.path.isfile(img_filename) and os.path.isfile(mask_filename):
                val_list.append((img_filename, mask_filename))

        test_list = []
        for item_name in test_item_name:
            img_filename = os.path.join(img_path, item_name)
            mask_filename = os.path.join(mask_path, item_name)
            if os.path.isfile(img_filename) and os.path.isfile(mask_filename):
                test_list.append((img_filename, mask_filename))

        return train_list, val_list, test_list

    def split(self):
        return self._train_data, self._val_data, self._test_data



class POCDataset(Dataset):
    """docstring for POCDataset"""
    def __init__(self, data, transform=None, target_transform=None, negative_mining: bool = False, load_on_gpu: bool = False, verbose: bool = False):
        super(POCDataset, self).__init__()
        self.sampler = None
        self.load_on_gpu = torch.cuda.is_available() and load_on_gpu
        self.transform = transform
        self.target_transform = target_transform

        if self.load_on_gpu:
            self.data = {k: (img.clone().cuda(), mask.clone().cuda(), file) for k, (img, mask, file) in data.items()}
        else:
            self.data = {k: (img.clone(), mask.clone(), file) for k, (img, mask, file) in data.items()}
        if verbose:
            print("\t- Loading done, {}".format(get_gpu_mem_usage() if self.load_on_gpu else get_ram_usage()))

        if negative_mining:
            self.sampler = WeightedRandomSampler(torch.ones(self.__len__()), num_samples=self.__len__(), replacement=True)

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
        if self.sampler is not None:
            self.sampler.weights.index_fill_(0, idx, weight)

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
