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


class POCFixedDataReader(object):
    """docstring for POCDataLoader"""
    def __init__(self, root_dir, load_on_gpu: bool = False, dataset: str = "cs9", seed: int = 1234, limit: int = None, verbose: bool = False):
        super(POCFixedDataReader, self).__init__()
        """
        Args:
            root_dir (string): Directory with all the images.
            load_on_gpu (bool): Should data tensor be on cuda device.
        """
        assert dataset in ["cs9", "poc2"]
        self.root_dir = root_dir
        self.load_on_gpu = torch.cuda.is_available() and load_on_gpu
        self._train_files, self._val_files, self._test_files = self._get_dataset_files(root_dir, dataset)

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

    @staticmethod
    def _get_dataset_files(root_dir: str, dataset: str):

        with open(os.path.join(root_dir, 'training.txt')) as f:
            train_item_name = f.read().splitlines()
        with open(os.path.join(root_dir, 'validation.txt')) as f:
            val_item_name = f.read().splitlines()
        with open(os.path.join(root_dir, 'testing_POCvsCS9.txt')) as f:
            test_item_name = f.read().splitlines()

        assert dataset in ["cs9", "poc2"]

        img_path = os.path.join(root_dir, 'image')
        mask_path = os.path.join(root_dir, f'mask.{dataset}')

        train_list = []
        for item_name in train_item_name:
            if os.path.isfile(os.path.join(root_dir, 'image', item_name)) and \
                os.path.isfile(os.path.join(root_dir, "mask.poc2", item_name)) and \
                os.path.isfile(os.path.join(root_dir, "mask.cs9", item_name)):
                train_list.append((os.path.join(img_path, item_name), os.path.join(mask_path, item_name)))

        val_list = []
        for item_name in val_item_name:
            if os.path.isfile(os.path.join(root_dir, 'image', item_name)) and \
                os.path.isfile(os.path.join(root_dir, "mask.poc2", item_name)) and \
                os.path.isfile(os.path.join(root_dir, "mask.cs9", item_name)):
                val_list.append((os.path.join(img_path, item_name), os.path.join(mask_path, item_name)))

        test_list = []
        for item_name in test_item_name:
            if os.path.isfile(os.path.join(root_dir, 'image', item_name)) and \
                os.path.isfile(os.path.join(root_dir, "mask.poc2", item_name)) and \
                os.path.isfile(os.path.join(root_dir, "mask.cs9", item_name)):
                test_list.append((os.path.join(img_path, item_name), os.path.join(mask_path, item_name)))

        return train_list, val_list, test_list

    def __len__(self):
        return len(self._train_data) + len(self._val_data) + len(self._test_data)

    def split(self):
        return self._train_data, self._val_data, self._test_data
