import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch.cuda as tcuda
from torch.nn.functional import softmax, normalize


def show_img(tensors):
    ncols = len(tensors)
    nrows = max(tensors.size(-3) - 2, 1)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False, figsize=(tensors.size(-1) * ncols / 100, tensors.size(-2) * nrows / 100))
    fig.tight_layout()
    for i, tensor in enumerate(tensors):
        tensor = tensor.detach().cpu()

        img = tensor[0:3]
        filters = tensor[3:]

        if img.size(0) == 1:
            array = np.array(255 * img[0] / img.max(), dtype=np.uint8)
            axs[0, i].imshow(array, cmap='gray')
        elif img.size(0) == 2:
            array = np.array(255 * img.argmax(dim=0, keepdim=False), dtype=np.uint8)
            axs[0, i].imshow(array, cmap='gray')
        else:
            array = np.array(255 * img / img.max(), dtype=np.uint8)
            axs[0, i].imshow(np.dstack(array))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        for j, layer in enumerate(filters):
            array = np.array(255 * layer / layer.max(), dtype=np.uint8)
            axs[j+1, i].imshow(array, cmap='gray')
            axs[j+1, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def get_gpu_mem_usage(complete=True):
    gpu_mem_free, gpu_mem_total = tcuda.mem_get_info()
    gpu_mem_free /= 1024**3; gpu_mem_total /= 1024**3
    if not complete:
        return "{:.2f}GiB".format(gpu_mem_total - gpu_mem_free)
    gpu_mem_used = tcuda.memory_allocated() / 1024**3
    return "GPU memory used: {:.2f}GiB / free: {:.2f}GiB / total: {:.2f}GiB".format(gpu_mem_used, gpu_mem_free, gpu_mem_total)

def get_ram_usage(complete=True):
    ram_infos = psutil.virtual_memory()
    if not complete:
        return "{:.2f}GiB".format(ram_infos.used / 1024**3)
    return "RAM used: {:.2f}GiB / free: {:.2f}GiB / total: {:.2f}GiB".format(ram_infos.used / 1024**3, ram_infos.free / 1024**3, ram_infos.total / 1024**3)

