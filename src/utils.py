import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch.cuda as tcuda
from torch.nn.functional import softmax, normalize


def show_img(tensors):
    fix, axs = plt.subplots(ncols=len(tensors), squeeze=False)
    for i, tensor in enumerate(tensors):
        tensor = tensor.detach().cpu()
        if tensor.size()[0] < 3:
            tensor = tensor[-1]
            array = np.array(255 * tensor / tensor.max(), dtype=np.uint8)
            axs[0, i].imshow(array, cmap='gray')
        else:
            if tensor.size()[0] > 3:
                tensor = tensor[0:3]
            array = np.array(255 * tensor / tensor.max(), dtype=np.uint8)
            axs[0, i].imshow(np.dstack(array))
        
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

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

