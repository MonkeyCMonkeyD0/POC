import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch.cuda as tcuda


def show_img(tensors):
    fix, axs = plt.subplots(ncols=len(tensors), squeeze=False)
    for i, tensor in enumerate(tensors):
        array = np.array((tensor[-1] * 255).cpu(), dtype=np.uint8)
        if np.ndim(array) > 2:
            assert array.shape[0] == 1
            array = array[0]
        axs[0, i].imshow(array, cmap='gray')
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

