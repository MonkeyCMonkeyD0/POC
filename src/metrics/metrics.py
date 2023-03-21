import pandas as pd
import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter

from .dice import DiceIndex
from .jaccard import JaccardIndex
from .tversky import TverskyIndex



class Metrics(nn.Module):
    def __init__(self, buffer_size, mode: str, hyperparam: dict, smooth=1e-7, device=None):
        super(Metrics, self).__init__()

        self.device = device
        # self.hyperparameters = {k: str(v) for k, v in hyperparam.items()}
        self.register_buffer("_losses", torch.zeros(buffer_size, dtype=torch.float, device=self.device))
        self.register_buffer("_scores_crack_IOU", torch.zeros(buffer_size, dtype=torch.float, device=self.device))
        self.register_buffer("_scores_mean_IOU", torch.zeros(buffer_size, dtype=torch.float, device=self.device))
        self.register_buffer("_scores_Tversky", torch.zeros(buffer_size, dtype=torch.float, device=self.device))

        self.jaccardCrackIndex = JaccardIndex(mesure_background=False, smooth=smooth).to(self.device)
        self.jaccardMeanIndex = JaccardIndex(mesure_background=True, smooth=smooth).to(self.device)
        self.tverskyIndex = TverskyIndex(alpha=.3, beta=.7, smooth=smooth).to(self.device)

        assert mode in ["Training", "Validation", "Evaluation"]

        self.hyperparameters = {k: v for k,v in hyperparam.items()}
        self.hyperparameters['Network'] = self.hyperparameters['Network'].__name__
        self.hyperparameters['Combine Loss'] = self.hyperparameters['Combine Loss'].__name__
        self.hyperparameters['Optimizer'] = self.hyperparameters['Optimizer'].__name__
        self.hyperparameters['Input Filter'] = self.hyperparameters['Input Filter'].__name__ if self.hyperparameters['Input Filter'] is not None else " "

        flags = "" + ("-NM" if self.hyperparameters['Negative Mining'] else "") + ("-SL" if self.hyperparameters['Smooth Labeling'] else "")

        self.log_folder = f"../logs/N:{self.hyperparameters['Network']}-O:{self.hyperparameters['Optimizer']}"
        self.log_folder += f"-L:{self.hyperparameters['Combine Loss']}_{self.hyperparameters['Pixel Loss']}_{self.hyperparameters['Volume Loss']}"
        self.log_folder += f"-P:{self.hyperparameters['Input Filter']}_{self.hyperparameters['Input Layer']}"
        self.log_folder += f"-BS:{self.hyperparameters['Batch Size']}-LR:{self.hyperparameters['Learning Rate']:.1e}{flags}/{mode}"
        self.writer = SummaryWriter(self.log_folder, max_queue=4)

    @property
    def loss(self):
        return self._losses.mean()

    @property
    def crackIoU(self):
        return self._scores_crack_IOU.mean()

    @property
    def meanIoU(self):
        return self._scores_mean_IOU.mean()

    @property
    def tversky(self):
        return self._scores_Tversky.mean()

    def get_metrics(self, epoch: int):
        return {"Epoch": epoch, "Loss": self.loss.item(), "CrackIoU": self.crackIoU.item(), "MeanIoU": self.meanIoU.item(), "Tversky": self.tversky.item()}


    def collect_metrics(self, batch_index: int, loss_value: float, preds: torch.Tensor, targets: torch.Tensor):
        preds, targets = preds.detach(), targets.detach()

        self._losses[batch_index] = loss_value
        self._scores_crack_IOU[batch_index] = self.jaccardCrackIndex.forward(preds, targets)
        self._scores_mean_IOU[batch_index] = self.jaccardMeanIndex.forward(preds, targets)
        self._scores_Tversky[batch_index] = self.tverskyIndex.forward(preds, targets)

        return self._scores_crack_IOU[batch_index]

    def write_epoch_tensorboard(self, epoch, lr=None):
        if self.hyperparameters['Combine Loss'] == "PixelLoss":
            loss_name = f"Losses/{self.hyperparameters['Pixel Loss']}"
        elif self.hyperparameters['Combine Loss'] == "VolumeLoss":
            loss_name = f"Losses/{self.hyperparameters['Volume Loss']}"
        else:
            loss_name = f"Losses/{self.hyperparameters['Combine Loss']}-{self.hyperparameters['Pixel Loss']}-{self.hyperparameters['Volume Loss']}"
        self.writer.add_scalar(loss_name, self.loss, epoch, new_style=True)
        self.writer.add_scalar("Indexes/Crack IOU", self.crackIoU, epoch, new_style=True)
        self.writer.add_scalar("Indexes/Mean IOU", self.meanIoU, epoch, new_style=True)
        self.writer.add_scalar("Indexes/Tversky", self.tversky, epoch, new_style=True)
        if lr is not None:
            self.writer.add_scalar("Learning Rate", lr, epoch, new_style=True)

    def close_tensorboard(self):
        self.writer.flush()
        self.writer.close()


class EvaluationMetrics(Metrics):
    """docstring for EvaluationMetrics"""
    def __init__(self, buffer_size, hyperparam: dict, smooth=1e-7, device=None):
        super(EvaluationMetrics, self).__init__(buffer_size, mode="Evaluation", hyperparam=hyperparam, smooth=smooth, device=device)

        self.name_list = []
        self.dataframe = pd.DataFrame(columns=["Image", "CrackIoU", "MeanIoU"])
        self.hyperparameters["Epochs"] = epochs


    def collect_metrics(self, item_index: int, img_name: str, preds: torch.Tensor, targets: torch.Tensor):
        super().collect_metrics(item_index, 0., preds, targets)
        self.name_list.append(img_name)

    def write_hyperparameters_tensorboard(self):
        self.writer.add_hparams(
            hparam_dict=self.hyperparameters, 
            metric_dict={
                "Crack IOU": self.crackIoU, 
                "Mean IOU": self.meanIoU,
                "Tversky": self.tversky},
            run_name=".")

    def write_metric_dataframe(self):
        self.dataframe["Image"] = self.name_list
        self.dataframe["CrackIoU"] = self._scores_crack_IOU.cpu().numpy()
        self.dataframe["MeanIoU"] = self._scores_mean_IOU.cpu().numpy()

        self.dataframe["CrackIoU"] = self.dataframe["CrackIoU"].astype(float)
        self.dataframe["MeanIoU"] = self.dataframe["MeanIoU"].astype(float)

        self.dataframe.sort_values(by=["CrackIoU", "MeanIoU"], inplace=True)
        self.dataframe.to_csv(path_or_buf= self.log_folder + "/metrics.csv")

        ax = self.dataframe.plot.bar(x="Image", y=["CrackIoU", "MeanIoU"], stacked=True)
        ax.figure.set_size_inches(12, 6)
        self.writer.add_figure("IoU Metrics for Evaluation Dataset", ax.get_figure())

    def write_example_segmentation_tensorboard(self, dataloader, model):
        images = []; masks = []; files = []
        for i in range(8):
            image, mask, file, _ = next(iter(dataloader))
            images.append(image); masks.append(mask[:,1:]); files.append(file)
        images = torch.cat(images, dim=0).to(self.device)
        masks = torch.cat(masks, dim=0)

        model.eval()
        with torch.inference_mode():
            preds = model(images)

            images = (255 * images).byte()
            masks = (255 * masks).byte()
            preds = (255 * preds.argmax(dim=1, keepdim=True)).byte()

        img_tensor = torch.cat((images.cpu(), preds.cpu(),masks), dim=0)
        self.writer.add_images(f"Segmentation example ({files})", img_tensor, dataformats='NCHW')

        # def write_model_graph_tensorboard(self, model: nn.Module, images):
        #     self.writer.add_graph(model, input_to_model=images.to(self.device))
