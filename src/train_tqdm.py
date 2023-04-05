import torch
from tqdm.notebook import tqdm

from my_utils import get_gpu_mem_usage



def training_loop(epoch, dataloader, model, loss_fn, optimizer, lr_scheduler, metric, device):

    train_tqdm = tqdm(dataloader, desc="[Training] Epoch: {} ".format(epoch), leave=False)

    model.train()
    for batch_idx, (X, Y, _, items_index) in enumerate(train_tqdm):
        X = X.to(device); Y = Y.to(device)

        # zero the parameter gradients
        model.zero_grad()
        optimizer.zero_grad()

        # Compute prediction and loss
        Y_hat = model(X)
        loss = loss_fn(Y_hat, Y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        # lr_scheduler.step(epoch + batch_idx / len(dataloader))

        if isinstance(Y_hat, tuple):
            Y_hat = Y_hat[0]
        batch_score = metric.collect_metrics(batch_index=batch_idx, loss_value=loss.item(), preds=Y_hat, targets=Y)
        # dataloader.dataset.set_weight(items_index, 1. - batch_score)

        train_tqdm.set_postfix({"loss": loss.item(),"gpu mem": get_gpu_mem_usage(False)})

    lr_scheduler.step()
    metric.write_epoch_tensorboard(epoch, lr_scheduler.get_last_lr()[0])


def validation_loop(epoch, dataloader, model, loss_fn, metric, device):

    model.eval()
    with torch.inference_mode():
        val_tqdm = tqdm(dataloader, desc="[Validating] Epoch: {} ".format(epoch), leave=False)
        for batch_idx, (X, Y, _, _) in enumerate(val_tqdm):
            X = X.to(device); Y = Y.to(device)

            Y_hat = model(X)
            loss_val = loss_fn(Y_hat, Y).item()

            if isinstance(Y_hat, tuple):
                Y_hat = Y_hat[0]
            metric.collect_metrics(batch_index=batch_idx, loss_value=loss_val, preds=Y_hat, targets=Y)
            val_tqdm.set_postfix({"loss": loss_val,"gpu mem": get_gpu_mem_usage(False)})

        metric.write_epoch_tensorboard(epoch)


def evaluation_loop(dataloader, model, metric, device):

    model.eval()
    with torch.inference_mode():
        val_tqdm = tqdm(dataloader, desc="[Evaluating]")
        for idx, (X, Y, img_name, _) in enumerate(val_tqdm):
            X = X.to(device); Y = Y.to(device)

            Y_hat = model(X)
            if isinstance(Y_hat, tuple):
                Y_hat = Y_hat[0]
            metric.collect_metrics(item_index=idx, img_name=img_name[0], preds=Y_hat, targets=Y)

        metric.write_hyperparameters_tensorboard()
        metric.write_metric_dataframe()
        metric.write_example_segmentation_tensorboard(dataloader, model)
