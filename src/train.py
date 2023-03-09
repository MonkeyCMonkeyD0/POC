import torch

from utils import get_gpu_mem_usage



def training_loop(epoch, dataloader, model, loss_fn, optimizer, lr_scheduler, metric, device):

    model.train()
    for batch_idx, (X, Y, _, items_index) in enumerate(dataloader):
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

        batch_score = metric.collect_metrics(batch_index=batch_idx, loss_value=loss.item(), preds=Y_hat, targets=Y)
        dataloader.dataset.set_weight(items_index, 1. - batch_score)

    lr_scheduler.step()
    metric.write_epoch_tensorboard(epoch, lr_scheduler.get_last_lr()[0])


def validation_loop(epoch, dataloader, model, loss_fn, metric, device):

    model.eval()
    with torch.inference_mode():
        for batch_idx, (X, Y, _, _) in enumerate(dataloader):
            X = X.to(device); Y = Y.to(device)

            Y_hat = model(X)
            loss_val = loss_fn(Y_hat, Y).item()

            metric.collect_metrics(batch_index=batch_idx, loss_value=loss_val, preds=Y_hat, targets=Y)

        metric.write_epoch_tensorboard(epoch)
