import os

import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import normalize
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import GaussianBlur

import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch


from Dataset import POCDataReader, data_augment_, POCDataset
from metrics import Metrics, EvaluationMetrics
from models import UNet
from loss import *
from train import training_loop, validation_loop, evaluation_loop



EPOCHS = 3 #20
NUM_SAMPLES = 3 #30

NB_AUGMENT = 0
LOAD_DATA_ON_GPU = False


def train(config, train_data, val_data):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = POCDataset(
        train_data,
        transform=normalize,
        target_transform= GaussianBlur(kernel_size=3, sigma=0.7) if config["SL"] else None,
        negative_mining=config["NM"])
    training_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        sampler=train_dataset.sampler,
        num_workers=8,
        pin_memory=True,
        pin_memory_device=device)

    val_dataset = POCDataset(val_data, transform=normalize, target_transform=None, negative_mining=False)
    validation_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        pin_memory_device=device)

    model = UNet(n_channels=1, n_classes=2, bilinear=True, crop=False)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    loss_fn = config["loss_fn"].to(device)
    optimizer = Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.99))
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS//2)

    # To restore a checkpoint, use `session.get_checkpoint()`.
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state, scheduler_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        lr_scheduler.load_state_dict(scheduler_state)

    train_metrics = Metrics(
        buffer_size=len(training_dataloader),
        mode="Training",
        model_name=model.__class__.__name__,
        loss_name=loss_fn.__class__.__name__,
        opt_name=optimizer.__class__.__name__,
        batch_size=config["batch_size"],
        learning_rate=config["lr"],
        negative_mining=config["NM"],
        soft_labels=config["SL"],
        device=device)

    val_metrics = Metrics(
        buffer_size=len(validation_dataloader),
        mode="Validation",
        model_name=model.__class__.__name__,
        loss_name=loss_fn.__class__.__name__,
        opt_name=optimizer.__class__.__name__,
        batch_size=config["batch_size"],
        learning_rate=config["lr"],
        negative_mining=config["NM"],
        soft_labels=config["SL"],
        device=device)


    for epoch in range(1, EPOCHS+1):  # loop over the dataset multiple times
        training_loop(epoch, training_dataloader, model, loss_fn, optimizer, lr_scheduler, train_metrics, device)
        validation_loop(epoch, validation_dataloader, model, loss_fn, val_metrics, device)

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `session.get_checkpoint()`
        # API in future iterations.
        os.makedirs("model", exist_ok=True)
        torch.save((model.state_dict(), optimizer.state_dict(), lr_scheduler.state_dict()), "model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("model")
        session.report(metrics=val_metrics.get_metrics(), checkpoint=checkpoint)

    train_metrics.close_tensorboard()
    val_metrics.close_tensorboard()
    print("Finished Training")


def evaluate(test_data, best_result):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    test_dataset = POCDataset(test_data, transform=normalize, target_transform=None, negative_mining=False)
    evaluation_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, pin_memory_device=device)

    best_trained_model = UNet(n_channels=1, n_classes=2, bilinear=True, crop=False).to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
    model_state, _, _ = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    test_metrics = EvaluationMetrics(
        buffer_size=len(evaluation_dataloader),
        model_name=best_trained_model.__class__.__name__,
        loss_name=best_result.config["loss_fn"].__class__.__name__,
        opt_name="Adam",
        epochs=EPOCHS,
        batch_size=best_result.config["batch_size"],
        learning_rate=best_result.config["learning_rate"],
        negative_mining=best_result.config["NM"],
        soft_labels=best_result.config["SL"],
        device=device)

    evaluation_loop(dataloader=evaluation_dataloader, model=best_trained_model, metric=test_metrics, device=device)


def loss_fn_sampler():
    pixel_losses_list = [
        CrossEntropyLoss(weight=torch.tensor([.3, .7])), 
        FocalLoss(weight=torch.tensor([.3, .7]), gamma=2)
    ]
    volume_losses_list = [
        JaccardLoss(),
        TverskyLoss(alpha=0.3, beta=0.7),
        FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=2)
    ]
    loss_combinators_list = [ CombinedLoss, BorderedLoss ]

    complete_list = pixel_losses_list + volume_losses_list

    for combinator in loss_combinators_list:
        complete_list += [combinator(loss1, loss2) for loss1 in pixel_losses_list for loss2 in volume_losses_list]

    return complete_list


def main(num_samples=NUM_SAMPLES, max_num_epochs=EPOCHS, gpus_per_trial=1):

    search_space = {
        "lr": tune.qloguniform(1e-5, 1e-2, 5e-6),
        "batch_size": tune.qrandint(2, 16, 2),
        "NM": tune.choice([True, False]),
        "SL": tune.choice([True, False]),
        "loss_fn": tune.choice(loss_fn_sampler()),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    data_reader = POCDataReader(root_dir="../data", load_on_gpu=LOAD_DATA_ON_GPU)
    train_data, val_data, test_data = data_reader.split([0.7, 0.1, 0.2])
    data_augment_(train_data, n=NB_AUGMENT, load_on_gpu=LOAD_DATA_ON_GPU)

    # Uncomment this to enable distributed execution
    # ray.init(address="auto")

    scheduler = ASHAScheduler(max_t=max_num_epochs, grace_period=1, reduction_factor=2)
    search_algo = HyperOptSearch()
    tune_config = tune.TuneConfig(metric="CrackIoU", mode="max", num_samples=NUM_SAMPLES, scheduler=scheduler, search_alg=search_algo)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train, train_data=train_data, val_data=val_data),
            resources={"cpu": 8, "gpu": gpus_per_trial}
        ),
        tune_config=tune_config,
        param_space=search_space,
    )
    results = tuner.fit()

    best_result = results.get_best_result(metric="CrackIoU", mode="max", scope="all")  # Get best result object
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["Loss"]))
    print("Best trial final validation CrackIoU: {}".format(best_result.metrics["CrackIoU"]))

    evaluate(test_data=test_data, best_result=best_result)


if __name__ == "__main__":

    # You can change the number of GPUs per trial here:
    main(num_samples=NUM_SAMPLES, max_num_epochs=EPOCHS, gpus_per_trial=1)

