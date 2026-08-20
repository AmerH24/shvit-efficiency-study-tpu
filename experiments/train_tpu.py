import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

from timm.models import create_model

# Import registrations for our custom SHViT variants.
import model  # noqa: F401


MODEL_NAMES = [
    "shvit_s1_ratio_1_8",
    "shvit_s1_ratio_default",
    "shvit_s1_ratio_1_2",
    "shvit_s1_progressive",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODEL_NAMES,
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="/kaggle/working/cifar100",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size PER TPU process/device.",
    )

    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.025,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    return parser.parse_args()


def build_datasets(args):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    train_dataset = datasets.CIFAR100(
        root=args.data_path,
        train=True,
        transform=train_transform,
        download=True,
    )

    test_dataset = datasets.CIFAR100(
        root=args.data_path,
        train=False,
        transform=test_transform,
        download=True,
    )

    return train_dataset, test_dataset


def accuracy_top1(outputs, targets):
    predictions = outputs.argmax(dim=1)

    correct = (
        predictions == targets
    ).sum()

    return correct


def train_one_epoch(
    model_instance,
    loader,
    optimizer,
    criterion,
    device,
):
    model_instance.train()

    total_loss = torch.tensor(
        0.0,
        device=device,
    )

    total_correct = torch.tensor(
        0,
        device=device,
        dtype=torch.long,
    )

    total_samples = torch.tensor(
        0,
        device=device,
        dtype=torch.long,
    )

    device_loader = pl.MpDeviceLoader(
        loader,
        device,
    )

    for images, targets in device_loader:
        optimizer.zero_grad()

        outputs = model_instance(images)

        loss = criterion(
            outputs,
            targets,
        )

        loss.backward()

        xm.optimizer_step(
            optimizer,
        )

        batch_size = targets.size(0)

        total_loss += (
            loss.detach() * batch_size
        )

        total_correct += accuracy_top1(
            outputs,
            targets,
        )

        total_samples += batch_size

    return (
        total_loss,
        total_correct,
        total_samples,
    )


@torch.no_grad()
def evaluate(
    model_instance,
    loader,
    criterion,
    device,
):
    model_instance.eval()

    total_loss = torch.tensor(
        0.0,
        device=device,
    )

    total_correct = torch.tensor(
        0,
        device=device,
        dtype=torch.long,
    )

    total_samples = torch.tensor(
        0,
        device=device,
        dtype=torch.long,
    )

    device_loader = pl.MpDeviceLoader(
        loader,
        device,
    )

    for images, targets in device_loader:
        outputs = model_instance(images)

        loss = criterion(
            outputs,
            targets,
        )

        batch_size = targets.size(0)

        total_loss += (
            loss * batch_size
        )

        total_correct += accuracy_top1(
            outputs,
            targets,
        )

        total_samples += batch_size

    return (
        total_loss,
        total_correct,
        total_samples,
    )


def reduce_metrics(
    loss_sum,
    correct_sum,
    sample_sum,
):
    total_loss = xm.mesh_reduce(
        "loss_sum",
        loss_sum.item(),
        sum,
    )

    total_correct = xm.mesh_reduce(
        "correct_sum",
        correct_sum.item(),
        sum,
    )

    total_samples = xm.mesh_reduce(
        "sample_sum",
        sample_sum.item(),
        sum,
    )

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        100.0
        * total_correct
        / total_samples
    )

    return (
        average_loss,
        accuracy,
    )


def worker(index, args):
    torch.manual_seed(
        args.seed
    )

    device = torch.device("xla")

    world_size = 1
    rank = 0

    if rank == 0:
        print(
            f"TPU processes: {world_size}"
        )

        print(
            f"Model: {args.model}"
        )

        print(
            f"Input size: {args.input_size}"
        )

        print(
            f"Epochs: {args.epochs}"
        )

        print(
            f"Per-device batch size: "
            f"{args.batch_size}"
        )

        print(
            f"Global batch size: "
            f"{args.batch_size * world_size}"
        )

    train_dataset, test_dataset = (
        build_datasets(args)
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
    )

    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model_instance = create_model(
        args.model,
        num_classes=100,
        pretrained=False,
        distillation=False,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    global_batch_size = (
        args.batch_size
        * world_size
    )

    scaled_lr = (
        args.lr
        * global_batch_size
        / 512.0
    )

    optimizer = torch.optim.AdamW(
        model_instance.parameters(),
        lr=scaled_lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    )

    if rank == 0:
        os.makedirs(
            args.output_dir,
            exist_ok=True,
        )

    xm.rendezvous(
        "output_dir_created"
    )

    best_accuracy = 0.0

    start_time = time.time()

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        epoch_start = time.time()

        (
            train_loss_sum,
            train_correct_sum,
            train_sample_sum,
        ) = train_one_epoch(
            model_instance,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        (
            train_loss,
            train_accuracy,
        ) = reduce_metrics(
            train_loss_sum,
            train_correct_sum,
            train_sample_sum,
        )

        (
            test_loss_sum,
            test_correct_sum,
            test_sample_sum,
        ) = evaluate(
            model_instance,
            test_loader,
            criterion,
            device,
        )

        (
            test_loss,
            test_accuracy,
        ) = reduce_metrics(
            test_loss_sum,
            test_correct_sum,
            test_sample_sum,
        )

        scheduler.step()

        epoch_time = (
            time.time() - epoch_start
        )

        if rank == 0:
            best_accuracy = max(
                best_accuracy,
                test_accuracy,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc1": train_accuracy,
                "test_loss": test_loss,
                "test_acc1": test_accuracy,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_seconds": epoch_time,
                "best_acc1": best_accuracy,
            }

            print(
                f"\nEpoch {epoch + 1}/"
                f"{args.epochs}"
            )

            print(
                f"Train loss: "
                f"{train_loss:.4f}"
            )

            print(
                f"Train Acc@1: "
                f"{train_accuracy:.2f}%"
            )

            print(
                f"Test loss: "
                f"{test_loss:.4f}"
            )

            print(
                f"Test Acc@1: "
                f"{test_accuracy:.2f}%"
            )

            print(
                f"Epoch time: "
                f"{epoch_time:.2f}s"
            )

            log_path = os.path.join(
                args.output_dir,
                "log.txt",
            )

            with open(
                log_path,
                "a",
            ) as log_file:
                log_file.write(
                    json.dumps(row)
                    + "\n"
                )

            checkpoint_path = os.path.join(
                args.output_dir,
                "checkpoint.pth",
            )

            xm.save(
                {
                    "model": (
                        model_instance.state_dict()
                    ),
                    "optimizer": (
                        optimizer.state_dict()
                    ),
                    "epoch": epoch,
                    "best_acc1": (
                        best_accuracy
                    ),
                },
                checkpoint_path,
            )

    total_time = (
        time.time() - start_time
    )

    if rank == 0:
        print(
            f"\nTraining complete."
        )

        print(
            f"Best Acc@1: "
            f"{best_accuracy:.2f}%"
        )

        print(
            f"Total time: "
            f"{total_time / 60:.2f} min"
        )


def main():
    args = parse_args()

    worker(0, args)


if __name__ == "__main__":
    main()