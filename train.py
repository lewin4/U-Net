import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet
from dataloader import SewageDataset
from dataloader import get_loaders
import torchvision.transforms as transforms

from utils import (load_checkpoint, save_checkpoint, check_accuracy, save_predictions_as_imgs,)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKER = 8
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768
PIN_MEMORY = True
LOAD_MODEL = False
IMG_DIR = r"D:\Code\data\sewage\small_dataset\small_img"
MASK_DIR = r"D:\Code\data\sewage\small_dataset\small_single_label"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_id, (data, targets) in loop:
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(deviec=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNet().to(device=DEVICE)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        IMG_DIR,
        MASK_DIR,
        BATCH_SIZE,
        NUM_WORKER,
        PIN_MEMORY,
        train_transform,
        val_transform,
    )

    scalar = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scalar)

    # save model
    checkpoint = {
        "checkpoint":model.state_dict(),
        "optimizer":optimizer.state_dict()
    }
    save_checkpoint(checkpoint)

    # check accuracy
    check_accuracy(val_loader, model, DEVICE)

    # print some samples to folder
    save_predictions_as_imgs(val_loader, model, device=DEVICE)


if __name__ == "__main__":
    main()
