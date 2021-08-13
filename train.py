import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from model import UNet
from dataloader import get_loaders
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from utils import (load_checkpoint, save_checkpoint, check_accuracy, save_predictions_as_imgs, DiceLoss)

# Hyperparameters etc.
LEARNING_RATE = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 6
NUM_EPOCHS = 20
NUM_WORKER = 8
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768
PIN_MEMORY = True
LOAD_MODEL = False
IMG_DIR = r"E:\code\LY\data\sewage\small_dataset\small_label"
MASK_DIR = r"E:\code\LY\data\sewage\small_dataset\small_label"


def train_fn(loader, model, optimizer, loss_fn, epoch, scaler):
    loop = tqdm(loader)
    loop.set_description(f"Epoch {epoch} train")
    losses = []
    for data, targets in loop:
        data = data.to(DEVICE)
        targets = targets.float().unsqueeze(1).to(DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            losses.append(loss)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # # forward
        # predictions = model(data)
        # loss = loss_fn(predictions, targets)
        # losses.append(loss)
        #
        # # backward
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    #compute loss mean
    logging.info(f"Epoch {epoch}'s mean train loss:{sum(losses)/len(losses)}")

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.5330, 0.5463, 0.5493],
                std=[0.1143, 0.1125, 0.1007],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.5330, 0.5463, 0.5493],
                std=[0.1143, 0.1125, 0.1007],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNet(retain_dim=True, out_size=(IMAGE_HEIGHT, IMAGE_WIDTH)).to(DEVICE)
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = DiceLoss()
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

    if LOAD_MODEL:
        load_checkpoint(torch.load("output/checkpoints/checkpoint.pth"), model)
        check_accuracy(val_loader, model, device=DEVICE)

    best_score = 0
    scalar = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, epoch, scalar)

        # every epoch check accuracy and save the best model
        iou_score = check_accuracy(val_loader, model, device=DEVICE, epoch=epoch)
        if iou_score >= best_score:
            save_checkpoint(model, optimizer, filename=f"output/checkpoints/best_checkpoint.pth")
            best_score = iou_score

    logging.info(f"Train done.\nThe best iou score: {best_score}.")
    # save model
    save_checkpoint(model, optimizer, filename="output/checkpoints/checkpoint.pth")

    # check accuracy
    check_accuracy(val_loader, model, device=DEVICE)

    # print some samples to folder
    save_predictions_as_imgs(val_loader, model, device=DEVICE)


if __name__ == "__main__":
    main()
