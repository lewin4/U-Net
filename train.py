import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet
from dataloader import SewageDataset
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2


# from utils import (load_checkpoint, save_checkout, get_loaders, check_accuracy, save_predictions_as_imgs,)

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
    for batch_id, (data, targets) in tqdm(loader):
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
        tqdm(loader).set_postfix(loss=loss.item())




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

    val_transforms = A.Compose(
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

    train_dataset = SewageDataset(IMG_DIR, MASK_DIR, transform=train_transform)
    img, label = train_dataset[195]
    print(len(train_dataset))
    print(type(img),img.shape)
    print(type(label), label.shape)
    unloader = transforms.ToPILImage()

    def imshow(tensor, title=None):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    def channels(tensor,):
        channel = {"0":0, "1":0, "2":0}
        shape = tensor.shape
        print(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if tensor[i][j].item() == 0:
                    channel["0"]+=1
                elif tensor[i][j].item() == 1:
                    channel["1"]+=1
                else:
                    channel["2"] += 1
        return channel

    def channelss(tensor,):
        channel = []
        shape = tensor.shape
        print(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if not tensor[i][j] in channel:
                    channel.append(tensor[i][j])
                    print(i,j,tensor[i][j])

        return channel

    def tensor_to_np(tensor):
        img = tensor.mul(255).byte()
        img = img.cpu().numpy().transpose((1, 2, 0))
        print(img.shape)
        return img

    def show_from_cv(img, title=None):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(img)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

    def tensor_to_nplabel(tensor):
        img = tensor.mul(255).byte()
        img = img.cpu().numpy()
        print(img.shape)
        return img






    # print(channels(label))
    # print(channelss(label))
    # imshow(img)
    # imshow(label)
    show_from_cv(tensor_to_np(img), "numpy_img")
    show_from_cv(tensor_to_nplabel(label), "numpy_label")




    # model = UNet().to(device=DEVICE)
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #
    # train_loader, val_loader = get_loaders(
    #     IMG_DIR,
    #     MASK_DIR,
    #     BATCH_SIZE,
    #     NUM_WORKER,
    #     PIN_MEMORY,
    #     train_transform,
    #     val_transforms,
    # )
    #
    # scaler = torch.cuda.amp.GradScaler()
    # for epoch in range(NUM_EPOCHS):
    #     train_fn(train_loader, model, optimizer, loss_fn, scaler)

    # save model
    # check accuracy
    # print some samples to folder


if __name__ == "__main__":
    main()