import torchvision
import torch
from tqdm import tqdm

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def save_checkpoint(model, optimizer, filename):
    checkpoint = {
        "checkpoint": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    logging.info("Saving checkpoint")
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint, model):
    logging.info("Loading checkpoint")
    model.load_state_dict(checkpoint["checkpoint"])


def check_accuracy(loader, model, device, epoch = 0):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        loop = tqdm(loader)
        loop.set_description(f"Epoch {epoch} val")
        for x, y in loop:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()  # 这是一个和原来的preds形状相同的的全是1.和0.的向量
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8)

    logging.info(
        f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
    )
    logging.info(f"Dice score: {dice_score / len(loader)}")
    model.train()


def save_predictions_as_imgs(
        loader, model, folder="output/saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
# if __name__ == "__main__":
#     x = torch.tensor([0,0,0,0,1,1,1,1,1,1])
#     x = (x>0.5)
#     print(x)
