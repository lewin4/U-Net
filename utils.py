import torchvision
import torch
from dataloader import SewageDataset

import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


def save_checkpoint(state, filename = "output/checkpoints/checkpoint.pth"):
    logging.info("Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    logging.info("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def check_accuracy(loader, model, device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds>0.5).float()     #这是一个和原来的preds形状相同的的全是1.和0.的向量
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
    )
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()


def save_predictions_as_imgs(
    loader, model, folder="output/saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
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

