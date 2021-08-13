import torchvision
import torch
from tqdm import tqdm
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Dice损失函数
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        return score


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


def check_accuracy(loader, model, device, epoch=0):
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
    return dice_score/len(loader)


def save_predictions_as_imgs(loader, model, folder="output/saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
        break

    model.train()
