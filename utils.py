import torchvision
import torch
from tqdm import tqdm
import torch.nn as nn
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        mIoU = np.nanmean(self.IntersectionOverUnion())  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


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
        iou = np.array([0.,0.])
        pa = 0
        for x, y in loop:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()  # 这是一个和原来的preds形状相同的的全是1.和0.的向量
            preds, y = preds.cpu().numpy(), y.cpu().numpy()  # 转化为ndarray类型才能进行计算
            preds, y = preds.astype(np.int32), y.astype(np.int32)  # 转化为整型
            evaluator = SegmentationMetric(2)
            evaluator.addBatch(preds, y)
            iou += np.array(evaluator.IntersectionOverUnion())
            pa += evaluator.pixelAccuracy()
            # num_correct += (preds == y).sum()
            # num_pixels += torch.numel(preds)
            # dice_score += (2 * (preds * y).sum()) / (
            #         (preds + y).sum() + 1e-8)

    logging.info(
        f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
    )
    logging.info(f"Dice score: {dice_score / len(loader)}")
    logging.info(
        f"Got acc: {pa / len(loader) * 100:.2f}\n",
        f"IOU-0: {(iou / len(loader))[0]}, IOU-1: {(iou / len(loader))[1]}",
        f"mIOU: {np.mean(iou / len(loader))}"
    )
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
