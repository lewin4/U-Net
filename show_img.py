from PIL import Image
import numpy as np
from model import UNet
import torch
from dataloader import SewageDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_DIR = r"D:\Code\data\sewage\small_dataset\small_img"
MASK_DIR = r"D:\Code\data\sewage\small_dataset\small_label"
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768

train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # A.Rotate(limit=35, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            # A.Normalize(
            #     mean=[0.5330, 0.5463, 0.5493],
            #     std=[0.1143, 0.1125, 0.1007],
            #     max_pixel_value=255.0,
            # ),
            # ToTensorV2(),
        ],
    )

dataset = SewageDataset(IMG_DIR, MASK_DIR, train=True, transform=train_transform)
img, label = dataset[0]
label *= 120
img = Image.fromarray(img)
label = Image.fromarray(label)
label.show()
img.show()

# img = img.unsqueeze(0)
# label = label.unsqueeze(0)
# print(img.dtype)
# print(type(img))
# print(img.shape)
#
# print(label.dtype)
# print(type(label))
# print(label.shape)

# zero = 0
# one = 0
# zero = (label == 0).sum()
# one =(label==1).sum()
#
# print(zero, one)

# img = np.array(img)
# img = img[:,:,0]
# shape = img.shape

# for i in range(shape[0]):
#     for j in range(shape[1]):
#         if not img[i][j] in lis:
#             frame[i][j]=0
#             lis.append(img[i][j])

# model = UNet(retain_dim=True, out_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
#
# with torch.no_grad():
#     model.eval()
#     pred = torch.sigmoid(model(img))
#     pred = (pred > 0.5).float()
#     print(pred.dtype)
#     print(type(pred))
#     print(pred.shape)
#     print(pred)
#     print(pred.mean())







