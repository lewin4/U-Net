import torchvision.transforms as transforms
from PIL import Image

image_path = "10_01.png"
img = Image.open(image_path)
transform = transforms.Compose([
    transforms.ToTensor()
])

img_tr = transform(img)
# calculate mean and std
mean, std = img_tr.mean([1, 2]), img_tr.std([1, 2])#计算第二维和第三维的像素的，不计算第一维的通道维

# print mean and std
print("mean and std before normalize:")
print("Mean of the image:", mean)
print("Std of the image:", std)