from PIL import Image
import numpy as np

img = Image.open(r"D:\Code\U-Net\output\checkpoints\checkpoints\0.png")
img.show()
img = np.array(img)
img = img[:,:,0]
shape = img.shape

# for i in range(shape[0]):
#     for j in range(shape[1]):
#         if not img[i][j] in lis:
#             frame[i][j]=0
#             lis.append(img[i][j])

print(img.dtype)
print(type(img))
print(img.shape)



