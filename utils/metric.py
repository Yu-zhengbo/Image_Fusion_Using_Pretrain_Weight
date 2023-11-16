import numpy as np
import math
from skimage.metrics import structural_similarity

def avgGradient(image):
    ag_x = image[1:,:] - image[:-1,:]
    ag_y = image[:,1:] - image[:,:-1]
    return np.mean(np.sqrt((ag_x[:,:-1]**2+ag_y[:-1,:]**2)/2))

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def ssim(img1, img2, channel_axis=2, data_range=1):
    return structural_similarity(img1, img2, channel_axis=channel_axis, data_range=data_range)


def mse(img1, img2):
    return np.mean( (img1 - img2) ** 2 )

if __name__ == '__main__':
    # img = np.random.randint(0,256,(640,640,3))
    img = np.random.uniform(0,1,(640,640,3))
    print(avgGradient(img))
    print(psnr(img,img))
    print(ssim(img,img,channel_axis=2,data_range=1))
    print(mse(img, img))
