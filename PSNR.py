import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np
from skimage import transform


img1 = cv2.imread(r'')
img2 = cv2.imread(r'')
img1 = transform.resize(img1, (512, 512))
img2 = transform.resize(img2, (512, 512))

if __name__ == "__main__":
    print(psnr(img1, img2))