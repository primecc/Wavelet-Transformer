from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np


def calculate_ssim(image1_path, image2_path, multichannel=False):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        raise ValueError("error!")
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    if multichannel or img1.shape[-1] > 1:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        channel_axis = 2
    else:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        channel_axis = None

    return ssim(img1, img2,
                data_range=255,
                channel_axis=channel_axis)



if __name__ == "__main__":

    img_path1 = ""
    img_path2 = ""
    try:
        ssim_value = calculate_ssim(img_path1, img_path2, multichannel=True)
        print(f"SSIM : {ssim_value:.4f}")
    except Exception as e:
        print(f"error: {str(e)}")
