import cv2
import numpy as np

img1 = cv2.imread('')
img2 = cv2.imread('')

diff = np.square(np.subtract(img1, img2))

rmse = np.sqrt(np.mean(diff))
print("RMSE: ", rmse)