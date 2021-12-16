import cv2
import numpy as np


def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    return img

original = cv2.imread(r"C:\Users\Ranajoy\Downloads\CASIA-WebFace\CASIA-WebFace\0000099\010.jpg")
frame = sharpen_image(original)

cv2.imshow("Image", original)
cv2.imshow("Detail_Enhance", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()