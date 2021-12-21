# import the necessary packages
import cv2

blur_threshold = 100.0
filepath = r"E:\Work\BLAB - Cirg\Casia Dataset Curation\Outputs\cropped_face_dataset_HAAR\0000102\274.jpg"


def blur_rating(filepath, image=None):
    if image:
        pass
    else:
        image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


print(blur_rating(filepath))