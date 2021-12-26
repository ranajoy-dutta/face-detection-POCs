import cv2
import os
import numpy as np


# load HAAR cascade model
# face_cascade = cv2.CascadeClassifier(r'face_detection_model\haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(r'face_detection_model\haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier(r'face_detection_model\haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(r'face_detection_model\haarcascade_smile.xml')


def find_faces_using_HAAR(image_path):
    """This function finds faces in given image.

    Args:
        image_path ([string]): Path of image

    Returns:
        np.ndarray: ndarray of face. Returns False 
        if no or more faces are found at the end of 
        processing the given image.
    """  
    image = cv2.imread(image_path)
    # if image is empty or not an image
    if not isinstance(image, np.ndarray):
        # print("[WARN] empty image")
        return False
    face_list=[]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        face = image[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(face_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.rectangle(face,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow("Detail_Enhance", face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if len(eyes)==2:
            face = image[y:y+h, x:x+w]
            face_list.append(face)
    return face_list
    
filename = r"E:\Work\BLAB - Cirg\Casia Dataset Curation\dataset\0000100\0.jpg"
res = find_faces_using_HAAR(filename)
if len(res)>0:
    cv2.imshow("Detail_Enhance", res[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()