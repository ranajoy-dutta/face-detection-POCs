# HAAR Cascade

# # import required module
import os
import numpy as np
import cv2
import pandas as pd



# input directory relative path
in_directory = r"dataset"  #input directory having subdirectories of each subject.
out_directory = r'Outputs\cropped_face_dataset_HAAR'      #output folder path
only_one_face = False       #False refers to the first detection with max confidence. True refers only those images will be processed which have 1 face detection. 
detection_confidence_threshold = 0.8        #face detection confidence threshold
adjust_brightness = 0       # increase brightness. 0 means no adjustment

# define model paths
current_script_directory = os.path.dirname(os.path.realpath(__file__))
detector_folder = os.path.join(current_script_directory, "face_detection_model")


print("[INFO] Initializing Models...")
# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([detector_folder, "deploy.prototxt"])
modelPath = os.path.sep.join([detector_folder,
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
print("[INFO] Models Initialized...")

# load HAAR cascade model
face_cascade = cv2.CascadeClassifier(r'face_detection_model\haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier(r'face_detection_model\haarcascade_frontalface_alt2.xml')

def find_faces_using_DNN(image_path): 
    """This function finds faces in given image.

    Args:
        image_path ([string]): Path of image

    Returns:
        np.ndarray: ndarray of face. Returns False 
        if no or more faces are found at the end of 
        processing the given image.
    """    
    # construct a blob from the image
    image = cv2.imread(image_path)
    # if image is empty or not an image
    if not isinstance(image, np.ndarray):
        # print("[WARN] empty image")
        return False
    face_list=[]
        
    (h, w) = image.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    # loop over the detections (face Detection)

    for i in range(0, detections.shape[2]):

        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > detection_confidence_threshold:
        # if confidence > 0.8:
            # compute the (x, y)-coordinates of the bounding box for the
            # face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20 or len(face)==0:
                # print("[INFO] Face too small for prediction or not a face")
                pass
            else:
                face_list.append(face)
    return face_list


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
        face = image[y:y + h, x:x + w]
        face_list.append(face)
    return face_list
    
    
    
    
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

# autocontrast

def curate(src, dst, filename, algo="DNN"):
    if algo=="DNN":
        image_array_list = find_faces_using_DNN(src)
    elif algo=="HAAR":
        image_array_list = find_faces_using_HAAR(src)
    else:
        raise Exception("Invalid Algo for Face Detection")
    
    num_faces_found = 0
    status = "skipped"
    if len(image_array_list)>0:
        num_faces_found = len(image_array_list)
        if (num_faces_found==1 and only_one_face==True) or only_one_face == False:
            first_face = image_array_list[0]
            if isinstance(first_face, np.ndarray) and len(first_face)>0:
                if adjust_brightness>0:
                    first_face = increase_brightness(first_face, adjust_brightness)
                os.makedirs(dst, exist_ok=True)
                cv2.imwrite(os.path.join(dst, filename), first_face)
                status = "processed"
    return num_faces_found, status
        


def main(in_directory, out_directory, algo="DNN", xlsx_filename="result.xlsx"):
    df = pd.DataFrame(columns=["Subject", "filename", "faces_found", "status"])
    # walk through the sub directories
    counter = 0
    for _path, subdirs, files in os.walk(in_directory):
        basename = os.path.basename(_path)
        dst = os.path.join(out_directory, basename)  
        src = None
        status = "skipped"
        
        for file in files:
            num_faces_found = 0
            new_row = {'Subject':basename, 'filename':os.path.basename(file), 'faces_found':num_faces_found}
            if file.endswith('jpg'):
                src = os.path.join(_path, file)
                new_row['faces_found'], new_row["status"] = curate(src, dst, file, algo=algo) 
            df = df.append(new_row, ignore_index=True)
        counter += 1
        print(f"{counter}. Processed {basename}")
        df.to_excel(xlsx_filename)
    return df

 
if __name__=="__main__":
    excel_filename = "result_HAAR.xlsx"         # defaults to result.xlsx
    algo="HAAR"     # HAAR, DNN. Defaults to DNN
    df = main(in_directory, out_directory, algo=algo, xlsx_filename=excel_filename)