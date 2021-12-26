import os
import shutil

def main(in_directory, min_image_threshold):
    counter = 0
    # walk through the sub directories
    subject_list = os.listdir(in_directory)
    for subject in subject_list:
        subject_path = os.path.join(in_directory, subject)
        if os.path.isdir(subject_path) and len(os.listdir(subject_path))<min_image_threshold:
            shutil.rmtree(subject_path) 
            counter += 1
    print(counter, " subjects removed.")
    return True


parent_folder = r"Outputs\cropped_face_dataset_HAARdef2312_80"
min_image_threshold = 5
main(parent_folder, min_image_threshold)
