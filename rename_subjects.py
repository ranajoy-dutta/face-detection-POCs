import os
char = r'A'

def rename_subjects(in_directory, suffix):
    counter = 0
    for _path, subdirs, files in os.walk(in_directory):
        if len(subdirs)>0:
            for folder in subdirs:
                src = os.path.join(_path, folder)
                dst = os.path.join(_path, suffix+folder)
                os.rename(src, dst)
                counter+=1
            break
    print(f"{counter} folders renamed.")
    return True

rename_subjects(r'E:\Work\BLAB - Cirg\Casia Dataset Curation\root\train', suffix=char)
# main('eval', filename)