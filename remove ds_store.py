import os

parent_folder = "root"
filename = ".DS_Store"

def main(in_directory, filename):
    counter = 0
    # walk through the sub directories
    for _path, subdirs, files in os.walk(in_directory):
        if filename in files:
            d=os.path.join(_path, filename)
            os.remove(d) 
            counter += 1
    print(counter, " files deleted.")
    return True

main(parent_folder, filename)