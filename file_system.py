import os
import shutil


def get_immediate_subdirectories(path):
    """
    Description: Get the immediate subdirectories in a specific path: path
    Reference:   https://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
    """
    return [name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))]

def get_all_files(path):
    """
    Description: Get the immediate subdirectories in a specific path: path
    Reference:   https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    """
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def get_all_files_with_extension(path, ext = ".*"):

    if ext == ".*":
        return get_all_files(path)
    else:
        return [f for f in os.listdir(path) if f.endswith(ext)]
