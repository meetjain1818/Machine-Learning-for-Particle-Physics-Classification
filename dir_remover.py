import os
import shutil
#Run this cell if you want to delete existing folder with the same name

dir_path = input('Paste here the complete file path of directory to be deleted: ') #Path of the directory to delete

# Check if the directory exists
if os.path.exists(dir_path) and os.path.isdir(dir_path):
    # Delete the directory and all its contents
    shutil.rmtree(dir_path)
    print(f'Directory {dir_path} has been deleted.')
else:
    print(f'Directory {dir_path} does not exist.')
