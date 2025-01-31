import shutil
dir_name = input('Enter the directory which you want to zip: ')
output_filename = input('Enter file name with which you want to save the file: ')

shutil.make_archive(output_filename, 'zip', dir_name)
print('File succesfully Zipped...')