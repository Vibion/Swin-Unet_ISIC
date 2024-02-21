import os

file_image = os.listdir('../data/test/images')
file_mask = os.listdir('../data/test/masks')

print(len(file_image))
print(len(file_mask))

for file in file_image:
    ext = file.split('.')[-1]
    if ext == 'txt':
        print(file)