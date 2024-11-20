import os
import random
import shutil

img_folder_path = 'train_data/images'
mask_folder_path = 'train_data/labels'

img_folder_val_path = 'test_data/test_images'
mask_folder_val_path = 'test_data/test_img_labels'
img_name_list = os.listdir(img_folder_path)

val_percent = 0.15  # 0

num = len(img_name_list)
list_index = range(num)

train_num = int(num * val_percent)

train = random.sample(list_index, train_num)

for index, number in enumerate(train):
    img_name = img_name_list[number]
    img_path = os.path.join(img_folder_path, img_name)
    mask_path = os.path.join(mask_folder_path, img_name)
    img_new_path = os.path.join(img_folder_val_path, img_name)
    mask_new_path = os.path.join(mask_folder_val_path, img_name)

    shutil.move(img_path, img_new_path)
    shutil.move(mask_path, mask_new_path)

