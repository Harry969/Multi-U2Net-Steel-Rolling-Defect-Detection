import cv2 as cv
import os
import numpy as np

source_img = 'E:/资料/4-惠州三协/1-语义分割的一个demo/1-数据集/1.3-数据增强/2.3-random_crop_resize/masks/'
resize_path = 'E:/资料/4-惠州三协/1-语义分割的一个demo/1-数据集/1.3-数据增强/2.3-random_crop_resize/masks/'
img_list = os.listdir(source_img)
for img_name in img_list:
    img_path = os.path.join(source_img, img_name)
    img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), cv.IMREAD_COLOR)
    img_resize = cv.resize(img, (480, 480))
    # img_name_new = img_name[:-4] + '_resize.bmp'
    file_path = os.path.join(resize_path, img_name)
    cv.imencode('.jpg', img_resize)[1].tofile(file_path)






