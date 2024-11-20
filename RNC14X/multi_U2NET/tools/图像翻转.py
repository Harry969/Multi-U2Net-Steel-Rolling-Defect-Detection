
import cv2
import os
import numpy as np

images = 'C:/Users/ASUS/Desktop/Pytorch-UNet-master/data/masks/'
save_path = 'E:/资料/8-惠州三协/1-语义分割的一个demo/1-数据集/1-数据增强/2-mask/2/'
video_path_list = os.listdir(images)
for s in video_path_list:
    # base data
    i = os.path.join(images + s)
    # img = cv2.imread(i)
    img = cv2.imdecode(np.fromfile(i, dtype=np.uint8), cv2.IMREAD_COLOR)
    # trans_img = cv2.transpose(img)
    # new_img = cv2.flip(trans_img, -2)#  0是逆时针90度，1是顺时针旋转90度，-1是左右翻转后

    new_img = cv2.flip(img, -1)  # 0是上下翻转，1是左右翻转，-1是上下左右翻转
    s_new = str(s)[:-4] + '_2.bmp'
    file_path = os.path.join(save_path, s_new)
    # cv2.imwrite(i, new_img)
    cv2.imencode('.jpg', new_img)[1].tofile(file_path)
