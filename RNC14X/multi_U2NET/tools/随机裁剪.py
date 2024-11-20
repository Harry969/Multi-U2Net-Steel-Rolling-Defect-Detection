# import numpy as np
# import random
# import cv2
#
# image_path = "E:/资料/8-惠州三协/1-语义分割的一个demo/1-数据集/1-数据增强/1-原size的图片翻转/masks/1.bmp"
#
#
# def random_crop(image, crop_shape, padding=None):
#     img_h = image.shape[0]
#     img_w = image.shape[1]
#     img_d = image.shape[2]
#
#     if padding:
#         oshape_h = img_h + 2 * padding
#         oshape_w = img_w + 2 * padding
#         img_pad = np.zeros([oshape_h, oshape_w, img_d], np.uint8)
#         img_pad[padding:padding + img_h, padding:padding + img_w, 0:img_d] = image
#
#         nh = random.randint(0, oshape_h - crop_shape[0])
#         nw = random.randint(0, oshape_w - crop_shape[1])
#         image_crop = img_pad[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
#
#         return image_crop
#     else:
#         print("WARNING!!! nothing to do!!!")
#         return image
#
#
# if __name__ == "__main__":
#     image_src = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
#     crop_width = image_src.shape[0] - 24
#     crop_height = image_src.shape[1] - 24
#     image_dst_crop = random_crop(image_src, [crop_width, crop_height], padding=10)
#     cv2.imwrite('save.jpg', image_dst_crop)
#
#     # cv2.imshow("oringin image", image_src)
#     # cv2.imshow("crop image", image_dst_crop)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()


from PIL import Image
import os
import random

# 定义待批量裁剪图像的路径地址
IMAGE_INPUT_PATH = 'E:/资料/4-惠州三协/1-语义分割的一个demo/1-数据集/1.2-数据增强/0-原size的图片翻转/masks/'
source_imgs_path = 'E:/资料/4-惠州三协/1-语义分割的一个demo/1-数据集/1.2-数据增强/0-原size的图片翻转/images/'
# 定义裁剪后的图像存放地址
IMAGE_OUTPUT_PATH = 'E:/资料/4-惠州三协/1-语义分割的一个demo/1-数据集/1.2-数据增强/2.3-random_crop_and_resize/masks'
source_output_path = 'E:/资料/4-惠州三协/1-语义分割的一个demo/1-数据集/1.2-数据增强/2.3-random_crop_and_resize/images'
for each_image in os.listdir(IMAGE_INPUT_PATH):
    # 每个图像全路径
    image_input_fullname = IMAGE_INPUT_PATH + "/" + each_image
    source_img_path = source_imgs_path + '/' + each_image
    # PIL库打开每一张图像
    img = Image.open(image_input_fullname)
    img_source = Image.open(source_img_path)
    # 定义裁剪图片左、上、右、下的像素坐标
    x_max = img.size[0]
    y_max = img.size[1]

    mid_point_x = int(x_max / 2)
    mid_point_y = int(y_max / 2)

    down = mid_point_y + random.randint(0, mid_point_y)
    up = mid_point_y - (down - mid_point_y)
    right = mid_point_x + (down - mid_point_y)
    left = mid_point_x - (down - mid_point_y)

    BOX_LEFT, BOX_UP, BOX_RIGHT, BOX_DOWN = left, up, right, down
    # 从原始图像返回一个矩形区域，区域是一个4元组定义左上右下像素坐标
    box = (BOX_LEFT, BOX_UP, BOX_RIGHT, BOX_DOWN)
    # 进行roi裁剪, 这里要分成   mask  和  img 两种图片的去同时进行裁剪保存
    roi_area = img.crop(box)
    roi_area_source = img_source.crop(box)
    # 裁剪后每个图像的路径+名称
    new_name_img = each_image[:-4] + '_random_crop_3.bmp'
    image_output_fullname = IMAGE_OUTPUT_PATH + "/" + new_name_img
    source_img_output_fullname = source_output_path + "/" + new_name_img
    # 存储裁剪得到的图像
    roi_area.save(image_output_fullname)
    roi_area_source.save(source_img_output_fullname)
    print('{0} crop done.'.format(each_image))
