import cv2
import os
import numpy as np
import json
from PIL import Image, ImageDraw


def draw_mask(img_name):
    # 传入的是图片的名字
    img = Image.open('./test/images/' + img_name + '.jpg').convert("RGBA")
    imArray = np.asarray(img)

    json_dir = './test/json/' + img_name + '.json'
    with open(json_dir) as json_file:
        json_data = json.load(json_file)
    objects = json_data['shapes']   # 所有的实例
    polygons = []
    for object in objects:
        polygon = []
        for point in object['points']:
            tmp = (int(point[0]), int(point[1]))
            polygon.append(tmp)
        polygons.append(polygon)
    # polygons = [np.array(object['points'], dtype=int) for object in objects]    # 把所有的object的边界的坐标存到一个list里

    # polygons = np.array(polygons, dtype=int) # 这样改类型会报错，因为不是每一张图片里的实例数目都一样，导致array里各个元素维度不一样
    # print(polygons[0])
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)

    for polygon in polygons:
        ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=255)   # outline为线条颜色，fill为填充颜色
    mask = np.array(maskIm)              # 生成了掩膜，只有多边形区域内为1，其余(含边界)全为0


    cv2.imwrite('./test/mask/' + img_name + '.jpg', mask)


if __name__ == '__main__':
    imgs_name = [i.split('.')[0] for i in os.listdir('./test/images') if i.split('.')[-1] == 'jpg']

    imgs_num = len(imgs_name)
    for index, img_name in enumerate(imgs_name):
        draw_mask(img_name)
        print('{} / {} is done '.format(index, imgs_num))
