import cv2
import torch
from torch.autograd import Variable

import numpy as np
from PIL import Image

from data_loader import RescaleT, Rescale
from data_loader import ToTensorLab

from model.u2net import U2NET, U2NETP
from skimage import io
import os
import time


def eval_main(model, sample, img_size):
    for i_test, data_test in enumerate([sample]):
        inputs_test = data_test['image']

        inputs_test = inputs_test.unsqueeze(0)
        inputs_test = inputs_test.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
        t1 = time.time()
        d1, d2, d3, d4, d5, d6, d7 = model(inputs_test)
        t2 = time.time()
        print(t2-t1)
        d1 = torch.softmax(d1, dim=1)
        predict_np = torch.argmax(d1, dim=1, keepdim=True)
        predict_np = predict_np.cpu().detach().numpy().squeeze()

        predict_np = cv2.resize(predict_np, img_size, interpolation=cv2.INTER_NEAREST)

        # red (255,0,0) b (0,0,255)
        cls = dict([(1, (0, 0, 255)),  # 蓝色
                    (2, (255, 0, 255)),  # 中间的那个，要对应粉红色
                    (3, (0, 255, 0)),  # 绿色
                    (4, (255, 0, 0)),  # 极耳 红色
                    (5, (255, 255, 0))])  # 双面胶 黄色


        r = predict_np.copy()
        b = predict_np.copy()
        g = predict_np.copy()
        for c in cls:
            r[r == c] = cls[c][0]
            g[g == c] = cls[c][1]
            b[b == c] = cls[c][2]

        rgb = np.zeros((img_size[1], img_size[0], 3))
        print('类别', np.unique(predict_np))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        Image.fromarray(rgb.astype(np.uint8)).save('test_out/' + '969.png')
        # Image.fromarray(rgb.astype(np.uint8)).save('test_out/' + '852.png')
        del d1, d2, d3, d4, d5, d6, d7


def label_deal(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])
    if 3 == len(label_3.shape):
        label = label_3[:, :, 0]
    elif 2 == len(label_3.shape):
        label = label_3
    if 3 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    elif 2 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    return label


if __name__ == "__main__":
    # # --------- 1. get image and model ---------
    model_dir = 'saved_models/u2netp/u2netp.pth'
    image = io.imread('E:/1-资料/6-工作/2-惠州三协/1-项目/1-深度学习/4-槟榔裁剪-finish-with-finish/2-模型训练/multi_U2NET/datasets/source_test_data/images/969.png')
    # image = io.imread('datasets_ButtonCell/test_data/images/Image_20230420111440899.bmp')

    if not os.path.exists('test_out/'):
        os.makedirs('test_out/')

    # --------- 2. pre-deal data ---------
    # 这一步是用来创建一个  跟 原图  一样的  shape 的 全 0 矩阵
    label = label_deal(image)
    if 2 == len(image.shape) and 2 == len(label.shape):
        image = image[:, :, np.newaxis]
    imidx = np.array([0])

    sample = {'imidx': imidx, 'image': image, 'label': label}
    # deal1 = RescaleT(512)
    # deal1 = Rescale(288)
    deal2 = ToTensorLab(flag=0)
    # sample = deal1(sample)
    sample = deal2(sample)

    # --------- 3. model define ---------
    net = U2NETP(3, 3)
    model = torch.load(model_dir)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    # --------- 4. model infer ---------
    model = net.eval()
    eval_main(model, sample, (image.shape[1], image.shape[0]))
