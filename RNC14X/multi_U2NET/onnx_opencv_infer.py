import numpy as np

import cv2 as cv
from PIL import Image
import os
from skimage import transform
import torch
import time


# tensor变量转ndarray
def ToTensorLab(image):
    # 生成一个  全0的3维数组
    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
    image = image / np.max(image)
    if image.shape[2] == 1:
        tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
    else:
        tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229  # r
        tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224  # g
        tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225  # b


    # 对 tmpImg 做一次 transpose 就把 tmpimg 给返回出去
    tmpImg = tmpImg.transpose((2, 0, 1))

    return torch.from_numpy(tmpImg.copy())


def onnx_infer(image_path, image_name):
    # opencv 读取 路径得到 img,    bgr  形式
    # img_cv = cv.imdecode(np.fromfile(image_path, dtype=np.uint8), cv.IMREAD_COLOR)
    img_cv = cv.imread(image_path)
    # 对 img_cv 进行通道拆分，再重组, 转成  rgb  形式
    # b, g, r = cv.split(img_cv)
    # image = cv.merge([r, g, b])
    image = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)
    # cv.imencode('.png', image)[1].tofile('C:/Users/ASUS/Desktop/tmp_2.png')
    # 这一步主要做的是 将 image resize 成 (320, 320 3),
    # 同时对所有的像素点都做一次归一化的处理
    # 如果是直接 resize 成 320*320 的shape，会onnx的模型input shape不符合，符合的应该是(1, 1, 320, 320)
    # onnx 模型接受的输入是  ndarray 的类型
    #                               高,  宽
    # img = transform.resize(image, (320, 320), mode='constant')  # 3 通道   (320, 320, 3)
    img = transform.resize(image, (1774, 2534), mode='constant')  # 3 通道   (320, 320, 3)
    # 做了一个值的调整，并转成 torch.Size([3, 320, 320]) 的尺寸
    sample = ToTensorLab(img)  # torch.Size([3, 320, 320])
    # 升维，再转成  nd.array 的形式，并送入  onnx  的模型当中
    inputs_test = sample.unsqueeze(0)  # torch.Size([1, 3, 320, 320]
    inputs_test = inputs_test.type(torch.FloatTensor)
    img = inputs_test.numpy()  # (1, 3, 320, 320)

    onnx_model = "saved_models/u2netp/u2netp.onnx"
    # onnx_model = "saved_models/u2net4p/u2net4p_bce_itr_13750-83-85_new.onnx"
    net = cv.dnn.readNetFromONNX(onnx_model)
    # Run a model
    net.setInput(img)
    # out 是个 numpy.ndarray 的类型, 1806 是 网络的其中一个输出层的id，选择这个层的输出作为这一个项目中  onnx  模型的推理结果输出
    time_start = time.time()
    out = net.forward('1802')  # (1, 1, 320, 320)   ndarray
    # out = net.forward('1876')  # (1, 1, 320, 320)   ndarray  # version 14 槟榔得原图还能到 0.6 秒以内
    time_stop = time.time()
    print(out.shape)
    print('totally cost', time_stop - time_start)
    # predict_np = torch.argmax(torch.tensor(out), dim=1, keepdim=True)
    # predict_np = predict_np.cpu().detach().numpy().squeeze()
    # predict_np = predict_np.squeeze()
    predict_np = out.squeeze()

    predict_np = cv.resize(predict_np, (img_cv.shape[1], img_cv.shape[0]), interpolation=cv.INTER_NEAREST)
    # 这个  序号 和 训练用的mask图的那个类别编号是一致的
    # cls = dict([(1, (0, 255, 0)),  # 绿色
    #             (2, (255, 0, 0)),
    #             (3, (255, 0, 255)),
    #             (4, (255, 255, 0)),
    #             (5, (0, 0, 255))])
    cls = dict([(1, (128, 0, 128)),
                (2, (255, 255, 0))])

    r = predict_np.copy()
    b = predict_np.copy()
    g = predict_np.copy()
    for c in cls:
        r[r == c] = cls[c][0]
        g[g == c] = cls[c][1]
        b[b == c] = cls[c][2]
    rgb = np.zeros((img_cv.shape[0], img_cv.shape[1], 3))
    print('类别', np.unique(predict_np))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    # Image.fromarray(rgb.astype(np.uint8)).save('./test_out/' + 'F5D2392BQE81FRFAE+234G_224539_210' + '.png')
    Image.fromarray(rgb.astype(np.uint8)).save('./test_out/' + image_name)


if __name__ == '__main__':
    # onnx_infer('datasets/180112_303.jpg')
    # images_path = 'datasets_ButtonCell/test_data/image'
    images_path = 'datasets/source_test_data/images'
    images_list = os.listdir(images_path)
    for image_name in images_list:
        image_path = os.path.join(images_path, image_name)
        onnx_infer(image_path, image_name)
