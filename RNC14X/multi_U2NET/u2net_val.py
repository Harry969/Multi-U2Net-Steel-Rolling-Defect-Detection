import os

from PIL import Image
from tqdm import tqdm

from utils_metrics import compute_mIoU, show_results

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import numpy as np
from data_loader import RescaleT, Rescale
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from model.u2net import U2NETP
import cv2 as cv
from skimage import io


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


def eval_print_miou(num_classes, name_classes, images_path, gt_dir, pred_dir, predict_label, miou_out_path, model_dir):
    miou_mode = 1

    images_list = os.listdir(images_path)

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model ", model_dir)

        net = U2NETP(3,num_classes )
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_dir))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        model = net.eval()
        print("Load model done.")

        print("Get predict result.")
        for image_name in tqdm(images_list):
            image_path = os.path.join(images_path, image_name)
            image = io.imread(image_path)
            label = label_deal(image)
            if 2 == len(image.shape) and 2 == len(label.shape):
                image = image[:, :, np.newaxis]
            imidx = np.array([0])

            sample = {'imidx': imidx, 'image': image, 'label': label}
            deal1 = RescaleT(512)
            deal2 = ToTensorLab(flag=0)
            sample = deal1(sample)
            sample = deal2(sample)

            for i_test, data_test in enumerate([sample]):
                inputs_test = data_test['image']
                inputs_test = inputs_test.unsqueeze(0)
                inputs_test = inputs_test.type(torch.FloatTensor)

                if torch.cuda.is_available():
                    inputs_test = Variable(inputs_test.cuda())
                else:
                    inputs_test = Variable(inputs_test)
                d1, d2, d3, d4, d5, d6, d7 = model(inputs_test)
                d1 = torch.softmax(d1, dim=1)
                predict_np = torch.argmax(d1, dim=1, keepdim=True)   # ？？
                predict_np = predict_np.cpu().detach().numpy().squeeze()

                predict_np = cv.resize(predict_np, (image.shape[1], image.shape[0]), interpolation=cv.INTER_NEAREST)

                cv.imwrite(pred_dir + str(image_name)[:-4] + '.png', predict_np)
                r = predict_np.copy()
                b = predict_np.copy()
                g = predict_np.copy()
                # red (255,0,0) b (0,0,255)
                cls = dict([(1, (0, 0, 255)),  # 蓝色
                            (2, (255, 0, 255)),  # 中间的那个，要对应粉红色
                            (3, (0, 255, 0)),  # 绿色
                            (4, (255, 0, 0)),  # 极耳 红色
                            (5, (255, 255, 0))])  # 双面胶 黄色
                for c in cls:               # 这个我懂了
                    r[r == c] = cls[c][0]
                    g[g == c] = cls[c][1]
                    b[b == c] = cls[c][2]
                rgb = np.zeros((image.shape[0], image.shape[1], 3))
                # print('类别', np.unique(predict_np))
                rgb[:, :, 0] = r
                rgb[:, :, 1] = g
                rgb[:, :, 2] = b
                # Image.fromarray(rgb.astype(np.uint8)).save(predict_label + str(image_name)[:-4] + '.bmp')
                img = cv.addWeighted((rgb.astype(np.uint8)), 0.15, image, 1, 0)  # ？
                cv.imwrite(predict_label + str(image_name)[:-4] + '.png', img)
                del d1, d2, d3, d4, d5, d6, d7
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, images_list, num_classes, name_classes)
        print("Get miou done.")
        print(IoUs, num_classes)
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)


if __name__ == "__main__":
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #   分类个数+1、如2+1
    # num_classes = 3
    # name_classes = ["background", "green", "red"]
    num_classes = 4
    name_classes = ["_background_", "Inclusions", "Patches", "Scratches"]

    # 原始图片路径
    images_path = r"D:\RNC14X\multi_U2NET\datasets\test_data\Image"
    # images_path = 'C:\\Users\\ASUS\\Desktop\\MJdatasets_source\\images_for_test\\'
    # 图片的标签路径
    gt_dir = r"D:\RNC14X\multi_U2NET\datasets\test_data\mask"
    # gt_dir = 'C:/Users/ASUS/Desktop/MJdatasets_source/masks_for_test/'
    # 存放推理结果图片的路径
    pred_dir = r"D:\RNC14X\multi_U2NET\datasets\test_data\predict_masks"
    predict_label = r"D:\RNC14X\multi_U2NET\predict\predict_lables"
    # 存放 miou 计算结果的 图片
    miou_out_path = "miou_out"
    # 模型路径
    model_dir = r'D:\RNC14X\multi_U2NET\saved_models\u2netp\u2netp_bce_itr_76230.pth'
    eval_print_miou(num_classes, name_classes, images_path, gt_dir, pred_dir, predict_label, miou_out_path, model_dir)
